"""Provenance registry: content registration, lineage and integrity verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.config import get_settings
from backend.core.errors import NotFound, ValidationFailed
from backend.core.hashing import config_hash, content_hash, normalized_hash
from backend.models import ContentItem, DetectionResult, ModelConfig, Transformation, WatermarkRun
from backend.models.entities import LABELS, SOURCE_TYPES
from backend.provenance.audit import record_event
from backend.provenance.signing import sign_record, verify_record

MAX_LINEAGE_DEPTH = 50


@dataclass
class NewContent:
    text: str
    source_type: str
    label: str = "unknown"
    label_source: str | None = None
    language: str | None = "en"
    dataset_version_id: str | None = None
    split: str | None = None
    group_id: str | None = None
    external_id: str | None = None
    parent_id: str | None = None
    model_config_id: str | None = None
    experiment_run_id: str | None = None
    generation_config: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _signable(item: ContentItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "content_hash": item.content_hash,
        "source_type": item.source_type,
        "label": item.label,
        "parent_id": item.parent_id,
        "model_config_id": item.model_config_id,
    }


def validate_text(text: str) -> None:
    if not isinstance(text, str) or not text.strip():
        raise ValidationFailed("Text must be a non-empty string")
    limit = get_settings().max_text_chars
    if len(text) > limit:
        raise ValidationFailed(f"Text exceeds MAX_TEXT_CHARS={limit}", details={"length": len(text)})


def register_content(session: Session, new: NewContent, *, actor: str = "system", audit: bool = True) -> ContentItem:
    validate_text(new.text)
    if new.source_type not in SOURCE_TYPES:
        raise ValidationFailed(f"source_type must be one of {SOURCE_TYPES}")
    if new.label not in LABELS:
        raise ValidationFailed(f"label must be one of {LABELS}")
    if new.label != "unknown" and not new.label_source:
        raise ValidationFailed("A non-'unknown' label requires label_source documenting its provenance")
    if new.parent_id is not None and session.get(ContentItem, new.parent_id) is None:
        raise NotFound(f"Parent content {new.parent_id} not found")

    settings = get_settings()
    item = ContentItem(
        content_hash=content_hash(new.text),
        normalized_hash=normalized_hash(new.text),
        text=new.text if settings.store_raw_text else None,
        char_len=len(new.text),
        language=new.language,
        source_type=new.source_type,
        label=new.label,
        label_source=new.label_source,
        dataset_version_id=new.dataset_version_id,
        split=new.split,
        group_id=new.group_id,
        external_id=new.external_id,
        parent_id=new.parent_id,
        model_config_id=new.model_config_id,
        experiment_run_id=new.experiment_run_id,
        generation_config=new.generation_config,
        meta=new.meta,
        created_by=actor,
    )
    session.add(item)
    session.flush()  # assigns id
    if settings.provenance_signing_key is not None:
        item.signature = sign_record(settings.provenance_signing_key.get_secret_value(), _signable(item))
    if audit:
        record_event(
            session,
            actor=actor,
            action="content.register",
            entity_type="content",
            entity_id=item.id,
            details={"content_hash": item.content_hash, "source_type": item.source_type, "parent_id": item.parent_id},
        )
    return item


def get_or_create_model_config(
    session: Session,
    *,
    provider: str,
    model_name: str,
    role: str,
    params: dict | None = None,
    revision: str | None = None,
) -> ModelConfig:
    params = params or {}
    h = config_hash({"provider": provider, "model": model_name, "role": role, "revision": revision, "params": params})
    existing = session.execute(select(ModelConfig).where(ModelConfig.config_hash == h)).scalar_one_or_none()
    if existing:
        return existing
    mc = ModelConfig(
        provider=provider, model_name=model_name, role=role, params=params, revision=revision, config_hash=h
    )
    session.add(mc)
    session.flush()
    return mc


def get_content(session: Session, content_id: str) -> ContentItem:
    item = session.get(ContentItem, content_id)
    if item is None:
        raise NotFound(f"Content {content_id} not found")
    return item


def ancestors(session: Session, content_id: str) -> list[ContentItem]:
    """Parent chain from the immediate parent up to the root (cycle- and depth-guarded)."""
    out: list[ContentItem] = []
    seen = {content_id}
    current = get_content(session, content_id)
    while current.parent_id and len(out) < MAX_LINEAGE_DEPTH:
        if current.parent_id in seen:
            break
        seen.add(current.parent_id)
        parent = session.get(ContentItem, current.parent_id)
        if parent is None:
            break
        out.append(parent)
        current = parent
    return out


def descendants(session: Session, content_id: str) -> list[ContentItem]:
    out: list[ContentItem] = []
    frontier = [content_id]
    seen = {content_id}
    depth = 0
    while frontier and depth < MAX_LINEAGE_DEPTH:
        children = session.execute(select(ContentItem).where(ContentItem.parent_id.in_(frontier))).scalars().all()
        frontier = []
        for c in children:
            if c.id not in seen:
                seen.add(c.id)
                out.append(c)
                frontier.append(c.id)
        depth += 1
    return out


def verify_integrity(session: Session, content_id: str, candidate_text: str | None = None) -> dict[str, Any]:
    """Check stored text/hash consistency and, optionally, a caller-supplied text.

    Returns which checks were possible; a missing check is reported as None rather than passing.
    """
    item = get_content(session, content_id)
    settings = get_settings()
    result: dict[str, Any] = {
        "content_id": item.id,
        "stored_hash": item.content_hash,
        "stored_text_matches_hash": None,
        "candidate_matches_hash": None,
        "candidate_normalized_match": None,
        "signature_valid": None,
        "note": "Integrity checks show the record is unaltered; they do not prove authorship.",
    }
    if item.text is not None:
        result["stored_text_matches_hash"] = content_hash(item.text) == item.content_hash
    if candidate_text is not None:
        result["candidate_matches_hash"] = content_hash(candidate_text) == item.content_hash
        result["candidate_normalized_match"] = normalized_hash(candidate_text) == item.normalized_hash
    if item.signature and settings.provenance_signing_key is not None:
        result["signature_valid"] = verify_record(
            settings.provenance_signing_key.get_secret_value(), _signable(item), item.signature
        )
    checks = [v for k, v in result.items() if k.endswith(("_hash", "_valid", "_match")) and isinstance(v, bool)]
    result["verified"] = bool(checks) and all(checks)
    return result


def provenance_record(session: Session, content_id: str) -> dict[str, Any]:
    item = get_content(session, content_id)
    wm_runs = session.execute(select(WatermarkRun).where(WatermarkRun.output_content_id == content_id)).scalars().all()
    transforms_in = (
        session.execute(select(Transformation).where(Transformation.child_content_id == content_id)).scalars().all()
    )
    transforms_out = (
        session.execute(select(Transformation).where(Transformation.parent_content_id == content_id)).scalars().all()
    )
    detections = (
        session.execute(
            select(DetectionResult).where(DetectionResult.content_id == content_id).order_by(DetectionResult.created_at)
        )
        .scalars()
        .all()
    )
    return {
        "content": item,
        "ancestors": ancestors(session, content_id),
        "descendants": descendants(session, content_id),
        "watermark_runs": wm_runs,
        "transformations_in": transforms_in,
        "transformations_out": transforms_out,
        "detections": detections,
        "integrity": verify_integrity(session, content_id),
    }
