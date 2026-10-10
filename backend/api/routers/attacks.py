from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.api.deps import Principal, get_db, researcher
from backend.api.schemas import AttackIn, AttackOut
from backend.attacks.registry import build_attack
from backend.core.errors import ProviderUnavailable, ValidationFailed
from backend.evaluation.quality import jaccard_similarity, semantic_similarity
from backend.models import Transformation
from backend.provenance.registry import NewContent, get_content, register_content, validate_text

router = APIRouter(prefix="/api/v1/attacks", tags=["attacks"])


@router.post("/run", response_model=AttackOut)
def run_attack(body: AttackIn, db: Session = Depends(get_db), p: Principal = Depends(researcher)):
    attack = build_attack(body.attack, body.config)
    parent = None
    if body.content_id is not None:
        parent = get_content(db, body.content_id)
        if parent.text is None:
            raise ValidationFailed("Raw text for this content was not retained (privacy settings)")
        text = parent.text
    else:
        text = body.text or ""
        validate_text(text)
    res = attack.apply(text, body.seed)

    sim = metric = None
    if res.status == "succeeded" and body.compute_similarity:
        try:
            s = semantic_similarity([(text, res.text)], allow_fallback=False)
            sim, metric = s["values"][0], s["metric"]
        except ProviderUnavailable:
            sim, metric = jaccard_similarity(text, res.text), "jaccard"

    child_id = tr_id = None
    if body.persist and parent is not None:
        tr = Transformation(
            parent_content_id=parent.id,
            attack_type=attack.attack_type,
            attack_name=attack.name,
            attack_version=attack.version,
            config=attack.config.model_dump(),
            seed=body.seed,
            status=res.status,
            error=res.error,
            latency_ms=res.latency_ms,
            semantic_similarity=sim,
            meta={**res.meta, "similarity_metric": metric},
        )
        if res.status == "succeeded":
            child = register_content(
                db,
                NewContent(
                    text=res.text,
                    source_type="transformed",
                    label=parent.label,
                    label_source=(
                        f"Inherited from parent {parent.id} via attack {attack.name}"
                        if parent.label != "unknown"
                        else None
                    ),
                    parent_id=parent.id,
                    group_id=parent.group_id,
                    split=parent.split,
                    meta={"attack": attack.name},
                ),
                actor=p.name,
            )
            tr.child_content_id = child.id
            child_id = child.id
        db.add(tr)
        db.flush()
        tr_id = tr.id
    return AttackOut(
        attack=attack.name,
        attack_type=attack.attack_type,
        status=res.status,
        text=res.text,
        error=res.error,
        latency_ms=res.latency_ms,
        meta=res.meta,
        similarity=sim,
        similarity_metric=metric,
        parent_content_id=parent.id if parent else None,
        child_content_id=child_id,
        transformation_id=tr_id,
    )
