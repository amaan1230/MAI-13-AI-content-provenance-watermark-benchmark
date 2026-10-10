from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.api.deps import Principal, get_db, researcher, viewer
from backend.api.schemas import DetectIn, DetectOut, EmbedIn, EmbedOut
from backend.core.config import get_settings
from backend.core.errors import ValidationFailed
from backend.core.hashing import config_hash
from backend.models import DetectionResult, WatermarkRun
from backend.provenance.registry import NewContent, get_content, register_content, validate_text
from backend.provenance.signing import key_fingerprint
from backend.watermark.registry import get_watermark

router = APIRouter(prefix="/api/v1/watermarks", tags=["watermarks"])


def interpret(decision: str, method: str) -> str:
    if decision == "positive":
        return (
            f"The text carries a {method} watermark signal consistent with this server's key "
            "(statistical evidence, not proof of authorship)."
        )
    if decision == "negative":
        return (
            f"No {method} signal above threshold. This does not show the text is human-written: it may be "
            "unwatermarked AI text, a different key, or heavily edited."
        )
    return "Not enough scorable tokens for a reliable decision."


@router.post("/embed", response_model=EmbedOut, status_code=status.HTTP_201_CREATED)
def embed(body: EmbedIn, db: Session = Depends(get_db), p: Principal = Depends(researcher)):
    validate_text(body.text)
    method = get_watermark(body.method)
    cfg = method.parse_config(body.config)
    res = method.embed(body.text, cfg)
    if not res.text.strip():
        raise ValidationFailed("Watermark embedding produced empty text")
    content_id = run_id = None
    if body.persist:
        if method.kind == "post_hoc":
            parent = body.parent_content_id
            if parent is None:
                parent = register_content(
                    db,
                    NewContent(text=body.text, source_type="unknown", meta={"role": "watermark_input"}),
                    actor=p.name,
                ).id
            item = register_content(
                db,
                NewContent(
                    text=res.text,
                    source_type="ai_watermarked",
                    label="unknown",
                    parent_id=parent,
                    meta={
                        "watermark_method": method.name,
                        "note": "post-hoc mark; authorship of the input is not asserted",
                    },
                ),
                actor=p.name,
            )
            input_id = parent
        else:
            prompt = register_content(
                db,
                NewContent(
                    text=body.text, source_type="unknown", meta={"role": "prompt"}, parent_id=body.parent_content_id
                ),
                actor=p.name,
            )
            item = register_content(
                db,
                NewContent(
                    text=res.text,
                    source_type="ai_watermarked",
                    label="ai",
                    label_source=f"Generated via API with {method.name} watermark "
                    f"({cfg.model_dump().get('model_name')})",
                    parent_id=prompt.id,
                    generation_config=cfg.model_dump(),
                ),
                actor=p.name,
            )
            input_id = prompt.id
        run = WatermarkRun(
            method=method.name,
            method_version=method.version,
            config=cfg.model_dump(),
            config_hash=config_hash(cfg.model_dump()),
            key_fingerprint=key_fingerprint(get_settings().watermark_secret_key.get_secret_value()),
            input_content_id=input_id,
            output_content_id=item.id,
            n_tokens=res.n_tokens,
            latency_ms=res.latency_ms,
        )
        db.add(run)
        db.flush()
        content_id, run_id = item.id, run.id
    return EmbedOut(
        method=method.name,
        kind=method.kind,
        text=res.text,
        n_tokens=res.n_tokens,
        latency_ms=res.latency_ms,
        content_id=content_id,
        watermark_run_id=run_id,
        metadata={k: v for k, v in res.metadata.items()},
    )


@router.post("/detect", response_model=DetectOut)
def detect(body: DetectIn, db: Session = Depends(get_db), p: Principal = Depends(researcher)):
    method = get_watermark(body.method)
    if body.content_id is not None:
        item = get_content(db, body.content_id)
        if item.text is None:
            raise ValidationFailed("Raw text for this content was not retained (privacy settings)")
        text = item.text
    else:
        text = body.text or ""
        validate_text(text)
    cfg = method.parse_config(body.config)
    r = method.detect(text, cfg)
    det_id = None
    if body.persist and body.content_id is not None:
        row = DetectionResult(
            content_id=body.content_id,
            detector_type="watermark",
            detector_name=method.name,
            detector_version=method.version,
            config=cfg.model_dump(),
            score=r.score,
            threshold=r.threshold,
            decision=r.decision,
            p_value=r.p_value,
            n_tokens_scored=r.n_tokens_scored,
            latency_ms=r.latency_ms,
            details={k: v for k, v in r.details.items() if k != "green_flags"},
        )
        db.add(row)
        db.flush()
        det_id = row.id
    return DetectOut(
        method=method.name,
        score=r.score,
        p_value=r.p_value,
        threshold=r.threshold,
        decision=r.decision,
        n_tokens_scored=r.n_tokens_scored,
        latency_ms=r.latency_ms,
        details={k: v for k, v in r.details.items() if k != "green_flags"},
        detection_id=det_id,
        interpretation=interpret(r.decision, method.name),
    )


@router.get("/methods")
def methods(_: Principal = Depends(viewer)):
    from backend.watermark.registry import WATERMARK_METHODS

    return [get_watermark(n).get_metadata() for n in WATERMARK_METHODS]
