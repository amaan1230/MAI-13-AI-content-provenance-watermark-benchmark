from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.api.deps import Principal, get_db, researcher, viewer
from backend.api.schemas import ContentOut, GenerateIn, GenerateOut, IngestIn, Page
from backend.core.config import get_settings
from backend.core.errors import ValidationFailed
from backend.core.hashing import config_hash
from backend.models import ContentItem, WatermarkRun
from backend.provenance.registry import NewContent, get_content, get_or_create_model_config, register_content
from backend.provenance.signing import key_fingerprint
from backend.providers.hf_local import get_lm
from backend.providers.registry import get_provider
from backend.watermark.kgw import _Warper
from backend.watermark.registry import get_watermark

router = APIRouter(prefix="/api/v1/content", tags=["content"])


@router.post("/ingest", response_model=ContentOut, status_code=status.HTTP_201_CREATED)
def ingest(body: IngestIn, db: Session = Depends(get_db), p: Principal = Depends(researcher)):
    item = register_content(
        db,
        NewContent(
            text=body.text,
            source_type=body.source_type,
            label=body.label,
            label_source=body.label_source,
            language=body.language,
            parent_id=body.parent_id,
            external_id=body.external_id,
            meta={**body.meta, "ingested_via": "api"},
        ),
        actor=p.name,
    )
    return item


@router.post("/generate", response_model=GenerateOut, status_code=status.HTTP_201_CREATED)
def generate(body: GenerateIn, db: Session = Depends(get_db), p: Principal = Depends(researcher)):
    s = get_settings()
    wm_meta = None
    if body.provider == "gemini":
        if body.watermark is not None:
            raise ValidationFailed(
                "Generation-time watermarks need logit access; use provider=hf_local "
                "(or apply a post-hoc method via /watermarks/embed)."
            )
        res = get_provider("gemini").complete(
            body.prompt,
            temperature=body.temperature,
            max_output_tokens=body.max_new_tokens,
            timeout_s=s.provider_timeout_s,
            max_retries=s.provider_max_retries,
            backoff_base_s=s.provider_backoff_base_s,
        )
        text, model, latency, n_tok = res.text, res.model, res.latency_ms, res.usage.get("output_tokens")
        gen_cfg = {
            "provider": "gemini",
            "model": model,
            "temperature": body.temperature,
            "max_output_tokens": body.max_new_tokens,
        }
    else:
        model = body.model_name or s.default_generator_model
        if body.watermark is not None:
            method = get_watermark(body.watermark.method)
            if method.kind != "generation_time":
                raise ValidationFailed(f"{method.name} is post-hoc; use /watermarks/embed on generated text")
            cfg = {
                "model_name": model,
                "max_new_tokens": body.max_new_tokens,
                "temperature": body.temperature,
                "top_p": body.top_p,
                "seed": body.seed,
                **body.watermark.config,
            }
            e = method.embed(body.prompt, cfg)
            text, latency, n_tok = e.text, e.latency_ms, e.n_tokens
            wm_meta = {
                "method": method.name,
                "version": method.version,
                "config": method.parse_config(cfg).model_dump(),
            }
        else:
            o = get_lm(model).generate(
                [body.prompt],
                max_new_tokens=body.max_new_tokens,
                seed=body.seed,
                processors=[_Warper(body.temperature, body.top_p)],
                batch_size=1,
            )[0]
            text, latency, n_tok = o.text, o.latency_ms, o.n_new_tokens
        gen_cfg = {
            "provider": "hf_local",
            "model": model,
            "max_new_tokens": body.max_new_tokens,
            "temperature": body.temperature,
            "top_p": body.top_p,
            "seed": body.seed,
            "watermark": wm_meta,
        }
    if not text.strip():
        raise ValidationFailed("The model returned empty text; try a different prompt or more tokens")
    content_id = prompt_id = None
    if body.persist:
        mc = get_or_create_model_config(db, provider=body.provider, model_name=model, role="generator", params=gen_cfg)
        prompt = register_content(
            db, NewContent(text=body.prompt, source_type="unknown", meta={"role": "prompt"}), actor=p.name
        )
        item = register_content(
            db,
            NewContent(
                text=text,
                source_type="ai_watermarked" if wm_meta else "ai",
                label="ai",
                label_source=f"Generated via API by {body.provider}:{model}",
                parent_id=prompt.id,
                model_config_id=mc.id,
                generation_config=gen_cfg,
            ),
            actor=p.name,
        )
        if wm_meta:
            db.add(
                WatermarkRun(
                    method=wm_meta["method"],
                    method_version=wm_meta["version"],
                    config=wm_meta["config"],
                    config_hash=config_hash(wm_meta["config"]),
                    key_fingerprint=key_fingerprint(s.watermark_secret_key.get_secret_value()),
                    model_config_id=mc.id,
                    input_content_id=prompt.id,
                    output_content_id=item.id,
                    n_tokens=n_tok,
                    latency_ms=latency,
                )
            )
        content_id, prompt_id = item.id, prompt.id
    return GenerateOut(
        text=text,
        provider=body.provider,
        model=model,
        latency_ms=latency,
        n_tokens=n_tok,
        watermark=wm_meta,
        content_id=content_id,
        prompt_content_id=prompt_id,
    )


@router.get("", response_model=Page[ContentOut])
def list_content(
    db: Session = Depends(get_db),
    _: Principal = Depends(viewer),
    source_type: str | None = None,
    label: str | None = None,
    experiment_id: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    q = select(ContentItem)
    cq = select(func.count()).select_from(ContentItem)
    for col, val in (
        (ContentItem.source_type, source_type),
        (ContentItem.label, label),
        (ContentItem.experiment_run_id, experiment_id),
    ):
        if val is not None:
            q, cq = q.where(col == val), cq.where(col == val)
    items = db.execute(q.order_by(ContentItem.created_at.desc()).limit(limit).offset(offset)).scalars().all()
    return Page[ContentOut](items=items, total=db.execute(cq).scalar_one(), limit=limit, offset=offset)


@router.get("/{content_id}", response_model=ContentOut)
def get_one(content_id: str, db: Session = Depends(get_db), _: Principal = Depends(viewer)):
    return get_content(db, content_id)
