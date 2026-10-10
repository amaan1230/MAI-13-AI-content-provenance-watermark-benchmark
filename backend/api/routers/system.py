from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import select, text, update
from sqlalchemy.orm import Session

from backend import __version__
from backend.api.deps import Principal, admin, get_db, viewer
from backend.api.schemas import DatasetVersionOut, HealthOut, ProviderOut, ProvidersOut
from backend.attacks.registry import ATTACKS
from backend.core.config import get_settings
from backend.core.errors import ValidationFailed
from backend.db.session import get_engine
from backend.detectors.registry import DETECTORS
from backend.models import ContentItem, DatasetVersion
from backend.models.base import utcnow
from backend.provenance.audit import record_event, verify_chain
from backend.providers.registry import list_statuses
from backend.watermark.registry import WATERMARK_METHODS, get_watermark

router = APIRouter()


@router.get("/health", response_model=HealthOut, tags=["system"])
def health() -> HealthOut:
    db_status = "ok"
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        db_status = f"error: {type(exc).__name__}"
    statuses = list_statuses()
    return HealthOut(
        status="ok" if db_status == "ok" else "degraded",
        version=__version__,
        database=db_status,
        database_dialect=get_engine().dialect.name,
        providers_available=sum(1 for s in statuses if s.available),
        auth_enabled=get_settings().auth_enabled,
    )


@router.get("/api/v1/providers", response_model=ProvidersOut, tags=["system"])
def providers(_: Principal = Depends(viewer)) -> ProvidersOut:
    wms = []
    for name in WATERMARK_METHODS:
        md = get_watermark(name).get_metadata()
        wms.append({k: v for k, v in md.items()})
    attacks = []
    for name, cls in ATTACKS.items():
        attacks.append(
            {
                "name": name,
                "attack_type": cls.attack_type,
                "version": cls.version,
                "deterministic": cls.deterministic,
                "requires": cls.requires,
                "config_schema": cls.config_cls.model_json_schema(),
                "doc": (cls.__doc__ or "").strip(),
            }
        )
    detectors = [
        {"name": n, "version": c.version, "supervised": c.supervised, "description": c.description}
        for n, c in DETECTORS.items()
    ]
    return ProvidersOut(
        providers=[ProviderOut(**s.__dict__) for s in list_statuses()],
        watermark_methods=wms,
        attacks=attacks,
        detectors=detectors,
    )


@router.get("/api/v1/datasets", response_model=list[DatasetVersionOut], tags=["datasets"])
def datasets(db: Session = Depends(get_db), _: Principal = Depends(viewer)):
    return db.execute(select(DatasetVersion).order_by(DatasetVersion.created_at.desc())).scalars().all()


@router.get("/api/v1/audit/verify", tags=["provenance"])
def audit_verify(db: Session = Depends(get_db), _: Principal = Depends(admin)):
    """Recompute the audit-log hash chain and report the first broken link, if any."""
    return verify_chain(db)


@router.post("/api/v1/admin/retention/purge", tags=["admin"])
def purge_raw_text(older_than_days: int | None = None, db: Session = Depends(get_db), p: Principal = Depends(admin)):
    """Privacy control: drop raw text older than N days, keeping hashes and lineage."""
    days = older_than_days if older_than_days is not None else get_settings().data_retention_days
    if days <= 0:
        raise ValidationFailed("Provide older_than_days > 0 or set DATA_RETENTION_DAYS")
    cutoff = utcnow() - timedelta(days=days)
    res = db.execute(
        update(ContentItem).where(ContentItem.created_at < cutoff, ContentItem.text.is_not(None)).values(text=None)
    )
    record_event(
        db,
        actor=p.name,
        action="retention.purge",
        entity_type="content",
        entity_id="*",
        details={"older_than_days": days, "rows": res.rowcount},
    )
    return {"purged_rows": res.rowcount, "cutoff": cutoff.isoformat()}
