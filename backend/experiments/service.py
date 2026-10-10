"""Experiment lifecycle operations shared by the API and the CLI."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend import __version__
from backend.core.errors import Conflict, NotFound, ValidationFailed
from backend.core.hashing import config_hash
from backend.experiments.config import ExperimentConfig
from backend.models import ExperimentRun
from backend.models.base import utcnow
from backend.provenance.audit import record_event

TERMINAL = {"succeeded", "failed", "cancelled"}


def validate_config(raw: dict[str, Any]) -> ExperimentConfig:
    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        errors = [{"loc": ".".join(str(x) for x in e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        raise ValidationFailed("Invalid experiment configuration", details={"errors": errors}) from None


def create_experiment(
    session: Session, cfg: ExperimentConfig, *, actor: str, rerun_of: str | None = None
) -> ExperimentRun:
    data = cfg.model_dump()
    exp = ExperimentRun(
        name=cfg.name,
        experiment_type=cfg.experiment_type,
        status="pending",
        config=data,
        config_hash=config_hash(data),
        seed=cfg.seed,
        software_version=__version__,
        created_by=actor,
        rerun_of_id=rerun_of,
    )
    session.add(exp)
    session.flush()
    record_event(
        session,
        actor=actor,
        action="experiment.create",
        entity_type="experiment",
        entity_id=exp.id,
        details={"config_hash": exp.config_hash, "rerun_of": rerun_of},
    )
    return exp


def get_experiment(session: Session, experiment_id: str) -> ExperimentRun:
    exp = session.get(ExperimentRun, experiment_id)
    if exp is None:
        raise NotFound(f"Experiment {experiment_id} not found")
    return exp


def list_experiments(
    session: Session, *, status: str | None, limit: int, offset: int
) -> tuple[list[ExperimentRun], int]:
    q = select(ExperimentRun)
    cq = select(func.count()).select_from(ExperimentRun)
    if status:
        q = q.where(ExperimentRun.status == status)
        cq = cq.where(ExperimentRun.status == status)
    total = session.execute(cq).scalar_one()
    items = session.execute(q.order_by(ExperimentRun.created_at.desc()).limit(limit).offset(offset)).scalars().all()
    return list(items), total


def rerun_experiment(session: Session, experiment_id: str, *, actor: str) -> ExperimentRun:
    src = get_experiment(session, experiment_id)
    cfg = validate_config(src.config)
    return create_experiment(session, cfg, actor=actor, rerun_of=src.id)


def cancel_experiment(session: Session, experiment_id: str, *, actor: str) -> ExperimentRun:
    exp = get_experiment(session, experiment_id)
    if exp.status in TERMINAL:
        raise Conflict(f"Experiment already {exp.status}")
    exp.status = "cancelled"
    if exp.started_at is None:
        exp.finished_at = utcnow()
    record_event(session, actor=actor, action="experiment.cancel", entity_type="experiment", entity_id=exp.id)
    return exp
