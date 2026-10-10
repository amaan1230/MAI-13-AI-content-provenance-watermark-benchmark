from __future__ import annotations

import json
import pickle
from typing import Any

from fastapi import APIRouter, Body, Depends, Query, status
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api.deps import Principal, get_db, researcher, viewer
from backend.api.schemas import (
    ArtifactOut,
    ClassifyIn,
    ClassifyOut,
    ExperimentBrief,
    ExperimentCreated,
    ExperimentOut,
    Page,
)
from backend.core.config import get_settings
from backend.core.errors import Conflict, NotFound, ValidationFailed
from backend.core.storage import LocalArtifactStore
from backend.detectors.registry import DETECTORS
from backend.experiments import service
from backend.experiments.executor import get_executor
from backend.models import ExperimentArtifact
from backend.provenance.registry import validate_text

router = APIRouter(tags=["experiments"])


def _artifact(db: Session, experiment_id: str, name: str) -> ExperimentArtifact:
    art = db.execute(
        select(ExperimentArtifact).where(
            ExperimentArtifact.experiment_run_id == experiment_id, ExperimentArtifact.name == name
        )
    ).scalar_one_or_none()
    if art is None:
        raise NotFound(f"Artifact {name!r} not found for experiment {experiment_id}")
    return art


def _read_artifact(art: ExperimentArtifact) -> bytes:
    store = LocalArtifactStore(get_settings().artifact_dir)
    data = store.get_bytes(art.storage_key)
    import hashlib

    if hashlib.sha256(data).hexdigest() != art.sha256:
        raise Conflict("Artifact failed integrity check (sha256 mismatch)", details={"artifact": art.name})
    return data


@router.post("/api/v1/experiments", response_model=ExperimentCreated, status_code=status.HTTP_202_ACCEPTED)
def create(
    config: dict[str, Any] = Body(
        ...,
        examples=[
            {
                "name": "smoke",
                "experiment_type": "watermark_robustness",
                "dataset": {"n_samples": 20},
                "watermarks": [{"method": "kgw"}],
                "attacks": [{"name": "truncation"}],
            }
        ],
    ),
    db: Session = Depends(get_db),
    p: Principal = Depends(researcher),
):
    """Validate the configuration, persist a pending experiment and queue it for background execution."""
    cfg = service.validate_config(config)
    exp = service.create_experiment(db, cfg, actor=p.name)
    db.commit()
    get_executor().submit(exp.id, p.name)
    return ExperimentCreated(id=exp.id, status=exp.status, status_url=f"/api/v1/experiments/{exp.id}")


@router.post("/api/v1/experiments/validate")
def validate(config: dict[str, Any] = Body(...), _: Principal = Depends(viewer)):
    cfg = service.validate_config(config)
    return {"valid": True, "normalized": cfg.model_dump()}


@router.get("/api/v1/experiments", response_model=Page[ExperimentBrief])
def list_(
    db: Session = Depends(get_db),
    _: Principal = Depends(viewer),
    status_filter: str | None = Query(None, alias="status"),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    items, total = service.list_experiments(db, status=status_filter, limit=limit, offset=offset)
    return Page[ExperimentBrief](items=items, total=total, limit=limit, offset=offset)


@router.get("/api/v1/experiments/compare")
def compare(
    ids: str = Query(..., description="Comma-separated experiment IDs"),
    db: Session = Depends(get_db),
    _: Principal = Depends(viewer),
):
    out = []
    for eid in [i.strip() for i in ids.split(",") if i.strip()][:10]:
        exp = service.get_experiment(db, eid)
        out.append(
            {
                "id": exp.id,
                "name": exp.name,
                "status": exp.status,
                "seed": exp.seed,
                "config_hash": exp.config_hash,
                "dataset_version_id": exp.dataset_version_id,
                "metrics": exp.metrics_summary or {},
            }
        )
    keys = sorted({k for e in out for k in e["metrics"]})
    return {"experiments": out, "metric_keys": keys}


@router.get("/api/v1/experiments/{experiment_id}", response_model=ExperimentOut)
def get(experiment_id: str, db: Session = Depends(get_db), _: Principal = Depends(viewer)):
    return service.get_experiment(db, experiment_id)


@router.post("/api/v1/experiments/{experiment_id}/cancel", response_model=ExperimentBrief)
def cancel(experiment_id: str, db: Session = Depends(get_db), p: Principal = Depends(researcher)):
    return service.cancel_experiment(db, experiment_id, actor=p.name)


@router.post(
    "/api/v1/experiments/{experiment_id}/rerun", response_model=ExperimentCreated, status_code=status.HTTP_202_ACCEPTED
)
def rerun(experiment_id: str, db: Session = Depends(get_db), p: Principal = Depends(researcher)):
    exp = service.rerun_experiment(db, experiment_id, actor=p.name)
    db.commit()
    get_executor().submit(exp.id, p.name)
    return ExperimentCreated(id=exp.id, status=exp.status, status_url=f"/api/v1/experiments/{exp.id}")


@router.get("/api/v1/experiments/{experiment_id}/results")
def results(experiment_id: str, db: Session = Depends(get_db), _: Principal = Depends(viewer)):
    exp = service.get_experiment(db, experiment_id)
    if exp.status != "succeeded":
        raise Conflict(
            f"Results are not available: experiment status is '{exp.status}'",
            details={"status": exp.status, "stage": exp.stage, "progress": exp.progress, "error": exp.error},
        )
    return json.loads(_read_artifact(_artifact(db, experiment_id, "results.json")))


@router.get("/api/v1/experiments/{experiment_id}/artifacts", response_model=list[ArtifactOut])
def artifacts(experiment_id: str, db: Session = Depends(get_db), _: Principal = Depends(viewer)):
    service.get_experiment(db, experiment_id)
    return (
        db.execute(
            select(ExperimentArtifact)
            .where(ExperimentArtifact.experiment_run_id == experiment_id)
            .order_by(ExperimentArtifact.name)
        )
        .scalars()
        .all()
    )


@router.get("/api/v1/experiments/{experiment_id}/artifacts/{name}")
def artifact(experiment_id: str, name: str, db: Session = Depends(get_db), _: Principal = Depends(viewer)):
    art = _artifact(db, experiment_id, name)
    if art.kind == "model":
        raise ValidationFailed("Model artifacts are not downloadable through the API")
    media = {"results_json": "application/json", "results_csv": "text/csv", "report_md": "text/markdown"}
    return PlainTextResponse(_read_artifact(art).decode("utf-8"), media_type=media.get(art.kind, "text/plain"))


@router.get("/api/v1/reports/{experiment_id}", tags=["reports"])
def report(
    experiment_id: str,
    format: str = Query("md", pattern="^(md|json)$"),
    db: Session = Depends(get_db),
    _: Principal = Depends(viewer),
):
    exp = service.get_experiment(db, experiment_id)
    if exp.status != "succeeded":
        raise Conflict(
            f"Report not available: experiment status is '{exp.status}'",
            details={"status": exp.status, "error": exp.error},
        )
    md = _read_artifact(_artifact(db, experiment_id, "report.md")).decode("utf-8")
    if format == "json":
        return {"experiment_id": experiment_id, "name": exp.name, "markdown": md}
    return PlainTextResponse(md, media_type="text/markdown")


@router.post("/api/v1/detectors/classify", response_model=ClassifyOut, tags=["detectors"])
def classify(body: ClassifyIn, db: Session = Depends(get_db), _: Principal = Depends(researcher)):
    """Score text with an AI-text classifier trained in a completed experiment (threshold from its validation set)."""
    validate_text(body.text)
    if body.detector not in DETECTORS:
        raise ValidationFailed(f"Unknown detector {body.detector!r}")
    exp = service.get_experiment(db, body.experiment_id)
    if exp.status != "succeeded":
        raise Conflict("Experiment has not succeeded")
    res = json.loads(_read_artifact(_artifact(db, exp.id, "results.json")))
    dres = res.get("detectors", {}).get("results", {}).get(body.detector)
    if dres is None:
        raise NotFound(f"Detector {body.detector} was not part of experiment {exp.id}")
    art = _artifact(db, exp.id, f"detector_{body.detector}")
    state = pickle.loads(_read_artifact(art))  # noqa: S301 - integrity-checked artifact written by this server
    det = DETECTORS[body.detector](**state["params"])
    det.fitted = state["fitted"]
    if "pipe" in state:
        det.pipe = state["pipe"]
    if "cols" in state:
        det._cols, det._fill = state["cols"], state["fill"]
    score = det.score([body.text])[0]
    score = None if score != score else score
    prim = f"val_fpr_{exp.config['evaluation']['primary_fpr']}"
    thr = dres["thresholds"].get(prim, {}).get("threshold")
    decision = "insufficient" if score is None or thr is None else ("positive" if score >= thr else "negative")
    return ClassifyOut(
        detector=body.detector,
        experiment_id=exp.id,
        score=score,
        threshold=thr,
        threshold_rule=prim,
        decision=decision,
        interpretation=(
            "Classifier output only: calibrated on this experiment's generator and domain. "
            "Not evidence of authorship; expect higher error rates out of distribution."
        ),
    )
