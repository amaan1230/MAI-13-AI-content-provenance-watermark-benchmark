"""End-to-end experiment on the toy model: lineage, metrics, artifacts and reproducibility."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import func, select

from backend.core.config import get_settings
from backend.core.storage import LocalArtifactStore
from backend.db.session import get_sessionmaker
from backend.experiments.config import ExperimentConfig
from backend.experiments.runner import ExperimentRunner
from backend.experiments.service import cancel_experiment, create_experiment
from backend.models import ContentItem, DetectionResult, ExperimentArtifact, ExperimentRun, Transformation, WatermarkRun
from backend.provenance.audit import verify_chain
from backend.provenance.registry import ancestors
from tests.integration.conftest import toy_experiment_config

pytestmark = pytest.mark.integration


def _run(cfg_dict) -> str:
    sm = get_sessionmaker()
    with sm() as s:
        exp = create_experiment(s, ExperimentConfig.model_validate(cfg_dict), actor="test")
        s.commit()
        eid = exp.id
    ExperimentRunner(eid, sm, actor="test").run()
    return eid


def test_full_experiment(db_session, toy_lm, toy_dataset):
    eid = _run(toy_experiment_config(toy_dataset))
    s = db_session
    exp = s.get(ExperimentRun, eid)
    assert exp.status == "succeeded", (exp.error, exp.error_details)
    assert exp.progress == 1.0 and exp.dataset_version_id and exp.environment["torch"]

    store = LocalArtifactStore(get_settings().artifact_dir)
    results = json.loads(store.get_bytes(f"experiments/{eid}/results.json"))

    # Watermark detection: on the near-uniform toy LM the KGW signal is strong.
    kgw = results["watermarks"]["kgw"]
    assert kgw["clean"]["auroc"]["auroc"] > 0.95
    assert kgw["thresholds"]["theoretical"]["threshold"] == 4.0
    assert "val_fpr_0.1" in kgw["thresholds"]
    # Thresholds are chosen on validation negatives only.
    assert "validation" in kgw["thresholds"]["val_fpr_0.1"]["source"]
    # Formatting normalisation destroys the zero-width mark but not KGW.
    zw_fmt = results["watermarks"]["zero_width"]["attacks"]["formatting"]
    assert zw_fmt["by_threshold"]["theoretical"]["tpr_after_attack"]["survival_rate"] == 0.0
    kgw_fmt = kgw["attacks"]["formatting"]
    assert kgw_fmt["by_threshold"]["theoretical"]["tpr_after_attack"]["survival_rate"] > 0.8
    # Skipped provider attack is reported, not silently dropped or faked.
    assert results["attacks"]["paraphrase"]["status"] == "skipped"
    assert "paraphrase" not in kgw["attacks"]
    assert any("paraphrase" in w for w in results["warnings"])

    # Detectors: trained, thresholded and compared against the baseline.
    det = results["detectors"]
    assert det["baseline"] == "tfidf_lr"
    assert set(det["results"]) == {"tfidf_lr", "lm_features_lr"}
    assert det["paired_comparisons"][0]["detector"] == "lm_features_lr"
    assert results["cross_task"]

    # Data integrity: group-aware split, no cross-split leakage.
    assert results["leakage"]["groups_crossing_splits"] == 0
    assert results["dataset"]["manifest"]["records_sha256"]

    # Provenance: every attacked text links back through its watermarked parent to the prompt.
    t = (
        s.execute(
            select(Transformation).where(Transformation.experiment_run_id == eid, Transformation.status == "succeeded")
        )
        .scalars()
        .first()
    )
    chain = ancestors(s, t.child_content_id)
    assert chain[0].id == t.parent_content_id
    assert chain[-1].meta.get("role") == "prompt"
    child = s.get(ContentItem, t.child_content_id)
    parent = s.get(ContentItem, t.parent_content_id)
    assert child.split == parent.split and child.group_id == parent.group_id
    assert (
        s.execute(
            select(func.count()).select_from(WatermarkRun).where(WatermarkRun.experiment_run_id == eid)
        ).scalar_one()
        > 0
    )
    assert (
        s.execute(
            select(func.count()).select_from(DetectionResult).where(DetectionResult.experiment_run_id == eid)
        ).scalar_one()
        > 0
    )

    # Artifacts registered with integrity hashes; Markdown report generated.
    names = set(s.execute(select(ExperimentArtifact.name).where(ExperimentArtifact.experiment_run_id == eid)).scalars())
    assert {"results.json", "samples.csv", "detections.csv", "report.md", "detector_tfidf_lr"} <= names
    report = store.get_bytes(f"experiments/{eid}/report.md").decode()
    assert "Robustness to attacks" in report and "not proof of authorship" in report
    assert verify_chain(s)["valid"]


def test_experiment_is_reproducible(db_session, toy_lm, toy_dataset):
    cfg = toy_experiment_config(
        toy_dataset, attacks=[{"name": "word_deletion", "config": {"rate": 0.2}}], detectors=[{"name": "tfidf_lr"}]
    )
    a, b = _run(cfg), _run(cfg)
    ea, eb = db_session.get(ExperimentRun, a), db_session.get(ExperimentRun, b)
    assert ea.status == eb.status == "succeeded"
    assert ea.config_hash == eb.config_hash
    assert ea.metrics_summary == eb.metrics_summary
    # Generated texts are identical too.
    ta = sorted(
        db_session.execute(select(ContentItem.content_hash).where(ContentItem.experiment_run_id == a)).scalars()
    )
    tb = sorted(
        db_session.execute(select(ContentItem.content_hash).where(ContentItem.experiment_run_id == b)).scalars()
    )
    assert ta == tb


def test_failure_is_recorded(db_session, toy_lm, settings_env):
    cfg = toy_experiment_config(settings_env / "missing.jsonl")
    eid = _run(cfg)
    exp = db_session.get(ExperimentRun, eid)
    assert exp.status == "failed"
    assert "not found" in exp.error
    assert exp.error_details["stage"] == "data"
    assert "traceback" in exp.error_details


def test_cancelled_before_start(db_session, toy_lm, toy_dataset):
    sm = get_sessionmaker()
    with sm() as s:
        exp = create_experiment(s, ExperimentConfig.model_validate(toy_experiment_config(toy_dataset)), actor="t")
        s.commit()
        eid = exp.id
        cancel_experiment(s, eid, actor="t")
        s.commit()
    ExperimentRunner(eid, sm).run()
    db_session.expire_all()
    exp = db_session.get(ExperimentRun, eid)
    assert exp.status == "cancelled"
