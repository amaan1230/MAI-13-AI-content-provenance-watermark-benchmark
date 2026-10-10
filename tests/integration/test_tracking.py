from __future__ import annotations

import pytest

from backend.core.config import reset_settings_cache
from backend.experiments.tracking import Tracker, flatten

pytestmark = pytest.mark.integration


def test_flatten():
    assert flatten({"a": {"b": 1}, "c": [{"d": 2}], "e": [1, 2]}) == {"a.b": 1, "c.0.d": 2, "e": [1, 2]}


def test_mlflow_tracking_roundtrip(settings_env, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(settings_env / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("MLFLOW_ARTIFACT_DIR", str(settings_env / "mlartifacts"))
    reset_settings_cache()
    t = Tracker(experiment_name="mai13-test")
    run_id = t.start("unit", {"k": "v"})
    assert run_id
    t.log_params({"config": {"seed": 1, "watermarks": [{"method": "kgw"}]}})
    t.log_metrics({"wm.kgw.auroc": 0.9, "skip_none": None, "flag": True})
    f = settings_env / "a.txt"
    f.write_text("x")
    t.log_artifact(f)
    t.end()

    import mlflow

    run = mlflow.get_run(run_id)
    assert run.data.params["config.seed"] == "1"
    assert run.data.metrics["wm.kgw.auroc"] == 0.9
    assert "skip_none" not in run.data.metrics and "flag" not in run.data.metrics
    assert (settings_env / "mlartifacts").exists()


def test_tracking_failure_does_not_raise(settings_env, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "bogus-scheme://nowhere")
    reset_settings_cache()
    t = Tracker()
    assert t.start("x", {}) is None and t.enabled is False
    t.log_metrics({"a": 1.0})  # no-op, no exception
    t.end()
