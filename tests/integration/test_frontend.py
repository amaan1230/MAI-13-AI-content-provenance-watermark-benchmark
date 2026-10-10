"""Render every dashboard section against the real API (in-process TestClient), offline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.core.config import reset_settings_cache
from backend.experiments.executor import set_executor
from tests.integration.conftest import toy_experiment_config

pytestmark = pytest.mark.integration

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


@pytest.fixture()
def wired_frontend(db_session, toy_lm, toy_dataset, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    reset_settings_cache()
    set_executor(None)
    from backend.api.main import create_app

    sys.path.insert(0, str(FRONTEND))
    import api_client

    with TestClient(create_app()) as client:

        def fake_request(method, path, *, json=None, params=None, timeout=120.0, raw=False):
            r = client.request(method, path, json=json, params=params)
            if r.is_success:
                return api_client.ApiResult(True, r.status_code, r.text if raw else r.json())
            err = r.json().get("error", {})
            return api_client.ApiResult(
                False,
                r.status_code,
                error_code=err.get("code"),
                message=err.get("message"),
                details=err.get("details"),
                request_id=err.get("request_id"),
            )

        monkeypatch.setattr(api_client, "request", fake_request)
        # Seed one completed experiment so result-driven pages have real data.
        from backend.db.session import get_sessionmaker
        from backend.experiments.config import ExperimentConfig
        from backend.experiments.runner import ExperimentRunner
        from backend.experiments.service import create_experiment

        sm = get_sessionmaker()
        with sm() as s:
            exp = create_experiment(
                s,
                ExperimentConfig.model_validate(toy_experiment_config(toy_dataset, detectors=[{"name": "tfidf_lr"}])),
                actor="test",
            )
            s.commit()
            eid = exp.id
        ExperimentRunner(eid, sm).run()
        yield eid
    set_executor(None)
    sys.path.remove(str(FRONTEND))


SECTIONS = [
    "1 · Overview & health",
    "2 · Datasets",
    "3 · Generate & ingest",
    "4 · Watermark embed/detect",
    "5 · Attacks",
    "6 · Experiments",
    "7 · Baseline & robustness comparison",
    "8 · Provenance explorer",
    "9 · Metrics & error analysis",
    "10 · Reports & settings",
]


@pytest.mark.parametrize("section", SECTIONS)
def test_section_renders_without_errors(wired_frontend, section):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(section).run()
    assert not at.exception, [e.value for e in at.exception]
    # Pages that depend on results must show real (non-placeholder) content.
    if section.startswith("7 "):
        assert at.dataframe, "comparison tables should render from results"
    if section.startswith("1 "):
        assert any(m.value == "ok" for m in at.metric), [str(at.main)[:3000]] + [e.value for e in at.error]
