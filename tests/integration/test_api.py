"""HTTP API tests (FastAPI TestClient, isolated SQLite database, toy model, no network)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from backend.core.config import reset_settings_cache
from backend.experiments.executor import set_executor
from tests.conftest import TOY_MODEL_NAME
from tests.integration.conftest import toy_experiment_config

pytestmark = pytest.mark.integration


@pytest.fixture()
def client(db_session, toy_lm, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    reset_settings_cache()
    set_executor(None)
    from backend.api.main import create_app

    with TestClient(create_app()) as c:
        yield c
    set_executor(None)


def test_health_and_request_id(client):
    r = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["database_dialect"] in ("sqlite", "postgresql")
    assert r.headers["X-Request-ID"] == "abc-123"
    assert len(client.get("/health", headers={"X-Request-ID": "bad id with spaces"}).headers["X-Request-ID"]) == 32


def test_providers_reports_missing_gemini_key(client):
    body = client.get("/api/v1/providers").json()
    gem = next(p for p in body["providers"] if p["name"] == "gemini")
    assert gem["available"] is False and "GEMINI_API_KEY" in gem["message"]
    assert {m["name"] for m in body["watermark_methods"]} == {"kgw", "unigram", "zero_width"}
    assert "AIza" not in str(body)


def test_generate_with_missing_gemini_key_returns_503(client):
    r = client.post("/api/v1/content/generate", json={"prompt": "hi", "provider": "gemini"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "provider_unavailable"
    assert r.json()["error"]["request_id"]


def test_generate_hf_local_with_watermark(client):
    r = client.post(
        "/api/v1/content/generate",
        json={
            "prompt": "w1 w2",
            "model_name": TOY_MODEL_NAME,
            "max_new_tokens": 60,
            "watermark": {"method": "kgw", "config": {"min_new_tokens": 60}},
        },
    )
    assert r.status_code == 201, r.text
    cid = r.json()["content_id"]
    d = client.post(
        "/api/v1/watermarks/detect", json={"method": "kgw", "content_id": cid, "config": {"model_name": TOY_MODEL_NAME}}
    )
    assert d.json()["decision"] == "positive", d.json()
    prov = client.get(f"/api/v1/provenance/{cid}").json()
    assert prov["watermark_runs"] and prov["ancestors"][0]["meta"]["role"] == "prompt"
    assert len(prov["detections"]) == 1
    assert "proof of authorship" in prov["disclaimer"]


def test_ingest_validation_and_errors(client):
    r = client.post("/api/v1/content/ingest", json={"text": "hello", "label": "human"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_failed"
    r = client.post("/api/v1/content/ingest", json={"text": ""})
    assert r.status_code == 422
    r = client.post("/api/v1/content/ingest", json={"text": "   "})
    assert r.status_code == 422
    r = client.post("/api/v1/content/ingest", json={"text": "hi", "parent_id": "nope"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = client.post(
        "/api/v1/content/ingest", json={"text": "hello", "label": "human", "label_source": "Written by test author"}
    )
    assert r.status_code == 201
    assert client.get("/api/v1/content?limit=1").json()["total"] >= 1


def test_watermark_embed_detect_and_attack_lineage(client):
    e = client.post(
        "/api/v1/watermarks/embed",
        json={"method": "zero_width", "text": "one two three four five six seven eight nine ten"},
    )
    assert e.status_code == 201
    cid = e.json()["content_id"]
    assert (
        client.post("/api/v1/watermarks/detect", json={"method": "zero_width", "content_id": cid}).json()["decision"]
        == "positive"
    )
    a = client.post(
        "/api/v1/attacks/run",
        json={"attack": "formatting", "content_id": cid, "config": {"operations": ["normalize_unicode"]}},
    )
    assert a.status_code == 200 and a.json()["status"] == "succeeded"
    child = a.json()["child_content_id"]
    assert (
        client.post("/api/v1/watermarks/detect", json={"method": "zero_width", "content_id": child}).json()["decision"]
        == "negative"
    )
    prov = client.get(f"/api/v1/provenance/{child}").json()
    assert prov["ancestors"][0]["id"] == cid
    assert prov["transformations_in"][0]["attack_name"] == "formatting"
    v = client.post(f"/api/v1/provenance/{cid}/verify", json={"text": e.json()["text"]}).json()
    assert v["verified"] is True


def test_detect_requires_exactly_one_input(client):
    r = client.post("/api/v1/watermarks/detect", json={"method": "kgw"})
    assert r.status_code == 422
    r = client.post("/api/v1/watermarks/detect", json={"method": "bogus", "text": "x"})
    assert r.status_code == 422 and "available" in r.json()["error"]["details"]


def test_unknown_ids_404(client):
    for path in (
        "/api/v1/experiments/nope",
        "/api/v1/provenance/nope",
        "/api/v1/reports/nope",
        "/api/v1/experiments/nope/results",
    ):
        assert client.get(path).status_code == 404, path


def test_experiment_lifecycle_via_api(client, toy_dataset):
    bad = client.post("/api/v1/experiments", json={"name": "x", "watermarks": []})
    assert bad.status_code == 422 and bad.json()["error"]["details"]["errors"]

    cfg = toy_experiment_config(toy_dataset, attacks=[{"name": "truncation"}], detectors=[{"name": "tfidf_lr"}])
    r = client.post("/api/v1/experiments", json=cfg)
    assert r.status_code == 202
    eid = r.json()["id"]
    early = client.get(f"/api/v1/experiments/{eid}/results")
    assert early.status_code in (200, 409)
    deadline = time.time() + 300
    while time.time() < deadline:
        st = client.get(f"/api/v1/experiments/{eid}").json()
        if st["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.5)
    assert st["status"] == "succeeded", st
    res = client.get(f"/api/v1/experiments/{eid}/results").json()
    assert res["watermarks"]["kgw"]["clean"]["auroc"]["auroc"] is not None
    rep = client.get(f"/api/v1/reports/{eid}")
    assert rep.status_code == 200 and rep.text.startswith("# Experiment report")
    arts = client.get(f"/api/v1/experiments/{eid}/artifacts").json()
    assert {a["name"] for a in arts} >= {"results.json", "report.md"}
    assert client.get(f"/api/v1/experiments/{eid}/artifacts/detector_tfidf_lr").status_code == 422
    cls = client.post(
        "/api/v1/detectors/classify", json={"experiment_id": eid, "detector": "tfidf_lr", "text": "w1 w2 w3 w4"}
    )
    assert cls.status_code == 200 and cls.json()["decision"] in ("positive", "negative")
    lst = client.get("/api/v1/experiments?status=succeeded").json()
    assert lst["total"] == 1
    cmp_ = client.get(f"/api/v1/experiments/compare?ids={eid}").json()
    assert cmp_["metric_keys"]
    assert client.post(f"/api/v1/experiments/{eid}/cancel").status_code == 409
    assert client.get("/api/v1/audit/verify").json()["valid"] is True


@pytest.fixture()
def auth_client(db_session, monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEYS", '{"view-key": "viewer", "res-key": "researcher", "admin-key": "admin"}')
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    reset_settings_cache()
    set_executor(None)
    from backend.api.main import create_app

    with TestClient(create_app(start_executor=False)) as c:
        yield c
    reset_settings_cache()


def test_auth_and_roles(auth_client):
    c = auth_client
    assert c.get("/health").status_code == 200  # health stays public
    assert c.get("/api/v1/experiments").status_code == 401
    assert c.get("/api/v1/experiments", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/api/v1/experiments", headers={"X-API-Key": "view-key"}).status_code == 200
    r = c.post("/api/v1/content/ingest", json={"text": "x"}, headers={"X-API-Key": "view-key"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"
    assert c.post("/api/v1/content/ingest", json={"text": "x"}, headers={"X-API-Key": "res-key"}).status_code == 201
    assert c.get("/api/v1/audit/verify", headers={"X-API-Key": "res-key"}).status_code == 403
    assert c.get("/api/v1/audit/verify", headers={"X-API-Key": "admin-key"}).status_code == 200


def test_rate_limit(db_session, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    reset_settings_cache()
    set_executor(None)
    from backend.api.main import create_app

    with TestClient(create_app(start_executor=False)) as c:
        codes = [c.get("/api/v1/experiments").status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429
    reset_settings_cache()
