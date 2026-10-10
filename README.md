# MAI-13 — AI Content Provenance and Watermark Robustness Benchmark

A reproducible research platform for **embedding, detecting and attacking text watermarks**, comparing
**AI-generated-text classifiers** against a baseline, and keeping a **provenance registry** with content
lineage and a tamper-evident audit log.

> **Scope:** the text modality is fully implemented and evaluated. Image and audio are *not*
> implemented; the architecture has adapter points for them (see `docs/PHASE6_MODALITY_DECISION.md`).
> Watermark detection and AI-text classification are separate tasks, and **neither is proof of authorship**.

## Project status

| Item | Status |
|---|---|
| Text pipeline (watermarks, attacks, detectors, evaluation, provenance, API, dashboard) | Implemented |
| Automated tests | 140 tests: 127 core tests pass on SQLite and PostgreSQL 16; 13 MLflow/dashboard tests pass on SQLite |
| Smoke experiment with real models (Qwen2.5-0.5B, 24 C4 documents) | Completed end-to-end; pipeline check only, too small for conclusions |
| Main benchmark (`main_benchmark.yaml`, 300 documents) | **Not yet run** — results and `docs/RESULTS.md` pending |
| Docker image / Compose | Written; not yet built on the development machine (built in CI) |
| `transformers` 5.x upgrade (pip-audit reports advisories for 4.57.6) | Pending |
| Image / audio modalities | Not implemented (see `docs/PHASE6_MODALITY_DECISION.md`) |

## What it does

| Capability | Implementation |
|---|---|
| Watermarks | **KGW** (Kirchenbauer et al. 2023), **Unigram** (Zhao et al. 2024), **zero-width** format baseline — pluggable `embed / detect / get_metadata` |
| Attacks | truncation, word deletion/swap/insertion, typos, Unicode/whitespace/homoglyph formatting (offline, seeded); masked-LM lexical substitution, MarianMT back-translation, LLM paraphrase via local Qwen or Gemini (optional) |
| AI-text detectors | TF-IDF + LR (**baseline**), LM-feature LR (GLTR/log-rank/entropy/Binoculars features), zero-shot Binoculars |
| Evaluation | AUROC with bootstrap CIs, TPR/FPR/precision/recall/F1 at validation-chosen thresholds (Wilson CIs), watermark survival (absolute & conditional), paired ΔAUROC vs baseline, semantic similarity, oracle perplexity, latency and cost, length subgroups, leakage audits |
| Provenance | Content hashes, parent→child lineage, watermark runs, transformations, detections, optional HMAC signatures, hash-chained audit log |
| Ops | FastAPI + OpenAPI, Streamlit dashboard, PostgreSQL/SQLite + Alembic, MLflow, structured JSON logs, OpenTelemetry hooks, API keys + roles, rate limiting, Docker Compose, GitHub Actions CI |

## Quick start (local, no Docker)

Requires Python 3.11+ (tested on 3.11 in CI and 3.13 locally). CPU is enough.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt          # or: pip install -r requirements.lock (exact versions)
cp .env.example .env                         # edit WATERMARK_SECRET_KEY at least

python -m backend.db.migrate                 # create the schema (SQLite by default)
pytest -q                                    # offline test suite (~1 min)

python scripts/download_models.py            # optional: pre-fetch models (~3 GB) for offline runs
uvicorn backend.api.main:app --port 8000     # API + docs at http://localhost:8000/docs
streamlit run frontend/app.py                # dashboard at http://localhost:8501
```

Run an experiment from the CLI (synchronous) or from the dashboard / API (background):

```bash
python -m backend.experiments.cli validate experiments/configs/*.yaml
python -m backend.experiments.cli run experiments/configs/smoke.yaml          # ~10 min CPU
python -m backend.experiments.cli run experiments/configs/main_benchmark.yaml # ~2-3 h CPU
python -m backend.experiments.cli rerun <experiment_id>                       # same config + seed
```

Each run writes `artifacts/experiments/<id>/{results.json, samples.csv, detections.csv, report.md}`,
copies the report to `reports/`, records an MLflow run (`mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`)
and registers every text, transformation and detection in the database.

## Quick start (Docker)

```bash
cp .env.example .env
docker compose up --build                 # PostgreSQL + API (:8000) + dashboard (:8501)
docker compose --profile mlflow up        # also the MLflow UI on :5000
```

## Configuration

All settings are environment variables (see `.env.example`). Nothing is required except a
`WATERMARK_SECRET_KEY` for anything beyond local experimentation.

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | `sqlite:///...` (default) or `postgresql+psycopg://...` |
| `GEMINI_API_KEY` | Optional. Without it Gemini is disabled, the API returns 503 `provider_unavailable`, and Gemini attacks are **skipped and reported as skipped**. |
| `HF_OFFLINE` | `true` to forbid downloads (after `scripts/download_models.py`) |
| `WATERMARK_SECRET_KEY` | Key for KGW/Unigram/zero-width. Only its fingerprint is ever stored or shown. |
| `PROVENANCE_SIGNING_KEY` | Optional HMAC key for content-record signatures |
| `AUTH_ENABLED`, `API_KEYS` | API-key auth with roles `viewer < researcher < admin` |
| `STORE_RAW_TEXT`, `DATA_RETENTION_DAYS` | Privacy controls |

## API

Interactive docs at `/docs` (OpenAPI). Required endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | DB connectivity, providers, version |
| GET | `/api/v1/providers` | Provider status + method/attack/detector catalogue |
| POST | `/api/v1/content/ingest` | Register text (label requires `label_source`) |
| POST | `/api/v1/content/generate` | Generate via `hf_local` (optionally watermarked) or `gemini` |
| POST | `/api/v1/watermarks/embed` · `/detect` | Embed / detect (by text or content id) |
| POST | `/api/v1/attacks/run` | Run one attack; with `content_id` records lineage |
| POST/GET | `/api/v1/experiments` | Create (202 + id) / list (paginated, filter by status) |
| GET | `/api/v1/experiments/{id}` · `/results` | Status & progress · results JSON (409 until succeeded) |
| GET | `/api/v1/provenance/{content_id}` | Lineage, watermark runs, transformations, detections, integrity |
| GET | `/api/v1/reports/{experiment_id}` | Markdown report (`?format=json` also) |

Additional: `POST /experiments/validate`, `/experiments/{id}/cancel`, `/experiments/{id}/rerun`,
`GET /experiments/compare?ids=`, `/experiments/{id}/artifacts[/{name}]`, `POST /detectors/classify`,
`POST /provenance/{id}/verify`, `GET /datasets`, `GET /audit/verify` (admin),
`POST /admin/retention/purge` (admin). Errors share one shape:
`{"error": {"code", "message", "details", "request_id"}}`. See `docs/API.md`.

## Repository layout

```
backend/
  api/          FastAPI app, routers, schemas, auth, middleware
  core/         settings, JSON logging + redaction, hashing, errors, storage, telemetry
  models/       SQLAlchemy ORM (9 tables)
  db/           engine/session, Alembic migrations
  data/         C4/JSONL sources, group-aware splits, leakage audit, versioned dataset builder
  providers/    HF local models, Gemini (retries/timeouts/quota handling)
  watermark/    PRF, KGW/Unigram, zero-width, registry
  attacks/      deterministic + model/provider attacks, registry
  detectors/    TF-IDF LR, LM-feature LR, Binoculars
  evaluation/   metrics + CIs, quality, Markdown report renderer
  provenance/   registry & lineage, HMAC signing, audit chain
  experiments/  config schema, runner, executor, MLflow tracking, CLI
frontend/       Streamlit dashboard (10 sections) + Plotly charts
experiments/configs/  smoke, main_benchmark, delta_ablation
datasets/manifests/   published dataset manifests
reports/        generated reports (published results under reports/published/)
docs/           architecture, methodology, dataset/model cards, security, API, results
tests/          unit + integration (offline, toy model); PostgreSQL via TEST_DATABASE_URL
```

## Testing

```bash
pytest -q                                                    # SQLite, offline
TEST_DATABASE_URL=postgresql+psycopg://user@host/db pytest   # whole suite on PostgreSQL
TEST_POSTGRES_URL=... pytest tests/integration/test_migrations.py
```

Tests never download models or call external APIs: a tiny randomly initialised GPT-2 with an
in-memory tokenizer stands in for real models, and providers are faked.

## Documentation

- `docs/ARCHITECTURE.md` — components, data flow, ER diagram, design decisions
- `docs/METHODOLOGY.md` — research questions, protocol, metrics, threats to validity (dissertation-ready)
- `docs/RESULTS.md` — *(pending: written after the main benchmark run completes)*
- `docs/DATASET_CARD.md`, `docs/MODEL_CARDS.md`
- `docs/SECURITY.md` — threat model, privacy, failure modes
- `docs/API.md` — endpoint reference with examples
- `docs/PHASE6_MODALITY_DECISION.md` — image/audio scope decision

## Licence

Code: MIT. Data and models retain their own licences (see the dataset and model cards).
