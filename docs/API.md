# API reference (v1)

The authoritative, always-current reference is the generated OpenAPI document:
`GET /openapi.json`, browsable at `/docs` (Swagger UI) and `/redoc`. This page explains conventions
and gives worked examples.

## Conventions

- **Base path:** `/api/v1` (plus unversioned `/health`).
- **Auth:** when `AUTH_ENABLED=true`, send `X-API-Key: <key>`. Roles: `viewer` (GET), `researcher`
  (create content, run watermarks/attacks/experiments), `admin` (audit verification, retention purge).
- **Request IDs:** send `X-Request-ID` (≤64 chars, `[A-Za-z0-9._-]`) or one is generated; it is echoed
  in the response header, in error bodies and in every log line.
- **Errors:** always
  ```json
  {"error": {"code": "provider_unavailable", "message": "...", "details": {...}, "request_id": "..."}}
  ```
  | HTTP | code examples |
  |---|---|
  | 401 / 403 | `unauthorized`, `forbidden` |
  | 404 | `not_found` |
  | 409 | `conflict` (e.g. results requested before the experiment succeeded) |
  | 422 | `validation_failed` (field list in `details.errors`), `insufficient_data` |
  | 429 | `rate_limited` (with `Retry-After`), `provider_rate_limited`, `provider_quota_exceeded` |
  | 502 / 503 / 504 | `provider_error`, `provider_invalid_response`, `provider_unavailable`, `provider_timeout` |
- **Pagination:** list endpoints take `limit` and `offset` and return `{items, total, limit, offset}`.
- **Long-running work:** `POST /experiments` returns **202** with an ID; poll `GET /experiments/{id}`
  (`status`, `stage`, `progress` ∈ [0,1]).

## Examples

```bash
API=http://localhost:8000

# Generate watermarked text with the local model, then detect it by content id
curl -s -X POST $API/api/v1/content/generate -H 'Content-Type: application/json' -d '{
  "prompt": "The city council met on Tuesday to discuss",
  "max_new_tokens": 150,
  "watermark": {"method": "kgw", "config": {"gamma": 0.25, "delta": 2.0}}}'
# -> {"text": "...", "content_id": "c1...", "watermark": {"method": "kgw", ...}, ...}

curl -s -X POST $API/api/v1/watermarks/detect -H 'Content-Type: application/json' \
  -d '{"method": "kgw", "content_id": "c1..."}'
# -> {"score": 6.8, "p_value": 5e-12, "threshold": 4.0, "decision": "positive",
#     "interpretation": "...statistical evidence, not proof of authorship."}

# Attack it and record lineage
curl -s -X POST $API/api/v1/attacks/run -H 'Content-Type: application/json' \
  -d '{"attack": "word_deletion", "config": {"rate": 0.3}, "content_id": "c1...", "seed": 1}'
# -> {"status": "succeeded", "child_content_id": "c2...", "similarity": 0.87, ...}

# Full lineage of the attacked text
curl -s $API/api/v1/provenance/c2...

# Launch an experiment (body = the YAML config as JSON)
curl -s -X POST $API/api/v1/experiments -H 'Content-Type: application/json' -d '{
  "name": "quick", "experiment_type": "watermark_robustness",
  "dataset": {"n_samples": 30}, "watermarks": [{"method": "kgw"}],
  "attacks": [{"name": "truncation", "config": {"keep_fraction": 0.5}}]}'
# -> 202 {"id": "e1...", "status": "pending", "status_url": "/api/v1/experiments/e1..."}

curl -s $API/api/v1/experiments/e1...            # status / progress
curl -s $API/api/v1/experiments/e1.../results    # 409 until succeeded, then results.json
curl -s $API/api/v1/reports/e1...                # Markdown report
```

## Endpoint summary

| Method | Path | Role | Notes |
|---|---|---|---|
| GET | `/health` | public | `status`, DB check, dialect, providers available |
| GET | `/api/v1/providers` | viewer | provider status + messages; watermark/attack/detector catalogue with JSON schemas |
| GET | `/api/v1/datasets` | viewer | registered dataset versions |
| POST | `/api/v1/content/ingest` | researcher | 201; `label` ≠ unknown requires `label_source` |
| POST | `/api/v1/content/generate` | researcher | `provider`: `hf_local` (supports generation-time watermark) or `gemini` |
| GET | `/api/v1/content`, `/content/{id}` | viewer | paginated, filters `source_type`, `label`, `experiment_id` |
| POST | `/api/v1/watermarks/embed` | researcher | generation-time methods treat `text` as prompt |
| POST | `/api/v1/watermarks/detect` | researcher | exactly one of `text` / `content_id`; persisted only for `content_id` |
| GET | `/api/v1/watermarks/methods` | viewer | method cards |
| POST | `/api/v1/attacks/run` | researcher | exactly one of `text` / `content_id` |
| POST | `/api/v1/experiments` | researcher | 202, queued in background |
| POST | `/api/v1/experiments/validate` | viewer | returns normalised config or 422 with field errors |
| GET | `/api/v1/experiments` | viewer | paginated, `status` filter |
| GET | `/api/v1/experiments/compare?ids=a,b` | viewer | metric summaries side by side |
| GET | `/api/v1/experiments/{id}` | viewer | full record incl. config, environment, error details |
| POST | `/api/v1/experiments/{id}/cancel` · `/rerun` | researcher | 409 if already terminal (cancel) |
| GET | `/api/v1/experiments/{id}/results` | viewer | 409 + status details until succeeded |
| GET | `/api/v1/experiments/{id}/artifacts[/{name}]` | viewer | SHA-256-verified; model pickles not downloadable |
| GET | `/api/v1/reports/{id}` | viewer | `?format=md|json` |
| POST | `/api/v1/detectors/classify` | researcher | score text with a classifier trained in a completed experiment |
| GET | `/api/v1/provenance/{content_id}` | viewer | ancestors, descendants, runs, transformations, detections, integrity |
| POST | `/api/v1/provenance/{content_id}/verify` | viewer | compare a candidate text with the stored hash |
| GET | `/api/v1/audit/verify` | admin | recompute the audit hash chain |
| POST | `/api/v1/admin/retention/purge` | admin | null raw text older than N days |
