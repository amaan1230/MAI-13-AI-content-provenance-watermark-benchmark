# Security, privacy and failure-mode analysis

## 1. What the system does and does not prove

| Artifact | What it supports | What it does **not** support |
|---|---|---|
| Content hash (SHA-256) | The stored bytes are the bytes that were registered. | Who wrote them, or when they were first written. |
| HMAC record signature | The record (hash, label, parent, model) was written by a holder of `PROVENANCE_SIGNING_KEY` and not altered since. | Third-party verification (symmetric key) or authorship. |
| Audit hash chain | Deletion/edit of audit rows after the fact is detectable. | Protection against an attacker who controls the DB *and* rewrites the whole chain; there is no external anchor. |
| Watermark detection (KGW/Unigram) | Statistical evidence (p-value) that text was sampled with this key and tokenizer. | That a specific person or model produced it; absence of a signal proves nothing. |
| AI-text classifier score | A domain- and generator-specific estimate. | Any decision about an individual without human review; calibration out of distribution. |

These caveats are surfaced in the API (`interpretation`, `disclaimer` fields), the dashboard and every report.

## 2. Threat model (text MVP)

| Threat | Mitigation | Residual risk |
|---|---|---|
| Secret leakage via logs | `SecretStr` settings; regex redaction of key/token patterns in every log line; query strings never logged; keys never returned by the API (only 16-hex fingerprints). | Secrets embedded in free text submitted by users are redacted only if they match known patterns. |
| Secret leakage via repo | `.env` git-ignored; `.env.example` has placeholders; CI greps for Google/HF key patterns. | Other secret formats are not scanned (add `gitleaks` for a fuller scan). |
| Unauthorised API use | Optional API-key auth (`AUTH_ENABLED`), constant-time comparison, roles viewer < researcher < admin; admin-only audit verification and retention purge. | Keys are static env values; no rotation/expiry. Use a reverse proxy with TLS for any network exposure. |
| Abuse / DoS | Per-client sliding-window rate limiter; `MAX_TEXT_CHARS`; bounded `max_new_tokens`; bounded job pool; provider timeouts and bounded retries with backoff. | Rate limiter is in-process (per worker). Model inference is CPU-bound, so long requests can still saturate a small host. |
| Malicious artifact deserialisation | Detector pickles are only loaded from this server's artifact store after SHA-256 verification against the DB record; model artifacts are not downloadable via the API. | An attacker with write access to both the artifact store and the DB could plant a pickle. |
| Path traversal | `LocalArtifactStore` rejects keys that resolve outside its root. | — |
| Watermark key compromise | Key held only in env; only fingerprint stored. A leaked key allows both **spoofing** (watermarking human text) and **scrubbing** (targeted removal). | Rotate by changing `WATERMARK_SECRET_KEY`; old texts then need the old key to verify (record the fingerprint). |
| Spoofing the zero-width baseline | Documented and tested (`test_zero_width_is_spoofable`): marks can be copied to human text. | By design — the baseline exists to show this failure mode. |
| Dataset poisoning / label noise | Labels require `label_source`; dataset files are hash-verified; leakage audits. | C4 may contain machine-written pages; generator outputs are labelled by construction. |

## 3. Privacy and retention

- `STORE_RAW_TEXT=false` stores only hashes, lengths and lineage; detection/attacks on stored content then
  require the caller to resubmit the text.
- `POST /api/v1/admin/retention/purge?older_than_days=N` (admin) nulls raw text older than N days while
  keeping hashes, lineage and audit records; the purge itself is audited.
- Logs contain IDs, status codes and timings, never request bodies.
- C4 documents carry their original URL in `meta.url` for attribution; do not ingest private data
  without a lawful basis and consider `STORE_RAW_TEXT=false`.

## 4. Failure modes and how they surface

| Failure | Behaviour |
|---|---|
| Missing provider key (Gemini) | Provider reports `available=false` with an actionable message; API returns 503 `provider_unavailable`; experiments mark that attack `skipped` with the reason and report no numbers for it. |
| Provider timeout / 5xx | Retried with exponential backoff + jitter up to `PROVIDER_MAX_RETRIES`, then 504 `provider_timeout`; inside experiments the individual transformation is `failed` and counted. |
| Rate limit (429) | Retried honouring `retry in Ns` hints; then 429 `provider_rate_limited`. |
| Quota exhausted | Not retried (cannot succeed); 429 `provider_quota_exceeded`. |
| Empty / blocked model response | `provider_invalid_response` (retried, then surfaced). |
| Model not in local cache with `HF_OFFLINE=true` | 503 `provider_unavailable` naming the model and the download script. |
| Too few tokens to score | Decision `insufficient`, counted as not detected; reported per kind and per attack. |
| Too few samples for a metric | Metric is `null` with a `reason` (e.g. "need >= 2 samples per class"); never imputed. |
| Threshold resolution (1/n_neg > target FPR) | Warning in results and report. |
| Experiment crash | Status `failed` with stage, error code and truncated traceback; partial lineage remains for analysis; audit event recorded. |
| Server restart mid-run | On startup: running → failed ("interrupted, use rerun"); pending → resubmitted. |
| Cancel | Checked between batches and stages; status `cancelled`. |
| Corrupted artifact / dataset cache | SHA-256 mismatch → 409 / `ValidationFailed`, never silently used. |

## 5. Dependency hygiene

- `requirements.txt` declares ranges; `requirements.lock` pins the exact resolved set used for the reported runs.
- CI runs `pip-audit` against `requirements.txt` on every push.
- The Docker image runs as an unprivileged user with data on mounted volumes.
