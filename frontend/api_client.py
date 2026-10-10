"""HTTP client for the MAI-13 backend. Runs server-side inside Streamlit; the API key never reaches the browser."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
_API_KEY = os.getenv("FRONTEND_API_KEY", "")


@dataclass
class ApiResult:
    ok: bool
    status: int
    data: Any = None
    error_code: str | None = None
    message: str | None = None
    details: dict | None = None
    request_id: str | None = None

    @property
    def actionable(self) -> str:
        hints = {
            "provider_unavailable": "Configure the provider (e.g. GEMINI_API_KEY in .env) or choose another provider.",
            "provider_quota_exceeded": "The provider quota is exhausted; wait or switch provider.",
            "provider_rate_limited": "The provider is rate limiting; retry shortly.",
            "provider_timeout": "The provider timed out; retry or increase PROVIDER_TIMEOUT_S.",
            "validation_failed": "Fix the highlighted fields and resubmit.",
            "unauthorized": "Set FRONTEND_API_KEY for the dashboard process.",
            "forbidden": "Your API key's role does not allow this action.",
            "rate_limited": "Too many requests; wait a minute.",
            "conflict": "The resource is not in the required state yet.",
            "connection_error": f"Is the API running at {API_BASE_URL}? Start it with `uvicorn backend.api.main:app`.",
        }
        return hints.get(self.error_code or "", "See details below.")


def _headers() -> dict[str, str]:
    return {"X-API-Key": _API_KEY} if _API_KEY else {}


def request(
    method: str, path: str, *, json: Any = None, params: dict | None = None, timeout: float = 120.0, raw: bool = False
) -> ApiResult:
    try:
        r = httpx.request(
            method, f"{API_BASE_URL}{path}", json=json, params=params, headers=_headers(), timeout=timeout
        )
    except httpx.TimeoutException:
        return ApiResult(False, 0, error_code="timeout", message=f"Request timed out after {timeout:.0f}s")
    except httpx.HTTPError as exc:
        return ApiResult(False, 0, error_code="connection_error", message=f"Cannot reach API: {exc}")
    rid = r.headers.get("X-Request-ID")
    if r.is_success:
        return ApiResult(True, r.status_code, r.text if raw else r.json(), request_id=rid)
    try:
        err = r.json().get("error", {})
    except ValueError:
        err = {"code": "http_error", "message": r.text[:500]}
    return ApiResult(
        False,
        r.status_code,
        error_code=err.get("code"),
        message=err.get("message"),
        details=err.get("details"),
        request_id=err.get("request_id") or rid,
    )


def get(path: str, **kw) -> ApiResult:
    return request("GET", path, **kw)


def post(path: str, json: Any = None, **kw) -> ApiResult:
    return request("POST", path, json=json, **kw)
