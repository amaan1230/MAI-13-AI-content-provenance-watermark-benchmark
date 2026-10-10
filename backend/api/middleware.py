"""Request-ID propagation, structured access logs, tracing spans and rate limiting."""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from backend.core.config import get_settings
from backend.core.logging import get_logger, request_id_var
from backend.core.telemetry import get_tracer

log = get_logger("backend.api.access")
_RID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = request.headers.get("X-Request-ID", "")
        rid = incoming if _RID_RE.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        tracer = get_tracer()
        try:
            with tracer.start_as_current_span(f"{request.method} {request.url.path}") as span:
                span.set_attribute("http.method", request.method)
                span.set_attribute("http.route", request.url.path)
                span.set_attribute("request.id", rid)
                response = await call_next(request)
                span.set_attribute("http.status_code", response.status_code)
        except Exception:
            log.exception("unhandled error", extra={"path": request.url.path, "method": request.method})
            response = JSONResponse(
                {
                    "error": {
                        "code": "internal_error",
                        "message": "Internal server error",
                        "details": {},
                        "request_id": rid,
                    }
                },
                status_code=500,
            )
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        log.info(
            "request",
            extra={
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,  # query strings are not logged (may carry content)
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
            },
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window limiter per client (API key if present, else IP). In-process only."""

    def __init__(self, app, per_minute: int | None = None):
        super().__init__(app)
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    async def dispatch(self, request: Request, call_next) -> Response:
        limit = self.per_minute if self.per_minute is not None else get_settings().rate_limit_per_minute
        if limit <= 0 or not request.url.path.startswith("/api/"):
            return await call_next(request)
        client = request.headers.get("X-API-Key") or (request.client.host if request.client else "unknown")
        key = str(hash(client))  # never keep raw keys in memory structures that could be dumped
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= limit:
                retry = max(1, int(60 - (now - q[0])))
                return JSONResponse(
                    {
                        "error": {
                            "code": "rate_limited",
                            "message": f"Rate limit of {limit}/min exceeded",
                            "details": {"retry_after_s": retry},
                            "request_id": request_id_var.get(),
                        }
                    },
                    status_code=429,
                    headers={"Retry-After": str(retry)},
                )
            q.append(now)
        return await call_next(request)
