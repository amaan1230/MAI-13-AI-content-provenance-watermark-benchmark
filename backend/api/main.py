"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import __version__
from backend.api.middleware import RateLimitMiddleware, RequestContextMiddleware
from backend.api.routers import attacks, content, experiments, provenance, system, watermarks
from backend.core.config import get_settings
from backend.core.errors import MAIError
from backend.core.logging import configure_logging, get_logger, request_id_var
from backend.core.telemetry import setup_tracing

log = get_logger(__name__)

DESCRIPTION = """
Research platform for **AI content provenance and watermark robustness** (text modality).

* Watermark embedding/detection (KGW, Unigram, zero-width baseline)
* Attack engine (truncation, edits, formatting, lexical substitution, back-translation, paraphrase)
* AI-vs-human text classifiers, provenance registry with lineage and hash-chained audit log
* Reproducible experiments with MLflow tracking and generated reports

Watermark detection and AI-text classification are separate tasks; neither constitutes proof of authorship.
"""


def _error(status: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message, "details": details or {}, "request_id": request_id_var.get()}},
        status_code=status,
    )


def create_app(*, run_migrations: bool = True, start_executor: bool = True) -> FastAPI:
    s = get_settings()
    configure_logging(s.log_level, s.log_format)
    setup_tracing()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s.ensure_dirs()
        if run_migrations:
            from backend.db.migrate import upgrade

            upgrade()
        if start_executor:
            from backend.experiments.executor import get_executor

            get_executor().recover()
        log.info("api started", extra={"version": __version__, "env": s.app_env, "auth": s.auth_enabled})
        yield
        if start_executor:
            from backend.experiments.executor import get_executor

            get_executor().shutdown()

    app = FastAPI(
        title="MAI-13 Provenance & Watermark Benchmark", version=__version__, description=DESCRIPTION, lifespan=lifespan
    )

    @app.exception_handler(MAIError)
    async def _mai(_: Request, exc: MAIError):
        if exc.status_code >= 500:
            log.warning("domain error", extra={"code": exc.code, "error_message": exc.message})
        return _error(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        errors = [{"loc": ".".join(str(x) for x in e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return _error(422, "validation_failed", "Request validation failed", {"errors": errors})

    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException):
        resp = _error(
            exc.status_code,
            {401: "unauthorized", 403: "forbidden", 404: "not_found"}.get(exc.status_code, "http_error"),
            str(exc.detail),
        )
        for k, v in (exc.headers or {}).items():
            resp.headers[k] = v
        return resp

    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origins,
        allow_methods=["*"],
        allow_headers=["X-API-Key", "Content-Type", "X-Request-ID"],
    )
    app.add_middleware(RequestContextMiddleware)

    for r in (system, content, watermarks, attacks, experiments, provenance):
        app.include_router(r.router)
    return app


app = create_app()
