"""Structured (JSON) logging with request/experiment correlation IDs and secret redaction."""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
from datetime import UTC, datetime

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
experiment_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("experiment_id", default=None)

# Patterns that look like credentials. Applied to every formatted message.
_REDACT_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),  # Google API keys
    re.compile(r"hf_[0-9A-Za-z]{20,}"),  # Hugging Face tokens
    re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization)(['\"]?\s*[:=]\s*['\"]?)([^\s'\",}]+)"),
]


def redact(text: str) -> str:
    for pat in _REDACT_PATTERNS:
        if pat.groups >= 3:
            text = pat.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
        else:
            text = pat.sub("[REDACTED]", text)
    return text


_STD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact(record.getMessage()),
        }
        if rid := request_id_var.get():
            payload["request_id"] = rid
        if eid := experiment_id_var.get():
            payload["experiment_id"] = eid
        for k, v in record.__dict__.items():
            if k not in _STD_ATTRS and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {k: v for k, v in record.__dict__.items() if k not in _STD_ATTRS and not k.startswith("_")}
        if extras:
            base += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        ids = []
        if rid := request_id_var.get():
            ids.append(f"rid={rid}")
        if eid := experiment_id_var.get():
            ids.append(f"exp={eid}")
        return redact(f"{base} {' '.join(ids)}".rstrip())


_configured = False


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    global _configured
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter() if fmt == "json" else TextFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    if _configured:
        for h in list(root.handlers):
            root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)
    # Third-party libraries are noisy at INFO.
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub", "datasets", "alembic"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, logging.getLevelName(level)))
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
