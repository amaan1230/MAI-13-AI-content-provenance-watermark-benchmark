"""Common attack interface. Every attack is a pure function of (text, config, seed)."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from backend.core.errors import MAIError
from backend.core.logging import get_logger

log = get_logger(__name__)

AttackType = Literal["paraphrase", "translation", "truncation", "edit", "formatting"]


class AttackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass
class AttackResult:
    text: str | None
    status: Literal["succeeded", "failed", "skipped"]
    latency_ms: float
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class Attack(ABC):
    name: ClassVar[str]
    attack_type: ClassVar[AttackType]
    version: ClassVar[str] = "1.0.0"
    config_cls: ClassVar[type[AttackConfig]]
    deterministic: ClassVar[bool] = True
    requires: ClassVar[list[str]] = []  # external resources (models / providers)

    def __init__(self, config: dict[str, Any] | AttackConfig | None = None):
        self.config = config if isinstance(config, self.config_cls) else self.config_cls.model_validate(config or {})

    @abstractmethod
    def _apply(self, text: str, seed: int) -> tuple[str, dict[str, Any]]: ...

    def apply(self, text: str, seed: int = 0) -> AttackResult:
        t0 = time.perf_counter()
        if not text or not text.strip():
            return AttackResult(None, "skipped", 0.0, error="empty input text")
        try:
            out, meta = self._apply(text, seed)
        except MAIError as exc:
            return AttackResult(None, "failed", (time.perf_counter() - t0) * 1000, error=f"{exc.code}: {exc.message}")
        except Exception as exc:  # noqa: BLE001 - attacks must never crash a whole experiment
            log.exception("attack failed", extra={"attack": self.name})
            return AttackResult(None, "failed", (time.perf_counter() - t0) * 1000, error=f"{type(exc).__name__}: {exc}")
        if not out or not out.strip():
            return AttackResult(None, "failed", (time.perf_counter() - t0) * 1000, error="attack produced empty text")
        return AttackResult(out, "succeeded", (time.perf_counter() - t0) * 1000, meta=meta)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "attack_type": self.attack_type,
            "version": self.version,
            "deterministic": self.deterministic,
            "requires": self.requires,
            "config": self.config.model_dump(),
        }
