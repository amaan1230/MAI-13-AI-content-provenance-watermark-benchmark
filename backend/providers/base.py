"""Text-generation provider interface with bounded retries and error classification."""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from backend.core.errors import (
    ProviderError,
    ProviderInvalidResponse,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)
from backend.core.logging import get_logger

log = get_logger(__name__)
T = TypeVar("T")

# Retrying a quota or configuration error cannot succeed, so only these are retried.
RETRYABLE = (ProviderTimeout, ProviderRateLimited, ProviderInvalidResponse)


@dataclass
class ProviderResult:
    text: str
    provider: str
    model: str
    latency_ms: float
    attempts: int
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderStatus:
    name: str
    available: bool
    kind: str  # "remote" | "local"
    capabilities: list[str]
    model: str | None
    message: str


class TextProvider(ABC):
    name: str
    kind: str

    @abstractmethod
    def status(self) -> ProviderStatus: ...

    @abstractmethod
    def _complete(
        self, prompt: str, *, temperature: float, max_output_tokens: int, timeout_s: float
    ) -> tuple[str, dict]:
        """Single attempt. Must raise a ProviderError subclass on failure."""

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_output_tokens: int = 512,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
    ) -> ProviderResult:
        if not self.status().available:
            raise ProviderUnavailable(self.status().message, details={"provider": self.name})
        t0 = time.perf_counter()

        def attempt() -> tuple[str, dict]:
            text, usage = self._complete(
                prompt, temperature=temperature, max_output_tokens=max_output_tokens, timeout_s=timeout_s
            )
            if not isinstance(text, str) or not text.strip():
                raise ProviderInvalidResponse("Provider returned an empty response", details={"provider": self.name})
            return text, usage

        (text, usage), attempts = with_retries(attempt, max_retries=max_retries, backoff_base_s=backoff_base_s)
        return ProviderResult(
            text=text.strip(),
            provider=self.name,
            model=self.status().model or "",
            latency_ms=(time.perf_counter() - t0) * 1000,
            attempts=attempts,
            usage=usage,
        )


def with_retries(
    fn: Callable[[], T],
    *,
    max_retries: int,
    backoff_base_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[T, int]:
    """Run `fn` with exponential backoff + jitter on retryable provider errors."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn(), attempt
        except RETRYABLE as exc:
            if attempt > max_retries:
                log.warning("provider retries exhausted", extra={"attempts": attempt, "error": exc.code})
                raise
            delay = backoff_base_s * (2 ** (attempt - 1)) * (0.5 + random.random())
            retry_after = exc.details.get("retry_after_s") if isinstance(exc, ProviderError) else None
            if retry_after:
                delay = max(delay, float(retry_after))
            log.info("provider retry", extra={"attempt": attempt, "error": exc.code, "delay_s": round(delay, 2)})
            sleep(delay)
