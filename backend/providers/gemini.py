"""Google Gemini provider (optional). Disabled unless GEMINI_API_KEY is set."""

from __future__ import annotations

import re
from typing import Any

from backend.core.config import get_settings
from backend.core.errors import (
    ProviderError,
    ProviderInvalidResponse,
    ProviderQuotaExceeded,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)
from backend.providers.base import ProviderStatus, TextProvider


def classify_gemini_error(exc: Exception) -> ProviderError:
    """Map SDK/HTTP failures onto our provider error taxonomy without leaking secrets."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    msg = str(exc)
    low = msg.lower()
    short = re.sub(r"\s+", " ", msg)[:300]
    if isinstance(exc, TimeoutError) or "timeout" in low or "timed out" in low or code in (408, 504):
        return ProviderTimeout("Gemini request timed out", details={"provider": "gemini"})
    if code == 429 or "resource_exhausted" in low or "rate limit" in low or "quota" in low:
        if "quota" in low and ("exceeded" in low or "billing" in low or "per day" in low):
            return ProviderQuotaExceeded(
                "Gemini quota exhausted; check plan/billing or try later", details={"provider": "gemini"}
            )
        m = re.search(r"retry in ([0-9.]+)s", low)
        return ProviderRateLimited(
            "Gemini rate limit hit",
            details={"provider": "gemini", "retry_after_s": float(m.group(1)) if m else None},
        )
    if code in (401, 403) or "api key" in low or "permission" in low:
        return ProviderUnavailable("Gemini rejected the API key or permissions", details={"provider": "gemini"})
    if code in (500, 502, 503):
        return ProviderTimeout(f"Gemini server error ({code})", details={"provider": "gemini"})
    return ProviderError(f"Gemini error: {short}", details={"provider": "gemini"})


class GeminiProvider(TextProvider):
    name = "gemini"
    kind = "remote"

    def __init__(self, client: Any | None = None):
        self._client = client

    def status(self) -> ProviderStatus:
        s = get_settings()
        if s.gemini_api_key is None or not s.gemini_api_key.get_secret_value().strip():
            return ProviderStatus(
                name=self.name,
                available=False,
                kind=self.kind,
                capabilities=["generate", "paraphrase", "translate"],
                model=s.gemini_model,
                message="Disabled: set GEMINI_API_KEY in .env to enable Gemini generation/paraphrasing.",
            )
        if self._client is None:
            try:
                import google.genai  # noqa: F401
            except ImportError:
                return ProviderStatus(
                    name=self.name,
                    available=False,
                    kind=self.kind,
                    capabilities=[],
                    model=s.gemini_model,
                    message="Disabled: install the 'google-genai' package.",
                )
        return ProviderStatus(
            name=self.name,
            available=True,
            kind=self.kind,
            capabilities=["generate", "paraphrase", "translate"],
            model=s.gemini_model,
            message="Configured",
        )

    def _get_client(self, timeout_s: float):
        if self._client is not None:
            return self._client
        from google import genai
        from google.genai import types

        s = get_settings()
        return genai.Client(
            api_key=s.gemini_api_key.get_secret_value(),
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
        )

    def _complete(self, prompt: str, *, temperature: float, max_output_tokens: int, timeout_s: float):
        s = get_settings()
        try:
            client = self._get_client(timeout_s)
            resp = client.models.generate_content(
                model=s.gemini_model,
                contents=prompt,
                config={"temperature": temperature, "max_output_tokens": max_output_tokens},
            )
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK raises several unrelated exception types
            raise classify_gemini_error(exc) from None
        text = getattr(resp, "text", None)
        if not text:
            reason = None
            try:
                reason = str(resp.candidates[0].finish_reason)
            except (AttributeError, IndexError, TypeError):
                reason = None
            raise ProviderInvalidResponse(
                "Gemini returned no text", details={"provider": "gemini", "finish_reason": reason}
            )
        usage = {}
        if getattr(resp, "usage_metadata", None) is not None:
            um = resp.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", None),
                "output_tokens": getattr(um, "candidates_token_count", None),
            }
        return text, usage
