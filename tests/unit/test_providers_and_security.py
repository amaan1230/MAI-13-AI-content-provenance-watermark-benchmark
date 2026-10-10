from __future__ import annotations

import logging

import pytest

from backend.core.config import Settings, reset_settings_cache
from backend.core.errors import (
    ProviderInvalidResponse,
    ProviderQuotaExceeded,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)
from backend.core.logging import JsonFormatter, redact
from backend.providers.base import with_retries
from backend.providers.gemini import GeminiProvider, classify_gemini_error


def test_gemini_disabled_without_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    reset_settings_cache()
    st = GeminiProvider().status()
    assert not st.available and "GEMINI_API_KEY" in st.message
    with pytest.raises(ProviderUnavailable):
        GeminiProvider().complete("hi")


class _Exc(Exception):
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (TimeoutError("read timed out"), ProviderTimeout),
        (_Exc("429 RESOURCE_EXHAUSTED rate limit, retry in 3s", 429), ProviderRateLimited),
        (_Exc("Quota exceeded for quota metric per day", 429), ProviderQuotaExceeded),
        (_Exc("API key not valid", 400), ProviderUnavailable),
        (_Exc("internal", 503), ProviderTimeout),
    ],
)
def test_gemini_error_classification(exc, expected):
    assert isinstance(classify_gemini_error(exc), expected)


def test_rate_limit_retry_after_parsed():
    e = classify_gemini_error(_Exc("429 rate limit. Please retry in 7.5s", 429))
    assert e.details["retry_after_s"] == 7.5


def test_retries_then_success():
    calls = {"n": 0}
    sleeps = []

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderTimeout("t")
        return "ok"

    out, attempts = with_retries(fn, max_retries=3, backoff_base_s=1.0, sleep=sleeps.append)
    assert out == "ok" and attempts == 3 and len(sleeps) == 2
    assert sleeps[1] > 0


def test_quota_is_not_retried():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise ProviderQuotaExceeded("q")

    with pytest.raises(ProviderQuotaExceeded):
        with_retries(fn, max_retries=5, backoff_base_s=0, sleep=lambda s: None)
    assert calls["n"] == 1


def test_retries_bounded():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise ProviderRateLimited("r", details={"retry_after_s": 0})

    with pytest.raises(ProviderRateLimited):
        with_retries(fn, max_retries=2, backoff_base_s=0, sleep=lambda s: None)
    assert calls["n"] == 3


def test_gemini_with_fake_client_empty_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaFAKEFAKEFAKEFAKEFAKEFAKE00")
    reset_settings_cache()

    class Resp:
        text = ""
        candidates = []

    class Client:
        class models:  # noqa: N801
            @staticmethod
            def generate_content(**kw):
                return Resp()

    p = GeminiProvider(client=Client())
    with pytest.raises(ProviderInvalidResponse):
        p.complete("x", max_retries=0)
    reset_settings_cache()


def test_gemini_with_fake_client_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaFAKEFAKEFAKEFAKEFAKEFAKE00")
    reset_settings_cache()

    class Resp:
        text = " generated "
        usage_metadata = None

    class Client:
        class models:  # noqa: N801
            @staticmethod
            def generate_content(**kw):
                return Resp()

    r = GeminiProvider(client=Client()).complete("x")
    assert r.text == "generated" and r.attempts == 1
    reset_settings_cache()


def test_secrets_not_in_repr_or_dump():
    s = Settings(gemini_api_key="AIzaSECRETSECRETSECRETSECRET12", watermark_secret_key="wm-secret")
    assert "AIzaSECRET" not in repr(s)
    assert "wm-secret" not in str(s.model_dump())


def test_log_redaction():
    assert "[REDACTED]" in redact("key=AIzaSyA1234567890abcdefghijklmnop")
    assert "hf_" not in redact("token hf_abcdefghijklmnopqrstuvwxyz123")
    rec = logging.LogRecord("x", logging.INFO, "", 0, "api_key=supersecret123 used", (), None)
    assert "supersecret123" not in JsonFormatter().format(rec)


def test_api_keys_parsing():
    s = Settings(api_keys='{"a": "admin", "b": "viewer"}')
    assert s.api_key_roles() == {"a": "admin", "b": "viewer"}
    with pytest.raises(ValueError):
        Settings(api_keys='{"a": "root"}').api_key_roles()
    with pytest.raises(ValueError):
        Settings(api_keys="not json").api_key_roles()
