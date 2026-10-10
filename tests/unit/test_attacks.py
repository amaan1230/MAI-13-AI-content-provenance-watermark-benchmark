from __future__ import annotations

import pytest

from backend.attacks.registry import ATTACKS, build_attack
from backend.core.errors import ProviderTimeout, ValidationFailed
from backend.providers.base import ProviderStatus, TextProvider
from backend.providers.registry import reset_providers, set_provider

TEXT = (
    "The committee approved the new budget on Tuesday. Members said the plan would fund schools, "
    "repair roads and expand public transport across the region over the next five years."
)

DETERMINISTIC = ["truncation", "word_deletion", "word_swap", "word_insertion", "char_noise", "formatting"]


@pytest.mark.parametrize("name", DETERMINISTIC)
def test_deterministic_attacks_are_seeded(name):
    a = build_attack(name, {"rate": 0.3} if name not in ("truncation", "formatting") else {})
    r1, r2 = a.apply(TEXT, seed=7), a.apply(TEXT, seed=7)
    assert r1.status == "succeeded" and r1.text == r2.text
    if name not in ("formatting",):
        r3 = a.apply(TEXT, seed=8)
        assert r3.status == "succeeded"


@pytest.mark.parametrize("name", ["word_deletion", "word_swap", "char_noise", "word_insertion"])
def test_edit_attacks_change_text(name):
    assert build_attack(name, {"rate": 0.5}).apply(TEXT, seed=1).text != TEXT


def test_rate_zero_is_identity_for_edits():
    for name in ("word_deletion", "word_swap", "word_insertion", "char_noise"):
        assert build_attack(name, {"rate": 0.0}).apply(TEXT, 0).text == " ".join(TEXT.split())


def test_truncation_lengths():
    n = len(TEXT.split())
    r = build_attack("truncation", {"keep_fraction": 0.5}).apply(TEXT)
    assert len(r.text.split()) == round(n * 0.5)
    r = build_attack("truncation", {"keep_fraction": None, "max_words": 5, "keep": "end"}).apply(TEXT)
    assert r.text.split() == TEXT.split()[-5:]


def test_formatting_removes_zero_width_and_homoglyphs_change_bytes():
    zw = "a​b‌c  d"
    r = build_attack("formatting", {"operations": ["normalize_unicode", "collapse_whitespace"]}).apply(zw)
    assert r.text == "abc d"
    assert r.meta["removed_format_chars"] == 2
    h = build_attack("formatting", {"operations": ["homoglyphs"], "homoglyph_rate": 1.0}).apply("peace")
    assert h.text != "peace" and len(h.text) == 5


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_input_skipped(text):
    r = build_attack("truncation").apply(text)
    assert r.status == "skipped" and r.text is None


def test_unknown_attack_and_bad_config():
    with pytest.raises(ValidationFailed):
        build_attack("nope")
    with pytest.raises(ValidationFailed):
        build_attack("word_deletion", {"rate": 2.0})
    with pytest.raises(ValidationFailed):
        build_attack("truncation", {"bogus": 1})


def test_registry_has_all_required_attack_types():
    types = {cls.attack_type for cls in ATTACKS.values()}
    assert {"paraphrase", "translation", "truncation", "edit", "formatting"} <= types


# ----------------------------------------------------------------------------- provider-backed attacks
class FakeProvider(TextProvider):
    name = "gemini"
    kind = "remote"

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = 0

    def status(self):
        return ProviderStatus(self.name, True, self.kind, ["paraphrase"], "fake-model", "ok")

    def _complete(self, prompt, *, temperature, max_output_tokens, timeout_s):
        self.calls += 1
        return self.behaviour(prompt, self.calls)


@pytest.fixture()
def restore_providers():
    yield
    reset_providers()


def test_paraphrase_without_api_key_fails_cleanly(restore_providers):
    reset_providers()
    r = build_attack("paraphrase", {"provider": "gemini"}).apply(TEXT)
    assert r.status == "failed"
    assert "provider_unavailable" in r.error
    assert "GEMINI_API_KEY" in r.error


def test_paraphrase_with_provider(restore_providers, monkeypatch):
    monkeypatch.setenv("PROVIDER_BACKOFF_BASE_S", "0")
    from backend.core.config import reset_settings_cache

    reset_settings_cache()
    set_provider(FakeProvider(lambda p, n: ("A rewritten passage about the budget.", {})))
    r = build_attack("paraphrase", {"provider": "gemini"}).apply(TEXT)
    assert r.status == "succeeded"
    assert r.text == "A rewritten passage about the budget."
    assert r.meta["model"] == "fake-model"


def test_paraphrase_timeout_exhausts_retries(restore_providers, monkeypatch):
    monkeypatch.setenv("PROVIDER_BACKOFF_BASE_S", "0")
    monkeypatch.setenv("PROVIDER_MAX_RETRIES", "2")
    from backend.core.config import reset_settings_cache

    reset_settings_cache()

    def boom(p, n):
        raise ProviderTimeout("slow")

    fp = FakeProvider(boom)
    set_provider(fp)
    r = build_attack("paraphrase", {"provider": "gemini"}).apply(TEXT)
    assert r.status == "failed" and "provider_timeout" in r.error
    assert fp.calls == 3  # 1 + 2 retries
    reset_settings_cache()
