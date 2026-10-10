from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from backend.watermark import prf
from backend.watermark.kgw import GreenListScheme, GreenListWatermark, UnigramWatermark
from backend.watermark.zero_width import ZeroWidthWatermark
from tests.conftest import TOY_MODEL_NAME, random_words

GEN = dict(model_name=TOY_MODEL_NAME, max_new_tokens=100, min_new_tokens=100, temperature=1.0, top_p=1.0, seed=3)


# ----------------------------------------------------------------------------- PRF
def test_prf_is_deterministic_and_version_pinned():
    """Guards against silent changes to green lists (which would break detection of old texts)."""
    seed = prf.context_seed(prf.key_to_int("k"), (42,))
    assert seed == prf.context_seed(prf.key_to_int("k"), (42,))
    scores = prf.uniform_scores(seed, np.array([0, 1, 2]))
    # Golden values computed once with this implementation; must never change.
    assert scores.tolist() == pytest.approx(GOLDEN_SCORES, abs=1e-12)


# Recorded from the reference implementation (v1.0.0). A change here means previously
# watermarked texts would no longer be detectable: bump the method version instead.
GOLDEN_SCORES = [0.05281977541422633, 0.7008590568481013, 0.9248928675316577]


def test_green_fraction_matches_gamma():
    for gamma in (0.1, 0.25, 0.5):
        mask = prf.green_mask(prf.context_seed(123, (7,)), 50_000, gamma)
        assert abs(mask.mean() - gamma) < 0.01


def test_different_contexts_and_keys_give_different_lists():
    a = prf.green_mask(prf.context_seed(1, (5,)), 1000, 0.25)
    b = prf.green_mask(prf.context_seed(1, (6,)), 1000, 0.25)
    c = prf.green_mask(prf.context_seed(2, (5,)), 1000, 0.25)
    assert (a != b).any() and (a != c).any()


def test_z_score_formula():
    s = GreenListScheme("k", gamma=0.25, context_width=1)
    assert s.z_score(25, 100) == pytest.approx(0.0)
    assert s.z_score(50, 100) == pytest.approx(25 / math.sqrt(100 * 0.25 * 0.75))


# ----------------------------------------------------------------------------- KGW / Unigram
@pytest.mark.parametrize("cls", [GreenListWatermark, UnigramWatermark])
def test_embed_then_detect_positive(toy_lm, cls):
    wm = cls(key="secret", lm=toy_lm)
    out = wm.embed_batch(["w1 w2 w3", "w9 w8"], GEN)
    assert len(out) == 2
    for o in out:
        assert o.n_tokens == 100
        r = wm.detect(o.text, GEN)
        assert r.decision == "positive"
        assert r.score > 4
        assert r.p_value < 1e-4


@pytest.mark.parametrize("cls", [GreenListWatermark, UnigramWatermark])
def test_unwatermarked_text_is_negative(toy_lm, cls):
    wm = cls(key="secret", lm=toy_lm)
    for seed in range(5):
        r = wm.detect(random_words(150, seed), GEN)
        assert r.decision == "negative"
        assert abs(r.score) < 4


def test_wrong_key_does_not_detect(toy_lm):
    text = GreenListWatermark(key="right", lm=toy_lm).embed("w1 w2", GEN).text
    r = GreenListWatermark(key="wrong", lm=toy_lm).detect(text, GEN)
    assert r.decision == "negative"


def test_short_text_is_insufficient_not_negative(toy_lm):
    wm = GreenListWatermark(key="k", lm=toy_lm)
    r = wm.detect("w1 w2 w3", {**GEN, "min_tokens": 16})
    assert r.decision == "insufficient"
    assert r.score is None and r.p_value is None


def test_empty_text_is_insufficient(toy_lm):
    r = GreenListWatermark(key="k", lm=toy_lm).detect("", GEN)
    assert r.decision == "insufficient"


def test_repeated_ngrams_counted_once(toy_lm):
    wm = GreenListWatermark(key="k", lm=toy_lm)
    rep = "w1 w2 " * 50
    r_ignore = wm.detect(rep, {**GEN, "ignore_repeated_ngrams": True, "min_tokens": 1})
    r_all = wm.detect(rep, {**GEN, "ignore_repeated_ngrams": False, "min_tokens": 1})
    assert r_ignore.n_tokens_scored == 2  # (w1->w2) and (w2->w1)
    assert r_all.n_tokens_scored == 99


def test_zero_delta_gives_no_signal(toy_lm):
    wm = GreenListWatermark(key="k", lm=toy_lm)
    cfg = {**GEN, "delta": 0.0}
    zs = [wm.detect(o.text, cfg).score for o in wm.embed_batch([f"w{i}" for i in range(6)], cfg)]
    assert abs(float(np.mean(zs))) < 2.5


def test_threshold_controls_decision(toy_lm):
    wm = GreenListWatermark(key="k", lm=toy_lm)
    text = wm.embed("w1", GEN).text
    z = wm.detect(text, GEN).score
    assert wm.detect(text, {**GEN, "z_threshold": z - 0.01}).decision == "positive"
    assert wm.detect(text, {**GEN, "z_threshold": z + 0.01}).decision == "negative"


def test_generation_is_reproducible(toy_lm):
    wm = GreenListWatermark(key="k", lm=toy_lm)
    assert wm.embed("w1 w2", GEN).text == wm.embed("w1 w2", GEN).text


@pytest.mark.parametrize(
    "bad", [{"gamma": 0}, {"gamma": 1.0}, {"delta": -1}, {"min_new_tokens": 500}, {"unknown_field": 1}]
)
def test_invalid_config_rejected(toy_lm, bad):
    with pytest.raises(ValidationError):
        GreenListWatermark(key="k", lm=toy_lm).parse_config({**GEN, **bad})


def test_unigram_forces_context_zero(toy_lm):
    with pytest.raises(ValidationError):
        UnigramWatermark(key="k", lm=toy_lm).parse_config({**GEN, "context_width": 1})


def test_metadata_never_exposes_key(toy_lm):
    md = GreenListWatermark(key="super-secret-key", lm=toy_lm).get_metadata()
    assert "super-secret-key" not in str(md)
    assert len(md["key_fingerprint"]) == 16


# ----------------------------------------------------------------------------- zero-width baseline
def test_zero_width_roundtrip_and_normalisation_fragility():
    from backend.attacks.deterministic import Formatting

    wm = ZeroWidthWatermark(key="k")
    marked = wm.embed("one two three four five six seven eight nine ten eleven twelve").text
    assert marked != "one two three four five six seven eight nine ten eleven twelve"
    assert wm.detect(marked).decision == "positive"
    stripped = Formatting({"operations": ["normalize_unicode"]}).apply(marked).text
    assert wm.detect(stripped).decision == "negative"


def test_zero_width_is_spoofable():
    """Documented weakness: copying the marks onto human text yields a positive."""
    wm = ZeroWidthWatermark(key="k")
    marked = wm.embed("a b c d e f g h").text
    marks = "".join(ch for ch in marked if ch in "​‌")
    assert wm.detect("Completely human sentence. " + marks).decision == "positive"


def test_zero_width_wrong_key_negative():
    marked = ZeroWidthWatermark(key="a").embed("a b c d e f g h i j").text
    r = ZeroWidthWatermark(key="b").detect(marked)
    assert r.decision == "negative" or r.score < 1.0
