from __future__ import annotations

import pytest

from backend.core.errors import InsufficientData, ValidationFailed
from backend.detectors.registry import build_detector
from backend.experiments.config import ExperimentConfig
from backend.experiments.service import validate_config
from tests.conftest import TOY_MODEL_NAME, random_words


def _synthetic(n=40):
    human = [f"I think {random_words(30, i)} honestly" for i in range(n)]
    ai = [f"In conclusion {random_words(30, 1000 + i)} furthermore" for i in range(n)]
    return human + ai, [0] * n + [1] * n


def test_tfidf_separates_synthetic_classes():
    texts, y = _synthetic()
    d = build_detector("tfidf_lr").fit(texts, y)
    s = d.score(["I think w1 w2 honestly", "In conclusion w1 w2 furthermore"])
    assert s[1] > s[0]


def test_detector_requires_both_classes():
    with pytest.raises(InsufficientData):
        build_detector("tfidf_lr").fit(["a", "b", "c"], [1, 1, 1])
    with pytest.raises(ValidationFailed):
        build_detector("tfidf_lr").fit(["a"], [1, 0])


def test_unfitted_detector_rejects_scoring():
    with pytest.raises(ValidationFailed):
        build_detector("tfidf_lr").score(["x"])


def test_lm_feature_detector_runs_offline(toy_lm):
    texts, y = _synthetic(10)
    d = build_detector("lm_features_lr", {"observer": TOY_MODEL_NAME, "performer": None}).fit(texts, y)
    s = d.score(texts[:2])
    assert all(0 <= v <= 1 for v in s)
    assert "binoculars" not in d.coefficients()  # all-NaN feature dropped when no performer


def test_binoculars_zero_shot(toy_lm):
    d = build_detector("binoculars", {"observer": TOY_MODEL_NAME, "performer": TOY_MODEL_NAME})
    s = d.score([random_words(30, 1)])
    assert isinstance(s[0], float)


def test_unknown_detector():
    with pytest.raises(ValidationFailed):
        build_detector("magic")


# ----------------------------------------------------------------------------- experiment config
BASE = {"name": "t", "watermarks": [{"method": "kgw"}], "detectors": [{"name": "tfidf_lr"}]}


def test_valid_config_defaults():
    cfg = ExperimentConfig.model_validate(BASE)
    assert cfg.experiment_type == "full" and cfg.seed == 42
    assert cfg.evaluation.primary_fpr in cfg.evaluation.target_fprs


@pytest.mark.parametrize(
    "patch",
    [
        {"watermarks": []},
        {"detectors": []},
        {"watermarks": [{"method": "kgw"}, {"method": "kgw"}]},
        {"watermarks": [{"method": "nope"}]},
        {"attacks": [{"name": "word_deletion", "config": {"rate": 5}}]},
        {"dataset": {"source": "jsonl"}},
        {"dataset": {"split_fractions": {"train": 0.9, "val": 0.2, "test": 0.1}}},
        {"evaluation": {"target_fprs": [0.01], "primary_fpr": 0.05}},
        {"generator": {"min_new_tokens": 300, "max_new_tokens": 100}},
        {"unexpected": 1},
    ],
)
def test_invalid_configs(patch):
    with pytest.raises(ValidationFailed) as ei:
        validate_config({**BASE, **patch})
    assert ei.value.details["errors"]


def test_duplicate_methods_allowed_with_labels():
    cfg = ExperimentConfig.model_validate(
        {
            **BASE,
            "watermarks": [{"method": "kgw", "label": "a"}, {"method": "kgw", "label": "b", "config": {"delta": 1}}],
        }
    )
    assert [w.key for w in cfg.watermarks] == ["a", "b"]


def test_shipped_configs_validate():
    from pathlib import Path

    from backend.experiments.config import load_config

    paths = sorted(Path("experiments/configs").glob("*.yaml"))
    assert paths, "expected shipped experiment configs"
    for p in paths:
        load_config(p)
