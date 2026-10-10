from __future__ import annotations

import json

import pytest

from tests.conftest import TOY_MODEL_NAME, random_words


@pytest.fixture()
def toy_dataset(settings_env):
    """Synthetic JSONL corpus with explicit (test-only) provenance fields."""
    path = settings_env / "toy.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i in range(30):
            fh.write(
                json.dumps(
                    {
                        "id": f"doc{i}",
                        "text": random_words(60, 100 + i),
                        "source": "synthetic-test-fixture",
                        "license": "CC0 (generated for tests)",
                        "label_source": "Synthetic random-word text written by the test suite (human stand-in)",
                    }
                )
                + "\n"
            )
    return path


def toy_experiment_config(dataset_path, **overrides) -> dict:
    cfg = {
        "name": "toy-e2e",
        "experiment_type": "full",
        "seed": 7,
        "dataset": {
            "source": "jsonl",
            "path": str(dataset_path),
            "n_samples": 30,
            "prompt_words": 10,
            "continuation_words": 40,
        },
        "generator": {
            "model_name": TOY_MODEL_NAME,
            "max_new_tokens": 50,
            "min_new_tokens": 50,
            "temperature": 1.0,
            "top_p": 1.0,
            "batch_size": 8,
        },
        "watermarks": [
            {"method": "kgw", "config": {"min_tokens": 10}},
            {"method": "unigram", "config": {"min_tokens": 10}},
            {"method": "zero_width"},
        ],
        "attacks": [
            {"name": "truncation", "config": {"keep_fraction": 0.5}},
            {"name": "word_deletion", "config": {"rate": 0.2}},
            {"name": "formatting", "config": {"operations": ["normalize_unicode"]}},
            {"name": "paraphrase", "config": {"provider": "gemini"}},  # no key in tests -> skipped
        ],
        "detectors": [
            {"name": "tfidf_lr", "is_baseline": True},
            {"name": "lm_features_lr", "params": {"observer": TOY_MODEL_NAME, "performer": None}},
        ],
        "evaluation": {
            "target_fprs": [0.05, 0.1],
            "primary_fpr": 0.1,
            "n_bootstrap": 50,
            "oracle_model": TOY_MODEL_NAME,
            "semantic_similarity": True,
        },
    }
    cfg.update(overrides)
    return cfg
