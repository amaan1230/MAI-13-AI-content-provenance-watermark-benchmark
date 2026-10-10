from __future__ import annotations

import json

import pytest

from backend.core.errors import ValidationFailed
from backend.data.leakage import LeakItem, check_leakage
from backend.data.sources import read_jsonl_records, split_document
from backend.data.splits import assign_split, assign_splits, validate_fractions
from tests.conftest import random_words


def test_split_is_deterministic_and_seed_dependent():
    groups = [f"g{i}" for i in range(500)]
    a = assign_splits(groups, seed=1)
    assert a == assign_splits(groups, seed=1)
    assert a != assign_splits(groups, seed=2)
    frac = {k: sum(1 for v in a.values() if v == k) / 500 for k in ("train", "val", "test")}
    assert abs(frac["train"] - 0.6) < 0.07 and abs(frac["test"] - 0.2) < 0.06


def test_invalid_fractions():
    with pytest.raises(ValueError):
        validate_fractions({"train": 0.5, "val": 0.2, "test": 0.2})
    with pytest.raises(ValueError):
        assign_split("g", 0, {"train": 1.0, "test": 0.0})


def test_leakage_detects_exact_and_near_duplicates_across_splits():
    base = random_words(120, 1)
    near = base.replace(base.split()[60], "zzz", 1)  # one-word edit
    items = [
        LeakItem("a", base, "train", "g1"),
        LeakItem("b", base.upper(), "test", "g2"),  # exact after normalisation
        LeakItem("c", near, "val", "g3"),  # near-duplicate
        LeakItem("d", random_words(120, 2), "test", "g4"),
    ]
    r = check_leakage(items, threshold=0.8)
    assert r["exact_cross_split_duplicates"] == 1
    assert r["near_cross_split_duplicates"] >= 1
    assert not r["leakage_free"]


def test_leakage_clean_and_same_group_allowed():
    t = random_words(100, 3)
    items = [
        LeakItem("a", t, "train", "g1"),
        LeakItem("b", t + " w1", "train", "g1"),
        LeakItem("c", random_words(100, 4), "test", "g2"),
    ]
    assert check_leakage(items)["leakage_free"]


def test_group_crossing_splits_is_flagged():
    items = [LeakItem("a", random_words(50, 1), "train", "g1"), LeakItem("b", random_words(50, 2), "test", "g1")]
    assert check_leakage(items)["groups_crossing_splits"] == 1


def test_split_document_preserves_text():
    text = "Hello,  world! This is\na test of splitting documents properly today."
    p, c = split_document(text, 2, 3)
    assert p == "Hello,  world!"
    assert c == "This is\na"
    assert split_document("too short", 5, 5) is None


def test_jsonl_requires_provenance_fields(tmp_path):
    f = tmp_path / "d.jsonl"
    f.write_text(json.dumps({"text": random_words(80, 1), "source": "x", "license": "CC0"}) + "\n")
    with pytest.raises(ValidationFailed, match="label_source"):
        list(read_jsonl_records(f, prompt_words=10, continuation_words=20))
    f.write_text(json.dumps({"text": random_words(80, 1), "source": "x", "license": "CC0", "label_source": "y"}) + "\n")
    recs = list(read_jsonl_records(f, prompt_words=10, continuation_words=20))
    assert len(recs) == 1 and len(recs[0].human_text.split()) == 20
