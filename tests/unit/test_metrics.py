from __future__ import annotations

import math

import numpy as np
import pytest

from backend.evaluation import metrics as M


def test_auroc_perfect_and_inverted():
    y = [1, 1, 1, 0, 0, 0]
    assert M.auroc(y, [3, 2, 1.5, 1, 0, -1], n_bootstrap=0)["auroc"] == 1.0
    assert M.auroc(y, [-3, -2, -1.5, 1, 2, 3], n_bootstrap=0)["auroc"] == 0.0


def test_auroc_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = [1] * 100 + [0] * 100
    s = list(rng.normal(1, 1, 100)) + list(rng.normal(0, 1, 100))
    r = M.auroc(y, s, n_bootstrap=300, seed=1)
    assert r["ci_low"] <= r["auroc"] <= r["ci_high"]
    assert 0.65 < r["auroc"] < 0.85


def test_auroc_matches_mann_whitney_with_ties():
    pos, neg = np.array([1.0, 2.0, 2.0]), np.array([2.0, 0.0])
    from sklearn.metrics import roc_auc_score

    expected = roc_auc_score([1, 1, 1, 0, 0], [1, 2, 2, 2, 0])
    assert M._auc_from_scores(pos, neg) == pytest.approx(expected)


def test_auroc_insufficient_data_reports_reason():
    r = M.auroc([1, 1, 1], [0.1, 0.2, 0.3])
    assert r["auroc"] is None and "need" in r["reason"]
    r = M.auroc([], [])
    assert r["auroc"] is None


def test_auroc_excludes_missing_scores():
    r = M.auroc([1, 1, 0, 0, 1], [2, 3, 0, 1, None], n_bootstrap=0)
    assert r["auroc"] == 1.0 and r["n_excluded_missing_score"] == 1


def test_threshold_at_fpr_respects_target():
    rng = np.random.default_rng(0)
    neg = list(rng.normal(0, 1, 1000))
    for f in (0.01, 0.05, 0.1):
        t = M.threshold_at_fpr(neg, f)
        assert t["achieved_fpr"] <= f
        assert np.mean(np.array(neg) >= t["threshold"]) <= f


def test_threshold_resolution_warning():
    t = M.threshold_at_fpr([0.1, 0.2, 0.3, 0.4], 0.01)
    assert t["achieved_fpr"] == 0.0
    assert "resolution" in t["reason"]
    assert M.threshold_at_fpr([1.0], 0.05)["threshold"] is None


def test_binary_metrics_counts_and_missing_as_negative():
    y = [1, 1, 1, 0, 0, 0]
    s = [0.9, 0.8, None, 0.7, 0.1, 0.2]
    bm = M.binary_metrics(y, s, 0.5)
    assert (bm["tp"], bm["fn"], bm["fp"], bm["tn"]) == (2, 1, 1, 2)
    assert bm["precision"] == pytest.approx(2 / 3)
    assert bm["recall"] == pytest.approx(2 / 3)
    assert bm["fpr"] == pytest.approx(1 / 3)
    assert bm["n_missing_scores"] == 1


def test_binary_metrics_no_positives_predicted():
    bm = M.binary_metrics([1, 0], [0.1, 0.2], 0.9)
    assert bm["precision"] is None and bm["f1"] is None


def test_wilson_ci():
    lo, hi = M.wilson_ci(0, 100)
    assert lo == 0.0 and 0 < hi < 0.05
    lo, hi = M.wilson_ci(50, 100)
    assert lo < 0.5 < hi
    assert M.wilson_ci(0, 0) == (None, None)


def test_survival_rate():
    r = M.survival_rate([True, True, False, False])
    assert r["survival_rate"] == 0.5 and r["n"] == 4
    assert M.survival_rate([])["survival_rate"] is None


def test_distribution_handles_missing():
    d = M.distribution([1.0, None, 3.0])
    assert d["n"] == 2 and d["n_missing"] == 1 and d["mean"] == 2.0
    assert M.distribution([None])["n"] == 0


def test_paired_difference_detects_better_detector():
    rng = np.random.default_rng(0)
    y = [1] * 80 + [0] * 80
    good = list(rng.normal(2, 1, 80)) + list(rng.normal(0, 1, 80))
    bad = list(rng.normal(0.2, 1, 80)) + list(rng.normal(0, 1, 80))
    r = M.paired_auroc_difference(y, good, bad, n_bootstrap=300)
    assert r["diff"] > 0.2 and r["ci_low"] > 0 and r["p_two_sided"] < 0.05
    same = M.paired_auroc_difference(y, good, good, n_bootstrap=100)
    assert same["diff"] == 0.0


def test_mean_ci_and_nan_handling():
    r = M.mean_ci([1.0, 2.0, 3.0, None], n_bootstrap=200)
    assert r["mean"] == 2.0 and r["n"] == 3 and r["ci_low"] <= 2.0 <= r["ci_high"]
    assert M.mean_ci([])["mean"] is None
    r = M.auroc([1, 1, 0, 0], [1.0, math.nan, 0.0, 0.5], n_bootstrap=0)
    assert r["n_excluded_missing_score"] == 1
