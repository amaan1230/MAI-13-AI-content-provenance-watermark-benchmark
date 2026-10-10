"""Detection metrics with confidence intervals and explicit handling of degenerate inputs.

Conventions
- `y` = 1 for the positive class (watermarked / AI), 0 otherwise.
- Scores may be None (e.g. a watermark detector returned "insufficient"). For AUROC
  these samples are excluded and counted; for rate metrics (TPR, survival) they are
  treated as *not detected*, which is the conservative choice for a defender.
- Every function returns `None` values plus a `reason` instead of raising or
  inventing a number when the data cannot support the metric.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

MIN_PER_CLASS = 2


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _clean(y: Sequence[int], s: Sequence[float | None]) -> tuple[np.ndarray, np.ndarray, int]:
    if len(y) != len(s):
        raise ValueError("labels and scores must have the same length")
    pairs = [(int(a), float(b)) for a, b in zip(y, s, strict=True) if b is not None and not math.isnan(float(b))]
    n_missing = len(y) - len(pairs)
    if not pairs:
        return np.array([], dtype=int), np.array([], dtype=float), n_missing
    yy, ss = zip(*pairs, strict=True)
    return np.array(yy), np.array(ss), n_missing


def auroc(
    y: Sequence[int], s: Sequence[float | None], *, n_bootstrap: int = 1000, seed: int = 0, ci: float = 0.95
) -> dict[str, Any]:
    yy, ss, n_missing = _clean(y, s)
    out: dict[str, Any] = {
        "auroc": None,
        "ci_low": None,
        "ci_high": None,
        "n_pos": int((yy == 1).sum()),
        "n_neg": int((yy == 0).sum()),
        "n_excluded_missing_score": n_missing,
        "reason": None,
    }
    if out["n_pos"] < MIN_PER_CLASS or out["n_neg"] < MIN_PER_CLASS:
        out["reason"] = f"need >= {MIN_PER_CLASS} samples per class (pos={out['n_pos']}, neg={out['n_neg']})"
        return out
    out["auroc"] = float(roc_auc_score(yy, ss))
    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        pos, neg = ss[yy == 1], ss[yy == 0]
        vals = []
        for _ in range(n_bootstrap):
            # Stratified resampling keeps both classes present in every replicate.
            bp = rng.choice(pos, size=len(pos), replace=True)
            bn = rng.choice(neg, size=len(neg), replace=True)
            vals.append(_auc_from_scores(bp, bn))
        alpha = (1 - ci) / 2
        out["ci_low"], out["ci_high"] = (float(np.quantile(vals, alpha)), float(np.quantile(vals, 1 - alpha)))
    return out


def _auc_from_scores(pos: np.ndarray, neg: np.ndarray) -> float:
    # Mann-Whitney U formulation, ties counted as 0.5.
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def threshold_at_fpr(neg_scores: Sequence[float | None], target_fpr: float) -> dict[str, Any]:
    """Smallest threshold t (decision: score >= t) whose empirical FPR on negatives <= target.

    Intended for *validation* negatives; the threshold is then frozen for the test set.
    """
    vals = np.array(sorted(float(v) for v in neg_scores if v is not None), dtype=float)
    if len(vals) < MIN_PER_CLASS:
        return {"threshold": None, "achieved_fpr": None, "n_neg": int(len(vals)), "reason": "too few negatives"}
    if target_fpr <= 0:
        return {
            "threshold": float(np.nextafter(vals[-1], np.inf)),
            "achieved_fpr": 0.0,
            "n_neg": len(vals),
            "reason": None,
        }
    # Allowed false positives: floor(target * n). Threshold just above the (n-k)-th largest negative.
    k = int(math.floor(target_fpr * len(vals)))
    if k == 0:
        t = float(np.nextafter(vals[-1], np.inf))
    else:
        t = float(np.nextafter(vals[-k - 1], np.inf)) if k < len(vals) else float(vals[0])
    achieved = float((vals >= t).mean())
    note = None
    if 1 / len(vals) > target_fpr:
        note = f"resolution limited: 1/n_neg={1 / len(vals):.4f} > target_fpr={target_fpr}"
    return {"threshold": t, "achieved_fpr": achieved, "n_neg": int(len(vals)), "reason": note}


def tpr_at_fpr(y: Sequence[int], s: Sequence[float | None], target_fpr: float) -> dict[str, Any]:
    """In-sample TPR at the ROC operating point with FPR <= target (reporting only)."""
    yy, ss, _ = _clean(y, s)
    if (yy == 1).sum() < MIN_PER_CLASS or (yy == 0).sum() < MIN_PER_CLASS:
        return {"tpr": None, "fpr": None, "reason": "insufficient class counts"}
    fpr, tpr, _ = roc_curve(yy, ss)
    ok = fpr <= target_fpr
    return {
        "tpr": float(tpr[ok].max()) if ok.any() else 0.0,
        "fpr": float(fpr[ok].max()) if ok.any() else None,
        "reason": None,
    }


def binary_metrics(y: Sequence[int], s: Sequence[float | None], threshold: float) -> dict[str, Any]:
    """Confusion-matrix metrics at a fixed threshold. Missing scores count as negative predictions."""
    y_arr = np.asarray(y, dtype=int)
    pred = np.array([(v is not None and not math.isnan(float(v)) and float(v) >= threshold) for v in s], dtype=bool)
    tp = int(((y_arr == 1) & pred).sum())
    fp = int(((y_arr == 0) & pred).sum())
    fn = int(((y_arr == 1) & ~pred).sum())
    tn = int(((y_arr == 0) & ~pred).sum())
    n_pos, n_neg = tp + fn, fp + tn

    def div(a, b):
        return a / b if b else None

    precision = div(tp, tp + fp)
    recall = div(tp, n_pos)
    f1 = (
        None
        if precision is None or recall is None or (precision + recall) == 0
        else 2 * precision * recall / (precision + recall)
    )
    fpr_ci = wilson_ci(fp, n_neg)
    tpr_ci = wilson_ci(tp, n_pos)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "precision": precision,
        "recall": recall,
        "tpr": recall,
        "tpr_ci": list(tpr_ci),
        "f1": f1,
        "fpr": div(fp, n_neg),
        "fpr_ci": list(fpr_ci),
        "accuracy": div(tp + tn, n_pos + n_neg),
        "n_missing_scores": int(sum(1 for v in s if v is None)),
    }


def survival_rate(detected: Sequence[bool]) -> dict[str, Any]:
    n = len(detected)
    k = int(sum(bool(d) for d in detected))
    lo, hi = wilson_ci(k, n)
    return {"survival_rate": k / n if n else None, "ci_low": lo, "ci_high": hi, "n": n, "n_detected": k}


def distribution(values: Sequence[float | None]) -> dict[str, Any]:
    v = np.array([float(x) for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return {"n": 0, "n_missing": len(values)}
    q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(len(v)),
        "n_missing": len(values) - int(len(v)),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "min": float(v.min()),
        "p05": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
        "max": float(v.max()),
    }


def mean_ci(values: Sequence[float | None], *, n_bootstrap: int = 1000, seed: int = 0) -> dict[str, Any]:
    v = np.array([float(x) for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    if len(v) == 1 or n_bootstrap == 0:
        return {"mean": float(v.mean()), "ci_low": None, "ci_high": None, "n": int(len(v))}
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_bootstrap, len(v)), replace=True).mean(axis=1)
    return {
        "mean": float(v.mean()),
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
        "n": int(len(v)),
    }


def roc_points(y: Sequence[int], s: Sequence[float | None], max_points: int = 200) -> dict[str, list[float]] | None:
    yy, ss, _ = _clean(y, s)
    if (yy == 1).sum() < 1 or (yy == 0).sum() < 1:
        return None
    fpr, tpr, _ = roc_curve(yy, ss)
    if len(fpr) > max_points:
        idx = np.linspace(0, len(fpr) - 1, max_points).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}


def paired_auroc_difference(
    y: Sequence[int],
    s_a: Sequence[float | None],
    s_b: Sequence[float | None],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """AUROC(a) - AUROC(b) on the same samples, with a paired stratified bootstrap CI.

    Samples where either score is missing are dropped from both, so the comparison stays paired.
    """
    keep = [
        i
        for i in range(len(y))
        if s_a[i] is not None and s_b[i] is not None and not math.isnan(float(s_a[i])) and not math.isnan(float(s_b[i]))
    ]
    yy = np.array([int(y[i]) for i in keep])
    a = np.array([float(s_a[i]) for i in keep])
    b = np.array([float(s_b[i]) for i in keep])
    out: dict[str, Any] = {
        "diff": None,
        "ci_low": None,
        "ci_high": None,
        "n": len(keep),
        "p_two_sided": None,
        "reason": None,
    }
    if (yy == 1).sum() < MIN_PER_CLASS or (yy == 0).sum() < MIN_PER_CLASS:
        out["reason"] = "insufficient class counts"
        return out
    out["diff"] = float(roc_auc_score(yy, a) - roc_auc_score(yy, b))
    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        pos, neg = np.where(yy == 1)[0], np.where(yy == 0)[0]
        diffs = []
        for _ in range(n_bootstrap):
            ip = rng.choice(pos, size=len(pos), replace=True)
            ineg = rng.choice(neg, size=len(neg), replace=True)
            diffs.append(_auc_from_scores(a[ip], a[ineg]) - _auc_from_scores(b[ip], b[ineg]))
        d = np.array(diffs)
        out["ci_low"], out["ci_high"] = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
        # Bootstrap percentile p-value for H0: diff = 0
        out["p_two_sided"] = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return out
