"""Language-model-statistic detectors.

LMFeatureDetector (supervised): per-text statistics under a scoring LM — mean/std
token log-probability, mean log-rank, mean predictive entropy, top-1 and top-10 rank
fractions (GLTR, Gehrmann et al. 2019; log-rank, Mitchell et al. 2023) — plus the
Binoculars ratio, fed to a standardised logistic regression.

BinocularsDetector (zero-shot): Hans et al. (2024), "Spotting LLMs with Binoculars".
B = log PPL_observer(s) / X-PPL(observer, performer); low B suggests machine text, so
the returned score is -B. Observer and performer must share a tokenizer. With small
CPU models (gpt2 / distilgpt2) this is a weak instantiation of the method and is
reported as such; it is not a reproduction of the paper's Falcon-7B results.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.core.config import get_settings
from backend.core.hashing import content_hash
from backend.detectors.base import AITextDetector
from backend.providers.hf_local import get_lm

FEATURES = ["mean_logp", "std_logp", "mean_logrank", "mean_entropy", "frac_rank1", "frac_rank10", "binoculars"]


_feature_cache: dict[tuple, dict[str, float]] = {}
_CACHE_MAX = 20_000


def text_features(text: str, observer: str, performer: str | None, max_tokens: int = 256) -> dict[str, float]:
    """Cached: several detectors in one experiment share the same LM passes."""
    key = (content_hash(text), observer, performer, max_tokens)
    if key not in _feature_cache:
        if len(_feature_cache) >= _CACHE_MAX:
            _feature_cache.clear()
        _feature_cache[key] = _text_features(text, observer, performer, max_tokens)
    return dict(_feature_cache[key])


@torch.inference_mode()
def _text_features(text: str, observer: str, performer: str | None, max_tokens: int) -> dict[str, float]:
    obs = get_lm(observer)
    st = obs.token_stats(text, max_tokens=max_tokens)
    n = st["n_tokens"]
    if n < 2:
        return {k: float("nan") for k in FEATURES}
    logp, rank, ent = st["logp"], st["rank"], st["entropy"]
    feats = {
        "mean_logp": float(logp.mean()),
        "std_logp": float(logp.std()),
        "mean_logrank": float(np.log(rank).mean()),
        "mean_entropy": float(ent.mean()),
        "frac_rank1": float((rank == 1).mean()),
        "frac_rank10": float((rank <= 10).mean()),
        "binoculars": float("nan"),
    }
    if performer:
        perf = get_lm(performer)
        p_logits = perf.model(st["input_ids"]).logits[0, :-1].float()
        o_logp = torch.log_softmax(st["logits"], dim=-1)
        p_prob = torch.softmax(p_logits, dim=-1)
        x_ppl = float(-(p_prob * o_logp).sum(dim=-1).mean())
        log_ppl = float(-logp.mean())
        feats["binoculars"] = log_ppl / x_ppl if x_ppl > 0 else float("nan")
    return feats


class LMFeatureDetector(AITextDetector):
    name = "lm_features_lr"
    description = "GLTR/log-rank/entropy/Binoculars features under a scoring LM + logistic regression."

    def __init__(
        self,
        observer: str | None = None,
        performer: str | None = "distilgpt2",
        max_tokens: int = 256,
        seed: int = 0,
        **kw: Any,
    ):
        observer = observer or get_settings().default_scoring_model
        super().__init__(observer=observer, performer=performer, max_tokens=max_tokens, seed=seed, **kw)
        self.pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
            ]
        )
        self._cols: list[str] = []
        self._fill: np.ndarray | None = None

    def featurize(self, texts: list[str]) -> np.ndarray:
        p = self.params
        rows = [text_features(t, p["observer"], p["performer"], p["max_tokens"]) for t in texts]
        return np.array([[r[c] for c in FEATURES] for r in rows], dtype=float)

    def _fit(self, texts: list[str], labels: list[int]) -> None:
        X = self.featurize(texts)
        keep = ~np.all(np.isnan(X), axis=0)
        self._cols = [c for c, k in zip(FEATURES, keep, strict=True) if k]
        X = X[:, keep]
        self._fill = np.nanmean(X, axis=0)
        X = np.where(np.isnan(X), self._fill, X)
        self.pipe.fit(X, labels)

    def _score(self, texts: list[str]) -> list[float]:
        X = self.featurize(texts)
        X = X[:, [FEATURES.index(c) for c in self._cols]]
        X = np.where(np.isnan(X), self._fill, X)
        return [float(v) for v in self.pipe.predict_proba(X)[:, 1]]

    def coefficients(self) -> dict[str, float]:
        return dict(zip(self._cols, (float(c) for c in self.pipe.named_steps["clf"].coef_[0]), strict=True))

    def _state(self) -> dict[str, Any]:
        return {**super()._state(), "pipe": self.pipe, "cols": self._cols, "fill": self._fill}


class BinocularsDetector(AITextDetector):
    name = "binoculars"
    supervised = False
    description = "Zero-shot Binoculars score (Hans et al., 2024) with small CPU observer/performer models."

    def __init__(self, observer: str | None = None, performer: str = "distilgpt2", max_tokens: int = 256, **kw: Any):
        observer = observer or get_settings().default_scoring_model
        super().__init__(observer=observer, performer=performer, max_tokens=max_tokens, **kw)

    def _fit(self, texts: list[str], labels: list[int]) -> None:
        pass  # zero-shot: thresholds are still chosen on validation data by the evaluator

    def _score(self, texts: list[str]) -> list[float]:
        p = self.params
        out = []
        for t in texts:
            b = text_features(t, p["observer"], p["performer"], p["max_tokens"])["binoculars"]
            out.append(float("nan") if np.isnan(b) else -b)
        return out
