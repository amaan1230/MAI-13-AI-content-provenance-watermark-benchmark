"""Baseline: TF-IDF n-grams + logistic regression (Solaiman et al., 2019 style)."""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from backend.detectors.base import AITextDetector


class TfidfLogRegDetector(AITextDetector):
    name = "tfidf_lr"
    description = "Word 1-2 gram + char 3-5 gram TF-IDF, L2 logistic regression (balanced classes)."

    def __init__(self, C: float = 1.0, max_features: int = 50_000, seed: int = 0, **kw: Any):
        super().__init__(C=C, max_features=max_features, seed=seed, **kw)
        self.pipe = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, sublinear_tf=True)),
                            (
                                "char",
                                TfidfVectorizer(
                                    analyzer="char_wb", ngram_range=(3, 5), max_features=max_features, sublinear_tf=True
                                ),
                            ),
                        ]
                    ),
                ),
                ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed)),
            ]
        )

    def _fit(self, texts: list[str], labels: list[int]) -> None:
        self.pipe.fit(texts, labels)

    def _score(self, texts: list[str]) -> list[float]:
        return [float(p) for p in self.pipe.predict_proba(texts)[:, 1]]

    def _state(self) -> dict[str, Any]:
        return {**super()._state(), "pipe": self.pipe}
