"""AI-generated vs human-written text classifiers.

These are deliberately independent of the watermark detectors: a watermark z-score
answers "was key K used?", while these estimate P(AI-written) from text statistics.
Neither score is used as a substitute for the other.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from backend.core.errors import InsufficientData, ValidationFailed


class AITextDetector(ABC):
    name: ClassVar[str]
    version: ClassVar[str] = "1.0.0"
    supervised: ClassVar[bool] = True
    description: ClassVar[str] = ""

    def __init__(self, **params: Any):
        self.params = params
        self.fitted = False

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> AITextDetector:
        if len(texts) != len(labels):
            raise ValidationFailed("texts and labels length mismatch")
        if len(set(labels)) < 2:
            raise InsufficientData("Training data must contain both classes", details={"labels": sorted(set(labels))})
        if min(sum(1 for y in labels if y == c) for c in (0, 1)) < 2:
            raise InsufficientData("Need at least 2 training samples per class")
        self._fit(list(texts), list(labels))
        self.fitted = True
        return self

    def score(self, texts: Sequence[str]) -> list[float]:
        """Higher = more likely AI-generated. Not necessarily a calibrated probability."""
        if self.supervised and not self.fitted:
            raise ValidationFailed(f"Detector {self.name} must be fitted before scoring")
        return self._score(list(texts))

    @abstractmethod
    def _fit(self, texts: list[str], labels: list[int]) -> None: ...

    @abstractmethod
    def _score(self, texts: list[str]) -> list[float]: ...

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "supervised": self.supervised,
            "description": self.description,
            "params": self.params,
        }

    # Pickles are only ever loaded from this application's own artifact store.
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self._state(), fh)

    def _state(self) -> dict[str, Any]:
        return {"params": self.params, "fitted": self.fitted}
