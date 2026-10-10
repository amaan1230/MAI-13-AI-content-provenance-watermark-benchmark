from __future__ import annotations

from typing import Any

from backend.core.errors import ValidationFailed
from backend.detectors.base import AITextDetector
from backend.detectors.lm_features import BinocularsDetector, LMFeatureDetector
from backend.detectors.tfidf import TfidfLogRegDetector

DETECTORS: dict[str, type[AITextDetector]] = {
    TfidfLogRegDetector.name: TfidfLogRegDetector,
    LMFeatureDetector.name: LMFeatureDetector,
    BinocularsDetector.name: BinocularsDetector,
}


def build_detector(name: str, params: dict[str, Any] | None = None) -> AITextDetector:
    if name not in DETECTORS:
        raise ValidationFailed(f"Unknown detector {name!r}", details={"available": sorted(DETECTORS)})
    try:
        return DETECTORS[name](**(params or {}))
    except TypeError as exc:
        raise ValidationFailed(f"Invalid params for detector {name!r}: {exc}") from exc
