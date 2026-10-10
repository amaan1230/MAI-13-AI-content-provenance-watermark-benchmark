from __future__ import annotations

from backend.core.errors import ValidationFailed
from backend.watermark.base import WatermarkMethod
from backend.watermark.kgw import GreenListWatermark, UnigramWatermark
from backend.watermark.zero_width import ZeroWidthWatermark

WATERMARK_METHODS: dict[str, type[WatermarkMethod]] = {
    GreenListWatermark.name: GreenListWatermark,
    UnigramWatermark.name: UnigramWatermark,
    ZeroWidthWatermark.name: ZeroWidthWatermark,
}

_instances: dict[str, WatermarkMethod] = {}


def get_watermark(name: str) -> WatermarkMethod:
    if name not in WATERMARK_METHODS:
        raise ValidationFailed(f"Unknown watermark method {name!r}", details={"available": list(WATERMARK_METHODS)})
    if name not in _instances:
        _instances[name] = WATERMARK_METHODS[name]()
    return _instances[name]


def reset_instances() -> None:
    _instances.clear()
