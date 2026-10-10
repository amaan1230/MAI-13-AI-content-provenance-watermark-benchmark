"""Post-hoc zero-width-character watermark — a transparent *simulation baseline*.

A key-derived bit string is hidden as invisible Unicode characters (U+200B = 0,
U+200C = 1) after every `interval`-th word. It is included as a contrast to the
statistical watermarks:
- It survives word-level edits that leave the invisible characters in place.
- It is destroyed by any Unicode normalisation / control-character stripping, and
- it is trivially *spoofable*: copying the marks onto human text produces a positive.
It therefore demonstrates why format-level marks are unsuitable as provenance evidence.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import time
from typing import Any, ClassVar

from pydantic import Field

from backend.core.config import get_settings
from backend.provenance.signing import key_fingerprint
from backend.watermark.base import DetectResult, EmbedResult, WatermarkConfig, WatermarkMethod

ZW0, ZW1 = "​", "‌"
_ZW_RE = re.compile(f"[{ZW0}{ZW1}]+")


class ZeroWidthConfig(WatermarkConfig):
    payload_bits: int = Field(default=16, ge=4, le=64)
    interval: int = Field(default=8, ge=1, le=100, description="Insert a payload after every N-th word")
    threshold: float = Field(default=0.9, gt=0.0, le=1.0, description="Min fraction of matching bits")


class ZeroWidthWatermark(WatermarkMethod):
    name: ClassVar[str] = "zero_width"
    version: ClassVar[str] = "1.0.0"
    kind: ClassVar = "post_hoc"
    config_cls: ClassVar = ZeroWidthConfig

    def __init__(self, key: str | None = None):
        self._key = key or get_settings().watermark_secret_key.get_secret_value()

    def _payload(self, n_bits: int) -> str:
        digest = hmac.new(self._key.encode(), b"zero-width-payload", hashlib.sha256).digest()
        bits = "".join(f"{b:08b}" for b in digest)[:n_bits]
        return "".join(ZW1 if c == "1" else ZW0 for c in bits)

    def embed(self, text: str, config: Any = None) -> EmbedResult:
        cfg = self.parse_config(config)
        t0 = time.perf_counter()
        payload = self._payload(cfg.payload_bits)
        parts = re.split(r"(\s+)", text)
        out, words = [], 0
        for part in parts:
            out.append(part)
            if part and not part.isspace():
                words += 1
                if words % cfg.interval == 0:
                    out.append(payload)
        if words < cfg.interval:  # always embed at least once
            out.append(payload)
        marked = "".join(out)
        return EmbedResult(
            text=marked,
            input_text=text,
            n_tokens=None,
            latency_ms=(time.perf_counter() - t0) * 1000,
            metadata={"method": self.name, "config": cfg.model_dump(), "n_payloads": max(1, words // cfg.interval)},
        )

    def detect(self, text: str, config: Any = None) -> DetectResult:
        cfg = self.parse_config(config)
        t0 = time.perf_counter()
        expected = self._payload(cfg.payload_bits)
        runs = _ZW_RE.findall(text)
        if not runs:
            return DetectResult(
                score=0.0,
                p_value=None,
                threshold=cfg.threshold,
                decision="negative",
                n_tokens_scored=0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                details={"n_payloads_found": 0},
            )
        # Best per-run bitwise agreement; robust to a run being truncated.
        best = 0.0
        for r in runs:
            n = min(len(r), len(expected))
            agree = sum(a == b for a, b in zip(r[:n], expected[:n], strict=True)) / len(expected)
            best = max(best, agree)
        return DetectResult(
            score=best,
            p_value=0.5 ** round(best * cfg.payload_bits) if best > 0 else None,
            threshold=cfg.threshold,
            decision="positive" if best >= cfg.threshold else "negative",
            n_tokens_scored=sum(len(r) for r in runs),
            latency_ms=(time.perf_counter() - t0) * 1000,
            details={"n_payloads_found": len(runs)},
        )

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "kind": self.kind,
            "reference": "Simulation baseline (format-level steganography)",
            "key_fingerprint": key_fingerprint(self._key),
            "config_schema": self.config_cls.model_json_schema(),
            "detection_requires": ["secret key"],
            "limitations": [
                "Removed by Unicode normalisation or control-character stripping.",
                "Spoofable: marks can be copied onto unrelated (human) text.",
                "Carries no information about the generating model.",
            ],
        }
