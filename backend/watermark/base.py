"""Pluggable watermark interface.

Embedding, watermark detection, AI-text classification and provenance verification
are deliberately separate concerns. A positive watermark detection means "this text
carries a statistical signal consistent with key K"; it is not proof of authorship.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

Decision = Literal["positive", "negative", "insufficient"]


class WatermarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass
class EmbedResult:
    text: str
    input_text: str
    n_tokens: int | None
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectResult:
    score: float | None
    p_value: float | None
    threshold: float
    decision: Decision
    n_tokens_scored: int
    latency_ms: float
    details: dict[str, Any] = field(default_factory=dict)


class WatermarkMethod(ABC):
    name: ClassVar[str]
    version: ClassVar[str]
    kind: ClassVar[Literal["generation_time", "post_hoc"]]
    config_cls: ClassVar[type[WatermarkConfig]]

    def parse_config(self, config: dict[str, Any] | WatermarkConfig | None) -> WatermarkConfig:
        if isinstance(config, self.config_cls):
            return config
        return self.config_cls.model_validate(config or {})

    @abstractmethod
    def embed(self, text: str, config: dict[str, Any] | WatermarkConfig | None = None) -> EmbedResult:
        """Generation-time methods treat `text` as the prompt; post-hoc methods mark `text`."""

    def embed_batch(
        self, texts: list[str], config: dict[str, Any] | WatermarkConfig | None = None
    ) -> list[EmbedResult]:
        return [self.embed(t, config) for t in texts]

    @abstractmethod
    def detect(self, text: str, config: dict[str, Any] | WatermarkConfig | None = None) -> DetectResult: ...

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]: ...
