"""Validated experiment configuration (YAML/JSON <-> pydantic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.attacks.registry import ATTACKS, build_attack
from backend.core.config import get_settings
from backend.core.errors import ValidationFailed
from backend.data.splits import DEFAULT_FRACTIONS, validate_fractions
from backend.detectors.registry import DETECTORS
from backend.watermark.registry import WATERMARK_METHODS


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetSpec(_Strict):
    source: Literal["c4_realnewslike", "jsonl"] = "c4_realnewslike"
    path: str | None = Field(default=None, description="Required when source=jsonl")
    name: str | None = None
    n_samples: int = Field(default=100, ge=4, le=10_000)
    prompt_words: int = Field(default=50, ge=5, le=500)
    continuation_words: int = Field(default=150, ge=10, le=1000)
    skip: int = Field(default=0, ge=0)
    split_fractions: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_FRACTIONS))
    leakage_threshold: float = Field(default=0.8, gt=0, le=1)

    @field_validator("split_fractions")
    @classmethod
    def _fractions(cls, v):
        validate_fractions(v)
        return v

    @model_validator(mode="after")
    def _path(self):
        if self.source == "jsonl" and not self.path:
            raise ValueError("dataset.path is required when source=jsonl")
        return self


class GeneratorSpec(_Strict):
    model_name: str = Field(default_factory=lambda: get_settings().default_generator_model)
    max_new_tokens: int = Field(default=200, ge=8, le=1024)
    min_new_tokens: int = Field(default=150, ge=0, le=1024)
    temperature: float = Field(default=0.8, gt=0, le=2.0)
    top_p: float = Field(default=0.95, gt=0, le=1.0)
    batch_size: int = Field(default=8, ge=1, le=64)
    max_prompt_tokens: int = Field(default=256, ge=8, le=2048)

    @model_validator(mode="after")
    def _lens(self):
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError("generator.min_new_tokens must be <= max_new_tokens")
        return self


class WatermarkSpec(_Strict):
    method: str
    label: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("method")
    @classmethod
    def _known(cls, v):
        if v not in WATERMARK_METHODS:
            raise ValueError(f"unknown watermark method {v!r}; available: {sorted(WATERMARK_METHODS)}")
        return v

    @property
    def key(self) -> str:
        return self.label or self.method


class AttackSpec(_Strict):
    name: str
    label: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    apply_to_negatives: bool = True
    max_groups: int | None = Field(
        default=None, ge=2, description="Limit costly attacks to a fixed, seeded subset of test documents"
    )

    @model_validator(mode="after")
    def _valid(self):
        if self.name not in ATTACKS:
            raise ValueError(f"unknown attack {self.name!r}; available: {sorted(ATTACKS)}")
        try:
            build_attack(self.name, self.config)  # validates config eagerly
        except ValidationFailed as exc:
            raise ValueError(exc.message) from None
        return self

    @property
    def key(self) -> str:
        return self.label or self.name


class DetectorSpec(_Strict):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    is_baseline: bool = False

    @field_validator("name")
    @classmethod
    def _known(cls, v):
        if v not in DETECTORS:
            raise ValueError(f"unknown detector {v!r}; available: {sorted(DETECTORS)}")
        return v


class EvaluationSpec(_Strict):
    target_fprs: list[float] = Field(default_factory=lambda: [0.01, 0.05])
    primary_fpr: float = 0.05
    n_bootstrap: int = Field(default=1000, ge=0, le=20_000)
    semantic_similarity: bool = True
    perplexity: bool = True
    oracle_model: str | None = None
    quality_max_samples: int = Field(default=200, ge=1)
    length_bins: list[int] = Field(default_factory=lambda: [25, 50, 100, 150, 250])

    @model_validator(mode="after")
    def _fprs(self):
        if not self.target_fprs or any(not 0 < f < 1 for f in self.target_fprs):
            raise ValueError("target_fprs must be non-empty values in (0, 1)")
        if self.primary_fpr not in self.target_fprs:
            raise ValueError("primary_fpr must be one of target_fprs")
        return self


class ExperimentConfig(_Strict):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    experiment_type: Literal["watermark_robustness", "detector_benchmark", "full"] = "full"
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    dataset: DatasetSpec = Field(default_factory=DatasetSpec)
    generator: GeneratorSpec = Field(default_factory=GeneratorSpec)
    watermarks: list[WatermarkSpec] = Field(default_factory=list)
    attacks: list[AttackSpec] = Field(default_factory=list)
    detectors: list[DetectorSpec] = Field(default_factory=list)
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)

    @model_validator(mode="after")
    def _consistency(self):
        for kind, keys in (("watermark", [w.key for w in self.watermarks]), ("attack", [a.key for a in self.attacks])):
            if len(keys) != len(set(keys)):
                raise ValueError(f"{kind} labels must be unique; set `label` to disambiguate")
        if self.experiment_type in ("watermark_robustness", "full") and not self.watermarks:
            raise ValueError(f"experiment_type={self.experiment_type} requires at least one watermark")
        if self.experiment_type in ("detector_benchmark", "full") and not self.detectors:
            raise ValueError(f"experiment_type={self.experiment_type} requires at least one detector")
        return self

    @property
    def run_watermarks(self) -> bool:
        return self.experiment_type in ("watermark_robustness", "full")

    @property
    def run_detectors(self) -> bool:
        return self.experiment_type in ("detector_benchmark", "full")


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(raw)
