"""Pydantic request/response schemas for the public API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T")


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = {}
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int


# ----------------------------------------------------------------------------- health / providers
class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    database: str
    database_dialect: str
    providers_available: int
    auth_enabled: bool


class ProviderOut(BaseModel):
    name: str
    available: bool
    kind: str
    capabilities: list[str]
    model: str | None
    message: str


class ProvidersOut(BaseModel):
    providers: list[ProviderOut]
    watermark_methods: list[dict[str, Any]]
    attacks: list[dict[str, Any]]
    detectors: list[dict[str, Any]]


# ----------------------------------------------------------------------------- content
class ContentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    content_hash: str
    text: str | None
    char_len: int
    modality: str
    language: str | None
    source_type: str
    label: str
    label_source: str | None
    dataset_version_id: str | None
    split: str | None
    group_id: str | None
    parent_id: str | None
    model_config_id: str | None
    experiment_run_id: str | None
    generation_config: dict[str, Any] | None
    meta: dict[str, Any]
    signature: str | None
    created_by: str | None
    created_at: datetime


class IngestIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    source_type: Literal["human", "ai", "ai_watermarked", "transformed", "unknown"] = "unknown"
    label: Literal["human", "ai", "unknown"] = "unknown"
    label_source: str | None = Field(
        default=None, max_length=500, description="Required unless label='unknown': why is this label justified?"
    )
    language: str | None = Field(default="en", max_length=16)
    parent_id: str | None = None
    external_id: str | None = Field(default=None, max_length=200)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _label(self):
        if self.label != "unknown" and not self.label_source:
            raise ValueError("label_source is required when label is not 'unknown'")
        return self


class WatermarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    config: dict[str, Any] = Field(default_factory=dict)


class GenerateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    provider: Literal["hf_local", "gemini"] = "hf_local"
    model_name: str | None = Field(default=None, description="hf_local only; defaults to DEFAULT_GENERATOR_MODEL")
    max_new_tokens: int = Field(default=200, ge=1, le=1024)
    temperature: float = Field(default=0.8, gt=0, le=2.0)
    top_p: float = Field(default=0.95, gt=0, le=1.0)
    seed: int = 0
    watermark: WatermarkRequest | None = Field(default=None, description="Generation-time watermark (hf_local only)")
    persist: bool = True


class GenerateOut(BaseModel):
    text: str
    provider: str
    model: str
    latency_ms: float
    n_tokens: int | None
    watermark: dict[str, Any] | None
    content_id: str | None
    prompt_content_id: str | None


class EmbedIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    text: str = Field(min_length=1, description="Prompt for generation-time methods; text to mark for post-hoc")
    config: dict[str, Any] = Field(default_factory=dict)
    persist: bool = True
    parent_content_id: str | None = None


class EmbedOut(BaseModel):
    method: str
    kind: str
    text: str
    n_tokens: int | None
    latency_ms: float
    content_id: str | None
    watermark_run_id: str | None
    metadata: dict[str, Any]


class DetectIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    text: str | None = None
    content_id: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    persist: bool = True

    @model_validator(mode="after")
    def _one(self):
        if (self.text is None) == (self.content_id is None):
            raise ValueError("Provide exactly one of text or content_id")
        return self


class DetectOut(BaseModel):
    method: str
    score: float | None
    p_value: float | None
    threshold: float
    decision: str
    n_tokens_scored: int
    latency_ms: float
    details: dict[str, Any]
    detection_id: str | None
    interpretation: str


class AttackIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attack: str
    config: dict[str, Any] = Field(default_factory=dict)
    text: str | None = None
    content_id: str | None = None
    seed: int = 0
    persist: bool = True
    compute_similarity: bool = True

    @model_validator(mode="after")
    def _one(self):
        if (self.text is None) == (self.content_id is None):
            raise ValueError("Provide exactly one of text or content_id")
        return self


class AttackOut(BaseModel):
    attack: str
    attack_type: str
    status: str
    text: str | None
    error: str | None
    latency_ms: float
    meta: dict[str, Any]
    similarity: float | None
    similarity_metric: str | None
    parent_content_id: str | None
    child_content_id: str | None
    transformation_id: str | None


class ClassifyIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    detector: str
    text: str = Field(min_length=1)


class ClassifyOut(BaseModel):
    detector: str
    experiment_id: str
    score: float | None
    threshold: float | None
    threshold_rule: str
    decision: str
    interpretation: str


# ----------------------------------------------------------------------------- experiments
class ExperimentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    experiment_type: str
    status: str
    stage: str | None
    progress: float
    config: dict[str, Any]
    config_hash: str
    seed: int
    dataset_version_id: str | None
    rerun_of_id: str | None
    software_version: str
    git_commit: str | None
    environment: dict[str, Any]
    mlflow_run_id: str | None
    metrics_summary: dict[str, Any] | None
    error: str | None
    error_details: dict[str, Any] | None
    created_by: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class ExperimentBrief(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    experiment_type: str
    status: str
    stage: str | None
    progress: float
    seed: int
    created_at: datetime
    finished_at: datetime | None
    error: str | None


class ExperimentCreated(BaseModel):
    id: str
    status: str
    status_url: str


class ArtifactOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    kind: str
    uri: str
    sha256: str
    size_bytes: int
    created_at: datetime


class DatasetVersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    version: str
    source: str | None
    license: str | None
    description: str | None
    manifest_uri: str | None
    manifest_sha256: str | None
    n_items: int
    split_seed: int | None
    meta: dict[str, Any]
    created_at: datetime


# ----------------------------------------------------------------------------- provenance
class LineageNode(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    parent_id: str | None
    source_type: str
    label: str
    content_hash: str
    char_len: int
    split: str | None
    created_at: datetime
    meta: dict[str, Any]


class TransformationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    parent_content_id: str
    child_content_id: str | None
    attack_type: str
    attack_name: str
    attack_version: str
    config: dict[str, Any]
    seed: int | None
    status: str
    error: str | None
    latency_ms: float | None
    semantic_similarity: float | None
    created_at: datetime


class WatermarkRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    method: str
    method_version: str
    config: dict[str, Any]
    key_fingerprint: str
    input_content_id: str | None
    output_content_id: str | None
    experiment_run_id: str | None
    n_tokens: int | None
    latency_ms: float | None
    created_at: datetime


class DetectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    detector_type: str
    detector_name: str
    detector_version: str
    score: float | None
    threshold: float | None
    decision: str
    p_value: float | None
    n_tokens_scored: int | None
    experiment_run_id: str | None
    created_at: datetime


class ProvenanceOut(BaseModel):
    content: ContentOut
    ancestors: list[LineageNode]
    descendants: list[LineageNode]
    watermark_runs: list[WatermarkRunOut]
    transformations_in: list[TransformationOut]
    transformations_out: list[TransformationOut]
    detections: list[DetectionOut]
    integrity: dict[str, Any]
    disclaimer: str = (
        "This record documents what this system observed and stored. Hash and signature checks show the "
        "record is unaltered; watermark detections are statistical evidence of key usage. Neither constitutes "
        "proof of authorship."
    )


class VerifyIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None = None
