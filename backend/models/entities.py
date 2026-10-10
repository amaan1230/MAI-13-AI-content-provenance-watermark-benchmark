"""Relational schema for the provenance registry and experiment tracking.

Large payloads (datasets, trained detectors, reports) live in the artifact store;
the database keeps metadata, hashes and URIs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base, JSONType, TZDateTime, new_id, utcnow

SOURCE_TYPES = ("human", "ai", "ai_watermarked", "transformed", "unknown")
LABELS = ("human", "ai", "unknown")
SPLITS = ("train", "val", "test")
EXPERIMENT_STATUSES = ("pending", "running", "succeeded", "failed", "cancelled")
DETECTOR_TYPES = ("watermark", "ai_classifier", "provenance")
DECISIONS = ("positive", "negative", "insufficient", "error")
TRANSFORM_STATUSES = ("succeeded", "failed", "skipped")


def _in(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IN ({', '.join(repr(v) for v in values)})"


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    source: Mapped[str | None] = mapped_column(String(500))
    license: Mapped[str | None] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    manifest_uri: Mapped[str | None] = mapped_column(String(1000))
    manifest_sha256: Mapped[str | None] = mapped_column(String(64))
    n_items: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    split_seed: Mapped[int | None] = mapped_column(Integer)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, default=utcnow, nullable=False)


class ModelConfig(Base):
    __tablename__ = "model_configs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str] = mapped_column(String(300), nullable=False)
    revision: Mapped[str | None] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    params: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, default=utcnow, nullable=False)


class ContentItem(Base):
    __tablename__ = "content_items"
    __table_args__ = (
        CheckConstraint(_in("source_type", SOURCE_TYPES), name="source_type"),
        CheckConstraint(_in("label", LABELS), name="label"),
        CheckConstraint(f"split IS NULL OR {_in('split', SPLITS)}", name="split"),
        Index("ix_content_items_dataset_split", "dataset_version_id", "split"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    content_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    normalized_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    # Nullable so deployments with STORE_RAW_TEXT=false keep only hashes.
    text: Mapped[str | None] = mapped_column(Text)
    char_len: Mapped[int] = mapped_column(Integer, nullable=False)
    modality: Mapped[str] = mapped_column(String(20), default="text", nullable=False)
    language: Mapped[str | None] = mapped_column(String(16))
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)
    label: Mapped[str] = mapped_column(String(10), default="unknown", nullable=False)
    label_source: Mapped[str | None] = mapped_column(
        String(500), doc="Documented justification for the label (dataset, generator run, ...)"
    )
    dataset_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="SET NULL"), index=True
    )
    split: Mapped[str | None] = mapped_column(String(10))
    group_id: Mapped[str | None] = mapped_column(String(100), index=True)
    external_id: Mapped[str | None] = mapped_column(String(200))
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("content_items.id", ondelete="SET NULL"), index=True)
    model_config_id: Mapped[str | None] = mapped_column(ForeignKey("model_configs.id", ondelete="SET NULL"), index=True)
    experiment_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("experiment_runs.id", ondelete="SET NULL"), index=True
    )
    generation_config: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    signature: Mapped[str | None] = mapped_column(String(128))
    created_by: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(TZDateTime, default=utcnow, nullable=False, index=True)

    parent: Mapped[ContentItem | None] = relationship(remote_side=[id], back_populates="children")
    children: Mapped[list[ContentItem]] = relationship(back_populates="parent")


class WatermarkRun(Base):
    __tablename__ = "watermark_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    method: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    method_version: Mapped[str] = mapped_column(String(20), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    key_fingerprint: Mapped[str] = mapped_column(
        String(16), nullable=False, doc="First 16 hex of sha256(key); the key itself is never stored"
    )
    model_config_id: Mapped[str | None] = mapped_column(ForeignKey("model_configs.id", ondelete="SET NULL"))
    input_content_id: Mapped[str | None] = mapped_column(ForeignKey("content_items.id", ondelete="SET NULL"))
    output_content_id: Mapped[str | None] = mapped_column(
        ForeignKey("content_items.id", ondelete="CASCADE"), index=True
    )
    experiment_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("experiment_runs.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(20), default="succeeded", nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    n_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, default=utcnow, nullable=False)


class Transformation(Base):
    __tablename__ = "transformations"
    __table_args__ = (CheckConstraint(_in("status", TRANSFORM_STATUSES), name="status"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    parent_content_id: Mapped[str] = mapped_column(
        ForeignKey("content_items.id", ondelete="CASCADE"), nullable=False, index=True
    )
    child_content_id: Mapped[str | None] = mapped_column(ForeignKey("content_items.id", ondelete="CASCADE"), index=True)
    attack_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    attack_name: Mapped[str] = mapped_column(String(100), nullable=False)
    attack_version: Mapped[str] = mapped_column(String(20), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    seed: Mapped[int | None] = mapped_column(BigInteger)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    semantic_similarity: Mapped[float | None] = mapped_column(Float)
    experiment_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("experiment_runs.id", ondelete="CASCADE"), index=True
    )
    meta: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, default=utcnow, nullable=False)


class DetectionResult(Base):
    __tablename__ = "detection_results"
    __table_args__ = (
        CheckConstraint(_in("detector_type", DETECTOR_TYPES), name="detector_type"),
        CheckConstraint(_in("decision", DECISIONS), name="decision"),
        Index("ix_detection_results_exp_detector", "experiment_run_id", "detector_name"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    content_id: Mapped[str] = mapped_column(
        ForeignKey("content_items.id", ondelete="CASCADE"), nullable=False, index=True
    )
    detector_type: Mapped[str] = mapped_column(String(20), nullable=False)
    detector_name: Mapped[str] = mapped_column(String(100), nullable=False)
    detector_version: Mapped[str] = mapped_column(String(20), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    score: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    p_value: Mapped[float | None] = mapped_column(Float)
    n_tokens_scored: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    experiment_run_id: Mapped[str | None] = mapped_column(ForeignKey("experiment_runs.id", ondelete="CASCADE"))
    details: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, default=utcnow, nullable=False)


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"
    __table_args__ = (CheckConstraint(_in("status", EXPERIMENT_STATUSES), name="status"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    experiment_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False, index=True)
    stage: Mapped[str | None] = mapped_column(String(100))
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    dataset_version_id: Mapped[str | None] = mapped_column(ForeignKey("dataset_versions.id", ondelete="SET NULL"))
    rerun_of_id: Mapped[str | None] = mapped_column(ForeignKey("experiment_runs.id", ondelete="SET NULL"))
    software_version: Mapped[str] = mapped_column(String(50), nullable=False)
    git_commit: Mapped[str | None] = mapped_column(String(40))
    environment: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64))
    metrics_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    error: Mapped[str | None] = mapped_column(Text)
    error_details: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    created_by: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(TZDateTime, default=utcnow, nullable=False, index=True)
    started_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(TZDateTime)

    artifacts: Mapped[list[ExperimentArtifact]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )


class ExperimentArtifact(Base):
    __tablename__ = "experiment_artifacts"
    __table_args__ = (UniqueConstraint("experiment_run_id", "name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    experiment_run_id: Mapped[str] = mapped_column(
        ForeignKey("experiment_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    kind: Mapped[str] = mapped_column(String(50), nullable=False)
    uri: Mapped[str] = mapped_column(String(1000), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(500), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, default=utcnow, nullable=False)

    experiment: Mapped[ExperimentRun] = relationship(back_populates="artifacts")


class AuditEvent(Base):
    """Append-only, hash-chained audit log.

    Each event stores the hash of the previous event, so deleting or editing a row
    breaks the chain and is detectable. This is tamper *evidence* for records held
    by this server, not a cryptographic proof of who authored the content.
    """

    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_events_entity", "entity_type", "entity_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(TZDateTime, default=utcnow, nullable=False, index=True)
    actor: Mapped[str] = mapped_column(String(100), nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
    request_id: Mapped[str | None] = mapped_column(String(64))
    details: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    prev_hash: Mapped[str | None] = mapped_column(String(64))
    event_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
