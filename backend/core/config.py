"""Application settings loaded from environment variables / `.env`.

Secrets are typed as `SecretStr` so they are never rendered by `repr()`, logs,
or API responses. Only `get_secret_value()` at the point of use exposes them.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

Role = Literal["admin", "researcher", "viewer"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- General ---
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"

    # --- Storage ---
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'mai13.db').as_posix()}"
    artifact_dir: Path = PROJECT_ROOT / "artifacts"
    dataset_dir: Path = PROJECT_ROOT / "datasets"
    reports_dir: Path = PROJECT_ROOT / "reports"
    # MLflow 3 no longer accepts the plain filesystem store; SQLite metadata + local artifacts.
    mlflow_tracking_uri: str = f"sqlite:///{(PROJECT_ROOT / 'mlruns' / 'mlflow.db').as_posix()}"
    mlflow_artifact_dir: Path = PROJECT_ROOT / "mlruns" / "artifacts"
    mlflow_enabled: bool = True

    # --- Privacy / retention ---
    store_raw_text: bool = True
    data_retention_days: int = Field(default=0, ge=0, description="0 = keep indefinitely")
    max_text_chars: int = Field(default=20_000, gt=0)

    # --- Providers ---
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"
    hf_token: SecretStr | None = None
    hf_offline: bool = False
    provider_timeout_s: float = Field(default=60.0, gt=0)
    provider_max_retries: int = Field(default=3, ge=0, le=10)
    provider_backoff_base_s: float = Field(default=1.0, ge=0)

    # --- Default models (all small enough to run on CPU) ---
    default_generator_model: str = "Qwen/Qwen2.5-0.5B"
    default_scoring_model: str = "gpt2"
    default_oracle_model: str = "gpt2-medium"
    default_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    torch_threads: int = Field(default=0, ge=0, description="0 = torch default")

    # --- Watermark & provenance keys ---
    watermark_secret_key: SecretStr = SecretStr("dev-insecure-watermark-key")
    provenance_signing_key: SecretStr | None = None

    # --- API security ---
    auth_enabled: bool = False
    api_keys: SecretStr | None = Field(
        default=None, description='JSON object mapping API key -> role, e.g. {"k1":"admin"}'
    )
    rate_limit_per_minute: int = Field(default=120, ge=0, description="0 disables")
    cors_origins: list[str] = ["http://localhost:8501"]

    # --- Background jobs ---
    job_workers: int = Field(default=1, ge=1, le=8)

    # --- Frontend ---
    api_base_url: str = "http://localhost:8000"

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    def api_key_roles(self) -> dict[str, Role]:
        if self.api_keys is None:
            return {}
        raw = self.api_keys.get_secret_value().strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("API_KEYS must be a JSON object of key -> role") from exc
        if not isinstance(parsed, dict):
            raise ValueError("API_KEYS must be a JSON object of key -> role")
        out: dict[str, Role] = {}
        for k, v in parsed.items():
            if v not in ("admin", "researcher", "viewer"):
                raise ValueError(f"Invalid role {v!r} in API_KEYS")
            out[str(k)] = v
        return out

    def ensure_dirs(self) -> None:
        for p in (self.artifact_dir, self.dataset_dir, self.reports_dir, self.mlflow_artifact_dir):
            Path(p).mkdir(parents=True, exist_ok=True)
        if self.database_url.startswith("sqlite:///"):
            Path(self.database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
