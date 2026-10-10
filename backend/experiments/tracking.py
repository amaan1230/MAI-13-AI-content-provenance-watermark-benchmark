"""MLflow tracking wrapper. Tracking failures are logged but never fail an experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.core.config import get_settings
from backend.core.logging import get_logger

log = get_logger(__name__)


def flatten(d: dict[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key, sep))
        elif isinstance(v, list) and all(isinstance(x, dict) for x in v):
            for i, x in enumerate(v):
                out.update(flatten(x, f"{key}{sep}{i}", sep))
        else:
            out[key] = v
    return out


class Tracker:
    def __init__(self, experiment_name: str = "MAI-13"):
        self.enabled = get_settings().mlflow_enabled
        self.experiment_name = experiment_name
        self.run_id: str | None = None
        self._mlflow = None

    def start(self, run_name: str, tags: dict[str, str]) -> str | None:
        if not self.enabled:
            return None
        try:
            import mlflow

            s = get_settings()
            s.ensure_dirs()
            mlflow.set_tracking_uri(s.mlflow_tracking_uri)
            if mlflow.get_experiment_by_name(self.experiment_name) is None:
                mlflow.create_experiment(self.experiment_name, artifact_location=Path(s.mlflow_artifact_dir).as_uri())
            mlflow.set_experiment(self.experiment_name)
            run = mlflow.start_run(run_name=run_name, tags=tags)
            self._mlflow = mlflow
            self.run_id = run.info.run_id
            return self.run_id
        except Exception as exc:  # noqa: BLE001
            log.warning("mlflow start failed; continuing without tracking", extra={"error": str(exc)})
            self.enabled = False
            return None

    def log_params(self, params: dict[str, Any]) -> None:
        if not (self.enabled and self._mlflow):
            return
        try:
            flat = {k: str(v)[:500] for k, v in flatten(params).items()}
            items = list(flat.items())
            for i in range(0, len(items), 100):
                self._mlflow.log_params(dict(items[i : i + 100]))
        except Exception as exc:  # noqa: BLE001
            log.warning("mlflow log_params failed", extra={"error": str(exc)})

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        if not (self.enabled and self._mlflow):
            return
        try:
            clean = {}
            for k, v in flatten(metrics).items():
                if isinstance(v, bool) or not isinstance(v, int | float) or v != v:
                    continue
                safe = "".join(c if c.isalnum() or c in "._-/ " else "_" for c in k)[:250]
                clean[safe] = float(v)
            if clean:
                self._mlflow.log_metrics(clean)
        except Exception as exc:  # noqa: BLE001
            log.warning("mlflow log_metrics failed", extra={"error": str(exc)})

    def log_artifact(self, path: Path) -> None:
        if not (self.enabled and self._mlflow):
            return
        try:
            self._mlflow.log_artifact(str(path))
        except Exception as exc:  # noqa: BLE001
            log.warning("mlflow log_artifact failed", extra={"error": str(exc)})

    def end(self, status: str = "FINISHED") -> None:
        if not (self.enabled and self._mlflow):
            return
        try:
            self._mlflow.end_run(status=status)
        except Exception as exc:  # noqa: BLE001
            log.warning("mlflow end_run failed", extra={"error": str(exc)})
