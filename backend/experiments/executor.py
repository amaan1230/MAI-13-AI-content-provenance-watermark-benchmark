"""In-process background execution for long-running experiments.

A bounded ThreadPoolExecutor is the simplest design that satisfies the requirements
for a single-node research deployment: the API returns an experiment ID immediately
and status/progress is persisted in the database. The `JobExecutor` interface can be
backed by Celery/Redis for multi-node deployments without touching the API.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor

from sqlalchemy import select

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.db.session import get_sessionmaker
from backend.experiments.runner import ExperimentRunner
from backend.models import ExperimentRun
from backend.models.base import utcnow

log = get_logger(__name__)


class JobExecutor:
    def __init__(self, workers: int | None = None):
        self._pool = ThreadPoolExecutor(max_workers=workers or get_settings().job_workers, thread_name_prefix="exp")
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit(self, experiment_id: str, actor: str = "system") -> None:
        runner = ExperimentRunner(experiment_id, get_sessionmaker(), actor=actor)
        with self._lock:
            self._futures[experiment_id] = self._pool.submit(runner.run)

    def is_active(self, experiment_id: str) -> bool:
        with self._lock:
            f = self._futures.get(experiment_id)
            return f is not None and not f.done()

    def wait(self, experiment_id: str, timeout: float | None = None) -> None:
        with self._lock:
            f = self._futures.get(experiment_id)
        if f is not None:
            f.result(timeout=timeout)

    def recover(self) -> dict[str, int]:
        """On startup: fail runs interrupted by a restart, resubmit queued ones."""
        sm = get_sessionmaker()
        failed = 0
        to_submit: list[tuple[str, str]] = []
        with sm() as s:
            exps = s.execute(select(ExperimentRun).where(ExperimentRun.status.in_(["running", "pending"]))).scalars()
            for exp in list(exps):
                if exp.status == "running":
                    exp.status = "failed"
                    exp.error = "Interrupted: the server restarted while this experiment was running. Use rerun."
                    exp.finished_at = utcnow()
                    failed += 1
                else:
                    to_submit.append((exp.id, exp.created_by or "system"))
            s.commit()
        for exp_id, actor in to_submit:
            self.submit(exp_id, actor)
        resubmitted = len(to_submit)
        if failed or resubmitted:
            log.info("recovered experiments", extra={"failed": failed, "resubmitted": resubmitted})
        return {"failed": failed, "resubmitted": resubmitted}

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


_executor: JobExecutor | None = None


def get_executor() -> JobExecutor:
    global _executor
    if _executor is None:
        _executor = JobExecutor()
    return _executor


def set_executor(executor: JobExecutor | None) -> None:
    global _executor
    _executor = executor
