"""Command-line entry point for running experiments synchronously.

python -m backend.experiments.cli run experiments/configs/smoke.yaml
python -m backend.experiments.cli rerun <experiment_id>
python -m backend.experiments.cli validate experiments/configs/*.yaml
"""

from __future__ import annotations

import argparse
import json
import sys

from backend.core.config import get_settings
from backend.core.logging import configure_logging
from backend.db.migrate import upgrade
from backend.db.session import get_sessionmaker
from backend.experiments.config import load_config
from backend.experiments.runner import ExperimentRunner
from backend.experiments.service import create_experiment, get_experiment, rerun_experiment


def _run(exp_id: str) -> int:
    sm = get_sessionmaker()
    ExperimentRunner(exp_id, sm, actor="cli").run()
    with sm() as s:
        exp = get_experiment(s, exp_id)
        print(
            json.dumps(
                {"experiment_id": exp.id, "status": exp.status, "error": exp.error, "summary": exp.metrics_summary},
                indent=2,
                default=str,
            )
        )
        return 0 if exp.status == "succeeded" else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mai13")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("config")
    rr = sub.add_parser("rerun")
    rr.add_argument("experiment_id")
    v = sub.add_parser("validate")
    v.add_argument("configs", nargs="+")
    args = p.parse_args(argv)

    s = get_settings()
    configure_logging(s.log_level, s.log_format)
    if args.cmd == "validate":
        for path in args.configs:
            cfg = load_config(path)
            print(f"OK {path}: {cfg.name} ({cfg.experiment_type})")
        return 0
    upgrade()
    sm = get_sessionmaker()
    with sm() as session:
        if args.cmd == "run":
            exp = create_experiment(session, load_config(args.config), actor="cli")
        else:
            exp = rerun_experiment(session, args.experiment_id, actor="cli")
        session.commit()
        exp_id = exp.id
    print(f"experiment_id={exp_id}", flush=True)
    return _run(exp_id)


if __name__ == "__main__":
    sys.exit(main())
