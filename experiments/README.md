# experiments/

`configs/` contains the shipped experiment definitions (validated by `ExperimentConfig`):

| Config | Purpose | Approx. CPU time |
|---|---|---|
| `smoke.yaml` | End-to-end check with real models; not for conclusions | ~10 min |
| `main_benchmark.yaml` | Baseline comparison (RQ3) + robustness/failure-mode study (RQ1, RQ2, RQ4) | 2–4 h |
| `delta_ablation.yaml` | Watermark strength δ ∈ {1, 2, 4}: detectability vs quality vs robustness | ~1.5 h |

Run: `python -m backend.experiments.cli run experiments/configs/<file>.yaml`
(or POST the config as JSON to `/api/v1/experiments`). Outputs land in `artifacts/experiments/<id>/`,
`reports/`, the database and MLflow.
