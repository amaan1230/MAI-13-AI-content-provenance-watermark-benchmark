"""End-to-end experiment pipeline.

Protocol (see docs/METHODOLOGY.md):
 1. Data      : versioned prompt / human-continuation pairs, group-aware splits.
 2. Generate  : unwatermarked AI continuation + one per generation-time watermark;
                post-hoc watermarks are applied to the unwatermarked AI text.
 3. WM detect : score every text; thresholds = theoretical z and validation-selected
                at target FPRs (negatives = human + unwatermarked AI, val split only);
                metrics reported on the held-out test split.
 4. Attacks   : transform test-split watermarked texts (and human texts) with full
                lineage; re-detect with frozen thresholds; survival + meaning preservation.
 5. Detectors : AI-vs-human classifiers trained on train, thresholded on val, tested on
                test, then probed with watermarked and attacked texts (distribution shift).
 6. Quality   : oracle-LM perplexity and repetition for human / AI / watermarked text.
 7. Finalise  : post-generation leakage audit, cost summary, JSON/CSV/Markdown artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import io
import json
import os
import platform
import subprocess
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy.orm import Session, sessionmaker

from backend import __version__
from backend.attacks.registry import build_attack
from backend.core.config import PROJECT_ROOT, get_settings
from backend.core.errors import MAIError, ProviderUnavailable
from backend.core.hashing import config_hash
from backend.core.logging import experiment_id_var, get_logger
from backend.core.seeding import derive_seed, set_global_seed
from backend.core.storage import LocalArtifactStore
from backend.data.builder import build_dataset
from backend.data.leakage import LeakItem, check_leakage
from backend.detectors.registry import DETECTORS, build_detector
from backend.evaluation import metrics as M
from backend.evaluation import quality as Q
from backend.evaluation.report import render_markdown
from backend.experiments.config import AttackSpec, ExperimentConfig, WatermarkSpec
from backend.experiments.tracking import Tracker
from backend.models import DetectionResult, ExperimentArtifact, ExperimentRun, Transformation, WatermarkRun
from backend.models.base import utcnow
from backend.provenance.audit import record_event
from backend.provenance.registry import NewContent, get_or_create_model_config, register_content
from backend.provenance.signing import key_fingerprint
from backend.providers.hf_local import get_lm
from backend.providers.registry import get_provider
from backend.watermark.base import DetectResult
from backend.watermark.kgw import _Warper
from backend.watermark.registry import get_watermark

log = get_logger(__name__)

STAGES = [
    ("data", 0.05),
    ("generate", 0.35),
    ("watermark_detect", 0.05),
    ("attacks", 0.25),
    ("detectors", 0.15),
    ("quality", 0.10),
    ("finalise", 0.05),
]


class Cancelled(Exception):
    pass


class NotRunnable(Exception):
    """The experiment is not pending (e.g. already finished); leave its state untouched."""


@dataclass
class Sample:
    content_id: str
    group_id: str
    split: str
    kind: str  # human | ai | wm
    text: str
    label: int  # 1 = AI-generated (incl. watermarked), 0 = human
    wm_key: str | None = None
    attack_key: str | None = None
    parent_id: str | None = None
    n_gen_tokens: int | None = None
    gen_latency_ms: float | None = None
    sim_to_parent: float | None = None


@dataclass
class RunState:
    samples: list[Sample] = field(default_factory=list)
    prompts: dict[str, str] = field(default_factory=dict)  # group_id -> prompt content_id
    wm_scores: dict[str, dict[str, DetectResult]] = field(default_factory=lambda: defaultdict(dict))
    det_scores: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    thresholds: dict[str, dict[str, Any]] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def select(self, **kw) -> list[Sample]:
        out = []
        for s in self.samples:
            ok = True
            for k, v in kw.items():
                val = getattr(s, k)
                if isinstance(v, list | tuple | set):
                    ok = val in v
                else:
                    ok = val == v
                if not ok:
                    break
            if ok:
                out.append(s)
        return out


def environment_info() -> dict[str, Any]:
    import sklearn
    import torch
    import transformers

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "transformers": transformers.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "cuda": torch.cuda.is_available(),
    }
    return info


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


class ExperimentRunner:
    def __init__(self, experiment_id: str, session_factory: sessionmaker[Session], actor: str = "system"):
        self.experiment_id = experiment_id
        self.session_factory = session_factory
        self.actor = actor
        s = get_settings()
        self.store = LocalArtifactStore(s.artifact_dir)
        self.tracker = Tracker()
        self.state = RunState()
        self.cfg: ExperimentConfig | None = None
        self._progress_base = 0.0
        self._stage_weight = 0.0
        self._session: Session | None = None

    # ------------------------------------------------------------------ status helpers
    def _update(self, **fields: Any) -> None:
        """Persist status/progress fields.

        Inside a stage this goes through the stage's own session and commits it: work done so
        far becomes visible (incremental lineage) and we never hold two write transactions at
        once, which SQLite cannot serve.
        """
        if self._session is not None:
            exp = self._session.get(ExperimentRun, self.experiment_id)
            for k, v in fields.items():
                setattr(exp, k, v)
            self._session.commit()
            return
        with self.session_factory() as s:
            exp = s.get(ExperimentRun, self.experiment_id)
            for k, v in fields.items():
                setattr(exp, k, v)
            s.commit()

    def _check_cancel(self) -> None:
        with self.session_factory() as s:
            exp = s.get(ExperimentRun, self.experiment_id)
            if exp is not None and exp.status == "cancelled":
                raise Cancelled()

    def _stage_progress(self, frac: float) -> None:
        self._update(progress=round(min(1.0, self._progress_base + self._stage_weight * frac), 4))

    # ------------------------------------------------------------------ entrypoint
    def run(self) -> None:
        token = experiment_id_var.set(self.experiment_id)
        status = "FAILED"
        try:
            with self.session_factory() as s:
                exp = s.get(ExperimentRun, self.experiment_id)
                if exp is None:
                    raise MAIError(f"Experiment {self.experiment_id} not found")
                self.cfg = ExperimentConfig.model_validate(exp.config)
                if exp.status == "cancelled":
                    raise Cancelled()
                if exp.status != "pending":
                    raise NotRunnable(exp.status)
                exp.status = "running"
                exp.started_at = utcnow()
                exp.environment = environment_info()
                exp.git_commit = git_commit()
                record_event(s, actor=self.actor, action="experiment.start", entity_type="experiment", entity_id=exp.id)
                s.commit()
            set_global_seed(self.cfg.seed)
            run_id = self.tracker.start(
                self.cfg.name, {"experiment_id": self.experiment_id, "type": self.cfg.experiment_type}
            )
            if run_id:
                self._update(mlflow_run_id=run_id)
            self.tracker.log_params({"config": self.cfg.model_dump(), "software_version": __version__})

            stage_fns: dict[str, Callable[[Session], None]] = {
                "data": self._stage_data,
                "generate": self._stage_generate,
                "watermark_detect": self._stage_wm_detect,
                "attacks": self._stage_attacks,
                "detectors": self._stage_detectors,
                "quality": self._stage_quality,
                "finalise": self._stage_finalise,
            }
            for name, weight in STAGES:
                self._check_cancel()
                self._stage_weight = weight
                self._update(stage=name)
                t0 = time.perf_counter()
                with self.session_factory() as session:
                    self._session = session
                    try:
                        stage_fns[name](session)
                        session.commit()
                    except Exception:
                        session.rollback()
                        raise
                    finally:
                        self._session = None
                self.state.timings[name] = round(time.perf_counter() - t0, 3)
                log.info("stage complete", extra={"stage": name, "seconds": self.state.timings[name]})
                self._progress_base += weight
                self._stage_progress(0)

            summary = self.state.results.get("summary", {})
            self.tracker.log_metrics(summary)
            self._update(status="succeeded", stage="done", progress=1.0, finished_at=utcnow(), metrics_summary=summary)
            with self.session_factory() as s:
                record_event(
                    s,
                    actor=self.actor,
                    action="experiment.succeeded",
                    entity_type="experiment",
                    entity_id=self.experiment_id,
                )
                s.commit()
            status = "FINISHED"
        except Cancelled:
            self._update(finished_at=utcnow(), error="Cancelled by user")
            status = "KILLED"
        except NotRunnable as exc:
            log.warning("experiment not runnable", extra={"status": str(exc)})
            status = "KILLED"
        except Exception as exc:  # noqa: BLE001
            log.exception("experiment failed")
            code = exc.code if isinstance(exc, MAIError) else type(exc).__name__
            msg = exc.message if isinstance(exc, MAIError) else str(exc)
            self._update(
                status="failed",
                finished_at=utcnow(),
                error=f"{code}: {msg}"[:2000],
                error_details={
                    "stage": self._current_stage(),
                    "type": type(exc).__name__,
                    "details": getattr(exc, "details", {}),
                    "traceback": traceback.format_exc()[-4000:],
                },
            )
            with self.session_factory() as s:
                record_event(
                    s,
                    actor=self.actor,
                    action="experiment.failed",
                    entity_type="experiment",
                    entity_id=self.experiment_id,
                    details={"error": code},
                )
                s.commit()
        finally:
            self.tracker.end(status)
            experiment_id_var.reset(token)

    def _current_stage(self) -> str | None:
        with self.session_factory() as s:
            exp = s.get(ExperimentRun, self.experiment_id)
            return exp.stage if exp else None

    # ------------------------------------------------------------------ helpers
    def _register(self, session: Session, sample: Sample, **kw) -> Sample:
        item = register_content(
            session,
            NewContent(
                text=sample.text,
                experiment_run_id=self.experiment_id,
                group_id=sample.group_id,
                split=sample.split,
                **kw,
            ),
            actor=self.actor,
        )
        sample.content_id = item.id
        self.state.samples.append(sample)
        return sample

    def _wm_config(self, spec: WatermarkSpec) -> dict[str, Any]:
        method = get_watermark(spec.method)
        cfg: dict[str, Any] = {}
        if method.kind == "generation_time":
            g = self.cfg.generator
            cfg = {
                "model_name": g.model_name,
                "max_new_tokens": g.max_new_tokens,
                "min_new_tokens": g.min_new_tokens,
                "temperature": g.temperature,
                "top_p": g.top_p,
                "batch_size": g.batch_size,
                "max_prompt_tokens": g.max_prompt_tokens,
                "seed": self.cfg.seed,
            }
        cfg.update(spec.config)
        return method.parse_config(cfg).model_dump()

    # ------------------------------------------------------------------ 1. data
    def _stage_data(self, session: Session) -> None:
        built = build_dataset(session, self.cfg.dataset, self.cfg.seed, actor=self.actor)
        self._update(dataset_version_id=built.version.id)
        self.state.results["dataset"] = {
            "id": built.version.id,
            "name": built.version.name,
            "version": built.version.version,
            "manifest": built.manifest,
        }
        self._records = built.records
        self._splits = built.splits
        for r in built.records:
            split = built.splits[r.group_id]
            prompt = register_content(
                session,
                NewContent(
                    text=r.prompt,
                    source_type="human",
                    label="human",
                    label_source=r.label_source,
                    dataset_version_id=built.version.id,
                    split=split,
                    group_id=r.group_id,
                    external_id=r.external_id,
                    experiment_run_id=self.experiment_id,
                    meta={"role": "prompt", "url": r.url},
                ),
                actor=self.actor,
            )
            self.state.prompts[r.group_id] = prompt.id
            self._register(
                session,
                Sample("", r.group_id, split, "human", r.human_text, 0),
                source_type="human",
                label="human",
                label_source=r.label_source,
                dataset_version_id=built.version.id,
                external_id=r.external_id,
                meta={"role": "human_continuation", "url": r.url},
            )

    # ------------------------------------------------------------------ 2. generate
    def _stage_generate(self, session: Session) -> None:
        if not self.cfg.run_watermarks and not self.cfg.run_detectors:
            return
        g = self.cfg.generator
        records = self._records
        prompts = [r.prompt for r in records]
        lm = get_lm(g.model_name)
        gen_params = g.model_dump()
        mc = get_or_create_model_config(
            session, provider="hf_local", model_name=g.model_name, role="generator", params=gen_params
        )
        gen_cost: dict[str, Any] = {}
        failures: dict[str, int] = defaultdict(int)

        # Unwatermarked AI continuation
        t0 = time.perf_counter()
        outs = []
        for i in range(0, len(prompts), g.batch_size):
            self._check_cancel()
            outs += lm.generate(
                prompts[i : i + g.batch_size],
                max_new_tokens=g.max_new_tokens,
                min_new_tokens=g.min_new_tokens,
                seed=self.cfg.seed + i,
                processors=[_Warper(g.temperature, g.top_p)],
                batch_size=g.batch_size,
                max_prompt_tokens=g.max_prompt_tokens,
            )
            self._stage_progress(min(1.0, (i + g.batch_size) / len(prompts)) / (1 + len(self.cfg.watermarks)))
        elapsed = time.perf_counter() - t0
        n_tok = sum(o.n_new_tokens for o in outs)
        gen_cost["unwatermarked"] = {
            "seconds": round(elapsed, 2),
            "tokens": n_tok,
            "tokens_per_s": round(n_tok / elapsed, 2) if elapsed else None,
        }
        ai_by_group: dict[str, Sample] = {}
        label_src = f"Generated by {g.model_name} (unwatermarked) in experiment {self.experiment_id}"
        for r, o in zip(records, outs, strict=True):
            if not o.text.strip():
                failures["unwatermarked_empty"] += 1
                continue
            s = self._register(
                session,
                Sample(
                    "",
                    r.group_id,
                    self._splits[r.group_id],
                    "ai",
                    o.text,
                    1,
                    parent_id=self.state.prompts[r.group_id],
                    n_gen_tokens=o.n_new_tokens,
                    gen_latency_ms=o.latency_ms,
                ),
                source_type="ai",
                label="ai",
                label_source=label_src,
                parent_id=self.state.prompts[r.group_id],
                model_config_id=mc.id,
                generation_config={**gen_params, "seed": self.cfg.seed, "watermark": None},
            )
            ai_by_group[r.group_id] = s

        # Watermarked variants
        if self.cfg.run_watermarks:
            for wi, spec in enumerate(self.cfg.watermarks):
                self._check_cancel()
                self._stage_progress((wi + 1) / (1 + len(self.cfg.watermarks)))  # commit before long compute
                method = get_watermark(spec.method)
                wcfg = self._wm_config(spec)
                fp = key_fingerprint(get_settings().watermark_secret_key.get_secret_value())
                t0 = time.perf_counter()
                if method.kind == "generation_time":
                    # Same chunking and per-chunk seeds as the unwatermarked pass, so the two samples
                    # for a prompt share their random stream and differ only by the watermark bias.
                    embeds = []
                    bs = g.batch_size
                    for i in range(0, len(prompts), bs):
                        self._check_cancel()
                        embeds += method.embed_batch(prompts[i : i + bs], {**wcfg, "seed": self.cfg.seed + i})
                        frac = (wi + 1 + min(1.0, (i + bs) / len(prompts))) / (1 + len(self.cfg.watermarks))
                        self._stage_progress(frac)
                    parents = [self.state.prompts[r.group_id] for r in records]
                    groups = [r.group_id for r in records]
                else:
                    srcs = list(ai_by_group.values())
                    embeds = [method.embed(s.text, wcfg) for s in srcs]
                    parents = [s.content_id for s in srcs]
                    groups = [s.group_id for s in srcs]
                elapsed = time.perf_counter() - t0
                n_tok = sum(e.n_tokens or 0 for e in embeds)
                gen_cost[spec.key] = {
                    "seconds": round(elapsed, 2),
                    "tokens": n_tok or None,
                    "tokens_per_s": round(n_tok / elapsed, 2) if n_tok and elapsed else None,
                }
                for gid, parent, e in zip(groups, parents, embeds, strict=True):
                    if not e.text.strip():
                        failures[f"{spec.key}_empty"] += 1
                        continue
                    s = self._register(
                        session,
                        Sample(
                            "",
                            gid,
                            self._splits[gid],
                            "wm",
                            e.text,
                            1,
                            wm_key=spec.key,
                            parent_id=parent,
                            n_gen_tokens=e.n_tokens,
                            gen_latency_ms=e.latency_ms,
                        ),
                        source_type="ai_watermarked",
                        label="ai",
                        label_source=f"Generated by {g.model_name} with watermark {spec.key} in experiment "
                        f"{self.experiment_id}",
                        parent_id=parent,
                        model_config_id=mc.id,
                        generation_config={**gen_params, "seed": self.cfg.seed, "watermark": spec.key},
                        meta={"watermark_method": spec.method},
                    )
                    session.add(
                        WatermarkRun(
                            method=method.name,
                            method_version=method.version,
                            config=wcfg,
                            config_hash=config_hash(wcfg),
                            key_fingerprint=fp,
                            model_config_id=mc.id if method.kind == "generation_time" else None,
                            input_content_id=parent,
                            output_content_id=s.content_id,
                            experiment_run_id=self.experiment_id,
                            status="succeeded",
                            n_tokens=e.n_tokens,
                            latency_ms=e.latency_ms,
                            meta={"label": spec.key},
                        )
                    )
                session.flush()
                self._stage_progress((wi + 2) / (1 + len(self.cfg.watermarks)))
        self.state.results["generation"] = {
            "cost": gen_cost,
            "failures": dict(failures),
            "model": g.model_name,
            "params": gen_params,
        }
        if failures:
            self.state.warnings.append(f"Generation produced empty outputs: {dict(failures)}")

    # ------------------------------------------------------------------ 3. watermark detection
    def _detect_wm(self, session: Session, spec: WatermarkSpec, samples: list[Sample]) -> None:
        method = get_watermark(spec.method)
        wcfg = self._wm_config(spec)
        rows = []
        for s in samples:
            if s.content_id in self.state.wm_scores[spec.key]:
                continue
            r = method.detect(s.text, wcfg)
            self.state.wm_scores[spec.key][s.content_id] = r
            rows.append(
                DetectionResult(
                    content_id=s.content_id,
                    detector_type="watermark",
                    detector_name=f"{spec.method}:{spec.key}",
                    detector_version=method.version,
                    config=wcfg,
                    score=r.score,
                    threshold=r.threshold,
                    decision=r.decision,
                    p_value=r.p_value,
                    n_tokens_scored=r.n_tokens_scored,
                    latency_ms=r.latency_ms,
                    experiment_run_id=self.experiment_id,
                    details={k: v for k, v in r.details.items() if k != "green_flags"},
                )
            )
        session.add_all(rows)
        session.flush()

    def _score(self, key: str, s: Sample) -> float | None:
        r = self.state.wm_scores[key].get(s.content_id)
        return None if r is None else r.score

    def _stage_wm_detect(self, session: Session) -> None:
        if not self.cfg.run_watermarks:
            return
        ev = self.cfg.evaluation
        out: dict[str, Any] = {}
        for spec in self.cfg.watermarks:
            k = spec.key
            method = get_watermark(spec.method)
            relevant = self.state.select(kind=["human", "ai"], attack_key=None) + self.state.select(kind="wm", wm_key=k)
            self._detect_wm(session, spec, relevant)
            theoretical = self._wm_config(spec).get("z_threshold", self._wm_config(spec).get("threshold"))

            def negs(split):
                return self.state.select(kind=["human", "ai"], split=split, attack_key=None)

            def poss(split, k=k):
                return self.state.select(kind="wm", wm_key=k, split=split, attack_key=None)

            val_neg_scores = [self._score(k, s) for s in negs("val")]
            thresholds: dict[str, Any] = {"theoretical": {"threshold": theoretical, "source": "method default"}}
            for f in ev.target_fprs:
                t = M.threshold_at_fpr(val_neg_scores, f)
                thresholds[f"val_fpr_{f}"] = {**t, "source": "validation negatives (human + unwatermarked AI)"}
                if t.get("reason"):
                    self.state.warnings.append(f"{k}: threshold at FPR {f}: {t['reason']}")
            self.state.thresholds[k] = thresholds

            test_neg, test_pos = negs("test"), poss("test")
            y = [1] * len(test_pos) + [0] * len(test_neg)
            sc = [self._score(k, s) for s in test_pos + test_neg]
            clean = {
                "n_pos": len(test_pos),
                "n_neg": len(test_neg),
                "auroc": M.auroc(y, sc, n_bootstrap=ev.n_bootstrap, seed=self.cfg.seed),
                "roc": M.roc_points(y, sc),
                "by_threshold": {
                    name: M.binary_metrics(y, sc, t["threshold"])
                    for name, t in thresholds.items()
                    if t.get("threshold") is not None
                },
                "neg_subgroups": {},
                "insufficient": {
                    kind: sum(
                        1
                        for s in self.state.select(
                            kind=kind, split="test", attack_key=None, **({"wm_key": k} if kind == "wm" else {})
                        )
                        if self.state.wm_scores[k][s.content_id].decision == "insufficient"
                    )
                    for kind in ("human", "ai", "wm")
                },
                "distributions": {
                    kind: M.distribution(
                        [
                            self._score(k, s)
                            for s in self.state.select(
                                kind=kind, split="test", attack_key=None, **({"wm_key": k} if kind == "wm" else {})
                            )
                        ]
                    )
                    for kind in ("human", "ai", "wm")
                },
                "length_bins": self._length_bins(k, test_pos, thresholds["theoretical"]["threshold"]),
            }
            for kind in ("human", "ai"):
                sub = [s for s in test_neg if s.kind == kind]
                yy = [1] * len(test_pos) + [0] * len(sub)
                ss = [self._score(k, s) for s in test_pos + sub]
                clean["neg_subgroups"][kind] = {
                    "auroc": M.auroc(yy, ss, n_bootstrap=ev.n_bootstrap, seed=self.cfg.seed),
                    "fpr": {
                        name: M.binary_metrics([0] * len(sub), [self._score(k, s) for s in sub], t["threshold"])["fpr"]
                        for name, t in thresholds.items()
                        if t.get("threshold") is not None
                    },
                }
            lat = [r.latency_ms for r in self.state.wm_scores[k].values()]
            out[k] = {
                "method": spec.method,
                "metadata": {kk: vv for kk, vv in method.get_metadata().items() if kk != "config_schema"},
                "config": self._wm_config(spec),
                "thresholds": thresholds,
                "clean": clean,
                "detection_latency_ms": M.distribution(lat),
                "attacks": {},
            }
            if len(test_pos) < 30:
                self.state.warnings.append(f"{k}: only {len(test_pos)} watermarked test samples; CIs are wide.")
        self.state.results["watermarks"] = out

    def _length_bins(self, k: str, pos: list[Sample], threshold: float | None) -> list[dict[str, Any]]:
        if threshold is None:
            return []
        edges = [0, *self.cfg.evaluation.length_bins, 10**9]
        out = []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            members = [s for s in pos if lo <= self.state.wm_scores[k][s.content_id].n_tokens_scored < hi]
            if not members:
                continue
            det = [(self._score(k, s) or float("-inf")) >= threshold for s in members]
            out.append({"tokens_scored": f"[{lo},{hi if hi < 10**9 else 'inf'})", **M.survival_rate(det)})
        return out

    # ------------------------------------------------------------------ 4. attacks
    def _attack_available(self, spec: AttackSpec) -> str | None:
        """Return a reason string if the attack cannot run in this environment."""
        if spec.name == "paraphrase" or (spec.name == "back_translation" and spec.config.get("backend") == "provider"):
            prov = spec.config.get("provider", "gemini")
            try:
                st = get_provider(prov).status()
            except MAIError as exc:
                return exc.message
            if not st.available:
                return f"provider {prov}: {st.message}"
        return None

    def _stage_attacks(self, session: Session) -> None:
        if not self.cfg.attacks or not (self.cfg.run_watermarks or self.cfg.run_detectors):
            self.state.results["attacks"] = {}
            return
        ev = self.cfg.evaluation
        attack_summary: dict[str, Any] = {}
        n_attacks = len(self.cfg.attacks)
        for ai, spec in enumerate(self.cfg.attacks):
            self._check_cancel()
            attack = build_attack(spec.name, spec.config)
            reason = self._attack_available(spec)
            targets = self.state.select(kind="wm", split="test", attack_key=None)
            if self.cfg.run_detectors and not self.cfg.run_watermarks:
                targets = self.state.select(kind="ai", split="test", attack_key=None)
            if spec.apply_to_negatives:
                targets += self.state.select(kind="human", split="test", attack_key=None)
            if spec.max_groups is not None:
                # Same documents for every method, so per-method results stay paired.
                groups = sorted({t.group_id for t in targets}, key=lambda g: derive_seed(self.cfg.seed, g))
                keep = set(groups[: spec.max_groups])
                targets = [t for t in targets if t.group_id in keep]
            if reason:
                attack_summary[spec.key] = {
                    "status": "skipped",
                    "reason": reason,
                    "describe": attack.describe(),
                    "n_targets": len(targets),
                }
                self.state.warnings.append(f"Attack {spec.key} skipped: {reason}")
                continue
            counts: dict[str, int] = defaultdict(int)
            lat: list[float] = []
            children: list[Sample] = []
            for ti, parent in enumerate(targets):
                if ti % 10 == 0:
                    self._check_cancel()
                    self._stage_progress((ai + ti / max(1, len(targets))) / n_attacks)
                # Stable identifiers only (content IDs are random UUIDs) so reruns apply identical edits.
                seed = derive_seed(self.cfg.seed, parent.group_id, parent.kind, parent.wm_key or "", spec.key)
                res = attack.apply(parent.text, seed)
                counts[res.status] += 1
                lat.append(res.latency_ms)
                tr = Transformation(
                    parent_content_id=parent.content_id,
                    attack_type=attack.attack_type,
                    attack_name=spec.key,
                    attack_version=attack.version,
                    config=attack.config.model_dump(),
                    seed=seed,
                    status=res.status,
                    error=res.error,
                    latency_ms=res.latency_ms,
                    experiment_run_id=self.experiment_id,
                    meta=res.meta,
                )
                if res.status == "succeeded":
                    child = Sample(
                        "",
                        parent.group_id,
                        parent.split,
                        parent.kind,
                        res.text,
                        parent.label,
                        wm_key=parent.wm_key,
                        attack_key=spec.key,
                        parent_id=parent.content_id,
                    )
                    self._register(
                        session,
                        child,
                        source_type="transformed",
                        label="ai" if parent.label else "human",
                        label_source=f"Inherited from parent {parent.content_id} via attack {spec.key}",
                        parent_id=parent.content_id,
                        meta={"attack": spec.key},
                    )
                    tr.child_content_id = child.content_id
                    children.append(child)
                session.add(tr)
            session.flush()
            # Meaning preservation
            sim_metric = None
            if ev.semantic_similarity and children:
                parents_by_id = {s.content_id: s for s in self.state.samples}
                sims = Q.semantic_similarity([(parents_by_id[c.parent_id].text, c.text) for c in children])
                sim_metric = sims["metric"]
                for c, v in zip(children, sims["values"], strict=True):
                    c.sim_to_parent = v
                trs = {
                    t.child_content_id: t
                    for t in session.query(Transformation).filter(
                        Transformation.experiment_run_id == self.experiment_id, Transformation.attack_name == spec.key
                    )
                }
                for c in children:
                    if c.content_id in trs:
                        trs[c.content_id].semantic_similarity = c.sim_to_parent
                        trs[c.content_id].meta = {**(trs[c.content_id].meta or {}), "similarity_metric": sim_metric}
            if counts.get("failed"):
                self.state.warnings.append(
                    f"Attack {spec.key}: {counts['failed']} of {len(targets)} transformations failed"
                )
            attack_summary[spec.key] = {
                "status": "ran",
                "describe": attack.describe(),
                "counts": dict(counts),
                "n_targets": len(targets),
                "latency_ms": M.distribution(lat),
                "similarity_metric": sim_metric,
                "similarity": {
                    kind: M.mean_ci(
                        [c.sim_to_parent for c in children if c.kind == kind],
                        n_bootstrap=ev.n_bootstrap,
                        seed=self.cfg.seed,
                    )
                    for kind in ("wm", "human", "ai")
                    if any(c.kind == kind for c in children)
                },
            }
            if self.cfg.run_watermarks:
                self._evaluate_wm_attack(session, spec, children, {t.group_id for t in targets})
        self.state.results["attacks"] = attack_summary

    def _evaluate_wm_attack(
        self, session: Session, aspec: AttackSpec, children: list[Sample], groups: set[str]
    ) -> None:
        """`groups` = documents the attack was applied to; clean baselines are restricted to them."""
        ev = self.cfg.evaluation
        parents_by_id = {s.content_id: s for s in self.state.samples}
        for wspec in self.cfg.watermarks:
            k = wspec.key
            pos = [c for c in children if c.kind == "wm" and c.wm_key == k]
            neg_att = [c for c in children if c.kind == "human"]
            self._detect_wm(session, wspec, pos + neg_att)
            clean_neg = self.state.select(kind=["human", "ai"], split="test", attack_key=None)
            clean_pos = [
                s for s in self.state.select(kind="wm", wm_key=k, split="test", attack_key=None) if s.group_id in groups
            ]
            res: dict[str, Any] = {
                "n_groups_attacked": len(groups),
                "n_attacked_pos": len(pos),
                "n_attacked_neg": len(neg_att),
                "n_clean_pos": len(clean_pos),
            }
            y1 = [1] * len(pos) + [0] * len(clean_neg)
            s1 = [self._score(k, s) for s in pos + clean_neg]
            res["auroc_vs_clean_neg"] = M.auroc(y1, s1, n_bootstrap=ev.n_bootstrap, seed=self.cfg.seed)
            res["roc_vs_clean_neg"] = M.roc_points(y1, s1)
            if neg_att:
                y2 = [1] * len(pos) + [0] * len(neg_att)
                s2 = [self._score(k, s) for s in pos + neg_att]
                res["auroc_vs_attacked_human"] = M.auroc(y2, s2, n_bootstrap=ev.n_bootstrap, seed=self.cfg.seed)
            res["by_threshold"] = {}
            for name, t in self.state.thresholds[k].items():
                thr = t.get("threshold")
                if thr is None:
                    continue
                # Absolute: attack failures count as not detected (denominator = all clean positives).
                det_abs = []
                pos_by_parent = {c.parent_id: c for c in pos}
                cond = []
                for cp in clean_pos:
                    child = pos_by_parent.get(cp.content_id)
                    after = child is not None and (self._score(k, child) or float("-inf")) >= thr
                    det_abs.append(after)
                    if (self._score(k, cp) or float("-inf")) >= thr:
                        cond.append(after)
                entry = {
                    "tpr_after_attack": M.survival_rate(det_abs),
                    "conditional_survival": M.survival_rate(cond),
                }
                if neg_att:
                    entry["fpr_attacked_human"] = M.binary_metrics(
                        [0] * len(neg_att), [self._score(k, s) for s in neg_att], thr
                    )
                res["by_threshold"][name] = entry
            deltas = []
            for c in pos:
                a, b = self._score(k, c), self._score(k, parents_by_id[c.parent_id])
                if a is not None and b is not None:
                    deltas.append(a - b)
            res["score_change"] = M.mean_ci(deltas, n_bootstrap=ev.n_bootstrap, seed=self.cfg.seed)
            res["insufficient_after_attack"] = sum(
                1 for c in pos if self.state.wm_scores[k][c.content_id].decision == "insufficient"
            )
            res["similarity_wm"] = M.mean_ci(
                [c.sim_to_parent for c in pos], n_bootstrap=ev.n_bootstrap, seed=self.cfg.seed
            )
            res["distribution_attacked"] = M.distribution([self._score(k, c) for c in pos])
            self.state.results["watermarks"][k]["attacks"][aspec.key] = res

    # ------------------------------------------------------------------ 5. AI-text detectors
    def _stage_detectors(self, session: Session) -> None:
        if not self.cfg.run_detectors:
            return
        ev = self.cfg.evaluation
        out: dict[str, Any] = {}
        clean = lambda split, kinds: self.state.select(kind=kinds, split=split, attack_key=None)  # noqa: E731
        train = clean("train", ["human", "ai"])
        val = clean("val", ["human", "ai"])
        test = clean("test", ["human", "ai"])
        for di, dspec in enumerate(self.cfg.detectors):
            self._check_cancel()
            params = {**dspec.params}
            if "seed" in inspect.signature(DETECTORS[dspec.name].__init__).parameters:
                params.setdefault("seed", self.cfg.seed)
            det = build_detector(dspec.name, params)
            t0 = time.perf_counter()
            det.fit([s.text for s in train], [s.label for s in train])
            fit_s = time.perf_counter() - t0
            key = f"detectors/{self.experiment_id}/{dspec.name}.pkl"
            det.save(self.store.local_path(key))
            self._add_artifact(session, f"detector_{dspec.name}", "model", key)

            to_score = (
                val
                + test
                + [s for s in self.state.samples if s.split == "test" and (s.kind == "wm" or s.attack_key is not None)]
            )
            uniq = {s.content_id: s for s in to_score}
            t0 = time.perf_counter()
            scores = det.score([s.text for s in uniq.values()])
            per_text_ms = (time.perf_counter() - t0) * 1000 / max(1, len(uniq))
            sc = {cid: (None if (v is None or v != v) else v) for cid, v in zip(uniq, scores, strict=True)}
            self.state.det_scores[dspec.name] = sc

            val_neg = [sc[s.content_id] for s in val if s.label == 0]
            thresholds = {f"val_fpr_{f}": M.threshold_at_fpr(val_neg, f) for f in ev.target_fprs}
            session.add_all(
                [
                    DetectionResult(
                        content_id=cid,
                        detector_type="ai_classifier",
                        detector_name=dspec.name,
                        detector_version=det.version,
                        config=det.describe()["params"],
                        score=v,
                        threshold=thresholds[f"val_fpr_{ev.primary_fpr}"]["threshold"],
                        decision=(
                            "error"
                            if v is None
                            else "positive"
                            if (
                                thresholds[f"val_fpr_{ev.primary_fpr}"]["threshold"] is not None
                                and v >= thresholds[f"val_fpr_{ev.primary_fpr}"]["threshold"]
                            )
                            else "negative"
                        ),
                        latency_ms=per_text_ms,
                        experiment_run_id=self.experiment_id,
                    )
                    for cid, v in sc.items()
                ]
            )
            y = [s.label for s in test]
            st = [sc[s.content_id] for s in test]
            res: dict[str, Any] = {
                "describe": det.describe(),
                "is_baseline": dspec.is_baseline,
                "n_train": {
                    "human": sum(1 for s in train if s.label == 0),
                    "ai": sum(1 for s in train if s.label == 1),
                },
                "n_val": {"human": sum(1 for s in val if s.label == 0), "ai": sum(1 for s in val if s.label == 1)},
                "n_test": {"human": sum(1 for s in test if s.label == 0), "ai": sum(1 for s in test if s.label == 1)},
                "thresholds": thresholds,
                "clean_test": {
                    "auroc": M.auroc(y, st, n_bootstrap=ev.n_bootstrap, seed=self.cfg.seed),
                    "roc": M.roc_points(y, st),
                    "by_threshold": {
                        n: M.binary_metrics(y, st, t["threshold"])
                        for n, t in thresholds.items()
                        if t["threshold"] is not None
                    },
                },
                "fit_seconds": round(fit_s, 2),
                "score_ms_per_text": round(per_text_ms, 2),
                "shift": {},
            }
            if hasattr(det, "coefficients"):
                res["coefficients"] = det.coefficients()
            human_test = [s for s in test if s.label == 0]
            prim = thresholds[f"val_fpr_{ev.primary_fpr}"]["threshold"]
            for wspec in self.cfg.watermarks:
                wm = self.state.select(kind="wm", wm_key=wspec.key, split="test", attack_key=None)
                yy = [1] * len(wm) + [0] * len(human_test)
                ss = [sc.get(s.content_id) for s in wm + human_test]
                res["shift"][f"watermarked:{wspec.key}"] = {
                    "auroc": M.auroc(yy, ss, n_bootstrap=ev.n_bootstrap, seed=self.cfg.seed),
                    "tpr_at_primary": None if prim is None else M.binary_metrics(yy, ss, prim)["tpr"],
                }
            for aspec in self.cfg.attacks:
                att_ai = self.state.select(attack_key=aspec.key, split="test", label=1)
                att_h = self.state.select(attack_key=aspec.key, split="test", label=0)
                if not att_ai:
                    continue
                negs = att_h or human_test
                yy = [1] * len(att_ai) + [0] * len(negs)
                ss = [sc.get(s.content_id) for s in att_ai + negs]
                bm = None if prim is None else M.binary_metrics(yy, ss, prim)
                res["shift"][f"attacked:{aspec.key}"] = {
                    "negatives": "attacked human" if att_h else "clean human",
                    "auroc": M.auroc(yy, ss, n_bootstrap=ev.n_bootstrap, seed=self.cfg.seed),
                    "tpr_at_primary": bm["tpr"] if bm else None,
                    "fpr_at_primary": bm["fpr"] if bm else None,
                }
            out[dspec.name] = res
            self._stage_progress((di + 1) / len(self.cfg.detectors))

        # Paired comparisons against the baseline on the identical clean test set.
        comparisons = []
        base = next((d.name for d in self.cfg.detectors if d.is_baseline), self.cfg.detectors[0].name)
        y = [s.label for s in test]
        for d in self.cfg.detectors:
            if d.name == base:
                continue
            comparisons.append(
                {
                    "detector": d.name,
                    "baseline": base,
                    **M.paired_auroc_difference(
                        y,
                        [self.state.det_scores[d.name][s.content_id] for s in test],
                        [self.state.det_scores[base][s.content_id] for s in test],
                        n_bootstrap=ev.n_bootstrap,
                        seed=self.cfg.seed,
                    ),
                }
            )
        self.state.results["detectors"] = {"results": out, "baseline": base, "paired_comparisons": comparisons}

        # Cross-task: identifying watermarked text vs human text — watermark detector vs classifiers.
        cross = []
        human_test = [s for s in test if s.label == 0]
        for wspec in self.cfg.watermarks:
            wm = self.state.select(kind="wm", wm_key=wspec.key, split="test", attack_key=None)
            yy = [1] * len(wm) + [0] * len(human_test)
            row = {
                "task": f"{wspec.key}-watermarked vs human (test)",
                "n_pos": len(wm),
                "n_neg": len(human_test),
                "watermark_detector_auroc": M.auroc(
                    yy, [self._score(wspec.key, s) for s in wm + human_test], n_bootstrap=0
                )["auroc"]
                if self.cfg.run_watermarks
                else None,
            }
            for d in self.cfg.detectors:
                row[f"{d.name}_auroc"] = M.auroc(
                    yy, [self.state.det_scores[d.name].get(s.content_id) for s in wm + human_test], n_bootstrap=0
                )["auroc"]
            cross.append(row)
        self.state.results["cross_task"] = cross

    # ------------------------------------------------------------------ 6. quality
    def _stage_quality(self, session: Session) -> None:
        ev = self.cfg.evaluation
        groups: dict[str, list[Sample]] = {
            "human": self.state.select(kind="human", split="test", attack_key=None),
            "ai_unwatermarked": self.state.select(kind="ai", split="test", attack_key=None),
        }
        for w in self.cfg.watermarks:
            groups[f"wm:{w.key}"] = self.state.select(kind="wm", wm_key=w.key, split="test", attack_key=None)
        out: dict[str, Any] = {"oracle_model": None, "groups": {}}
        oracle = ev.oracle_model or get_settings().default_oracle_model
        ppl_ok = ev.perplexity
        if ppl_ok:
            try:
                get_lm(oracle)
                out["oracle_model"] = oracle
            except ProviderUnavailable as exc:
                ppl_ok = False
                self.state.warnings.append(f"Perplexity skipped: {exc.message}")
        for gi, (name, members) in enumerate(groups.items()):
            members = members[: ev.quality_max_samples]
            entry: dict[str, Any] = {"n": len(members)}
            if ppl_ok and members:
                ppl = Q.perplexity([s.text for s in members], oracle)
                entry["perplexity"] = M.mean_ci(ppl, n_bootstrap=ev.n_bootstrap, seed=self.cfg.seed)
                entry["perplexity_median"] = M.distribution(ppl).get("median")
            entry["repetition_3gram"] = M.mean_ci(
                [Q.repetition_rate(s.text) for s in members], n_bootstrap=ev.n_bootstrap, seed=self.cfg.seed
            )
            entry["words"] = M.distribution([len(s.text.split()) for s in members])
            out["groups"][name] = entry
            self._stage_progress((gi + 1) / len(groups))
        self.state.results["quality"] = out

    # ------------------------------------------------------------------ 7. finalise
    def _add_artifact(self, session: Session, name: str, kind: str, key: str) -> None:
        p = self.store.local_path(key)
        data = p.read_bytes()
        session.add(
            ExperimentArtifact(
                experiment_run_id=self.experiment_id,
                name=name,
                kind=kind,
                uri=self.store.uri(key),
                storage_key=key,
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
            )
        )

    def _stage_finalise(self, session: Session) -> None:
        # Leakage audit over every generated / derived text (not just the source documents).
        items = [LeakItem(id=s.content_id, text=s.text, split=s.split, group_id=s.group_id) for s in self.state.samples]
        leak = check_leakage(items, threshold=self.cfg.dataset.leakage_threshold, seed=self.cfg.seed)
        if not leak["leakage_free"]:
            self.state.warnings.append(
                f"Leakage audit: {leak['exact_cross_split_duplicates']} exact and {leak['near_cross_split_duplicates']}"
                f" near-duplicate cross-split pairs found (see results.leakage)."
            )
        self.state.results["leakage"] = leak

        counts: dict[str, int] = defaultdict(int)
        for s in self.state.samples:
            counts[
                f"{s.split}:{s.kind}{':' + s.wm_key if s.wm_key else ''}{':' + s.attack_key if s.attack_key else ''}"
            ] += 1
        self.state.results["counts"] = dict(sorted(counts.items()))
        self.state.results["cost"] = {
            "stage_seconds": self.state.timings,
            **self.state.results.get("generation", {}).get("cost", {}),
        }
        self.state.results["summary"] = self._summary()
        self.state.results["warnings"] = self.state.warnings
        self.state.results["experiment"] = {
            "id": self.experiment_id,
            "name": self.cfg.name,
            "type": self.cfg.experiment_type,
            "seed": self.cfg.seed,
            "config": self.cfg.model_dump(),
            "config_hash": config_hash(self.cfg.model_dump()),
            "software_version": __version__,
            "environment": environment_info(),
            "git_commit": git_commit(),
        }

        base = f"experiments/{self.experiment_id}"
        self.store.put_json(f"{base}/results.json", self.state.results)
        self.store.put_bytes(f"{base}/samples.csv", self._samples_csv().encode("utf-8"))
        self.store.put_bytes(f"{base}/detections.csv", self._detections_csv().encode("utf-8"))
        self.store.put_bytes(f"{base}/report.md", render_markdown(self.state.results).encode("utf-8"))
        for name, kind in (
            ("results.json", "results_json"),
            ("samples.csv", "results_csv"),
            ("detections.csv", "results_csv"),
            ("report.md", "report_md"),
        ):
            self._add_artifact(session, name, kind, f"{base}/{name}")
            self.tracker.log_artifact(self.store.local_path(f"{base}/{name}"))
        # Also copy the report into the top-level reports/ directory for convenience.
        rep_dir = get_settings().reports_dir
        rep_dir.mkdir(parents=True, exist_ok=True)
        (rep_dir / f"{self.cfg.name}-{self.experiment_id[:8]}.md").write_bytes(
            self.store.get_bytes(f"{base}/report.md")
        )
        record_event(
            session,
            actor=self.actor,
            action="experiment.artifacts",
            entity_type="experiment",
            entity_id=self.experiment_id,
            details={"artifacts": 4},
        )

    def _summary(self) -> dict[str, Any]:
        r = self.state.results
        prim = f"val_fpr_{self.cfg.evaluation.primary_fpr}"
        out: dict[str, Any] = {}
        for k, w in r.get("watermarks", {}).items():
            out[f"wm.{k}.auroc"] = w["clean"]["auroc"]["auroc"]
            bt = w["clean"]["by_threshold"]
            if "theoretical" in bt:
                out[f"wm.{k}.tpr_theoretical"] = bt["theoretical"]["tpr"]
                out[f"wm.{k}.fpr_theoretical"] = bt["theoretical"]["fpr"]
            if prim in bt:
                out[f"wm.{k}.tpr_{prim}"] = bt[prim]["tpr"]
                out[f"wm.{k}.fpr_{prim}"] = bt[prim]["fpr"]
            for a, ar in w["attacks"].items():
                out[f"wm.{k}.{a}.auroc"] = ar["auroc_vs_clean_neg"]["auroc"]
                if "theoretical" in ar["by_threshold"]:
                    out[f"wm.{k}.{a}.tpr_theoretical"] = ar["by_threshold"]["theoretical"]["tpr_after_attack"][
                        "survival_rate"
                    ]
                out[f"wm.{k}.{a}.similarity"] = ar["similarity_wm"]["mean"]
        for d, dr in r.get("detectors", {}).get("results", {}).items():
            out[f"det.{d}.auroc"] = dr["clean_test"]["auroc"]["auroc"]
            if prim in dr["clean_test"]["by_threshold"]:
                out[f"det.{d}.tpr_{prim}"] = dr["clean_test"]["by_threshold"][prim]["tpr"]
                out[f"det.{d}.fpr_{prim}"] = dr["clean_test"]["by_threshold"][prim]["fpr"]
        for g, q in r.get("quality", {}).get("groups", {}).items():
            if "perplexity" in q:
                out[f"quality.{g}.ppl"] = q["perplexity"]["mean"]
        out["n_samples_total"] = len(self.state.samples)
        return out

    def _samples_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(
            [
                "content_id",
                "group_id",
                "split",
                "kind",
                "label",
                "wm_key",
                "attack_key",
                "parent_id",
                "n_words",
                "n_gen_tokens",
                "sim_to_parent",
            ]
        )
        for s in self.state.samples:
            w.writerow(
                [
                    s.content_id,
                    s.group_id,
                    s.split,
                    s.kind,
                    s.label,
                    s.wm_key or "",
                    s.attack_key or "",
                    s.parent_id or "",
                    len(s.text.split()),
                    s.n_gen_tokens or "",
                    "" if s.sim_to_parent is None else round(s.sim_to_parent, 5),
                ]
            )
        return buf.getvalue()

    def _detections_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(
            ["content_id", "detector_type", "detector", "score", "decision", "n_tokens_scored", "p_value", "latency_ms"]
        )
        for k, scores in self.state.wm_scores.items():
            for cid, r in scores.items():
                w.writerow(
                    [cid, "watermark", k, r.score, r.decision, r.n_tokens_scored, r.p_value, round(r.latency_ms, 3)]
                )
        for d, scores in self.state.det_scores.items():
            for cid, v in scores.items():
                w.writerow([cid, "ai_classifier", d, v, "", "", "", ""])
        return buf.getvalue()


def results_to_json(results: dict[str, Any]) -> str:
    return json.dumps(results, indent=2, default=str)
