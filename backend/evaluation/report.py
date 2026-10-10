"""Render experiment results as a human-readable Markdown research report.

Every number in the report is read from the results dictionary produced by the
runner; nothing is filled in by hand. Missing metrics are shown as "n/a" with the
reason where one was recorded.
"""

from __future__ import annotations

from typing import Any

STANDING_LIMITATIONS = [
    "Watermark detection indicates use of a specific key/tokenizer; it is not proof of authorship.",
    "AI-text classifiers are trained and tested on text from one generator and one domain (news-like web "
    "text); performance on other generators, domains or languages is not established by this experiment.",
    "Human labels rely on C4 (crawled April 2019); the corpus may contain some template/machine-written pages.",
    "Small CPU-scale models were used; absolute numbers will differ for larger generators.",
    "Confidence intervals are percentile bootstrap (AUROC, means) or Wilson (proportions) and assume i.i.d. "
    "samples; documents sharing a source domain may be correlated.",
]


def _f(v: Any, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _ci(d: dict | None, key: str, nd: int = 3) -> str:
    if not d or d.get(key) is None:
        reason = (d or {}).get("reason")
        return f"n/a ({reason})" if reason else "n/a"
    lo, hi = d.get("ci_low"), d.get("ci_high")
    if lo is None or hi is None:
        return _f(d[key], nd)
    return f"{d[key]:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def _rate(d: dict | None) -> str:
    if not d or d.get("survival_rate") is None:
        return "n/a"
    return f"{d['survival_rate']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}] (n={d['n']})"


def _prop(bm: dict, key: str) -> str:
    v = bm.get(key)
    ci = bm.get(f"{key}_ci")
    if v is None:
        return "n/a"
    if ci and ci[0] is not None:
        return f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{v:.3f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render_markdown(r: dict[str, Any]) -> str:
    exp = r.get("experiment", {})
    cfg = exp.get("config", {})
    ds = r.get("dataset", {})
    man = ds.get("manifest", {})
    L: list[str] = []
    L.append(f"# Experiment report: {exp.get('name', 'unnamed')}\n")
    L.append(
        f"- **Experiment ID:** `{exp.get('id')}`  \n- **Type:** {exp.get('type')}  \n"
        f"- **Seed:** {exp.get('seed')}  \n- **Config hash:** `{exp.get('config_hash')}`  \n"
        f"- **Software version:** {exp.get('software_version')}  \n- **Git commit:** {exp.get('git_commit')}"
    )
    if cfg.get("description"):
        L.append(f"\n> {cfg['description']}")

    # ----------------------------------------------------------- setup
    L.append("\n## 1. Setup\n")
    L.append(
        f"- **Dataset:** {ds.get('name')} version `{ds.get('version')}` — source `{man.get('source')}`, "
        f"license: {man.get('license')}"
    )
    L.append(f"- **Records SHA-256:** `{man.get('records_sha256')}`")
    L.append(
        f"- **Documents:** {man.get('n_documents')}; split counts (by document): {man.get('split_counts')}; "
        f"split unit: {man.get('split_unit')}"
    )
    gen = r.get("generation", {})
    if gen:
        L.append(f"- **Generator:** `{gen.get('model')}` with params `{gen.get('params')}`")
    env = exp.get("environment", {})
    if env:
        L.append(
            f"- **Environment:** Python {env.get('python')}, torch {env.get('torch')}, transformers "
            f"{env.get('transformers')}, {env.get('cpu_count')} CPUs, CUDA={env.get('cuda')}"
        )
    counts = r.get("counts", {})
    if counts:
        L.append("\n<details><summary>Sample counts (split:kind[:watermark][:attack])</summary>\n")
        L.append(_table(["group", "n"], [[k, str(v)] for k, v in counts.items()]))
        L.append("\n</details>")

    # ----------------------------------------------------------- leakage
    lk = r.get("leakage")
    if lk:
        L.append("\n## 2. Data integrity and leakage\n")
        L.append(
            f"Post-generation audit over {lk['n_items']} texts (MinHash/LSH, Jaccard ≥ {lk['threshold']}): "
            f"**{lk['exact_cross_split_duplicates']}** exact and **{lk['near_cross_split_duplicates']}** "
            f"near-duplicate cross-split pairs; **{lk['groups_crossing_splits']}** document groups crossing "
            f"splits. Leakage-free: **{lk['leakage_free']}**."
        )
        src = man.get("leakage_check")
        if src:
            L.append(f"Source-document audit at build time: leakage-free = {src.get('leakage_free')}.")

    # ----------------------------------------------------------- watermark detection
    wms = r.get("watermarks", {})
    if wms:
        L.append("\n## 3. Watermark detection (clean, held-out test split)\n")
        L.append(
            "Negatives = human continuations + unwatermarked AI continuations. Thresholds are either the "
            "method's theoretical default or chosen on **validation** negatives at the target FPR and then "
            "frozen for the test set. Missing scores (too few tokens) count as *not detected*.\n"
        )
        rows = []
        for k, w in wms.items():
            c = w["clean"]
            for tname, bm in c["by_threshold"].items():
                rows.append(
                    [
                        k,
                        tname,
                        _f(bm["threshold"], 3),
                        _ci(c["auroc"], "auroc"),
                        _prop(bm, "tpr"),
                        _prop(bm, "fpr"),
                        _f(bm.get("f1")),
                        f"{c['n_pos']}/{c['n_neg']}",
                    ]
                )
        L.append(
            _table(
                [
                    "method",
                    "threshold rule",
                    "threshold",
                    "AUROC [95% CI]",
                    "TPR [95% CI]",
                    "FPR [95% CI]",
                    "F1",
                    "n pos/neg",
                ],
                rows,
            )
        )
        L.append(
            "\n**Negative subgroups** (is the false-positive behaviour different on human vs. unwatermarked AI "
            "text?):\n"
        )
        rows = []
        for k, w in wms.items():
            for kind, sg in w["clean"]["neg_subgroups"].items():
                fprs = ", ".join(f"{n}={_f(v)}" for n, v in sg["fpr"].items())
                rows.append([k, kind, _ci(sg["auroc"], "auroc"), fprs])
        L.append(_table(["method", "negatives", "AUROC", "FPR by threshold"], rows))
        L.append("\n**Score distributions (test):**\n")
        rows = []
        for k, w in wms.items():
            for kind, d in w["clean"]["distributions"].items():
                rows.append(
                    [
                        k,
                        kind,
                        str(d.get("n", 0)),
                        _f(d.get("mean")),
                        _f(d.get("std")),
                        _f(d.get("median")),
                        _f(d.get("p05")),
                        _f(d.get("p95")),
                        str(w["clean"]["insufficient"].get(kind, 0)),
                    ]
                )
        L.append(_table(["method", "kind", "n", "mean", "std", "median", "p05", "p95", "insufficient"], rows))
        L.append("\n**Edge case — detection rate vs. number of scored tokens (theoretical threshold):**\n")
        rows = []
        for k, w in wms.items():
            for b in w["clean"]["length_bins"]:
                rows.append([k, b["tokens_scored"], _rate(b)])
        L.append(_table(["method", "tokens scored", "TPR [95% CI]"], rows) if rows else "_No data._")

        # ------------------------------------------------------- robustness
        L.append("\n## 4. Robustness to attacks\n")
        att = r.get("attacks", {})
        skipped = {k: v for k, v in att.items() if v.get("status") == "skipped"}
        if skipped:
            L.append(
                "Skipped attacks (not run, **no results reported**): "
                + "; ".join(f"`{k}` — {v['reason']}" for k, v in skipped.items())
                + "\n"
            )
        L.append(
            "`TPR after attack` uses all clean test positives as the denominator (failed transformations count "
            "as not detected). `Conditional survival` = P(detected after | detected before). AUROC is attacked "
            "positives vs clean negatives. `sim` = meaning preservation of the attacked text vs its parent.\n"
        )
        rows = []
        for k, w in wms.items():
            for a, ar in w["attacks"].items():
                th = ar["by_threshold"].get("theoretical", {})
                prim = next((v for n, v in ar["by_threshold"].items() if n.startswith("val_fpr")), {})
                fpr_h = (prim.get("fpr_attacked_human") or {}).get("fpr")
                rows.append(
                    [
                        k,
                        a,
                        _ci(ar["auroc_vs_clean_neg"], "auroc"),
                        _rate(th.get("tpr_after_attack")),
                        _rate(th.get("conditional_survival")),
                        _rate(prim.get("tpr_after_attack")),
                        _f(fpr_h),
                        _ci(ar["score_change"], "mean", 2),
                        _ci(ar["similarity_wm"], "mean"),
                        str(ar["insufficient_after_attack"]),
                    ]
                )
        L.append(
            _table(
                [
                    "method",
                    "attack",
                    "AUROC",
                    "TPR@theoretical",
                    "cond. survival@theoretical",
                    "TPR@val-FPR (primary)",
                    "FPR attacked human",
                    "Δscore",
                    "sim",
                    "insufficient",
                ],
                rows,
            )
            if rows
            else "_No attack results._"
        )
        if att:
            L.append("\n**Attack execution:**\n")
            rows = []
            for a, v in att.items():
                if v.get("status") != "ran":
                    rows.append([a, v["describe"]["attack_type"], "skipped", "-", "-", "-"])
                    continue
                c = v["counts"]
                rows.append(
                    [
                        a,
                        v["describe"]["attack_type"],
                        f"{c.get('succeeded', 0)}/{v['n_targets']} ok, {c.get('failed', 0)} failed",
                        _f(v["latency_ms"].get("mean"), 1),
                        v.get("similarity_metric") or "n/a",
                        "; ".join(f"{k}={_ci(s, 'mean')}" for k, s in v.get("similarity", {}).items()),
                    ]
                )
            L.append(
                _table(
                    ["attack", "type", "status", "mean latency (ms)", "similarity metric", "similarity by kind"], rows
                )
            )

    # ----------------------------------------------------------- detectors
    det = r.get("detectors")
    if det:
        L.append("\n## 5. AI-generated vs human text classifiers\n")
        L.append(
            "Trained on the train split (human vs **unwatermarked** AI), thresholds chosen on validation human "
            "texts at the target FPR, evaluated on test. Baseline: "
            f"`{det['baseline']}`.\n"
        )
        rows = []
        for name, d in det["results"].items():
            ct = d["clean_test"]
            for tname, bm in ct["by_threshold"].items():
                rows.append(
                    [
                        name + (" (baseline)" if name == det["baseline"] else ""),
                        tname,
                        _ci(ct["auroc"], "auroc"),
                        _prop(bm, "tpr"),
                        _prop(bm, "fpr"),
                        _f(bm.get("precision")),
                        _f(bm.get("f1")),
                        f"{d['n_train']['human']}/{d['n_train']['ai']}",
                        f"{d['n_test']['human']}/{d['n_test']['ai']}",
                        _f(d["score_ms_per_text"], 1),
                    ]
                )
        L.append(
            _table(
                [
                    "detector",
                    "threshold rule",
                    "AUROC [95% CI]",
                    "TPR",
                    "FPR",
                    "precision",
                    "F1",
                    "train h/ai",
                    "test h/ai",
                    "ms/text",
                ],
                rows,
            )
        )
        if det["paired_comparisons"]:
            L.append("\n**Paired comparison vs baseline (same test samples, stratified paired bootstrap):**\n")
            rows = [
                [
                    c["detector"],
                    c["baseline"],
                    _f(c["diff"]),
                    f"[{_f(c['ci_low'])}, {_f(c['ci_high'])}]",
                    _f(c["p_two_sided"]),
                    str(c["n"]),
                ]
                for c in det["paired_comparisons"]
            ]
            L.append(_table(["detector", "baseline", "ΔAUROC", "95% CI", "p (two-sided)", "n"], rows))
        L.append("\n**Distribution shift (watermarked / attacked AI text):**\n")
        rows = []
        for name, d in det["results"].items():
            for sname, sv in d["shift"].items():
                rows.append(
                    [
                        name,
                        sname,
                        _ci(sv["auroc"], "auroc"),
                        _f(sv.get("tpr_at_primary")),
                        _f(sv.get("fpr_at_primary")),
                        sv.get("negatives", "clean human"),
                    ]
                )
        L.append(
            _table(["detector", "condition", "AUROC", "TPR@primary", "FPR@primary", "negatives"], rows)
            if rows
            else "_None._"
        )
        for name, d in det["results"].items():
            if d.get("coefficients"):
                L.append(
                    f"\n`{name}` standardised logistic-regression coefficients: "
                    + ", ".join(f"{k}={v:+.2f}" for k, v in d["coefficients"].items())
                )

    cross = r.get("cross_task")
    if cross:
        L.append("\n## 6. Watermark detector vs. AI-text classifiers on watermarked text\n")
        L.append(
            "Both tasks differ: the watermark detector needs the key but no training; the classifiers need no "
            "key but were trained on unwatermarked AI text. AUROC on the same test samples:\n"
        )
        keys = [k for k in cross[0] if k.endswith("_auroc")]
        rows = [[c["task"], f"{c['n_pos']}/{c['n_neg']}", *[_f(c.get(k)) for k in keys]] for c in cross]
        L.append(_table(["task", "n pos/neg", *keys], rows))

    # ----------------------------------------------------------- quality
    q = r.get("quality")
    if q and q.get("groups"):
        L.append("\n## 7. Text quality (test split)\n")
        L.append(
            f"Oracle perplexity model: `{q.get('oracle_model') or 'not computed'}` (lower = more fluent under the "
            "oracle; very low values can also indicate repetition).\n"
        )
        rows = [
            [
                g,
                str(v["n"]),
                _ci(v.get("perplexity"), "mean", 2),
                _f(v.get("perplexity_median"), 2),
                _ci(v.get("repetition_3gram"), "mean"),
                _f(v["words"].get("mean"), 1),
            ]
            for g, v in q["groups"].items()
        ]
        L.append(_table(["group", "n", "PPL mean [95% CI]", "PPL median", "3-gram repetition", "mean words"], rows))

    # ----------------------------------------------------------- cost
    cost = r.get("cost")
    if cost:
        L.append("\n## 8. Latency and computational cost\n")
        L.append(f"Stage wall-clock seconds: {cost.get('stage_seconds')}\n")
        rows = [
            [k, _f(v.get("seconds"), 1), _f(v.get("tokens")), _f(v.get("tokens_per_s"), 1)]
            for k, v in cost.items()
            if isinstance(v, dict) and "seconds" in v
        ]
        if rows:
            L.append(_table(["generation", "seconds", "tokens", "tokens/s"], rows))
        for k, w in wms.items():
            d = w.get("detection_latency_ms", {})
            L.append(
                f"\n- `{k}` detection latency: mean {_f(d.get('mean'), 2)} ms, p95 {_f(d.get('p95'), 2)} ms "
                f"(n={d.get('n')})"
            )

    # ----------------------------------------------------------- warnings
    L.append("\n## 9. Warnings, limitations and failure analysis\n")
    for w in r.get("warnings", []):
        L.append(f"- ⚠ {w}")
    for lim in STANDING_LIMITATIONS:
        L.append(f"- {lim}")

    L.append("\n## 10. Reproduction\n")
    L.append(
        "```bash\n# Re-run with the identical configuration (new experiment ID, same seed):\n"
        f"curl -X POST $API/api/v1/experiments/{exp.get('id')}/rerun\n"
        "# or from the stored config:\npython -m backend.experiments.cli run <config.yaml>\n```"
    )
    return "\n".join(L) + "\n"
