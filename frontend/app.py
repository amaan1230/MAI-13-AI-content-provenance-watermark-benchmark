"""MAI-13 research dashboard (Streamlit). All data shown comes from the backend API.

streamlit run frontend/app.py
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import api_client as api  # noqa: E402
import charts  # noqa: E402

st.set_page_config(page_title="MAI-13 Provenance Benchmark", layout="wide")

SECTIONS = [
    "1 · Overview & health",
    "2 · Datasets",
    "3 · Generate & ingest",
    "4 · Watermark embed/detect",
    "5 · Attacks",
    "6 · Experiments",
    "7 · Baseline & robustness comparison",
    "8 · Provenance explorer",
    "9 · Metrics & error analysis",
    "10 · Reports & settings",
]
CONFIG_DIR = Path(__file__).resolve().parents[1] / "experiments" / "configs"


# ----------------------------------------------------------------------------- helpers
def show_error(res: api.ApiResult, what: str) -> None:
    st.error(f"**{what} failed** — `{res.error_code or res.status}`: {res.message}")
    st.caption(f"What to do: {res.actionable}" + (f"  ·  request id `{res.request_id}`" if res.request_id else ""))
    if res.details:
        with st.expander("Error details"):
            st.json(res.details)


def fmt(v, nd=3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def ci(d: dict | None, key: str = "auroc", nd: int = 3) -> str:
    if not d or d.get(key) is None:
        return f"n/a ({d.get('reason')})" if d and d.get("reason") else "n/a"
    if d.get("ci_low") is None:
        return fmt(d[key], nd)
    return f"{d[key]:.{nd}f} [{d['ci_low']:.{nd}f}, {d['ci_high']:.{nd}f}]"


@st.cache_data(ttl=10, show_spinner=False)
def _catalog():
    r = api.get("/api/v1/providers")
    return r.data if r.ok else None


def experiments_list(status: str | None = None, limit: int = 100) -> list[dict]:
    r = api.get("/api/v1/experiments", params={"limit": limit, **({"status": status} if status else {})})
    if not r.ok:
        show_error(r, "Listing experiments")
        return []
    return r.data["items"]


def pick_experiment(label: str, key: str, only_succeeded: bool = True) -> str | None:
    items = experiments_list("succeeded" if only_succeeded else None)
    if not items:
        st.info("No completed experiments yet. Create one in **6 · Experiments**.")
        return None
    opts = {f"{e['name']} · {e['id'][:8]} · {e['created_at'][:16]}": e["id"] for e in items}
    return opts[st.selectbox(label, list(opts), key=key)]


@st.cache_data(ttl=300, show_spinner=False)
def load_results(eid: str):
    r = api.get(f"/api/v1/experiments/{eid}/results")
    return r.data if r.ok else {"__error__": r.__dict__}


@st.cache_data(ttl=300, show_spinner=False)
def load_csv(eid: str, name: str):
    r = api.get(f"/api/v1/experiments/{eid}/artifacts/{name}", raw=True)
    return pd.read_csv(io.StringIO(r.data)) if r.ok else None


# ----------------------------------------------------------------------------- 1. overview
def page_overview():
    st.header("Overview & system health")
    h = api.get("/health")
    if not h.ok:
        show_error(h, "Health check")
        return
    c = st.columns(4)
    c[0].metric("API status", h.data["status"])
    c[1].metric("Database", f"{h.data['database']} ({h.data['database_dialect']})")
    c[2].metric("Providers available", h.data["providers_available"])
    c[3].metric("Version", h.data["version"])
    cat = _catalog()
    if cat:
        st.subheader("Providers")
        for p in cat["providers"]:
            (st.success if p["available"] else st.warning)(
                f"**{p['name']}** ({p['kind']}, model `{p['model']}`): {p['message']}"
            )
    st.subheader("Recent experiments")
    items = experiments_list(limit=10)
    if items:
        df = pd.DataFrame(items)[["name", "status", "stage", "progress", "experiment_type", "seed", "created_at", "id"]]
        st.dataframe(df, hide_index=True, width="stretch")
    st.caption("Watermark detection and AI-text classification are separate tasks; neither proves authorship.")


# ----------------------------------------------------------------------------- 2. datasets
def page_datasets():
    st.header("Dataset management")
    r = api.get("/api/v1/datasets")
    if not r.ok:
        show_error(r, "Listing datasets")
        return
    if not r.data:
        st.info(
            "No dataset versions registered yet. Datasets are materialised and versioned automatically when an "
            "experiment runs (source documents are cached under DATASET_DIR with a SHA-256 manifest)."
        )
        return
    df = pd.DataFrame(r.data)
    st.dataframe(
        df[["name", "version", "source", "license", "n_items", "split_seed", "created_at"]],
        hide_index=True,
        width="stretch",
    )
    sel = st.selectbox("Inspect version", [f"{d['name']} / {d['version']}" for d in r.data])
    d = r.data[[f"{x['name']} / {x['version']}" for x in r.data].index(sel)]
    c = st.columns(3)
    c[0].metric("Documents", d["n_items"])
    c[1].metric("Leakage-free at build", str(d["meta"].get("leakage_free")))
    c[2].metric("Split seed", d["split_seed"])
    st.write("Split counts (by source document):", d["meta"].get("split_counts"))
    st.caption(f"Manifest: `{d['manifest_uri']}` · sha256 `{d['manifest_sha256']}`")


# ----------------------------------------------------------------------------- 3. generate & ingest
def page_content():
    st.header("Content generation and ingestion")
    tab_g, tab_i, tab_l = st.tabs(["Generate", "Ingest", "Browse"])
    cat = _catalog() or {"providers": [], "watermark_methods": []}
    with tab_g:
        avail = {p["name"]: p for p in cat["providers"]}
        provider = st.radio("Provider", ["hf_local", "gemini"], horizontal=True)
        if provider == "gemini" and not avail.get("gemini", {}).get("available"):
            st.warning(avail.get("gemini", {}).get("message", "Gemini is not configured."))
        prompt = st.text_area("Prompt", "The city council met on Tuesday to discuss", height=100)
        c = st.columns(4)
        model = c[0].text_input("HF model (blank = default)", "") if provider == "hf_local" else None
        max_new = c[1].number_input("Max new tokens", 8, 1024, 120)
        temp = c[2].number_input("Temperature", 0.1, 2.0, 0.8, 0.1)
        seed = c[3].number_input("Seed", 0, 10**9, 0)
        wm = None
        if provider == "hf_local":
            gen_methods = [m["name"] for m in cat["watermark_methods"] if m["kind"] == "generation_time"]
            choice = st.selectbox("Generation-time watermark", ["none", *gen_methods])
            if choice != "none":
                c2 = st.columns(2)
                wm = {
                    "method": choice,
                    "config": {
                        "gamma": c2[0].slider("gamma", 0.05, 0.75, 0.25, 0.05),
                        "delta": c2[1].slider("delta", 0.0, 10.0, 2.0, 0.5),
                    },
                }
        if st.button("Generate", type="primary"):
            body = {
                "prompt": prompt,
                "provider": provider,
                "max_new_tokens": int(max_new),
                "temperature": temp,
                "seed": int(seed),
                "watermark": wm,
            }
            if model:
                body["model_name"] = model
            with st.spinner("Generating (first call loads the model; CPU generation can take a minute)…"):
                r = api.post("/api/v1/content/generate", body, timeout=600)
            if r.ok:
                st.success(
                    f"Generated {r.data['n_tokens']} tokens with `{r.data['model']}` in "
                    f"{r.data['latency_ms']:.0f} ms · content id `{r.data['content_id']}`"
                )
                st.text_area("Output", r.data["text"], height=200)
                if r.data["watermark"]:
                    st.json(r.data["watermark"])
            else:
                show_error(r, "Generation")
    with tab_i:
        text = st.text_area("Text", height=160, key="ingest_text")
        c = st.columns(3)
        source_type = c[0].selectbox("Source type", ["unknown", "human", "ai", "ai_watermarked", "transformed"])
        label = c[1].selectbox("Label", ["unknown", "human", "ai"])
        parent = c[2].text_input("Parent content id (optional)")
        label_source = st.text_input(
            "Label justification (required for human/ai labels)",
            help="E.g. dataset name and licence, or how authorship was established.",
        )
        if st.button("Ingest", type="primary"):
            r = api.post(
                "/api/v1/content/ingest",
                {
                    "text": text,
                    "source_type": source_type,
                    "label": label,
                    "label_source": label_source or None,
                    "parent_id": parent or None,
                },
            )
            if r.ok:
                st.success(f"Registered `{r.data['id']}` · sha256 `{r.data['content_hash'][:16]}…`")
            else:
                show_error(r, "Ingest")
    with tab_l:
        c = st.columns(3)
        stype = c[0].selectbox(
            "Source type filter", ["(any)", "human", "ai", "ai_watermarked", "transformed", "unknown"]
        )
        page = c[1].number_input("Page", 1, 10**6, 1)
        size = c[2].selectbox("Page size", [25, 50, 100])
        params = {"limit": size, "offset": (page - 1) * size}
        if stype != "(any)":
            params["source_type"] = stype
        r = api.get("/api/v1/content", params=params)
        if r.ok:
            st.caption(f"{r.data['total']} items")
            if r.data["items"]:
                df = pd.DataFrame(r.data["items"])
                df["text"] = df["text"].fillna("[not retained]").str.slice(0, 120)
                st.dataframe(
                    df[["id", "source_type", "label", "split", "char_len", "text", "created_at"]],
                    hide_index=True,
                    width="stretch",
                )
        else:
            show_error(r, "Listing content")


# ----------------------------------------------------------------------------- 4. watermark
def page_watermark():
    st.header("Watermark embedding and detection")
    cat = _catalog()
    if not cat:
        st.error("Cannot load method catalogue from the API.")
        return
    methods = {m["name"]: m for m in cat["watermark_methods"]}
    name = st.selectbox("Method", list(methods))
    m = methods[name]
    with st.expander("Method card: assumptions and limitations"):
        st.write(
            f"**Kind:** {m['kind']} · **Reference:** {m['reference']} · **Key fingerprint:** `{m['key_fingerprint']}`"
        )
        for lim in m["limitations"]:
            st.write(f"- {lim}")
    cfg_text = st.text_area(
        "Config (JSON, optional)", "{}", height=80, help='e.g. {"gamma": 0.25, "delta": 2.0, "z_threshold": 4}'
    )
    try:
        cfg = json.loads(cfg_text or "{}")
    except json.JSONDecodeError as exc:
        st.error(f"Config is not valid JSON: {exc}")
        return
    col_e, col_d = st.columns(2)
    with col_e:
        st.subheader("Embed")
        hint = "Prompt (the model continues it)" if m["kind"] == "generation_time" else "Text to mark"
        t = st.text_area(hint, height=140, key="embed_text")
        if st.button("Embed", type="primary"):
            with st.spinner("Embedding…"):
                r = api.post("/api/v1/watermarks/embed", {"method": name, "text": t, "config": cfg}, timeout=600)
            if r.ok:
                st.success(
                    f"content id `{r.data['content_id']}` · {fmt(r.data['n_tokens'])} tokens · "
                    f"{r.data['latency_ms']:.0f} ms"
                )
                st.text_area("Watermarked text", r.data["text"], height=180)
                st.session_state["last_wm_content"] = r.data["content_id"]
            else:
                show_error(r, "Embedding")
    with col_d:
        st.subheader("Detect")
        mode = st.radio("Input", ["Text", "Content id"], horizontal=True)
        if mode == "Text":
            payload = {"text": st.text_area("Text to test", height=140, key="detect_text")}
        else:
            payload = {"content_id": st.text_input("Content id", st.session_state.get("last_wm_content", ""))}
        if st.button("Detect"):
            r = api.post("/api/v1/watermarks/detect", {"method": name, "config": cfg, **payload})
            if r.ok:
                d = r.data
                c = st.columns(3)
                c[0].metric("Score", fmt(d["score"], 2))
                c[1].metric("Threshold", fmt(d["threshold"], 2))
                c[2].metric("Decision", d["decision"])
                st.caption(f"p-value {fmt(d['p_value'], 6)} · tokens scored {d['n_tokens_scored']}")
                st.info(d["interpretation"])
            else:
                show_error(r, "Detection")


# ----------------------------------------------------------------------------- 5. attacks
def page_attacks():
    st.header("Attack configuration")
    cat = _catalog()
    if not cat:
        st.error("Cannot load attack catalogue from the API.")
        return
    attacks = {a["name"]: a for a in cat["attacks"]}
    name = st.selectbox("Attack", list(attacks))
    a = attacks[name]
    st.caption(
        f"Type: **{a['attack_type']}** · deterministic: {a['deterministic']} · requires: "
        f"{', '.join(a['requires']) or 'nothing'}"
    )
    defaults = {k: v.get("default") for k, v in a["config_schema"].get("properties", {}).items()}
    cfg_text = st.text_area("Config (JSON)", json.dumps(defaults, indent=1), height=140)
    mode = st.radio("Input", ["Content id (records lineage)", "Free text"], horizontal=True)
    payload = (
        {"content_id": st.text_input("Content id", st.session_state.get("last_wm_content", ""))}
        if mode.startswith("Content")
        else {"text": st.text_area("Text", height=140)}
    )
    seed = st.number_input("Seed", 0, 10**9, 0)
    if st.button("Run attack", type="primary"):
        try:
            cfg = json.loads(cfg_text)
        except json.JSONDecodeError as exc:
            st.error(f"Config is not valid JSON: {exc}")
            return
        with st.spinner("Running attack…"):
            r = api.post(
                "/api/v1/attacks/run", {"attack": name, "config": cfg, "seed": int(seed), **payload}, timeout=600
            )
        if not r.ok:
            show_error(r, "Attack")
            return
        d = r.data
        if d["status"] != "succeeded":
            st.error(f"Attack {d['status']}: {d['error']}")
            return
        c = st.columns(3)
        c[0].metric("Status", d["status"])
        c[1].metric(f"Similarity ({d['similarity_metric']})", fmt(d["similarity"]))
        c[2].metric("Latency (ms)", fmt(d["latency_ms"], 1))
        st.text_area("Attacked text", d["text"], height=180)
        if d["child_content_id"]:
            st.success(
                f"Lineage recorded: `{d['parent_content_id']}` → `{d['child_content_id']}` "
                f"(transformation `{d['transformation_id']}`)"
            )
            st.session_state["last_wm_content"] = d["child_content_id"]


# ----------------------------------------------------------------------------- 6. experiments
def page_experiments():
    st.header("Experiment creation and execution")
    tab_new, tab_status = st.tabs(["New experiment", "Status & control"])
    with tab_new:
        presets = sorted(p.name for p in CONFIG_DIR.glob("*.yaml"))
        preset = st.selectbox("Start from preset", presets) if presets else None
        text = (CONFIG_DIR / preset).read_text(encoding="utf-8") if preset else "name: my-experiment\n"
        cfg_text = st.text_area("Configuration (YAML)", text, height=420, key=f"cfg_{preset}")
        c = st.columns(2)
        try:
            cfg = yaml.safe_load(cfg_text)
        except yaml.YAMLError as exc:
            st.error(f"YAML error: {exc}")
            return
        if c[0].button("Validate"):
            r = api.post("/api/v1/experiments/validate", cfg)
            st.success("Configuration is valid.") if r.ok else show_error(r, "Validation")
        if c[1].button("Create & run", type="primary"):
            r = api.post("/api/v1/experiments", cfg)
            if r.ok:
                st.success(f"Queued experiment `{r.data['id']}`. Follow it in *Status & control*.")
                st.session_state["watch_exp"] = r.data["id"]
            else:
                show_error(r, "Experiment creation")
    with tab_status:
        items = experiments_list(limit=50)
        if not items:
            st.info("No experiments yet.")
            return
        opts = {f"{e['name']} · {e['status']} · {e['id'][:8]}": e["id"] for e in items}
        default = next((i for i, v in enumerate(opts.values()) if v == st.session_state.get("watch_exp")), 0)
        eid = opts[st.selectbox("Experiment", list(opts), index=default)]
        r = api.get(f"/api/v1/experiments/{eid}")
        if not r.ok:
            show_error(r, "Loading experiment")
            return
        e = r.data
        c = st.columns(4)
        c[0].metric("Status", e["status"])
        c[1].metric("Stage", e["stage"] or "-")
        c[2].metric("Seed", e["seed"])
        c[3].metric("Software", e["software_version"])
        st.progress(float(e["progress"]), text=f"{e['progress'] * 100:.0f}%")
        if e["status"] == "failed":
            st.error(f"Failed: {e['error']}")
            with st.expander("Error details (stage, traceback)"):
                st.json(e["error_details"])
        st.caption(
            f"config hash `{e['config_hash'][:16]}` · dataset version `{e['dataset_version_id']}` · "
            f"MLflow run `{e['mlflow_run_id']}` · git `{e['git_commit']}`"
        )
        b = st.columns(3)
        if e["status"] in ("pending", "running") and b[0].button("Cancel"):
            rr = api.post(f"/api/v1/experiments/{eid}/cancel")
            st.success("Cancellation requested.") if rr.ok else show_error(rr, "Cancel")
        if e["status"] in ("succeeded", "failed", "cancelled") and b[1].button("Re-run (same config & seed)"):
            rr = api.post(f"/api/v1/experiments/{eid}/rerun")
            if rr.ok:
                st.session_state["watch_exp"] = rr.data["id"]
                st.success(f"Re-run queued: `{rr.data['id']}`")
            else:
                show_error(rr, "Re-run")
        with st.expander("Configuration"):
            st.json(e["config"])
        if e["metrics_summary"]:
            with st.expander("Metrics summary", expanded=True):
                st.dataframe(
                    pd.DataFrame(sorted(e["metrics_summary"].items()), columns=["metric", "value"]),
                    hide_index=True,
                    width="stretch",
                )
        if e["status"] in ("pending", "running") and b[2].toggle("Auto-refresh", value=True):
            time.sleep(3)
            st.rerun()


# ----------------------------------------------------------------------------- 7. comparison
def page_comparison():
    st.header("Baseline and robustness comparison")
    eid = pick_experiment("Experiment", "cmp_exp")
    if not eid:
        return
    res = load_results(eid)
    if "__error__" in res:
        st.error(f"Cannot load results: {res['__error__'].get('message')}")
        return
    wms = res.get("watermarks", {})
    if wms:
        st.subheader("Watermark detection on clean held-out text")
        st.plotly_chart(
            charts.roc_figure(
                {k: {"roc": w["clean"]["roc"], "auroc": w["clean"]["auroc"]["auroc"]} for k, w in wms.items()},
                "ROC — watermarked vs human + unwatermarked AI",
            ),
            width="stretch",
        )
        rule = st.radio("Threshold rule", sorted({n for w in wms.values() for n in w["thresholds"]}), horizontal=True)
        rows, table = [], []
        for k, w in wms.items():
            for a, ar in w["attacks"].items():
                t = ar["by_threshold"].get(rule)
                if not t:
                    continue
                s = t["tpr_after_attack"]
                rows.append(
                    {
                        "method": k,
                        "attack": a,
                        "rate": s["survival_rate"],
                        "lo": s["ci_low"],
                        "hi": s["ci_high"],
                        "n": s["n"],
                        "sim": ar["similarity_wm"]["mean"],
                    }
                )
                table.append(
                    {
                        "method": k,
                        "attack": a,
                        "TPR after attack": ci(s, "survival_rate"),
                        "conditional survival": ci(t["conditional_survival"], "survival_rate"),
                        "AUROC": ci(ar["auroc_vs_clean_neg"]),
                        "FPR attacked human": fmt((t.get("fpr_attacked_human") or {}).get("fpr")),
                        "similarity": ci(ar["similarity_wm"], "mean"),
                    }
                )
        if rows:
            st.subheader("Robustness: detection after attack")
            st.plotly_chart(charts.survival_figure(rows, f"Detection rate after attack ({rule})"), width="stretch")
            st.plotly_chart(charts.tradeoff_figure(rows, "Meaning preservation vs watermark survival"), width="stretch")
            st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        skipped = {k: v for k, v in res.get("attacks", {}).items() if v.get("status") == "skipped"}
        for k, v in skipped.items():
            st.warning(f"Attack `{k}` was skipped (no results): {v['reason']}")
    det = res.get("detectors")
    if det:
        st.subheader("AI-text classifiers vs baseline")
        rs = det["results"]
        st.plotly_chart(
            charts.roc_figure(
                {k: {"roc": d["clean_test"]["roc"], "auroc": d["clean_test"]["auroc"]["auroc"]} for k, d in rs.items()},
                "ROC — human vs unwatermarked AI (test)",
            ),
            width="stretch",
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "detector": k + (" (baseline)" if k == det["baseline"] else ""),
                        "AUROC": ci(d["clean_test"]["auroc"]),
                        **{f"TPR@{n}": fmt(b["tpr"]) for n, b in d["clean_test"]["by_threshold"].items()},
                        **{f"FPR@{n}": fmt(b["fpr"]) for n, b in d["clean_test"]["by_threshold"].items()},
                        "ms/text": d["score_ms_per_text"],
                    }
                    for k, d in rs.items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        if det["paired_comparisons"]:
            st.write("**Paired ΔAUROC vs baseline (same test samples):**")
            st.dataframe(pd.DataFrame(det["paired_comparisons"]), hide_index=True, width="stretch")
    if res.get("cross_task"):
        st.subheader("Watermark detector vs classifiers on watermarked text")
        st.dataframe(pd.DataFrame(res["cross_task"]), hide_index=True, width="stretch")
    cmp_ids = st.multiselect(
        "Compare metric summaries across experiments", [e["id"] for e in experiments_list("succeeded")], default=[eid]
    )
    if len(cmp_ids) > 1:
        r = api.get("/api/v1/experiments/compare", params={"ids": ",".join(cmp_ids)})
        if r.ok:
            df = pd.DataFrame({f"{e['name']}·{e['id'][:6]}": e["metrics"] for e in r.data["experiments"]})
            st.dataframe(df, width="stretch")


# ----------------------------------------------------------------------------- 8. provenance
def page_provenance():
    st.header("Provenance explorer")
    cid = st.text_input("Content id", st.session_state.get("last_wm_content", ""))
    if not cid:
        st.info("Enter a content id (from generation, ingestion, attacks, or an experiment's samples.csv).")
        return
    r = api.get(f"/api/v1/provenance/{cid}")
    if not r.ok:
        show_error(r, "Provenance lookup")
        return
    p = r.data
    c = p["content"]
    st.warning(p["disclaimer"])
    cols = st.columns(4)
    cols[0].metric("Source type", c["source_type"])
    cols[1].metric("Label", c["label"])
    cols[2].metric("Integrity", "verified" if p["integrity"]["verified"] else "not verified")
    cols[3].metric("Split", c["split"] or "-")
    st.caption(
        f"label source: {c['label_source'] or '—'} · sha256 `{c['content_hash']}` · created by "
        f"{c['created_by']} at {c['created_at']}"
    )
    st.text_area("Text", c["text"] or "[raw text not retained]", height=140)
    st.subheader("Lineage")
    chain = list(reversed(p["ancestors"])) + [c]
    st.markdown(" → ".join(f"`{n['id'][:8]}` ({n['source_type']})" for n in chain))
    if p["transformations_in"]:
        t = p["transformations_in"][0]
        st.write(
            f"Produced by **{t['attack_name']}** ({t['attack_type']} v{t['attack_version']}), seed {t['seed']}, "
            f"similarity {fmt(t['semantic_similarity'])}"
        )
    if p["descendants"]:
        st.write(f"{len(p['descendants'])} descendant(s):")
        st.dataframe(
            pd.DataFrame(p["descendants"])[["id", "parent_id", "source_type", "label", "char_len"]],
            hide_index=True,
            width="stretch",
        )
    if p["watermark_runs"]:
        st.subheader("Watermark runs")
        st.dataframe(
            pd.DataFrame(p["watermark_runs"])[
                ["method", "method_version", "key_fingerprint", "n_tokens", "latency_ms", "created_at"]
            ],
            hide_index=True,
            width="stretch",
        )
    if p["detections"]:
        st.subheader("Detection history")
        st.dataframe(
            pd.DataFrame(p["detections"])[
                [
                    "detector_type",
                    "detector_name",
                    "score",
                    "threshold",
                    "decision",
                    "p_value",
                    "n_tokens_scored",
                    "created_at",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
    with st.expander("Integrity checks"):
        st.json(p["integrity"])
    cand = st.text_area("Verify a candidate copy of this text", height=100)
    if st.button("Verify copy") and cand:
        v = api.post(f"/api/v1/provenance/{cid}/verify", {"text": cand})
        st.json(v.data) if v.ok else show_error(v, "Verification")


# ----------------------------------------------------------------------------- 9. metrics & errors
def page_metrics():
    st.header("Metrics and error analysis")
    eid = pick_experiment("Experiment", "met_exp")
    if not eid:
        return
    res = load_results(eid)
    if "__error__" in res:
        st.error(f"Cannot load results: {res['__error__'].get('message')}")
        return
    samples, dets = load_csv(eid, "samples.csv"), load_csv(eid, "detections.csv")
    wms = res.get("watermarks", {})
    if wms and samples is not None and dets is not None:
        k = st.selectbox("Watermark method", list(wms))
        attack = st.selectbox("Condition", ["clean", *wms[k]["attacks"]])
        d = dets[(dets.detector_type == "watermark") & (dets.detector == k)].merge(samples, on="content_id")
        d = d[d.split == "test"]
        d = d[d.attack_key.isna()] if attack == "clean" else d[(d.attack_key == attack) | (d.kind == "ai")]
        d = d[(d.kind != "wm") | (d.wm_key == k)]
        groups = {kind: d[(d.kind == kind)]["score"].dropna().tolist() for kind in ("human", "ai", "wm")}
        groups = {("watermarked" if g == "wm" else g): v for g, v in groups.items() if v}
        thr = wms[k]["thresholds"]["theoretical"]["threshold"]
        st.plotly_chart(
            charts.histogram_figure(groups, f"{k} score distribution · test · {attack}", "Score", thr), width="stretch"
        )
        ins = d[d.decision == "insufficient"].groupby("kind").size().to_dict()
        if ins:
            st.warning(f"Insufficient-token decisions (counted as not detected): {ins}")
        lb = wms[k]["clean"]["length_bins"]
        if lb:
            st.subheader("Edge case: detection vs text length")
            st.plotly_chart(
                charts.bar_ci_figure(
                    [
                        {"name": b["tokens_scored"], "mean": b["survival_rate"], "lo": b["ci_low"], "hi": b["ci_high"]}
                        for b in lb
                    ],
                    f"{k}: TPR by tokens scored (theoretical threshold)",
                    "TPR",
                ),
                width="stretch",
            )
        st.subheader("Negative subgroups")
        st.dataframe(
            pd.DataFrame(
                [
                    {"negatives": g, "AUROC": ci(v["auroc"]), **{f"FPR {n}": fmt(x) for n, x in v["fpr"].items()}}
                    for g, v in wms[k]["clean"]["neg_subgroups"].items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.subheader("Hardest cases (lowest-scoring watermarked test texts)")
        hard = (
            d[(d.kind == "wm")]
            .sort_values("score")
            .head(10)[["content_id", "score", "decision", "n_tokens_scored", "attack_key", "n_words"]]
        )
        st.dataframe(hard, hide_index=True, width="stretch")
        st.caption("Open a content id in the Provenance explorer to inspect the text and its lineage.")
    q = res.get("quality", {}).get("groups")
    if q:
        st.subheader("Text quality (oracle perplexity)")
        rows = [
            {
                "name": g,
                "mean": v["perplexity"]["mean"],
                "lo": v["perplexity"]["ci_low"],
                "hi": v["perplexity"]["ci_high"],
            }
            for g, v in q.items()
            if v.get("perplexity")
        ]
        if rows:
            st.plotly_chart(
                charts.bar_ci_figure(
                    rows, f"Perplexity under {res['quality']['oracle_model']} (lower = more fluent)", "PPL"
                ),
                width="stretch",
            )
    st.subheader("Data integrity")
    lk = res.get("leakage", {})
    st.write(
        f"Leakage-free: **{lk.get('leakage_free')}** · exact cross-split duplicates "
        f"{lk.get('exact_cross_split_duplicates')} · near duplicates {lk.get('near_cross_split_duplicates')} · "
        f"groups crossing splits {lk.get('groups_crossing_splits')}"
    )
    st.write("Sample counts (split:kind[:wm][:attack]):")
    st.dataframe(pd.DataFrame(sorted(res.get("counts", {}).items()), columns=["group", "n"]), hide_index=True)
    st.subheader("Cost")
    st.json(res.get("cost", {}))
    for w in res.get("warnings", []):
        st.warning(w)


# ----------------------------------------------------------------------------- 10. reports & settings
def page_reports():
    st.header("Experiment reports and settings")
    eid = pick_experiment("Experiment", "rep_exp")
    if eid:
        r = api.get(f"/api/v1/reports/{eid}", raw=True)
        if r.ok:
            st.download_button("Download report.md", r.data, file_name=f"report-{eid[:8]}.md")
            arts = api.get(f"/api/v1/experiments/{eid}/artifacts")
            if arts.ok:
                st.dataframe(
                    pd.DataFrame(arts.data)[["name", "kind", "size_bytes", "sha256"]], hide_index=True, width="stretch"
                )
            st.markdown(r.data)
        else:
            show_error(r, "Loading report")
    st.divider()
    st.subheader("Settings (read-only)")
    st.write(
        f"API base URL: `{api.API_BASE_URL}` · API key configured for dashboard: {'yes' if api._API_KEY else 'no'}"
    )
    h = api.get("/health")
    if h.ok:
        st.json(h.data)
    st.caption("Secrets are configured in the backend's .env and never displayed here.")


PAGES = dict(
    zip(
        SECTIONS,
        [
            page_overview,
            page_datasets,
            page_content,
            page_watermark,
            page_attacks,
            page_experiments,
            page_comparison,
            page_provenance,
            page_metrics,
            page_reports,
        ],
        strict=True,
    )
)

with st.sidebar:
    st.title("MAI-13")
    st.caption("AI content provenance & watermark robustness benchmark — text modality")
    section = st.radio("Section", SECTIONS, label_visibility="collapsed")
PAGES[section]()
