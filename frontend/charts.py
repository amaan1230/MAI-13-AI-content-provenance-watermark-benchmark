"""Plotly figures built only from backend results (never from hard-coded values)."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

# Reference categorical slots 1-3 (validated all-pairs in light and dark); fixed order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
GRID = "rgba(128,128,128,0.18)"
MUTED = "#898781"


def color_map(keys: list[str]) -> dict[str, str]:
    """Colour follows the entity: assignment depends on the sorted key, not on display order."""
    return {k: SERIES[i % len(SERIES)] if i < len(SERIES) else MUTED for i, k in enumerate(sorted(keys))}


def _layout(fig: go.Figure, title: str, x: str, y: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 15}},
        xaxis_title=x,
        yaxis_title=y,
        height=height,
        margin={"l": 50, "r": 20, "t": 50, "b": 45},
        legend={"orientation": "h", "y": -0.22},
        hovermode="closest",
        bargap=0.25,
        bargroupgap=0.08,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def roc_figure(curves: dict[str, dict[str, Any]], title: str) -> go.Figure:
    fig = go.Figure()
    cmap = color_map(list(curves))
    for name, c in curves.items():
        if not c or not c.get("roc"):
            continue
        auc = c.get("auroc")
        label = f"{name} (AUROC {auc:.3f})" if auc is not None else name
        fig.add_trace(
            go.Scatter(
                x=c["roc"]["fpr"],
                y=c["roc"]["tpr"],
                mode="lines",
                name=label,
                line={"width": 2, "color": cmap[name]},
                hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra>" + name + "</extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="chance",
            line={"width": 1, "dash": "dot", "color": MUTED},
            hoverinfo="skip",
        )
    )
    fig = _layout(fig, title, "False-positive rate", "True-positive rate")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    return fig


def survival_figure(rows: list[dict[str, Any]], title: str) -> go.Figure:
    """Grouped bars: TPR after attack per (attack, method) with Wilson 95% CI error bars."""
    methods = sorted({r["method"] for r in rows})
    attacks = list(dict.fromkeys(r["attack"] for r in rows))
    cmap = color_map(methods)
    fig = go.Figure()
    for m in methods:
        pts = {r["attack"]: r for r in rows if r["method"] == m}
        ys = [pts.get(a, {}).get("rate") for a in attacks]
        hi = [(pts[a]["hi"] - pts[a]["rate"]) if a in pts and pts[a]["rate"] is not None else 0 for a in attacks]
        lo = [(pts[a]["rate"] - pts[a]["lo"]) if a in pts and pts[a]["rate"] is not None else 0 for a in attacks]
        fig.add_trace(
            go.Bar(
                x=attacks,
                y=ys,
                name=m,
                marker={"color": cmap[m], "line": {"width": 0}},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": hi,
                    "arrayminus": lo,
                    "thickness": 1.2,
                    "width": 3,
                    "color": MUTED,
                },
                customdata=[[pts.get(a, {}).get("n")] for a in attacks],
                hovertemplate="%{x}<br>TPR %{y:.3f}<br>n=%{customdata[0]}<extra>" + m + "</extra>",
            )
        )
    fig = _layout(fig, title, "Attack", "Detection rate (95% CI)")
    fig.update_yaxes(range=[0, 1.05])
    fig.update_layout(barmode="group")
    return fig


def tradeoff_figure(rows: list[dict[str, Any]], title: str) -> go.Figure:
    """Meaning preservation (x) vs detection after attack (y): the robustness frontier."""
    methods = sorted({r["method"] for r in rows})
    cmap = color_map(methods)
    fig = go.Figure()
    for m in methods:
        pts = [r for r in rows if r["method"] == m and r["sim"] is not None and r["rate"] is not None]
        fig.add_trace(
            go.Scatter(
                x=[p["sim"] for p in pts],
                y=[p["rate"] for p in pts],
                mode="markers",
                name=m,
                text=[p["attack"] for p in pts],
                marker={"size": 10, "color": cmap[m], "line": {"width": 2, "color": "white"}},
                hovertemplate="%{text}<br>similarity %{x:.3f}<br>TPR %{y:.3f}<extra>" + m + "</extra>",
            )
        )
    fig = _layout(fig, title, "Semantic similarity to original", "Detection rate after attack")
    fig.update_yaxes(range=[-0.03, 1.05])
    return fig


def histogram_figure(
    groups: dict[str, list[float]], title: str, x_label: str, threshold: float | None = None
) -> go.Figure:
    cmap = color_map(list(groups))
    fig = go.Figure()
    for name, vals in groups.items():
        fig.add_trace(
            go.Histogram(
                x=vals,
                name=f"{name} (n={len(vals)})",
                opacity=0.65,
                nbinsx=40,
                marker={"color": cmap[name], "line": {"width": 0}},
                hovertemplate="%{x}<br>count %{y}<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(barmode="overlay")
    if threshold is not None:
        fig.add_vline(
            x=threshold,
            line={"dash": "dash", "width": 1.5, "color": MUTED},
            annotation_text=f"threshold {threshold:.2f}",
            annotation_position="top right",
        )
    return _layout(fig, title, x_label, "Count")


def bar_ci_figure(rows: list[dict[str, Any]], title: str, y_label: str) -> go.Figure:
    """Single-series bars (one per group) with CI whiskers - one colour, the title names the measure."""
    fig = go.Figure(
        go.Bar(
            x=[r["name"] for r in rows],
            y=[r["mean"] for r in rows],
            marker={"color": SERIES[0], "line": {"width": 0}},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": [(r["hi"] - r["mean"]) if r.get("hi") is not None else 0 for r in rows],
                "arrayminus": [(r["mean"] - r["lo"]) if r.get("lo") is not None else 0 for r in rows],
                "thickness": 1.2,
                "width": 3,
                "color": MUTED,
            },
            hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
        )
    )
    return _layout(fig, title, "", y_label)
