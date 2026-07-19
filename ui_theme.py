"""
Shared design-system helpers for the Streamlit presentation layer.

Pure rendering/formatting only - no analysis logic lives here, and nothing
in this module calls the LLM, RAG pipeline, or any backend agent. It exists
so every tab across the app (Overview & Analytics, Knowledge Extraction,
Contradictions, Root Cause, Ideas & Recommendations, Risk Indicators,
Proposal Preview) shares one consistent visual language - one color
palette, one chip/badge style, one set of chart wrappers - instead of each
section inventing its own look.

Chart backgrounds are left transparent (paper_bgcolor/plot_bgcolor =
"rgba(0,0,0,0)") so Streamlit's own light/dark theme shows through rather
than fighting it with a hardcoded template.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Palette - reused for category bars/treemap/network nodes so the same
# category (e.g. "Agentic AI") always reads as the same color everywhere.
# ---------------------------------------------------------------------------

CATEGORY_PALETTE = [
    "#6366f1", "#0ea5e9", "#14b8a6", "#f59e0b",
    "#ef4444", "#8b5cf6", "#22c55e", "#ec4899",
]

CLASSIFICATION_COLORS = {
    "Agreement": "#22c55e",
    "Contradiction": "#ef4444",
    "Partial Contradiction": "#f59e0b",
    "Different Context": "#0ea5e9",
    "Insufficient Evidence": "#94a3b8",
}

EVIDENCE_TYPE_COLORS = {
    "Supported by Retrieved Literature": "#22c55e",
    "General AI Best Practice": "#64748b",
}

SOURCE_COLORS = {
    "Full PDF": "#22c55e",
    "Abstract fallback": "#f59e0b",
    "Not Available": "#94a3b8",
}


def category_color(index: int) -> str:
    return CATEGORY_PALETTE[index % len(CATEGORY_PALETTE)]


# ---------------------------------------------------------------------------
# CSS injection + chips
# ---------------------------------------------------------------------------

def inject_css() -> None:
    """Call once near the top of main(). Only styles elements this module
    itself renders (chips, tag-cloud, hero card) - never touches Streamlit's
    internal DOM classes, so it can't break across Streamlit version bumps."""
    st.markdown(
        """
        <style>
        .rs-chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 10px 0; }
        .rs-chip {
            display: inline-block; padding: 3px 11px; margin: 0 6px 6px 0;
            border-radius: 999px; font-size: 0.82rem; font-weight: 600;
            line-height: 1.6; white-space: nowrap;
        }
        .rs-tagcloud { line-height: 2.4; padding: 8px 4px; }
        .rs-hero {
            border: 1px solid rgba(128,128,128,0.25); border-radius: 14px;
            padding: 22px 26px; margin: 10px 0 18px 0;
        }
        .rs-hero h3 { margin-top: 0; }
        .rs-muted { opacity: 0.7; font-size: 0.9rem; }
        .rs-agent-dot {
            display: inline-flex; align-items: center; gap: 6px;
            margin: 0 14px 6px 0; font-size: 0.88rem;
        }
        .rs-agent-dot .dot {
            width: 10px; height: 10px; border-radius: 50%; display: inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def chip(label: str, color: str) -> str:
    return (
        f'<span class="rs-chip" style="background:{color}22;color:{color};'
        f'border:1px solid {color}55;">{label}</span>'
    )


def chip_row(labels: Sequence[str], color: str = CATEGORY_PALETTE[0]) -> None:
    """Renders a wrapped row of chips, all the same color, in one call."""
    labels = [l for l in labels if l]
    if not labels:
        return
    html = '<div class="rs-chip-row">' + "".join(chip(l, color) for l in labels) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def mixed_chip_row(items: Sequence[Tuple[str, str]]) -> None:
    """items = [(label, color), ...] - for rows where each chip needs its
    own color (e.g. per-classification chips)."""
    items = [(l, c) for l, c in items if l]
    if not items:
        return
    html = '<div class="rs-chip-row">' + "".join(chip(l, c) for l, c in items) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def tag_cloud(items: Sequence[Tuple[str, int]], *, color: str = CATEGORY_PALETTE[0]) -> None:
    """items = [(value, count), ...]. CSS-only tag cloud (font-size scaled
    by count) - deliberately avoids a wordcloud image library."""
    items = [(v, c) for v, c in items if v]
    if not items:
        return
    max_count = max(c for _, c in items) or 1
    spans = []
    for value, count in items:
        size = 0.85 + 1.15 * (count / max_count)
        opacity = 0.55 + 0.45 * (count / max_count)
        spans.append(
            f'<span style="font-size:{size:.2f}em;color:{color};opacity:{opacity:.2f};'
            f'font-weight:600;margin:0 10px 8px 0;display:inline-block;">{value}</span>'
        )
    st.markdown(f'<div class="rs-tagcloud">{"".join(spans)}</div>', unsafe_allow_html=True)


def agent_status_row(statuses: Sequence[Tuple[str, bool]]) -> None:
    """statuses = [(label, done), ...] - a row of filled/hollow status dots."""
    spans = []
    for label, done in statuses:
        color = "#22c55e" if done else "#94a3b8"
        spans.append(
            f'<span class="rs-agent-dot"><span class="dot" style="background:{color};"></span>{label}</span>'
        )
    st.markdown("".join(spans), unsafe_allow_html=True)


def section_header(icon: str, title: str, caption: str = "") -> None:
    st.markdown(f"## {icon} {title}")
    if caption:
        st.caption(caption)


def metric_row(items: List[Tuple[str, object]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


# ---------------------------------------------------------------------------
# Plotly wrappers - thin, consistent theming only
# ---------------------------------------------------------------------------

def _base_layout(fig: go.Figure, *, height: int = 380, title: str = "") -> go.Figure:
    fig.update_layout(
        title=title or None,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plotly_bar_ranking(
    labels: Sequence[str], values: Sequence[float], *, title: str = "",
    color: Optional[str] = None, height: int = 380,
) -> go.Figure:
    order = sorted(range(len(values)), key=lambda i: values[i])
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=color or CATEGORY_PALETTE[0],
        text=values, textposition="outside",
    ))
    return _base_layout(fig, height=max(height, 36 * len(labels) + 60), title=title)


def plotly_donut(labels: Sequence[str], values: Sequence[float], *, title: str = "", height: int = 380) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=[category_color(i) for i in range(len(labels))]),
    ))
    return _base_layout(fig, height=height, title=title)


def plotly_heatmap(
    z: Sequence[Sequence[float]], x_labels: Sequence[str], y_labels: Sequence[str], *,
    colorscale="RdYlGn", title: str = "", height: Optional[int] = None,
    hover_text: Optional[Sequence[Sequence[str]]] = None,
) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=z, x=list(x_labels), y=list(y_labels), colorscale=colorscale,
        showscale=False, xgap=2, ygap=2,
        text=hover_text,
        hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>" if hover_text else None,
    ))
    fig.update_layout(xaxis=dict(side="top"))
    return _base_layout(fig, height=height or max(320, 28 * len(y_labels) + 80), title=title)


def plotly_stacked_bar(
    x_categories: Sequence, series: Dict[str, Sequence[float]], *,
    title: str = "", x_title: str = "", y_title: str = "", height: int = 380,
) -> go.Figure:
    fig = go.Figure()
    for i, (name, values) in enumerate(series.items()):
        fig.add_trace(go.Bar(x=list(x_categories), y=list(values), name=name, marker_color=category_color(i)))
    fig.update_layout(
        barmode="stack",
        xaxis=dict(type="category", title=x_title),
        yaxis=dict(title=y_title),
    )
    return _base_layout(fig, height=height, title=title)


def plotly_radar(
    categories: Sequence[str], values: Sequence[float], *, title: str = "",
    color: Optional[str] = None, max_value: float = 100, height: int = 420,
) -> go.Figure:
    categories = list(categories)
    values = list(values)
    if categories:
        categories = categories + [categories[0]]
        values = values + [values[0]]
    fig = go.Figure(go.Scatterpolar(
        r=values, theta=categories, fill="toself",
        line_color=color or CATEGORY_PALETTE[4],
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max_value])),
        showlegend=False,
    )
    return _base_layout(fig, height=height, title=title)


def plotly_gauge(
    value: float, *, title: str = "", color: Optional[str] = None,
    max_value: float = 100, height: int = 220,
) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        gauge=dict(
            axis=dict(range=[0, max_value]),
            bar=dict(color=color or CATEGORY_PALETTE[0]),
        ),
    ))
    return _base_layout(fig, height=height, title=title)


def plotly_network(
    nodes: List[dict], edges: List[dict], *, title: str = "", height: int = 460,
) -> go.Figure:
    """nodes: [{id, label, color?, size?, hover?}],
    edges: [{source, target, color?, width?, label?}]. Layout via
    networkx.spring_layout - a lightweight, dependency-free (networkx is
    already installed) way to draw a relationship graph without a JS
    graph-visualization library."""
    import networkx as nx

    g = nx.Graph()
    for n in nodes:
        g.add_node(n["id"])
    for e in edges:
        g.add_edge(e["source"], e["target"])
    pos = nx.spring_layout(g, seed=42)

    edge_traces = []
    for e in edges:
        x0, y0 = pos[e["source"]]
        x1, y1 = pos[e["target"]]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line=dict(width=e.get("width", 1.5), color=e.get("color", "#94a3b8")),
            hoverinfo="text", text=e.get("label", ""), showlegend=False,
        ))

    node_trace = go.Scatter(
        x=[pos[n["id"]][0] for n in nodes],
        y=[pos[n["id"]][1] for n in nodes],
        mode="markers+text",
        text=[n["label"] for n in nodes], textposition="top center",
        marker=dict(
            size=[n.get("size", 26) for n in nodes],
            color=[n.get("color", CATEGORY_PALETTE[0]) for n in nodes],
            line=dict(width=1, color="#ffffff"),
        ),
        hovertext=[n.get("hover", n["label"]) for n in nodes], hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    return _base_layout(fig, height=height, title=title)


def plotly_sankey(
    labels: Sequence[str], sources: Sequence[int], targets: Sequence[int],
    values: Sequence[float], *, link_colors: Optional[Sequence[str]] = None,
    node_colors: Optional[Sequence[str]] = None, title: str = "", height: int = 460,
) -> go.Figure:
    fig = go.Figure(go.Sankey(
        node=dict(
            label=list(labels), pad=14, thickness=16,
            color=list(node_colors) if node_colors else [category_color(i) for i in range(len(labels))],
        ),
        link=dict(source=list(sources), target=list(targets), value=list(values), color=link_colors),
    ))
    return _base_layout(fig, height=height, title=title)


def plotly_treemap(
    labels: Sequence[str], parents: Sequence[str], values: Sequence[float], *,
    ids: Optional[Sequence[str]] = None, title: str = "", height: int = 460,
    colors: Optional[Sequence[str]] = None,
) -> go.Figure:
    fig = go.Figure(go.Treemap(
        ids=list(ids) if ids else None,
        labels=list(labels), parents=list(parents), values=list(values),
        marker=dict(colors=list(colors)) if colors else None,
        textinfo="label+value", branchvalues="total",
    ))
    return _base_layout(fig, height=height, title=title)


def plotly_scatter_clusters(
    x: Sequence[float], y: Sequence[float], labels: Sequence[str], cluster_ids: Sequence[int],
    *, title: str = "", height: int = 460,
) -> go.Figure:
    fig = go.Figure()
    for ci in sorted(set(cluster_ids)):
        idx = [i for i, c in enumerate(cluster_ids) if c == ci]
        fig.add_trace(go.Scatter(
            x=[x[i] for i in idx], y=[y[i] for i in idx],
            mode="markers", name=f"Cluster {ci + 1}",
            marker=dict(size=14, color=category_color(ci)),
            text=[labels[i] for i in idx], hoverinfo="text",
        ))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    return _base_layout(fig, height=height, title=title)
