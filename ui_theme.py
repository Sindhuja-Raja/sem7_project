"""
Shared design-system helpers for the Streamlit presentation layer.

Pure rendering/formatting only — no analysis logic lives here, and nothing
in this module calls the LLM, RAG pipeline, or any backend agent. It exists
so every tab across the app shares one consistent visual language: one color
palette, one chip/badge style, one set of chart wrappers.

Chart backgrounds are left transparent (paper_bgcolor/plot_bgcolor =
"rgba(0,0,0,0)") so Streamlit's dark theme shows through cleanly.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Palette — consistent across bars, treemap, network nodes, chips.
# ---------------------------------------------------------------------------

CATEGORY_PALETTE = [
    "#6366f1",  # indigo
    "#0ea5e9",  # sky
    "#14b8a6",  # teal
    "#f59e0b",  # amber
    "#ef4444",  # red
    "#8b5cf6",  # violet
    "#22c55e",  # green
    "#ec4899",  # pink
]

CLASSIFICATION_COLORS = {
    "Agreement":              "#22c55e",
    "Contradiction":          "#ef4444",
    "Partial Contradiction":  "#f59e0b",
    "Different Context":      "#0ea5e9",
    "Insufficient Evidence":  "#94a3b8",
}

EVIDENCE_TYPE_COLORS = {
    "Supported by Retrieved Literature": "#22c55e",
    "General AI Best Practice":          "#64748b",
}

SOURCE_COLORS = {
    "Full PDF":         "#22c55e",
    "Abstract fallback": "#f59e0b",
    "Not Available":    "#94a3b8",
}


def category_color(index: int) -> str:
    return CATEGORY_PALETTE[index % len(CATEGORY_PALETTE)]


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

def inject_css() -> None:
    """Call once near the top of main(). Injects only classes that this
    module renders — never touches Streamlit's internal DOM selectors."""
    st.markdown(
        """
        <style>
        /* ---- chip / badge ---- */
        .rs-chip-row {
            display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0 12px 0;
        }
        .rs-chip {
            display: inline-flex; align-items: center; padding: 3px 12px;
            border-radius: 999px; font-size: 0.80rem; font-weight: 600;
            line-height: 1.6; white-space: nowrap; letter-spacing: 0.01em;
            transition: opacity 0.15s;
        }
        .rs-chip:hover { opacity: 0.85; }

        /* ---- tag cloud ---- */
        .rs-tagcloud { line-height: 2.6; padding: 8px 4px; }

        /* ---- hero / overview banner ---- */
        .rs-hero {
            background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(14,165,233,0.10) 100%);
            border: 1px solid rgba(99,102,241,0.35);
            border-radius: 14px;
            padding: 24px 28px 20px 28px;
            margin: 8px 0 20px 0;
            position: relative;
            overflow: hidden;
        }
        .rs-hero::before {
            content: "";
            position: absolute; top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, #6366f1, #0ea5e9, #14b8a6);
            border-radius: 14px 14px 0 0;
        }

        /* ---- section header ---- */
        .rs-section-header {
            border-left: 4px solid #6366f1;
            padding: 4px 0 4px 14px;
            margin: 20px 0 8px 0;
        }
        .rs-section-header h2 {
            margin: 0 0 2px 0;
            font-size: 1.25rem;
            font-weight: 700;
        }
        .rs-section-caption {
            font-size: 0.84rem;
            opacity: 0.72;
            margin: 0;
        }

        /* ---- agent status dots ---- */
        .rs-agent-row {
            display: flex; flex-wrap: wrap; gap: 4px 20px; margin: 6px 0;
        }
        .rs-agent-dot {
            display: inline-flex; align-items: center; gap: 7px;
            font-size: 0.85rem; opacity: 0.92;
        }
        .rs-agent-dot .dot {
            width: 9px; height: 9px; border-radius: 50%;
            display: inline-block; flex-shrink: 0;
        }
        .rs-agent-dot .dot.done   { background: #22c55e; box-shadow: 0 0 6px #22c55e88; }
        .rs-agent-dot .dot.pending { background: #334155; }

        /* ---- metric row ---- */
        .rs-metric-row {
            display: flex; flex-wrap: wrap; gap: 12px; margin: 10px 0;
        }
        .rs-metric-card {
            flex: 1; min-width: 100px;
            background: rgba(30,41,59,0.7);
            border: 1px solid rgba(99,102,241,0.2);
            border-radius: 10px;
            padding: 12px 16px;
            text-align: center;
        }
        .rs-metric-label {
            font-size: 0.78rem; opacity: 0.65; text-transform: uppercase;
            letter-spacing: 0.06em; margin-bottom: 4px;
        }
        .rs-metric-value {
            font-size: 1.5rem; font-weight: 700; color: #a5b4fc;
        }

        .st-key-problem-solution-summary [data-testid="stMetricValue"] {
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chips / badges
# ---------------------------------------------------------------------------

def chip(label: str, color: str) -> str:
    """Returns an HTML string for a single pill chip."""
    return (
        f'<span class="rs-chip" style="background:{color}1a;color:{color};'
        f'border:1.5px solid {color}50;">{label}</span>'
    )


def chip_row(labels: Sequence[str], color: str = CATEGORY_PALETTE[0]) -> None:
    """Renders a wrapped row of same-colour chips."""
    labels = [la for la in labels if la]
    if not labels:
        return
    html = '<div class="rs-chip-row">' + "".join(chip(la, color) for la in labels) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def mixed_chip_row(items: Sequence[Tuple[str, str]]) -> None:
    """items = [(label, color), ...] for rows where each chip has its own colour."""
    items = [(la, c) for la, c in items if la]
    if not items:
        return
    html = '<div class="rs-chip-row">' + "".join(chip(la, c) for la, c in items) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def tag_cloud(items: Sequence[Tuple[str, int]], *, color: str = CATEGORY_PALETTE[0]) -> None:
    """CSS-only tag cloud — font-size scaled by frequency count."""
    items = [(v, c) for v, c in items if v]
    if not items:
        return
    max_count = max(c for _, c in items) or 1
    spans = []
    for value, count in items:
        size = 0.82 + 1.18 * (count / max_count)
        opacity = 0.50 + 0.50 * (count / max_count)
        spans.append(
            f'<span style="font-size:{size:.2f}em;color:{color};opacity:{opacity:.2f};'
            f'font-weight:600;margin:0 10px 8px 0;display:inline-block;">{value}</span>'
        )
    st.markdown(f'<div class="rs-tagcloud">{"".join(spans)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Agent status row
# ---------------------------------------------------------------------------

def agent_status_row(statuses: Sequence[Tuple[str, bool]]) -> None:
    """A horizontal row of filled/hollow status dots with labels."""
    spans = []
    for label, done in statuses:
        dot_class = "dot done" if done else "dot pending"
        spans.append(
            f'<span class="rs-agent-dot">'
            f'<span class="{dot_class}"></span>{label}</span>'
        )
    st.markdown(
        f'<div class="rs-agent-row">{"".join(spans)}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section header — accented left-border style
# ---------------------------------------------------------------------------

def section_header(icon: str, title: str, caption: str = "") -> None:
    caption_html = f'<p class="rs-section-caption">{caption}</p>' if caption else ""
    st.markdown(
        f'<div class="rs-section-header">'
        f'<h2>{icon}&nbsp;{title}</h2>'
        f'{caption_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Metric row — custom cards instead of plain st.metric for the hero strip
# ---------------------------------------------------------------------------

def metric_row(items: List[Tuple[str, object]]) -> None:
    """Renders a row of native st.metric widgets inside equal columns."""
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def hero_metric_row(items: List[Tuple[str, object]]) -> None:
    """Custom styled metric cards for use inside the hero banner."""
    cards = "".join(
        f'<div class="rs-metric-card">'
        f'<div class="rs-metric-label">{label}</div>'
        f'<div class="rs-metric-value">{value}</div>'
        f'</div>'
        for label, value in items
    )
    st.markdown(f'<div class="rs-metric-row">{cards}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly wrappers — consistent dark-mode transparent background
# ---------------------------------------------------------------------------

_PLOTLY_FONT = dict(family="Inter, sans-serif", color="#f1f5f9")
_GRID_COLOR  = "rgba(51,65,85,0.6)"


def _base_layout(fig: go.Figure, *, height: int = 380, title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title or None, font=dict(size=15, weight="bold", color="#f1f5f9")) if title else None,
        margin=dict(l=14, r=14, t=48 if title else 14, b=14),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=_PLOTLY_FONT,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def plotly_bar_ranking(
    labels: Sequence[str], values: Sequence[float], *, title: str = "",
    color: Optional[str] = None, height: int = 380,
) -> go.Figure:
    order  = sorted(range(len(values)), key=lambda i: values[i])
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    bar_color = color or CATEGORY_PALETTE[0]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(
            color=bar_color,
            opacity=0.88,
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        text=values, textposition="outside",
        textfont=dict(color="#f1f5f9", size=12),
    ))
    fig.update_layout(
        xaxis=dict(gridcolor=_GRID_COLOR, zeroline=False, showline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", zeroline=False),
    )
    return _base_layout(fig, height=max(height, 38 * len(labels) + 70), title=title)


def plotly_donut(
    labels: Sequence[str], values: Sequence[float], *, title: str = "", height: int = 380,
) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.60,
        marker=dict(
            colors=[category_color(i) for i in range(len(labels))],
            line=dict(color="#0f172a", width=2),
        ),
        textfont=dict(color="#f1f5f9"),
    ))
    return _base_layout(fig, height=height, title=title)


def plotly_heatmap(
    z: Sequence[Sequence[float]], x_labels: Sequence[str], y_labels: Sequence[str], *,
    colorscale="RdYlGn", title: str = "", height: Optional[int] = None,
    hover_text: Optional[Sequence[Sequence[str]]] = None,
) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=z, x=list(x_labels), y=list(y_labels), colorscale=colorscale,
        showscale=False, xgap=3, ygap=3,
        text=hover_text,
        hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>" if hover_text else None,
    ))
    fig.update_layout(xaxis=dict(side="top"))
    return _base_layout(fig, height=height or max(320, 30 * len(y_labels) + 90), title=title)


def plotly_stacked_bar(
    x_categories: Sequence, series: Dict[str, Sequence[float]], *,
    title: str = "", x_title: str = "", y_title: str = "", height: int = 380,
) -> go.Figure:
    fig = go.Figure()
    for i, (name, vals) in enumerate(series.items()):
        fig.add_trace(go.Bar(
            x=list(x_categories), y=list(vals), name=name,
            marker_color=category_color(i), opacity=0.88,
        ))
    fig.update_layout(
        barmode="stack",
        xaxis=dict(type="category", title=x_title, gridcolor=_GRID_COLOR),
        yaxis=dict(title=y_title, gridcolor=_GRID_COLOR, zeroline=False),
    )
    return _base_layout(fig, height=height, title=title)


def plotly_radar(
    categories: Sequence[str], values: Sequence[float], *, title: str = "",
    color: Optional[str] = None, max_value: float = 100, height: int = 420,
) -> go.Figure:
    categories = list(categories)
    values     = list(values)
    if categories:
        categories = categories + [categories[0]]
        values     = values + [values[0]]
    c = color or CATEGORY_PALETTE[4]
    fill_color = "rgba(239, 68, 68, 0.15)" if c == "#ef4444" else f"rgba(0, 0, 0, 0.15)"
    fig = go.Figure(go.Scatterpolar(
        r=values, theta=categories, fill="toself",
        line=dict(color=c, width=2),
        fillcolor=fill_color,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max_value], gridcolor=_GRID_COLOR),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
    )
    return _base_layout(fig, height=height, title=title)


def plotly_gauge(
    value: float, *, title: str = "", color: Optional[str] = None,
    max_value: float = 100, height: int = 220,
) -> go.Figure:
    c = color or CATEGORY_PALETTE[0]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(font=dict(color=c, size=32)),
        gauge=dict(
            axis=dict(range=[0, max_value], tickcolor="#94a3b8"),
            bar=dict(color=c),
            bgcolor="rgba(30,41,59,0.5)",
            bordercolor="#334155",
        ),
    ))
    return _base_layout(fig, height=height, title=title)


def plotly_network(
    nodes: List[dict], edges: List[dict], *, title: str = "", height: int = 460,
) -> go.Figure:
    """nodes: [{id, label, color?, size?, hover?}],
    edges: [{source, target, color?, width?, label?}]."""
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
            line=dict(width=e.get("width", 1.8), color=e.get("color", "#4f46e5")),
            hoverinfo="text", text=e.get("label", ""), showlegend=False,
        ))

    node_trace = go.Scatter(
        x=[pos[n["id"]][0] for n in nodes],
        y=[pos[n["id"]][1] for n in nodes],
        mode="markers+text",
        text=[n["label"] for n in nodes], textposition="top center",
        textfont=dict(color="#f1f5f9", size=11),
        marker=dict(
            size=[n.get("size", 26) for n in nodes],
            color=[n.get("color", CATEGORY_PALETTE[0]) for n in nodes],
            line=dict(width=2, color="#0f172a"),
            opacity=0.9,
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
        arrangement="snap",
        node=dict(
            label=list(labels), pad=18, thickness=18,
            color=list(node_colors) if node_colors else [category_color(i) for i in range(len(labels))],
            line=dict(color="#0f172a", width=1),
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
        marker=dict(
            colors=list(colors) if colors else None,
            line=dict(color="#0f172a", width=1),
        ),
        textinfo="label+value", branchvalues="total",
        textfont=dict(color="#f1f5f9"),
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
            marker=dict(size=16, color=category_color(ci), opacity=0.85,
                        line=dict(color="#0f172a", width=1.5)),
            text=[labels[i] for i in idx], hoverinfo="text",
        ))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    return _base_layout(fig, height=height, title=title)
