"""Interactive Plotly visualizations for the scientific dashboard."""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import plotly.graph_objects as go

from src.dashboard.config import PALETTE


def plot_belief_distribution(weights: np.ndarray, hypothesis_names: List[str]) -> go.Figure:
    """Renders a horizontal bar chart of the agent's posterior belief over candidate rules."""
    w = np.asarray(weights, dtype=float)
    map_idx = int(np.argmax(w))

    colors = [
        PALETTE["hex_wm"] if i == map_idx and w[i] > 0.4 else PALETTE["card_bg"]
        for i in range(len(w))
    ]
    border_colors = [
        PALETTE["hex_wm"] if i == map_idx else PALETTE["text_muted"]
        for i in range(len(w))
    ]

    fig = go.Figure(go.Bar(
        x=w * 100.0,
        y=hypothesis_names,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color=border_colors, width=1.5)
        ),
        text=[f"{val * 100.0:.1f}%" for val in w],
        textposition="outside",
        textfont=dict(color=PALETTE["text_main"], size=12)
    ))

    fig.update_layout(
        title=dict(text="<b>Posterior Belief P(H | Data)</b>", font=dict(color=PALETTE["text_main"], size=15)),
        xaxis=dict(
            title="Probability (%)",
            range=[0, 115],
            gridcolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color=PALETTE["text_muted"])
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color=PALETTE["text_main"], size=11)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=40, t=40, b=30),
        height=240
    )
    return fig


def plot_entropy_trajectory(entropy_history: List[float], mutation_events: Optional[List[int]] = None) -> go.Figure:
    """Renders the Shannon entropy reduction curve over experiment batches."""
    steps = list(range(len(entropy_history)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps,
        y=entropy_history,
        mode="lines+markers",
        line=dict(color=PALETTE["accent"], width=3),
        marker=dict(size=7, color=PALETTE["accent"]),
        fill="tozeroy",
        fillcolor="rgba(6, 182, 212, 0.1)",
        name="Shannon Entropy"
    ))

    if mutation_events:
        for m_step in mutation_events:
            fig.add_vline(
                x=m_step,
                line_dash="dash",
                line_color=PALETTE["danger"],
                annotation_text="Rule Mutation",
                annotation_font_color=PALETTE["danger"],
                annotation_position="top right"
            )

    fig.update_layout(
        title=dict(text="<b>Uncertainty Decay: H(b_t) [Nats]</b>", font=dict(color=PALETTE["text_main"], size=15)),
        xaxis=dict(
            title="Experiment Batch",
            gridcolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color=PALETTE["text_muted"]),
            dtick=1
        ),
        yaxis=dict(
            title="Entropy (nats)",
            range=[-0.05, max(1.6, max(entropy_history) + 0.2)],
            gridcolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color=PALETTE["text_muted"])
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=20, t=40, b=30),
        height=240
    )
    return fig


def plot_surprise_gauge(current_surprise: float, threshold: float = 15.0) -> go.Figure:
    """Renders a real-time observation surprise indicator gauge."""
    val = float(np.clip(current_surprise, -10.0, 100.0))
    is_anomaly = current_surprise > threshold

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number=dict(suffix=" nats", font=dict(color=PALETTE["danger"] if is_anomaly else PALETTE["text_main"], size=24)),
        gauge=dict(
            axis=dict(range=[-10, 100], tickcolor=PALETTE["text_muted"]),
            bar=dict(color=PALETTE["danger"] if is_anomaly else PALETTE["hex_wm"]),
            bgcolor="rgba(255, 255, 255, 0.05)",
            threshold=dict(
                line=dict(color=PALETTE["danger"], width=3),
                thickness=0.75,
                value=threshold
            ),
            steps=[
                dict(range=[-10, threshold], color="rgba(16, 185, 129, 0.1)"),
                dict(range=[threshold, 100], color="rgba(239, 68, 68, 0.2)")
            ]
        ),
        title=dict(text="<b>Observation Surprise (-ln p)</b>", font=dict(color=PALETTE["text_main"], size=14))
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=10),
        height=190
    )
    return fig


def plot_batch_yields(experiments: List[Dict[str, Any]]) -> go.Figure:
    """Renders product vs waste yield history across physical batches."""
    if not experiments:
        fig = go.Figure()
        fig.update_layout(
            title="No experiments executed yet.",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=240
        )
        return fig

    batch_labels = [f"#{e['batch_num']}" for e in experiments]
    cP = [e["yields"]["C_P"] for e in experiments]
    cW = [e["yields"]["C_W"] for e in experiments]
    temps = [e["params"]["T"] for e in experiments]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=batch_labels,
        y=cP,
        name="Target Product P",
        marker_color=PALETTE["hex_wm"]
    ))
    fig.add_trace(go.Bar(
        x=batch_labels,
        y=cW,
        name="Toxic Waste W",
        marker_color=PALETTE["danger"]
    ))

    fig.update_layout(
        title=dict(text="<b>Yield Composition: Target P vs Waste W (M)</b>", font=dict(color=PALETTE["text_main"], size=15)),
        barmode="group",
        xaxis=dict(title="Batch Number", gridcolor="rgba(255, 255, 255, 0.08)", tickfont=dict(color=PALETTE["text_muted"])),
        yaxis=dict(title="Concentration (mol/L)", gridcolor="rgba(255, 255, 255, 0.08)", tickfont=dict(color=PALETTE["text_muted"])),
        legend=dict(font=dict(color=PALETTE["text_main"])),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=30),
        height=240
    )
    return fig


def plot_safety_barrier_gauge(
    cbf_value: float,
    threshold: float = 0.0,
    title: str = "Control Barrier Function h(s)",
) -> go.Figure:
    """Renders a gauge showing current barrier margin and safety boundary."""
    color = PALETTE["hex_wm"] if cbf_value >= threshold else PALETTE["danger"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cbf_value,
        domain=dict(x=[0, 1], y=[0, 1]),
        title=dict(text=f"<b>{title}</b>", font=dict(color=PALETTE["text_main"], size=14)),
        number=dict(font=dict(color=color, size=24), valueformat=".2f"),
        gauge=dict(
            axis=dict(range=[-1.0, 1.0], tickcolor=PALETTE["text_muted"], tickfont=dict(color=PALETTE["text_muted"])),
            bar=dict(color=color),
            steps=[
                dict(range=[-1.0, 0.0], color="rgba(239, 68, 68, 0.25)"),
                dict(range=[0.0, 1.0], color="rgba(16, 185, 129, 0.20)"),
            ],
            threshold=dict(
                line=dict(color=PALETTE["danger"], width=4),
                thickness=0.8,
                value=threshold,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text_main"]),
        margin=dict(l=20, r=20, t=40, b=20),
        height=200,
    )
    return fig


def plot_safe_action_envelope(
    nominal_actions: np.ndarray,
    safe_actions: np.ndarray,
    is_safe: np.ndarray,
    action_dims: tuple[int, int] = (0, 1),
) -> go.Figure:
    """Plots sampled exploration actions and their epistemic barrier classification."""
    dim_x, dim_y = action_dims
    nom = np.asarray(nominal_actions)
    safe = np.asarray(safe_actions)
    mask = np.asarray(is_safe, dtype=bool)

    fig = go.Figure()

    # Rejected / Unsafe candidates
    if np.any(~mask):
        fig.add_trace(go.Scatter(
            x=nom[~mask, dim_x],
            y=nom[~mask, dim_y],
            mode="markers",
            name="Epistemically Hazardous (Filtered)",
            marker=dict(color=PALETTE["danger"], symbol="x", size=8, opacity=0.7),
        ))

    # Safe Candidates
    if np.any(mask):
        fig.add_trace(go.Scatter(
            x=nom[mask, dim_x],
            y=nom[mask, dim_y],
            mode="markers",
            name="Safe Envelope Action",
            marker=dict(color=PALETTE["hex_wm"], symbol="circle", size=8, opacity=0.8),
        ))

    fig.update_layout(
        title=dict(text="<b>Safe Action Envelope & Epistemic Filter Projection</b>", font=dict(color=PALETTE["text_main"], size=14)),
        xaxis=dict(title=f"Action Dim {dim_x}", gridcolor="rgba(255, 255, 255, 0.08)", tickfont=dict(color=PALETTE["text_muted"])),
        yaxis=dict(title=f"Action Dim {dim_y}", gridcolor="rgba(255, 255, 255, 0.08)", tickfont=dict(color=PALETTE["text_muted"])),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color=PALETTE["text_main"])),
        margin=dict(l=20, r=20, t=40, b=30),
        height=260,
    )
    return fig

