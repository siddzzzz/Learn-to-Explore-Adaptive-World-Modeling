"""Causal Lab DAG Visualizer and Interventional Explorer Page.

Interactive exploration of Benchmark A: Pearl's Structural Causal Models,
interventional do-calculus do(X_i = v), and the Noisy TV aleatoric distractor channel.
"""

from __future__ import annotations
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.config import PALETTE
from src.envs.causal_lab import (
    CausalLabEnv,
    CausalHypothesis,
    StructuralCausalModel,
    generate_canonical_hypothesis_space
)


def render_causal_studio():
    """Renders the Causal Lab DAG Explorer view."""
    st.markdown("## 🕸️ Causal Lab & Interventional DAG Explorer")
    st.markdown(
        "*Benchmark A: Pearl's Structural Causal Models (Pearl, 2009). "
        "Test atomic experimental interventions $\\text{do}(X_i = v)$ and observe causal propagation.*"
    )

    hypotheses = generate_canonical_hypothesis_space(num_variables=4)

    # Initialize causal lab session state
    if "causal_env" not in st.session_state:
        st.session_state.causal_env = CausalLabEnv(num_variables=4, ground_truth_id=1, noisy_tv_dim=3)
        st.session_state.causal_obs, _ = st.session_state.causal_env.reset(seed=42)

    c_sel, c_interv = st.columns([1.2, 1.8])

    with c_sel:
        st.markdown("### 🔍 Select Causal Hypothesis")
        h_idx = st.selectbox(
            "Candidate Structural Mechanism:",
            options=range(len(hypotheses)),
            format_func=lambda i: f"H{i}: {hypotheses[i].name}"
        )
        selected_h = hypotheses[h_idx]
        st.info(f"**Mechanism:** {selected_h.description}")

        # Render DAG Network Visualization using Plotly
        adj = selected_h.adjacency_matrix
        G = nx.DiGraph()
        for i in range(4):
            G.add_node(f"X_{i}")
        for i in range(4):
            for j in range(4):
                if adj[i, j] == 1:
                    G.add_edge(f"X_{i}", f"X_{j}")

        # Fixed layout for consistency
        pos = {
            "X_0": (0.1, 0.5),
            "X_1": (0.4, 0.8),
            "X_2": (0.4, 0.2),
            "X_3": (0.8, 0.5)
        }

        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2.5, color=PALETTE["accent"]),
            hoverinfo="none",
            mode="lines"
        )

        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=list(G.nodes()),
            textposition="top center",
            textfont=dict(color=PALETTE["text_main"], size=13, family="JetBrains Mono"),
            marker=dict(size=24, color=PALETTE["card_bg"], line=dict(color=PALETTE["hex_wm"], width=2.5))
        )

        fig_dag = go.Figure(data=[edge_trace, node_trace])
        fig_dag.update_layout(
            title=dict(text=f"<b>Causal DAG: {selected_h.name}</b>", font=dict(color=PALETTE["text_main"], size=14)),
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=35, b=10),
            height=260
        )
        st.plotly_chart(fig_dag, use_container_width=True)

    with c_interv:
        st.markdown("### ⚡ Execute Pearl's do-operator")
        st.markdown(
            "An intervention $\\text{do}(X_i = v)$ **severs all incoming edges** into $X_i$, "
            "forcing its value while allowing causal effects to propagate downstream."
        )

        ci1, ci2 = st.columns(2)
        with ci1:
            target_var = st.selectbox(
                "Intervention Target (do(X_i)):",
                options=[-1, 0, 1, 2, 3],
                format_func=lambda v: "No Intervention (Observe)" if v == -1 else f"do(X_{v})"
            )
        with ci2:
            stimulus_val = st.slider("Forced Stimulus Value (v)", min_value=-3.0, max_value=3.0, value=2.0, step=0.5)

        if st.button("🔬 Trigger Intervention in Lab", type="primary", use_container_width=True):
            interv_dict = {} if target_var == -1 else {target_var: stimulus_val}
            env = st.session_state.causal_env
            # Map to discrete action
            if target_var == -1:
                action = 0
            else:
                stim_type = 0 if stimulus_val > 0 else (1 if stimulus_val < 0 else 2)
                action = 1 + target_var + stim_type * 4

            obs, _, _, _, info = env.step(action)
            st.session_state.causal_obs = obs
            st.session_state.last_causal_info = info
            st.rerun()

        # Display State Readings
        obs = st.session_state.causal_obs
        causal_vals = obs[:4]
        noisy_tv_vals = obs[4:]

        st.markdown("#### Real-Time Sensor Readings")
        c_var0, c_var1, c_var2, c_var3 = st.columns(4)
        for i, col in enumerate([c_var0, c_var1, c_var2, c_var3]):
            with col:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-title'>Variable X_{i}</div>"
                    f"<div class='metric-value' style='font-size:1.3rem; color:#34D399;'>{causal_vals[i]:.2f}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # Noisy TV Channel Monitor
        st.markdown("---")
        st.markdown("#### 📺 Aleatoric Distractor: The Noisy TV Channel")
        st.markdown(
            "*Theorem 1 states that Contrastive Hypothesis Discrepancy (CHD) is provably invariant "
            "to these unmodelable noise channels, whereas ICM Curiosity and Plan2Explore get trapped exploring them.*"
        )

        fig_noise = go.Figure()
        fig_noise.add_trace(go.Bar(
            x=[f"Channel #{i+1}" for i in range(len(noisy_tv_vals))],
            y=noisy_tv_vals,
            marker_color=PALETTE["danger"],
            text=[f"{v:.2f}" for v in noisy_tv_vals],
            textposition="auto"
        ))
        fig_noise.update_layout(
            title=dict(text="<b>White Noise Channels Y_noise ~ N(0, 1)</b>", font=dict(color=PALETTE["text_main"], size=13)),
            yaxis=dict(title="Noise Amplitude", range=[-3.5, 3.5], gridcolor="rgba(255, 255, 255, 0.08)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=200,
            margin=dict(l=10, r=10, t=35, b=20)
        )
        st.plotly_chart(fig_noise, use_container_width=True)
