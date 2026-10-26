"""Live Rule Mutation & Anomaly Cockpit Page.

Demonstrates real-time non-stationary adaptation where environment kinetics silently mutate
and HEx-WM automatically detects surprise, broadens belief, and re-identifies the new rule.
"""

from __future__ import annotations
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.state_engine import DashboardStateEngine
from src.dashboard.config import PALETTE
from src.dashboard.plots import plot_surprise_gauge
from src.envs.chemical_reactor import ChemicalKineticsModel


def render_mutation_cockpit():
    """Renders the Live Rule Mutation Cockpit view."""
    DashboardStateEngine.get_chemical_state()

    st.markdown("## ⚡ Live Rule Mutation & Anomaly Cockpit")
    st.markdown(
        "*Test real-time non-stationary adaptation: Silently swap the governing kinetics mid-run "
        "and watch HEx-WM detect observation surprise, broaden its belief, and re-discover the new mechanism.*"
    )

    env = st.session_state.chem_env
    agent = st.session_state.chem_agent
    current_gt = env.ground_truth_id
    hypotheses = env.get_hypothesis_space()

    # Mutation Control Panel
    st.markdown("### 🎛️ Inject Unannounced Physical Shift")
    col_sel, col_mut = st.columns([2, 1])

    with col_sel:
        target_h = st.selectbox(
            "Select New Governing Mechanism to Inject:",
            options=[h.hypothesis_id for h in hypotheses],
            format_func=lambda h_id: f"H{h_id}: {hypotheses[h_id].name}",
            index=(current_gt + 1) % len(hypotheses)
        )

    with col_mut:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("💥 Inject Silent Mutation", type="primary", use_container_width=True):
            new_name = DashboardStateEngine.mutate_rule(target_h)
            st.toast(f"⚡ Reactor kinetics silently mutated to: {new_name}!", icon="⚠️")
            st.rerun()

    st.markdown("---")

    # Step Reaction and Watch Adaptation
    c_action1, c_action2, _ = st.columns([1, 1, 2])
    with c_action1:
        if st.button("▶️ Step Next Reaction Batch", use_container_width=True):
            exp = DashboardStateEngine.run_hex_experiment()
            st.rerun()

    with c_action2:
        if st.button("⏩ Run 3 Post-Mutation Batches", use_container_width=True):
            for _ in range(3):
                DashboardStateEngine.run_hex_experiment()
            st.rerun()

    # Status & Anomaly Banner
    surprise_hist = st.session_state.chem_history_surprise
    last_surprise = surprise_hist[-1] if surprise_hist else 0.0
    is_anomaly = last_surprise > 15.0

    if is_anomaly:
        st.error(
            f"🚨 **ANOMALY DETECTED!** Observation surprise spiked to **{last_surprise:.1f} nats** "
            f"(threshold: 15.0). Agent automatically broadened its posterior belief and resumed targeted experimental probing."
        )
    else:
        map_idx, map_prob = agent.belief_tracker.get_map_hypothesis()
        st.success(
            f"✅ **STEADY STATE:** Operating under **{hypotheses[map_idx].name}** "
            f"with confidence **{map_prob * 100:.1f}%** (Surprise: {last_surprise:.1f} nats)."
        )

    # Visualization Columns
    c_left, c_right = st.columns([1.2, 1.8])

    with c_left:
        st.plotly_chart(plot_surprise_gauge(last_surprise, threshold=15.0), use_container_width=True)

    with c_right:
        # Plot dual-axis recovery curve
        beliefs_hist = st.session_state.chem_history_beliefs
        steps = list(range(len(beliefs_hist)))

        fig = go.Figure()
        colors = [PALETTE["hex_wm"], PALETTE["accent"], PALETTE["plan2explore"], PALETTE["danger"]]

        for h_id in range(len(hypotheses)):
            probs = [b[h_id] for b in beliefs_hist]
            fig.add_trace(go.Scatter(
                x=steps,
                y=probs,
                mode="lines+markers",
                name=f"H{h_id}: {hypotheses[h_id].name[:18]}...",
                line=dict(color=colors[h_id % len(colors)], width=2.5)
            ))

        for m_step in st.session_state.chem_mutation_events:
            fig.add_vline(
                x=m_step,
                line_dash="dash",
                line_color=PALETTE["danger"],
                annotation_text="Mutation Event",
                annotation_font_color=PALETTE["danger"],
                annotation_position="top left"
            )

        fig.update_layout(
            title=dict(text="<b>Posterior Probability Trajectories P(H_k)</b>", font=dict(color=PALETTE["text_main"], size=15)),
            xaxis=dict(title="Batch Number", gridcolor="rgba(255, 255, 255, 0.08)", tickfont=dict(color=PALETTE["text_muted"])),
            yaxis=dict(title="Probability P(H)", range=[-0.05, 1.05], gridcolor="rgba(255, 255, 255, 0.08)", tickfont=dict(color=PALETTE["text_muted"])),
            legend=dict(font=dict(color=PALETTE["text_main"], size=10)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=30),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
