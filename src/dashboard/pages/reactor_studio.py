"""Chemical Reactor Experiment Studio Page.

Allows interactive execution of autonomous HEx-WM experimental design
or manual knob control in an automated chemical reactor.
"""

from __future__ import annotations
import numpy as np
import streamlit as st

from src.dashboard.state_engine import DashboardStateEngine
from src.dashboard.plots import (
    plot_belief_distribution,
    plot_entropy_trajectory,
    plot_batch_yields,
    plot_surprise_gauge
)
from src.envs.chemical_reactor import ChemicalKineticsModel


def render_reactor_studio():
    """Renders the main Chemical Reactor Studio view."""
    DashboardStateEngine.get_chemical_state()

    st.markdown("## 🧪 Autonomous Chemical Reactor Studio")
    st.markdown(
        "*Real-world Self-Driving Lab benchmark: The agent controls continuous reaction knobs "
        "to discover unknown non-linear reaction rate laws with minimal batch experiments.*"
    )

    env = st.session_state.chem_env
    agent = st.session_state.chem_agent
    experiments = st.session_state.chem_history_experiments
    entropy_hist = st.session_state.chem_history_entropy
    beliefs_hist = st.session_state.chem_history_beliefs
    surprise_hist = st.session_state.chem_history_surprise

    # Status Overview Cards
    map_idx, map_prob = agent.belief_tracker.get_map_hypothesis()
    current_h = agent.belief_tracker.get_shannon_entropy()
    hypotheses = env.get_hypothesis_space()
    h_names = [f"H{h.hypothesis_id}: {h.name}" for h in hypotheses]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Top Candidate Rule</div>"
            f"<div class='metric-value' style='font-size:1.15rem; color:#10B981;'>{hypotheses[map_idx].name}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Agent Confidence</div>"
            f"<div class='metric-value'>{map_prob * 100:.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Shannon Entropy</div>"
            f"<div class='metric-value'>{current_h:.3f} <span style='font-size:0.9rem; color:#94A3B8;'>nats</span></div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Batches Executed</div>"
            f"<div class='metric-value'>{len(experiments)} / 20</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Layout: Controls on Left, Visualizations on Right
    c_left, c_right = st.columns([1.1, 1.4])

    with c_left:
        tab_auto, tab_manual = st.tabs(["🤖 HEx-WM Autonomous AI", "🎛️ Manual Chemist Control"])

        with tab_auto:
            st.markdown("#### Active Experimental Design (sBED)")
            st.markdown(
                "HEx-WM samples continuous candidate operating conditions and maximizes the "
                "**Contrastive Hypothesis Discrepancy (CHD)** between candidate ODE mechanisms."
            )

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("▶️ Run Next Optimal Experiment", use_container_width=True, type="primary"):
                    with st.spinner("Synthesizing batch in reactor..."):
                        exp = DashboardStateEngine.run_hex_experiment()
                        st.success(
                            f"Batch #{exp['batch_num']} complete: "
                            f"Product P = {exp['yields']['C_P']:.2f}M | Purity = {exp['purity']:.1f}%"
                        )
                        st.rerun()

            with col_btn2:
                if st.button("⏩ Run 5-Batch Discovery Sequence", use_container_width=True):
                    with st.spinner("Executing 5-batch sequence..."):
                        for _ in range(5):
                            DashboardStateEngine.run_hex_experiment()
                        st.rerun()

            if st.button("🔄 Reset Reactor Session", use_container_width=True):
                DashboardStateEngine.reset_chemical_state(ground_truth_id=st.session_state.chem_ground_truth_id)
                st.rerun()

        with tab_manual:
            st.markdown("#### Manually Configure Reactor Knobs")
            m_temp = st.slider("Reactor Temperature (K)", min_value=300.0, max_value=480.0, value=350.0, step=5.0)
            m_ca0 = st.slider("Substrate Feed C_A0 (mol/L)", min_value=0.5, max_value=4.0, value=2.0, step=0.1)
            m_cb0 = st.slider("Catalyst Concentration C_B0 (mol/L)", min_value=0.05, max_value=0.8, value=0.2, step=0.05)
            m_dur = st.slider("Reaction Duration (minutes)", min_value=10.0, max_value=100.0, value=40.0, step=5.0)

            if st.button("🧪 Inject Custom Experiment", use_container_width=True):
                with st.spinner("Running manual experiment in reactor..."):
                    exp = DashboardStateEngine.run_manual_experiment(m_temp, m_ca0, m_cb0, m_dur)
                    st.info(
                        f"Manual Batch #{exp['batch_num']} yield: "
                        f"Product P = {exp['yields']['C_P']:.2f}M, Waste W = {exp['yields']['C_W']:.2f}M"
                    )
                    st.rerun()

        # Observation Surprise Gauge
        last_surprise = surprise_hist[-1] if surprise_hist else 0.0
        st.plotly_chart(plot_surprise_gauge(last_surprise, threshold=15.0), use_container_width=True)

    with c_right:
        # Dynamic Belief Distribution Plot
        latest_belief = beliefs_hist[-1]
        st.plotly_chart(plot_belief_distribution(latest_belief, h_names), use_container_width=True)

        # Shannon Entropy Decay Curve
        st.plotly_chart(plot_entropy_trajectory(entropy_hist, st.session_state.chem_mutation_events), use_container_width=True)

    # Historical Experiments Table & Yields
    st.markdown("---")
    st.markdown("### 📊 Reactor Experiment Log")

    if experiments:
        st.plotly_chart(plot_batch_yields(experiments), use_container_width=True)

        log_data = []
        for e in experiments:
            log_data.append({
                "Batch": f"#{e['batch_num']}",
                "Temp (K)": f"{e['params']['T']:.1f}",
                "Substrate Ca0 (M)": f"{e['params']['C_A0']:.2f}",
                "Catalyst Cb0 (M)": f"{e['params']['C_B0']:.2f}",
                "Duration (m)": f"{e['params']['duration']:.1f}",
                "Product P (M)": f"{e['yields']['C_P']:.2f}",
                "Waste W (M)": f"{e['yields']['C_W']:.2f}",
                "Purity": f"{e['purity']:.1f}%",
                "Entropy (nats)": f"{e['entropy']:.3f}"
            })
        st.dataframe(log_data, use_container_width=True)
    else:
        st.info("No batches executed yet. Click 'Run Next Optimal Experiment' above to begin.")
