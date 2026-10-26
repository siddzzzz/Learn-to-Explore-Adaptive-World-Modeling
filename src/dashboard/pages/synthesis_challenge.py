"""Zero-Shot Synthesis Challenge & Safety Guardrails Page.

Allows configuring custom customer drug synthesis orders with strict thermal limits
and executing head-to-head physical synthesis runs comparing HEx-WM, GP-BO, and Random Search.
"""

from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.state_engine import DashboardStateEngine
from src.dashboard.config import PALETTE
from src.envs.chemical_reactor import ChemicalReactorEnv
from src.agents.chemical_hex_agent import ChemicalHExAgent
from src.agents.baselines.bayesian_opt_agent import BayesianOptimizationAgent


def render_synthesis_challenge():
    """Renders the Zero-Shot Synthesis Challenge view."""
    DashboardStateEngine.get_chemical_state()

    st.markdown("## 🎯 Zero-Shot Chemical Synthesis Challenge")
    st.markdown(
        "*Real-world test: Configure a custom customer drug order with strict thermal safety limits. "
        "Each AI gets **1 single physical batch** to hit the specification without prior task training.*"
    )

    env = st.session_state.chem_env
    agent = st.session_state.chem_agent
    map_idx, map_prob = agent.belief_tracker.get_map_hypothesis()

    # Model confidence warning if not explored yet
    if map_prob < 0.6:
        st.warning(
            f"⚠️ **Scientific Discovery Incomplete:** Agent confidence in the current kinetic mechanism "
            f"is only **{map_prob * 100:.1f}%**. For best results, run 3-5 exploratory batches in the Reactor Studio first!"
        )

    # Order Configuration Form
    st.markdown("### 📋 Configure Customer Specification")
    c_tgt, c_wst, c_temp = st.columns(3)

    with c_tgt:
        req_p = st.slider("Target Product P (mol/L)", min_value=1.0, max_value=4.0, value=1.8, step=0.1)
    with c_wst:
        max_w = st.slider("Maximum Allowed Toxic Waste W (mol/L)", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    with c_temp:
        max_t = st.slider("Strict Thermal Safety Limit (K)", min_value=320.0, max_value=420.0, value=380.0, step=5.0)

    target_spec = {"min_P": req_p, "max_W": max_w, "max_T": max_t}

    if st.button("🚀 Execute 1-Shot Physical Synthesis (All 3 Systems)", type="primary", use_container_width=True):
        with st.spinner("Synthesizing physical batches in reactor across all 3 systems..."):
            # Setup fresh evaluation environments with same ground truth
            gt_id = st.session_state.chem_ground_truth_id
            eval_env = ChemicalReactorEnv(ground_truth_id=gt_id, max_experiments=1)
            eval_env.set_evaluation_mode(target_spec)

            # 1. HEx-WM Run (Discovered ODE World Model)
            agent.target_goal = target_spec
            act_hex = agent.select_action()
            obs_hex, _, _, _, info_hex = eval_env.step(act_hex)
            p_hex = info_hex["params"]

            # 2. Bayesian Optimization Run (GP-BO trained on previous data)
            bo_agent = BayesianOptimizationAgent(action_dim=4, num_candidates=200, rng=np.random.RandomState(42))
            # Fit BO on historical runs
            for exp in st.session_state.chem_history_experiments:
                r = float(exp["yields"]["C_P"] - 3.0 * exp["yields"]["C_W"])
                bo_agent.observe_experiment(exp["action"], r)

            eval_env_bo = ChemicalReactorEnv(ground_truth_id=gt_id, max_experiments=1)
            eval_env_bo.set_evaluation_mode(target_spec)
            act_bo = bo_agent.select_action(np.zeros(4))
            obs_bo, _, _, _, info_bo = eval_env_bo.step(act_bo)
            p_bo = info_bo["params"]

            # 3. Random Search Run
            eval_env_rand = ChemicalReactorEnv(ground_truth_id=gt_id, max_experiments=1)
            eval_env_rand.set_evaluation_mode(target_spec)
            act_rand = np.random.uniform(-1.0, 1.0, 4).astype(np.float32)
            obs_rand, _, _, _, info_rand = eval_env_rand.step(act_rand)
            p_rand = info_rand["params"]

            st.session_state.synthesis_eval_results = {
                "hex": {"params": p_hex, "yields": obs_hex},
                "bo": {"params": p_bo, "yields": obs_bo},
                "rand": {"params": p_rand, "yields": obs_rand},
                "spec": target_spec
            }

    # Render Results if Available
    if "synthesis_eval_results" in st.session_state:
        res = st.session_state.synthesis_eval_results
        spec = res["spec"]

        st.markdown("---")
        st.markdown("### 🏆 Synthesis Results (Single Physical Batch)")

        cols = st.columns(3)
        systems = [
            ("HEx-WM (Ours)", res["hex"], PALETTE["hex_wm"]),
            ("Bayesian Opt (GP-BO)", res["bo"], PALETTE["bayesian_opt"]),
            ("Random Search", res["rand"], PALETTE["random"])
        ]

        for i, (sys_name, data, col_color) in enumerate(systems):
            p = data["params"]
            y = data["yields"]
            cP = float(y[2])
            cW = float(y[3])
            is_safe = p["T"] <= spec["max_T"]
            purity = (cP / max(1e-4, cP + cW)) * 100.0
            is_success = is_safe and (cP >= spec["min_P"]) and (cW <= spec["max_W"])

            with cols[i]:
                badge_class = "badge-success" if is_success else "badge-danger"
                status_text = "SUCCESS" if is_success else ("THERMAL VIOLATION" if not is_safe else "POOR PURITY")
                
                st.markdown(
                    f"<div class='metric-card' style='border-left: 4px solid {col_color};'>"
                    f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                    f"<div class='metric-title' style='color:{col_color}; font-size:1rem;'>{sys_name}</div>"
                    f"<span class='badge {badge_class}'>{status_text}</span>"
                    f"</div>"
                    f"<div style='margin-top:12px; font-size:0.9rem;'>"
                    f"<b>Temperature:</b> {p['T']:.1f} K {'✅' if is_safe else '❌ EXCEEDED'}<br>"
                    f"<b>Target Product P:</b> <span style='color:#34D399; font-weight:700;'>{cP:.2f} M</span><br>"
                    f"<b>Toxic Waste W:</b> <span style='color:#F87171; font-weight:700;'>{cW:.2f} M</span><br>"
                    f"<b>Overall Purity:</b> {purity:.1f}%"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # Plotly Bar Chart Comparison
        sys_labels = ["HEx-WM (Ours)", "Bayesian Opt (GP-BO)", "Random Search"]
        p_vals = [res["hex"]["yields"][2], res["bo"]["yields"][2], res["rand"]["yields"][2]]
        w_vals = [res["hex"]["yields"][3], res["bo"]["yields"][3], res["rand"]["yields"][3]]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=sys_labels, y=p_vals, name="Target Product P", marker_color=PALETTE["hex_wm"]))
        fig.add_trace(go.Bar(x=sys_labels, y=w_vals, name="Toxic Waste W", marker_color=PALETTE["danger"]))
        fig.add_hline(y=spec["min_P"], line_dash="dot", line_color="#34D399", annotation_text="Target Minimum P")
        fig.add_hline(y=spec["max_W"], line_dash="dot", line_color="#F87171", annotation_text="Max Allowed W")

        fig.update_layout(
            title=dict(text="<b>Yield & Byproduct Comparison Across Methods</b>", font=dict(color=PALETTE["text_main"], size=15)),
            barmode="group",
            yaxis=dict(title="Concentration (mol/L)", gridcolor="rgba(255, 255, 255, 0.08)", tickfont=dict(color=PALETTE["text_muted"])),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
