"""Demonstration comparing unconstrained vs safe exploration (Safe-HEx-WM).

Demonstrates Theorem 2 in action on the Batch Chemical Reactor:
An agent exploring an unfamiliar environment with physical safety bounds
(e.g., maximum reactor core temperature or bounded actuator limits)
utilizes Cantelli's Lower Confidence Bound on the candidate hypothesis ensemble
to proactively filter hazardous exploratory interventions.
"""

from __future__ import annotations
import numpy as np

from src.envs.chemical_reactor import ChemicalKineticsModel
from src.safety.barrier_function import (
    create_thermal_barrier,
    create_toxicity_barrier,
)
from src.safety.epistemic_safety_filter import EpistemicSafetyFilter
from src.safety.safe_hex_agent import SafeHExWMAgent


def run_demo() -> None:
    print("=" * 75)
    print("  Safe-HEx-WM: Safety-Constrained Bayesian Experimental Design Demo")
    print("=" * 75)

    # 1. Physics boundaries: Thermal explosion limit (420 K) and Toxicity limit (0.35 M)
    thermal_barrier = create_thermal_barrier(max_temperature_kelvin=420.0, temp_index=0)
    toxic_barrier = create_toxicity_barrier(max_waste_molar=0.35, waste_index=3)
    
    # 2. Epistemic Safety Filter with Cantelli Bound (delta=0.05 => >=95% confidence)
    safety_filter = EpistemicSafetyFilter(
        barriers=[thermal_barrier, toxic_barrier],
        risk_tolerance_delta=0.05,
        default_alpha=0.5,
    )

    
    # 3. Ensemble of Candidate Reaction Mechanisms
    models = [
        ChemicalKineticsModel(0),  # Endothermic Arrhenius
        ChemicalKineticsModel(1),  # Highly Exothermic Secondary Path
    ]
    
    bounds = {
        "T": (300.0, 480.0),
        "C_A0": (0.5, 4.0),
        "C_B0": (0.05, 0.8),
        "duration": (10.0, 100.0)
    }

    def denorm(act: np.ndarray) -> tuple[float, float, float, float]:
        t = float(bounds["T"][0] + 0.5 * (act[0] + 1.0) * (bounds["T"][1] - bounds["T"][0]))
        ca = float(bounds["C_A0"][0] + 0.5 * (act[1] + 1.0) * (bounds["C_A0"][1] - bounds["C_A0"][0]))
        cb = float(bounds["C_B0"][0] + 0.5 * (act[2] + 1.0) * (bounds["C_B0"][1] - bounds["C_B0"][0]))
        dur = float(bounds["duration"][0] + 0.5 * (act[3] + 1.0) * (bounds["duration"][1] - bounds["duration"][0]))
        return t, ca, cb, dur

    agent = SafeHExWMAgent(
        action_dim=4,
        hypothesis_models=models,
        safety_filter=safety_filter,
        denormalize_action_fn=denorm,
        bounds=bounds,
        num_candidates=30,
        planning_horizon=3,
    )


    current_state = np.array([330.0, 2.0, 0.4, 0.0])  # T=330K, CA=2M, CB=0.4M, CW=0.0M
    print(f"\n[Initial Operating State]")
    print(f"  Reactor Core Temperature: {current_state[0]:.1f} K (Limit: 420.0 K)")
    print(f"  Toxic Waste Concentration: {current_state[3]:.2f} M (Limit: 0.35 M)")
    print(f"  Risk Tolerance delta: {safety_filter.delta} (Cantelli Confidence: >= {100*(1-safety_filter.delta):.1f}%)")

    # Evaluate candidate predictions: Safe vs Hazardous
    safe_predictions = np.array([
        [360.0, 0.8, 0.5, 0.05],
        [375.0, 0.7, 0.4, 0.08],
    ])
    hazardous_predictions = np.array([
        [390.0, 0.5, 0.3, 0.10],
        [445.0, 0.1, 0.0, 0.45],  # Thermal runaway beyond 420K and toxic spill > 0.35M
    ])

    is_safe_1, margin_1, _ = safety_filter.evaluate_action_safety(current_state, safe_predictions)
    is_safe_2, margin_2, _ = safety_filter.evaluate_action_safety(current_state, hazardous_predictions)

    print(f"\n[Epistemic Barrier Filter Evaluation]")
    print(f"  Candidate Intervention A (Safe):      Passed = {is_safe_1}, LCB Safety Margin = {margin_1:.2f}")
    print(f"  Candidate Intervention B (Hazardous): Passed = {is_safe_2}, LCB Safety Margin = {margin_2:.2f} (Rejected!)")

    # Safe Action Selection via Safe-HEx-WM CEM Planner
    print(f"\n[Optimizing Next Experiment via Safe-HEx-WM]")
    action, provably_safe, safety_margin = agent.select_action(current_state)
    t_opt, ca_opt, cb_opt, dur_opt = denorm(action)
    print(f"  Certified Action: {action.round(3).tolist()}")
    print(f"  Physical Parameters: T = {t_opt:.1f} K, C_A0 = {ca_opt:.2f} M, C_B0 = {cb_opt:.2f} M, time = {dur_opt:.1f} s")
    print(f"  Provably Safe (Theorem 2): {provably_safe} with LCB Margin: {safety_margin:.2f}")
    print("=" * 75)


if __name__ == "__main__":
    run_demo()
