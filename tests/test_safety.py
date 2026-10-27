"""Unit tests for Safety-Constrained Bayesian Experimental Design (Safe-HEx-WM).

Tests Control Barrier Functions, Epistemic Safety Filters, and Safe-HEx-WM action selection
without executing long training loops.
"""

import numpy as np
import pytest

from src.safety.barrier_function import (
    ControlBarrierFunction,
    create_box_barrier,
    create_thermal_barrier,
    create_toxicity_barrier
)
from src.safety.epistemic_safety_filter import EpistemicSafetyFilter
from src.safety.safe_hex_agent import SafeHExWMAgent
from src.envs.chemical_reactor import ChemicalKineticsModel


def test_control_barrier_function_evaluation():
    """Verify Control Barrier Functions evaluate safe/unsafe states correctly."""
    # Thermal barrier: T < 400K -> h(s) = 400 - T
    barrier = create_thermal_barrier(max_temperature_kelvin=400.0, temp_index=0)

    safe_state = np.array([350.0, 1.0, 0.2, 0.05])
    unsafe_state = np.array([425.0, 1.0, 0.2, 0.05])

    assert barrier.evaluate(safe_state) == 50.0
    assert barrier.is_safe(safe_state)

    assert barrier.evaluate(unsafe_state) == -25.0
    assert not barrier.is_safe(unsafe_state)


def test_box_and_toxicity_barriers():
    """Verify box boundaries and toxic byproduct constraints."""
    tox_barrier = create_toxicity_barrier(max_waste_molar=0.3, waste_index=3)

    clean_state = np.array([350.0, 1.0, 1.5, 0.05])
    toxic_state = np.array([350.0, 1.0, 0.2, 0.45])

    assert tox_barrier.is_safe(clean_state)
    assert not tox_barrier.is_safe(toxic_state)


def test_epistemic_safety_filter_conservative_lcb():
    """Verify that epistemic uncertainty across models triggers safety filtering."""
    barrier = create_thermal_barrier(max_temperature_kelvin=400.0, temp_index=0)
    safety_filter = EpistemicSafetyFilter(
        barriers=[barrier],
        risk_tolerance_delta=0.05,
        default_alpha=0.5
    )

    current_state = np.array([350.0, 1.0, 0.0, 0.0])

    # Case A: Models agree on a safe outcome
    safe_predictions = np.array([
        [360.0, 0.5, 0.5, 0.01],
        [362.0, 0.5, 0.5, 0.01]
    ])
    is_safe_a, margin_a, _ = safety_filter.evaluate_action_safety(current_state, safe_predictions)
    assert is_safe_a
    assert margin_a > 0.0

    # Case B: High epistemic divergence where one model predicts thermal runaway
    divergent_predictions = np.array([
        [360.0, 0.5, 0.5, 0.01],
        [430.0, 0.1, 0.0, 0.90]  # Thermal explosion
    ])
    is_safe_b, margin_b, details = safety_filter.evaluate_action_safety(current_state, divergent_predictions)
    # LCB must penalize the variance and reject the action
    assert not is_safe_b
    assert margin_b < 0.0


def test_safe_hex_agent_action_selection():
    """Verify SafeHExWMAgent selects safe actions and tracks margins."""
    models = [ChemicalKineticsModel(0), ChemicalKineticsModel(1)]
    bounds = {
        "T": (300.0, 480.0),
        "C_A0": (0.5, 4.0),
        "C_B0": (0.05, 0.8),
        "duration": (10.0, 100.0)
    }

    def denorm(act):
        t = float(bounds["T"][0] + 0.5 * (act[0] + 1.0) * (bounds["T"][1] - bounds["T"][0]))
        ca = float(bounds["C_A0"][0] + 0.5 * (act[1] + 1.0) * (bounds["C_A0"][1] - bounds["C_A0"][0]))
        cb = float(bounds["C_B0"][0] + 0.5 * (act[2] + 1.0) * (bounds["C_B0"][1] - bounds["C_B0"][0]))
        dur = float(bounds["duration"][0] + 0.5 * (act[3] + 1.0) * (bounds["duration"][1] - bounds["duration"][0]))
        return t, ca, cb, dur

    barrier = create_thermal_barrier(max_temperature_kelvin=430.0, temp_index=0)
    safety_filter = EpistemicSafetyFilter(barriers=[barrier], risk_tolerance_delta=0.05)

    agent = SafeHExWMAgent(
        action_dim=4,
        hypothesis_models=models,
        safety_filter=safety_filter,
        denormalize_action_fn=denorm,
        bounds=bounds,
        num_candidates=30
    )

    current_state = np.array([320.0, 2.0, 0.4, 0.0])
    action, is_provably_safe, margin = agent.select_action(current_state)

    assert action.shape == (4,)
    assert len(agent.history_actions) == 1
    assert len(agent.history_safety_margins) == 1
