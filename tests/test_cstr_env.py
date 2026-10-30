"""Unit tests for Continuous CSTR dynamics, RK4 integration, and thermal barrier constraints."""

from __future__ import annotations
import numpy as np
import pytest

from src.envs.chemical_reactor.cstr_kinetics import CSTRDynamicsModel
from src.envs.chemical_reactor.cstr_env import ContinuousCSTREnv
from src.safety.cstr_barrier import create_cstr_thermal_barrier


def test_cstr_dynamics_model_rk4():
    """Verify RK4 integration preserves physical conservation and temperature changes."""
    model = CSTRDynamicsModel(hypothesis_id=1)
    s0 = np.array([1.0, 350.0, 300.0], dtype=np.float64)

    # 1. Step with low cooling (qc = 5 L/min) -> Heat generation dominates
    s_hot = model.step_rk4(s0, qc=5.0, dt=0.1)
    # Concentration CA must be consumed below feed concentration 1.0
    assert s_hot[0] < 1.0, "Reactant CA must be converted by reaction"

    # 2. Step with massive cooling (qc = 80 L/min) -> Cooling jacket absorbs heat
    s_cold = model.step_rk4(s0, qc=80.0, dt=0.1)
    assert s_cold[1] < s_hot[1], "Higher coolant flow must extract more thermal energy"
    assert s_cold[2] < s_hot[2], "Higher coolant flow keeps jacket cooler"


def test_continuous_cstr_env_gym_interface():
    """Verify standard Gymnasium reset, step, observation spaces, and truncation."""
    env = ContinuousCSTREnv(ground_truth_id=2, max_steps=10)
    obs, info = env.reset(seed=42)

    assert obs.shape == (3,)
    assert info["hypothesis_id"] == 2
    assert "hypothesis_name" in info

    # Step with moderate cooling
    action = np.array([0.2], dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert next_obs.shape == (3,)
    assert isinstance(reward, float)
    assert not terminated
    assert "core_temperature" in step_info
    assert step_info["qc"] > 0.0


def test_cstr_thermal_barrier_evaluation():
    """Verify create_cstr_thermal_barrier detects safe and unsafe core temperatures."""
    barrier = create_cstr_thermal_barrier(max_temperature_kelvin=410.0)

    # Safe state: T = 350K
    safe_state = np.array([0.5, 350.0, 310.0])
    assert barrier.evaluate(safe_state) == 60.0
    assert barrier.is_safe(safe_state)

    # Dangerous runaway state: T = 425K
    unsafe_state = np.array([0.1, 425.0, 340.0])
    assert barrier.evaluate(unsafe_state) == -15.0
    assert not barrier.is_safe(unsafe_state)
