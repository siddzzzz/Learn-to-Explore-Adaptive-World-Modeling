"""Unit tests for Phase B Safe-BED components:

Verifies ReactorSafetyEvaluator, ConstrainedBayesianOptimizationAgent,
and SafetyMetricsLogger under known boundary and violation conditions.
"""

from __future__ import annotations
import numpy as np
import pytest

from src.envs.chemical_reactor.safety_evaluator import ReactorSafetyEvaluator
from src.agents.baselines.constrained_bo_agent import ConstrainedBayesianOptimizationAgent
from src.safety.safety_logger import SafetyMetricsLogger


def test_reactor_safety_evaluator():
    """Verify ReactorSafetyEvaluator correctly marks thermal runaway and toxic waste."""
    evaluator = ReactorSafetyEvaluator(max_temperature_kelvin=420.0, max_waste_molar=0.35)

    # 1. Safe condition: T=350K, W=0.05M
    safe_params = {"T": 350.0}
    safe_concs = np.array([1.0, 0.5, 2.0, 0.05])
    is_safe, margin, details = evaluator.evaluate_safety(safe_params, safe_concs)
    assert is_safe
    assert margin > 0.0
    assert details["is_thermal_safe"]
    assert details["is_toxic_safe"]

    # 2. Thermal runaway violation: T=435K
    hot_params = {"T": 435.0}
    is_safe_hot, margin_hot, details_hot = evaluator.evaluate_safety(hot_params, safe_concs)
    assert not is_safe_hot
    assert margin_hot < 0.0
    assert not details_hot["is_thermal_safe"]

    # 3. Toxic spill violation: W=0.45M
    toxic_concs = np.array([1.0, 0.5, 2.0, 0.45])
    is_safe_tox, margin_tox, details_tox = evaluator.evaluate_safety(safe_params, toxic_concs)
    assert not is_safe_tox
    assert not details_tox["is_toxic_safe"]


def test_constrained_bayesian_optimization_agent():
    """Verify ConstrainedBayesianOptimizationAgent generates actions and fits dual GPs."""
    agent = ConstrainedBayesianOptimizationAgent(action_dim=4, kappa=1.5, num_candidates=30)
    agent.reset()

    # Cold start proposals
    act1 = agent.select_action(np.zeros(4))
    assert act1.shape == (4,)
    assert np.all(act1 >= -1.0) and np.all(act1 <= 1.0)

    # Observe mock results (safe vs unsafe)
    for _ in range(4):
        a = agent.select_action(np.zeros(4))
        # Feed positive margin for safe, negative for dangerous
        agent.observe_result(a, reward=1.0, safety_margin=5.0)

    assert agent.is_fitted
    act_fitted = agent.select_action(np.zeros(4))
    assert act_fitted.shape == (4,)


def test_safety_metrics_logger():
    """Verify SafetyMetricsLogger calculates accurate statistics."""
    logger = SafetyMetricsLogger(risk_delta=0.05)
    logger.reset()

    # Step 1: Safe step
    logger.record_step(
        is_safe=True,
        worst_margin=15.0,
        margins={"is_thermal_safe": True, "is_toxic_safe": True},
        filter_approved=True,
        entropy=1.2,
        temperature=350.0,
        toxic_waste=0.05
    )

    # Step 2: Thermal violation
    logger.record_step(
        is_safe=False,
        worst_margin=-10.0,
        margins={"is_thermal_safe": False, "is_toxic_safe": True},
        filter_approved=False,
        entropy=0.5,
        temperature=430.0,
        toxic_waste=0.10
    )

    summary = logger.get_summary()
    assert summary["total_steps"] == 2
    assert summary["cumulative_violations"] == 1
    assert summary["thermal_violations"] == 1
    assert summary["toxic_violations"] == 0
    assert summary["worst_safety_margin"] == -10.0
    assert summary["filter_approval_rate_pct"] == 50.0
    assert summary["final_entropy"] == 0.5
    assert summary["max_temperature"] == 430.0
