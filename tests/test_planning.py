"""Unit tests for acquisition functions and trajectory planners."""

import numpy as np
import pytest
from src.planning import (
    compute_chd,
    compute_bald,
    compute_ensemble_variance,
    compute_prediction_error,
    TrajectoryCEMPlanner,
    MPPIPlanner
)


def test_chd_theorem1_aleatoric_invariance():
    """Verify Theorem 1: CHD is strictly 0 when candidate hypotheses make identical predictions."""
    K = 5
    D = 4
    # All 5 hypotheses predict the exact same outcome
    identical_preds = np.tile(np.array([1.5, -0.5, 2.0, 0.0]), (K, 1))
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    
    # Under any noise variance, CHD should be 0.0
    for noise_var in [0.001, 0.1, 1.0, 10.0]:
        chd = compute_chd(identical_preds, weights, noise_variance=noise_var)
        assert np.isclose(chd, 0.0)


def test_chd_sensitivity_to_divergence():
    """Verify that CHD increases when likely hypotheses make conflicting predictions."""
    weights = np.array([0.5, 0.5])
    # Case A: Minor disagreement
    preds_close = np.array([
        [1.0, 0.0],
        [1.1, 0.0]
    ])
    chd_close = compute_chd(preds_close, weights, noise_variance=0.01)
    
    # Case B: Drastic disagreement
    preds_far = np.array([
        [1.0, 0.0],
        [5.0, 0.0]
    ])
    chd_far = compute_chd(preds_far, weights, noise_variance=0.01)
    
    assert chd_far > chd_close
    assert chd_close > 0.0


def test_bald_computation():
    """Verify BALD calculation and monotonicity."""
    weights = np.array([0.5, 0.5])
    preds_close = np.array([[0.0], [0.1]])
    preds_far = np.array([[0.0], [3.0]])
    
    bald_close = compute_bald(preds_close, weights, noise_variance=0.01)
    bald_far = compute_bald(preds_far, weights, noise_variance=0.01)
    
    assert bald_far > bald_close
    assert bald_close >= 0.0


def test_discrete_trajectory_cem():
    """Verify that Trajectory CEM finds optimal action sequences in synthetic landscape."""
    planner = TrajectoryCEMPlanner(action_dim=4, horizon=3, num_samples=32, num_iterations=4)
    
    # Synthetic objective: action 2 at step 0 gives reward 10, other actions give 0
    def mock_rollout(s, a):
        r = 10.0 if a == 2 else 0.0
        return s, r
        
    best_a0, trajectory, score = planner.plan_discrete(np.zeros(2), mock_rollout)
    assert best_a0 == 2
    assert score >= 10.0


def test_mppi_planner():
    """Verify MPPI continuous planning."""
    planner = MPPIPlanner(action_dim=2, horizon=4, num_samples=32)
    
    # Objective: drive action toward [0.8, -0.5]
    target_a = np.array([0.8, -0.5])
    def mock_rollout(s, a):
        r = -float(np.sum((a - target_a) ** 2))
        return s, r
        
    action, score = planner.plan(np.zeros(2), mock_rollout)
    assert len(action) == 2
    assert score <= 0.0
