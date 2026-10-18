"""Unit tests for Bayesian Belief Tracker and Deep Ensemble World Model."""

import numpy as np
import pytest
import torch
from src.models import (
    HypothesisState,
    BayesianBeliefTracker,
    EnsembleWorldModel
)


def test_belief_tracker_initialization():
    """Verify uniform prior and initial Shannon entropy."""
    K = 8
    tracker = BayesianBeliefTracker(num_hypotheses=K)
    
    assert len(tracker.weights) == K
    assert np.allclose(tracker.weights, 1.0 / K)
    # Entropy of uniform over 8 is ln(8) ≈ 2.0794
    assert np.isclose(tracker.get_shannon_entropy(), np.log(K), atol=1e-4)
    assert np.isclose(tracker.get_normalized_entropy(), 1.0, atol=1e-4)


def test_belief_tracker_update():
    """Verify Bayesian update concentrates mass on favored hypothesis."""
    K = 4
    tracker = BayesianBeliefTracker(num_hypotheses=K)
    
    # Hypothesis 2 has much higher log-likelihood
    log_liks = np.array([-10.0, -8.0, -1.0, -12.0])
    weights, surprise, is_anomaly = tracker.update_from_log_likelihoods(log_liks)
    
    # Hypothesis 2 should dominate
    map_idx, map_prob = tracker.get_map_hypothesis()
    assert map_idx == 2
    assert map_prob > 0.95
    assert tracker.get_shannon_entropy() < np.log(K)


def test_belief_anomaly_detection_and_broadening():
    """Verify surprise calculation and belief broadening upon anomaly."""
    K = 4
    tracker = BayesianBeliefTracker(num_hypotheses=K, surprise_threshold=10.0)
    
    # Highly improbable observation under all models
    catastrophic_log_liks = np.array([-25.0, -30.0, -28.0, -26.0])
    _, surprise, is_anomaly = tracker.update_from_log_likelihoods(catastrophic_log_liks)
    
    assert is_anomaly
    assert surprise > 10.0
    
    # Broaden belief
    broadened_w = tracker.broaden_belief(mixing_factor=0.5)
    assert len(broadened_w) == K
    # Should have higher entropy than before broadening
    assert tracker.get_shannon_entropy() > 0.5


def test_ensemble_world_model():
    """Verify forward prediction, disagreement metric, and training pass of Deep Ensemble."""
    state_dim = 4
    action_dim = 2
    num_models = 3
    ensemble = EnsembleWorldModel(state_dim=state_dim, action_dim=action_dim, num_models=num_models)
    
    # Test batch prediction
    s = np.random.randn(5, state_dim).astype(np.float32)
    a = np.random.randn(5, action_dim).astype(np.float32)
    preds = ensemble.predict_all(s, a)
    assert preds.shape == (num_models, 5, state_dim)
    
    # Test single state disagreement
    s_single = np.random.randn(state_dim).astype(np.float32)
    a_single = np.random.randn(action_dim).astype(np.float32)
    disagreement = ensemble.compute_ensemble_disagreement(s_single, a_single)
    assert isinstance(disagreement, float)
    assert disagreement >= 0.0
    
    # Test training step
    ns = s + 0.1 * np.random.randn(5, state_dim).astype(np.float32)
    metrics = ensemble.train_step(s, a, ns, batch_size=4)
    assert "mean_loss" in metrics
    assert metrics["mean_loss"] >= 0.0
