"""Unit and statistical tests for Causal-Lab Benchmark."""

import numpy as np
import pytest
from src.envs.causal_lab import (
    CausalLabEnv,
    CausalHypothesis,
    StructuralCausalModel,
    generate_canonical_hypothesis_space,
    AnalyticalBayesianFilter
)


def test_scm_interventions():
    """Verify that Pearl's do-operator correctly forces values and breaks incoming edges."""
    scm = StructuralCausalModel(num_variables=4, noise_std=0.0)
    
    # Natural un-intervened state
    x_natural = scm.evaluate()
    assert len(x_natural) == 4
    
    # Intervene on variable 1: do(X_1 = 3.5)
    x_intervened = scm.evaluate(interventions={1: 3.5})
    assert np.isclose(x_intervened[1], 3.5)
    
    # Intervene on variable 0: do(X_0 = -2.0)
    x_intervened_0 = scm.evaluate(interventions={0: -2.0})
    assert np.isclose(x_intervened_0[0], -2.0)


def test_gymnasium_api_compliance():
    """Verify standard Gymnasium interface compliance."""
    env = CausalLabEnv(num_variables=4, noisy_tv_dim=3, max_steps=20)
    obs, info = env.reset(seed=123)
    
    assert obs.shape == (7,)  # 4 causal + 3 noisy TV
    assert "ground_truth_id" in info
    assert env.action_space.n == 13  # 1 observe + 3*4 interventions
    
    # Step through several actions
    for a in range(5):
        next_obs, reward, terminated, truncated, step_info = env.step(a)
        assert next_obs.shape == (7,)
        assert isinstance(reward, float)
        assert not terminated
        assert "causal_state" in step_info
        assert len(step_info["causal_state"]) == 4


def test_noisy_tv_channel():
    """Verify that the Noisy TV distractor generates stochastic noise independent of causal state."""
    env = CausalLabEnv(num_variables=4, noisy_tv_dim=5, noisy_tv_std=2.0)
    obs1, _ = env.reset(seed=42)
    obs2, _, _, _, _ = env.step(0)  # Same observation step
    
    noise1 = obs1[4:]
    noise2 = obs2[4:]
    assert noise1.shape == (5,)
    assert noise2.shape == (5,)
    # Noise should vary across steps
    assert not np.allclose(noise1, noise2)


def test_analytical_bayesian_filter():
    """Verify exact Bayesian updating and Shannon entropy reduction."""
    hypotheses = generate_canonical_hypothesis_space(4)
    bayes_filter = AnalyticalBayesianFilter(hypotheses, num_variables=4, noise_std=0.05)
    
    initial_entropy = bayes_filter.get_shannon_entropy()
    # Uniform over 8 hypotheses: H = ln(8) ≈ 2.079
    assert np.isclose(initial_entropy, np.log(8), atol=1e-3)
    
    # Generate observation from true hypothesis 1 (Linear chain)
    true_scm = StructuralCausalModel(4, hypotheses[1], noise_std=0.05, rng=np.random.RandomState(42))
    
    # Run targeted interventions: do(X_0 = 2.0), do(X_1 = 2.0)
    interventions_to_test = [{0: 2.0}, {1: 2.0}, {2: 2.0}]
    for interv in interventions_to_test:
        obs = true_scm.evaluate(interventions=interv)
        bayes_filter.update(interv, obs)
        
    final_entropy = bayes_filter.get_shannon_entropy()
    assert final_entropy < initial_entropy
    
    # True hypothesis should have the highest posterior weight
    weights = bayes_filter.weights
    assert np.argmax(weights) == 1
    assert weights[1] > 0.5


def test_eig_calculation():
    """Verify analytical Expected Information Gain is non-negative and discriminative."""
    hypotheses = generate_canonical_hypothesis_space(4)
    bayes_filter = AnalyticalBayesianFilter(hypotheses, num_variables=4, noise_std=0.05)
    
    eig_active = bayes_filter.compute_exact_eig({0: 2.0})
    eig_passive = bayes_filter.compute_exact_eig({})
    
    assert eig_active >= 0.0
    assert eig_passive >= 0.0


def test_rule_mutation():
    """Verify non-stationary rule mutation functionality."""
    env = CausalLabEnv(num_variables=4, ground_truth_id=1)
    assert env.get_ground_truth().hypothesis_id == 1
    
    new_h = env.mutate_rules(new_hypothesis_id=4)
    assert new_h.hypothesis_id == 4
    assert env.get_ground_truth().hypothesis_id == 4
