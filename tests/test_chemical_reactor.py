"""Unit tests for Chemical Reactor Environment and Agents."""

import numpy as np
import pytest
from src.envs.chemical_reactor import ChemicalReactorEnv, ChemicalKineticsModel
from src.agents.chemical_hex_agent import ChemicalHExAgent
from src.agents.baselines.bayesian_opt_agent import BayesianOptimizationAgent


def test_ode_kinetics_simulation():
    """Verify continuous ODE solver runs smoothly and outputs non-negative concentrations."""
    model = ChemicalKineticsModel(hypothesis_id=0)
    # T = 350K, C_A0 = 2.0M, C_B0 = 0.2M, duration = 30min
    concs = model.simulate(T_kelvin=350.0, C_A0=2.0, C_B0=0.2, duration_min=30.0)
    
    assert len(concs) == 4
    assert np.all(concs >= 0.0)
    # Target product P must be produced
    assert concs[2] > 0.05


def test_hypotheses_divergence():
    """Verify that different reaction mechanisms produce divergent predictions."""
    model_ideal = ChemicalKineticsModel(0)
    model_poison = ChemicalKineticsModel(1)
    model_crack = ChemicalKineticsModel(3)
    
    # At high temperature and long duration, thermal cracking should reduce P and increase W
    concs_ideal = model_ideal.simulate(440.0, 3.0, 0.5, 60.0)
    concs_crack = model_crack.simulate(440.0, 3.0, 0.5, 60.0)
    
    # Under thermal cracking, product should be lower due to decomposition P -> W
    assert concs_crack[2] < concs_ideal[2]
    assert concs_crack[3] > concs_ideal[3]


def test_chemical_reactor_env():
    """Verify Gymnasium environment compliance."""
    env = ChemicalReactorEnv(ground_truth_id=1, max_experiments=5)
    obs, info = env.reset(seed=42)
    assert obs.shape == (4,)
    assert info["experiments_remaining"] == 5
    
    action = np.array([0.0, 0.5, -0.2, 0.8], dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    assert next_obs.shape == (4,)
    assert np.all(next_obs >= 0.0)
    assert "params" in step_info
    assert step_info["params"]["T"] >= 300.0


def test_chemical_hex_agent():
    """Verify active chemical discovery reduces entropy in batch trials."""
    env = ChemicalReactorEnv(ground_truth_id=3, max_experiments=5)
    agent = ChemicalHExAgent(bounds=env.bounds, num_candidates=50, rng=np.random.RandomState(42))
    
    obs, _ = env.reset(seed=42)
    agent.reset()
    initial_h = agent.belief_tracker.get_shannon_entropy()
    
    for _ in range(4):
        action = agent.select_action()
        next_obs, _, _, _, _ = env.step(action)
        metrics = agent.observe_transition(action, next_obs)
        
    final_h = agent.belief_tracker.get_shannon_entropy()
    assert final_h < initial_h
    # True rule 3 (thermal cracking) should be identified
    map_idx, prob = agent.belief_tracker.get_map_hypothesis()
    assert map_idx == 3
    assert prob > 0.7


def test_bayesian_optimization_agent():
    """Verify GP-BO baseline fitting and action proposal."""
    agent = BayesianOptimizationAgent(action_dim=4, num_candidates=50)
    agent.reset()
    
    # Run 3 dummy steps
    for _ in range(3):
        a = agent.select_action(np.zeros(4))
        assert a.shape == (4,)
        agent.observe_experiment(a, reward=1.5)
        
    assert agent.is_fitted
