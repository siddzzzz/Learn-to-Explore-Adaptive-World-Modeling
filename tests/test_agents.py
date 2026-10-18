"""Unit tests for HEx-WM and Baseline Agents."""

import numpy as np
import pytest
from src.envs.causal_lab import CausalLabEnv, StructuralCausalModel
from src.agents import HExWMAgent
from src.agents.baselines import RandomAgent, CuriosityAgent, Plan2ExploreAgent, OracleAgent


@pytest.fixture
def test_setup():
    env = CausalLabEnv(num_variables=4, ground_truth_id=1, noisy_tv_dim=2)
    hypothesis_space = env.get_hypothesis_space()
    evaluators = [StructuralCausalModel(4, h, noise_std=0.0) for h in hypothesis_space]
    action_decoder = env.decode_action
    return env, hypothesis_space, evaluators, action_decoder


def test_hex_wm_agent_discovery(test_setup):
    """Verify that HEx-WM actively selects discriminating experiments and discovers the rule."""
    env, hypothesis_space, evaluators, action_decoder = test_setup
    
    agent = HExWMAgent(
        hypothesis_evaluators=evaluators,
        num_actions=env.num_actions,
        action_decoder_fn=action_decoder,
        noise_variance=0.0025,
        entropy_threshold=0.1,
        planning_horizon=2,
        num_cem_samples=16,
        num_cem_iters=2
    )
    
    obs, _ = env.reset(seed=42)
    initial_entropy = agent.belief_tracker.get_shannon_entropy()
    
    # Run 10 steps of active exploration
    for step in range(10):
        action = agent.select_action(obs[:4])
        next_obs, reward, terminated, truncated, _ = env.step(action)
        metrics = agent.observe_transition(obs[:4], action, next_obs[:4])
        obs = next_obs
        
    final_entropy = agent.belief_tracker.get_shannon_entropy()
    assert final_entropy < initial_entropy
    
    # Ground truth was 1, so hypothesis 1 should have high weight
    map_idx, map_prob = agent.belief_tracker.get_map_hypothesis()
    assert map_idx == 1
    assert map_prob > 0.6


def test_baseline_agents_execution(test_setup):
    """Verify that all baseline agents run cleanly in the benchmark."""
    env, hypothesis_space, evaluators, action_decoder = test_setup
    
    agents = [
        RandomAgent(env.num_actions, evaluators, action_decoder),
        CuriosityAgent(env.num_actions, evaluators, action_decoder),
        Plan2ExploreAgent(env.num_actions, evaluators, action_decoder),
        OracleAgent(env.num_actions, hypothesis_space, action_decoder)
    ]
    
    for agent in agents:
        obs, _ = env.reset(seed=123)
        agent.reset()
        for step in range(3):
            action = agent.select_action(obs[:4])
            next_obs, _, _, _, _ = env.step(action)
            metrics = agent.observe_transition(obs[:4], action, next_obs[:4])
            assert "entropy" in metrics
            obs = next_obs
