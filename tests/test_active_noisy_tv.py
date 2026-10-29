"""Unit tests for the Active Noisy-TV Environment and Agents."""

from __future__ import annotations
import numpy as np
import pytest

from src.envs.causal_lab.active_noisy_tv_env import ActiveNoisyTVLabEnv
from src.envs.causal_lab.scm import StructuralCausalModel
from src.agents import HExWMAgent
from src.agents.baselines import CuriosityAgent, Plan2ExploreAgent, RandomAgent


def test_active_noisy_tv_environment_initialization():
    """Verify state, observation, and action dimensions of ActiveNoisyTVLabEnv."""
    env = ActiveNoisyTVLabEnv(
        num_variables=4,
        noisy_tv_dim=4,
        num_tv_actions=3,
        noisy_tv_std=1.5,
        max_steps=20
    )
    # Causal actions = 3 * 4 + 1 = 13. TV actions = 3. Total = 16.
    assert env.num_causal_actions == 13
    assert env.num_actions == 16
    assert env.action_space.n == 16
    assert env.obs_dim == 8  # 4 physical + 4 TV

    obs, info = env.reset(seed=42)
    assert obs.shape == (8,)
    assert info["current_step"] == 0
    assert not env.is_distractor_action(0)
    assert env.is_distractor_action(13)
    assert env.is_distractor_action(15)


def test_active_noisy_tv_step_dynamics():
    """Verify physical interventions vs. TV distractor interventions."""
    env = ActiveNoisyTVLabEnv(
        num_variables=4,
        noisy_tv_dim=4,
        num_tv_actions=2,
        max_steps=10
    )
    obs, _ = env.reset(seed=123)

    # 1. Physical action: do(X_0 = 2.0) -> action 1
    obs_next, r, term, trunc, info = env.step(1)
    assert not info["is_distractor_action"]
    assert info["interventions"] == {0: 2.0}
    assert info["causal_state"][0] == 2.0

    # 2. Distractor action: TV channel button -> action 13
    obs_tv, r, term, trunc, info_tv = env.step(13)
    assert info_tv["is_distractor_action"]
    assert info_tv["interventions"] == {}
    # TV channels produce significant values
    assert np.linalg.norm(obs_tv[4:]) > 0.1


def test_agent_interaction_with_active_noisy_tv():
    """Verify HEx-WM, Plan2Explore, and Curiosity agents step cleanly without errors."""
    env = ActiveNoisyTVLabEnv(
        num_variables=4,
        noisy_tv_dim=4,
        num_tv_actions=3,
        max_steps=5
    )
    hypothesis_space = env.get_hypothesis_space()
    evaluators = [StructuralCausalModel(4, h, noise_std=0.0) for h in hypothesis_space]
    action_decoder = env.decode_action
    rng = np.random.RandomState(42)

    # HEx-WM
    hex_agent = HExWMAgent(
        hypothesis_evaluators=evaluators,
        num_actions=env.num_actions,
        action_decoder_fn=action_decoder,
        planning_horizon=2,
        num_cem_samples=16,
        num_cem_iters=2,
        rng=rng
    )

    # Curiosity
    curiosity_agent = CuriosityAgent(
        num_actions=env.num_actions,
        hypothesis_evaluators=evaluators,
        action_decoder_fn=action_decoder,
        rng=rng
    )

    # Plan2Explore
    p2e_agent = Plan2ExploreAgent(
        num_actions=env.num_actions,
        hypothesis_evaluators=evaluators,
        action_decoder_fn=action_decoder,
        rng=rng
    )

    for ag in [hex_agent, curiosity_agent, p2e_agent]:
        obs, _ = env.reset(seed=42)
        ag.reset()
        for _ in range(3):
            action = ag.select_action(obs[:4])
            next_obs, _, _, _, _ = env.step(action)
            metrics = ag.observe_transition(obs, action, next_obs)
            assert "entropy" in metrics
            obs = next_obs
