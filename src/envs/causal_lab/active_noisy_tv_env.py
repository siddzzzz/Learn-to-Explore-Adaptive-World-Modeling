"""Interactive Active Noisy-TV Causal Environment.

In this environment, an agent can explicitly choose whether to perform
informative causal experiments on the underlying physical system or interact
with an aleatoric noise source ("Noisy TV") by pressing distractor channels.
Tests the Noisy TV trap directly on curiosity and ensemble disagreement baselines.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.envs.causal_lab.scm import (
    CausalHypothesis,
    StructuralCausalModel,
    generate_canonical_hypothesis_space,
)


class ActiveNoisyTVLabEnv(gym.Env):
    """Gymnasium environment featuring an active Noisy-TV distractor channel.
    
    Action Space:
    - 0 .. 3*N: Physical Causal Interventions:
        - 0: Observational step (no intervention)
        - 1 .. N: do(X_i = +2.0)
        - N+1 .. 2N: do(X_i = -2.0)
        - 2N+1 .. 3N: do(X_i = 0.0)
    - 3*N+1 .. 3*N+K_tv: Interactive TV Distractor Actions:
        - Agent changes TV channel / sends bursts to pure stochastic noise generator.
        - Provides immediate huge prediction error and variance, but zero structural information.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(
        self,
        num_variables: int = 4,
        noise_std: float = 0.05,
        num_tv_actions: int = 4,
        noisy_tv_dim: int = 4,
        noisy_tv_std: float = 2.0,
        max_steps: int = 50,
        ground_truth_id: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.num_variables = num_variables
        self.noise_std = noise_std
        self.num_tv_actions = num_tv_actions
        self.noisy_tv_dim = noisy_tv_dim
        self.noisy_tv_std = noisy_tv_std
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Generate hypothesis space
        self.hypothesis_space: List[CausalHypothesis] = generate_canonical_hypothesis_space(num_variables)
        self.num_hypotheses = len(self.hypothesis_space)

        # Ground truth
        self._ground_truth_id = ground_truth_id if ground_truth_id is not None else 1
        self.ground_truth_hypothesis: CausalHypothesis = self.hypothesis_space[self._ground_truth_id]

        # Active SCM instance
        self.scm = StructuralCausalModel(
            num_variables=self.num_variables,
            hypothesis=self.ground_truth_hypothesis,
            noise_std=self.noise_std,
        )

        # Total Actions:
        # Causal actions: 3 * num_variables + 1
        self.num_causal_actions = 3 * self.num_variables + 1
        self.num_actions = self.num_causal_actions + self.num_tv_actions
        self.action_space = spaces.Discrete(self.num_actions)

        # Observation Space:
        # [X_0, ..., X_{N-1}, TV_0, ..., TV_{D-1}]
        self.obs_dim = self.num_variables + self.noisy_tv_dim
        low = np.full(self.obs_dim, -20.0, dtype=np.float32)
        high = np.full(self.obs_dim, 20.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Internal state
        self.current_state: np.ndarray = np.zeros(self.num_variables, dtype=np.float32)
        self.current_tv_state: np.ndarray = np.zeros(self.noisy_tv_dim, dtype=np.float32)
        self.current_step: int = 0
        self.last_action: Optional[int] = None
        self.last_action_is_distractor: bool = False

    def is_distractor_action(self, action: int) -> bool:
        """Returns True if action interacts with the Noisy TV distractor."""
        return action >= self.num_causal_actions

    def decode_action(self, action: int) -> Dict[int, float]:
        """Maps action index to Pearl's do-operator dict."""
        if action >= self.num_causal_actions:
            # Distractor action does not intervene on physical variables
            return {}
        if action == 0:
            return {}
        
        adjusted = action - 1
        var_idx = adjusted % self.num_variables
        stimulus_type = adjusted // self.num_variables

        if stimulus_type == 0:
            val = 2.0
        elif stimulus_type == 1:
            val = -2.0
        else:
            val = 0.0

        return {var_idx: val}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.scm.rng = np.random.RandomState(seed)

        self.current_step = 0
        self.last_action = None
        self.last_action_is_distractor = False

        # Reset physical variables
        self.current_state = self.scm.evaluate(interventions={})
        # Reset TV channel state
        self.current_tv_state = self.scm.rng.normal(0.0, self.noisy_tv_std, size=self.noisy_tv_dim).astype(np.float32)

        obs = self._get_observation()
        info = {
            "ground_truth_id": self._ground_truth_id,
            "hypothesis_name": self.ground_truth_hypothesis.name,
            "current_step": self.current_step,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        self.last_action = action
        is_tv = self.is_distractor_action(action)
        self.last_action_is_distractor = is_tv

        if is_tv:
            # Distractor intervention: does not perturb physical state,
            # but creates massive unpredictable changes on the TV channel
            tv_channel = action - self.num_causal_actions
            channel_offset = (tv_channel + 1.0) * 2.0
            noise = self.scm.rng.normal(channel_offset, self.noisy_tv_std, size=self.noisy_tv_dim)
            self.current_tv_state = noise.astype(np.float32)
            interventions = {}
        else:
            interventions = self.decode_action(action)
            self.current_state = self.scm.evaluate(interventions=interventions)
            # Default TV background noise
            self.current_tv_state = self.scm.rng.normal(0.0, self.noisy_tv_std, size=self.noisy_tv_dim).astype(np.float32)

        obs = self._get_observation()
        reward = 0.0
        terminated = False
        truncated = self.current_step >= self.max_steps

        info = {
            "interventions": interventions,
            "is_distractor_action": is_tv,
            "ground_truth_id": self._ground_truth_id,
            "causal_state": self.current_state.copy(),
            "tv_state": self.current_tv_state.copy(),
            "step": self.current_step,
        }
        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[:self.num_variables] = self.current_state
        obs[self.num_variables:] = self.current_tv_state
        return obs

    def get_hypothesis_space(self) -> List[CausalHypothesis]:
        return self.hypothesis_space
