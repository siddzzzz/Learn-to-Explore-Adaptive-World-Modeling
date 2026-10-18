"""Causal-Lab Gymnasium Environment.

An interventional causal laboratory benchmark with exact hypothesis ground truth,
parameterized Structural Causal Models, and controllable aleatoric noise (Noisy TV).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.envs.causal_lab.scm import (
    CausalHypothesis,
    StructuralCausalModel,
    generate_canonical_hypothesis_space
)


class CausalLabEnv(gym.Env):
    """Gymnasium environment for Active Hypothesis Testing in Causal Systems.
    
    The environment is parameterized by an unobserved Structural Causal Model (SCM)
    hypothesis θ* ∈ Θ. The agent executes interventional experiments to identify θ*.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(
        self,
        num_variables: int = 4,
        noise_std: float = 0.05,
        noisy_tv_dim: int = 2,
        noisy_tv_std: float = 1.0,
        max_steps: int = 100,
        ground_truth_id: Optional[int] = None,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        self.num_variables = num_variables
        self.noise_std = noise_std
        self.noisy_tv_dim = noisy_tv_dim
        self.noisy_tv_std = noisy_tv_std
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Generate hypothesis space Θ
        self.hypothesis_space: List[CausalHypothesis] = generate_canonical_hypothesis_space(num_variables)
        self.num_hypotheses = len(self.hypothesis_space)

        # Ground truth hypothesis θ*
        self._ground_truth_id = ground_truth_id if ground_truth_id is not None else 1
        self.ground_truth_hypothesis: CausalHypothesis = self.hypothesis_space[self._ground_truth_id]

        # Active SCM instance
        self.scm = StructuralCausalModel(
            num_variables=self.num_variables,
            hypothesis=self.ground_truth_hypothesis,
            noise_std=self.noise_std
        )

        # Action Space:
        # 0: Observational step (no intervention)
        # 1 .. N: do(X_i = +2.0) [High stimulus]
        # N+1 .. 2N: do(X_i = -2.0) [Low stimulus]
        # 2N+1 .. 3N: do(X_i = 0.0) [Zero stimulus]
        self.num_actions = 3 * self.num_variables + 1
        self.action_space = spaces.Discrete(self.num_actions)

        # Observation Space:
        # [X_0, ..., X_{N-1}, Noisy_TV_0, ..., Noisy_TV_{D-1}]
        self.obs_dim = self.num_variables + self.noisy_tv_dim
        low = np.full(self.obs_dim, -10.0, dtype=np.float32)
        high = np.full(self.obs_dim, 10.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Task evaluation goal
        self.target_goal: Optional[np.ndarray] = None
        self.in_evaluation_mode: bool = False

        # Internal state
        self.current_state: np.ndarray = np.zeros(self.num_variables, dtype=np.float32)
        self.current_step: int = 0
        self.last_action: Optional[int] = None
        self.last_intervention: Dict[int, float] = {}

    def decode_action(self, action: int) -> Dict[int, float]:
        """Maps discrete action integer to Pearl's do-operator dict."""
        if action == 0:
            return {}  # Observational step (no intervention)
        
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
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.scm.rng = np.random.RandomState(seed)

        self.current_step = 0
        self.last_action = None
        self.last_intervention = {}

        # Reset state from natural un-intervened distribution
        self.current_state = self.scm.evaluate(interventions={})
        obs = self._get_observation()
        info = {
            "ground_truth_id": self._ground_truth_id,
            "hypothesis_name": self.ground_truth_hypothesis.name,
            "current_step": self.current_step
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        self.last_action = action
        interventions = self.decode_action(action)
        self.last_intervention = interventions

        # Step the causal physical mechanisms
        self.current_state = self.scm.evaluate(interventions=interventions)
        obs = self._get_observation()

        # Compute reward
        if self.in_evaluation_mode and self.target_goal is not None:
            # Exploitation: negative MSE to target state
            reward = float(-np.sum((self.current_state - self.target_goal) ** 2))
        else:
            # Epistemic discovery phase: zero extrinsic reward
            reward = 0.0

        terminated = False
        truncated = self.current_step >= self.max_steps

        info = {
            "interventions": interventions,
            "ground_truth_id": self._ground_truth_id,
            "causal_state": self.current_state.copy(),
            "step": self.current_step
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Constructs observation with genuine causal variables and Noisy TV distractor."""
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[:self.num_variables] = self.current_state

        if self.noisy_tv_dim > 0:
            # Generate pure aleatoric noise (unpredictable distractor)
            noise = self.scm.rng.normal(0.0, self.noisy_tv_std, size=self.noisy_tv_dim)
            obs[self.num_variables:] = noise

        return obs

    def mutate_rules(self, new_hypothesis_id: Optional[int] = None) -> CausalHypothesis:
        """Silently swaps the environment's governing rule to test non-stationary adaptation."""
        if new_hypothesis_id is None:
            # Pick a different rule
            candidates = [h for h in range(self.num_hypotheses) if h != self._ground_truth_id]
            new_hypothesis_id = int(self.scm.rng.choice(candidates))

        self._ground_truth_id = new_hypothesis_id
        self.ground_truth_hypothesis = self.hypothesis_space[new_hypothesis_id]
        self.scm.hypothesis = self.ground_truth_hypothesis
        return self.ground_truth_hypothesis

    def set_evaluation_mode(self, target_goal: np.ndarray) -> None:
        """Transitions environment to downstream task evaluation mode."""
        self.in_evaluation_mode = True
        self.target_goal = np.asarray(target_goal, dtype=np.float32)

    def get_hypothesis_space(self) -> List[CausalHypothesis]:
        return self.hypothesis_space

    def get_ground_truth(self) -> CausalHypothesis:
        return self.ground_truth_hypothesis

    def render(self) -> Optional[str]:
        output = [
            f"--- Causal Lab Step {self.current_step} ---",
            f"Active Rule: {self.ground_truth_hypothesis.name} (ID: {self._ground_truth_id})",
            f"Last Action: {self.last_action} -> {self.last_intervention}",
            f"State: {np.round(self.current_state, 3)}"
        ]
        text = "\n".join(output)
        if self.render_mode == "human":
            print(text)
        return text
