"""Model Predictive Path Integral (MPPI) Controller for Continuous Control.

References:
    - Williams et al. (2017), Information-Theoretic Model Predictive Control
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple
import numpy as np


class MPPIPlanner:
    """Model Predictive Path Integral trajectory optimizer."""

    def __init__(
        self,
        action_dim: int,
        horizon: int = 10,
        num_samples: int = 64,
        temperature: float = 0.5,
        noise_sigma: float = 0.3,
        gamma: float = 0.95,
        rng: Optional[np.random.RandomState] = None
    ):
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.gamma = gamma
        self.rng = rng or np.random.RandomState(42)

        # Nominal control sequence [H, D]
        self.U = np.zeros((horizon, action_dim), dtype=np.float64)

    def reset(self) -> None:
        self.U = np.zeros((self.horizon, self.action_dim), dtype=np.float64)

    def plan(
        self,
        initial_state: np.ndarray,
        rollout_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, float]],
        action_low: float = -1.0,
        action_high: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """Plans optimal continuous action using MPPI importance sampling."""
        # Generate control perturbations ε ~ N(0, σ^2)
        noise = self.rng.normal(0.0, self.noise_sigma, size=(self.num_samples, self.horizon, self.action_dim))
        perturbed_actions = np.clip(self.U[np.newaxis, :, :] + noise, action_low, action_high)

        # Rollout trajectories in world model
        returns = np.zeros(self.num_samples, dtype=np.float64)
        for i in range(self.num_samples):
            s = initial_state.copy()
            total_return = 0.0
            discount = 1.0
            for h in range(self.horizon):
                a = perturbed_actions[i, h]
                s, r = rollout_fn(s, a)
                total_return += discount * r
                discount *= self.gamma
            returns[i] = total_return

        # Importance weights: w_i = exp( (S_i - max S) / λ )
        max_return = np.max(returns)
        exp_weights = np.exp((returns - max_return) / max(1e-4, self.temperature))
        weights = exp_weights / np.sum(exp_weights)

        # Update nominal sequence: U = ∑ w_i * A_i
        weighted_actions = np.sum(weights[:, np.newaxis, np.newaxis] * perturbed_actions, axis=0)
        self.U = weighted_actions

        best_action = self.U[0].copy()

        # Receding horizon shift: slide forward by 1 step
        self.U[:-1] = self.U[1:]
        self.U[-1] = 0.0

        return best_action, float(max_return)
