"""Trajectory-Level Cross-Entropy Method (CEM) for Non-Myopic Experimental Design.

Plans multi-step action trajectories in the imagined world model to maximize
cumulative Information Gain (CHD) or downstream task rewards.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


class TrajectoryCEMPlanner:
    """Non-myopic Cross-Entropy Method planner over action sequences."""

    def __init__(
        self,
        action_dim: int,
        is_discrete: bool = True,
        horizon: int = 5,
        num_samples: int = 64,
        num_elites: int = 12,
        num_iterations: int = 5,
        gamma: float = 0.95,
        temperature: float = 0.1,
        rng: Optional[np.random.RandomState] = None
    ):
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.gamma = gamma
        self.temperature = temperature
        self.rng = rng or np.random.RandomState(42)

    def plan_discrete(
        self,
        initial_state: np.ndarray,
        rollout_fn: Callable[[np.ndarray, int], Tuple[np.ndarray, float]]
    ) -> Tuple[int, List[int], float]:
        """Plans discrete action sequence over horizon H using categorical CEM.
        
        Args:
            initial_state: Current state of the environment.
            rollout_fn: Function (state, action) -> (next_state, reward).
            
        Returns:
            Tuple of:
                - best first action a_0*
                - best full trajectory [a_0*, ..., a_{H-1}*]
                - expected trajectory return J(A*)
        """
        # Categorical probabilities [horizon, action_dim]
        probs = np.full((self.horizon, self.action_dim), 1.0 / self.action_dim, dtype=np.float64)

        best_trajectory = [0] * self.horizon
        best_return = -np.inf

        for iteration in range(self.num_iterations):
            trajectories = np.zeros((self.num_samples, self.horizon), dtype=int)
            returns = np.zeros(self.num_samples, dtype=np.float64)

            # Sample candidate action sequences
            for i in range(self.num_samples):
                for h in range(self.horizon):
                    trajectories[i, h] = self.rng.choice(self.action_dim, p=probs[h])

            # Evaluate each candidate in the imagined world model
            for i in range(self.num_samples):
                s = initial_state.copy()
                total_return = 0.0
                discount = 1.0
                for h in range(self.horizon):
                    a = trajectories[i, h]
                    s, r = rollout_fn(s, a)
                    total_return += discount * r
                    discount *= self.gamma
                returns[i] = total_return

            # Select elite candidates
            elite_indices = np.argsort(returns)[-self.num_elites:]
            elite_trajectories = trajectories[elite_indices]

            # Track best found
            if returns[elite_indices[-1]] > best_return:
                best_return = returns[elite_indices[-1]]
                best_trajectory = list(elite_trajectories[-1])

            # Refine categorical probabilities using smoothed elite frequencies
            new_probs = np.zeros_like(probs)
            for h in range(self.horizon):
                counts = np.bincount(elite_trajectories[:, h], minlength=self.action_dim)
                smoothed = counts + self.temperature
                new_probs[h] = smoothed / np.sum(smoothed)

            probs = 0.5 * probs + 0.5 * new_probs

        return int(best_trajectory[0]), best_trajectory, float(best_return)

    def plan_continuous(
        self,
        initial_state: np.ndarray,
        rollout_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, float]],
        action_low: float = -1.0,
        action_high: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Plans continuous action sequences using Gaussian CEM."""
        mean = np.zeros((self.horizon, self.action_dim), dtype=np.float64)
        std = np.ones((self.horizon, self.action_dim), dtype=np.float64)

        best_trajectory = mean.copy()
        best_return = -np.inf

        for iteration in range(self.num_iterations):
            # Sample candidate trajectories [N, H, D]
            samples = self.rng.normal(mean, std, size=(self.num_samples, self.horizon, self.action_dim))
            samples = np.clip(samples, action_low, action_high)

            returns = np.zeros(self.num_samples, dtype=np.float64)
            for i in range(self.num_samples):
                s = initial_state.copy()
                total_return = 0.0
                discount = 1.0
                for h in range(self.horizon):
                    a = samples[i, h]
                    s, r = rollout_fn(s, a)
                    total_return += discount * r
                    discount *= self.gamma
                returns[i] = total_return

            elite_indices = np.argsort(returns)[-self.num_elites:]
            elites = samples[elite_indices]

            if returns[elite_indices[-1]] > best_return:
                best_return = returns[elite_indices[-1]]
                best_trajectory = elites[-1].copy()

            # Update Gaussian parameters
            new_mean = np.mean(elites, axis=0)
            new_std = np.std(elites, axis=0) + 1e-4

            mean = 0.5 * mean + 0.5 * new_mean
            std = 0.5 * std + 0.5 * new_std

        return best_trajectory[0], best_trajectory, float(best_return)
