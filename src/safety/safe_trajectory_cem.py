"""Safe Cross-Entropy Method (Safe-CEM) trajectory planner with barrier filtering."""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from src.safety.epistemic_safety_filter import EpistemicSafetyFilter


class SafeTrajectoryCEMPlanner:
    """Non-myopic Cross-Entropy Method planner enforcing epistemic barrier constraints."""

    def __init__(
        self,
        action_dim: int,
        safety_filter: EpistemicSafetyFilter,
        horizon: int = 4,
        num_samples: int = 64,
        num_elites: int = 12,
        num_iterations: int = 5,
        gamma: float = 0.95,
        rng: Optional[np.random.RandomState] = None
    ):
        self.action_dim = action_dim
        self.safety_filter = safety_filter
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.gamma = gamma
        self.rng = rng or np.random.RandomState(42)

    def plan_safe_continuous(
        self,
        initial_state: np.ndarray,
        rollout_hypotheses_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        reward_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
        hypothesis_weights: Optional[np.ndarray] = None,
        action_low: float = -1.0,
        action_high: float = 1.0
    ) -> Tuple[np.ndarray, bool, float]:
        """Plans optimal continuous action sequence subject to epistemic barrier constraints."""
        mean = np.zeros((self.horizon, self.action_dim), dtype=np.float64)
        std = np.ones((self.horizon, self.action_dim), dtype=np.float64)

        best_action = np.zeros(self.action_dim, dtype=np.float64)
        best_return = -np.inf
        found_safe = False

        for iteration in range(self.num_iterations):
            samples = self.rng.normal(mean, std, size=(self.num_samples, self.horizon, self.action_dim))
            samples = np.clip(samples, action_low, action_high)

            returns = np.zeros(self.num_samples, dtype=np.float64)
            trajectory_is_safe = np.ones(self.num_samples, dtype=bool)

            for i in range(self.num_samples):
                s = initial_state.copy()
                total_return = 0.0
                discount = 1.0

                for h in range(self.horizon):
                    a = samples[i, h]
                    # Evaluate predictions across hypotheses
                    preds = rollout_hypotheses_fn(s, a)
                    is_safe, margin, _ = self.safety_filter.evaluate_action_safety(
                        s, preds, hypothesis_weights
                    )

                    if not is_safe:
                        trajectory_is_safe[i] = False
                        # Severe penalty for safety violation
                        total_return -= 1000.0 * (1.0 + abs(margin))
                        break

                    r = reward_fn(s, a, preds)
                    total_return += discount * r
                    discount *= self.gamma

                    # Rollout to mean predicted next state
                    if hypothesis_weights is not None:
                        w = hypothesis_weights / np.sum(hypothesis_weights)
                        s = np.sum(w[:, np.newaxis] * preds, axis=0)
                    else:
                        s = np.mean(preds, axis=0)

                returns[i] = total_return

            # Check if any safe trajectories found
            safe_mask = trajectory_is_safe
            if np.any(safe_mask):
                found_safe = True
                safe_indices = np.where(safe_mask)[0]
                elite_subset = safe_indices[np.argsort(returns[safe_indices])[-self.num_elites:]]
            else:
                # Fallback to least unsafe trajectories
                elite_subset = np.argsort(returns)[-self.num_elites:]

            elites = samples[elite_subset]
            if returns[elite_subset[-1]] > best_return:
                best_return = returns[elite_subset[-1]]
                best_action = elites[-1, 0].copy()

            # Update distribution
            new_mean = np.mean(elites, axis=0)
            new_std = np.std(elites, axis=0) + 1e-4

            mean = 0.5 * mean + 0.5 * new_mean
            std = 0.5 * std + 0.5 * new_std

        return best_action, found_safe, float(best_return)
