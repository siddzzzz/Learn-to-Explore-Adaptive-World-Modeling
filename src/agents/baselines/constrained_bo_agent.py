"""Constrained Bayesian Optimization (Constrained GP-BO) Baseline Agent.

Industry standard constrained black-box optimizer for automated labs.
Maintains two Gaussian Processes:
1. Objective GP: Models expected downstream product yield / discovery objective.
2. Constraint GP: Models probability of satisfying physical safety constraints (e.g. T < 420K, W < 0.35M).
Computes Constrained Expected Improvement (cEI) or Penalty-Penalized UCB.
Crucially: Lacks any mechanistic kinetic world model, so it cannot anticipate non-linear runaway thresholds.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


class ConstrainedBayesianOptimizationAgent:
    """Constrained Gaussian Process Bayesian Optimization Agent."""

    def __init__(
        self,
        action_dim: int = 4,
        kappa: float = 2.0,
        constraint_penalty: float = 50.0,
        num_candidates: int = 500,
        rng: Optional[np.random.RandomState] = None
    ):
        self.action_dim = action_dim
        self.kappa = kappa
        self.constraint_penalty = constraint_penalty
        self.num_candidates = num_candidates
        self.rng = rng or np.random.RandomState(42)

        # Objective GP
        obj_kernel = Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=1e-3)
        self.gp_obj = GaussianProcessRegressor(
            kernel=obj_kernel,
            alpha=1e-4,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=self.rng.randint(0, 10000)
        )

        # Safety Margin Constraint GP (predicts min safety margin h(s))
        con_kernel = Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=1e-3)
        self.gp_con = GaussianProcessRegressor(
            kernel=con_kernel,
            alpha=1e-4,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=self.rng.randint(0, 10000)
        )

        self.history_actions: List[np.ndarray] = []
        self.history_rewards: List[float] = []
        self.history_margins: List[float] = []
        self.is_fitted = False

    def reset(self) -> None:
        self.history_actions = []
        self.history_rewards = []
        self.history_margins = []
        self.is_fitted = False

    def select_action(self, current_obs: np.ndarray) -> np.ndarray:
        """Selects action maximizing Penalized GP-UCB subject to predicted safety margin >= 0."""
        # Initial cold start with moderate safe-prior actions
        if len(self.history_actions) < 3:
            # Sample moderately around center
            action = self.rng.uniform(-0.6, 0.2, size=self.action_dim).astype(np.float32)
            return np.clip(action, -1.0, 1.0)

        # Generate candidate proposals
        candidates = self.rng.uniform(-1.0, 1.0, size=(self.num_candidates, self.action_dim)).astype(np.float32)

        # Predict objective mean and variance
        mu_obj, sigma_obj = self.gp_obj.predict(candidates, return_std=True)
        ucb_score = mu_obj + self.kappa * sigma_obj

        # Predict constraint margin: h >= 0 is safe
        mu_con, sigma_con = self.gp_con.predict(candidates, return_std=True)
        # Conservative constraint estimate (Lower Confidence Bound on margin)
        con_lcb = mu_con - self.kappa * sigma_con

        # Apply barrier penalty to candidates with negative predicted margin
        penalized_scores = ucb_score.copy()
        unsafe_mask = con_lcb < 0.0
        penalized_scores[unsafe_mask] -= self.constraint_penalty * np.abs(con_lcb[unsafe_mask])

        best_idx = int(np.argmax(penalized_scores))
        chosen_action = candidates[best_idx]
        return np.clip(chosen_action, -1.0, 1.0)

    def observe_result(
        self,
        action: np.ndarray,
        reward: float,
        safety_margin: float = 0.0
    ) -> None:
        """Updates both objective GP and safety constraint GP."""
        self.history_actions.append(np.asarray(action, dtype=np.float32))
        self.history_rewards.append(float(reward))
        self.history_margins.append(float(safety_margin))

        X = np.array(self.history_actions)
        y_obj = np.array(self.history_rewards)
        y_con = np.array(self.history_margins)

        if len(X) >= 3:
            try:
                self.gp_obj.fit(X, y_obj)
                self.gp_con.fit(X, y_con)
                self.is_fitted = True
            except Exception:
                pass
