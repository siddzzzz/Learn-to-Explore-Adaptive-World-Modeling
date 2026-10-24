"""Bayesian Optimization (GP-BO) Baseline Agent.

The current industry standard for automated chemistry labs (e.g., BoTorch, Ax).
Uses Gaussian Process regression with Upper Confidence Bound (UCB) acquisition.
Crucially: Lacks any mechanistic world model (treats the reactor as a black box).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


class BayesianOptimizationAgent:
    """Black-box Bayesian Optimization agent with GP-UCB."""

    def __init__(
        self,
        action_dim: int = 4,
        kappa: float = 2.0,
        num_candidates: int = 500,
        rng: Optional[np.random.RandomState] = None
    ):
        self.action_dim = action_dim
        self.kappa = kappa
        self.num_candidates = num_candidates
        self.rng = rng or np.random.RandomState(42)

        # Matern 5/2 kernel with noise regularization
        kernel = Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=1e-3)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-4,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=self.rng.randint(0, 10000)
        )

        self.history_actions: List[np.ndarray] = []
        self.history_rewards: List[float] = []
        self.is_fitted = False

    def reset(self) -> None:
        self.history_actions = []
        self.history_rewards = []
        self.is_fitted = False

    def select_action(self, current_obs: np.ndarray) -> np.ndarray:
        """Selects continuous action maximizing GP-UCB acquisition."""
        # Initial exploration: first 2 steps are random trials
        if len(self.history_actions) < 2 or not self.is_fitted:
            action = self.rng.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)
            return action

        # Generate dense candidate pool in [-1, 1]^4
        candidates = self.rng.uniform(-1.0, 1.0, size=(self.num_candidates, self.action_dim))
        
        # Predict mean and epistemic variance from GP
        mu, std = self.gp.predict(candidates, return_std=True)
        
        # UCB Acquisition: alpha(x) = mu(x) + kappa * sigma(x)
        ucb = mu + self.kappa * std
        best_idx = int(np.argmax(ucb))
        
        return candidates[best_idx].astype(np.float32)

    def observe_experiment(self, action: np.ndarray, reward: float) -> None:
        """Updates Gaussian Process posterior with new experiment yield."""
        self.history_actions.append(action.copy())
        self.history_rewards.append(float(reward))

        X = np.array(self.history_actions)
        y = np.array(self.history_rewards)

        if len(X) >= 2:
            self.gp.fit(X, y)
            self.is_fitted = True
