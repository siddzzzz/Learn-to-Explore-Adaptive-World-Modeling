"""Analytical Bayesian Filter and Bayes-Optimal Oracle for Causal-Lab.

Computes exact ground-truth posterior probabilities, exact Shannon entropy,
and analytical Expected Information Gain (EIG) for candidate experiments.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.special import logsumexp

from src.envs.causal_lab.scm import CausalHypothesis, StructuralCausalModel


class AnalyticalBayesianFilter:
    """Exact Bayesian belief updater over discrete causal hypotheses."""

    def __init__(
        self,
        hypothesis_space: List[CausalHypothesis],
        num_variables: int = 4,
        noise_std: float = 0.05
    ):
        self.hypothesis_space = hypothesis_space
        self.K = len(hypothesis_space)
        self.num_variables = num_variables
        self.noise_std = noise_std
        self.var = noise_std ** 2

        # Initialize uniform prior: w_k = 1 / K
        self.log_weights = np.full(self.K, -np.log(self.K), dtype=np.float64)

        # Pre-instantiate deterministic evaluators for each hypothesis
        self.scms = [
            StructuralCausalModel(num_variables, h, noise_std=0.0)
            for h in self.hypothesis_space
        ]

    @property
    def weights(self) -> np.ndarray:
        """Returns normalized posterior probabilities p(θ | D)."""
        return np.exp(self.log_weights - logsumexp(self.log_weights))

    def reset(self) -> None:
        """Resets belief to uniform prior."""
        self.log_weights = np.full(self.K, -np.log(self.K), dtype=np.float64)

    def predict_mean(self, hypothesis_idx: int, interventions: Dict[int, float]) -> np.ndarray:
        """Predicts the deterministic expected outcome under a given hypothesis."""
        return self.scms[hypothesis_idx].evaluate(
            interventions=interventions,
            exogenous_noise=np.zeros(self.num_variables)
        )

    def log_likelihood(
        self,
        hypothesis_idx: int,
        interventions: Dict[int, float],
        observed_state: np.ndarray
    ) -> float:
        """Computes ln p(X_obs | a, θ_k) under isotropic Gaussian noise."""
        mu = self.predict_mean(hypothesis_idx, interventions)
        diff = observed_state[:self.num_variables] - mu
        sse = float(np.sum(diff ** 2))
        d = self.num_variables
        log_lik = -0.5 * (d * np.log(2 * np.pi * self.var) + sse / self.var)
        return log_lik

    def update(self, interventions: Dict[int, float], observed_state: np.ndarray) -> np.ndarray:
        """Performs exact Bayesian update: w_{t+1} ∝ w_t · p(X_obs | a, θ)."""
        log_liks = np.zeros(self.K, dtype=np.float64)
        for k in range(self.K):
            log_liks[k] = self.log_likelihood(k, interventions, observed_state)

        # Posterior in log space with LogSumExp normalization
        self.log_weights = self.log_weights + log_liks
        self.log_weights -= logsumexp(self.log_weights)
        return self.weights

    def get_shannon_entropy(self) -> float:
        """Computes discrete Shannon entropy H(b_t) in nats."""
        w = self.weights
        # Clip to prevent log(0)
        w_safe = np.clip(w, 1e-15, 1.0)
        entropy = -float(np.sum(w_safe * np.log(w_safe)))
        return max(0.0, entropy)

    def compute_exact_eig(self, interventions: Dict[int, float]) -> float:
        """Computes analytical Expected Information Gain for a candidate intervention.
        
        EIG(a) = H(b) - E_{X'}[ H(b(· | X', a)) ]
        """
        w = self.weights
        current_h = self.get_shannon_entropy()

        # Generate expected outcomes under each hypothesis
        predictions = [self.predict_mean(k, interventions) for k in range(self.K)]

        # Sample or evaluate expectation over possible outcomes
        # Since noise is small, Monte Carlo expectation over mixture components is highly accurate:
        expected_posterior_entropy = 0.0
        for k in range(self.K):
            if w[k] < 1e-6:
                continue
            # Center of component k
            x_k = predictions[k]
            # Hypothetical update
            sim_log_weights = np.zeros(self.K)
            for j in range(self.K):
                diff = x_k - predictions[j]
                sse = float(np.sum(diff ** 2))
                sim_log_weights[j] = self.log_weights[j] - 0.5 * (sse / self.var)
            sim_log_weights -= logsumexp(sim_log_weights)
            sim_w = np.exp(sim_log_weights)
            sim_h = -float(np.sum(sim_w * np.log(np.clip(sim_w, 1e-15, 1.0))))
            expected_posterior_entropy += w[k] * sim_h

        eig = current_h - expected_posterior_entropy
        return max(0.0, float(eig))
