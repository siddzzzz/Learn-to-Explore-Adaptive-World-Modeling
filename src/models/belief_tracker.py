"""Bayesian Belief Tracker over Hypothesis Space.

Maintains posterior probability distribution P(θ | D_t), tracks Shannon entropy,
measures observation surprise, and detects non-stationary rule mutations.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from scipy.special import logsumexp

from src.models.hypothesis import HypothesisState


class BayesianBeliefTracker:
    """Maintains and updates posterior distribution b_t(θ) over discrete hypotheses."""

    def __init__(
        self,
        num_hypotheses: int,
        prior: Optional[np.ndarray] = None,
        surprise_threshold: float = 15.0
    ):
        self.num_hypotheses = num_hypotheses
        self.surprise_threshold = surprise_threshold

        if prior is not None:
            assert len(prior) == num_hypotheses
            prior = np.asarray(prior, dtype=np.float64)
            self.log_weights = np.log(np.clip(prior / np.sum(prior), 1e-15, 1.0))
        else:
            self.log_weights = np.full(num_hypotheses, -np.log(num_hypotheses), dtype=np.float64)

        self.initial_log_weights = self.log_weights.copy()
        self.history_entropy: List[float] = [self.get_shannon_entropy()]
        self.history_surprise: List[float] = []

    @property
    def weights(self) -> np.ndarray:
        """Returns normalized posterior probabilities w_1, ..., w_K."""
        return np.exp(self.log_weights - logsumexp(self.log_weights))

    def reset(self) -> None:
        """Resets belief to initial prior."""
        self.log_weights = self.initial_log_weights.copy()
        self.history_entropy = [self.get_shannon_entropy()]
        self.history_surprise = []

    def update_from_log_likelihoods(self, log_likelihoods: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Performs Bayesian update from log p(y | a, θ_k).
        
        Args:
            log_likelihoods: Array of shape [num_hypotheses] containing log p(y | a, θ_k).
            
        Returns:
            Tuple of:
                - updated normalized weights
                - observation surprise (negative marginal log-likelihood)
                - is_anomaly (boolean flag if surprise exceeds threshold)
        """
        assert len(log_likelihoods) == self.num_hypotheses
        current_weights = self.weights

        # Marginal log-likelihood of observation: ln p(y | a, b) = ln ∑_k w_k p(y | a, θ_k)
        marginal_log_lik = logsumexp(self.log_weights + log_likelihoods)
        surprise = -float(marginal_log_lik)
        self.history_surprise.append(surprise)

        is_anomaly = surprise > self.surprise_threshold

        # Bayesian posterior update in log space
        self.log_weights = self.log_weights + log_likelihoods
        self.log_weights -= logsumexp(self.log_weights)

        entropy = self.get_shannon_entropy()
        self.history_entropy.append(entropy)

        return self.weights, surprise, is_anomaly

    def get_shannon_entropy(self) -> float:
        """Computes discrete Shannon entropy H(b_t) in nats: -∑ w_k ln w_k."""
        w = self.weights
        w_safe = np.clip(w, 1e-15, 1.0)
        h = -float(np.sum(w_safe * np.log(w_safe)))
        return max(0.0, h)

    def get_normalized_entropy(self) -> float:
        """Computes entropy normalized to [0, 1] relative to uniform maximum ln(K)."""
        max_h = np.log(self.num_hypotheses)
        if max_h == 0.0:
            return 0.0
        return float(np.clip(self.get_shannon_entropy() / max_h, 0.0, 1.0))

    def get_map_hypothesis(self) -> Tuple[int, float]:
        """Returns the Maximum a Posteriori (MAP) hypothesis index and its probability."""
        w = self.weights
        idx = int(np.argmax(w))
        return idx, float(w[idx])

    def broaden_belief(self, mixing_factor: float = 0.3) -> np.ndarray:
        """Mixes current belief with uniform distribution when domain shift / rule mutation is detected.
        
        w_new = (1 - α) * w_current + α * (1 / K)
        """
        uniform = np.full(self.num_hypotheses, 1.0 / self.num_hypotheses)
        current_w = self.weights
        new_w = (1.0 - mixing_factor) * current_w + mixing_factor * uniform
        self.log_weights = np.log(np.clip(new_w, 1e-15, 1.0))
        return self.weights
