"""Curiosity-driven Exploration Baseline (ICM-style).

Selects actions that maximize one-step prediction error.
Vulnerable to the Noisy TV problem (Theorem 1).
Reference: Pathak et al. (2017), Curiosity-driven Exploration by Self-supervised Prediction.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from src.models.belief_tracker import BayesianBeliefTracker


class CuriosityAgent:
    """Agent driven by prediction error curiosity."""

    def __init__(
        self,
        num_actions: int,
        hypothesis_evaluators: List[Any],
        action_decoder_fn: Any,
        noise_variance: float = 0.0025,
        rng: Optional[np.random.RandomState] = None
    ):
        self.num_actions = num_actions
        self.hypothesis_evaluators = hypothesis_evaluators
        self.K = len(hypothesis_evaluators)
        self.action_decoder_fn = action_decoder_fn
        self.noise_variance = noise_variance
        self.rng = rng or np.random.RandomState(42)

        self.belief_tracker = BayesianBeliefTracker(num_hypotheses=self.K)
        # Running prediction model of transitions (mean over past seen transitions)
        # Initialize with small prior error
        self.seen_errors: Dict[int, float] = {a: float(self.rng.uniform(0.01, 0.05)) for a in range(num_actions)}

    def reset(self) -> None:
        self.belief_tracker.reset()
        self.seen_errors = {a: float(self.rng.uniform(0.01, 0.05)) for a in range(self.num_actions)}

    def select_action(self, current_state: np.ndarray) -> int:
        # Greedily pick action with highest recent prediction error
        # In presence of distractor actions, prediction error on the TV dominates
        if self.rng.rand() < 0.15:
            return int(self.rng.choice(self.num_actions))

        best_a = max(self.seen_errors, key=self.seen_errors.get)
        return int(best_a)

    def observe_transition(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray
    ) -> Dict[str, Any]:
        interv = self.action_decoder_fn(action)
        map_idx, _ = self.belief_tracker.get_map_hypothesis()
        pred_mu = self.hypothesis_evaluators[map_idx].evaluate(interventions=interv)
        
        causal_dim = len(pred_mu)
        # Physical prediction error
        phys_error = float(np.sum((next_state[:causal_dim] - pred_mu) ** 2))
        error = phys_error
        if len(next_state) > causal_dim:
            # TV channel distractor has huge unpredictable aleatoric error
            tv_noise = next_state[causal_dim:]
            error += float(np.sum(tv_noise ** 2))

        self.seen_errors[action] = 0.7 * self.seen_errors[action] + 0.3 * error

        # Bayesian belief update
        log_liks = np.zeros(self.K, dtype=np.float64)
        for k, model in enumerate(self.hypothesis_evaluators):
            mu_k = model.evaluate(interventions=interv)
            diff = next_state[:len(mu_k)] - mu_k
            sse = float(np.sum(diff ** 2))
            var = self.noise_variance
            log_liks[k] = -0.5 * (len(mu_k) * np.log(2 * np.pi * var) + sse / var)

        weights, surprise, is_anomaly = self.belief_tracker.update_from_log_likelihoods(log_liks)
        return {
            "entropy": self.belief_tracker.get_shannon_entropy(),
            "weights": weights,
            "prediction_error": error
        }
