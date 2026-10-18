"""Random Exploration Baseline Agent."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from src.models.belief_tracker import BayesianBeliefTracker


class RandomAgent:
    """Agent executing uniform random exploration."""

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

    def reset(self) -> None:
        self.belief_tracker.reset()

    def select_action(self, current_state: np.ndarray) -> int:
        return int(self.rng.choice(self.num_actions))

    def observe_transition(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray
    ) -> Dict[str, Any]:
        interv = self.action_decoder_fn(action)
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
            "map_hypothesis": np.argmax(weights)
        }
