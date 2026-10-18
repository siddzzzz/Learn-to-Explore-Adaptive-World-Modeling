"""Plan2Explore Disagreement Exploration Baseline.

Selects actions that maximize unweighted ensemble disagreement:
    r_disagreement = (1 / M) ∑_m || f_m(s, a) - f_bar(s, a) ||^2
Reference: Sekar et al. (2020), Planning to Explore via Self-Supervised World Models.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from src.models.belief_tracker import BayesianBeliefTracker
from src.planning.acquisition import compute_ensemble_variance


class Plan2ExploreAgent:
    """Agent driven by ensemble model disagreement."""

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
        """Selects action maximizing unweighted ensemble disagreement across models."""
        best_action = 0
        max_disagreement = -1.0

        for a in range(self.num_actions):
            interv = self.action_decoder_fn(a)
            preds = [m.evaluate(interventions=interv) for m in self.hypothesis_evaluators]
            disagreement = compute_ensemble_variance(np.array(preds))
            if disagreement > max_disagreement:
                max_disagreement = disagreement
                best_action = a

        return best_action

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
