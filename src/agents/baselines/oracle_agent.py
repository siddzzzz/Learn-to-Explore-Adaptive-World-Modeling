"""Greedy Bayes-Optimal Oracle Agent.

Greedily selects the action that maximizes exact analytical Expected Information Gain.
Represents the theoretical performance upper bound.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from src.envs.causal_lab.oracle import AnalyticalBayesianFilter


class OracleAgent:
    """Theoretical upper bound agent maximizing analytical Expected Information Gain."""

    def __init__(
        self,
        num_actions: int,
        hypothesis_space: List[Any],
        action_decoder_fn: Any,
        num_variables: int = 4,
        noise_std: float = 0.05
    ):
        self.num_actions = num_actions
        self.action_decoder_fn = action_decoder_fn
        self.oracle_filter = AnalyticalBayesianFilter(
            hypothesis_space=hypothesis_space,
            num_variables=num_variables,
            noise_std=noise_std
        )

    def reset(self) -> None:
        self.oracle_filter.reset()

    def select_action(self, current_state: np.ndarray) -> int:
        best_a = 0
        max_eig = -1.0

        for a in range(self.num_actions):
            interv = self.action_decoder_fn(a)
            eig = self.oracle_filter.compute_exact_eig(interv)
            if eig > max_eig:
                max_eig = eig
                best_a = a

        return best_a

    def observe_transition(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray
    ) -> Dict[str, Any]:
        interv = self.action_decoder_fn(action)
        weights = self.oracle_filter.update(interv, next_state)
        return {
            "entropy": self.oracle_filter.get_shannon_entropy(),
            "weights": weights,
            "map_hypothesis": np.argmax(weights)
        }
