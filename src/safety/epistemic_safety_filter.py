"""Epistemic Safety Filter implementing Lower Confidence Bound (LCB) barrier constraints.

Evaluates safety under candidate model hypotheses, penalizing epistemic uncertainty
to guarantee safe exploration under ambiguous dynamics.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from src.safety.barrier_function import ControlBarrierFunction


class EpistemicSafetyFilter:
    """Filters candidate actions using epistemic Lower Confidence Bound on barrier functions."""

    def __init__(
        self,
        barriers: List[ControlBarrierFunction],
        risk_tolerance_delta: float = 0.05,
        default_alpha: float = 0.5
    ):
        self.barriers = barriers
        self.delta = float(np.clip(risk_tolerance_delta, 1e-4, 0.5))
        self.default_alpha = default_alpha

        # Cantelli multiplier: κ_δ = sqrt((1 - δ) / δ)
        self.kappa = float(np.sqrt((1.0 - self.delta) / self.delta))

    def evaluate_action_safety(
        self,
        current_state: np.ndarray,
        predicted_next_states_across_hypotheses: np.ndarray,
        hypothesis_weights: Optional[np.ndarray] = None
    ) -> Tuple[bool, float, Dict[str, float]]:
        """Evaluates whether an action is provably safe under candidate models.
        
        Args:
            current_state: State vector s.
            predicted_next_states_across_hypotheses: Array [K, d] of predictions ŝ'_k.
            hypothesis_weights: Array [K] of posterior weights w_k (sum to 1).
            
        Returns:
            Tuple of:
                - is_safe (boolean flag)
                - min_lcb_margin (scalar margin >= 0 means safe)
                - barrier_details (dict of barrier diagnostics)
        """
        preds = np.asarray(predicted_next_states_across_hypotheses, dtype=np.float64)
        K = preds.shape[0]

        if hypothesis_weights is not None:
            w = np.asarray(hypothesis_weights, dtype=np.float64)
            w = w / np.sum(w)
        else:
            w = np.full(K, 1.0 / K, dtype=np.float64)

        all_safe = True
        min_margin = np.inf
        details = {}

        for barrier in self.barriers:
            h_curr = barrier.evaluate(current_state)
            alpha = barrier.alpha

            # Evaluate barrier values across all candidate model outcomes: h(ŝ'_k)
            h_preds = np.array([barrier.evaluate(preds[k]) for k in range(K)], dtype=np.float64)

            # Expected barrier value: μ_h = ∑_k w_k h(ŝ'_k)
            mu_h = float(np.sum(w * h_preds))

            # Epistemic variance of barrier: σ_h^2 = ∑_k w_k (h_k - μ_h)^2
            var_h = float(np.sum(w * ((h_preds - mu_h) ** 2)))
            sigma_h = float(np.sqrt(max(0.0, var_h)))

            # Conservative Lower Confidence Bound: LCB_h = μ_h - κ_δ * σ_h
            lcb_h = mu_h - self.kappa * sigma_h

            # Discrete-time CBF condition: LCB_h >= (1 - α) * h(s_curr)
            required_h = (1.0 - alpha) * h_curr
            margin = lcb_h - required_h

            details[barrier.name] = {
                "mu_h": mu_h,
                "sigma_h": sigma_h,
                "lcb_h": lcb_h,
                "required_h": required_h,
                "margin": margin,
                "is_safe": bool(margin >= 0.0)
            }

            if margin < 0.0:
                all_safe = False

            if margin < min_margin:
                min_margin = margin

        return all_safe, float(min_margin), details

    def filter_safe_actions(
        self,
        current_state: np.ndarray,
        candidate_actions: List[np.ndarray],
        rollout_hypotheses_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        hypothesis_weights: Optional[np.ndarray] = None
    ) -> Tuple[List[int], List[float]]:
        """Filters a candidate pool of actions, returning safe indices and their LCB margins."""
        safe_indices = []
        lcb_margins = []

        for idx, a in enumerate(candidate_actions):
            preds = rollout_hypotheses_fn(current_state, a)
            is_safe, margin, _ = self.evaluate_action_safety(
                current_state, preds, hypothesis_weights
            )
            lcb_margins.append(margin)
            if is_safe:
                safe_indices.append(idx)

        return safe_indices, lcb_margins
