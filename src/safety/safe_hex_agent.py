"""Safe HEx-WM Agent (Safe-BED).

Combines Contrastive Hypothesis Discrepancy active learning with Control Barrier Functions,
ensuring provably safe epistemic exploration in high-risk continuous environments.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from src.models.belief_tracker import BayesianBeliefTracker
from src.planning.acquisition import compute_chd
from src.safety.epistemic_safety_filter import EpistemicSafetyFilter
from src.safety.safe_trajectory_cem import SafeTrajectoryCEMPlanner


class SafeHExWMAgent:
    """Safety-Constrained Hypothesis-Experimenting World Model Agent."""

    def __init__(
        self,
        action_dim: int,
        hypothesis_models: List[Any],
        safety_filter: EpistemicSafetyFilter,
        denormalize_action_fn: Callable[[np.ndarray], Tuple[float, ...]],
        bounds: Dict[str, Tuple[float, float]],
        noise_variance: float = 0.0004,
        planning_horizon: int = 3,
        num_candidates: int = 150,
        rng: Optional[np.random.RandomState] = None
    ):
        self.action_dim = action_dim
        self.hypothesis_models = hypothesis_models
        self.K = len(hypothesis_models)
        self.safety_filter = safety_filter
        self.denormalize_action_fn = denormalize_action_fn
        self.bounds = bounds
        self.noise_variance = noise_variance
        self.planning_horizon = planning_horizon
        self.num_candidates = num_candidates
        self.rng = rng or np.random.RandomState(42)

        # Belief Tracker over candidate hypotheses
        self.belief_tracker = BayesianBeliefTracker(num_hypotheses=self.K)

        # Safe CEM Planner
        self.safe_planner = SafeTrajectoryCEMPlanner(
            action_dim=action_dim,
            safety_filter=safety_filter,
            horizon=planning_horizon,
            num_samples=64,
            num_elites=12,
            rng=self.rng
        )

        self.target_goal: Optional[Dict[str, float]] = None
        self.is_confident = False
        self.history_actions: List[np.ndarray] = []
        self.history_safety_margins: List[float] = []

    def reset(self) -> None:
        self.belief_tracker.reset()
        self.is_confident = False
        self.target_goal = None
        self.history_actions = []
        self.history_safety_margins = []

    def _rollout_hypotheses(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Evaluates predicted next state across all K candidate models."""
        params = self.denormalize_action_fn(action)
        # Expected signature: simulate(T, Ca, Cb, dur)
        preds = []
        for model in self.hypothesis_models:
            out = model.simulate(*params)
            preds.append(out)
        return np.array(preds)

    def select_action(self, current_state: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """Selects a safe, information-maximizing or goal-directed experimental action."""
        entropy = self.belief_tracker.get_shannon_entropy()
        self.is_confident = entropy < 0.15
        weights = self.belief_tracker.weights

        # 1. Sample candidate continuous action pool in [-1, 1]^d
        candidates = self.rng.uniform(-1.0, 1.0, size=(self.num_candidates, self.action_dim)).astype(np.float32)

        safe_candidates = []
        cgd_scores = []
        margins = []

        # 2. Filter actions through Epistemic Control Barrier Function
        for cand in candidates:
            preds = self._rollout_hypotheses(current_state, cand)
            is_safe, margin, _ = self.safety_filter.evaluate_action_safety(
                current_state, preds, weights
            )
            margins.append(margin)

            if is_safe:
                safe_candidates.append(cand)
                # Compute epistemic information gain
                info_score = compute_chd(preds, weights, noise_variance=self.noise_variance)
                cgd_scores.append(info_score)

        # 3. Decision rule:
        if safe_candidates:
            # Maximally informative action among SAFE actions
            best_idx = int(np.argmax(cgd_scores))
            chosen_action = safe_candidates[best_idx]
            chosen_margin = margins[best_idx]
            is_provably_safe = True
        else:
            # Fallback: Minimum-risk recovery action maximizing barrier margin
            best_margin_idx = int(np.argmax(margins))
            chosen_action = candidates[best_margin_idx]
            chosen_margin = margins[best_margin_idx]
            is_provably_safe = False

        self.history_actions.append(chosen_action.copy())
        self.history_safety_margins.append(chosen_margin)
        return chosen_action, is_provably_safe, chosen_margin

    def observe_transition(
        self,
        action: np.ndarray,
        measured_state: np.ndarray
    ) -> Dict[str, Any]:
        """Performs Bayesian belief update under continuous likelihood."""
        params = self.denormalize_action_fn(action)
        d = len(measured_state)
        log_liks = np.zeros(self.K, dtype=np.float64)
        var = self.noise_variance

        for k, model in enumerate(self.hypothesis_models):
            mu_k = model.simulate(*params)
            sse = float(np.sum((measured_state - mu_k) ** 2))
            log_liks[k] = -0.5 * (d * np.log(2.0 * np.pi * var) + sse / var)

        weights, surprise, is_anomaly = self.belief_tracker.update_from_log_likelihoods(log_liks)
        map_idx, map_prob = self.belief_tracker.get_map_hypothesis()

        return {
            "entropy": self.belief_tracker.get_shannon_entropy(),
            "weights": weights,
            "map_hypothesis": map_idx,
            "map_confidence": map_prob,
            "surprise": surprise
        }
