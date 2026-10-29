"""HEx-WM: Hypothesis-Experimenting World Model Agent.

Combines Bayesian belief tracking, Contrastive Hypothesis Discrepancy (CHD)
acquisition, non-myopic trajectory planning, and adaptive rule-shift recovery.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.models.belief_tracker import BayesianBeliefTracker
from src.planning.acquisition import compute_chd
from src.planning.trajectory_cem import TrajectoryCEMPlanner


class HExWMAgent:
    """Hypothesis-Experimenting World Model Agent (Algorithm 1)."""

    def __init__(
        self,
        hypothesis_evaluators: List[Any],
        num_actions: int,
        action_decoder_fn: Any,
        noise_variance: float = 0.0025,
        entropy_threshold: float = 0.05,
        planning_horizon: int = 3,
        num_cem_samples: int = 64,
        num_cem_elites: int = 12,
        num_cem_iters: int = 5,
        surprise_threshold: float = 20.0,
        rng: Optional[np.random.RandomState] = None
    ):
        self.hypothesis_evaluators = hypothesis_evaluators
        self.K = len(hypothesis_evaluators)
        self.num_actions = num_actions
        self.action_decoder_fn = action_decoder_fn
        self.noise_variance = noise_variance
        self.entropy_threshold = entropy_threshold
        self.planning_horizon = planning_horizon
        self.rng = rng or np.random.RandomState(42)

        # Belief Tracker
        self.belief_tracker = BayesianBeliefTracker(
            num_hypotheses=self.K,
            surprise_threshold=surprise_threshold
        )

        # CEM Trajectory Planner
        self.planner = TrajectoryCEMPlanner(
            action_dim=num_actions,
            is_discrete=True,
            horizon=planning_horizon,
            num_samples=num_cem_samples,
            num_elites=num_cem_elites,
            num_iterations=num_cem_iters,
            rng=self.rng
        )

        # Agent state
        self.is_confident = False
        self.goal_state: Optional[np.ndarray] = None
        self.history_actions: List[int] = []
        self.history_beliefs: List[np.ndarray] = []

    def reset(self) -> None:
        """Resets agent belief and internal state."""
        self.belief_tracker.reset()
        self.is_confident = False
        self.goal_state = None
        self.history_actions = []
        self.history_beliefs = [self.belief_tracker.weights.copy()]

    def set_goal(self, goal_state: np.ndarray) -> None:
        """Sets target downstream goal for exploitation phase."""
        self.goal_state = np.asarray(goal_state, dtype=np.float32)

    def select_action(self, current_state: np.ndarray) -> int:
        """Selects optimal experiment (exploration) or optimal task action (exploitation)."""
        current_entropy = self.belief_tracker.get_shannon_entropy()
        self.is_confident = current_entropy < self.entropy_threshold

        weights = self.belief_tracker.weights

        if self.is_confident and self.goal_state is not None:
            # Exploitation Phase: Plan to reach goal using MAP hypothesis model
            map_idx, _ = self.belief_tracker.get_map_hypothesis()
            best_model = self.hypothesis_evaluators[map_idx]

            def goal_rollout(s: np.ndarray, a: int) -> Tuple[np.ndarray, float]:
                interv = self.action_decoder_fn(a)
                next_s = best_model.evaluate(interventions=interv)
                reward = -float(np.sum((next_s[:len(self.goal_state)] - self.goal_state) ** 2))
                return next_s, reward

            action, _, _ = self.planner.plan_discrete(current_state, goal_rollout)
        else:
            # Epistemic Discovery Phase: Plan trajectory to maximize Contrastive Hypothesis Discrepancy
            def epistemic_rollout(s: np.ndarray, a: int) -> Tuple[np.ndarray, float]:
                interv = self.action_decoder_fn(a)
                # Compute predictions under all candidate models
                preds = [model.evaluate(interventions=interv) for model in self.hypothesis_evaluators]
                preds_arr = np.array(preds)

                # Compute Contrastive Hypothesis Discrepancy (Jensen-Shannon divergence)
                r_chd = compute_chd(preds_arr, weights, noise_variance=self.noise_variance)

                # Small intervention effort regularization so uninformative distractor actions (r_chd = 0) are not arbitrarily sampled
                reward = r_chd - 1e-4 * float(a)

                # Imagined next state is expectation under current belief
                expected_next_s = np.sum(weights[:, np.newaxis] * preds_arr, axis=0)
                return expected_next_s, reward

            action, _, _ = self.planner.plan_discrete(current_state, epistemic_rollout)

        self.history_actions.append(action)
        return action

    def observe_transition(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray
    ) -> Dict[str, Any]:
        """Performs Bayesian update from real transition (s, a, s')."""
        interv = self.action_decoder_fn(action)
        d = len(next_state)

        # Compute log-likelihood under each hypothesis model
        log_liks = np.zeros(self.K, dtype=np.float64)
        for k, model in enumerate(self.hypothesis_evaluators):
            mu_k = model.evaluate(interventions=interv)
            diff = next_state[:len(mu_k)] - mu_k
            sse = float(np.sum(diff ** 2))
            var = self.noise_variance
            log_liks[k] = -0.5 * (len(mu_k) * np.log(2 * np.pi * var) + sse / var)

        weights, surprise, is_anomaly = self.belief_tracker.update_from_log_likelihoods(log_liks)
        self.history_beliefs.append(weights.copy())

        # Anomaly Recovery: If observation is catastrophic surprise, broaden belief
        if is_anomaly:
            self.belief_tracker.broaden_belief(mixing_factor=0.4)
            self.is_confident = False

        map_idx, map_prob = self.belief_tracker.get_map_hypothesis()
        entropy = self.belief_tracker.get_shannon_entropy()

        return {
            "entropy": entropy,
            "weights": weights,
            "map_hypothesis": map_idx,
            "map_confidence": map_prob,
            "surprise": surprise,
            "is_anomaly": is_anomaly
        }
