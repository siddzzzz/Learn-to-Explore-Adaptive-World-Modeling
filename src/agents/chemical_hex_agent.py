"""Chemical HEx-WM Agent for Automated Scientific Discovery in Chemical Reactors.

Designs continuous reactor experiments to discriminate competing ODE kinetic mechanisms,
then zero-shot optimizes chemical synthesis for arbitrary target specifications.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.envs.chemical_reactor.kinetics import ChemicalKineticsModel
from src.models.belief_tracker import BayesianBeliefTracker
from src.planning.acquisition import compute_chd


class ChemicalHExAgent:
    """Hypothesis-Experimenting World Model agent for chemical discovery."""

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        noise_variance: float = 0.0004,
        num_candidates: int = 500,
        rng: Optional[np.random.RandomState] = None
    ):
        self.bounds = bounds
        self.noise_variance = noise_variance
        self.num_candidates = num_candidates
        self.rng = rng or np.random.RandomState(42)

        # 4 Mechanistic ODE models
        self.models = [ChemicalKineticsModel(h_id) for h_id in range(4)]
        self.num_hypotheses = len(self.models)

        # Bayesian belief over mechanisms
        self.belief_tracker = BayesianBeliefTracker(num_hypotheses=self.num_hypotheses)
        self.is_confident = False
        self.target_goal: Optional[Dict[str, float]] = None

    def reset(self) -> None:
        self.belief_tracker.reset()
        self.is_confident = False
        self.target_goal = None

    def denormalize(self, action: np.ndarray) -> Tuple[float, float, float, float]:
        """Converts [-1, 1]^4 to (T, C_A0, C_B0, duration)."""
        act = np.clip(action, -1.0, 1.0)
        t = float(self.bounds["T"][0] + 0.5 * (act[0] + 1.0) * (self.bounds["T"][1] - self.bounds["T"][0]))
        ca = float(self.bounds["C_A0"][0] + 0.5 * (act[1] + 1.0) * (self.bounds["C_A0"][1] - self.bounds["C_A0"][0]))
        cb = float(self.bounds["C_B0"][0] + 0.5 * (act[2] + 1.0) * (self.bounds["C_B0"][1] - self.bounds["C_B0"][0]))
        dur = float(self.bounds["duration"][0] + 0.5 * (act[3] + 1.0) * (self.bounds["duration"][1] - self.bounds["duration"][0]))
        return t, ca, cb, dur

    def select_action(self) -> np.ndarray:
        """Selects optimal experiment (exploration) or synthesis recipe (exploitation)."""
        entropy = self.belief_tracker.get_shannon_entropy()
        self.is_confident = entropy < 0.15

        weights = self.belief_tracker.weights
        candidates = self.rng.uniform(-1.0, 1.0, size=(self.num_candidates, 4)).astype(np.float32)

        if self.is_confident and self.target_goal is not None:
            # Exploitation: Optimize continuous conditions using the discovered MAP kinetic model
            map_idx, _ = self.belief_tracker.get_map_hypothesis()
            best_model = self.models[map_idx]

            best_action = candidates[0]
            best_score = -1e9
            max_T = self.target_goal.get("max_T", 450.0)

            for cand in candidates:
                t, ca, cb, dur = self.denormalize(cand)
                concs = best_model.simulate(t, ca, cb, dur)
                cP = concs[2]
                cW = concs[3]
                
                score = cP - 3.0 * cW
                if t > max_T:
                    score -= 5.0
                if score > best_score:
                    best_score = score
                    best_action = cand

            return best_action

        else:
            # Epistemic Discovery: Maximize Contrastive Hypothesis Discrepancy across candidate ODEs
            best_action = candidates[0]
            max_chd = -1.0

            for cand in candidates:
                t, ca, cb, dur = self.denormalize(cand)
                preds = np.array([m.simulate(t, ca, cb, dur) for m in self.models])
                chd = compute_chd(preds, weights, noise_variance=self.noise_variance)

                if chd > max_chd:
                    max_chd = chd
                    best_action = cand

            return best_action

    def observe_transition(self, action: np.ndarray, measured_concs: np.ndarray) -> Dict[str, Any]:
        """Performs Bayesian update using continuous ODE likelihoods."""
        t, ca, cb, dur = self.denormalize(action)

        log_liks = np.zeros(self.num_hypotheses, dtype=np.float64)
        var = self.noise_variance
        d = len(measured_concs)

        for k, model in enumerate(self.models):
            mu_k = model.simulate(t, ca, cb, dur)
            sse = float(np.sum((measured_concs - mu_k) ** 2))
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
