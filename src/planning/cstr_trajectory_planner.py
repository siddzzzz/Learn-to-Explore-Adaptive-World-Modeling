"""Dynamic Trajectory Safe-CEM Controller for Closed-Loop Continuous CSTR Systems.

Plans continuous multi-step cooling flow sequences u_{t:t+H} to maximize
active hypothesis information gain while strictly enforcing Cantelli Lower Confidence Bound
on the continuous temperature Control Barrier Function h(s) = T_max - T >= 0.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from src.models.belief_tracker import BayesianBeliefTracker
from src.planning.acquisition import compute_chd
from src.envs.chemical_reactor.cstr_kinetics import CSTRDynamicsModel
from src.safety.barrier_function import ControlBarrierFunction
from src.safety.epistemic_safety_filter import EpistemicSafetyFilter


class CSTRDynamicTrajectoryPlanner:
    """Non-myopic trajectory planner for dynamic continuous CSTR under safety constraints."""

    def __init__(
        self,
        horizon: int = 5,
        num_candidates: int = 50,
        elite_ratio: float = 0.2,
        num_iterations: int = 3,
        dt: float = 0.05,
        max_temperature: float = 410.0,
        risk_delta: float = 0.05,
        rng: Optional[np.random.RandomState] = None
    ):
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.num_elites = max(2, int(num_candidates * elite_ratio))
        self.num_iterations = num_iterations
        self.dt = dt
        self.max_T = max_temperature
        self.risk_delta = risk_delta
        self.rng = rng or np.random.RandomState(42)

        # Candidate kinetic models
        self.models = [CSTRDynamicsModel(i) for i in range(4)]
        self.K = len(self.models)
        self.belief_tracker = BayesianBeliefTracker(num_hypotheses=self.K)

        # Cantelli multiplier
        self.kappa = float(np.sqrt((1.0 - risk_delta) / risk_delta))

    def reset(self) -> None:
        self.belief_tracker.reset()

    def select_safe_action(
        self,
        current_state: np.ndarray,
        use_safety_barrier: bool = True
    ) -> Tuple[np.ndarray, bool, float]:
        """Plans optimal coolant flow rate u_t using Safe-CEM trajectory optimization."""
        weights = self.belief_tracker.weights
        s0 = np.asarray(current_state, dtype=np.float64)

        # Initialize Gaussian over action trajectory in [-1, 1]^H
        mean_u = np.zeros(self.horizon, dtype=np.float64)
        std_u = np.ones(self.horizon, dtype=np.float64) * 0.5

        best_trajectory = mean_u.copy()
        best_score = -np.inf
        best_margin = 0.0
        is_safe_plan = False

        for iteration in range(self.num_iterations):
            # Sample trajectory candidates
            samples = self.rng.normal(mean_u, std_u, size=(self.num_candidates, self.horizon))
            samples = np.clip(samples, -1.0, 1.0)

            scores = []
            margins = []
            safe_flags = []

            for traj in samples:
                score = 0.0
                traj_margin = np.inf
                traj_is_safe = True
                
                # Rollout forward in time across candidate models
                # State particles for each model: shape (K, 3)
                sim_states = np.tile(s0, (self.K, 1))

                for t in range(self.horizon):
                    # Action to physical coolant flow qc
                    u_t = traj[t]
                    qc_t = 50.0 + 50.0 * u_t  # Maps [-1, 1] to [0, 100] L/min

                    # Step each model
                    next_sim_states = np.zeros_like(sim_states)
                    for k in range(self.K):
                        next_sim_states[k] = self.models[k].step_rk4(sim_states[k], qc=qc_t, dt=self.dt)

                    # Dynamic CBF: h_k = T_max - T_k
                    h_preds = self.max_T - next_sim_states[:, 1]
                    mu_h = float(np.sum(weights * h_preds))
                    var_h = float(np.sum(weights * ((h_preds - mu_h) ** 2)))
                    sigma_h = float(np.sqrt(max(0.0, var_h)))

                    # Cantelli LCB
                    lcb_h = mu_h - self.kappa * sigma_h
                    traj_margin = min(traj_margin, lcb_h)

                    # Safety check: LCB must be positive
                    if use_safety_barrier and (lcb_h < 0.0):
                        traj_is_safe = False

                    # Active Information Gain (Contrastive Discrepancy on next temperature)
                    temp_preds = next_sim_states[:, 1:2]
                    chd_val = compute_chd(temp_preds, weights, noise_variance=0.04)
                    score += (0.95 ** t) * chd_val

                    sim_states = next_sim_states

                # Heavily penalize unsafe trajectories
                if use_safety_barrier and not traj_is_safe:
                    score -= 1000.0 + 10.0 * max(0.0, -traj_margin)

                scores.append(score)
                margins.append(traj_margin)
                safe_flags.append(traj_is_safe)

            scores = np.array(scores)
            elite_indices = np.argsort(scores)[-self.num_elites:]

            # Update sampling distribution
            mean_u = np.mean(samples[elite_indices], axis=0)
            std_u = np.std(samples[elite_indices], axis=0) + 1e-4

            if scores[elite_indices[-1]] > best_score:
                best_score = scores[elite_indices[-1]]
                best_trajectory = samples[elite_indices[-1]]
                best_margin = margins[elite_indices[-1]]
                is_safe_plan = safe_flags[elite_indices[-1]]

        first_action = np.array([best_trajectory[0]], dtype=np.float32)
        return first_action, is_safe_plan, best_margin

    def observe_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray
    ) -> Dict[str, Any]:
        """Updates Bayesian belief distribution over candidate CSTR models."""
        u = float(action[0] if hasattr(action, '__len__') else action)
        qc = 50.0 + 50.0 * u
        s_obs = np.asarray(next_state, dtype=np.float64)

        log_liks = np.zeros(self.K, dtype=np.float64)
        var = 0.04

        for k in range(self.K):
            pred_s = self.models[k].step_rk4(state, qc=qc, dt=self.dt)
            # Evaluate log-likelihood on temperature (index 1)
            diff = s_obs[1] - pred_s[1]
            log_liks[k] = -0.5 * (np.log(2.0 * np.pi * var) + (diff ** 2) / var)

        weights, surprise, is_anomaly = self.belief_tracker.update_from_log_likelihoods(log_liks)
        map_idx, map_prob = self.belief_tracker.get_map_hypothesis()

        return {
            "entropy": self.belief_tracker.get_shannon_entropy(),
            "weights": weights,
            "map_hypothesis": map_idx,
            "map_prob": map_prob,
        }
