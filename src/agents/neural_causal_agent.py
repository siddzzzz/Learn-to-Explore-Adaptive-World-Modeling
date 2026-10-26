"""End-to-End Neural Causal World Model Agent.

Discovers causal graph structures and non-linear mechanisms from scratch without
predefined hypothesis menus, using Contrastive Graph Discrepancy (CGD) to actively
break observational Markov Equivalence Classes.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from src.models.causal.causal_world_model import DifferentiableCausalWorldModel
from src.planning.causal_acquisition import compute_contrastive_graph_discrepancy


class NeuralCausalAgent:
    """Agent that learns causal world models end-to-end via active experimental design."""

    def __init__(
        self,
        num_variables: int,
        action_dim: int,
        action_decoder_fn: Any,
        num_actions: int,
        num_graph_particles: int = 8,
        hidden_dim: int = 64,
        noise_variance: float = 0.0025,
        rng: Optional[np.random.RandomState] = None,
        device: Optional[str] = None
    ):
        self.num_variables = num_variables
        self.action_dim = action_dim
        self.action_decoder_fn = action_decoder_fn
        self.num_actions = num_actions
        self.num_graph_particles = num_graph_particles
        self.noise_variance = noise_variance
        self.rng = rng or np.random.RandomState(42)

        # End-to-end differentiable causal world model
        self.world_model = DifferentiableCausalWorldModel(
            num_variables=num_variables,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=device
        )

        # Candidate discrete DAG particles currently active
        self.candidate_dags = self.world_model.sample_candidate_hypotheses(num_graph_particles)
        self.dag_weights = np.full(num_graph_particles, 1.0 / num_graph_particles, dtype=np.float64)

        # History tracking
        self.history_actions: List[int] = []
        self.history_graph_entropy: List[float] = [float(self.world_model.graph_posterior.get_shannon_entropy().item())]
        self.target_goal: Optional[np.ndarray] = None
        self.is_confident = False

    def reset(self) -> None:
        """Resamples candidate graph hypotheses and resets internal state."""
        self.candidate_dags = self.world_model.sample_candidate_hypotheses(self.num_graph_particles)
        self.dag_weights = np.full(self.num_graph_particles, 1.0 / self.num_graph_particles, dtype=np.float64)
        self.history_actions = []
        self.history_graph_entropy = [float(self.world_model.graph_posterior.get_shannon_entropy().item())]
        self.target_goal = None
        self.is_confident = False

    def select_action(self, current_state: np.ndarray) -> int:
        """Selects optimal experiment (active discovery) or goal-directed intervention (exploitation)."""
        graph_entropy = float(self.world_model.graph_posterior.get_shannon_entropy().item())
        self.is_confident = graph_entropy < 0.2

        if self.is_confident and self.target_goal is not None:
            # Exploitation: Plan toward goal using MAP estimated causal DAG
            map_dag = torch.from_numpy(self.world_model.get_most_probable_dag()).float().unsqueeze(0)
            best_action = 0
            best_mse = 1e9

            for a in range(self.num_actions):
                interv = self.action_decoder_fn(a)
                a_vec = np.zeros(self.action_dim, dtype=np.float32)
                pred = self.world_model.predict_under_hypotheses(
                    current_state, a_vec, map_dag, interventions=interv
                )[0]
                mse = float(np.sum((pred[:len(self.target_goal)] - self.target_goal) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_action = a

            self.history_actions.append(best_action)
            return best_action

        else:
            # Epistemic Discovery: Maximize Contrastive Graph Discrepancy across candidate DAGs
            best_action = 0
            max_cgd = -1.0

            for a in range(self.num_actions):
                interv = self.action_decoder_fn(a)
                a_vec = np.zeros(self.action_dim, dtype=np.float32)
                # Predict outcomes under all K candidate DAG structures
                preds = self.world_model.predict_under_hypotheses(
                    current_state, a_vec, self.candidate_dags, interventions=interv
                )
                cgd = compute_contrastive_graph_discrepancy(
                    preds, weights=self.dag_weights, noise_variance=self.noise_variance
                )

                if cgd > max_cgd:
                    max_cgd = cgd
                    best_action = a

            self.history_actions.append(best_action)
            return best_action

    def observe_transition(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray
    ) -> Dict[str, Any]:
        """Performs Bayesian update over candidate DAG structures using transition likelihoods."""
        interv = self.action_decoder_fn(action)
        a_vec = np.zeros(self.action_dim, dtype=np.float32)

        preds = self.world_model.predict_under_hypotheses(
            state, a_vec, self.candidate_dags, interventions=interv
        )

        K = len(self.dag_weights)
        d = len(next_state)
        log_liks = np.zeros(K, dtype=np.float64)
        var = self.noise_variance

        for k in range(K):
            mu_k = preds[k]
            sse = float(np.sum((next_state[:len(mu_k)] - mu_k) ** 2))
            log_liks[k] = -0.5 * (len(mu_k) * np.log(2.0 * np.pi * var) + sse / var)

        # Numerically stable Bayes update
        max_ll = np.max(log_liks)
        unnorm = np.exp(log_liks - max_ll) * self.dag_weights
        self.dag_weights = unnorm / max(1e-12, np.sum(unnorm))

        entropy = float(self.world_model.graph_posterior.get_shannon_entropy().item())
        self.history_graph_entropy.append(entropy)
        map_dag_idx = int(np.argmax(self.dag_weights))

        return {
            "graph_entropy": entropy,
            "map_dag_idx": map_dag_idx,
            "map_confidence": float(self.dag_weights[map_dag_idx]),
            "dag_weights": self.dag_weights.copy()
        }
