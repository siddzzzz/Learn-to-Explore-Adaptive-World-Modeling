"""Differentiable Causal World Model integrating graph posterior with neural mechanisms.

Unifies variational graph induction q_ψ(G) and neural SCM dynamics
for active interventional world modeling.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.models.causal.graph_posterior import VariationalGraphPosterior
from src.models.causal.neural_scm import NeuralSCM
from src.models.causal.dag_utils import is_acyclic, threshold_graph


class DifferentiableCausalWorldModel(nn.Module):
    """End-to-End Differentiable Causal World Model."""

    def __init__(
        self,
        num_variables: int,
        action_dim: int,
        hidden_dim: int = 64,
        initial_edge_prior: float = 0.25,
        device: Optional[str] = None
    ):
        super().__init__()
        self.num_variables = num_variables
        self.action_dim = action_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # 1. Variational distribution over DAG space q_ψ(G)
        self.graph_posterior = VariationalGraphPosterior(num_variables, initial_prior=initial_edge_prior)

        # 2. Neural Structural Causal Model mechanisms F_φ
        self.neural_scm = NeuralSCM(num_variables, action_dim, hidden_dim)

        self.to(self.device)

    def sample_candidate_hypotheses(self, num_hypotheses: int = 8) -> torch.Tensor:
        """Samples K candidate DAG hypothesis particles A^(1), ..., A^(K) ~ q_ψ(G)."""
        return self.graph_posterior.sample_discrete(num_hypotheses)

    def predict_under_hypotheses(
        self,
        state: np.ndarray,
        action: np.ndarray,
        candidate_graphs: torch.Tensor,
        interventions: Optional[Dict[int, float]] = None
    ) -> np.ndarray:
        """Evaluates predicted next states across candidate DAG hypotheses.
        
        Args:
            state: Array of shape [d] or [B, d].
            action: Array of shape [action_dim] or [B, action_dim].
            candidate_graphs: Tensor of shape [K, d, d].
            interventions: Pearl do-operator forced values.
            
        Returns:
            Numpy array of shape [K, d] of predictions under the K candidate graphs.
        """
        s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        graphs = candidate_graphs.to(self.device)
        K = graphs.shape[0]

        predictions = []
        with torch.no_grad():
            for k in range(K):
                pred = self.neural_scm(s_t, a_t, graphs[k], interventions=interventions)
                predictions.append(pred)

        stacked = torch.stack(predictions, dim=0).cpu().numpy()
        return stacked

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        interventions: Optional[Dict[int, float]] = None,
        temperature: float = 0.5,
        acyclicity_weight: float = 2.0,
        sparsity_weight: float = 0.05
    ) -> Dict[str, torch.Tensor]:
        """Computes the differentiable training objective:
        
        L = Reconstruction MSE + λ_dag * h(P) + λ_sparse * ||P||_1
        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        B = states.shape[0]

        # 1. Differentiable relaxed graph sample A_tilde ~ q_ψ(G)
        relaxed_dag = self.graph_posterior.sample_gumbel(batch_size=B, temperature=temperature)

        # 2. Forward rollout through neural mechanisms
        pred_next_states = self.neural_scm(states, actions, relaxed_dag, interventions=interventions)

        # 3. Prediction reconstruction loss
        recon_loss = nn.functional.mse_loss(pred_next_states, next_states)

        # 4. NOTEARS continuous acyclicity constraint penalty
        edge_probs = self.graph_posterior.get_edge_probabilities()
        acyclicity_penalty = self.graph_posterior.get_acyclicity_penalty()

        # 5. L1 edge sparsity regularization
        sparsity_penalty = edge_probs.sum()

        total_loss = (
            recon_loss
            + acyclicity_weight * acyclicity_penalty
            + sparsity_weight * sparsity_penalty
        )

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "acyclicity_penalty": acyclicity_penalty,
            "sparsity_penalty": sparsity_penalty,
            "entropy": self.graph_posterior.get_shannon_entropy()
        }

    def get_most_probable_dag(self, threshold: float = 0.4) -> np.ndarray:
        """Returns the MAP estimated binary DAG adjacency matrix."""
        with torch.no_grad():
            probs = self.graph_posterior.get_edge_probabilities()
            binary_dag = threshold_graph(probs, threshold=threshold)
            return binary_dag.cpu().numpy().astype(int)
