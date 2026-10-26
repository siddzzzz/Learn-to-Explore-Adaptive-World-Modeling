"""Variational Graph Posterior Distribution for Differentiable Causal Learning.

Maintains a variational distribution q_ψ(G) over directed graphs using
Gumbel-Softmax continuous relaxation and discrete Bernoulli sampling.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.models.causal.dag_utils import compute_acyclicity_constraint


class VariationalGraphPosterior(nn.Module):
    """Variational distribution q_ψ(G) over causal graph adjacency matrices."""

    def __init__(self, num_variables: int, initial_prior: float = 0.2):
        super().__init__()
        self.num_variables = num_variables
        d = num_variables

        # Initialize variational edge logits: logit(prior)
        prior_clipped = np.clip(initial_prior, 1e-4, 1.0 - 1e-4)
        init_logit = np.log(prior_clipped / (1.0 - prior_clipped))
        
        # Logit parameter tensor [d, d]
        self.edge_logits = nn.Parameter(torch.full((d, d), float(init_logit), dtype=torch.float32))

        # Self-loop mask (diagonal is strictly zero)
        self.register_buffer("off_diagonal_mask", 1.0 - torch.eye(d, dtype=torch.float32))

    def get_logits(self) -> torch.Tensor:
        """Returns masked logits where self-loops are clamped to -1e9."""
        return self.edge_logits * self.off_diagonal_mask - 1e9 * (1.0 - self.off_diagonal_mask)

    def get_edge_probabilities(self) -> torch.Tensor:
        """Returns marginal edge existence probabilities P(X_i -> X_j) ∈ [0, 1]."""
        logits = self.get_logits()
        probs = torch.sigmoid(logits) * self.off_diagonal_mask
        return probs

    def sample_gumbel(self, batch_size: int = 1, temperature: float = 0.5) -> torch.Tensor:
        """Draws differentiable relaxed graph samples using the Gumbel-Softmax trick.
        
        Args:
            batch_size: Number of graph samples to generate.
            temperature: Gumbel-Softmax temperature parameter τ > 0.
            
        Returns:
            Continuous soft adjacency matrices of shape [batch_size, d, d] in (0, 1).
        """
        logits = self.get_logits().unsqueeze(0).expand(batch_size, -1, -1)
        
        # Sample Gumbel perturbations: g = -log(-log(u))
        u = torch.rand_like(logits).clamp(1e-7, 1.0 - 1e-7)
        g1 = -torch.log(-torch.log(u))
        
        u0 = torch.rand_like(logits).clamp(1e-7, 1.0 - 1e-7)
        g0 = -torch.log(-torch.log(u0))

        # Continuous relaxation: σ( (logit + g1 - g0) / τ )
        relaxed = torch.sigmoid((logits + g1 - g0) / max(1e-4, temperature))
        return relaxed * self.off_diagonal_mask.unsqueeze(0)

    def sample_discrete(self, num_samples: int = 1) -> torch.Tensor:
        """Draws discrete binary DAG samples A^(k) ~ Bernoulli(σ(Ψ)).
        
        Returns:
            Binary adjacency matrices of shape [num_samples, d, d].
        """
        probs = self.get_edge_probabilities().unsqueeze(0).expand(num_samples, -1, -1)
        samples = torch.bernoulli(probs)
        return samples * self.off_diagonal_mask.unsqueeze(0)

    def get_shannon_entropy(self) -> torch.Tensor:
        """Computes discrete Shannon entropy H(q_ψ(G)) = ∑_{i≠j} H(Bernoulli(p_{ij})).
        
        Represents total structural epistemic uncertainty over causal graphs.
        """
        probs = self.get_edge_probabilities().clamp(1e-7, 1.0 - 1e-7)
        ent = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs))
        masked_ent = ent * self.off_diagonal_mask
        return masked_ent.sum()

    def get_acyclicity_penalty(self) -> torch.Tensor:
        """Evaluates NOTEARS constraint h(P) = Tr(exp(P ∘ P)) - d on edge probabilities."""
        probs = self.get_edge_probabilities()
        return compute_acyclicity_constraint(probs)
