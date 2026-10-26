"""Modular Neural Structural Causal Model (Neural SCM).

Implements independent neural mechanism networks f_i(PA_i, a; φ_i) with
Pearl's do-calculus interventional edge mutilation.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class VariableMechanism(nn.Module):
    """Neural mechanism f_i computing state transition for a single causal variable."""

    def __init__(self, num_variables: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_variables + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, masked_parents: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predicts delta ΔX_i given parent inputs and action vector."""
        x = torch.cat([masked_parents, action], dim=-1)
        return self.net(x).squeeze(-1)


class NeuralSCM(nn.Module):
    """Full Neural Structural Causal Model with modular variable mechanisms."""

    def __init__(
        self,
        num_variables: int,
        action_dim: int,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.num_variables = num_variables
        self.action_dim = action_dim

        # d independent neural mechanisms for the d causal variables
        self.mechanisms = nn.ModuleList([
            VariableMechanism(num_variables, action_dim, hidden_dim)
            for _ in range(num_variables)
        ])

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        interventions: Optional[Dict[int, float]] = None
    ) -> torch.Tensor:
        """Evaluates causal forward rollout under a specified DAG structure.
        
        Args:
            state: Tensor of shape [B, d] or [d].
            action: Tensor of shape [B, action_dim] or [action_dim].
            adjacency_matrix: Continuous or binary DAG matrix [B, d, d] or [d, d].
            interventions: Dict mapping variable index k to forced interventional value v_k.
            
        Returns:
            Predicted next state tensor of shape [B, d].
        """
        is_single = state.dim() == 1
        if is_single:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)

        B, d = state.shape
        if adjacency_matrix.dim() == 2:
            adjacency_matrix = adjacency_matrix.unsqueeze(0).expand(B, -1, -1)

        interventions = interventions or {}
        next_state_cols = []

        for i in range(self.num_variables):
            if i in interventions:
                # Pearl's do-operator: cut incoming edges, force value
                forced_val = torch.full((B,), float(interventions[i]), dtype=state.dtype, device=state.device)
                next_state_cols.append(forced_val)
            else:
                # Gated parent input: A[:, i] ⊙ state
                parent_mask = adjacency_matrix[:, :, i]  # [B, d] incoming edges into variable i
                masked_parents = state * parent_mask     # [B, d]
                
                # Predict delta: X'_i = X_i + f_i(PA_i, a)
                delta = self.mechanisms[i](masked_parents, action)
                next_val = state[:, i] + delta
                next_state_cols.append(next_val)

        next_state = torch.stack(next_state_cols, dim=-1)

        if is_single:
            return next_state.squeeze(0)
        return next_state
