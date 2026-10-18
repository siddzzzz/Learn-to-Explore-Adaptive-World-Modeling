"""Epistemic Deep Ensemble World Model in PyTorch.

Implements an ensemble of M forward transition dynamics models f_θ_m(s, a)
for estimating both epistemic (model disagreement) and aleatoric uncertainty.
References:
    - Chua et al. (2018), PETS: Deep Reinforcement Learning in a Handful of Trials
    - Sekar et al. (2020), Planning to Explore via Self-Supervised World Models
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SingleDynamicsModel(nn.Module):
    """Single MLP forward transition model predicting state delta Δs = s' - s."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predicts next state s' = s + Δs."""
        x = torch.cat([state, action], dim=-1)
        delta = self.net(x)
        return state + delta


class EnsembleWorldModel:
    """Epistemic Deep Ensemble containing M independent dynamics models."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_models: int = 5,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        device: Optional[str] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_models = num_models
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Instantiate M independent models
        self.models = nn.ModuleList([
            SingleDynamicsModel(state_dim, action_dim, hidden_dim)
            for _ in range(num_models)
        ]).to(self.device)

        self.optimizers = [
            optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            for model in self.models
        ]

    def predict_all(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Evaluates all M models on input (s, a).
        
        Args:
            state: Array of shape [B, state_dim] or [state_dim].
            action: Array of shape [B, action_dim] or [action_dim].
            
        Returns:
            Array of shape [num_models, B, state_dim] of predicted next states.
        """
        s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        is_single = s_t.dim() == 1
        if is_single:
            s_t = s_t.unsqueeze(0)
            a_t = a_t.unsqueeze(0)

        preds = []
        with torch.no_grad():
            for model in self.models:
                preds.append(model(s_t, a_t))

        stacked = torch.stack(preds, dim=0).cpu().numpy()  # [M, B, state_dim]
        if is_single:
            stacked = stacked[:, 0, :]  # [M, state_dim]
        return stacked

    def compute_ensemble_disagreement(self, state: np.ndarray, action: np.ndarray) -> float:
        """Computes epistemic disagreement variance across ensemble members.
        
        Used by the Plan2Explore baseline:
            Var_m[f_m(s, a)] = (1 / M) ∑_m || f_m(s, a) - f_bar(s, a) ||^2
        """
        preds = self.predict_all(state, action)  # [M, state_dim]
        mean_pred = np.mean(preds, axis=0, keepdims=True)
        variance = np.mean(np.sum((preds - mean_pred) ** 2, axis=-1))
        return float(variance)

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """Trains the ensemble on a batch of transition data."""
        num_samples = len(states)
        if num_samples == 0:
            return {"loss": 0.0}

        s_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a_tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        ns_tensor = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)

        losses = []
        criterion = nn.MSELoss()

        for m_idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            # Bootstrap subsample per model for epistemic diversity
            indices = torch.randint(0, num_samples, (min(batch_size, num_samples),), device=self.device)
            s_batch = s_tensor[indices]
            a_batch = a_tensor[indices]
            ns_batch = ns_tensor[indices]

            optimizer.zero_grad()
            pred = model(s_batch, a_batch)
            loss = criterion(pred, ns_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return {"mean_loss": float(np.mean(losses))}
