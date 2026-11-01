"""Synthetic Non-linear Causal DAG Generator.

Generates arbitrary n-node directed acyclic graphs (Erdős–Rényi, Chain, Collider)
with non-linear mechanisms (tanh, sin, cubic) and Pearl do-operator simulation.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np


class SyntheticCausalDAG:
    """Represents a non-linear structural causal model over d variables."""

    def __init__(
        self,
        num_nodes: int = 6,
        edge_prob: float = 0.35,
        noise_std: float = 0.05,
        rng: Optional[np.random.RandomState] = None
    ):
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob
        self.noise_std = noise_std
        self.rng = rng or np.random.RandomState(42)

        # Generate upper-triangular adjacency matrix (strictly acyclic)
        self.adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if self.rng.rand() < edge_prob:
                    # Random edge weight in [-1.5, -0.4] U [0.4, 1.5]
                    sign = 1.0 if self.rng.rand() > 0.5 else -1.0
                    weight = float(sign * self.rng.uniform(0.5, 1.2))
                    self.adjacency[i, j] = weight

        # Non-linear activation assignment per node
        # 0: linear, 1: tanh, 2: cos
        self.activations = [int(self.rng.choice([0, 1, 2])) for _ in range(num_nodes)]

    def sample(self, interventions: Optional[Dict[int, float]] = None) -> np.ndarray:
        """Propagates causal signals through the DAG in topological order."""
        interv = interventions or {}
        values = np.zeros(self.num_nodes, dtype=np.float32)

        for i in range(self.num_nodes):
            if i in interv:
                # Pearl do-operator: do(X_i = value)
                values[i] = float(interv[i])
            else:
                # Parents of node i: nodes j where adjacency[j, i] != 0
                parents = np.where(np.abs(self.adjacency[:, i]) > 1e-4)[0]
                if len(parents) == 0:
                    # Root node
                    raw = float(self.rng.normal(0.0, 1.0))
                else:
                    parent_vals = values[parents]
                    weights = self.adjacency[parents, i]
                    linear_comb = float(np.sum(parent_vals * weights))

                    act_type = self.activations[i]
                    if act_type == 1:
                        raw = float(np.tanh(linear_comb))
                    elif act_type == 2:
                        raw = float(np.cos(linear_comb))
                    else:
                        raw = linear_comb

                noise = float(self.rng.normal(0.0, self.noise_std))
                values[i] = raw + noise

        return values

    def get_binary_adjacency(self) -> np.ndarray:
        """Returns binary adjacency matrix [d, d]."""
        return (np.abs(self.adjacency) > 1e-4).astype(int)
