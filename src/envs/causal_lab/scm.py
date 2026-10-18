"""Structural Causal Model (SCM) definitions and hypothesis generation.

Based on Judea Pearl's Causal Inference framework (Pearl, 2009) and
interventional experimental design principles.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class CausalHypothesis:
    """Represents a specific structural mechanism hypothesis θ.
    
    Attributes:
        hypothesis_id: Unique integer index for the hypothesis.
        name: Human-readable identifier.
        adjacency_matrix: Binary matrix [N, N] where A[i, j] = 1 indicates edge X_i -> X_j.
        description: Description of the causal mechanism.
    """
    hypothesis_id: int
    name: str
    adjacency_matrix: np.ndarray
    description: str = ""

    def __repr__(self) -> str:
        return f"CausalHypothesis(id={self.hypothesis_id}, name='{self.name}')"


class StructuralCausalModel:
    """Parameterized Structural Causal Model for Causal-Lab.
    
    System of structural equations:
        X_0 = f_0(U_0)
        X_1 = f_1(X_0, U_1)
        ...
        X_N = f_N(PA_N, U_N)
    
    Supports interventional queries do(X_i = x) which sever incoming edges
    into X_i while preserving downstream mechanisms.
    """

    def __init__(
        self,
        num_variables: int = 4,
        hypothesis: Optional[CausalHypothesis] = None,
        noise_std: float = 0.05,
        rng: Optional[np.random.RandomState] = None
    ):
        self.num_variables = num_variables
        self.noise_std = noise_std
        self.rng = rng or np.random.RandomState(42)
        self.hypothesis = hypothesis or self._default_hypothesis()

    @staticmethod
    def _default_hypothesis() -> CausalHypothesis:
        # Default chain: X_0 -> X_1 -> X_2 -> X_3
        adj = np.zeros((4, 4), dtype=int)
        adj[0, 1] = 1
        adj[1, 2] = 1
        adj[2, 3] = 1
        return CausalHypothesis(0, "Linear Chain", adj, "Linear cascade: 0->1->2->3")

    def evaluate(
        self,
        interventions: Optional[Dict[int, float]] = None,
        exogenous_noise: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Evaluates the SCM under optional interventions do(X_i = v_i).
        
        Args:
            interventions: Dict mapping variable index i to forced value v_i.
            exogenous_noise: Optional noise array of shape [num_variables].
            
        Returns:
            State vector X of shape [num_variables].
        """
        if interventions is None:
            interventions = {}
        if exogenous_noise is None:
            exogenous_noise = self.rng.normal(0, self.noise_std, size=self.num_variables)

        x = np.zeros(self.num_variables, dtype=np.float32)
        adj = self.hypothesis.adjacency_matrix
        h_id = self.hypothesis.hypothesis_id

        # Topological forward pass through variables
        for i in range(self.num_variables):
            if i in interventions:
                # Pearl's do-operator: cut incoming edges, force value
                x[i] = float(interventions[i])
            else:
                parents = np.where(adj[:, i] == 1)[0]
                if len(parents) == 0:
                    # Root variable: base signal + exogenous noise
                    x[i] = np.sin(i * 1.5) + exogenous_noise[i]
                else:
                    # Mechanism depends on parents and hypothesis-specific non-linearities
                    parent_vals = x[parents]
                    x[i] = self._compute_mechanism(h_id, i, parent_vals) + exogenous_noise[i]

        return np.clip(x, -5.0, 5.0)

    @staticmethod
    def _compute_mechanism(h_id: int, var_idx: int, parent_vals: np.ndarray) -> float:
        """Computes structural mechanism f_i(PA_i; θ)."""
        # Linear or non-linear combination depending on hypothesis ID
        if h_id % 4 == 0:
            # Additive linear mechanism
            return float(np.sum(parent_vals) * 0.8)
        elif h_id % 4 == 1:
            # Multiplicative / gating mechanism
            return float(np.prod(parent_vals) * 1.2 if len(parent_vals) > 1 else np.tanh(parent_vals[0]) * 1.5)
        elif h_id % 4 == 2:
            # Threshold / step activation
            return float(np.sum(np.maximum(0.0, parent_vals - 0.2)) * 1.1)
        else:
            # Periodic / oscillatory response
            return float(np.sin(np.sum(parent_vals) * 2.0))


def generate_canonical_hypothesis_space(num_variables: int = 4) -> List[CausalHypothesis]:
    """Generates a well-defined family of candidate structural hypotheses.
    
    Includes canonical causal structures:
    - Independent / Null graph
    - Direct Chain (0->1->2->3)
    - Fork / Common Cause (0->1, 0->2, 0->3)
    - Collider / Immorality (0->3, 1->3, 2->3)
    - Diamond graph (0->1->3, 0->2->3)
    - Alternative mechanisms on identical topologies (distinguishable only via magnitude)
    
    Returns:
        List of candidate CausalHypothesis objects.
    """
    hypotheses = []

    # 1. Null / Independent graph (no edges)
    adj0 = np.zeros((num_variables, num_variables), dtype=int)
    hypotheses.append(CausalHypothesis(0, "Independent Graph", adj0, "Variables are fully independent"))

    # 2. Sequential Cascade (Chain: 0->1->2->3)
    adj1 = np.zeros((num_variables, num_variables), dtype=int)
    for i in range(num_variables - 1):
        adj1[i, i + 1] = 1
    hypotheses.append(CausalHypothesis(1, "Sequential Chain", adj1, "Cascade: 0->1->2->3"))

    # 3. Fork: Variable 0 is common confounder of 1, 2, 3
    adj2 = np.zeros((num_variables, num_variables), dtype=int)
    for i in range(1, num_variables):
        adj2[0, i] = 1
    hypotheses.append(CausalHypothesis(2, "Common Cause Fork", adj2, "0 causes 1, 2, 3"))

    # 4. Collider: Variables 0, 1, 2 all cause 3
    adj3 = np.zeros((num_variables, num_variables), dtype=int)
    for i in range(num_variables - 1):
        adj3[i, num_variables - 1] = 1
    hypotheses.append(CausalHypothesis(3, "Collider / Immorality", adj3, "0, 1, 2 all cause 3"))

    # 5. Diamond Graph: 0 -> 1 -> 3 and 0 -> 2 -> 3
    adj4 = np.zeros((num_variables, num_variables), dtype=int)
    adj4[0, 1] = 1
    adj4[0, 2] = 1
    adj4[1, 3] = 1
    adj4[2, 3] = 1
    hypotheses.append(CausalHypothesis(4, "Diamond Graph", adj4, "Dual pathway 0->(1,2)->3"))

    # 6. Inverted Chain: 3 -> 2 -> 1 -> 0
    adj5 = np.zeros((num_variables, num_variables), dtype=int)
    for i in range(num_variables - 1, 0, -1):
        adj5[i, i - 1] = 1
    hypotheses.append(CausalHypothesis(5, "Inverted Chain", adj5, "Reverse cascade: 3->2->1->0"))

    # 7. Cycle-breaking Feedback: 0->1->2->0 (with 3 observed)
    adj6 = np.zeros((num_variables, num_variables), dtype=int)
    adj6[0, 1] = 1
    adj6[1, 2] = 1
    adj6[2, 3] = 1
    adj6[0, 3] = 1
    hypotheses.append(CausalHypothesis(6, "Dense Forward DAG", adj6, "Feedforward with skip connection"))

    # 8. Bipartite Interaction: (0, 1) cause (2, 3)
    adj7 = np.zeros((num_variables, num_variables), dtype=int)
    adj7[0, 2] = 1
    adj7[0, 3] = 1
    adj7[1, 2] = 1
    adj7[1, 3] = 1
    hypotheses.append(CausalHypothesis(7, "Bipartite Causal Graph", adj7, "(0, 1) independently drive (2, 3)"))

    return hypotheses
