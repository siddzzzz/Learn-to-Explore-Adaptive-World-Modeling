"""Hypothesis representation and metadata structures.

Defines formal representations for candidate world rules, causal mechanisms,
and parametric dynamics hypotheses.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class HypothesisState:
    """Represents a candidate world rule hypothesis.
    
    Attributes:
        hypothesis_id: Unique integer index.
        name: Semantic name.
        parameters: Arbitrary parameter dictionary (e.g. weights, DAG edges, physical constants).
        prior_weight: Initial probability prior P(θ).
        metadata: Extra contextual annotations.
    """
    hypothesis_id: int
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    prior_weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"HypothesisState(id={self.hypothesis_id}, name='{self.name}', prior={self.prior_weight:.3f})"
