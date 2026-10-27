"""Control Barrier Functions (CBFs) for defining safe operational envelopes.

Defines mathematical barrier functions h(s) >= 0 where h(s) < 0 indicates a safety violation.
References:
    - Ames et al. (2019), Control Barrier Functions: Theory and Applications
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class ControlBarrierFunction:
    """Represents a continuous or discrete Control Barrier Function h(s)."""

    def __init__(
        self,
        barrier_fn: Callable[[np.ndarray], float],
        name: str = "Generic Barrier",
        alpha: float = 0.5
    ):
        self.barrier_fn = barrier_fn
        self.name = name
        self.alpha = float(np.clip(alpha, 0.01, 1.0))

    def evaluate(self, state: np.ndarray) -> float:
        """Evaluates barrier value h(s). Positive means safe, negative means violation."""
        return float(self.barrier_fn(np.asarray(state, dtype=np.float64)))

    def is_safe(self, state: np.ndarray) -> bool:
        """Returns True if state is within the safe set C = {s | h(s) >= 0}."""
        return self.evaluate(state) >= 0.0


def create_box_barrier(
    lower_bounds: Optional[np.ndarray] = None,
    upper_bounds: Optional[np.ndarray] = None,
    alpha: float = 0.5
) -> ControlBarrierFunction:
    """Constructs a multidimensional box barrier: h(s) = min_i (s_i - low_i, high_i - s_i)."""
    def box_fn(s: np.ndarray) -> float:
        margins = []
        if lower_bounds is not None:
            for i, low in enumerate(lower_bounds):
                if low is not None and not np.isnan(low):
                    margins.append(s[i] - low)
        if upper_bounds is not None:
            for i, high in enumerate(upper_bounds):
                if high is not None and not np.isnan(high):
                    margins.append(high - s[i])
        if not margins:
            return 1.0
        return float(np.min(margins))

    return ControlBarrierFunction(box_fn, name="Box Safety Boundary", alpha=alpha)


def create_thermal_barrier(
    max_temperature_kelvin: float = 430.0,
    temp_index: int = 0,
    alpha: float = 0.4
) -> ControlBarrierFunction:
    """Specialized barrier for chemical reactors: h(s) = T_max - T."""
    def thermal_fn(s: np.ndarray) -> float:
        # If scalar or dict or array
        t_val = s[temp_index] if len(s) > temp_index else s[0]
        return float(max_temperature_kelvin - t_val)

    return ControlBarrierFunction(
        thermal_fn,
        name=f"Thermal Runaway Barrier (T < {max_temperature_kelvin}K)",
        alpha=alpha
    )


def create_toxicity_barrier(
    max_waste_molar: float = 0.35,
    waste_index: int = 3,
    alpha: float = 0.5
) -> ControlBarrierFunction:
    """Specialized barrier for toxic byproduct accumulation: h(s) = C_W_max - C_W."""
    def waste_fn(s: np.ndarray) -> float:
        w_val = s[waste_index] if len(s) > waste_index else s[-1]
        return float(max_waste_molar - w_val)

    return ControlBarrierFunction(
        waste_fn,
        name=f"Toxicity Accumulation Barrier (W < {max_waste_molar}M)",
        alpha=alpha
    )
