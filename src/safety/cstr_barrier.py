"""Dynamic Control Barrier Functions for Continuous Stirred Tank Reactors.

Enforces physical temperature ceilings (e.g. T < 410K) to strictly prevent
exothermic thermal explosions along continuous trajectories.
"""

from __future__ import annotations
from typing import Callable
import numpy as np

from src.safety.barrier_function import ControlBarrierFunction


def create_cstr_thermal_barrier(
    max_temperature_kelvin: float = 410.0,
    alpha: float = 0.5
) -> ControlBarrierFunction:
    """Creates a Control Barrier Function h(s) = T_max - T for CSTR states.
    
    State is [CA, T, Tc] where index 1 is core temperature T.
    """
    def cstr_thermal_fn(state: np.ndarray) -> float:
        s = np.asarray(state, dtype=np.float64)
        T_val = float(s[1]) if len(s) > 1 else float(s[0])
        return float(max_temperature_kelvin - T_val)

    return ControlBarrierFunction(
        barrier_fn=cstr_thermal_fn,
        name=f"CSTR Thermal Runaway Barrier (T < {max_temperature_kelvin}K)",
        alpha=alpha
    )
