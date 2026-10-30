"""Safety evaluation harness for the Chemical Reactor Environment.

Evaluates thermal runaway risks, toxic byproduct concentrations, and physical bounds.
Provides exact constraint violation checking and Control Barrier Function (CBF) status.
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from src.safety.barrier_function import (
    ControlBarrierFunction,
    create_thermal_barrier,
    create_toxicity_barrier,
)


class ReactorSafetyEvaluator:
    """Evaluates safety margins and violations for continuous chemical reactor trials."""

    def __init__(
        self,
        max_temperature_kelvin: float = 420.0,
        max_waste_molar: float = 0.35,
        alpha: float = 0.5
    ):
        self.max_T = max_temperature_kelvin
        self.max_W = max_waste_molar
        self.alpha = alpha

        # State representation: [Temperature (K), C_A (M), C_P (M), C_W (M)]
        self.thermal_barrier = create_thermal_barrier(
            max_temperature_kelvin=max_temperature_kelvin,
            temp_index=0,
            alpha=alpha
        )
        self.toxicity_barrier = create_toxicity_barrier(
            max_waste_molar=max_waste_molar,
            waste_index=3,
            alpha=alpha
        )
        self.barriers = [self.thermal_barrier, self.toxicity_barrier]

    def construct_state(self, params: Dict[str, float], measured_concs: np.ndarray) -> np.ndarray:
        """Assembles 4D reactor operational state: [T, C_A, C_P, C_W]."""
        T = float(params.get("T", 300.0))
        ca = float(measured_concs[0]) if len(measured_concs) > 0 else 0.0
        cp = float(measured_concs[2]) if len(measured_concs) > 2 else 0.0
        cw = float(measured_concs[3]) if len(measured_concs) > 3 else 0.0
        return np.array([T, ca, cp, cw], dtype=np.float64)

    def evaluate_safety(
        self,
        params: Dict[str, float],
        measured_concs: np.ndarray
    ) -> Tuple[bool, float, Dict[str, float]]:
        """Checks whether an experiment violated any physical safety boundary.
        
        Returns:
            is_safe: True if all barrier functions h_m(s) >= 0.
            worst_margin: Minimum barrier value across all constraints.
            margins: Dict of individual barrier values.
        """
        state = self.construct_state(params, measured_concs)
        h_thermal = float(self.thermal_barrier.evaluate(state))
        h_toxic = float(self.toxicity_barrier.evaluate(state))

        worst_margin = min(h_thermal, h_toxic)
        is_safe = (worst_margin >= 0.0)

        margins = {
            "h_thermal": h_thermal,
            "h_toxic": h_toxic,
            "worst_margin": worst_margin,
            "is_thermal_safe": h_thermal >= 0.0,
            "is_toxic_safe": h_toxic >= 0.0,
        }
        return is_safe, worst_margin, margins
