"""Non-linear Chemical Reaction Kinetics and Mechanistic ODE Models.

Implements real chemical rate laws with Arrhenius thermal activation:
    k(T) = A * exp( -E_a / (R * T) )
References:
    - Levenspiel, O. (1998). Chemical Reaction Engineering (3rd ed.).
    - Cronin et al. (2018). Organic synthesis in a modular automated system. Nature.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp

# Universal gas constant in kJ / (mol * K)
R_GAS = 8.314e-3


@dataclass(frozen=True)
class ReactionHypothesis:
    """Represents a candidate reaction kinetics mechanism θ."""
    hypothesis_id: int
    name: str
    description: str


class ChemicalKineticsModel:
    """Evaluates continuous ODE reaction kinetics for candidate mechanisms."""

    HYPOTHESES = [
        ReactionHypothesis(
            0,
            "Direct Catalysis (Ideal)",
            "A + B -> P + B; A + A -> W (No catalyst decay, stable product)"
        ),
        ReactionHypothesis(
            1,
            "Catalyst Poisoning (Byproduct Deactivation)",
            "Waste W irreversibly binds and poisons catalyst B (B + W -> Inert)"
        ),
        ReactionHypothesis(
            2,
            "Saturation / Michaelis-Menten Kinetics",
            "Catalytic rate saturates at high substrate concentration: r = Vmax*Ca*Cb / (Km + Ca)"
        ),
        ReactionHypothesis(
            3,
            "Thermal Degradation / Cracking",
            "High temperature thermally degrades target product into waste: P -> W"
        )
    ]

    def __init__(self, hypothesis_id: int = 0):
        self.hypothesis_id = hypothesis_id
        self.hypothesis = self.HYPOTHESES[hypothesis_id]

        # Arrhenius kinetic parameters: A (frequency factor), E_a (activation energy in kJ/mol)
        self.A1 = 5.0e4      # Main reaction: A + B -> P
        self.Ea1 = 35.0

        self.A2 = 2.0e3      # Byproduct reaction: 2A -> W
        self.Ea2 = 45.0

        self.Ad = 1.0e3      # Poisoning / deactivation
        self.Ead = 30.0

        self.Acrack = 8.0e6  # Thermal cracking of P -> W
        self.Eacrack = 65.0

        self.Km = 1.5        # Michaelis constant in mol/L

    def arrhenius_rate(self, A: float, Ea: float, T_kelvin: float) -> float:
        """Computes rate constant k(T) = A * exp(-E_a / (R * T))."""
        T_safe = np.clip(T_kelvin, 280.0, 550.0)
        return float(A * np.exp(-Ea / (R_GAS * T_safe)))

    def simulate(
        self,
        T_kelvin: float,
        C_A0: float,
        C_B0: float,
        duration_min: float
    ) -> np.ndarray:
        """Solves the continuous ODE system for a batch reaction.
        
        Args:
            T_kelvin: Temperature in Kelvin [300, 480].
            C_A0: Initial substrate A concentration (mol/L) [0.5, 5.0].
            C_B0: Initial catalyst B concentration (mol/L) [0.05, 1.0].
            duration_min: Reaction duration in minutes [5, 120].
            
        Returns:
            Final concentrations [C_A, C_B, C_P, C_W] at t = duration_min.
        """
        # Rate constants at operating temperature
        k1 = self.arrhenius_rate(self.A1, self.Ea1, T_kelvin)
        k2 = self.arrhenius_rate(self.A2, self.Ea2, T_kelvin)
        kd = self.arrhenius_rate(self.Ad, self.Ead, T_kelvin)
        kcrack = self.arrhenius_rate(self.Acrack, self.Eacrack, T_kelvin)

        # Initial state vector: [C_A, C_B, C_P, C_W]
        y0 = [max(0.0, C_A0), max(0.0, C_B0), 0.0, 0.0]
        h_id = self.hypothesis_id

        def ode_system(t, y):
            cA, cB, cP, cW = np.maximum(0.0, y)

            if h_id == 0:
                # Direct Catalysis (Ideal)
                r_prod = k1 * cA * cB
                r_waste = k2 * (cA ** 2)
                d_cA = -r_prod - 2.0 * r_waste
                d_cB = 0.0
                d_cP = r_prod
                d_cW = r_waste

            elif h_id == 1:
                # Catalyst Poisoning
                r_prod = k1 * cA * cB
                r_waste = k2 * (cA ** 2)
                r_poison = kd * cB * cW
                d_cA = -r_prod - 2.0 * r_waste
                d_cB = -r_poison
                d_cP = r_prod
                d_cW = r_waste

            elif h_id == 2:
                # Saturation / Michaelis-Menten
                r_prod = (k1 * cA * cB) / (self.Km + cA)
                r_waste = k2 * (cA ** 2)
                d_cA = -r_prod - 2.0 * r_waste
                d_cB = 0.0
                d_cP = r_prod
                d_cW = r_waste

            else:
                # Thermal Cracking
                r_prod = k1 * cA * cB
                r_waste = k2 * (cA ** 2)
                r_crack = kcrack * cP
                d_cA = -r_prod - 2.0 * r_waste
                d_cB = 0.0
                d_cP = r_prod - r_crack
                d_cW = r_waste + r_crack

            return [d_cA, d_cB, d_cP, d_cW]

        # Numerical integration using Runge-Kutta (RK45)
        t_span = (0.0, max(0.1, duration_min))
        sol = solve_ivp(ode_system, t_span, y0, method="RK45", rtol=1e-5, atol=1e-7)

        final_state = np.maximum(0.0, sol.y[:, -1])
        return np.asarray(final_state, dtype=np.float32)
