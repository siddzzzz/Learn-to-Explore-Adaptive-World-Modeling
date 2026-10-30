"""Kinetic Models and Runge-Kutta 4th-Order ODE Integrator for Dynamic CSTR.

Implements non-linear differential dynamics for mass and energy balances
in an exothermic Continuous Stirred Tank Reactor with external cooling jacket.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import numpy as np


@dataclass(frozen=True)
class CSTRKineticParameters:
    """Arrhenius and thermodynamic parameters for candidate reaction mechanisms."""
    name: str
    k0: float             # Pre-exponential factor (1/min)
    E_over_R: float       # Activation energy parameter E_a / R (K)
    delta_H: float        # Heat of reaction (J/mol) - negative for exothermic
    UA: float = 20000.0   # Heat transfer coefficient (J/(min*K))


class CSTRDynamicsModel:
    """Simulates dynamic continuous trajectories of an exothermic CSTR via RK4."""

    # Candidate Hypotheses representing different chemical reaction kinetics
    HYPOTHESES: List[CSTRKineticParameters] = [
        CSTRKineticParameters(
            name="Mild Exotherm (H0)",
            k0=5.0e9,
            E_over_R=8000.0,
            delta_H=-45000.0,
        ),
        CSTRKineticParameters(
            name="Nominal Exotherm (H1)",
            k0=7.2e10,
            E_over_R=8750.0,
            delta_H=-50000.0,
        ),
        CSTRKineticParameters(
            name="High-Heat Runaway Exotherm (H2)",
            k0=1.2e11,
            E_over_R=9100.0,
            delta_H=-68000.0,
        ),
        CSTRKineticParameters(
            name="Severe Thermal Explosion Risk (H3)",
            k0=2.0e11,
            E_over_R=9400.0,
            delta_H=-80000.0,
        ),
    ]

    def __init__(self, hypothesis_id: int = 1):
        self.hypothesis_id = int(np.clip(hypothesis_id, 0, len(self.HYPOTHESES) - 1))
        self.params = self.HYPOTHESES[self.hypothesis_id]

        # Standard physical reactor constants
        self.V: float = 100.0         # L
        self.Vc: float = 20.0         # L
        self.q: float = 100.0         # L/min
        self.CAf: float = 1.0         # mol/L
        self.Tf: float = 350.0        # K
        self.Tcf: float = 300.0       # K
        self.rho_Cp: float = 1000.0   # J/(L*K)
        self.rhoc_Cpc: float = 1000.0 # J/(L*K)
        self.UA: float = self.params.UA

    def derivatives(self, state: np.ndarray, qc: float) -> np.ndarray:
        """Computes continuous time derivatives [dCA/dt, dT/dt, dTc/dt].
        
        Args:
            state: [CA (mol/L), T (K), Tc (K)]
            qc: Coolant jacket flow rate (L/min)
        """
        CA, T, Tc = float(state[0]), float(state[1]), float(state[2])
        # Prevent non-physical negative concentrations or absolute zero temperatures
        CA = max(0.0, CA)
        T = max(200.0, T)
        Tc = max(200.0, Tc)
        qc = max(0.0, float(qc))

        # Arrhenius reaction rate
        k = self.params.k0 * np.exp(-self.params.E_over_R / T)
        rA = k * CA

        # dCA/dt
        dCA_dt = (self.q / self.V) * (self.CAf - CA) - rA

        # dT/dt
        heat_gen = (-self.params.delta_H / self.rho_Cp) * rA
        heat_rem = (self.UA / (self.V * self.rho_Cp)) * (Tc - T)
        dT_dt = (self.q / self.V) * (self.Tf - T) + heat_gen + heat_rem

        # dTc/dt
        cooling_transfer = (self.UA / (self.Vc * self.rhoc_Cpc)) * (T - Tc)
        dTc_dt = (qc / self.Vc) * (self.Tcf - Tc) + cooling_transfer

        return np.array([dCA_dt, dT_dt, dTc_dt], dtype=np.float64)

    def step_rk4(self, state: np.ndarray, qc: float, dt: float = 0.1) -> np.ndarray:
        """Integrates CSTR state forward by dt using Runge-Kutta 4th Order."""
        s = np.asarray(state, dtype=np.float64)
        k1 = self.derivatives(s, qc)
        k2 = self.derivatives(s + 0.5 * dt * k1, qc)
        k3 = self.derivatives(s + 0.5 * dt * k2, qc)
        k4 = self.derivatives(s + dt * k3, qc)

        s_next = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # Apply physical floor
        s_next[0] = max(0.0, s_next[0])
        s_next[1] = max(200.0, s_next[1])
        s_next[2] = max(200.0, s_next[2])
        return s_next
