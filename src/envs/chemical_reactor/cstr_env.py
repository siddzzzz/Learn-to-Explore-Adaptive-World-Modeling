"""Continuous Stirred Tank Reactor (CSTR) Gymnasium Environment.

Dynamic trajectory control benchmark where an agent regulates active cooling
jacket flow qc to explore unknown exothermic reaction mechanisms without thermal runaway.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.envs.chemical_reactor.cstr_kinetics import (
    CSTRDynamicsModel,
    CSTRKineticParameters,
)


class ContinuousCSTREnv(gym.Env):
    """Gymnasium environment for Dynamic Continuous CSTR Trajectory Control.
    
    State: [CA (mol/L), T (K), Tc (K)]
    Action: Normalized cooling flow rate u in [-1, 1] mapped to qc in [0, 100] L/min.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        ground_truth_id: int = 1,
        dt: float = 0.05,            # Integration timestep (minutes)
        max_steps: int = 40,         # Trajectory length (2.0 minutes total process time)
        noise_std: float = 0.1,      # Sensor measurement noise on temperature
        qc_bounds: Tuple[float, float] = (0.0, 100.0),
        render_mode: Optional[str] = None
    ):
        super().__init__()
        self.ground_truth_id = ground_truth_id
        self.dt = dt
        self.max_steps = max_steps
        self.noise_std = noise_std
        self.qc_bounds = qc_bounds
        self.render_mode = render_mode

        # True physical dynamics
        self.model = CSTRDynamicsModel(hypothesis_id=ground_truth_id)

        # Action: Continuous cooling flow rate in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation: [CA, T, Tc]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 200.0, 200.0], dtype=np.float32),
            high=np.array([5.0, 600.0, 600.0], dtype=np.float32),
            dtype=np.float32
        )

        # Initial steady state near operating point
        self.initial_state = np.array([0.8, 330.0, 310.0], dtype=np.float64)
        self.state = self.initial_state.copy()
        self.current_step = 0
        self.rng = np.random.RandomState(42)

    def denormalize_action(self, action: np.ndarray) -> float:
        """Maps normalized action in [-1, 1] to coolant flow rate qc in [qc_min, qc_max]."""
        u = float(np.clip(action[0] if hasattr(action, '__len__') else action, -1.0, 1.0))
        q_min, q_max = self.qc_bounds
        qc = q_min + 0.5 * (u + 1.0) * (q_max - q_min)
        return float(qc)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # Start near operating point with small perturbation
        ca0 = float(self.initial_state[0] + self.rng.uniform(-0.05, 0.05))
        t0 = float(self.initial_state[1] + self.rng.uniform(-3.0, 3.0))
        tc0 = float(self.initial_state[2] + self.rng.uniform(-2.0, 2.0))
        self.state = np.array([ca0, t0, tc0], dtype=np.float64)
        self.current_step = 0

        obs = self.state.astype(np.float32)
        info = {
            "hypothesis_id": self.ground_truth_id,
            "hypothesis_name": self.model.params.name,
            "step": self.current_step
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        qc = self.denormalize_action(action)

        # Integrate forward in time via RK4
        self.state = self.model.step_rk4(self.state, qc, dt=self.dt)

        # Add sensor measurement noise on temperature
        obs = self.state.copy()
        obs[1] += float(self.rng.normal(0.0, self.noise_std))

        # Check for catastrophic thermal runaway: T > 450K terminates episode
        is_explosion = bool(self.state[1] > 450.0)
        terminated = is_explosion
        truncated = self.current_step >= self.max_steps

        # Reward: penalize temperature deviations from safe setpoint (350K) and cooling effort
        temp_penalty = -0.01 * ((self.state[1] - 350.0) ** 2)
        effort_penalty = -0.001 * (qc ** 2)
        reward = float(temp_penalty + effort_penalty)
        if is_explosion:
            reward -= 100.0

        info = {
            "qc": qc,
            "core_temperature": float(self.state[1]),
            "coolant_temperature": float(self.state[2]),
            "reactant_concentration": float(self.state[0]),
            "is_explosion": is_explosion,
            "step": self.current_step
        }
        return obs.astype(np.float32), reward, terminated, truncated, info

    def get_hypothesis_space(self) -> List[CSTRKineticParameters]:
        return CSTRDynamicsModel.HYPOTHESES
