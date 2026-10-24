"""Batch Chemical Reactor Gymnasium Environment.

Real-world benchmark representing automated chemical discovery in a Self-Driving Lab.
The agent designs continuous experiments (temperature, feeds, catalyst, duration)
to discover unknown reaction kinetics and zero-shot optimize synthesis.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.envs.chemical_reactor.kinetics import (
    ChemicalKineticsModel,
    ReactionHypothesis
)


class ChemicalReactorEnv(gym.Env):
    """Gymnasium environment for Autonomous Chemical Reaction Mechanism Discovery."""

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        ground_truth_id: int = 1,
        noise_std: float = 0.02,
        max_experiments: int = 10,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        self.ground_truth_id = ground_truth_id
        self.noise_std = noise_std
        self.max_experiments = max_experiments
        self.render_mode = render_mode

        # True underlying kinetic mechanism
        self.ground_truth_model = ChemicalKineticsModel(ground_truth_id)

        # Action: Continuous 4D vector normalized in [-1, 1]
        # [Temperature, C_A0, C_B0, Duration]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation: Measured concentrations [C_A, C_B, C_P, C_W] (mol/L)
        self.observation_space = spaces.Box(
            low=0.0, high=10.0, shape=(4,), dtype=np.float32
        )

        # Experiment bounds in physical units
        self.bounds = {
            "T": (300.0, 480.0),       # Kelvin
            "C_A0": (0.5, 4.0),        # mol/L
            "C_B0": (0.05, 0.8),       # mol/L
            "duration": (10.0, 100.0)  # minutes
        }

        # Internal state
        self.experiment_count = 0
        self.last_experiment_params: Dict[str, float] = {}
        self.last_measured_yield: np.ndarray = np.zeros(4, dtype=np.float32)

        # Downstream task goal: (Target P, Max Allowed W, Max Temp)
        self.target_goal: Optional[Dict[str, float]] = None
        self.in_evaluation_mode: bool = False
        self.rng = np.random.RandomState(42)

    def denormalize_action(self, action: np.ndarray) -> Dict[str, float]:
        """Maps normalized action [-1, 1]^4 to physical reactor units."""
        act = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        
        # Scale to physical units
        t_low, t_high = self.bounds["T"]
        ca_low, ca_high = self.bounds["C_A0"]
        cb_low, cb_high = self.bounds["C_B0"]
        dur_low, dur_high = self.bounds["duration"]

        return {
            "T": float(t_low + 0.5 * (act[0] + 1.0) * (t_high - t_low)),
            "C_A0": float(ca_low + 0.5 * (act[1] + 1.0) * (ca_high - ca_low)),
            "C_B0": float(cb_low + 0.5 * (act[2] + 1.0) * (cb_high - cb_low)),
            "duration": float(dur_low + 0.5 * (act[3] + 1.0) * (dur_high - dur_low))
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.experiment_count = 0
        self.last_experiment_params = {}
        self.last_measured_yield = np.zeros(4, dtype=np.float32)

        obs = np.zeros(4, dtype=np.float32)
        info = {
            "ground_truth_id": self.ground_truth_id,
            "hypothesis_name": self.ground_truth_model.hypothesis.name,
            "experiments_remaining": self.max_experiments
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.experiment_count += 1
        params = self.denormalize_action(action)
        self.last_experiment_params = params

        # Run physical batch reaction via non-linear ODE integration
        true_concs = self.ground_truth_model.simulate(
            T_kelvin=params["T"],
            C_A0=params["C_A0"],
            C_B0=params["C_B0"],
            duration_min=params["duration"]
        )

        # Spectroscopic measurement noise
        noise = self.rng.normal(0.0, self.noise_std, size=4).astype(np.float32)
        measured_concs = np.maximum(0.0, true_concs + noise)
        self.last_measured_yield = measured_concs

        # Reward computation
        if self.in_evaluation_mode and self.target_goal is not None:
            # Downstream objective: Maximize pure Product P while penalizing toxic Waste W
            cP = float(measured_concs[2])
            cW = float(measured_concs[3])
            reward = cP - 3.0 * cW
            # Penalty if safe thermal limit exceeded
            if params["T"] > self.target_goal.get("max_T", 450.0):
                reward -= 5.0
        else:
            # Epistemic discovery phase: zero extrinsic reward
            reward = 0.0

        terminated = False
        truncated = self.experiment_count >= self.max_experiments

        info = {
            "params": params,
            "true_concs": true_concs,
            "measured_concs": measured_concs,
            "ground_truth_id": self.ground_truth_id,
            "experiment_num": self.experiment_count
        }

        return measured_concs, reward, terminated, truncated, info

    def set_evaluation_mode(self, target_goal: Dict[str, float]) -> None:
        self.in_evaluation_mode = True
        self.target_goal = target_goal

    def get_hypothesis_space(self) -> List[ReactionHypothesis]:
        return ChemicalKineticsModel.HYPOTHESES

    def get_ground_truth(self) -> ReactionHypothesis:
        return self.ground_truth_model.hypothesis

    def render(self) -> Optional[str]:
        p = self.last_experiment_params
        y = self.last_measured_yield
        text = (
            f"[Exp {self.experiment_count}] T={p.get('T', 0):.1f}K, "
            f"C_A0={p.get('C_A0', 0):.2f}M, C_B0={p.get('C_B0', 0):.2f}M, dur={p.get('duration', 0):.1f}m | "
            f"Yield -> P={y[2]:.2f}M, W={y[3]:.2f}M"
        )
        if self.render_mode == "human":
            print(text)
        return text
