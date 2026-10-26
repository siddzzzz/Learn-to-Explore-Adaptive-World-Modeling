"""Dashboard Session State Engine.

Manages interactive session states for the chemical reactor and causal lab,
orchestrating active experimentation, belief history, manual overrides,
and live rule mutations.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import streamlit as st

from src.envs.chemical_reactor import ChemicalReactorEnv, ChemicalKineticsModel
from src.agents.chemical_hex_agent import ChemicalHExAgent


class DashboardStateEngine:
    """Manages active simulation state across Streamlit reruns."""

    @staticmethod
    def get_chemical_state() -> Dict[str, Any]:
        """Initializes or retrieves the active chemical reactor session state."""
        if "chem_initialized" not in st.session_state:
            DashboardStateEngine.reset_chemical_state(ground_truth_id=3)
        return st.session_state

    @staticmethod
    def reset_chemical_state(ground_truth_id: int = 3) -> None:
        """Resets the chemical reactor simulation state."""
        env = ChemicalReactorEnv(ground_truth_id=ground_truth_id, max_experiments=20)
        agent = ChemicalHExAgent(
            bounds=env.bounds,
            noise_variance=0.0004,
            num_candidates=150,
            rng=np.random.RandomState(42)
        )
        obs, info = env.reset(seed=42)
        agent.reset()

        st.session_state.chem_env = env
        st.session_state.chem_agent = agent
        st.session_state.chem_ground_truth_id = ground_truth_id
        st.session_state.chem_history_entropy = [agent.belief_tracker.get_shannon_entropy()]
        st.session_state.chem_history_beliefs = [agent.belief_tracker.weights.copy()]
        st.session_state.chem_history_experiments: List[Dict[str, Any]] = []
        st.session_state.chem_history_surprise: List[float] = [0.0]
        st.session_state.chem_mutation_events: List[int] = []
        st.session_state.chem_initialized = True

    @staticmethod
    def run_hex_experiment() -> Dict[str, Any]:
        """Executes a single HEx-WM optimal experiment."""
        env: ChemicalReactorEnv = st.session_state.chem_env
        agent: ChemicalHExAgent = st.session_state.chem_agent

        action = agent.select_action()
        obs, reward, terminated, truncated, info = env.step(action)
        metrics = agent.observe_transition(action, obs)

        p = info["params"]
        exp_record = {
            "batch_num": len(st.session_state.chem_history_experiments) + 1,
            "action": action,
            "params": p,
            "yields": {
                "C_A": float(obs[0]),
                "C_B": float(obs[1]),
                "C_P": float(obs[2]),
                "C_W": float(obs[3])
            },
            "purity": float((obs[2] / max(1e-4, obs[2] + obs[3])) * 100.0),
            "entropy": metrics["entropy"],
            "surprise": metrics["surprise"],
            "map_hypothesis": metrics["map_hypothesis"],
            "map_confidence": metrics["map_confidence"]
        }

        st.session_state.chem_history_experiments.append(exp_record)
        st.session_state.chem_history_entropy.append(metrics["entropy"])
        st.session_state.chem_history_beliefs.append(agent.belief_tracker.weights.copy())
        st.session_state.chem_history_surprise.append(metrics["surprise"])

        return exp_record

    @staticmethod
    def run_manual_experiment(T: float, Ca0: float, Cb0: float, duration: float) -> Dict[str, Any]:
        """Allows manual user control of reactor parameters and records agent belief update."""
        env: ChemicalReactorEnv = st.session_state.chem_env
        agent: ChemicalHExAgent = st.session_state.chem_agent

        # Normalize physical parameters to [-1, 1]^4
        t_low, t_high = env.bounds["T"]
        ca_low, ca_high = env.bounds["C_A0"]
        cb_low, cb_high = env.bounds["C_B0"]
        dur_low, dur_high = env.bounds["duration"]

        a0 = 2.0 * (T - t_low) / (t_high - t_low) - 1.0
        a1 = 2.0 * (Ca0 - ca_low) / (ca_high - ca_low) - 1.0
        a2 = 2.0 * (Cb0 - cb_low) / (cb_high - cb_low) - 1.0
        a3 = 2.0 * (duration - dur_low) / (dur_high - dur_low) - 1.0

        action = np.array([a0, a1, a2, a3], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        metrics = agent.observe_transition(action, obs)

        p = info["params"]
        exp_record = {
            "batch_num": len(st.session_state.chem_history_experiments) + 1,
            "action": action,
            "params": p,
            "yields": {
                "C_A": float(obs[0]),
                "C_B": float(obs[1]),
                "C_P": float(obs[2]),
                "C_W": float(obs[3])
            },
            "purity": float((obs[2] / max(1e-4, obs[2] + obs[3])) * 100.0),
            "entropy": metrics["entropy"],
            "surprise": metrics["surprise"],
            "map_hypothesis": metrics["map_hypothesis"],
            "map_confidence": metrics["map_confidence"]
        }

        st.session_state.chem_history_experiments.append(exp_record)
        st.session_state.chem_history_entropy.append(metrics["entropy"])
        st.session_state.chem_history_beliefs.append(agent.belief_tracker.weights.copy())
        st.session_state.chem_history_surprise.append(metrics["surprise"])

        return exp_record

    @staticmethod
    def mutate_rule(new_hypothesis_id: int) -> str:
        """Silently swaps the true kinetic mechanism inside the reactor."""
        env: ChemicalReactorEnv = st.session_state.chem_env
        env.ground_truth_id = new_hypothesis_id
        env.ground_truth_model = ChemicalKineticsModel(new_hypothesis_id)
        st.session_state.chem_ground_truth_id = new_hypothesis_id

        current_step = len(st.session_state.chem_history_experiments)
        st.session_state.chem_mutation_events.append(current_step)
        return env.ground_truth_model.hypothesis.name
