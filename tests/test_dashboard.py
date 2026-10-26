"""Unit tests for Dashboard configuration, state engine, and Plotly visualizers."""

import numpy as np
import pytest
import streamlit as st
import plotly.graph_objects as go

from src.dashboard.config import PALETTE, CUSTOM_CSS, NAVIGATION_PAGES
from src.dashboard.plots import (
    plot_belief_distribution,
    plot_entropy_trajectory,
    plot_surprise_gauge,
    plot_batch_yields
)
from src.dashboard.state_engine import DashboardStateEngine


def test_dashboard_config():
    """Verify visual palette and navigation pages are defined properly."""
    assert "hex_wm" in PALETTE
    assert "oracle" in PALETTE
    assert len(NAVIGATION_PAGES) == 5
    assert "<style>" in CUSTOM_CSS


def test_dashboard_plot_generators():
    """Verify that all Plotly figures generate without exceptions."""
    weights = np.array([0.1, 0.7, 0.1, 0.1])
    h_names = ["H0: Ideal", "H1: Poisoning", "H2: Saturation", "H3: Cracking"]

    # 1. Belief distribution
    fig1 = plot_belief_distribution(weights, h_names)
    assert isinstance(fig1, go.Figure)

    # 2. Entropy trajectory
    fig2 = plot_entropy_trajectory([1.38, 0.5, 0.0], mutation_events=[1])
    assert isinstance(fig2, go.Figure)

    # 3. Surprise gauge
    fig3 = plot_surprise_gauge(current_surprise=18.5, threshold=15.0)
    assert isinstance(fig3, go.Figure)

    # 4. Batch yields
    exp_mock = [{
        "batch_num": 1,
        "params": {"T": 350.0},
        "yields": {"C_A": 0.5, "C_B": 0.2, "C_P": 1.5, "C_W": 0.1}
    }]
    fig4 = plot_batch_yields(exp_mock)
    assert isinstance(fig4, go.Figure)


def test_state_engine_lifecycle():
    """Verify DashboardStateEngine initializes and steps the reactor."""
    DashboardStateEngine.reset_chemical_state(ground_truth_id=1)
    state = DashboardStateEngine.get_chemical_state()
    
    assert state.chem_initialized
    assert state.chem_ground_truth_id == 1
    
    # Step an experiment
    exp = DashboardStateEngine.run_hex_experiment()
    assert exp["batch_num"] == 1
    assert "yields" in exp
    assert "C_P" in exp["yields"]
    assert len(state.chem_history_experiments) == 1

    # Mutate rule
    new_name = DashboardStateEngine.mutate_rule(new_hypothesis_id=3)
    assert state.chem_ground_truth_id == 3
    assert len(state.chem_mutation_events) == 1
