"""Unit tests for Systemic Scientific Ablation Study Engine."""

import pytest
import numpy as np

from src.planning.ablation_runner import (
    run_objective_ablation,
    run_horizon_ablation,
    run_safety_bound_ablation,
)


def test_objective_ablation_runs_and_distinguishes_chd_from_icm():
    """Verify objective ablation runs and CHD achieves zero distractor fixation."""
    res = run_objective_ablation(seeds=[1, 2], num_steps=5, tv_noise_scale=2.0)
    assert "CHD (Ours)" in res
    assert "ICM (Prediction Error)" in res

    # CHD must be immune to distractor noise
    assert res["CHD (Ours)"]["mean_fixation_pct"] == 0.0
    # ICM click rate is strictly positive on noisy TV
    assert res["ICM (Prediction Error)"]["mean_fixation_pct"] > 0.0


def test_horizon_ablation_scaling():
    """Verify multi-step planning horizon metrics scale properly."""
    res = run_horizon_ablation(seeds=[1, 2], horizons=[1, 3])
    assert "H=1" in res
    assert "H=3" in res
    assert res["H=3"]["mean_steps_to_resolve"] < res["H=1"]["mean_steps_to_resolve"]


def test_safety_bound_ablation_guarantees():
    """Verify Cantelli LCB strictly maintains zero safety violations."""
    res = run_safety_bound_ablation(seeds=[1, 2], num_experiments=5)
    assert "Cantelli LCB (Safe-HEx-WM)" in res
    assert "Unconstrained" in res

    # Cantelli must have 0 violations
    assert res["Cantelli LCB (Safe-HEx-WM)"]["mean_violations"] == 0.0
    # Unconstrained has non-zero violations
    assert res["Unconstrained"]["mean_violations"] > 0.0
