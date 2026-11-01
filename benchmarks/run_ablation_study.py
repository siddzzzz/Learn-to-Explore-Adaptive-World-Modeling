"""Benchmark Suite Runner: Systemic Scientific Ablation Matrix.

Executes 3 targeted scientific ablation experiments:
1. Epistemic Exploration Objective Ablation (CHD vs BALD vs P2E vs ICM vs Random)
2. Trajectory Planning Horizon Ablation (H in {1, 2, 3, 5})
3. Epistemic Safety Bound Ablation (Cantelli LCB vs Gaussian LCB vs Mean-Only vs Unconstrained)

Generates:
- benchmarks/results/ablation_matrix_results.json
- benchmarks/figures/fig9_ablation_matrix.png
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from src.planning.ablation_runner import (
    run_objective_ablation,
    run_horizon_ablation,
    run_safety_bound_ablation,
)

console = Console()


def plot_ablation_figure(results: Dict[str, Any], output_path: str = "benchmarks/figures/fig9_ablation_matrix.png"):
    """Plots a 3-panel publication figure summarizing the ablation matrix."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Acquisition Criterion vs Distractor Fixation
    obj_data = results["objective_ablation"]
    obj_names = list(obj_data.keys())
    fixation_rates = [obj_data[m]["mean_fixation_pct"] for m in obj_names]
    fixation_errs = [obj_data[m]["std_fixation_pct"] for m in obj_names]

    colors_obj = ["#2ecc71" if "CHD" in m else "#95a5a6" if "Random" in m else "#e74c3c" if "ICM" in m else "#3498db" for m in obj_names]
    ax1.bar(range(len(obj_names)), fixation_rates, yerr=fixation_errs, capsize=4, color=colors_obj, alpha=0.85, edgecolor="black")
    ax1.set_xticks(range(len(obj_names)))
    ax1.set_xticklabels(obj_names, rotation=18, ha="right", fontsize=9)
    ax1.set_ylabel("Distractor Fixation Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Objective Ablation: Distractor Fixation", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel B: Planning Horizon vs Steps to Disambiguate
    hor_data = results["horizon_ablation"]
    hor_names = list(hor_data.keys())
    steps_resolve = [hor_data[h]["mean_steps_to_resolve"] for h in hor_names]
    steps_errs = [hor_data[h]["std_steps_to_resolve"] for h in hor_names]

    ax2.plot(range(len(hor_names)), steps_resolve, marker="o", linewidth=2.5, color="#8e44ad", markersize=8)
    ax2.fill_between(
        range(len(hor_names)),
        np.array(steps_resolve) - np.array(steps_errs),
        np.array(steps_resolve) + np.array(steps_errs),
        color="#8e44ad",
        alpha=0.2,
    )
    ax2.set_xticks(range(len(hor_names)))
    ax2.set_xticklabels(hor_names, fontsize=10)
    ax2.set_ylabel("Steps to Discard False Rules", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Horizon Ablation: Multi-Step Non-Myopy", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel C: Safety Bound vs Cumulative Runaway Violations
    safe_data = results["safety_bound_ablation"]
    safe_names = list(safe_data.keys())
    violations = [safe_data[b]["mean_violations"] for b in safe_names]
    viol_errs = [safe_data[b]["std_violations"] for b in safe_names]

    colors_safe = ["#2ecc71" if "Cantelli" in b else "#e67e22" if "Gaussian" in b else "#e74c3c" for b in safe_names]
    ax3.bar(range(len(safe_names)), violations, yerr=viol_errs, capsize=4, color=colors_safe, alpha=0.85, edgecolor="black")
    ax3.set_xticks(range(len(safe_names)))
    ax3.set_xticklabels(safe_names, rotation=18, ha="right", fontsize=9)
    ax3.set_ylabel("Safety Violations per Run", fontsize=11, fontweight="bold")
    ax3.set_title("(c) Safety Bound: Runaway Immunity", fontsize=12, fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    console.print(f"[bold green]Saved publication Figure 9 to {output_path}[/bold green]")


def main():
    console.rule("[bold cyan]Executing Systemic Scientific Ablation Study Matrix[/bold cyan]")
    os.makedirs("benchmarks/results", exist_ok=True)
    os.makedirs("benchmarks/figures", exist_ok=True)

    seeds = [11, 22, 33, 44, 55]

    console.print("[yellow]1. Running Epistemic Exploration Objective Ablation...[/yellow]")
    obj_results = run_objective_ablation(seeds=seeds, num_steps=12, tv_noise_scale=3.0)

    console.print("[yellow]2. Running Planning Horizon Ablation...[/yellow]")
    hor_results = run_horizon_ablation(seeds=seeds, horizons=[1, 2, 3, 5])

    console.print("[yellow]3. Running Epistemic Safety Bound Ablation...[/yellow]")
    safe_results = run_safety_bound_ablation(seeds=seeds, num_experiments=8)

    results = {
        "objective_ablation": obj_results,
        "horizon_ablation": hor_results,
        "safety_bound_ablation": safe_results,
    }

    # Print summary tables
    t1 = Table(title="Ablation Axis 1: Acquisition Function vs Distractor Fixation")
    t1.add_column("Acquisition Objective", style="bold")
    t1.add_column("Fixation Rate (%)", justify="right")
    t1.add_column("Steps to Identify", justify="right")
    t1.add_column("Final Entropy", justify="right")
    for k, v in obj_results.items():
        t1.add_row(k, f"{v['mean_fixation_pct']:.1f}%", f"{v['mean_steps_to_solve']:.1f}", f"{v['mean_final_entropy']:.3f}")
    console.print(t1)

    t2 = Table(title="Ablation Axis 2: Receding Horizon vs Identification Speed")
    t2.add_column("Horizon H", style="bold")
    t2.add_column("Steps to Resolve", justify="right")
    t2.add_column("Cumulative Info Gain (nats)", justify="right")
    for k, v in hor_results.items():
        t2.add_row(k, f"{v['mean_steps_to_resolve']:.2f}", f"{v['cumulative_info_gain']:.3f}")
    console.print(t2)

    t3 = Table(title="Ablation Axis 3: Safety Concentration Bound vs Thermal Breaches")
    t3.add_column("Boundary Formulation", style="bold")
    t3.add_column("Mean Violations", justify="right")
    t3.add_column("Worst Margin (K)", justify="right")
    t3.add_column("Max Temp (K)", justify="right")
    for k, v in safe_results.items():
        t3.add_row(k, f"{v['mean_violations']:.2f}", f"{v['worst_margin']:.1f}", f"{v['max_temperature']:.1f}")
    console.print(t3)

    output_json = "benchmarks/results/ablation_matrix_results.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[bold green]Saved raw ablation data to {output_json}[/bold green]")

    plot_ablation_figure(results)


if __name__ == "__main__":
    main()
