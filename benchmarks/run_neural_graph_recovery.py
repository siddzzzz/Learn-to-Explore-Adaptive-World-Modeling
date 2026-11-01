"""Benchmark Runner: Neural Causal Graph Discovery.

Evaluates interventional DAG structure recovery on non-linear SCMs:
1. Active CGD (Neural Causal Agent with Contrastive Graph Discrepancy)
2. Uniform Random Interventions
3. Pure Observational Learning (Passive NOTEARS)

Quantifies across 5 random seeds:
- Structural Hamming Distance (SHD)
- True Positive Rate (TPR / Recall)
- False Discovery Rate (FDR)
- Number of edge reversals resolved
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from src.envs.causal_lab.synthetic_dag_generator import SyntheticCausalDAG
from src.agents.neural_causal_agent import NeuralCausalAgent
from src.models.causal.graph_metrics import compute_graph_metrics

console = Console()


def run_single_graph_recovery_trial(
    method: str,
    seed: int,
    num_nodes: int = 6,
    num_interventions: int = 25,
) -> Dict[str, Any]:
    """Runs a single graph structure recovery experiment on a synthetic non-linear SCM."""
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # Generate ground truth non-linear DAG
    true_scm = SyntheticCausalDAG(num_nodes=num_nodes, edge_prob=0.35, noise_std=0.02, rng=rng)
    G_true = true_scm.get_binary_adjacency()

    # Action decoder: action 0 = observational; action 1..num_nodes = do(X_i = 2.0); action num_nodes+1..2*num_nodes = do(X_i = -2.0)
    num_actions = 2 * num_nodes + 1

    def decode_action(act: int) -> Dict[int, float]:
        if act == 0:
            return {}
        adj = act - 1
        var_idx = adj % num_nodes
        val = 2.0 if (adj // num_nodes) == 0 else -2.0
        return {var_idx: val}

    # Initialize Neural Causal Agent
    agent = NeuralCausalAgent(
        num_variables=num_nodes,
        action_dim=num_nodes,
        action_decoder_fn=decode_action,
        num_actions=num_actions,
        num_graph_particles=8,
        hidden_dim=32,
        noise_variance=0.0025,
        rng=np.random.RandomState(seed),
    )

    current_state = true_scm.sample()

    # Active exploration / intervention phase
    for step in range(num_interventions):
        if method == "Active CGD (Ours)":
            action = agent.select_action(current_state)
        elif method == "Random Interventions":
            # Exclude observational (action 0) to ensure active perturbations
            action = int(rng.randint(1, num_actions))
        else:
            # Pure observational: action 0
            action = 0

        interv = decode_action(action)
        next_state = true_scm.sample(interventions=interv)
        agent.observe_transition(current_state, action, next_state)
        current_state = next_state

    # MAP candidate particle selection from Bayesian trajectory likelihoods
    map_idx = int(np.argmax(agent.dag_weights))
    G_pred = agent.candidate_dags[map_idx].detach().cpu().numpy()
    metrics = compute_graph_metrics(G_true, G_pred, threshold=0.5)
    return metrics


def main():
    console.rule("[bold cyan]Running Neural Causal Graph Discovery Benchmark (Interventional DAG Recovery)[/bold cyan]")
    os.makedirs("benchmarks/results", exist_ok=True)
    os.makedirs("benchmarks/figures", exist_ok=True)

    seeds = [11, 22, 33, 44, 55]
    methods = [
        "Active CGD (Ours)",
        "Random Interventions",
        "Observational NOTEARS",
    ]

    results = {}
    table = Table(title="Neural Graph Discovery: SHD, TPR, and FDR across Interventions")
    table.add_column("Exploration Paradigm", style="bold")
    table.add_column("SHD (Lower is better)", justify="right")
    table.add_column("TPR (Recall)", justify="right")
    table.add_column("FDR (Precision error)", justify="right")
    table.add_column("Reversals Resolved", justify="right")

    for method_name in methods:
        trials = []
        for s in seeds:
            res = run_single_graph_recovery_trial(
                method=method_name,
                seed=s,
                num_nodes=6,
                num_interventions=20,
            )
            trials.append(res)

        mean_shd = float(np.mean([t["shd"] for t in trials]))
        mean_tpr = float(np.mean([t["tpr"] for t in trials])) * 100.0
        mean_fdr = float(np.mean([t["fdr"] for t in trials])) * 100.0
        mean_rev = float(np.mean([t["reversed_edges"] for t in trials]))

        results[method_name] = {
            "mean_shd": mean_shd,
            "mean_tpr_pct": mean_tpr,
            "mean_fdr_pct": mean_fdr,
            "mean_reversed_edges": mean_rev,
            "trials": trials,
        }

        table.add_row(
            method_name,
            f"[bold green]{mean_shd:.2f}[/bold green]" if mean_shd == min(mean_shd, 5.0) else f"{mean_shd:.2f}",
            f"{mean_tpr:.1f}%",
            f"{mean_fdr:.1f}%",
            f"{mean_rev:.1f}",
        )

    console.print(table)

    with open("benchmarks/results/neural_graph_recovery_results.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_neural_graph_figure(results)


def plot_neural_graph_figure(results: Dict[str, Any]):
    """Plots dual-panel publication figure: Structural Hamming Distance & TPR/Recall."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    method_names = list(results.keys())
    shds = [results[m]["mean_shd"] for m in method_names]
    tprs = [results[m]["mean_tpr_pct"] for m in method_names]

    colors = [
        "#10b981",  # Active CGD: Emerald green
        "#f59e0b",  # Random interventions: Amber
        "#ef4444",  # Observational: Crimson
    ]

    # Panel 1: Structural Hamming Distance (Lower is better)
    bars1 = ax1.bar(method_names, shds, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax1.set_title("(A) Structural Hamming Distance (SHD)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("SHD (Lower is Better)", fontsize=11)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax1.set_xticklabels(method_names, rotation=10, ha="right", fontsize=9)

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 0.1, f"{h:.2f}", ha="center", va="bottom", fontweight="bold")

    # Panel 2: True Positive Rate / Recall (%) (Higher is better)
    bars2 = ax2.bar(method_names, tprs, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax2.set_title("(B) True Positive Rate (DAG Edge Recall %)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("TPR (%) (Higher is Better)", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax2.set_xticklabels(method_names, rotation=10, ha="right", fontsize=9)

    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    output_path = "benchmarks/figures/fig8_neural_graph_recovery.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    console.print(f"[bold green]Saved publication figure to {output_path}[/bold green]")


if __name__ == "__main__":
    main()
