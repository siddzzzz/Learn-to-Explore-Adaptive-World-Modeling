"""Benchmark 1: Sample Efficiency of Hypothesis Discovery.

Compares HEx-WM against Baselines (Oracle, Plan2Explore, ICM Curiosity, Random)
measuring Shannon entropy reduction rate across environment steps.
"""

from __future__ import annotations
import json
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from src.envs.causal_lab import CausalLabEnv, StructuralCausalModel
from src.agents import HExWMAgent
from src.agents.baselines import RandomAgent, CuriosityAgent, Plan2ExploreAgent, OracleAgent

console = Console()


def run_single_trial(agent_name: str, seed: int, max_steps: int = 15, ground_truth_id: int = 1) -> List[float]:
    """Runs a single exploration trial and returns entropy trajectory."""
    env = CausalLabEnv(num_variables=4, ground_truth_id=ground_truth_id, max_steps=max_steps)
    hypothesis_space = env.get_hypothesis_space()
    evaluators = [StructuralCausalModel(4, h, noise_std=0.0) for h in hypothesis_space]
    action_decoder = env.decode_action
    rng = np.random.RandomState(seed)

    if agent_name == "HEx-WM (Ours)":
        agent = HExWMAgent(
            hypothesis_evaluators=evaluators,
            num_actions=env.num_actions,
            action_decoder_fn=action_decoder,
            planning_horizon=2,
            num_cem_samples=32,
            num_cem_iters=3,
            rng=rng
        )
    elif agent_name == "Oracle (Bayes-Optimal)":
        agent = OracleAgent(
            num_actions=env.num_actions,
            hypothesis_space=hypothesis_space,
            action_decoder_fn=action_decoder
        )
    elif agent_name == "Plan2Explore":
        agent = Plan2ExploreAgent(
            num_actions=env.num_actions,
            hypothesis_evaluators=evaluators,
            action_decoder_fn=action_decoder,
            rng=rng
        )
    elif agent_name == "ICM Curiosity":
        agent = CuriosityAgent(
            num_actions=env.num_actions,
            hypothesis_evaluators=evaluators,
            action_decoder_fn=action_decoder,
            rng=rng
        )
    elif agent_name == "Random":
        agent = RandomAgent(
            num_actions=env.num_actions,
            hypothesis_evaluators=evaluators,
            action_decoder_fn=action_decoder,
            rng=rng
        )
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    obs, _ = env.reset(seed=seed)
    agent.reset()
    entropies = [agent.belief_tracker.get_shannon_entropy() if hasattr(agent, "belief_tracker") else agent.oracle_filter.get_shannon_entropy()]

    for step in range(max_steps):
        action = agent.select_action(obs[:4])
        next_obs, _, _, _, _ = env.step(action)
        metrics = agent.observe_transition(obs[:4], action, next_obs[:4])
        entropies.append(metrics["entropy"])
        obs = next_obs

    return entropies


def main():
    console.rule("[bold cyan]Running Benchmark 1: Sample Efficiency of Hypothesis Discovery[/bold cyan]")
    os.makedirs("benchmarks/results", exist_ok=True)
    os.makedirs("benchmarks/figures", exist_ok=True)

    agents = [
        "Oracle (Bayes-Optimal)",
        "HEx-WM (Ours)",
        "Plan2Explore",
        "ICM Curiosity",
        "Random"
    ]

    seeds = [10, 21, 32, 43, 54]
    max_steps = 15
    results: Dict[str, Dict[str, List[float]]] = {}

    table = Table(title="Benchmark 1: Final Posterior Shannon Entropy (Nats)")
    table.add_column("Agent", style="cyan", no_wrap=True)
    table.add_column("Initial Entropy", style="magenta")
    table.add_column("Final Mean Entropy", style="green")
    table.add_column("Std Err", style="yellow")
    table.add_column("Discovery Rate (<0.1 nats)", style="bold blue")

    for agent_name in agents:
        trial_curves = []
        for seed in seeds:
            gt_id = (seed % 7) + 1  # Test across diverse ground-truth rules
            curve = run_single_trial(agent_name, seed, max_steps=max_steps, ground_truth_id=gt_id)
            trial_curves.append(curve)

        curves_arr = np.array(trial_curves)  # [num_seeds, max_steps + 1]
        mean_curve = np.mean(curves_arr, axis=0).tolist()
        std_err = (np.std(curves_arr, axis=0) / np.sqrt(len(seeds))).tolist()
        final_vals = curves_arr[:, -1]
        success_rate = float(np.mean(final_vals < 0.1) * 100.0)

        results[agent_name] = {
            "mean": mean_curve,
            "std_err": std_err,
            "final_mean": float(np.mean(final_vals)),
            "final_stderr": float(np.std(final_vals) / np.sqrt(len(seeds))),
            "success_rate": success_rate
        }

        table.add_row(
            agent_name,
            f"{mean_curve[0]:.3f}",
            f"{results[agent_name]['final_mean']:.3f}",
            f"±{results[agent_name]['final_stderr']:.3f}",
            f"{success_rate:.0f}%"
        )

    console.print(table)

    # Save JSON results
    json_path = "benchmarks/results/entropy_reduction.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[green]Saved benchmark data to {json_path}[/green]")

    # Plot publication-grade figure
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    colors = {
        "Oracle (Bayes-Optimal)": "#10B981", # Emerald
        "HEx-WM (Ours)": "#3B82F6",          # Blue
        "Plan2Explore": "#F59E0B",           # Amber
        "ICM Curiosity": "#EF4444",          # Red
        "Random": "#6B7280"                  # Gray
    }

    styles = {
        "Oracle (Bayes-Optimal)": "--",
        "HEx-WM (Ours)": "-",
        "Plan2Explore": "-.",
        "ICM Curiosity": ":",
        "Random": ":"
    }

    steps = list(range(max_steps + 1))
    for agent_name in agents:
        mean = np.array(results[agent_name]["mean"])
        err = np.array(results[agent_name]["std_err"])
        color = colors.get(agent_name, "#000000")
        ls = styles.get(agent_name, "-")

        ax.plot(steps, mean, label=agent_name, color=color, linestyle=ls, linewidth=2.5)
        ax.fill_between(steps, np.maximum(0, mean - err), mean + err, color=color, alpha=0.15)

    ax.set_title("Sample Efficiency of Hypothesis Discovery: Shannon Entropy Reduction", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Environment Interaction Steps (Experiments)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Posterior Shannon Entropy H(b_t) [Nats]", fontsize=11, fontweight="medium")
    ax.set_ylim(-0.05, 2.2)
    ax.set_xlim(0, max_steps)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    fig_path = "benchmarks/figures/fig1_entropy_reduction.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    console.print(f"[bold green]Generated publication figure: {fig_path}[/bold green]")


if __name__ == "__main__":
    main()
