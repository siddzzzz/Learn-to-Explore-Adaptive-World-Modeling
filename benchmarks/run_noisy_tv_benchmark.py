"""Benchmark 2: The Noisy TV Robustness Experiment (Theorem 1 Validation).

Tests agent performance as irreducible aleatoric observation noise increases:
σ_noise ∈ {0.0, 0.5, 1.0, 2.0, 4.0}.
Demonstrates Theorem 1: HEx-WM (via CHD) is invariant to aleatoric noise distractors.
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
from src.agents.baselines import RandomAgent, CuriosityAgent, Plan2ExploreAgent

console = Console()


def evaluate_noise_level(agent_name: str, noise_std: float, seeds: List[int], steps: int = 10) -> float:
    """Evaluates an agent under a specific Noisy TV distractor level."""
    final_entropies = []

    for seed in seeds:
        env = CausalLabEnv(
            num_variables=4,
            noisy_tv_dim=4,
            noisy_tv_std=noise_std,
            max_steps=steps,
            ground_truth_id=(seed % 7) + 1
        )
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

        for step in range(steps):
            action = agent.select_action(obs[:4])
            next_obs, _, _, _, _ = env.step(action)
            metrics = agent.observe_transition(obs[:4], action, next_obs[:4])
            obs = next_obs

        final_entropies.append(agent.belief_tracker.get_shannon_entropy())

    return float(np.mean(final_entropies))


def main():
    console.rule("[bold cyan]Running Benchmark 2: Noisy TV Robustness (Theorem 1)[/bold cyan]")
    os.makedirs("benchmarks/results", exist_ok=True)
    os.makedirs("benchmarks/figures", exist_ok=True)

    noise_levels = [0.0, 0.5, 1.0, 2.0, 4.0]
    agents = [
        "HEx-WM (Ours)",
        "Plan2Explore",
        "ICM Curiosity",
        "Random"
    ]
    seeds = [12, 23, 34, 45, 56]

    results: Dict[str, List[float]] = {name: [] for name in agents}

    table = Table(title="Benchmark 2: Final Entropy vs Noisy TV Distractor Intensity (sigma)")
    table.add_column("Agent", style="cyan", no_wrap=True)
    for nl in noise_levels:
        table.add_column(f"sigma = {nl}", justify="center")

    for agent_name in agents:
        row_vals = []
        for nl in noise_levels:
            score = evaluate_noise_level(agent_name, nl, seeds)
            results[agent_name].append(score)
            row_vals.append(f"{score:.3f}")
        table.add_row(agent_name, *row_vals)

    console.print(table)

    json_path = "benchmarks/results/noisy_tv_results.json"
    with open(json_path, "w") as f:
        json.dump({"noise_levels": noise_levels, "results": results}, f, indent=2)

    # Plot publication figure
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    colors = {
        "HEx-WM (Ours)": "#3B82F6",
        "Plan2Explore": "#F59E0B",
        "ICM Curiosity": "#EF4444",
        "Random": "#6B7280"
    }

    markers = {
        "HEx-WM (Ours)": "o",
        "Plan2Explore": "s",
        "ICM Curiosity": "^",
        "Random": "d"
    }

    for agent_name in agents:
        y = results[agent_name]
        ax.plot(
            noise_levels,
            y,
            label=agent_name,
            color=colors.get(agent_name, "#000"),
            marker=markers.get(agent_name, "o"),
            linewidth=2.5,
            markersize=8
        )

    ax.set_title("Noisy TV Robustness: Resistance to Aleatoric Distractors (Theorem 1)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Distractor Noise Standard Deviation σ_noise", fontsize=11, fontweight="medium")
    ax.set_ylabel("Final Uncertainty H(b_T) after 10 Steps [Nats]", fontsize=11, fontweight="medium")
    ax.set_ylim(-0.05, 2.2)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    fig_path = "benchmarks/figures/fig2_noisy_tv_robustness.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    console.print(f"[bold green]Generated publication figure: {fig_path}[/bold green]")


if __name__ == "__main__":
    main()
