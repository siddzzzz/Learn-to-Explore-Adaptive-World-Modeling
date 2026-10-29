"""Benchmark Runner: Interactive Active Noisy-TV Distractor.

Directly benchmarks the Noisy TV trap:
Agents can choose between performing genuine causal interventions or clicking
aleatoric distractor channels (Noisy TV).
Quantifies:
- Distractor Fixation Rate (% of steps trapped on the TV)
- Entropy Reduction Trajectory
- Steps to rule identification
"""

from __future__ import annotations
import json
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from src.envs.causal_lab.active_noisy_tv_env import ActiveNoisyTVLabEnv
from src.envs.causal_lab.scm import StructuralCausalModel
from src.agents import HExWMAgent
from src.agents.baselines import CuriosityAgent, Plan2ExploreAgent, RandomAgent

console = Console()


def evaluate_active_noisy_tv_agent(
    agent_name: str,
    seeds: List[int],
    max_steps: int = 25,
    noisy_tv_std: float = 3.0,
) -> Dict[str, Any]:
    """Runs an agent across multiple seeds in the ActiveNoisyTVLabEnv."""
    all_entropies = []
    all_tv_fixations = []
    all_steps_to_collapse = []

    for seed in seeds:
        env = ActiveNoisyTVLabEnv(
            num_variables=4,
            noisy_tv_dim=4,
            num_tv_actions=4,
            noisy_tv_std=noisy_tv_std,
            max_steps=max_steps,
            ground_truth_id=(seed % 7) + 1,
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
                rng=rng,
            )
        elif agent_name == "Plan2Explore":
            agent = Plan2ExploreAgent(
                num_actions=env.num_actions,
                hypothesis_evaluators=evaluators,
                action_decoder_fn=action_decoder,
                rng=rng,
            )
        elif agent_name == "ICM Curiosity":
            agent = CuriosityAgent(
                num_actions=env.num_actions,
                hypothesis_evaluators=evaluators,
                action_decoder_fn=action_decoder,
                rng=rng,
            )
        elif agent_name == "Random":
            agent = RandomAgent(
                num_actions=env.num_actions,
                hypothesis_evaluators=evaluators,
                action_decoder_fn=action_decoder,
                rng=rng,
            )
        else:
            raise ValueError(f"Unknown agent: {agent_name}")

        obs, _ = env.reset(seed=seed)
        agent.reset()

        entropy_history = [agent.belief_tracker.get_shannon_entropy()]
        distractor_count = 0
        step_collapsed = max_steps

        for step in range(max_steps):
            action = agent.select_action(obs)
            next_obs, _, _, _, info = env.step(action)
            metrics = agent.observe_transition(obs, action, next_obs)

            if info.get("is_distractor_action", False):
                distractor_count += 1

            ent = agent.belief_tracker.get_shannon_entropy()
            entropy_history.append(ent)
            if ent < 0.05 and step_collapsed == max_steps:
                step_collapsed = step + 1

            obs = next_obs

        all_entropies.append(entropy_history)
        all_tv_fixations.append(distractor_count / float(max_steps))
        all_steps_to_collapse.append(step_collapsed)

    mean_entropy = np.mean(all_entropies, axis=0).tolist()
    mean_fixation = float(np.mean(all_tv_fixations)) * 100.0  # percentage
    mean_collapse = float(np.mean(all_steps_to_collapse))

    return {
        "entropy_curve": mean_entropy,
        "final_entropy": float(mean_entropy[-1]),
        "distractor_fixation_pct": mean_fixation,
        "steps_to_discovery": mean_collapse,
    }


def main():
    console.rule("[bold cyan]Running Active Noisy-TV Distractor Benchmark[/bold cyan]")
    os.makedirs("benchmarks/results", exist_ok=True)
    os.makedirs("benchmarks/figures", exist_ok=True)

    seeds = [10, 21, 32, 43, 54]
    agents = [
        "HEx-WM (Ours)",
        "Plan2Explore",
        "ICM Curiosity",
        "Random",
    ]

    results = {}
    table = Table(title="Active Noisy-TV Distractor Benchmark: Fixation vs Discovery")
    table.add_column("Agent", style="bold")
    table.add_column("Distractor Fixation Rate (%)", justify="right")
    table.add_column("Final Shannon Entropy", justify="right")
    table.add_column("Steps to Discovery", justify="right")

    for agent_name in agents:
        res = evaluate_active_noisy_tv_agent(agent_name, seeds=seeds, max_steps=20, noisy_tv_std=3.0)
        results[agent_name] = res
        table.add_row(
            agent_name,
            f"{res['distractor_fixation_pct']:.1f}%",
            f"{res['final_entropy']:.3f}",
            f"{res['steps_to_discovery']:.1f}" if res['steps_to_discovery'] < 20 else ">20",
        )

    console.print(table)

    # Save results json
    with open("benchmarks/results/active_noisy_tv_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot publication figure
    plot_active_noisy_tv_figure(results)


def plot_active_noisy_tv_figure(results: Dict[str, Any]):
    """Plots dual-panel publication figure showing distractor fixation and entropy collapse."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Entropy reduction over steps
    colors = {
        "HEx-WM (Ours)": "#10b981",  # Emerald
        "Plan2Explore": "#f59e0b",   # Amber
        "ICM Curiosity": "#ef4444",  # Crimson
        "Random": "#6b7280",         # Gray
    }

    for name, res in results.items():
        curve = res["entropy_curve"]
        ax1.plot(range(len(curve)), curve, label=name, color=colors.get(name, "#888"), linewidth=2.5)

    ax1.set_title("(A) Shannon Entropy Reduction under Active Noisy TV", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Environment Steps", fontsize=11)
    ax1.set_ylabel("Belief Shannon Entropy (nats)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, fontsize=10)

    # Panel 2: Distractor Fixation Rate (%)
    agent_names = list(results.keys())
    fixation_rates = [results[n]["distractor_fixation_pct"] for n in agent_names]
    bar_colors = [colors.get(n, "#888") for n in agent_names]

    bars = ax2.bar(agent_names, fixation_rates, color=bar_colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax2.set_title("(B) Distractor Fixation Rate (% of Steps Clicked TV)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("TV Fixation Rate (%)", fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 2.0, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    output_path = "benchmarks/figures/fig5_active_noisy_tv_benchmark.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    console.print(f"[bold green]Saved publication figure to {output_path}[/bold green]")


if __name__ == "__main__":
    main()
