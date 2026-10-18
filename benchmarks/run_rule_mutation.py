"""Benchmark 3: Non-Stationary Adaptation under Mid-Run Rule Mutation.

Evaluates how quickly HEx-WM detects an unannounced rule swap θ* -> θ_new,
broadens its belief, resumes scientific experiments, and recovers the new ground truth.
"""

from __future__ import annotations
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

from src.envs.causal_lab import CausalLabEnv, StructuralCausalModel
from src.agents import HExWMAgent

console = Console()


def run_mutation_experiment(mutation_step: int = 12, total_steps: int = 25, seed: int = 42):
    console.rule("[bold cyan]Running Benchmark 3: Non-Stationary Rule Mutation Recovery[/bold cyan]")
    os.makedirs("benchmarks/results", exist_ok=True)
    os.makedirs("benchmarks/figures", exist_ok=True)

    env = CausalLabEnv(num_variables=4, ground_truth_id=1, max_steps=total_steps)
    hypothesis_space = env.get_hypothesis_space()
    evaluators = [StructuralCausalModel(4, h, noise_std=0.0) for h in hypothesis_space]
    action_decoder = env.decode_action
    rng = np.random.RandomState(seed)

    agent = HExWMAgent(
        hypothesis_evaluators=evaluators,
        num_actions=env.num_actions,
        action_decoder_fn=action_decoder,
        planning_horizon=2,
        num_cem_samples=32,
        num_cem_iters=3,
        surprise_threshold=12.0,
        rng=rng
    )

    obs, _ = env.reset(seed=seed)
    agent.reset()

    history_entropy = [agent.belief_tracker.get_shannon_entropy()]
    history_h1_prob = [agent.belief_tracker.weights[1]]
    history_h4_prob = [agent.belief_tracker.weights[4]]
    history_surprise = [0.0]

    mutated = False

    for step in range(total_steps):
        if step == mutation_step:
            console.print(f"[bold red][RULE MUTATION] Step {step}: Environment silently mutating rule H1 (Sequential Chain) -> H4 (Diamond Graph)![/bold red]")
            env.mutate_rules(new_hypothesis_id=4)
            mutated = True

        action = agent.select_action(obs[:4])
        next_obs, _, _, _, _ = env.step(action)
        metrics = agent.observe_transition(obs[:4], action, next_obs[:4])
        obs = next_obs

        history_entropy.append(metrics["entropy"])
        history_h1_prob.append(metrics["weights"][1])
        history_h4_prob.append(metrics["weights"][4])
        history_surprise.append(metrics["surprise"])

        console.print(
            f"Step {step+1:02d} | "
            f"H(b): {metrics['entropy']:.3f} | "
            f"P(H1): {metrics['weights'][1]:.3f} | "
            f"P(H4): {metrics['weights'][4]:.3f} | "
            f"Surprise: {metrics['surprise']:.1f} | "
            f"{'[ANOMALY DETECTED]' if metrics['is_anomaly'] else ''}"
        )

    # Save results
    results = {
        "mutation_step": mutation_step,
        "total_steps": total_steps,
        "entropy": history_entropy,
        "p_h1": history_h1_prob,
        "p_h4": history_h4_prob,
        "surprise": history_surprise
    }
    with open("benchmarks/results/mutation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot dual-axis recovery trajectory
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, dpi=300)

    steps = list(range(len(history_entropy)))

    # Subplot 1: Hypothesis Probabilities
    ax1.plot(steps, history_h1_prob, label="P(H1: Sequential Chain)", color="#EF4444", linewidth=2.5)
    ax1.plot(steps, history_h4_prob, label="P(H4: Diamond Graph)", color="#10B981", linewidth=2.5)
    ax1.axvline(x=mutation_step, color="#000000", linestyle="--", alpha=0.8, label="Rule Mutation (H1 -> H4)")
    ax1.set_ylabel("Posterior Probability P(H)", fontsize=11, fontweight="medium")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Non-Stationary Adaptation: Rapid Recovery under Unannounced Rule Swap", fontsize=13, fontweight="bold")
    ax1.legend(frameon=True, facecolor="white", framealpha=0.9, loc="center right", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Subplot 2: Shannon Entropy & Surprise
    ax2.plot(steps, history_entropy, label="Shannon Entropy H(b_t)", color="#3B82F6", linewidth=2.5)
    ax2.axvline(x=mutation_step, color="#000000", linestyle="--", alpha=0.8)
    ax2.set_xlabel("Environment Steps", fontsize=11, fontweight="medium")
    ax2.set_ylabel("Entropy [Nats]", fontsize=11, fontweight="medium")
    ax2.set_ylim(-0.05, 2.2)
    ax2.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig_path = "benchmarks/figures/fig3_rule_mutation_recovery.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    console.print(f"[bold green]Generated publication figure: {fig_path}[/bold green]")


if __name__ == "__main__":
    run_mutation_experiment()
