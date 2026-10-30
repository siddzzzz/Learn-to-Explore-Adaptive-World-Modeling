"""Benchmark Runner: Continuous Stirred Tank Reactor (CSTR) Dynamic Trajectory Control.

Benchmarks closed-loop continuous control under non-linear exothermic runaway curves:
1. Safe-HEx-WM (Dynamic Safe-CEM with Cantelli LCB on continuous thermal CBF)
2. Unconstrained HEx-WM (Active trajectory exploration without CBF)
3. Proportional-Integral (PI) Temperature Feedback Controller
4. Random Cooling Flow Policy

Quantifies:
- Thermal Runaway Explosions (T > 410.0 K breaches)
- Core Temperature Trajectories across 40 continuous control timesteps
- Minimum Safety Barrier Margin min_t (T_max - T)
- Information Gain / Entropy Collapse Speed
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from src.envs.chemical_reactor.cstr_env import ContinuousCSTREnv
from src.planning.cstr_trajectory_planner import CSTRDynamicTrajectoryPlanner

console = Console()


def run_single_cstr_trial(
    agent_type: str,
    seed: int,
    ground_truth_id: int = 2,
    max_steps: int = 30,
    max_temperature: float = 410.0,
) -> Dict[str, Any]:
    """Runs a single continuous CSTR trajectory trial."""
    env = ContinuousCSTREnv(
        ground_truth_id=ground_truth_id,
        dt=0.05,
        max_steps=max_steps,
        noise_std=0.05,
    )
    obs, _ = env.reset(seed=seed)

    planner = CSTRDynamicTrajectoryPlanner(
        horizon=3,
        num_candidates=25,
        num_iterations=2,
        dt=0.05,
        max_temperature=max_temperature,
        risk_delta=0.05,
        rng=np.random.RandomState(seed),
    )

    temperatures = [float(obs[1])]
    coolant_flows = []
    safety_violations = 0
    worst_margin = np.inf
    entropy_history = [planner.belief_tracker.get_shannon_entropy()]

    # PI Controller state for baseline
    integral_err = 0.0

    for step in range(max_steps):
        s_curr = env.state.copy()
        current_temp = float(s_curr[1])

        if agent_type == "Safe-HEx-WM (Ours)":
            action, is_safe, margin = planner.select_safe_action(s_curr, use_safety_barrier=True)
            planner_entropy = planner.belief_tracker.get_shannon_entropy()
        elif agent_type == "Unconstrained HEx-WM":
            action, is_safe, margin = planner.select_safe_action(s_curr, use_safety_barrier=False)
            planner_entropy = planner.belief_tracker.get_shannon_entropy()
        elif agent_type == "PI Feedback Control":
            # Traditional PI feedback trying to track 350K setpoint
            err = current_temp - 350.0
            integral_err += err * 0.05
            # Coolant flow qc in [0, 100] L/min
            qc_cmd = 40.0 + 0.8 * err + 0.1 * integral_err
            u = (qc_cmd - 50.0) / 50.0
            action = np.array([float(np.clip(u, -1.0, 1.0))], dtype=np.float32)
            planner_entropy = 1.386
        else:
            # Aggressive random exploratory actions (explores low cooling to see kinetics)
            action = np.random.uniform(-0.8, 0.4, size=(1,)).astype(np.float32)
            planner_entropy = 1.386

        # Step CSTR continuous physics
        next_obs, reward, terminated, truncated, info = env.step(action)
        qc_val = info["qc"]
        t_next = info["core_temperature"]

        margin_now = max_temperature - t_next
        if margin_now < worst_margin:
            worst_margin = margin_now

        if t_next > max_temperature:
            safety_violations += 1

        temperatures.append(t_next)
        coolant_flows.append(qc_val)

        # Update belief
        if agent_type in ["Safe-HEx-WM (Ours)", "Unconstrained HEx-WM"]:
            planner.observe_transition(s_curr, action, next_obs)
            entropy_history.append(planner.belief_tracker.get_shannon_entropy())
        else:
            entropy_history.append(planner_entropy)

        obs = next_obs
        if terminated:
            break

    return {
        "temperatures": temperatures,
        "coolant_flows": coolant_flows,
        "safety_violations": safety_violations,
        "worst_margin": float(worst_margin),
        "max_temperature": float(np.max(temperatures)),
        "final_entropy": float(entropy_history[-1]),
        "steps_completed": len(temperatures) - 1,
    }


def main():
    console.rule("[bold cyan]Running CSTR Dynamic Closed-Loop Trajectory Control Benchmark[/bold cyan]")
    os.makedirs("benchmarks/results", exist_ok=True)
    os.makedirs("benchmarks/figures", exist_ok=True)

    seeds = [15, 26, 37, 48, 59]
    agents = [
        "Safe-HEx-WM (Ours)",
        "Unconstrained HEx-WM",
        "PI Feedback Control",
        "Random",
    ]

    results = {}
    table = Table(title="CSTR Dynamic Benchmark: Closed-Loop Continuous Thermal Safety")
    table.add_column("Agent Paradigm", style="bold")
    table.add_column("Thermal Violations", justify="right")
    table.add_column("Worst Margin (h)", justify="right")
    table.add_column("Max Temp (K)", justify="right")
    table.add_column("Final Entropy", justify="right")

    for agent_name in agents:
        trials = []
        for s in seeds:
            res = run_single_cstr_trial(
                agent_type=agent_name,
                seed=s,
                ground_truth_id=2,  # High-Heat Runaway Exotherm model
                max_steps=25,
                max_temperature=410.0,
            )
            trials.append(res)

        mean_violations = float(np.mean([t["safety_violations"] for t in trials]))
        min_margin = float(np.min([t["worst_margin"] for t in trials]))
        max_temp = float(np.max([t["max_temperature"] for t in trials]))
        mean_entropy = float(np.mean([t["final_entropy"] for t in trials]))

        results[agent_name] = {
            "mean_violations": mean_violations,
            "min_margin": min_margin,
            "max_temp": max_temp,
            "final_entropy": mean_entropy,
            "sample_trajectory": trials[0]["temperatures"],
        }

        table.add_row(
            agent_name,
            f"[bold green]0.0[/bold green]" if mean_violations == 0 else f"[bold red]{mean_violations:.1f}[/bold red]",
            f"{min_margin:.1f}",
            f"{max_temp:.1f} K",
            f"{mean_entropy:.3f}",
        )

    console.print(table)

    with open("benchmarks/results/cstr_dynamic_results.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_cstr_figure(results)


def plot_cstr_figure(results: Dict[str, Any]):
    """Plots dual-panel publication figure: Dynamic Temperature Trajectory & Safety Violations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = {
        "Safe-HEx-WM (Ours)": "#10b981",  # Emerald
        "Unconstrained HEx-WM": "#f59e0b",  # Amber
        "PI Feedback Control": "#3b82f6",  # Blue
        "Random": "#6b7280",  # Gray
    }

    # Panel 1: Dynamic Continuous Temperature Trajectories
    for name, data in results.items():
        traj = data["sample_trajectory"]
        ax1.plot(range(len(traj)), traj, label=name, color=colors.get(name, "#888"), linewidth=2.2)

    ax1.axhline(410.0, color="#ef4444", linestyle="--", linewidth=2.0, label="Thermal Runaway Limit (410K)")
    ax1.set_title("(A) Dynamic Core Temperature Trajectory T(t)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Continuous Control Timesteps", fontsize=11)
    ax1.set_ylabel("Core Temperature T (K)", fontsize=11)
    ax1.set_ylim(320, 440)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, fontsize=9, loc="upper left")

    # Panel 2: Violations Bar Chart
    agent_names = list(results.keys())
    violations = [results[n]["mean_violations"] for n in agent_names]
    bar_colors = [colors.get(n, "#888") for n in agent_names]

    bars = ax2.bar(agent_names, violations, color=bar_colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax2.set_title("(B) Cumulative Thermal Runaway Violations", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Mean Violations per Trajectory", fontsize=11)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax2.set_xticklabels(agent_names, rotation=15, ha="right", fontsize=9)

    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 0.1, f"{h:.1f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    output_path = "benchmarks/figures/fig7_cstr_dynamic_benchmark.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    console.print(f"[bold green]Saved publication figure to {output_path}[/bold green]")


if __name__ == "__main__":
    main()
