"""Benchmark 4: Downstream Goal-Directed Task Planning (Exploitation Phase).

Evaluates the agent's ability to plan actions toward arbitrary target goals
using the world model discovered during the scientific exploration phase.
"""

from __future__ import annotations
import json
import os
import numpy as np
from rich.console import Console
from rich.table import Table

from src.envs.causal_lab import CausalLabEnv, StructuralCausalModel
from src.agents import HExWMAgent

console = Console()


def run_goal_planning_benchmark(num_tasks: int = 10, seed: int = 42):
    console.rule("[bold cyan]Running Benchmark 4: Zero-Shot Goal Planning[/bold cyan]")
    os.makedirs("benchmarks/results", exist_ok=True)

    rng = np.random.RandomState(seed)
    successes = []
    regrets = []

    table = Table(title="Benchmark 4: Downstream Goal Execution Results")
    table.add_column("Task ID", style="cyan", justify="center")
    table.add_column("Ground Truth Rule", style="magenta")
    table.add_column("Target Goal (X_3)", justify="center")
    table.add_column("Achieved (X_3)", justify="center")
    table.add_column("Error (MSE)", justify="center", style="green")
    table.add_column("Status", justify="center", style="bold")

    for task_id in range(num_tasks):
        gt_id = (task_id % 7) + 1
        env = CausalLabEnv(num_variables=4, ground_truth_id=gt_id, max_steps=20)
        hypothesis_space = env.get_hypothesis_space()
        evaluators = [StructuralCausalModel(4, h, noise_std=0.0) for h in hypothesis_space]
        action_decoder = env.decode_action

        agent = HExWMAgent(
            hypothesis_evaluators=evaluators,
            num_actions=env.num_actions,
            action_decoder_fn=action_decoder,
            planning_horizon=3,
            num_cem_samples=64,
            num_cem_iters=5,
            rng=np.random.RandomState(seed + task_id)
        )

        # 1. Scientific Exploration (5 steps)
        obs, _ = env.reset(seed=seed + task_id)
        agent.reset()
        for _ in range(5):
            action = agent.select_action(obs[:4])
            next_obs, _, _, _, _ = env.step(action)
            agent.observe_transition(obs[:4], action, next_obs[:4])
            obs = next_obs

        # 2. Downstream Task: Drive variable X_3 to target value
        target_val = float(rng.choice([-1.5, 0.0, 1.5, 2.0]))
        target_goal = np.array([0.0, 0.0, 0.0, target_val], dtype=np.float32)
        env.set_evaluation_mode(target_goal)
        agent.set_goal(target_goal)

        # 3. Plan and execute action
        action = agent.select_action(obs[:4])
        next_obs, reward, _, _, info = env.step(action)
        achieved = info["causal_state"]

        mse = float((achieved[3] - target_val) ** 2)
        is_success = mse < 0.25
        successes.append(is_success)
        regrets.append(mse)

        table.add_row(
            f"#{task_id+1:02d}",
            hypothesis_space[gt_id].name,
            f"{target_val:+.2f}",
            f"{achieved[3]:+.2f}",
            f"{mse:.4f}",
            "[green]SUCCESS[/green]" if is_success else "[yellow]NEAR[/yellow]"
        )

    console.print(table)
    mean_mse = float(np.mean(regrets))
    success_rate = float(np.mean(successes) * 100.0)
    console.print(f"[bold green]Overall Goal Achievement Rate: {success_rate:.1f}% | Mean MSE: {mean_mse:.4f}[/bold green]")

    with open("benchmarks/results/zero_shot_planning.json", "w") as f:
        json.dump({"success_rate": success_rate, "mean_mse": mean_mse, "regrets": regrets}, f, indent=2)


if __name__ == "__main__":
    run_goal_planning_benchmark()
