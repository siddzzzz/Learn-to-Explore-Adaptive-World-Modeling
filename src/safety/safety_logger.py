"""Safety logger and quantitative metrics aggregator for Safe-BED benchmarks.

Tracks cumulative constraint violations, minimum safety barrier margins,
safe action filter pass rates, and entropy reduction histories.
"""

from __future__ import annotations
from typing import Any, Dict, List
import numpy as np


class SafetyMetricsLogger:
    """Aggregates and computes statistical metrics for safety-constrained active learning."""

    def __init__(self, risk_delta: float = 0.05):
        self.risk_delta = risk_delta
        self.reset()

    def reset(self) -> None:
        self.step_count: int = 0
        self.cumulative_violations: int = 0
        self.thermal_violations: int = 0
        self.toxic_violations: int = 0
        self.barrier_margins: List[float] = []
        self.filter_approvals: int = 0
        self.total_proposals: int = 0
        self.entropy_history: List[float] = []
        self.temperatures: List[float] = []
        self.toxic_waste: List[float] = []

    def record_step(
        self,
        is_safe: bool,
        worst_margin: float,
        margins: Dict[str, Any],
        filter_approved: bool = True,
        entropy: float = 0.0,
        temperature: float = 300.0,
        toxic_waste: float = 0.0,
    ) -> None:
        self.step_count += 1
        if not is_safe:
            self.cumulative_violations += 1
        if not margins.get("is_thermal_safe", True):
            self.thermal_violations += 1
        if not margins.get("is_toxic_safe", True):
            self.toxic_violations += 1

        self.barrier_margins.append(float(worst_margin))
        if filter_approved:
            self.filter_approvals += 1
        self.total_proposals += 1

        self.entropy_history.append(float(entropy))
        self.temperatures.append(float(temperature))
        self.toxic_waste.append(float(toxic_waste))

    def get_summary(self) -> Dict[str, Any]:
        """Returns quantitative metrics dictionary for publication reporting."""
        min_margin = float(np.min(self.barrier_margins)) if self.barrier_margins else 0.0
        mean_margin = float(np.mean(self.barrier_margins)) if self.barrier_margins else 0.0
        approval_rate = (
            float(self.filter_approvals) / float(self.total_proposals)
            if self.total_proposals > 0
            else 1.0
        )
        final_entropy = self.entropy_history[-1] if self.entropy_history else 0.0

        return {
            "total_steps": self.step_count,
            "cumulative_violations": self.cumulative_violations,
            "thermal_violations": self.thermal_violations,
            "toxic_violations": self.toxic_violations,
            "worst_safety_margin": min_margin,
            "mean_safety_margin": mean_margin,
            "filter_approval_rate_pct": approval_rate * 100.0,
            "final_entropy": final_entropy,
            "max_temperature": float(np.max(self.temperatures)) if self.temperatures else 300.0,
            "max_toxic_waste": float(np.max(self.toxic_waste)) if self.toxic_waste else 0.0,
        }
