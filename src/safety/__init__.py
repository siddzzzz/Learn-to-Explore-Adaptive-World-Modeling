"""Safety-Constrained Bayesian Experimental Design (Safe-HEx-WM) Module.

Provides Control Barrier Functions (CBFs), Epistemic Safety Filters with
Distribution-Free Cantelli Bounds, Safe Trajectory CEM Planners, and the
complete SafeHExWMAgent.
"""

from src.safety.barrier_function import (
    ControlBarrierFunction,
    create_box_barrier,
    create_thermal_barrier,
    create_toxicity_barrier,
)
from src.safety.epistemic_safety_filter import (
    EpistemicSafetyFilter,
)
from src.safety.safe_trajectory_cem import (
    SafeTrajectoryCEMPlanner,
)
from src.safety.safe_hex_agent import (
    SafeHExWMAgent,
)

__all__ = [
    "ControlBarrierFunction",
    "create_box_barrier",
    "create_thermal_barrier",
    "create_toxicity_barrier",
    "EpistemicSafetyFilter",
    "SafeTrajectoryCEMPlanner",
    "SafeHExWMAgent",
]
