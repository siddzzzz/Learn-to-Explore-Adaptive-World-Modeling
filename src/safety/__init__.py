"""Safety-Constrained Bayesian Experimental Design (Safe-HEx-WM) Module.

Provides Control Barrier Functions (CBFs), Epistemic Safety Filters with
Distribution-Free Cantelli Bounds, Safe Trajectory CEM Planners, and the
complete SafeHExWMAgent.
"""

from src.safety.barrier_function import (
    ControlBarrierFunction,
    StateBoundBarrier,
    LinearBarrier,
)
from src.safety.epistemic_safety_filter import (
    EpistemicSafetyFilter,
    EpistemicCBFResult,
)
from src.safety.safe_trajectory_cem import (
    SafeTrajectoryCEMPlanner,
    SafeCEMResult,
)
from src.safety.safe_hex_agent import (
    SafeHExWMAgent,
)

__all__ = [
    "ControlBarrierFunction",
    "StateBoundBarrier",
    "LinearBarrier",
    "EpistemicSafetyFilter",
    "EpistemicCBFResult",
    "SafeTrajectoryCEMPlanner",
    "SafeCEMResult",
    "SafeHExWMAgent",
]
