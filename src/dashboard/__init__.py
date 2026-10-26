"""Dashboard package for interactive scientific experimentation and visualization."""

from src.dashboard.config import PALETTE, CUSTOM_CSS, NAVIGATION_PAGES
from src.dashboard.state_engine import DashboardStateEngine
from src.dashboard.plots import (
    plot_belief_distribution,
    plot_entropy_trajectory,
    plot_surprise_gauge,
    plot_batch_yields
)

__all__ = [
    "PALETTE",
    "CUSTOM_CSS",
    "NAVIGATION_PAGES",
    "DashboardStateEngine",
    "plot_belief_distribution",
    "plot_entropy_trajectory",
    "plot_surprise_gauge",
    "plot_batch_yields"
]
