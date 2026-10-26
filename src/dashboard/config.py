"""Dashboard configuration, themes, typography, and styling constants."""

from __future__ import annotations
from typing import Dict, Any

# Scientific Visual Palette
PALETTE = {
    "hex_wm": "#10B981",       # Emerald green for HEx-WM
    "oracle": "#3B82F6",       # Blue for Bayes-optimal Oracle
    "plan2explore": "#F59E0B", # Amber for Plan2Explore
    "curiosity": "#EC4899",    # Pink/Magenta for ICM
    "bayesian_opt": "#8B5CF6", # Purple for GP-BO
    "random": "#6B7280",       # Slate gray for Random
    "danger": "#EF4444",       # Red for anomalies / waste
    "card_bg": "#1E293B",      # Slate dark card
    "bg_dark": "#0F172A",      # Deep midnight background
    "text_main": "#F8FAFC",    # Bright white text
    "text_muted": "#94A3B8",   # Soft silver text
    "accent": "#06B6D4"        # Cyan accent
}

CUSTOM_CSS = """
<style>
    /* Global styles */
    .stApp {
        background-color: #0B1120;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Metrics and cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    
    .metric-title {
        color: #94A3B8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    
    .metric-value {
        color: #F8FAFC;
        font-size: 1.7rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    
    .badge-success {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-danger {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .badge-info {
        background-color: rgba(6, 182, 212, 0.15);
        color: #22D3EE;
        border: 1px solid rgba(6, 182, 212, 0.3);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
    }
</style>
"""

NAVIGATION_PAGES = [
    "Chemical Reactor Studio",
    "Live Rule Mutation Cockpit",
    "Zero-Shot Synthesis Challenge",
    "Causal Lab Explorer",
    "Benchmark & Paper Results"
]
