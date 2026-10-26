"""HEx-WM Interactive Scientific Studio - Main Streamlit Application Entrypoint."""

import streamlit as st

from src.dashboard.config import CUSTOM_CSS, NAVIGATION_PAGES
from src.dashboard.pages.reactor_studio import render_reactor_studio
from src.dashboard.pages.mutation_cockpit import render_mutation_cockpit
from src.dashboard.pages.synthesis_challenge import render_synthesis_challenge
from src.dashboard.pages.causal_studio import render_causal_studio
from src.dashboard.pages.benchmark_viewer import render_benchmark_viewer

# 1. Page Configuration
st.set_page_config(
    page_title="HEx-WM: Hypothesis-Experimenting World Models",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Custom Theme Styles
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 3. Sidebar Navigation
with st.sidebar:
    st.markdown("## 🔬 **HEx-WM Studio**")
    st.markdown(
        "<span class='badge badge-success'>NeurIPS Target</span> "
        "<span class='badge badge-info'>AI for Science</span>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    selected_page = st.radio(
        "Navigation",
        options=NAVIGATION_PAGES,
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📚 Research Context")
    st.markdown(
        "- **Framework**: Sequential Bayesian Experimental Design (sBED)\n"
        "- **Core Metric**: Contrastive Hypothesis Discrepancy (CHD)\n"
        "- **Guarantee**: Provably immune to aleatoric noise (Theorem 1)\n"
        "- **Benchmark**: Real-world Chemical Reaction Kinetics"
    )
    st.markdown("---")
    st.caption("Learn-to-Explore-Adaptive-World-Modeling © 2026")

# 4. Route to Selected Page
if selected_page == "Chemical Reactor Studio":
    render_reactor_studio()
elif selected_page == "Live Rule Mutation Cockpit":
    render_mutation_cockpit()
elif selected_page == "Zero-Shot Synthesis Challenge":
    render_synthesis_challenge()
elif selected_page == "Causal Lab Explorer":
    render_causal_studio()
elif selected_page == "Benchmark & Paper Results":
    render_benchmark_viewer()
