# HEx-WM: Hypothesis-Experimenting World Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Venue: NeurIPS/ICML Target](https://img.shields.io/badge/Target-NeurIPS%20%2F%20ICML-red.svg)]()

> **An AI that discovers the governing rules of an unfamiliar environment through Bayesian experimental design, active hypothesis discrimination, and model-based planning.**

---

## Overview

Modern reinforcement learning agents often explore using heuristic novelty (ICM, RND) or continuous model disagreement (Plan2Explore). These methods fail in environments with irreducible stochasticity (the **Noisy TV** problem) and cannot reason over structured, competing causal hypotheses.

**HEx-WM** formalizes active exploration as **Sequential Bayesian Experimental Design (sBED)** in an Epistemic POMDP:
1. **Bayesian Hypothesis Particle Posterior**: Explicitly tracks belief distributions over competing structural mechanisms $p(\theta \mid \mathcal{D})$.
2. **Contrastive Hypothesis Discrepancy (CHD)**: Measures the Generalized Jensen-Shannon Divergence between model predictions, provably eliminating aleatoric distraction while maximizing information gain.
3. **Non-Myopic Trajectory Planning**: Employs Cross-Entropy Method (CEM) and Model Predictive Path Integral (MPPI) control to sequence prerequisite actions that set up decisive experiments.
4. **Adaptive Exploitation**: Seamlessly transitions from scientific exploration to goal-directed execution, rapidly detecting and recovering from unannounced rule mutations.

For complete mathematical derivations and proofs, see [`docs/RESEARCH_FOUNDATIONS.md`](docs/RESEARCH_FOUNDATIONS.md).  
For the detailed implementation roadmap, see [`ROADMAP.md`](ROADMAP.md).

---

## Repository Structure

```
├── docs/                     # Mathematical proofs, derivations & research foundations
├── src/
│   ├── envs/                 # Gymnasium environments with latent SCMs & Noisy TV
│   │   └── causal_lab/       # Analytical Causal DAG benchmark (closed-form Bayes optimality)
│   ├── models/               # Bayesian belief trackers & epistemic dynamics models
│   ├── planning/             # Non-myopic CEM / MPPI planners & CHD acquisition metrics
│   └── agents/               # HEx-WM and baselines (Plan2Explore, ICM, Random, Oracle)
├── tests/                    # Unit, statistical, and mathematical correctness tests
├── benchmarks/               # Reproducible benchmark experiments and figure generators
└── configs/                  # Experiment configurations
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -e .

# 2. Launch the Interactive Scientific Studio (Dashboard)
streamlit run app.py
# (or: python run_dashboard.py)

# 3. Run all verification tests (25 tests passing)
pytest -v

# 4. Run the benchmark suite
python benchmarks/run_chemical_lab_benchmark.py  # Real-world Self-Driving Lab benchmark
python benchmarks/run_entropy_benchmark.py       # Sample efficiency benchmark
python benchmarks/run_noisy_tv_benchmark.py      # Noisy TV (Theorem 1) benchmark
python benchmarks/run_rule_mutation.py           # Non-stationary adaptation benchmark
python benchmarks/run_zero_shot_planning.py      # Downstream task planning
```

---

## Interactive Scientific Studio (`app.py`)

The repository includes a live Streamlit visualizer for interactive experimentation:
1. **Chemical Reactor Studio**: Run autonomous batch discovery or manually turn reactor knobs (temperature, feeds, duration) and observe real-time Bayesian belief updates.
2. **Live Rule Mutation Cockpit**: Silently swap reaction kinetics mid-run to test real-time anomaly detection and rapid recovery.
3. **Zero-Shot Synthesis Challenge**: Configure custom drug manufacturing orders with strict thermal safety constraints and test physical synthesis against Bayesian Optimization.
4. **Causal Lab Explorer**: Interactively test Pearl's do-operator $\text{do}(X_i = v)$ on DAG networks and inspect the Noisy TV aleatoric channel.
5. **Benchmark & Paper Results**: Explore high-resolution figures and download raw benchmark JSON data.