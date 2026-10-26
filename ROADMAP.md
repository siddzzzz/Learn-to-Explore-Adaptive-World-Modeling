# Research Roadmap & Implementation Checklist: HEx-WM

## High-Standard NeurIPS/ICML Research Plan

---

### [x] Phase 0: Mathematical Foundations & Theoretical Framework
- [x] Formalize Epistemic POMDP $(\mathcal{S}, \mathcal{A}, \Theta, \mathcal{T}_\theta, \Omega, \mathcal{O})$.
- [x] Formulate Bayesian Experimental Design (BED) & Expected Information Gain (EIG).
- [x] Derive Contrastive Hypothesis Discrepancy (CHD) via Generalized Jensen-Shannon Divergence.
- [x] Formal Proof: Theorem 1 (Aleatoric Invariance / Noisy TV Immunity of CHD).
- [x] Complete formal literature matrix comparing HEx-WM with Plan2Explore, ICM, RND, VIME, and Alchemy.
- [x] Algorithm 1 Pseudocode specification.
- *Artifact*: `docs/RESEARCH_FOUNDATIONS.md`

---

### [x] Phase 1: Environment Engine (Causal Lab Benchmark)
- [x] Implement `src/envs/causal_lab/dag_generator.py`:
  - Parameterized Structural Causal Models (SCMs).
  - Graph isomorphism and candidate hypothesis space generation.
  - Interventional semantics ($\text{do}(X=x)$).
- [x] Implement `src/envs/causal_lab/lab_env.py`:
  - Gymnasium standard API (`reset`, `step`, `render`).
  - Active experiment observation emission + controllable aleatoric noise channel ("Noisy TV").
  - Hidden rule oracle API: `get_hypothesis_space()`, `get_ground_truth()`, `mutate_rule()`.
- [x] Implement `src/envs/causal_lab/oracle.py`:
  - Exact analytical Bayesian posterior and entropy computation.
- [x] Tests: `tests/test_causal_lab.py` (Verify SCM transitions, Gymnasium compatibility, determinism, seed reproducibility).

---

### [x] Phase 2: Hypothesis Representation & Bayesian Belief Tracker
- [x] Implement `src/models/hypothesis.py`:
  - Explicit & parameterized structural mechanism definitions.
- [x] Implement `src/models/belief_tracker.py`:
  - Bayesian belief state $b_t = (w_1, \dots, w_K)$.
  - Exact log-likelihood belief updating with numerical stability tricks (LogSumExp).
  - Shannon Entropy $\mathcal{H}(b_t)$, normalized uncertainty, and maximum a posteriori (MAP) rule extractor.
- [x] Implement `src/models/ensemble_world_model.py`:
  - PyTorch-based epistemic deep ensemble for continuous dynamics prediction.
- [x] Tests: `tests/test_belief_tracker.py` (Verify Bayes update correctness, entropy monotonicity, edge cases).

---

### [x] Phase 3: Active Planning & Acquisition Functions
- [x] Implement `src/planning/acquisition.py`:
  - `compute_chd()`: Contrastive Hypothesis Discrepancy (Jensen-Shannon divergence).
  - `compute_bald()`: Bayesian Active Learning by Disagreement.
  - `compute_ensemble_variance()`: Plan2Explore baseline metric.
  - `compute_prediction_error()`: ICM curiosity baseline metric.
- [x] Implement `src/planning/trajectory_cem.py`:
  - Multi-step Cross-Entropy Method optimizer over action sequences.
  - Imagined trajectory rollouts under current belief dynamics.
- [x] Implement `src/planning/mppi.py`:
  - Model Predictive Path Integral controller for smooth continuous trajectories.
- [x] Tests: `tests/test_planning.py` (Verify CHD computation, CEM optimization convergence, noisy TV invariance).

---

### [x] Phase 4: Full Agent Architecture & Baseline Suite
- [x] Implement `src/agents/hex_wm_agent.py`:
  - Dual-phase agent: (1) Active Epistemic Discovery $\to$ (2) Goal-directed task execution.
- [x] Implement Baseline Agents:
  - `src/agents/baselines/random_agent.py`: Uniform random explorer.
  - `src/agents/baselines/curiosity_agent.py`: ICM-style curiosity agent.
  - `src/agents/baselines/plan2explore_agent.py`: Ensemble disagreement agent.
  - `src/agents/baselines/oracle_agent.py`: Greedy Bayes-optimal upper bound.
- [x] Tests: `tests/test_agents.py` (Verify end-to-end runs for all agents on the benchmark).

---

### [x] Phase 5: Empirical Benchmarking & Publication Results
- [x] `benchmarks/run_entropy_benchmark.py`:
  - Evaluate sample efficiency (Entropy reduction vs. environment steps).
  - Generate publication-quality comparison curves (HEx-WM vs. Baselines).
- [x] `benchmarks/run_noisy_tv_benchmark.py`:
  - Test robustness against aleatoric noise distractors.
- [x] `benchmarks/run_zero_shot_planning.py`:
  - Evaluate downstream task success rate after active exploration.
- [x] `benchmarks/run_rule_mutation.py`:
  - Test non-stationary adaptation when the ground truth rule changes mid-evaluation.
- [x] Generate comprehensive paper figures and summary report (`docs/EXPERIMENT_RESULTS.md`).

---

### [x] Phase 6: Interactive Scientific Studio (Streamlit Web Dashboard)
- [x] **Commit 1**: Dashboard Architecture & Configuration (`src/dashboard/config.py`, `src/dashboard/__init__.py`)
- [x] **Commit 2**: Session State Engine for Live Simulation (`src/dashboard/state_engine.py`)
- [x] **Commit 3**: Interactive Plotly Visualizers (`src/dashboard/plots.py`)
- [x] **Commit 4**: Chemical Reactor Experiment Studio Page (`src/dashboard/pages/reactor_studio.py`)
- [x] **Commit 5**: Live Rule Mutation Cockpit (`src/dashboard/pages/mutation_cockpit.py`)
- [x] **Commit 6**: Zero-Shot Synthesis Challenge Page (`src/dashboard/pages/synthesis_challenge.py`)
- [x] **Commit 7**: Causal Lab DAG Explorer (`src/dashboard/pages/causal_studio.py`)
- [x] **Commit 8**: Benchmark Explorer & Publication Results Viewer (`src/dashboard/pages/benchmark_viewer.py`)
- [x] **Commit 9**: Streamlit Main Entrypoint & Unit Tests (`app.py`, `run_dashboard.py`, `tests/test_dashboard.py`)
- [x] **Commit 10**: Documentation & Quickstart Integration (`README.md`, `ROADMAP.md`)

---

### [x] Phase 7: End-to-End Neural Causal Induction (Differentiable Neural World Models)
- [x] **Commit 1**: Mathematical formulation & theory document (`docs/NEURAL_CAUSAL_INDUCTION.md`)
- [x] **Commit 2**: NOTEARS continuous acyclicity constraint & graph utilities (`src/models/causal/dag_utils.py`)
- [x] **Commit 3**: Variational graph posterior with Gumbel-Softmax sampling (`src/models/causal/graph_posterior.py`)
- [x] **Commit 4**: Modular Neural SCM with Pearl do-operator masking (`src/models/causal/neural_scm.py`)
- [x] **Commit 5**: Differentiable causal world model unifying posterior & dynamics (`src/models/causal/causal_world_model.py`)
- [x] **Commit 6**: Contrastive Graph Discrepancy (CGD) acquisition metric (`src/planning/causal_acquisition.py`)
- [x] **Commit 7**: End-to-end neural causal agent (`src/agents/neural_causal_agent.py`)
- [x] **Commit 8**: Unit test suite for neural causal discovery (`tests/test_neural_causal.py`)
- [x] **Commit 9**: Package exports integration (`src/models/causal/__init__.py`, `src/models/__init__.py`, `src/agents/__init__.py`)
- [x] **Commit 10**: Documentation & Roadmap update (`README.md`, `ROADMAP.md`)

---

### Next Research Phases
- [ ] **Phase 8: Safety-Constrained Bayesian Experimental Design (Safe-HEx-WM)**
- [ ] **Phase 9: Complete Conference Paper LaTeX Submission Package (`paper/`)**
