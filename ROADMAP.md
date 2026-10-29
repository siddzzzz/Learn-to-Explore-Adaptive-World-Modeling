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

### [x] Phase 8: Safety-Constrained Bayesian Experimental Design (Safe-HEx-WM)
- [x] **Commit 1**: Mathematical formulation & Theorem 2 proof of Forward Invariance under Cantelli LCB (`docs/SAFE_EXPLORATION_THEORY.md`)
- [x] **Commit 2**: Control Barrier Functions (CBFs) & safe boundary specifications (`src/safety/barrier_function.py`)
- [x] **Commit 3**: Epistemic Safety Filter with distribution-free Cantelli Lower Confidence Bound (`src/safety/epistemic_safety_filter.py`)
- [x] **Commit 4**: Safe Trajectory CEM Planner with dynamic rejection & penalization (`src/safety/safe_trajectory_cem.py`)
- [x] **Commit 5**: Safe-HEx-WM Agent integrating active epistemic BED and provable CBF guarantees (`src/safety/safe_hex_agent.py`)
- [x] **Commit 6**: Comprehensive pytest suite for CBF bounds, epistemic filtering, and SafeHExWMAgent (`tests/test_safety.py`)
- [x] **Commit 7**: Safety module public API exports and agent registration (`src/safety/__init__.py`, `src/agents/__init__.py`)
- [x] **Commit 8**: Interactive Plotly visualizers for safety barrier gauge and safe action envelope (`src/dashboard/plots.py`)
- [x] **Commit 9**: Demonstration script comparing unconstrained vs safe exploration (`benchmarks/run_safe_exploration_demo.py`)
- [x] **Commit 10**: Documentation & Roadmap integration (`ROADMAP.md`, `README.md`)

---

### [x] Phase 10: Deep Empirical Expansion & Interactive Active Noisy-TV Benchmark
- [x] **Commit 1**: Formal active distractor protocol and mathematical fixation metrics (`docs/ACTIVE_NOISY_TV_PROTOCOL.md`)
- [x] **Commit 2**: Interactive environment `ActiveNoisyTVLabEnv` with controllable distractor channels (`src/envs/causal_lab/active_noisy_tv_env.py`)
- [x] **Commit 3**: Baseline agents updated with full sensory observation routing (`src/agents/baselines/curiosity_agent.py`, `plan2explore_agent.py`)
- [x] **Commit 4**: Pytest validation suite for active noisy-TV environment and agents (`tests/test_active_noisy_tv.py`)
- [x] **Commit 5**: Benchmark runner `benchmarks/run_active_noisy_tv_benchmark.py` evaluating multi-seed distractor fixation
- [x] **Commit 6**: Distractor Fixation Rate ($\Phi_{\text{TV}}$) and entropy trajectory data logging (`benchmarks/results/active_noisy_tv_results.json`)
- [x] **Commit 7**: Dual-panel publication figure generation (`benchmarks/figures/fig5_active_noisy_tv_benchmark.png`)
- [x] **Commit 8**: Empirical findings documentation in `docs/EXPERIMENT_RESULTS.md` and roadmap update

---

### Upcoming Empirical Benchmarks (Before Paper Drafting)
- [ ] **Phase B**: Rigorous Multi-Seed Safety-Constrained BED Benchmark (Quantitative Safety Violations vs. Discovery Speed across diverse risk tolerances $\delta$)
- [ ] **Phase C**: Continuous Stirred Tank Reactor (CSTR) Dynamic Control Lab (Continuous ODE control with thermal runaway curves)
- [ ] **Phase D**: Neural Causal Graph Discovery Benchmark (Structural Hamming Distance, TPR, and FDR on 10/20-node DAGs)
- [ ] **Phase E**: Systemic Scientific Ablation Matrix (EIG vs BALD vs CHD, Planning Horizons $H \in \{1, 3, 5\}$, Cantelli vs Gaussian bounds)



