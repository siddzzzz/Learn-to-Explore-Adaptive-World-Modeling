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

### [ ] Phase 1: Environment Engine (Causal Lab Benchmark)
- [ ] Implement `src/envs/causal_lab/dag_generator.py`:
  - Parameterized Structural Causal Models (SCMs).
  - Graph isomorphism and candidate hypothesis space generation.
  - Interventional semantics ($\text{do}(X=x)$).
- [ ] Implement `src/envs/causal_lab/lab_env.py`:
  - Gymnasium standard API (`reset`, `step`, `render`).
  - Active experiment observation emission + controllable aleatoric noise channel ("Noisy TV").
  - Hidden rule oracle API: `get_hypothesis_space()`, `get_ground_truth()`, `mutate_rule()`.
- [ ] Implement `src/envs/causal_lab/oracle.py`:
  - Exact analytical Bayesian posterior and entropy computation.
- [ ] Tests: `tests/test_causal_lab.py` (Verify SCM transitions, Gymnasium compatibility, determinism, seed reproducibility).

---

### [ ] Phase 2: Hypothesis Representation & Bayesian Belief Tracker
- [ ] Implement `src/models/hypothesis.py`:
  - Explicit & parameterized structural mechanism definitions.
- [ ] Implement `src/models/belief_tracker.py`:
  - Bayesian belief state $b_t = (w_1, \dots, w_K)$.
  - Exact log-likelihood belief updating with numerical stability tricks (LogSumExp).
  - Shannon Entropy $\mathcal{H}(b_t)$, normalized uncertainty, and maximum a posteriori (MAP) rule extractor.
- [ ] Implement `src/models/ensemble_world_model.py`:
  - PyTorch-based epistemic deep ensemble for continuous dynamics prediction.
- [ ] Tests: `tests/test_belief_tracker.py` (Verify Bayes update correctness, entropy monotonicity, edge cases).

---

### [ ] Phase 3: Active Planning & Acquisition Functions
- [ ] Implement `src/planning/acquisition.py`:
  - `compute_chd()`: Contrastive Hypothesis Discrepancy (Jensen-Shannon divergence).
  - `compute_bald()`: Bayesian Active Learning by Disagreement.
  - `compute_ensemble_variance()`: Plan2Explore baseline metric.
  - `compute_prediction_error()`: ICM curiosity baseline metric.
- [ ] Implement `src/planning/trajectory_cem.py`:
  - Multi-step Cross-Entropy Method optimizer over action sequences.
  - Imagined trajectory rollouts under current belief dynamics.
- [ ] Implement `src/planning/mppi.py`:
  - Model Predictive Path Integral controller for smooth continuous trajectories.
- [ ] Tests: `tests/test_planning.py` (Verify CHD computation, CEM optimization convergence, noisy TV invariance).

---

### [ ] Phase 4: Full Agent Architecture & Baseline Suite
- [ ] Implement `src/agents/hex_wm_agent.py`:
  - Dual-phase agent: (1) Active Epistemic Discovery $\to$ (2) Goal-directed task execution.
- [ ] Implement Baseline Agents:
  - `src/agents/baselines/random_agent.py`: Uniform random explorer.
  - `src/agents/baselines/curiosity_agent.py`: ICM-style curiosity agent.
  - `src/agents/baselines/plan2explore_agent.py`: Ensemble disagreement agent.
  - `src/agents/baselines/oracle_agent.py`: Greedy Bayes-optimal upper bound.
- [ ] Tests: `tests/test_agents.py` (Verify end-to-end runs for all agents on the benchmark).

---

### [ ] Phase 5: Empirical Benchmarking & Publication Results
- [ ] `benchmarks/run_entropy_benchmark.py`:
  - Evaluate sample efficiency (Entropy reduction vs. environment steps).
  - Generate publication-quality comparison curves (HEx-WM vs. Baselines).
- [ ] `benchmarks/run_noisy_tv_benchmark.py`:
  - Test robustness against aleatoric noise distractors.
- [ ] `benchmarks/run_zero_shot_planning.py`:
  - Evaluate downstream task success rate after active exploration.
- [ ] `benchmarks/run_rule_mutation.py`:
  - Test non-stationary adaptation when the ground truth rule changes mid-evaluation.
- [ ] Generate comprehensive paper figures and summary report (`docs/EXPERIMENT_RESULTS.md`).

---

### Git Commit Strategy
- **Commit 1**: `feat(core): theoretical foundations, math derivation, and research roadmap`
- **Commit 2**: `feat(envs): causal lab gymnasium benchmark with exact SCM and Noisy TV`
- **Commit 3**: `feat(models): Bayesian belief tracker and epistemic dynamics models`
- **Commit 4**: `feat(planning): contrastive acquisition functions and non-myopic CEM planner`
- **Commit 5**: `feat(agents): HEx-WM active discovery agent and baseline implementations`
- **Commit 6**: `feat(benchmarks): empirical evaluation suite, figures, and experimental results`
