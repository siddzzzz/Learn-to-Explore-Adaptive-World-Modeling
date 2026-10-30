# Protocol: Safety-Constrained Bayesian Experimental Design (Safe-BED) Benchmark

## 1. Problem Formalization
In automated laboratory chemistry and autonomous material synthesis, exploratory interventions must satisfy hard state constraints representing physical safety:
$$\mathcal{C} = \{ s \in \mathcal{S} \mid h_m(s) \ge 0, \; \forall m \in \{1, \dots, M\} \}$$
where $h_m(s)$ are Control Barrier Functions.

During the scientific discovery phase, the transition dynamics $\mathcal{T}_{\theta^*}(s_t, a_t)$ are unknown, creating a tension between:
1. **Epistemic Exploration**: Visiting extreme or high-energy regimes to maximize information gain (contrastive hypothesis discrepancy).
2. **Operational Safety**: Preventing physical boundary breaches ($h_m(s) < 0$).

## 2. Theoretical Guarantee Under Test: Theorem 2
Theorem 2 states that under candidate hypothesis models with epistemic uncertainty $\sigma_{h, m}$, evaluating barrier conditions with Cantelli's distribution-free Lower Confidence Bound:
$$\text{LCB}_{h, m}(s_t, a_t) = \mu_{h, m}(s_t, a_t) - \sqrt{\frac{1 - \delta}{\delta}} \sigma_{h, m}(s_t, a_t) \ge (1 - \alpha) h_m(s_t)$$
guarantees forward invariance of the safe set with probability at least $1 - \delta$:
$$P(s_{t+1} \in \mathcal{C} \mid s_t \in \mathcal{C}) \ge 1 - \delta$$
Furthermore, as scientific experiments eliminate invalid hypotheses ($\sigma_{h, m} \to 0$), the certifiable safe exploration envelope monotonically expands.

## 3. Benchmark Objectives & Baselines
We benchmark 5 distinct paradigms on the continuous batch chemical reactor:
1. **Safe-HEx-WM (Ours)**: Active hypothesis discrimination via Contrastive Hypothesis Discrepancy (CHD) + Cantelli Epistemic Safety Filter.
2. **Unconstrained HEx-WM**: Active CHD without barrier filtering.
3. **Constrained GP-BO**: Industry standard Gaussian Process Bayesian Optimization with Expected Improvement penalization for predicted constraint violations.
4. **Curiosity (ICM)**: Prediction-error curiosity.
5. **Random Exploration**: Uniform random action selection.

## 4. Sweeps and Measurements
- **Risk Tolerance Sweep**: $\delta \in \{0.01, 0.05, 0.10, 0.25\}$.
- **Safety Violation Count**: Total steps where $T > 420.0\text{ K}$ or $C_W > 0.35\text{ M}$.
- **Worst-Case Safety Margin**: $\min_t h(s_t)$.
- **Entropy Reduction Rate**: $\mathcal{H}(b_t)$ across batch steps $t \in \{1, \dots, 10\}$.
- **Downstream Yield**: Product purity % when commanded to synthesize target compound.
