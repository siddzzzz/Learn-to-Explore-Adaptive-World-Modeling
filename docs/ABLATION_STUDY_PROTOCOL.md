# Scientific Ablation Protocol: Component Isolation in HEx-WM

## 1. Motivation & Research Questions

To understand why HEx-WM achieves superior sample efficiency, distractor robustness, and safety invariance, we perform systematic ablations across three foundational axes:

### Axis 1: Epistemic Exploration Criterion
- **CHD (Contrastive Hypothesis Discrepancy)**:
  $$\alpha_{\text{CHD}}(a) = \sum_{k=1}^K w_k D_{\text{KL}}\left(p(y \mid a, H_k) \parallel \bar{p}(y \mid a)\right)$$
- **BALD (Bayesian Active Learning by Disagreement)**:
  $$\alpha_{\text{BALD}}(a) = \mathcal{H}\left(\sum_{k=1}^K w_k p(y \mid a, H_k)\right) - \sum_{k=1}^K w_k \mathcal{H}(p(y \mid a, H_k))$$
- **EIG Oracle (Expected Information Gain)**:
  $$\alpha_{\text{EIG}}(a) = \mathbb{E}_{y \sim \bar{p}}\left[\mathcal{H}(b) - \mathcal{H}(b \mid y)\right]$$
- **Prediction Error (Curiosity / ICM)**:
  $$\alpha_{\text{ICM}}(a) = \|y - \hat{y}(a)\|_2^2$$
- **Ensemble Variance (Plan2Explore)**:
  $$\alpha_{\text{P2E}}(a) = \frac{1}{K}\sum_{k=1}^K \|\hat{y}_k(a) - \bar{y}(a)\|_2^2$$

**Research Question 1**: How do these acquisition functions behave in the presence of uninformative aleatoric distractors ("Noisy TV") versus informative epistemic branching?

---

### Axis 2: Planning Horizon $H \in \{1, 2, 3, 5\}$
In non-myopic experimental design, experiments must sometimes be staged sequentially (e.g. state-dependent activation where an initial action opens a valve or sets temperature before an informative intervention is observable).
We evaluate cumulative information gain and steps to full hypothesis identification under receding horizon trajectory CEM across horizons $H = 1, 2, 3, 5$.

**Research Question 2**: Does multi-step trajectory planning provide quantifiable sample-efficiency benefits in delayed-information causal topologies?

---

### Axis 3: Epistemic Safety Bound Mechanisms
Under physical runaway dynamics (Arrhenius chemical reactor), we compare safety boundary certification mechanisms:
- **Cantelli LCB (Distribution-Free, Safe-HEx-WM)**:
  $$\hat{h}_{\text{Cantelli}}(s) = \mu_h(s) - \sigma_h(s)\sqrt{\frac{1-\delta}{\delta}}$$
- **Gaussian LCB**:
  $$\hat{h}_{\text{Gauss}}(s) = \mu_h(s) - \Phi^{-1}(1-\delta)\sigma_h(s)$$
- **Mean-Only (Uncertainty-Blind)**:
  $$\hat{h}_{\text{Mean}}(s) = \mu_h(s)$$
- **Unconstrained**:
  $$\text{Filter disabled}$$

**Research Question 3**: Does heavy-tailed kinetic uncertainty require distribution-free concentration bounds (Cantelli) over Gaussian approximations to eliminate runaway explosions?

---

## 2. Experimental Specifications & Metrics
- **Seeds**: 5 independent seeds per condition.
- **Evaluation Metrics**:
  - Sample Efficiency: Steps to reach $\mathcal{H}(b_t) \le 0.05$ nats.
  - Distractor Fixation Rate: Percentage of actions directed at aleatoric noise.
  - Safety Violations: Count of transitions exceeding $T_{\max} = 420\text{ K}$.
  - Min Safety Margin: $\min_t h(s_t)$.
