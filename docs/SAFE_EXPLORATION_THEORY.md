# Theoretical Foundations of Safety-Constrained Bayesian Experimental Design (Safe-HEx-WM)

**Target**: NeurIPS / ICML Core Theoretical Framework  
**Topic**: Provably Safe Active Hypothesis Testing in Unknown Environments  

---

## 1. Problem Formulation: The Unsafe Exploration Dilemma

In standard active exploration (such as Plan2Explore, ICM, or unconstrained BED), the agent seeks to maximize information gain:
$$a^* = \arg\max_a \mathcal{U}_{\text{info}}(s, a)$$

In unfamiliar physical, chemical, or robotic environments, extreme interventions may trigger catastrophic outcomes:
- **Chemical Reactors**: Thermal runaway reactions where exothermicity exceeds cooling capacity ($T > T_{\text{critical}}$).
- **Robotics**: Actuator torque saturation, collisions, or structural joint overload.
- **Biomedicine / Clinical Trials**: Exceeding maximum tolerated toxicity thresholds.

We formalize **Safety-Constrained Bayesian Experimental Design (Safe-BED)** as a **Constrained Epistemic POMDP**:

$$\max_{a \in \mathcal{A}} \mathcal{U}_{\text{info}}(s, a) \quad \text{subject to} \quad \Pr_{\theta \sim b_t}\left( c(s, a, s'; \theta) \le 0 \right) \ge 1 - \delta$$

where:
- $c(s, a, s'; \theta) \le 0$ defines the safe operational envelope.
- $\delta \in (0, 1)$ is the maximum allowable risk tolerance (e.g., $\delta = 0.01$ for 99% safety confidence).
- The probability is evaluated under the agent's current posterior belief $b_t(\theta)$.

---

## 2. Control Barrier Functions (CBFs) in Epistemic Uncertainty

Let $h(s): \mathcal{S} \to \mathbb{R}$ be a continuously differentiable **Safety Barrier Function** defining the safe state set $\mathcal{C}$:
$$\mathcal{C} = \{s \in \mathcal{S} \mid h(s) \ge 0\}$$
$$\partial \mathcal{C} = \{s \in \mathcal{S} \mid h(s) = 0\}$$

A candidate experiment $a$ is safe if the change in barrier function satisfies the discrete-time barrier condition:
$$\Delta h(s, a) \ge -\alpha h(s), \quad \alpha \in (0, 1]$$

### 2.1 Robust Epistemic Barrier Condition
Under candidate model hypotheses $\{\theta^{(1)}, \dots, \theta^{(K)}\}$ with posterior weights $w_1, \dots, w_K$:

For each candidate model $k$, the predicted next state is $\hat{s}'_k = f_{\theta^{(k)}}(s, a)$.  
The expected safety margin and its epistemic uncertainty are:
$$\mu_h(s, a) = \sum_{k=1}^K w_k h(\hat{s}'_k)$$
$$\sigma_h^2(s, a) = \sum_{k=1}^K w_k \left( h(\hat{s}'_k) - \mu_h(s, a) \right)^2$$

By Cantelli's / Chebyshev's inequality, the **Conservative Lower Confidence Bound (LCB)** on safety is:
$$\text{LCB}_h(s, a) \triangleq \mu_h(s, a) - \kappa_\delta \sigma_h(s, a), \quad \kappa_\delta = \sqrt{\frac{1 - \delta}{\delta}}$$

**Safe Action Filter Condition**:
$$\mathcal{A}_{\text{safe}}(s; b_t) = \left\{ a \in \mathcal{A} \mid \text{LCB}_h(s, a) \ge (1 - \alpha) h(s) \right\}$$

---

## 3. Theorem 2: Guarantees on Safe Epistemic Exploration

### Theorem 2 (Forward Invariance under Epistemic Ambiguity)
**Statement**:  
Suppose the true world model $\theta^* \in \Theta$ is contained in the hypothesis support ($\exists k: \theta^{(k)} = \theta^*$). If the agent initializes at a strictly safe state $s_0 \in \text{Int}(\mathcal{C})$ and restricts all experimental actions to:
$$a_t \in \mathcal{A}_{\text{safe}}(s_t; b_t)$$
then the environment trajectory remains within the safe set $\mathcal{C}$ with probability at least $1 - \delta$:
$$\Pr\left( \forall t \ge 0, \; s_t \in \mathcal{C} \right) \ge 1 - \delta$$

### 3.1 Monotonic Expansion of the Safe Operational Envelope
As the agent conducts safe experiments, epistemic uncertainty diminishes:
$$\sigma_h(s, a) \to 0 \quad \text{as} \quad \mathcal{H}(b_t) \to 0$$
Consequently:
$$\text{LCB}_h(s, a) \to \mu_h(s, a)$$
The safe action set expands monotonically as knowledge is acquired:
$$\mathcal{A}_{\text{safe}}(s; b_t) \subseteq \mathcal{A}_{\text{safe}}(s; b_{t+1})$$
The agent begins conservatively, and as it eliminates false models, it unlocks previously uncertain experimental regimes without ever violating physical limits.

---

## 4. The Safe-HEx-WM Constrained Acquisition Algorithm

```
Algorithm: Safe-HEx-WM Action Selection
────────────────────────────────────────────────────────────────────────────────
Input: Current state s, Belief b_t = (w_1, ..., w_K), Barrier h(·), Risk δ
Output: Safe experiment action a*

1: Sample candidate actions {a_1, ..., a_N} ~ Uniform(A)
2: for each candidate action a_i do
3:     Evaluate predicted states {ŝ'_(i, 1), ..., ŝ'_(i, K)} across hypotheses
4:     Compute barrier values {h(ŝ'_(i, 1)), ..., h(ŝ'_(i, K))}
5:     Compute mean μ_h(a_i) and variance σ_h^2(a_i)
6:     Evaluate LCB: LCB_h(a_i) = μ_h(a_i) - κ_δ · σ_h(a_i)
7:     if LCB_h(a_i) >= (1 - α) · h(s) then
8:         Mark a_i as SAFE: a_i ∈ A_safe
9:         Compute Information Gain: r_CHD(s, a_i)
10:    else
11:        Mark a_i as UNSAFE (Pruned)
12:    end if
13: end for
14: if A_safe is non-empty then
15:    a* = argmax_{a ∈ A_safe} r_CHD(s, a)
16: else
17:    // Fallback: Minimal-risk recovery action maximizing LCB_h
18:    a* = argmax_{a_i} LCB_h(a_i)
19: end if
20: Return a*
────────────────────────────────────────────────────────────────────────────────
```
