# Active Noisy-TV Distractor Benchmark: Mathematical Protocol

## 1. Problem Definition & Setting
Consider an environment with decoupled state components:
$$s = (s_{\text{phys}}, s_{\text{dist}})$$
where:
- $s_{\text{phys}} \in \mathbb{R}^{d_s}$ is governed by an unknown structural causal mechanism $\theta^* \in \Theta$:
  $$s_{\text{phys}} = f_{\theta^*}(s_{\text{phys}}, a) + \epsilon_{\text{phys}}, \quad \epsilon_{\text{phys}} \sim \mathcal{N}(0, \sigma_{\text{phys}}^2 I)$$
- $s_{\text{dist}} \in \mathbb{R}^{d_{\text{dist}}}$ is a controllable aleatoric noise source ("Noisy TV"):
  $$s_{\text{dist}, t+1} \sim \mathcal{N}\left( g(a), \, \sigma_{\text{dist}}^2 I \right)$$
  where $g(a)$ provides a deterministic shift if $a \in \mathcal{A}_{\text{dist}}$, but the variance $\sigma_{\text{dist}}^2 \gg \sigma_{\text{phys}}^2$ is irreducible white noise.

## 2. Failure Mode of Prediction Error & Ensemble Disagreement

### Prediction Error (ICM Curiosity)
For an exploratory action $a$:
$$r_{\text{ICM}}(s, a) = \|s_{t+1} - \hat{s}_{t+1}\|^2 = \|s_{\text{phys}, t+1} - \hat{s}_{\text{phys}}\|^2 + \|s_{\text{dist}, t+1} - \hat{s}_{\text{dist}}\|^2$$
In expectation over noise:
$$\mathbb{E}[r_{\text{ICM}}(s, a)] = \text{MSE}_{\text{phys}}(a) + d_{\text{dist}} \cdot \sigma_{\text{dist}}^2$$
When $a \in \mathcal{A}_{\text{dist}}$ stimulates the TV screen, the distractor term dominates:
$$\mathbb{E}[r_{\text{ICM}}(s, a_{\text{dist}})] - \mathbb{E}[r_{\text{ICM}}(s, a_{\text{phys}})] \approx d_{\text{dist}} \cdot \sigma_{\text{dist}}^2 - \text{MSE}_{\text{phys}}(a_{\text{phys}})$$
For $\sigma_{\text{dist}} > 1$, the greedy curiosity policy strictly prefers $a_{\text{dist}}$, causing $100\%$ fixation on the television screen and complete exploratory starvation of $s_{\text{phys}}$.

### Ensemble Disagreement (Plan2Explore)
Plan2Explore measures the variance among $M$ ensemble predictions:
$$r_{\text{P2E}}(s, a) = \frac{1}{M} \sum_{m=1}^M \|f_m(s, a) - \bar{f}(s, a)\|^2$$
When the ensemble attempts to fit the high-variance stochastic channel $s_{\text{dist}}$, finite-sample variance and aleatoric parameter drift between bootstrap models create persistent pseudo-epistemic disagreement:
$$\text{Var}_m(\hat{s}_{\text{dist}}) \propto \frac{\sigma_{\text{dist}}^2}{N_{\text{samples}}}$$
which repeatedly misleads the planner towards the distractor until massive amounts of data are gathered.

## 3. Immunity of Contrastive Hypothesis Discrepancy (HEx-WM)
HEx-WM maintains belief weights $b_t(\theta) = P(\theta \mid \mathcal{D}_t)$ over hypothesized physical mechanisms:
$$r_{\text{CHD}}(s, a) = \sum_{k=1}^K b_t(\theta_k) D_{\text{KL}}\left( p(s' \mid s, a, \theta_k) \,||\, \bar{p}(s' \mid s, a) \right)$$
Because the distractor channel $s_{\text{dist}}$ does not differ across candidate structural hypotheses:
$$p(s_{\text{dist}}' \mid s, a, \theta_i) = p(s_{\text{dist}}' \mid s, a, \theta_j) = \mathcal{N}(g(a), \sigma_{\text{dist}}^2 I) \quad \forall i, j$$
By factorizability of independent noise channels:
$$D_{\text{KL}}\left( p(s_{\text{phys}}', s_{\text{dist}}' \mid \theta_i) \,||\, p(s_{\text{phys}}', s_{\text{dist}}' \mid \theta_j) \right) = D_{\text{KL}}\left( p(s_{\text{phys}}' \mid \theta_i) \,||\, p(s_{\text{phys}}' \mid \theta_j) \right) + 0$$
Hence:
$$r_{\text{CHD}}(s, a_{\text{dist}}) = 0$$
The HEx-WM agent assigns zero acquisition value to interacting with the Noisy TV, achieving strict theoretical and empirical immunity.

## 4. Quantitative Performance Metrics
1. **Distractor Fixation Rate ($\Phi_{\text{TV}}$)**:
   $$\Phi_{\text{TV}} = \frac{1}{T} \sum_{t=1}^T \mathbb{I}[a_t \in \mathcal{A}_{\text{dist}}]$$
2. **Hypothesis Discovery Steps ($T_{\text{disc}}$)**:
   $$T_{\text{disc}} = \min \{ t \mid \mathcal{H}(b_t) < 0.01 \}$$
3. **Exploratory Efficiency ($\eta_{\text{info}}$)**:
   $$\eta_{\text{info}} = \frac{\mathcal{H}(b_0) - \mathcal{H}(b_T)}{T_{\text{causal\_actions}}}$$
