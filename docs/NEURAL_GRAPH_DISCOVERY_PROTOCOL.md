# Neural Causal Graph Discovery Protocol: Interventional DAG Recovery

## 1. Mathematical Formalism
Let $\mathcal{G}^* = (\mathcal{V}, \mathcal{E}^*)$ be an unknown directed acyclic graph over $d$ random variables $\mathcal{V} = \{X_1, \dots, X_d\}$ with binary adjacency matrix $G^* \in \{0, 1\}^{d \times d}$:
$$G^*_{ij} = 1 \iff (X_i \to X_j) \in \mathcal{E}^*$$

The data-generating process is a non-linear Structural Causal Model (SCM) (Pearl, 2009):
$$X_j = f_j\left( \text{Pa}(X_j), \epsilon_j \right), \quad \epsilon_j \sim \mathcal{N}(0, \sigma_j^2)$$
where $\text{Pa}(X_j) = \{X_i \mid G^*_{ij} = 1\}$. Under an intervention $\text{do}(X_k = \xi)$, the structural mechanism for $X_k$ is replaced by the constant $\xi$, severing all incoming edges to $X_k$.

## 2. Markov Equivalence & The Interventional Imperative
From observational data alone, causal DAGs can only be identified up to their **Markov Equivalence Class** (MEC), sharing the same skeleton and v-structures ($X_i \to X_k \leftarrow X_j$). 
Directionality along causal chains ($X_i \to X_j \to X_k$) vs common cause ($X_i \leftarrow X_j \to X_k$) is mathematically unidentifiable from purely observational joint distributions:
$$P(X_i, X_j, X_k) = P(X_i) P(X_j \mid X_i) P(X_k \mid X_j) = P(X_j) P(X_i \mid X_j) P(X_k \mid X_j)$$
Only **targeted interventional experiments** $\text{do}(X_j = \xi)$ break the equivalence class, enabling exact DAG recovery.

## 3. Metrics for Graph Structure Recovery
Let $\hat{G} \in \{0, 1\}^{d \times d}$ be the estimated binary adjacency:
1. **Structural Hamming Distance (SHD)**:
   $$\text{SHD}(G^*, \hat{G}) = \sum_{i < j} \left( \mathbb{I}[G^*_{ij} \ne \hat{G}_{ij} \lor G^*_{ji} \ne \hat{G}_{ji}] \right)$$
   Penalizes missing edges (False Negatives), spurious edges (False Positives), and reversed edges.
2. **True Positive Rate (TPR / Sensitivity / Recall)**:
   $$\text{TPR} = \frac{|\mathcal{E}^* \cap \hat{\mathcal{E}}|}{|\mathcal{E}^*|} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
3. **False Discovery Rate (FDR)**:
   $$\text{FDR} = \frac{|\hat{\mathcal{E}} \setminus \mathcal{E}^*|}{|\hat{\mathcal{E}}|} = \frac{\text{FP}}{\text{TP} + \text{FP}}$$
4. **Continuous Acyclicity Measure**:
   $$h(\hat{W}) = \text{Tr}(\exp(\hat{W} \circ \hat{W})) - d$$

## 4. Evaluated Regimes
- **Active CGD (Neural Causal HEx-WM)**: The agent computes the Contrastive Graph Discrepancy:
  $$r_{\text{CGD}}(s, a) = \mathbb{E}_{G_1, G_2 \sim q_\psi(G)} \left[ \| f_{G_1}(s, a) - f_{G_2}(s, a) \|_2^2 \right]$$
  and actively executes interventions that maximize predictive divergence between competing graph topologies.
- **Random Interventions**: Interventions $\text{do}(X_k = \xi)$ selected uniformly at random.
- **Observational Baseline**: Pure passive observation without interventional perturbation.
