"""Information-theoretic acquisition functions for active causal graph discovery.

Computes Contrastive Graph Discrepancy (CGD) to actively design interventions
that break observational Markov Equivalence Classes (MECs).
"""

from __future__ import annotations
from typing import Optional
import numpy as np


def compute_contrastive_graph_discrepancy(
    predictions_under_graphs: np.ndarray,
    weights: Optional[np.ndarray] = None,
    noise_variance: float = 0.0025
) -> float:
    """Computes Contrastive Graph Discrepancy (CGD) across sampled DAG hypotheses.
    
    Given K candidate DAGs sampled from the graph posterior G^(1), ..., G^(K) ~ q_ψ(G),
    evaluates how much their interventional predictions diverge:
    
        r_CGD(a) = (1 / (2 * σ^2)) * ∑_k w_k || X_hat^(k)(a) - X_bar(a) ||^2
        
    Properties:
        - If candidate DAGs are in the same Markov Equivalence Class and the action
          is purely observational, r_CGD(a) = 0.
        - Under targeted interventions do(X_i = v) on collider or mediator nodes,
          predictions diverge, yielding a large positive information signal.
          
    Args:
        predictions_under_graphs: Array of shape [K, d] of predicted outcomes.
        weights: Optional prior/posterior weights over the K graphs (default uniform).
        noise_variance: Observation noise variance σ^2.
        
    Returns:
        Scalar acquisition value r_CGD >= 0.
    """
    preds = np.asarray(predictions_under_graphs, dtype=np.float64)
    K, d = preds.shape
    if K <= 1:
        return 0.0

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    else:
        w = np.full(K, 1.0 / K, dtype=np.float64)

    # Weighted mean prediction across candidate graphs: X_bar = ∑_k w_k X_k
    mean_pred = np.sum(w[:, np.newaxis] * preds, axis=0, keepdims=True)

    # Weighted sum of squared differences from the mean
    diff = preds - mean_pred
    weighted_var = float(np.sum(w[:, np.newaxis] * (diff ** 2)))

    cgd = weighted_var / (2.0 * max(1e-6, noise_variance))
    return max(0.0, float(cgd))


def evaluate_intervention_informativeness(
    unintervened_predictions: np.ndarray,
    intervened_predictions: np.ndarray
) -> float:
    """Measures the information gain achieved by an intervention over passive observation.
    
    Δ_info = CGD(intervened) - CGD(observational)
    """
    cgd_interv = compute_contrastive_graph_discrepancy(intervened_predictions)
    cgd_obs = compute_contrastive_graph_discrepancy(unintervened_predictions)
    return float(max(0.0, cgd_interv - cgd_obs))
