"""Information-theoretic acquisition functions for active experimental design.

Implements Contrastive Hypothesis Discrepancy (CHD), BALD, and baseline metrics
(Ensemble Disagreement, Prediction Error Curiosity).
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Union
import numpy as np


def compute_chd(
    predictions: np.ndarray,
    weights: np.ndarray,
    noise_variance: float = 0.0025
) -> float:
    """Computes Contrastive Hypothesis Discrepancy (CHD).
    
    Derived as the Generalized Jensen-Shannon Divergence across candidate hypothesis
    predictions (Theorem 1, docs/RESEARCH_FOUNDATIONS.md).
    
    r_CHD(s, a) = (1 / (2 * σ^2)) * ∑_k w_k || μ_k - μ_bar ||^2
    
    Properties:
        - Invariant to aleatoric noise (equals 0 if all active hypotheses agree).
        - Maximally rewards actions where likely hypotheses make conflicting predictions.
        
    Args:
        predictions: Array of shape [K, D] representing predicted outcomes under K hypotheses.
        weights: Array of shape [K] containing normalized posterior belief weights.
        noise_variance: σ^2 of the transition noise channel.
        
    Returns:
        Scalar acquisition value r_CHD >= 0.
    """
    assert len(predictions) == len(weights)
    K = len(weights)
    if K <= 1:
        return 0.0

    w = np.asarray(weights, dtype=np.float64)[:, np.newaxis]  # [K, 1]
    preds = np.asarray(predictions, dtype=np.float64)        # [K, D]

    # Weighted mean prediction: μ_bar = ∑_k w_k μ_k
    mean_pred = np.sum(w * preds, axis=0, keepdims=True)     # [1, D]

    # Weighted sum of squared differences from the mean
    diff = preds - mean_pred                                 # [K, D]
    squared_dist = np.sum(diff ** 2, axis=-1, keepdims=True) # [K, 1]
    weighted_variance = float(np.sum(w * squared_dist))

    # Scale by inverse noise variance (natural information metric in nats)
    chd = weighted_variance / (2.0 * max(1e-6, noise_variance))
    return max(0.0, float(chd))


def compute_bald(
    predictions: np.ndarray,
    weights: np.ndarray,
    noise_variance: float = 0.0025
) -> float:
    """Computes Bayesian Active Learning by Disagreement (BALD) mutual information.
    
    BALD = H(Y | x) - E_θ[H(Y | x, θ)]
    Reference: Houlsby et al. (2011), Bayesian Active Learning for Classification and Preferences.
    """
    w = np.asarray(weights, dtype=np.float64)
    preds = np.asarray(predictions, dtype=np.float64)
    
    # Under Gaussian likelihood with variance σ^2, each model has conditional entropy:
    # H(Y | x, θ_k) = 0.5 * ln( (2πe)^d * σ^2 )
    # The mixture entropy can be approximated using the total variance:
    # Var_mix = σ^2 + ∑_k w_k ||μ_k - μ_bar||^2
    d = preds.shape[-1]
    mean_pred = np.sum(w[:, np.newaxis] * preds, axis=0)
    epistemic_var = float(np.sum(w * np.sum((preds - mean_pred) ** 2, axis=-1)))
    
    total_var = noise_variance + epistemic_var / max(1, d)
    # I(Θ; Y | x) ≈ 0.5 * ln(1 + epistemic_var / (d * noise_variance))
    bald = 0.5 * np.log(1.0 + epistemic_var / (max(1e-6, noise_variance)))
    return max(0.0, float(bald))


def compute_ensemble_variance(predictions: np.ndarray) -> float:
    """Computes unweighted ensemble disagreement variance.
    
    Baseline acquisition function used by Plan2Explore (Sekar et al., 2020):
    r_disagreement = (1 / M) ∑_m || f_m(s, a) - f_bar(s, a) ||^2
    """
    preds = np.asarray(predictions, dtype=np.float64)
    mean_pred = np.mean(preds, axis=0, keepdims=True)
    variance = np.mean(np.sum((preds - mean_pred) ** 2, axis=-1))
    return float(variance)


def compute_prediction_error(
    predicted_next_state: np.ndarray,
    actual_next_state: np.ndarray
) -> float:
    """Computes prediction error curiosity reward.
    
    Baseline acquisition function used by ICM (Pathak et al., 2017):
    r_curiosity = 0.5 * || s_{t+1} - f_hat(s_t, a_t) ||^2
    """
    diff = np.asarray(actual_next_state) - np.asarray(predicted_next_state)
    return float(0.5 * np.sum(diff ** 2))
