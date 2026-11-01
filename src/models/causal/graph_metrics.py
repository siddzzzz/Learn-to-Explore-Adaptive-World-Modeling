"""Quantitative evaluation metrics for Causal DAG discovery.

Implements Structural Hamming Distance (SHD), True Positive Rate (TPR),
False Discovery Rate (FDR), and Precision/Recall calculations.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, Union
import numpy as np
import torch


def compute_graph_metrics(
    true_adjacency: Union[np.ndarray, torch.Tensor],
    pred_adjacency: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.3
) -> Dict[str, float]:
    """Computes standard causal structure discovery metrics.
    
    Args:
        true_adjacency: Binary ground-truth adjacency matrix [d, d] (A[i, j] = 1 means i -> j).
        pred_adjacency: Continuous or binary predicted adjacency matrix [d, d].
        threshold: Binarization cutoff threshold for predicted edges.
        
    Returns:
        Dict containing:
            - shd: Structural Hamming Distance (lower is better, 0 is perfect).
            - tpr: True Positive Rate / Recall (higher is better, 1.0 is perfect).
            - fdr: False Discovery Rate (lower is better, 0.0 is perfect).
            - precision: TP / (TP + FP).
            - tp: True Positive count.
            - fp: False Positive count.
            - fn: False Negative count.
            - reversed_edges: Count of edges predicted in opposite direction.
    """
    if isinstance(true_adjacency, torch.Tensor):
        G_true = true_adjacency.detach().cpu().numpy()
    else:
        G_true = np.asarray(true_adjacency)

    if isinstance(pred_adjacency, torch.Tensor):
        G_pred = pred_adjacency.detach().cpu().numpy()
    else:
        G_pred = np.asarray(pred_adjacency)

    # Binarize
    G_true = (np.abs(G_true) > 0.1).astype(int)
    G_pred = (np.abs(G_pred) > threshold).astype(int)

    # Ensure zero diagonal (no self-loops)
    np.fill_diagonal(G_true, 0)
    np.fill_diagonal(G_pred, 0)

    d = G_true.shape[0]

    # Calculate Edge Components
    # True Positives: Predicted edge i->j matches True edge i->j
    tp = int(np.sum((G_true == 1) & (G_pred == 1)))
    
    # False Positives: Predicted edge i->j where no True edge i->j
    fp = int(np.sum((G_true == 0) & (G_pred == 1)))
    
    # False Negatives: True edge i->j where no Predicted edge i->j
    fn = int(np.sum((G_true == 1) & (G_pred == 0)))

    # Reversed edges: Predicted i->j while ground truth is j->i
    reversed_edges = 0
    for i in range(d):
        for j in range(i + 1, d):
            if G_true[i, j] == 1 and G_pred[j, i] == 1 and G_pred[i, j] == 0:
                reversed_edges += 1
            elif G_true[j, i] == 1 and G_pred[i, j] == 1 and G_pred[j, i] == 0:
                reversed_edges += 1

    # Structural Hamming Distance
    # SHD = (FP + FN) where reversed edges count as 1 edge error instead of 2 (1 FP + 1 FN)
    # Standard Tsamardinos (2006) formulation:
    shd = (fp + fn) - reversed_edges

    total_true_edges = int(np.sum(G_true))
    total_pred_edges = int(np.sum(G_pred))

    tpr = float(tp) / float(total_true_edges) if total_true_edges > 0 else 1.0
    fdr = float(fp) / float(total_pred_edges) if total_pred_edges > 0 else 0.0
    precision = float(tp) / float(total_pred_edges) if total_pred_edges > 0 else 1.0

    return {
        "shd": float(shd),
        "tpr": float(tpr),
        "fdr": float(fdr),
        "precision": float(precision),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "reversed_edges": float(reversed_edges),
        "total_true_edges": float(total_true_edges),
        "total_pred_edges": float(total_pred_edges),
    }
