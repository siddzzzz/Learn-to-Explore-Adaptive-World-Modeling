"""Unit tests for Phase D: Graph metrics (SHD, TPR, FDR) and Synthetic Causal DAGs."""

from __future__ import annotations
import numpy as np
import pytest

from src.models.causal.graph_metrics import compute_graph_metrics
from src.envs.causal_lab.synthetic_dag_generator import SyntheticCausalDAG
from src.models.causal.dag_utils import is_acyclic


def test_compute_graph_metrics_exact_match():
    """Verify metrics when predicted graph exactly matches ground truth."""
    true_adj = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0]
    ])
    pred_adj = true_adj.copy()

    metrics = compute_graph_metrics(true_adj, pred_adj)
    assert metrics["shd"] == 0.0
    assert metrics["tpr"] == 1.0
    assert metrics["fdr"] == 0.0
    assert metrics["precision"] == 1.0
    assert metrics["reversed_edges"] == 0.0


def test_compute_graph_metrics_reversed_and_missing_edges():
    """Verify SHD penalizes edge reversal by 1 and missing edges correctly."""
    true_adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])
    # Predicted reverses 0->1 as 1->0, and misses 1->2
    pred_adj = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0]
    ])

    metrics = compute_graph_metrics(true_adj, pred_adj)
    assert metrics["reversed_edges"] == 1.0
    assert metrics["tpr"] == 0.0  # Zero correct directed edges
    # SHD = (FP + FN) - reversed = (1 + 2) - 1 = 2
    assert metrics["shd"] == 2.0


def test_synthetic_causal_dag_generation_and_sampling():
    """Verify SyntheticCausalDAG produces valid DAGs and respects Pearl interventions."""
    dag = SyntheticCausalDAG(num_nodes=6, edge_prob=0.4, noise_std=0.01)
    adj = dag.get_binary_adjacency()

    assert adj.shape == (6, 6)
    assert is_acyclic(adj), "Generated synthetic DAG must be strictly acyclic"

    # Observational sample
    obs = dag.sample()
    assert obs.shape == (6,)

    # Interventional sample do(X_2 = 5.0)
    interv_obs = dag.sample(interventions={2: 5.0})
    assert np.isclose(interv_obs[2], 5.0, atol=0.05)
