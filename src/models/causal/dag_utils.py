"""Graph and acyclicity mathematical utilities for differentiable causal discovery.

Implements the continuous characterization of acyclicity (Zheng et al., 2018; NOTEARS)
and graph-theoretic operations in PyTorch.
"""

from __future__ import annotations
from typing import List, Tuple, Union
import numpy as np
import torch


def compute_acyclicity_constraint(W: torch.Tensor) -> torch.Tensor:
    """Computes the smooth acyclicity characterization h(W) = Tr(exp(W ∘ W)) - d.
    
    Theorem (Zheng et al., 2018):
        A directed graph with weighted adjacency W is acyclic (a DAG)
        if and only if h(W) = 0.
        
    Args:
        W: Weighted adjacency matrix tensor of shape [d, d] or batched [B, d, d].
        
    Returns:
        Scalar acyclicity penalty tensor h(W) >= 0.
    """
    is_batched = W.dim() == 3
    if not is_batched:
        W = W.unsqueeze(0)

    B, d, _ = W.shape
    M = W * W  # Hadamard product W ∘ W

    # Matrix exponential: exp(M)
    exp_M = torch.matrix_exp(M)
    
    # Trace: ∑_i exp(M)_{ii}
    traces = torch.diagonal(exp_M, dim1=-2, dim2=-1).sum(dim=-1)
    h = traces - d

    if not is_batched:
        return h.squeeze(0)
    return h


def compute_acyclicity_gradient(W: torch.Tensor) -> torch.Tensor:
    """Computes analytical gradient ∇_W h(W) = 2 (exp(W ∘ W))^T ∘ W."""
    M = W * W
    exp_M = torch.matrix_exp(M)
    grad = 2.0 * (exp_M.transpose(-1, -2) * W)
    return grad


def is_acyclic(adjacency_matrix: Union[np.ndarray, torch.Tensor], threshold: float = 0.3) -> bool:
    """Checks whether a directed graph is acyclic via Kahn's topological sort.
    
    Args:
        adjacency_matrix: Adjacency matrix [d, d] where A[i, j] > 0 means edge i -> j.
        threshold: Binarization threshold.
        
    Returns:
        True if the graph is a strict DAG, False if it contains directed cycles.
    """
    if isinstance(adjacency_matrix, torch.Tensor):
        adj = (adjacency_matrix.abs() > threshold).cpu().numpy().astype(int)
    else:
        adj = (np.abs(adjacency_matrix) > threshold).astype(int)

    d = adj.shape[0]
    in_degrees = np.sum(adj, axis=0)

    # Queue of nodes with zero in-degree
    queue = [i for i in range(d) if in_degrees[i] == 0]
    visited_count = 0

    while queue:
        curr = queue.pop(0)
        visited_count += 1
        for neighbor in np.where(adj[curr, :] > 0)[0]:
            in_degrees[neighbor] -= 1
            if in_degrees[neighbor] == 0:
                queue.append(neighbor)

    # Acyclic if and only if all nodes can be topologically ordered
    return visited_count == d


def threshold_graph(adjacency_matrix: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """Hard-binarizes continuous adjacency matrix and zeroes diagonal."""
    adj = (adjacency_matrix.abs() > threshold).float()
    # Remove self-loops
    adj.fill_diagonal_(0.0)
    return adj
