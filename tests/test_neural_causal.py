"""Unit tests for Differentiable Neural Causal World Models and Active Discovery.

Tests mathematical acyclicity, variational graph sampling, Pearl do-operator masking,
and Contrastive Graph Discrepancy without running heavy training loops.
"""

import numpy as np
import pytest
import torch

from src.models.causal.dag_utils import (
    compute_acyclicity_constraint,
    compute_acyclicity_gradient,
    is_acyclic,
    threshold_graph
)
from src.models.causal.graph_posterior import VariationalGraphPosterior
from src.models.causal.neural_scm import NeuralSCM
from src.models.causal.causal_world_model import DifferentiableCausalWorldModel
from src.planning.causal_acquisition import (
    compute_contrastive_graph_discrepancy,
    evaluate_intervention_informativeness
)
from src.agents.neural_causal_agent import NeuralCausalAgent


def test_acyclicity_constraint_dag_vs_cycle():
    """Verify NOTEARS acyclicity characterization h(W) = 0 for DAGs and h(W) > 0 for cycles."""
    d = 4
    # Case A: Strict DAG (Linear Chain: 0->1->2->3)
    W_dag = torch.zeros(d, d)
    W_dag[0, 1] = 1.0
    W_dag[1, 2] = 1.0
    W_dag[2, 3] = 1.0
    h_dag = compute_acyclicity_constraint(W_dag)
    assert torch.isclose(h_dag, torch.tensor(0.0), atol=1e-5)
    assert is_acyclic(W_dag)

    # Case B: Directed Cycle (0->1->2->0)
    W_cycle = torch.zeros(d, d)
    W_cycle[0, 1] = 1.0
    W_cycle[1, 2] = 1.0
    W_cycle[2, 0] = 1.0
    h_cycle = compute_acyclicity_constraint(W_cycle)
    assert h_cycle > 0.5
    assert not is_acyclic(W_cycle)


def test_acyclicity_gradient_shape():
    """Verify analytical gradient shape ∇_W h(W)."""
    W = torch.randn(4, 4, requires_grad=True)
    h = compute_acyclicity_constraint(W)
    h.backward()
    
    grad_analytical = compute_acyclicity_gradient(W.detach())
    assert torch.allclose(W.grad, grad_analytical, atol=1e-4)


def test_variational_graph_posterior_sampling():
    """Verify Gumbel-Softmax relaxed graph and discrete Bernoulli sampling."""
    num_vars = 4
    posterior = VariationalGraphPosterior(num_variables=num_vars, initial_prior=0.3)
    
    # 1. Edge probabilities should be in (0, 1) and diagonal strictly zero
    probs = posterior.get_edge_probabilities()
    assert probs.shape == (num_vars, num_vars)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert torch.diagonal(probs).sum() == 0.0

    # 2. Gumbel-Softmax continuous relaxation
    relaxed = posterior.sample_gumbel(batch_size=5, temperature=0.5)
    assert relaxed.shape == (5, num_vars, num_vars)
    assert (relaxed >= 0.0).all() and (relaxed <= 1.0).all()

    # 3. Discrete Bernoulli DAG sampling
    discrete = posterior.sample_discrete(num_samples=3)
    assert discrete.shape == (3, num_vars, num_vars)
    # Binary entries {0, 1}
    assert ((discrete == 0.0) | (discrete == 1.0)).all()

    # 4. Shannon entropy should be positive
    ent = posterior.get_shannon_entropy()
    assert ent > 0.0


def test_neural_scm_pearl_intervention_masking():
    """Verify that Neural SCM respects parent gating and Pearl's do-operator."""
    num_vars = 4
    action_dim = 2
    scm = NeuralSCM(num_variables=num_vars, action_dim=action_dim, hidden_dim=32)

    state = torch.tensor([1.0, 2.0, 3.0, 4.0])
    action = torch.zeros(action_dim)
    adj = torch.zeros(num_vars, num_vars)
    adj[0, 1] = 1.0  # 0 causes 1

    # Normal un-intervened pass
    next_s = scm(state, action, adj)
    assert next_s.shape == (4,)

    # Intervene on variable 1: do(X_1 = 9.99)
    next_s_interv = scm(state, action, adj, interventions={1: 9.99})
    assert torch.isclose(next_s_interv[1], torch.tensor(9.99))
    # Other variables shouldn't be forced
    assert not torch.isclose(next_s_interv[0], torch.tensor(9.99))


def test_contrastive_graph_discrepancy():
    """Verify CGD metric correctly measures divergence across candidate DAGs."""
    # Case A: Candidate graphs make identical predictions (e.g. passive observation under MEC)
    identical_preds = np.tile(np.array([1.0, 2.0, 3.0]), (4, 1))
    cgd_zero = compute_contrastive_graph_discrepancy(identical_preds)
    assert np.isclose(cgd_zero, 0.0)

    # Case B: Candidate graphs diverge under an intervention
    divergent_preds = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 5.0, 0.0],
        [1.0, -5.0, 0.0]
    ])
    cgd_pos = compute_contrastive_graph_discrepancy(divergent_preds)
    assert cgd_pos > 0.0


def test_neural_causal_agent_step():
    """Verify NeuralCausalAgent initializes and executes one active action selection step."""
    def dummy_action_decoder(a_idx: int):
        return {0: 2.0} if a_idx == 1 else {}

    agent = NeuralCausalAgent(
        num_variables=4,
        action_dim=2,
        action_decoder_fn=dummy_action_decoder,
        num_actions=3,
        num_graph_particles=4,
        hidden_dim=32
    )

    dummy_state = np.zeros(4, dtype=np.float32)
    action = agent.select_action(dummy_state)
    assert action in [0, 1, 2]

    # Observe mock transition
    dummy_next = np.ones(4, dtype=np.float32)
    metrics = agent.observe_transition(dummy_state, action, dummy_next)
    assert "graph_entropy" in metrics
    assert "map_dag_idx" in metrics
    assert len(metrics["dag_weights"]) == 4
