"""Tests for the learnable adjacency.

The load-bearing property is that the learnable model is **identical to the fixed
baseline at initialisation**. Without that, any difference in the ablation could
be a different starting point rather than the graph being learned, and the whole
comparison would be uninterpretable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graphs.adjacency import pcc_adjacency
from graphs.coarsening import coarsen
from graphs.laplacian import chebyshev_basis, laplacian, rescale_laplacian
from models.dynamic_adjacency import (
    DynamicChebConv,
    LearnableAdjacency,
    torch_chebyshev_basis,
    torch_laplacian,
    torch_rescale_laplacian,
)
from models.gcns_net import GCNsNet, GCNsNetConfig, build_bases, dense_adjacencies


def _graphs(n_channels=64, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(6, n_channels, 256)).astype(np.float32)
    graphs, _ = coarsen(pcc_adjacency(X), levels=5, seed=0)
    return graphs


# --------------------------------------------------------------------------
# the torch Laplacian must match the numpy one used by the baseline
# --------------------------------------------------------------------------

def test_torch_laplacian_matches_the_numpy_implementation():
    A = pcc_adjacency(np.random.default_rng(0).normal(size=(4, 16, 128)).astype(np.float32))
    got = torch_laplacian(torch.as_tensor(A, dtype=torch.float64)).numpy()
    assert np.allclose(got, laplacian(A), atol=1e-8)


def test_torch_rescale_matches_numpy():
    A = pcc_adjacency(np.random.default_rng(0).normal(size=(4, 12, 128)).astype(np.float32))
    L = torch.as_tensor(laplacian(A), dtype=torch.float64)
    got = torch_rescale_laplacian(L, lmax=2.0).numpy()
    assert np.allclose(got, rescale_laplacian(laplacian(A), lmax=2.0), atol=1e-8)


def test_torch_chebyshev_matches_numpy():
    A = pcc_adjacency(np.random.default_rng(0).normal(size=(4, 10, 128)).astype(np.float32))
    Lt = rescale_laplacian(laplacian(A))
    got = torch_chebyshev_basis(torch.as_tensor(Lt, dtype=torch.float64), K=4).numpy()
    assert np.allclose(got, chebyshev_basis(Lt, K=4), atol=1e-8)


# --------------------------------------------------------------------------
# the adjacency parameterisation
# --------------------------------------------------------------------------

def test_starts_exactly_at_the_pcc_graph():
    """The property the whole ablation depends on."""
    A = pcc_adjacency(np.random.default_rng(0).normal(size=(4, 16, 128)).astype(np.float32))
    adj = LearnableAdjacency(A)
    assert torch.allclose(adj(), torch.as_tensor(A, dtype=torch.float32), atol=1e-6)


def test_stays_symmetric_and_non_negative_after_an_arbitrary_update():
    A = pcc_adjacency(np.random.default_rng(0).normal(size=(4, 16, 128)).astype(np.float32))
    adj = LearnableAdjacency(A)
    with torch.no_grad():
        adj.delta.copy_(torch.randn(16, 16) * 5)   # deliberately wild and asymmetric
    out = adj()
    assert torch.allclose(out, out.t(), atol=1e-6)
    assert (out >= 0).all()
    assert torch.allclose(torch.diagonal(out), torch.zeros(16))


def test_the_learned_graph_produces_a_valid_laplacian_even_when_edges_die():
    """relu can zero every edge of a node; the Laplacian must stay finite."""
    A = pcc_adjacency(np.random.default_rng(0).normal(size=(4, 8, 128)).astype(np.float32))
    adj = LearnableAdjacency(A)
    with torch.no_grad():
        adj.delta.fill_(-100.0)      # drive the entire graph to zero
    L = torch_laplacian(adj())
    assert torch.isfinite(L).all()


def test_gradients_reach_the_adjacency():
    A = pcc_adjacency(np.random.default_rng(0).normal(size=(4, 16, 128)).astype(np.float32))
    conv = DynamicChebConv(2, 4, A, K=2)
    conv(torch.randn(3, 16, 2)).sum().backward()
    assert conv.adjacency.delta.grad is not None
    assert conv.adjacency.delta.grad.abs().sum() > 0


def test_delta_l1_is_zero_at_init_and_positive_after_change():
    A = pcc_adjacency(np.random.default_rng(0).normal(size=(4, 8, 128)).astype(np.float32))
    adj = LearnableAdjacency(A)
    assert adj.delta_l1().item() == pytest.approx(0.0)
    with torch.no_grad():
        adj.delta.fill_(0.1)
    assert adj.delta_l1().item() > 0


# --------------------------------------------------------------------------
# the full model
# --------------------------------------------------------------------------

def test_dynamic_conv_matches_the_fixed_conv_at_initialisation():
    A = pcc_adjacency(np.random.default_rng(0).normal(size=(4, 16, 128)).astype(np.float32))
    conv = DynamicChebConv(3, 5, A, K=2)
    x = torch.randn(4, 16, 3)
    # the parent's buffer holds the fixed basis; both must agree at delta = 0
    fixed = torch.einsum("kmn,bnf->bmfk", conv.basis, x)
    fixed = (fixed.reshape(4 * 16, 3 * 2) @ conv.weight).view(4, 16, 5)
    assert torch.allclose(conv(x), fixed, atol=1e-5)


def test_learnable_model_equals_fixed_model_at_init():
    graphs = _graphs()
    bases = build_bases(graphs, (2,) * 6)
    adjs = dense_adjacencies(graphs)
    torch.manual_seed(0)
    fixed = GCNsNet(bases, GCNsNetConfig()).eval()
    torch.manual_seed(0)
    learn = GCNsNet(bases, GCNsNetConfig(learnable_graph=True), adjacencies=adjs).eval()
    x = torch.randn(6, 64, 1)
    assert torch.allclose(fixed(x), learn(x), atol=1e-4)


def test_learnable_graph_adds_only_a_small_number_of_parameters():
    graphs = _graphs()
    bases, adjs = build_bases(graphs, (2,) * 6), dense_adjacencies(graphs)
    fixed = GCNsNet(bases, GCNsNetConfig()).n_parameters()
    learn = GCNsNet(bases, GCNsNetConfig(learnable_graph=True),
                    adjacencies=adjs).n_parameters()
    added = learn - fixed
    assert added == 64**2 + 32**2 + 16**2 + 8**2 + 4**2 + 2**2 == 5460
    assert added / fixed < 0.02   # under 2%, so gains are not just capacity


def test_learnable_graph_requires_the_adjacencies():
    bases = build_bases(_graphs(), (2,) * 6)
    with pytest.raises(ValueError, match="needs the per-layer adjacencies"):
        GCNsNet(bases, GCNsNetConfig(learnable_graph=True))


def test_adjacency_penalty_is_zero_for_the_fixed_baseline():
    bases = build_bases(_graphs(), (2,) * 6)
    assert GCNsNet(bases, GCNsNetConfig()).adjacency_penalty().item() == pytest.approx(0.0)


def test_learned_adjacencies_are_exposed_for_interpretability():
    graphs = _graphs()
    model = GCNsNet(build_bases(graphs, (2,) * 6), GCNsNetConfig(learnable_graph=True),
                    adjacencies=dense_adjacencies(graphs))
    got = model.learned_adjacencies()
    assert [g.shape[0] for g in got] == [64, 32, 16, 8, 4, 2]


def test_training_actually_changes_the_graph():
    graphs = _graphs()
    model = GCNsNet(build_bases(graphs, (2,) * 6), GCNsNetConfig(learnable_graph=True),
                    adjacencies=dense_adjacencies(graphs))
    before = model.learned_adjacencies()[0].copy()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(5):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            model(torch.randn(16, 64, 1)), torch.randint(0, 4, (16,)))
        loss.backward()
        opt.step()
    assert not np.allclose(before, model.learned_adjacencies()[0])
