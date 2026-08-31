"""Tests for the GCNs-Net port.

The parameter-count test is the sharpest check that the port matches the
reference: `b2relu` sizes its bias per vertex per filter, so getting `b1relu` by
mistake would change the count by tens of thousands and this test would catch it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graphs.adjacency import pcc_adjacency
from graphs.coarsening import coarsen
from models.cheb_conv import ChebConv, GraphMaxPool, VertexBiasNorm
from models.gcns_net import GCNsNet, GCNsNetConfig, build_bases


def _bases(n_channels=64, seed=1, lmax=2.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(6, n_channels, 256)).astype(np.float32)
    graphs, perm = coarsen(pcc_adjacency(X), levels=5, seed=0)
    return build_bases(graphs, (2,) * 6, lmax=lmax), perm


# --------------------------------------------------------------------------
# ChebConv
# --------------------------------------------------------------------------

def test_chebconv_shapes():
    basis = np.stack([np.eye(8), np.random.default_rng(0).normal(size=(8, 8))]).astype(np.float32)
    conv = ChebConv(3, 16, basis)
    assert conv(torch.zeros(5, 8, 3)).shape == (5, 8, 16)


def test_chebconv_k1_is_a_pointwise_linear_map():
    """With K=1 the basis is just the identity, so the conv is a per-node linear."""
    basis = np.eye(6, dtype=np.float32)[None]  # (1, 6, 6)
    conv = ChebConv(4, 3, basis)
    x = torch.randn(2, 6, 4)
    expected = x @ conv.weight
    assert torch.allclose(conv(x), expected, atol=1e-5)


def test_chebconv_weight_row_order_matches_the_reference():
    """Rows are ordered (fin, k) with k fastest, as chebyshev5's reshape implies."""
    rng = np.random.default_rng(0)
    basis = np.stack([np.eye(4), rng.normal(size=(4, 4))]).astype(np.float32)
    conv = ChebConv(2, 1, basis)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[1, 0] = 1.0  # fin=0, k=1  ->  T_1 @ x[..., 0]
    x = torch.randn(1, 4, 2)
    expected = torch.as_tensor(basis[1]) @ x[0, :, 0]
    assert torch.allclose(conv(x)[0, :, 0], expected, atol=1e-5)


def test_chebconv_rejects_a_node_count_mismatch():
    basis = np.eye(8, dtype=np.float32)[None]
    with pytest.raises(ValueError, match="nodes"):
        ChebConv(1, 4, basis)(torch.zeros(2, 16, 1))


# --------------------------------------------------------------------------
# pooling and bias/norm
# --------------------------------------------------------------------------

def test_graph_max_pool_halves_the_node_axis():
    x = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3)
    out = GraphMaxPool(2)(x)
    assert out.shape == (2, 4, 3)
    # each output node is the max of its two siblings
    assert torch.allclose(out[:, 0, :], torch.maximum(x[:, 0, :], x[:, 1, :]))


def test_graph_max_pool_rejects_an_odd_node_count():
    with pytest.raises(ValueError, match="divisible"):
        GraphMaxPool(2)(torch.zeros(1, 7, 2))


def test_vertex_bias_is_per_node_per_filter():
    """b2relu, not b1relu: bias shape is [1, M, F]."""
    norm = VertexBiasNorm(n_nodes=16, n_features=8)
    assert norm.bias.shape == (1, 16, 8)
    assert torch.allclose(norm.bias, torch.full_like(norm.bias, 0.1))


def test_batchnorm_respects_eval_mode():
    """Our deliberate fix: the reference hardcodes training=True."""
    norm = VertexBiasNorm(4, 6)
    norm.train()
    x = torch.randn(8, 4, 6)
    norm(x)  # populate running stats
    norm.eval()
    a, b = norm(x[:4]), norm(x[:4])
    assert torch.allclose(a, b)  # deterministic in eval
    norm.train()
    # in train mode the same input batched differently gives different output
    assert not torch.allclose(norm(x[:4]), a, atol=1e-4)


def test_always_train_bn_reproduces_the_leaky_behaviour():
    norm = VertexBiasNorm(4, 6, always_train_bn=True)
    norm.train()
    x = torch.randn(16, 4, 6)
    norm(x)
    norm.eval()
    # even in eval it uses batch statistics, so different batches differ
    assert not torch.allclose(norm(x[:8]), norm(x[8:]), atol=1e-4)


# --------------------------------------------------------------------------
# full model
# --------------------------------------------------------------------------

def test_forward_produces_class_logits():
    bases, _ = _bases()
    model = GCNsNet(bases)
    out = model(torch.randn(7, 64, 1))
    assert out.shape == (7, 4)
    assert torch.isfinite(out).all()


def test_two_dimensional_input_is_treated_as_one_feature():
    """Matches the reference's tf.expand_dims(x, 2)."""
    bases, _ = _bases()
    model = GCNsNet(bases).eval()
    x = torch.randn(5, 64)
    assert torch.allclose(model(x), model(x.unsqueeze(2)))


def test_node_axis_collapses_to_one():
    bases, _ = _bases()
    model = GCNsNet(bases)
    assert model.n_nodes_out == 1


def test_parameter_count_matches_the_reference_architecture():
    """~350K params, as CLAUDE.md states. Catches b1relu/b2relu confusion."""
    bases, _ = _bases()
    n = GCNsNet(bases).n_parameters()
    assert 340_000 < n < 370_000, n


def test_vertex_biases_are_sized_on_the_pre_pool_graphs():
    bases, _ = _bases()
    model = GCNsNet(bases)
    assert [m.bias.shape[1] for m in model.norms] == [64, 32, 16, 8, 4, 2]


def test_l2_penalty_covers_only_the_fully_connected_layers():
    bases, _ = _bases()
    model = GCNsNet(bases)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.convs[0].weight.fill_(10.0)   # conv weights must be excluded
    assert model.l2_penalty().item() == pytest.approx(0.0)
    with torch.no_grad():
        model.logits.weight.fill_(1.0)
    assert model.l2_penalty().item() == pytest.approx(0.5 * model.logits.weight.numel())


def test_gradients_reach_every_parameter():
    bases, _ = _bases()
    model = GCNsNet(bases)
    loss = torch.nn.functional.cross_entropy(
        model(torch.randn(8, 64, 1)), torch.randint(0, 4, (8,)))
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.any()]
    assert not missing, missing


def test_config_rejects_mismatched_layer_lists():
    with pytest.raises(ValueError):
        GCNsNetConfig(filters=(16, 32), orders=(2,), pools=(2, 2))


def test_build_bases_with_true_lmax_differs_from_the_fixed_two():
    fixed, _ = _bases(lmax=2.0)
    exact, _ = _bases(lmax=None)
    assert not np.allclose(fixed[0], exact[0])
