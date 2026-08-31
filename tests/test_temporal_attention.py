"""Tests for temporal attention.

The important ones check that attention *does what attention is supposed to do*:
weights sum to one over time, and a signal planted in one part of a trial draws
the weights toward it. Shape tests alone would pass on a module that silently
averages, which would make the whole H2 comparison meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graphs.adjacency import pcc_adjacency
from graphs.coarsening import coarsen
from models.gcns_net import GCNsNet, GCNsNetConfig, build_bases, dense_adjacencies
from models.temporal_attention import TemporalAttention


def _bases(n_channels=64, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(6, n_channels, 256)).astype(np.float32)
    graphs, _ = coarsen(pcc_adjacency(X), levels=5, seed=0)
    return build_bases(graphs, (2,) * 6), dense_adjacencies(graphs)


# --------------------------------------------------------------------------
# the mechanism
# --------------------------------------------------------------------------

def test_pools_time_into_heads():
    att = TemporalAttention(n_nodes=16, n_heads=4)
    assert att(torch.randn(5, 16, 40)).shape == (5, 16, 4)


def test_attention_weights_are_a_distribution_over_time():
    att = TemporalAttention(n_nodes=8, n_heads=3)
    att(torch.randn(4, 8, 25))
    w = att.last_attention()
    assert w.shape == (4, 25, 3)
    assert torch.allclose(w.sum(dim=1), torch.ones(4, 3), atol=1e-5)
    assert (w >= 0).all()


def test_attention_moves_toward_a_planted_burst():
    """A module that just averages would fail this."""
    torch.manual_seed(0)
    n_time = 40
    x = torch.randn(64, 8, n_time) * 0.05
    x[:, :, 30:35] += 4.0                      # a loud burst late in the window
    att = TemporalAttention(n_nodes=8, n_heads=1, hidden=16)
    opt = torch.optim.Adam(att.parameters(), lr=0.05)
    # train the attention to produce a large output: it should find the burst
    for _ in range(120):
        opt.zero_grad()
        (-att(x).mean()).backward()
        opt.step()
    w = att.last_attention().mean(dim=(0, 2))
    assert int(w.argmax()) in range(28, 37), int(w.argmax())
    assert w[30:35].sum() > w[0:25].sum()


def test_uniform_input_gives_near_uniform_attention():
    att = TemporalAttention(n_nodes=8, n_heads=2, hidden=16)
    att(torch.ones(3, 8, 20))
    w = att.last_attention()
    assert torch.allclose(w, torch.full_like(w, 1 / 20), atol=1e-5)


def test_per_node_gives_a_profile_per_electrode():
    att = TemporalAttention(n_nodes=6, n_heads=2, per_node=True)
    assert att(torch.randn(3, 6, 30)).shape == (3, 6, 2)
    assert att.last_attention().shape == (3, 30, 2, 6)


def test_rejects_a_single_timepoint():
    """The reference's sample definition has no time axis to attend over."""
    att = TemporalAttention(n_nodes=8)
    with pytest.raises(ValueError, match="needs a time axis"):
        att(torch.randn(2, 8, 1))


def test_rejects_a_node_count_mismatch():
    att = TemporalAttention(n_nodes=8)
    with pytest.raises(ValueError, match="expected 8 nodes"):
        att(torch.randn(2, 16, 20))


def test_gradients_reach_the_score_network():
    att = TemporalAttention(n_nodes=8, n_heads=2)
    att(torch.randn(4, 8, 20)).sum().backward()
    grads = [p.grad for p in att.parameters()]
    assert all(g is not None and g.abs().sum() > 0 for g in grads)


# --------------------------------------------------------------------------
# integration with the model
# --------------------------------------------------------------------------

def test_model_with_attention_accepts_a_window_and_returns_logits():
    bases, _ = _bases()
    model = GCNsNet(bases, GCNsNetConfig(in_features=64, temporal_attention=True))
    out = model(torch.randn(6, 64, 64))
    assert out.shape == (6, 4)
    assert torch.isfinite(out).all()


def test_attention_refuses_the_single_timepoint_configuration():
    bases, _ = _bases()
    with pytest.raises(ValueError, match="needs a time axis"):
        GCNsNet(bases, GCNsNetConfig(in_features=1, temporal_attention=True))


def test_attention_changes_the_first_conv_width():
    """Fin becomes n_heads, so the first conv shrinks; logged, not hidden."""
    bases, _ = _bases()
    plain = GCNsNet(bases, GCNsNetConfig(in_features=64))
    att = GCNsNet(bases, GCNsNetConfig(in_features=64, temporal_attention=True,
                                       attention_heads=4))
    assert plain.convs[0].in_features == 64
    assert att.convs[0].in_features == 4


def test_attention_and_learnable_graph_compose():
    """H3 needs both at once."""
    bases, adjs = _bases()
    model = GCNsNet(bases,
                    GCNsNetConfig(in_features=64, temporal_attention=True,
                                  learnable_graph=True),
                    adjacencies=adjs)
    loss = model(torch.randn(4, 64, 64)).sum()
    loss.backward()
    assert model.attention.score[0].weight.grad.abs().sum() > 0
    assert model.convs[0].adjacency.delta.grad.abs().sum() > 0


def test_attention_is_absent_by_default():
    bases, _ = _bases()
    assert GCNsNet(bases, GCNsNetConfig()).attention is None
