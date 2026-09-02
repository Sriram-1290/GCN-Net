"""Tests for the unconstrained spatial operator (A1, project_documentation.md 17.2).

The load-bearing property is the same one the learnable-adjacency arm needed: the
free-operator model must be **identical to the fixed baseline at initialisation**.
Without that, any difference in the ablation could be a different starting point
rather than the constraint being lifted, and the comparison that 9.7.1 rests on
would be uninterpretable.

The second property tested here is that the operators are genuinely *free* — not
symmetric, not non-negative, not confined to the graph's 1-hop neighbourhood.
That is exactly what separates this arm from the learnable adjacency (9.7), which
relaxed *which* nodes are neighbours while keeping all three constraints.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graphs.adjacency import pcc_adjacency
from graphs.coarsening import coarsen
from models.cheb_conv import ChebConv
from models.free_spatial import FreeSpatialConv
from models.gcns_net import GCNsNet, GCNsNetConfig, build_bases, dense_adjacencies


def _graphs(n_channels=64, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(6, n_channels, 256)).astype(np.float32)
    graphs, _ = coarsen(pcc_adjacency(X), levels=5, self_connections=False, seed=seed)
    return graphs


def _basis(K=2, M=16, seed=0):
    rng = np.random.default_rng(seed)
    A = pcc_adjacency(rng.normal(size=(4, M, 128)).astype(np.float32))
    from graphs.laplacian import chebyshev_basis, laplacian, rescale_laplacian
    return chebyshev_basis(rescale_laplacian(laplacian(A), lmax=2.0), K)


# ---------------------------------------------------------- identity at init

def test_free_conv_matches_the_fixed_conv_at_initialisation():
    basis = _basis()
    torch.manual_seed(0)
    fixed = ChebConv(3, 5, basis)
    torch.manual_seed(0)
    free = FreeSpatialConv(3, 5, basis)
    x = torch.randn(4, basis.shape[1], 3)
    assert torch.allclose(free(x), fixed(x), atol=1e-6)


def test_free_model_equals_fixed_model_at_init():
    bases = build_bases(_graphs(), (2,) * 6)
    torch.manual_seed(0)
    fixed = GCNsNet(bases, GCNsNetConfig()).eval()
    torch.manual_seed(0)
    free = GCNsNet(bases, GCNsNetConfig(free_spatial=True)).eval()
    x = torch.randn(6, 64, 1)
    assert torch.allclose(fixed(x), free(x), atol=1e-4)


# ---------------------------------------------------- the constraint is lifted

def test_operators_are_trainable_and_the_fixed_basis_is_not():
    basis = _basis()
    free = FreeSpatialConv(3, 5, basis)
    fixed = ChebConv(3, 5, basis)
    assert free.operators.requires_grad
    assert not isinstance(getattr(fixed, "basis", None), torch.nn.Parameter)
    names = {n for n, _ in free.named_parameters()}
    assert "operators" in names and "weight" in names


def test_gradients_reach_the_operators():
    basis = _basis()
    free = FreeSpatialConv(3, 5, basis)
    free(torch.randn(4, basis.shape[1], 3)).sum().backward()
    assert free.operators.grad is not None
    assert float(free.operators.grad.abs().sum()) > 0


def test_operators_need_not_stay_symmetric_or_non_negative():
    """The point of A1: none of the graph constraints survive an update.

    The learnable adjacency (9.7) enforces symmetry and non-negativity *by
    construction*, so it can only ever re-weight a valid graph. This arm has no
    such structure, which is the whole difference between the two experiments.
    """
    basis = _basis()
    free = FreeSpatialConv(3, 5, basis)
    with torch.no_grad():
        free.operators[0, 0, 1] = -5.0    # negative
        free.operators[0, 1, 0] = +3.0    # and asymmetric
    op = free.operators[0].detach()
    assert float(op[0, 1]) < 0
    assert not torch.allclose(op, op.t())
    # and it still runs: no validity check can reject it, because there is none
    free(torch.randn(2, basis.shape[1], 3))


def test_drift_from_init_is_zero_at_init_and_positive_after_change():
    basis = _basis()
    free = FreeSpatialConv(3, 5, basis)
    assert free.drift_from_init()["l2"] == pytest.approx(0.0)
    with torch.no_grad():
        free.operators.add_(torch.randn_like(free.operators) * 0.1)
    drift = free.drift_from_init()
    assert drift["l2"] > 0 and drift["frac_changed_1pct"] > 0


# ------------------------------------------------------------- cost and guards

def test_free_spatial_adds_a_small_number_of_parameters():
    bases = build_bases(_graphs(), (2,) * 6)
    fixed = GCNsNet(bases, GCNsNetConfig()).n_parameters()
    free = GCNsNet(bases, GCNsNetConfig(free_spatial=True)).n_parameters()
    added = free - fixed
    # K=2 operators per layer over the 64/32/16/8/4/2 hierarchy
    assert added == 2 * (64**2 + 32**2 + 16**2 + 8**2 + 4**2 + 2**2) == 10920
    assert added / fixed < 0.04   # under 4%, so gains are not just capacity


def test_free_spatial_and_learnable_graph_cannot_be_stacked():
    """CLAUDE.md forbids stacking changes; these two answer different questions."""
    with pytest.raises(ValueError, match="alternative relaxations"):
        GCNsNetConfig(free_spatial=True, learnable_graph=True)
