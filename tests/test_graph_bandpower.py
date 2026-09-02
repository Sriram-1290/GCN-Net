"""Tests for the band-power architecture with a graph spatial stage.

Three properties carry the design and each has a test:

* **power is taken after the spatial stage.** This is the whole point — CSP combines
  channels and then takes variance, and 9.6's null result tested the reverse ordering.
  A test that only checked output shape would pass on an architecture that got this
  backwards, so the ordering is asserted directly.
* **the graph is a prior, not a constraint, in the regulariser variant.** The penalty
  must be zero for a spatially smooth operator and positive for a rough one, and it
  must reach the operator's gradient. Otherwise it is decoration.
* **the config refuses combinations that would confound the comparison**, in the same
  spirit as `--free-spatial` refusing to stack with `--learnable-graph`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graphs.adjacency import pcc_adjacency
from graphs.coarsening import coarsen
from graphs.laplacian import laplacian
from models.gcns_net import build_bases, dense_adjacencies
from models.graph_bandpower import GraphBandPowerConfig, GraphBandPowerNet


def _graphs(n_channels=64, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(6, n_channels, 256)).astype(np.float32)
    return coarsen(pcc_adjacency(X), levels=5, self_connections=False, seed=seed)[0]


@pytest.fixture(scope="module")
def graph_bits():
    graphs = _graphs()
    return build_bases(graphs, (2,) * 6), dense_adjacencies(graphs)


# ------------------------------------------------------------------- shapes

@pytest.mark.parametrize("n_samples", [64, 160, 320])
def test_forward_shape(graph_bits, n_samples):
    bases, _ = graph_bits
    net = GraphBandPowerNet(bases, n_samples, GraphBandPowerConfig(n_classes=2))
    out = net(torch.randn(4, bases[0].shape[1], n_samples))
    assert out.shape == (4, 2)
    assert torch.isfinite(out).all()


def test_wrong_shape_fails_loudly(graph_bits):
    bases, _ = graph_bits
    net = GraphBandPowerNet(bases, 64, GraphBandPowerConfig())
    with pytest.raises(ValueError, match="expected"):
        net(torch.randn(4, 32, 64))
    with pytest.raises(ValueError, match="expected"):
        net(torch.randn(4, bases[0].shape[1], 99))


def test_default_depth_is_one_not_six(graph_bits):
    """Change 3. GCNs-Net's six layers were adopted from the paper, never justified."""
    bases, _ = graph_bits
    assert GraphBandPowerConfig().n_layers == 1
    net = GraphBandPowerNet(bases, 64, GraphBandPowerConfig())
    assert len(net.spatial) == 1


def test_node_pooling_is_off_by_default_but_available(graph_bits):
    """Change 4, kept as an ablation rather than an assumption."""
    bases, _ = graph_bits
    assert GraphBandPowerConfig().pool_nodes is False
    plain = GraphBandPowerNet(bases, 64, GraphBandPowerConfig())
    pooled = GraphBandPowerNet(bases, 64, GraphBandPowerConfig(pool_nodes=True))
    assert plain.pool is None and pooled.pool is not None
    # pooling halves the node axis, so the classifier sees half the features
    assert pooled.classifier.in_features == plain.classifier.in_features // 2
    pooled(torch.randn(2, bases[0].shape[1], 64))


# ------------------------------------------- power comes AFTER the spatial stage

def test_power_is_taken_after_the_spatial_stage(graph_bits):
    """The ordering that distinguishes this from 9.6, asserted rather than assumed.

    If power were taken per channel *before* mixing, the network could not
    distinguish two inputs that share per-channel power but differ in the relative
    sign between channels — that sign is exactly what a spatial filter cancels on.
    Feeding such a pair must therefore produce different outputs.
    """
    bases, _ = graph_bits
    m = bases[0].shape[1]
    net = GraphBandPowerNet(bases, 64, GraphBandPowerConfig()).eval()

    rng = np.random.default_rng(0)
    base = torch.tensor(rng.normal(size=(1, m, 64)), dtype=torch.float32)
    flipped = base.clone()
    flipped[:, 1::2, :] *= -1.0        # invert every other channel: same power, new phase

    assert torch.allclose((base ** 2).sum(dim=2), (flipped ** 2).sum(dim=2))
    with torch.no_grad():
        assert not torch.allclose(net(base), net(flipped), atol=1e-5), (
            "output is invariant to per-channel sign, so the spatial stage is acting "
            "on power rather than on voltage -- the 9.6 ordering"
        )


def test_log_is_guarded_against_zero(graph_bits):
    bases, _ = graph_bits
    net = GraphBandPowerNet(bases, 64, GraphBandPowerConfig()).eval()
    with torch.no_grad():
        out = net(torch.zeros(2, bases[0].shape[1], 64))
    assert torch.isfinite(out).all()


# ------------------------------------------------ the graph as a prior, not a wall

def test_smoothness_penalty_is_small_for_smooth_operators_and_large_for_rough(graph_bits):
    bases, adjs = graph_bits
    lap = laplacian(adjs[0], normalized=False)
    cfg = GraphBandPowerConfig(spatial="free", graph_smoothness=1e-3)
    net = GraphBandPowerNet(bases, 64, cfg, laplacian=lap)

    with torch.no_grad():
        net.spatial[0].operators.fill_(1.0)          # perfectly smooth: constant
    smooth = float(net.graph_penalty().detach())
    with torch.no_grad():
        net.spatial[0].operators.copy_(
            torch.randn_like(net.spatial[0].operators) * 5.0)   # rough
    rough = float(net.graph_penalty().detach())

    assert smooth == pytest.approx(0.0, abs=1e-3), (
        f"a constant operator has zero Dirichlet energy, got {smooth}")
    assert rough > 1.0, f"a rough operator should be penalised, got {rough}"


def test_smoothness_penalty_reaches_the_operator_gradient(graph_bits):
    bases, adjs = graph_bits
    cfg = GraphBandPowerConfig(spatial="free", graph_smoothness=1e-3)
    net = GraphBandPowerNet(bases, 64, cfg, laplacian=laplacian(adjs[0], normalized=False))
    net.graph_penalty().backward()
    grad = net.spatial[0].operators.grad
    assert grad is not None and float(grad.abs().sum()) > 0


def test_penalty_is_zero_and_free_when_disabled(graph_bits):
    bases, _ = graph_bits
    net = GraphBandPowerNet(bases, 64, GraphBandPowerConfig(spatial="free"))
    assert net.laplacian is None
    assert float(net.graph_penalty()) == 0.0


def test_free_spatial_starts_at_the_chebyshev_basis(graph_bits):
    """Same guarantee as A1: the two spatial modes are identical at step 0."""
    bases, _ = graph_bits
    torch.manual_seed(0)
    graph = GraphBandPowerNet(bases, 64, GraphBandPowerConfig(spatial="graph")).eval()
    torch.manual_seed(0)
    free = GraphBandPowerNet(bases, 64, GraphBandPowerConfig(spatial="free")).eval()
    x = torch.randn(3, bases[0].shape[1], 64)
    with torch.no_grad():
        assert torch.allclose(graph(x), free(x), atol=1e-4)


# ----------------------------------------------------------------- config guards

def test_smoothness_on_a_constrained_graph_is_refused():
    """The prior and the hard constraint would be confounded; pick one."""
    with pytest.raises(ValueError, match="already a hard constraint"):
        GraphBandPowerConfig(spatial="graph", graph_smoothness=1e-3)


def test_smoothness_without_a_laplacian_is_refused(graph_bits):
    bases, _ = graph_bits
    with pytest.raises(ValueError, match="needs the layer-0 Laplacian"):
        GraphBandPowerNet(bases, 64,
                          GraphBandPowerConfig(spatial="free", graph_smoothness=1e-3))


def test_invalid_config_values_are_refused():
    with pytest.raises(ValueError, match="spatial must be"):
        GraphBandPowerConfig(spatial="dense")
    with pytest.raises(ValueError, match="n_layers must be"):
        GraphBandPowerConfig(n_layers=0)
    with pytest.raises(ValueError, match="graph_smoothness must be"):
        GraphBandPowerConfig(spatial="free", graph_smoothness=-1.0)


def test_it_is_small_enough_that_gains_are_not_capacity(graph_bits):
    """~45 trials per problem; a model with GCNs-Net's 359k parameters is not the fix."""
    bases, _ = graph_bits
    n = GraphBandPowerNet(bases, 64, GraphBandPowerConfig()).n_parameters()
    assert n < 100_000, f"{n} parameters is too many for ~45 trials"
