"""Tests for Graclus coarsening and the binary-tree permutation.

The invariant everything depends on: after permutation, nodes ``2i`` and ``2i+1``
of layer k are the two children of node ``i`` of layer k+1. If that breaks, the
model still trains but pools unrelated electrodes, so it is tested directly
rather than inferred from output shapes.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

from graphs.adjacency import pcc_adjacency
from graphs.coarsening import (
    coarsen,
    compute_perm,
    graclus,
    perm_adjacency,
    perm_data,
)


def _ring(n: int) -> np.ndarray:
    """Ring graph — every node has degree 2, so matching is unambiguous."""
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = 1.0
        A[(i + 1) % n, i] = 1.0
    return A


def _random_adjacency(n=16, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(6, n, 128)).astype(np.float32)
    return pcc_adjacency(X)


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------

def test_graclus_roughly_halves_each_level():
    graphs, parents = graclus(scipy.sparse.csr_matrix(_ring(32)), levels=3)
    sizes = [g.shape[0] for g in graphs]
    assert sizes[0] == 32
    for a, b in zip(sizes, sizes[1:]):
        assert b <= a
        assert b >= a // 2  # at best a perfect matching halves it


def test_graclus_assigns_every_node_to_exactly_one_cluster():
    _, parents = graclus(scipy.sparse.csr_matrix(_ring(16)), levels=2)
    for parent in parents:
        assert parent.min() >= 0
        assert set(np.unique(parent)) == set(range(parent.max() + 1))


def test_graclus_clusters_hold_at_most_two_nodes():
    """Heavy-edge matching pairs nodes; a cluster of three would break pooling."""
    _, parents = graclus(scipy.sparse.csr_matrix(_random_adjacency(32)), levels=3)
    for parent in parents:
        counts = np.bincount(parent)
        assert counts.max() <= 2


def test_graclus_is_reproducible_under_the_same_seed():
    A = scipy.sparse.csr_matrix(_random_adjacency(24))
    _, p1 = graclus(A, levels=3, seed=7)
    _, p2 = graclus(A, levels=3, seed=7)
    for a, b in zip(p1, p2):
        assert np.array_equal(a, b)


def test_graclus_pairs_neighbours_on_a_ring():
    """On a ring every cluster must be two adjacent nodes (or a singleton)."""
    A = _ring(16)
    _, parents = graclus(scipy.sparse.csr_matrix(A), levels=1, seed=0)
    for cluster in np.unique(parents[0]):
        members = np.flatnonzero(parents[0] == cluster)
        if len(members) == 2:
            assert A[members[0], members[1]] > 0


# --------------------------------------------------------------------------
# permutation / binary tree
# --------------------------------------------------------------------------

def test_compute_perm_produces_a_permutation_per_layer():
    _, parents = graclus(scipy.sparse.csr_matrix(_random_adjacency(32)), levels=3)
    perms = compute_perm(parents)
    for i, layer in enumerate(perms):
        assert sorted(layer) == list(range(len(layer)))
        if i > 0:
            assert len(perms[i - 1]) == 2 * len(layer)


def test_permuted_siblings_share_a_parent():
    """The core invariant: nodes 2i and 2i+1 map to the same coarse node."""
    _, parents = graclus(scipy.sparse.csr_matrix(_random_adjacency(32)), levels=3)
    perms = compute_perm(parents)

    parent = parents[0]
    n_real = len(parent)
    layer = perms[0]
    for i in range(0, len(layer), 2):
        a, b = layer[i], layer[i + 1]
        pa = parent[a] if a < n_real else None
        pb = parent[b] if b < n_real else None
        if pa is not None and pb is not None:
            assert pa == pb, f"nodes {a},{b} at slots {i},{i+1} have parents {pa},{pb}"


def test_perm_data_places_real_channels_and_zeros_the_fakes():
    X = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    indices = [2, 0, 5, 1, 4, 3]  # 4 and 5 are fake (>= n_channels == 4)
    out = perm_data(X, indices)
    assert out.shape == (2, 6, 3)
    assert np.array_equal(out[:, 0, :], X[:, 2, :])
    assert np.array_equal(out[:, 1, :], X[:, 0, :])
    assert np.array_equal(out[:, 3, :], X[:, 1, :])
    assert np.all(out[:, 2, :] == 0)  # fake
    assert np.all(out[:, 4, :] == 0)  # fake


def test_perm_data_zeros_let_max_pooling_pick_the_real_sibling():
    """Why fakes are zero: a stride-2 max must return the real node's value."""
    X = np.full((1, 2, 1), 5.0, dtype=np.float32)
    out = perm_data(X, [0, 2, 1, 3])  # each real node paired with a fake
    pooled = out.reshape(1, 2, 2, 1).max(axis=2)
    assert np.allclose(pooled, 5.0)


def test_perm_data_is_identity_for_none():
    X = np.zeros((2, 3, 4))
    assert perm_data(X, None) is X


def test_perm_data_rejects_a_too_small_permutation():
    with pytest.raises(ValueError):
        perm_data(np.zeros((1, 8, 4)), [0, 1, 2])


def test_perm_adjacency_pads_and_stays_symmetric():
    A = scipy.sparse.csr_matrix(_ring(8))
    out = perm_adjacency(A, list(range(10)))
    assert out.shape == (10, 10)
    assert np.abs(out - out.T).mean() < 1e-9


# --------------------------------------------------------------------------
# full pipeline at the reference's configuration
# --------------------------------------------------------------------------

def test_coarsen_at_the_reference_configuration():
    """64 electrodes, levels=5 -> 6 graphs, each half the previous."""
    A = _random_adjacency(64, seed=1)
    graphs, perm = coarsen(A, levels=5, seed=0)

    assert len(graphs) == 6
    sizes = [g.shape[0] for g in graphs]
    for a, b in zip(sizes, sizes[1:]):
        assert b == a // 2, f"sizes {sizes} are not a clean binary hierarchy"
    assert len(perm) == sizes[0]


def test_coarsened_graphs_have_no_self_loops_by_default():
    graphs, _ = coarsen(_random_adjacency(32), levels=3, seed=0)
    for g in graphs:
        assert np.allclose(g.diagonal(), 0.0)


def test_pooling_six_times_reaches_one_node():
    """p=[2]*6 over the reference hierarchy must land on a single node."""
    A = _random_adjacency(64, seed=1)
    graphs, perm = coarsen(A, levels=5, seed=0)
    n = len(perm)
    for _ in range(6):
        assert n % 2 == 0
        n //= 2
    assert n == 1


def test_perm_data_then_pooling_preserves_trial_count():
    A = _random_adjacency(64, seed=1)
    _, perm = coarsen(A, levels=5, seed=0)
    X = np.random.default_rng(0).normal(size=(7, 64, 640)).astype(np.float32)
    Xp = perm_data(X, perm)
    assert Xp.shape == (7, len(perm), 640)
    assert np.count_nonzero(Xp.sum(axis=(0, 2))) <= 64
