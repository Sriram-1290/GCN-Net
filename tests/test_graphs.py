"""Tests for adjacency construction and the Laplacian pipeline.

The Chebyshev tests check the recurrence against hand-computable values on a tiny
graph, which is the only way to be confident the dense port matches the
reference's ``chebyshev5`` before any training happens.
"""

from __future__ import annotations

import numpy as np
import pytest

from graphs.adjacency import pcc_adjacency, threshold_adjacency
from graphs.laplacian import (
    chebyshev_basis,
    laplacian,
    largest_eigenvalue,
    rescale_laplacian,
)


def _synthetic(n_trials=8, n_channels=6, n_samples=64, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_trials, n_channels, n_samples)).astype(np.float32)


# --------------------------------------------------------------------------
# adjacency
# --------------------------------------------------------------------------

def test_adjacency_is_symmetric_zero_diagonal_and_bounded():
    A = pcc_adjacency(_synthetic())
    assert np.allclose(A, A.T)
    assert np.allclose(np.diag(A), 0.0)
    assert A.min() >= 0.0 and A.max() <= 1.0


def test_adjacency_recovers_a_known_correlation():
    """Two channels that are exact copies must have weight 1; sign is discarded."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(4, 1, 50))
    X = np.concatenate([base, base, -base], axis=1)  # ch0 == ch1 == -ch2
    A = pcc_adjacency(X)
    assert A[0, 1] == pytest.approx(1.0, abs=1e-6)
    assert A[0, 2] == pytest.approx(1.0, abs=1e-6)  # |P|, so anticorrelation is 1 too


def test_adjacency_rejects_a_constant_channel():
    X = _synthetic(n_channels=4)
    X[:, 2, :] = 3.0
    with pytest.raises(ValueError, match="constant"):
        pcc_adjacency(X)


def test_adjacency_rejects_wrong_rank():
    with pytest.raises(ValueError):
        pcc_adjacency(np.zeros((10, 5)))


def test_adjacency_depends_only_on_the_data_it_is_given():
    """Train-only and all-data adjacencies must differ — the leakage this prevents."""
    X = _synthetic(n_trials=20, seed=3)
    A_train = pcc_adjacency(X[:10])
    A_all = pcc_adjacency(X)
    assert not np.allclose(A_train, A_all)


def test_threshold_keeps_symmetry_and_zero_diagonal():
    A = pcc_adjacency(_synthetic(n_channels=8))
    for kwargs in ({"keep_top_k": 3}, {"min_weight": 0.2}, {"keep_top_k": 2, "min_weight": 0.1}):
        T = threshold_adjacency(A, **kwargs)
        assert np.allclose(T, T.T)
        assert np.allclose(np.diag(T), 0.0)


def test_threshold_without_arguments_is_a_no_op():
    A = pcc_adjacency(_synthetic())
    assert threshold_adjacency(A) is A


# --------------------------------------------------------------------------
# Laplacian
# --------------------------------------------------------------------------

def test_normalized_laplacian_spectrum_is_within_zero_and_two():
    A = pcc_adjacency(_synthetic(n_channels=10))
    eig = np.linalg.eigvalsh(laplacian(A))
    assert eig.min() >= -1e-8
    assert eig.max() <= 2.0 + 1e-8


def test_normalized_laplacian_has_a_zero_eigenvalue():
    """A connected graph has exactly one zero eigenvalue (the constant vector)."""
    A = pcc_adjacency(_synthetic(n_channels=10))
    eig = np.linalg.eigvalsh(laplacian(A))
    assert eig[0] == pytest.approx(0.0, abs=1e-8)


def test_combinatorial_laplacian_rows_sum_to_zero():
    A = pcc_adjacency(_synthetic(n_channels=6))
    L = laplacian(A, normalized=False)
    assert np.allclose(L.sum(axis=1), 0.0)


def test_laplacian_matches_the_hand_written_formula():
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    # d = [1, 1] -> D^-1/2 = I -> L = I - A = [[1,-1],[-1,1]]
    assert np.allclose(laplacian(A), np.array([[1.0, -1.0], [-1.0, 1.0]]))


def test_laplacian_rejects_asymmetric_input():
    with pytest.raises(ValueError, match="symmetric"):
        laplacian(np.array([[0.0, 1.0], [0.5, 0.0]]))


def test_laplacian_rejects_self_loops():
    with pytest.raises(ValueError, match="self-loops"):
        laplacian(np.array([[1.0, 1.0], [1.0, 1.0]]))


# --------------------------------------------------------------------------
# rescaling and Chebyshev basis
# --------------------------------------------------------------------------

def test_rescaled_spectrum_fits_inside_minus_one_to_one():
    A = pcc_adjacency(_synthetic(n_channels=10))
    eig = np.linalg.eigvalsh(rescale_laplacian(laplacian(A), lmax=2.0))
    assert eig.min() >= -1.0 - 1e-8
    assert eig.max() <= 1.0 + 1e-8


def test_rescaling_with_true_lmax_reaches_the_upper_bound():
    """lmax=2 is loose; the computed lambda_max uses the full range."""
    A = pcc_adjacency(_synthetic(n_channels=10))
    L = laplacian(A)
    fixed = np.linalg.eigvalsh(rescale_laplacian(L, lmax=2.0)).max()
    exact = np.linalg.eigvalsh(rescale_laplacian(L, lmax=None)).max()
    assert exact == pytest.approx(1.0, abs=1e-8)
    assert fixed < exact


def test_rescale_matches_the_reference_formula():
    A = pcc_adjacency(_synthetic(n_channels=5))
    L = laplacian(A)
    expected = L / 1.0 - np.eye(5)  # lmax=2 -> divide by 1, subtract I
    assert np.allclose(rescale_laplacian(L, lmax=2.0), expected)


def test_chebyshev_basis_obeys_the_recurrence():
    A = pcc_adjacency(_synthetic(n_channels=6))
    Lt = rescale_laplacian(laplacian(A))
    T = chebyshev_basis(Lt, K=5)
    assert np.allclose(T[0], np.eye(6))
    assert np.allclose(T[1], Lt)
    for k in range(2, 5):
        assert np.allclose(T[k], 2.0 * Lt @ T[k - 1] - T[k - 2])


def test_chebyshev_k2_is_what_the_reference_uses():
    """K=2 means the filter sees [I, L_tilde] only — one-hop neighbourhoods."""
    A = pcc_adjacency(_synthetic(n_channels=6))
    Lt = rescale_laplacian(laplacian(A))
    T = chebyshev_basis(Lt, K=2)
    assert T.shape == (2, 6, 6)
    assert np.allclose(T[0], np.eye(6))
    assert np.allclose(T[1], Lt)


def test_chebyshev_rejects_zero_order():
    with pytest.raises(ValueError):
        chebyshev_basis(np.eye(3), K=0)
