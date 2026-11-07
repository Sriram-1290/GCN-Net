"""Tests for Euclidean alignment on synthetic data (Phase A, Review Focus 1)."""

from __future__ import annotations

import numpy as np
import pytest

from data.alignment import (
    EAReference,
    align,
    euclidean_reference,
    fit_ea,
    inv_sqrt_psd,
    trial_covariances,
)


def _car(X: np.ndarray) -> np.ndarray:
    """Common average reference over the channel axis (rank C-1 afterwards)."""
    return X - X.mean(axis=1, keepdims=True)


def test_full_rank_whitens_to_identity():
    rng = np.random.default_rng(0)
    mix = rng.standard_normal((8, 8))
    X = np.einsum("ij,njt->nit", mix, rng.standard_normal((50, 8, 200)))
    ref = fit_ea(X)
    assert isinstance(ref, EAReference)
    assert ref.rank == 8 and ref.n_trials == 50
    aligned = align(X, ref.W)
    assert aligned.dtype == np.float32 and aligned.shape == X.shape
    np.testing.assert_allclose(euclidean_reference(aligned), np.eye(8), atol=1e-5)


def test_car_rank_deficient_no_blowup():
    rng = np.random.default_rng(1)
    mix = rng.standard_normal((8, 8))
    X = _car(np.einsum("ij,njt->nit", mix, rng.standard_normal((50, 8, 200))))
    X = X.astype(np.float32)
    ref = fit_ea(X)
    assert ref.rank == 7
    aligned = align(X, ref.W)
    eig = np.linalg.eigvalsh(euclidean_reference(aligned))
    np.testing.assert_allclose(eig[1:], 1.0, atol=1e-4)
    assert abs(eig[0]) < 1e-4
    assert eig.max() < 1 + 1e-4
    assert np.abs(aligned).max() < 1e3 * np.abs(X).max()


def test_reference_uses_only_given_trials():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((30, 6, 100))
    W0 = fit_ea(X[:20]).W
    X2 = X.copy()
    X2[20:] *= 100.0  # trials not handed to fit_ea
    W1 = fit_ea(X2[:20]).W
    np.testing.assert_array_equal(W0, W1)


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        trial_covariances(np.zeros((8, 100)))
    with pytest.raises(ValueError):
        fit_ea(np.zeros((8, 100)))
    with pytest.raises(ValueError):
        euclidean_reference(np.zeros((8, 100)))
    with pytest.raises(ValueError):
        inv_sqrt_psd(np.zeros((4, 5)))
    with pytest.raises(ValueError):
        align(np.zeros((3, 8, 50)), np.eye(7))
