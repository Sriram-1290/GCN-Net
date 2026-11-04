"""Tests for graphs.subset: subset graphs and standardisers from source moments."""

import numpy as np
import pytest

from data.montage import common_average, region_of
from data.normalize import fit_standardizer
from graphs.adjacency import pcc_adjacency
from graphs.laplacian import chebyshev_basis, laplacian, rescale_laplacian
from graphs.subset import moments, subset_representation

_NAMES12 = [f"C{2 * i + 1}" for i in range(12)]


def _random_data(seed=0, shape=(20, 12, 100)):
    """Correlated channels with a per-channel offset, so centring actually matters."""
    rng = np.random.default_rng(seed)
    mixing = rng.standard_normal((shape[1], shape[1]))
    X = np.einsum("cd,ndt->nct", mixing, rng.standard_normal(shape))
    return X + rng.standard_normal((1, shape[1], 1)) * 3.0


def test_moments_add():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((5, 8, 30))
    b = rng.standard_normal((7, 8, 30))
    both = moments(np.concatenate([a, b]))
    added = moments(a) + moments(b)
    assert added.n == both.n == 12 * 30
    np.testing.assert_allclose(added.sum, both.sum, atol=1e-10)
    np.testing.assert_allclose(added.outer, both.outer, atol=1e-10)
    assert added.sum.dtype == np.float64 and added.outer.dtype == np.float64


def test_matches_direct_computation():
    X = _random_data()
    keep = np.zeros(12, dtype=bool)
    keep[[0, 2, 3, 6, 8, 9, 11]] = True

    rep = subset_representation(moments(X), keep, _NAMES12, K=2)

    direct = common_average(X, keep)
    std = fit_standardizer(direct)
    assert rep.standardizer.mean.shape == (7, 1)
    np.testing.assert_allclose(rep.standardizer.mean, std.mean, atol=1e-6)
    np.testing.assert_allclose(rep.standardizer.std, std.std, atol=1e-6)

    # The basis is a function of the adjacency alone, so equality of the bases is
    # equality of the adjacencies that produced them.
    expected = chebyshev_basis(rescale_laplacian(laplacian(pcc_adjacency(direct)), 2.0), 2)
    assert rep.basis.shape == (2, 7, 7)
    assert rep.basis.dtype == np.float32
    np.testing.assert_allclose(rep.basis, expected, atol=1e-6)
    np.testing.assert_array_equal(rep.keep, keep)


def test_full_keep_matches_existing_pipeline():
    X = common_average(_random_data(seed=3))
    rep = subset_representation(moments(X), np.ones(12, dtype=bool), _NAMES12)
    expected = chebyshev_basis(rescale_laplacian(laplacian(pcc_adjacency(X)), 2.0), 2)
    np.testing.assert_allclose(rep.basis, expected, atol=1e-6)


def test_basis_and_regions_share_order():
    names = ["Fp1", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "FC2", "CP1"]
    X = _random_data(seed=4, shape=(15, 12, 80))
    keep = np.zeros(12, dtype=bool)
    keep[[9, 1, 4, 7, 2, 10, 6]] = True  # set in arbitrary order; mask order is montage order
    rep = subset_representation(moments(X), keep, names)

    kept_index = np.flatnonzero(keep)
    assert rep.regions.dtype == np.int64
    assert rep.regions.shape == (7,)
    assert len(set(rep.regions.tolist())) > 2  # spans several regions
    for i, idx in enumerate(kept_index):
        assert rep.regions[i] == region_of(names[idx])

    # Row i of the standardiser and basis must be the same i-th kept channel.
    direct = common_average(X, keep)
    std = fit_standardizer(direct)
    np.testing.assert_allclose(rep.standardizer.mean, std.mean, atol=1e-6)
    expected = chebyshev_basis(rescale_laplacian(laplacian(pcc_adjacency(direct)), 2.0), 2)
    np.testing.assert_allclose(rep.basis, expected, atol=1e-6)


def test_dead_channel_raises():
    X = _random_data(seed=6)
    X[:, 3, :] = 0.0
    X[:, 7, :] = 0.0
    keep = np.zeros(12, dtype=bool)
    keep[[3, 7]] = True  # both identically zero, so both centre to zero
    with pytest.raises(ValueError, match="variance"):
        subset_representation(moments(X), keep, _NAMES12)


def test_input_validation():
    m = moments(_random_data())
    with pytest.raises(ValueError):
        subset_representation(m, np.zeros(12, dtype=bool), _NAMES12)
    with pytest.raises(ValueError):
        subset_representation(m, np.ones(11, dtype=bool), _NAMES12)
    with pytest.raises(ValueError):
        subset_representation(m, np.ones(12, dtype=bool), _NAMES12[:-1])
    with pytest.raises(ValueError):
        moments(_random_data()[0])
