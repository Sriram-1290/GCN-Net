"""Tests for data.montage: common-average re-referencing over a channel subset."""

import numpy as np
import pytest

from data.montage import common_average


def test_common_average_all_channels_has_zero_channel_mean():
    X = np.random.default_rng(0).normal(size=(3, 8, 50))
    out = common_average(X)
    assert out.shape == (3, 8, 50)
    assert out.dtype == np.float32
    assert np.abs(out.mean(axis=1)).max() < 1e-6


def test_common_average_subset_uses_only_kept_channels():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3, 8, 50))
    keep = np.zeros(8, dtype=bool)
    keep[:4] = True

    out = common_average(X, keep)
    expected = X[:, :4] - X[:, :4].mean(axis=1, keepdims=True)
    assert out.shape == (3, 4, 50)
    np.testing.assert_allclose(out, expected, atol=1e-6)

    X2 = X.copy()
    X2[:, 4:] += 100.0  # dropped electrodes must have no influence
    np.testing.assert_array_equal(common_average(X2, keep), out)


def test_common_average_rejects_bad_shapes():
    with pytest.raises(ValueError):
        common_average(np.zeros((8, 50)))
    with pytest.raises(ValueError):
        common_average(np.zeros((3, 8, 50)), np.ones(7, dtype=bool))
    with pytest.raises(ValueError):
        common_average(np.zeros((3, 8, 50)), np.zeros(8, dtype=bool))
