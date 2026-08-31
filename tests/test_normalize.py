"""Tests that normalisation statistics never see data they were not fit on."""

from __future__ import annotations

import numpy as np
import pytest

from data.normalize import fit_standardizer


def _synthetic(n_trials=20, n_channels=5, n_samples=64, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Give each channel a distinct scale, so per-channel statistics matter.
    scale = np.arange(1, n_channels + 1)[None, :, None]
    return (rng.normal(size=(n_trials, n_channels, n_samples)) * scale).astype(np.float32)


def test_standardising_the_training_split_gives_zero_mean_unit_variance():
    X = _synthetic()
    std = fit_standardizer(X)
    Z = std.apply(X)
    assert np.allclose(Z.mean(axis=(0, 2)), 0.0, atol=1e-4)
    assert np.allclose(Z.std(axis=(0, 2)), 1.0, atol=1e-4)


def test_statistics_come_only_from_the_trials_passed_in():
    X = _synthetic(n_trials=40)
    train, test = X[:20], X[20:]

    from_train = fit_standardizer(train)
    from_all = fit_standardizer(X)

    assert not np.allclose(from_train.mean, from_all.mean)
    assert from_train.n_trials_fit == 20


def test_test_data_is_not_forced_to_zero_mean():
    """If the test split came out exactly standardised, the stats saw it."""
    X = _synthetic(n_trials=40)
    std = fit_standardizer(X[:20])
    Z_test = std.apply(X[20:])
    assert not np.allclose(Z_test.mean(axis=(0, 2)), 0.0, atol=1e-6)


def test_statistics_are_per_channel():
    std = fit_standardizer(_synthetic(n_channels=5))
    assert std.mean.shape == (5, 1)
    assert std.std.shape == (5, 1)
    # channel scales were 1..5, so the fitted stds must be clearly distinct
    assert std.std.ravel()[-1] > 3 * std.std.ravel()[0]


def test_apply_rejects_a_channel_count_mismatch():
    std = fit_standardizer(_synthetic(n_channels=5))
    with pytest.raises(ValueError, match="channel count"):
        std.apply(_synthetic(n_channels=6))


def test_fit_rejects_empty_and_constant_input():
    with pytest.raises(ValueError):
        fit_standardizer(np.zeros((0, 4, 10)))
    X = _synthetic(n_channels=4)
    X[:, 1, :] = 2.0
    with pytest.raises(ValueError, match="zero variance"):
        fit_standardizer(X)


def test_output_is_float32():
    X = _synthetic()
    assert fit_standardizer(X).apply(X).dtype == np.float32
