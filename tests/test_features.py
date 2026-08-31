"""Tests for band-power node features.

These check that the feature actually measures what it claims — a synthetic sine
at a known frequency must produce power in the band containing it and not in the
others. Without that, a subtle indexing error would silently feed the network
noise and every downstream result would be meaningless but plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.features import BAND_SETS, band_power_features, resolve_bands
from data.normalize import fit_standardizer

SFREQ = 160.0


def _sine(freq: float, n_trials=2, n_channels=3, n_time=320, amp=1.0, seed=0):
    t = np.arange(n_time) / SFREQ
    rng = np.random.default_rng(seed)
    sig = amp * np.sin(2 * np.pi * freq * t)
    return (sig + 0.01 * rng.normal(size=(n_trials, n_channels, n_time))).astype(np.float32)


def test_power_lands_in_the_band_containing_the_signal():
    bands = ((8, 12), (12, 16), (16, 20), (20, 24))
    X = _sine(14.0)  # squarely inside the second band
    feat, _ = band_power_features(X, SFREQ, bands, window=160, stride=160)
    mean_per_band = feat.mean(axis=(0, 1))
    assert int(np.argmax(mean_per_band)) == 1, mean_per_band


def test_a_different_frequency_moves_the_peak():
    bands = ((8, 12), (12, 16), (16, 20), (20, 24))
    for freq, expect in [(10.0, 0), (14.0, 1), (18.0, 2), (22.0, 3)]:
        feat, _ = band_power_features(_sine(freq), SFREQ, bands, 160, 160)
        assert int(np.argmax(feat.mean(axis=(0, 1)))) == expect, freq


def test_amplitude_increases_log_power():
    bands = ((8, 12),)
    quiet, _ = band_power_features(_sine(10.0, amp=1.0), SFREQ, bands, 160, 160)
    loud, _ = band_power_features(_sine(10.0, amp=10.0), SFREQ, bands, 160, 160)
    # log power of a 10x amplitude signal is ~log(100) higher
    assert loud.mean() - quiet.mean() == pytest.approx(np.log(100), rel=0.05)


def test_output_shape_and_trial_index():
    X = _sine(10.0, n_trials=4, n_channels=8, n_time=640)
    bands = BAND_SETS["mu_beta"]
    feat, idx = band_power_features(X, SFREQ, bands, window=160, stride=160)
    assert feat.shape == (4 * 4, 8, len(bands))   # 4 windows per trial
    assert idx.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]


def test_windows_do_not_cross_trials():
    X = np.zeros((3, 2, 320), dtype=np.float32)
    X[1] = _sine(10.0, n_trials=1, n_channels=2, n_time=320)[0]
    feat, idx = band_power_features(X, SFREQ, ((8, 12),), 160, 160)
    # only trial 1 carries signal, so only its rows have appreciably more power
    assert feat[idx == 1].mean() > feat[idx == 0].mean()


def test_too_coarse_a_window_is_rejected():
    """0.4 s gives 2.5 Hz resolution, too coarse to resolve a 4 Hz band."""
    with pytest.raises(ValueError, match="resolution"):
        band_power_features(_sine(10.0), SFREQ, ((8, 12),), window=64, stride=64)


def test_rejects_bad_shapes_and_strides():
    with pytest.raises(ValueError):
        band_power_features(np.zeros((2, 3)), SFREQ, ((8, 12),), 160, 160)
    with pytest.raises(ValueError):
        band_power_features(_sine(10.0), SFREQ, ((8, 12),), window=160, stride=0)


def test_log_can_be_disabled():
    lin, _ = band_power_features(_sine(10.0), SFREQ, ((8, 12),), 160, 160, log=False)
    assert lin.min() >= 0.0


def test_band_sets_resolve_by_name_and_reject_unknown():
    assert resolve_bands("mu_beta") == BAND_SETS["mu_beta"]
    assert resolve_bands(((8, 13),)) == ((8, 13),)
    with pytest.raises(ValueError, match="unknown band set"):
        resolve_bands("not_a_band_set")


def test_all_named_band_sets_are_usable_at_one_second():
    X = _sine(10.0, n_time=640)
    for name in BAND_SETS:
        feat, _ = band_power_features(X, SFREQ, BAND_SETS[name], 160, 160)
        assert np.isfinite(feat).all(), name


def test_per_feature_standardiser_keeps_bands_separate():
    """Bands differ in scale by orders of magnitude; collapsing them is wrong."""
    feat = np.stack([np.full((4, 3), 1.0), np.full((4, 3), 100.0)], axis=-1)
    feat += np.random.default_rng(0).normal(0, .1, feat.shape)
    std = fit_standardizer(feat.astype(np.float32), per_feature=True)
    assert std.mean.shape == (3, 2)
    z = std.apply(feat.astype(np.float32))
    assert abs(z[..., 0].mean()) < 0.2 and abs(z[..., 1].mean()) < 0.2


def test_channel_standardiser_would_conflate_bands():
    """Contrast: the default mode gives one statistic per channel, not per band."""
    feat = np.stack([np.full((4, 3), 1.0), np.full((4, 3), 100.0)], axis=-1)
    feat += np.random.default_rng(0).normal(0, .1, feat.shape)
    std = fit_standardizer(feat.astype(np.float32), per_feature=False)
    assert std.mean.shape == (3, 1)
