"""Tests for data.interpolate: spherical-spline repair of missing channels."""

from pathlib import Path

import numpy as np
import pytest

from data.interpolate import interpolation_matrix, repair_full_cap
from data.montage import common_average, positions_3d

_CACHE = Path(__file__).resolve().parents[1] / "data" / "cache_bp830"


#: The 64 standardised channel names in cache order, hard-coded so the tests that need
#: only the names run on a fresh clone. test_channel_names_match_the_cache asserts they
#: equal what data/cache_bp830 holds whenever that cache exists.
_CH_NAMES = [
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "Fp1", "Fpz", "Fp2", "AF7",
    "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8", "P7", "P5", "P3", "P1", "Pz",
    "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz",
]


def _cached_ch_names() -> list[str]:
    return list(_CH_NAMES)


def test_channel_names_match_the_cache():
    if not (_CACHE / "S001.npz").exists():
        pytest.skip("data/cache_bp830/S001.npz absent; fresh clone")
    from data.cache import load_subject

    assert len(_CH_NAMES) == 64
    assert list(load_subject("S001", _CACHE).ch_names) == _CH_NAMES


def _mask(n: int, missing: list[int]) -> np.ndarray:
    keep = np.ones(n, dtype=bool)
    keep[missing] = False
    return keep


def test_repair_with_nothing_missing_is_car():
    X = np.random.default_rng(0).normal(size=(3, 8, 20))
    keep = np.ones(8, dtype=bool)
    out = repair_full_cap(X, keep, None)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, common_average(X), atol=1e-6)


def test_missing_channels_without_a_matrix_is_rejected():
    X = np.zeros((1, 4, 5))
    with pytest.raises(ValueError, match="W"):
        repair_full_cap(X, _mask(4, [1]), None)


def test_matrix_shape_is_checked_against_the_mask():
    X = np.zeros((1, 4, 5))
    with pytest.raises(ValueError, match="shape"):
        repair_full_cap(X, _mask(4, [1]), np.zeros((2, 3)))


def test_interpolation_matrix_shape_and_dtype():
    names = _cached_ch_names()
    keep = _mask(len(names), [3, 10, 20])
    W = interpolation_matrix(names, keep)
    assert W.shape == (3, len(names) - 3)
    assert W.dtype == np.float64


def test_matrix_matches_mne_interpolate_bads():
    """Guards the private MNE function against upstream changes.

    ``data.interpolate`` calls ``mne.channels.interpolation._make_interpolation_matrix``
    directly. This test runs the same repair through MNE's public
    ``interpolate_bads`` and demands agreement, so a change to the private
    function's behaviour fails here rather than silently shifting results.
    """
    import mne

    names = _cached_ch_names()
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, len(names), 40))
    bad = sorted(rng.choice(len(names), size=6, replace=False).tolist())
    keep = _mask(len(names), bad)

    ours = repair_full_cap(X, keep, interpolation_matrix(names, keep))

    info = mne.create_info(names, 160.0, "eeg")
    epochs = mne.EpochsArray(X.copy(), info, verbose=False)
    epochs.set_montage(
        mne.channels.make_standard_montage("standard_1005"), match_case=False, verbose=False
    )
    epochs.info["bads"] = [names[i] for i in bad]
    epochs.set_eeg_reference("average", projection=False, verbose=False)  # kept channels only
    epochs.interpolate_bads(reset_bads=True, verbose=False)
    epochs.set_eeg_reference("average", projection=False, verbose=False)

    theirs = epochs.get_data()
    scale = np.abs(theirs).max()
    np.testing.assert_allclose(ours, theirs, rtol=1e-5, atol=1e-5 * scale)


def test_smooth_field_is_recovered():
    names = _cached_ch_names()
    x = positions_3d(names)[:, 0]
    s = np.random.default_rng(2).normal(size=(5, 1, 30))
    X = (x[None, :, None] * s).astype(np.float64)
    missing = list(range(3, 64, 8))
    assert len(missing) == 8
    keep = _mask(len(names), missing)

    out = repair_full_cap(X, keep, interpolation_matrix(names, keep))
    truth = common_average(X)
    r = np.corrcoef(out[:, ~keep, :].ravel(), truth[:, ~keep, :].ravel())[0, 1]
    assert r > 0.95
