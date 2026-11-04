"""Tests for data.montage: common-average re-referencing over a channel subset."""

from pathlib import Path

import numpy as np
import pytest

from data.montage import (
    CONDITIONS,
    IV2A_CHANNELS,
    PRIMARY,
    REGION_NAMES,
    TEN_TWENTY_CHANNELS,
    available_masks,
    common_average,
    positions_3d,
    region_of,
    region_vector,
)

_CACHE = Path(__file__).resolve().parents[1] / "data" / "cache_bp830"


def _cached_ch_names() -> list[str]:
    """The 64 standardised channel names; skip when the cache is not on disk."""
    if not (_CACHE / "S001.npz").exists():
        pytest.skip("data/cache_bp830/S001.npz absent; fresh clone")
    from data.cache import load_subject

    return list(load_subject("S001", _CACHE).ch_names)


def _cond(name: str):
    return next(c for c in CONDITIONS if c.name == name)


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


def test_region_examples():
    expected = {
        "C3": "left-central",
        "Cz": "midline-central",
        "C4": "right-central",
        "FC1": "left-frontal",
        "T10": "right-central",
        "POz": "midline-posterior",
        "Iz": "midline-posterior",
        "Fp2": "right-frontal",
    }
    for name, region in expected.items():
        assert REGION_NAMES[region_of(name)] == region, name
    with pytest.raises(ValueError):
        region_of("Nonsense")


def test_every_dataset_channel_has_a_region():
    ch_names = _cached_ch_names()
    vec = region_vector(ch_names)
    assert vec.dtype == np.int64 and vec.shape == (len(ch_names),)
    assert set(vec.tolist()) == set(range(9))


@pytest.mark.parametrize("montage", [IV2A_CHANNELS, TEN_TWENTY_CHANNELS])
def test_montages_occupy_every_region(montage):
    assert set(region_vector(list(montage)).tolist()) == set(range(9))


def test_named_montages_are_subsets_of_the_dataset():
    ch_names = _cached_ch_names()
    lowered = {c.lower() for c in ch_names}
    assert len(IV2A_CHANNELS) == 22 and len(TEN_TWENTY_CHANNELS) == 19
    for ch in IV2A_CHANNELS + TEN_TWENTY_CHANNELS:
        assert ch.lower() in lowered, ch
    for name, size in (("iv2a", 22), ("tentwenty", 19)):
        (mask,) = available_masks(_cond(name), ch_names, "S001", 0)
        assert mask.dtype == bool and int(mask.sum()) == size


def test_condition_table():
    assert [c.name for c in CONDITIONS] == [
        "full", "iv2a", "random25", "tentwenty",
        "random10", "random50", "cluster8", "cluster16",
    ]
    assert PRIMARY == ("iv2a", "random25")


def test_random25_shape_and_occupancy():
    ch_names = _cached_ch_names()
    regions = region_vector(ch_names)
    masks = available_masks(_cond("random25"), ch_names, "S001", 0)
    assert len(masks) == 10
    for m in masks:
        assert m.dtype == bool and m.shape == (64,)
        assert int(m.sum()) == 48
        assert set(regions[m].tolist()) == set(range(9))


def test_masks_are_deterministic_and_seed_dependent():
    ch_names = _cached_ch_names()
    cond = _cond("random25")
    a = available_masks(cond, ch_names, "S001", 0)
    b = available_masks(cond, ch_names, "S001", 0)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    c = available_masks(cond, ch_names, "S001", 1)
    assert any(not np.array_equal(x, y) for x, y in zip(a, c))
    d = available_masks(cond, ch_names, "S002", 0)
    assert any(not np.array_equal(x, y) for x, y in zip(a, d))


def test_full_returns_one_all_true_mask():
    ch_names = _cached_ch_names()
    (mask,) = available_masks(_cond("full"), ch_names, "S001", 0)
    assert mask.all() and mask.shape == (64,)


def test_cluster8_removes_nearest_neighbours():
    ch_names = _cached_ch_names()
    pos = positions_3d(ch_names)
    assert pos.shape == (64, 3)
    masks = available_masks(_cond("cluster8"), ch_names, "S001", 0)
    assert len(masks) == 10
    for m in masks:
        removed = set(np.flatnonzero(~m).tolist())
        assert len(removed) == 8
        # Some removed electrode is a centre whose 7 nearest neighbours are the rest.
        is_cluster = False
        for centre in removed:
            d = np.linalg.norm(pos - pos[centre], axis=1)
            if set(np.argsort(d, kind="stable")[:8].tolist()) == removed:
                is_cluster = True
        assert is_cluster


def test_cluster_and_random_masks_keep_every_region():
    ch_names = _cached_ch_names()
    regions = region_vector(ch_names)
    for name in ("random10", "random50", "cluster8", "cluster16"):
        for m in available_masks(_cond(name), ch_names, "S003", 2):
            assert set(regions[m].tolist()) == set(range(9)), name
