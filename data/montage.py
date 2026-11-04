"""Montage utilities: re-reference epochs over a chosen subset of electrodes.

Why every condition is re-referenced (montage-robustness spec, section 4.1).
The standard cache stores epochs already referenced to the mean of all 64
electrodes. If a reduced-montage test condition simply *dropped* channels from
that cache, the surviving signals would still carry the average of 64 electrodes
that are, in the condition being simulated, absent from the cap. A real
low-density cap can only average over the electrodes it has. So each condition
re-references over its own kept channels, and that needs epochs stored *before*
the common average -- the no-reference cache (``data/cache_bp830_noref``).

Equivalence with the cached pipeline: the common average is linear and acts on
each time sample independently (it subtracts the across-channel mean at that
sample), so it commutes with epoching. Applying it to epochs here is the same as
applying it to the continuous recording before epoching, as ``edf_loader`` does.

Leakage: the operation uses no statistics across trials or subjects, only the
channel mean within one sample of one trial, so it is safe to apply to test data.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np


def common_average(X: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """Re-reference to the mean of the kept channels and drop the others.

    Args:
        X: epochs, shape ``(trials, channels, time)``.
        keep: boolean mask of length ``channels``; ``None`` keeps every channel.

    Returns:
        float32 array of shape ``(trials, keep.sum(), time)`` in which, at every
        sample, the kept channels sum to zero. Channels outside ``keep`` do not
        influence the result.
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"X must be (trials, channels, time); got shape {X.shape}")
    n_channels = X.shape[1]
    if keep is None:
        keep = np.ones(n_channels, dtype=bool)
    else:
        keep = np.asarray(keep)
        if keep.dtype != bool or keep.shape != (n_channels,):
            raise ValueError(
                f"keep must be a boolean vector of length {n_channels}; "
                f"got dtype {keep.dtype}, shape {keep.shape}"
            )
    if not keep.any():
        raise ValueError("keep selects no channels; a common average needs at least one")

    # Accumulate in float64: the mean of up to 64 channels in float32 would add
    # rounding error comparable to what the equivalence check tolerates.
    sub = X[:, keep, :].astype(np.float64)
    return (sub - sub.mean(axis=1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------
# Why regions at all: a random channel pattern can, by bad luck, delete every
# electrode over one part of the scalp (e.g. all of the left-posterior cortex).
# Such a pattern is not "a sparser cap", it is a cap with a hole, and the model
# comparison would then measure how each model copes with an absent region
# rather than with fewer electrodes. Every pattern we score must therefore keep
# at least one electrode in each of nine coarse regions; patterns that do not
# are rejected and redrawn (see ``available_masks``).
#
# The nine regions are {left, midline, right} x {frontal, central, posterior}.
# Both axes are read off the 10-10 electrode name: the numeric suffix gives the
# side (``z`` = midline, odd = left, even = right) and the letter prefix gives
# the row. Parsing the name keeps the map free of any montage file and makes it
# deterministic and auditable.

REGION_NAMES: tuple[str, ...] = (
    "left-frontal", "midline-frontal", "right-frontal",
    "left-central", "midline-central", "right-central",
    "left-posterior", "midline-posterior", "right-posterior",
)

_ROW_OF_PREFIX = {
    "fp": 0, "af": 0, "f": 0, "fc": 0, "ft": 0,   # frontal
    "c": 1, "t": 1, "cp": 1, "tp": 1,             # central (incl. temporal)
    "p": 2, "po": 2, "o": 2, "i": 2,              # posterior
}
# Lazy prefix so that "POz" splits as ("PO", "z") and "Fp1" as ("Fp", "1").
_NAME_RE = re.compile(r"^([A-Za-z]+?)(z|\d+)$", re.IGNORECASE)


def region_of(name: str) -> int:
    """Index (into ``REGION_NAMES``) of the scalp region an electrode lies in.

    Raises:
        ValueError: if the name does not parse or its prefix is not one of the
            known rows. Failing loudly matters: a silently misplaced channel
            would make the occupancy check pass for the wrong reason.
    """
    match = _NAME_RE.match(name)
    if match is None:
        raise ValueError(f"unrecognised electrode name: {name!r}")
    prefix, suffix = match.group(1).lower(), match.group(2).lower()
    if prefix not in _ROW_OF_PREFIX:
        raise ValueError(f"unrecognised electrode prefix {prefix!r} in {name!r}")
    if suffix == "z":
        column = 1
    else:
        column = 0 if int(suffix) % 2 == 1 else 2
    return 3 * _ROW_OF_PREFIX[prefix] + column


def region_vector(ch_names: list[str]) -> np.ndarray:
    """Region index of every channel, as an int64 vector aligned with ``ch_names``."""
    return np.array([region_of(c) for c in ch_names], dtype=np.int64)


# ---------------------------------------------------------------------------
# Named montages
# ---------------------------------------------------------------------------

#: The 22 electrodes of BCI Competition IV-2a, as the nearest 10-10 sites. Names
#: follow the PhysioNet 64-channel cap, where IV-2a's exact sites exist.
IV2A_CHANNELS: tuple[str, ...] = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
)

#: The 19 electrodes of the classic clinical 10-20 system (modern T7/T8/P7/P8
#: names for T3/T4/T5/T6).
TEN_TWENTY_CHANNELS: tuple[str, ...] = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
)


def positions_3d(ch_names: list[str]) -> np.ndarray:
    """3-D scalp positions ``(C, 3)`` from MNE's ``standard_1005`` montage.

    Names are matched case-insensitively, as in ``visualization.topography``.
    Three dimensions, not two: a top-down projection squeezes the temporal rim
    of the head, so electrodes there would look closer to one another than they
    are, and a "cluster" grown by 2-D distance would not be a physical patch.
    """
    import mne

    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    lookup = {k.lower(): v for k, v in pos.items()}
    missing = [c for c in ch_names if c.lower() not in lookup]
    if missing:
        raise ValueError(f"channels absent from standard_1005: {missing}")
    return np.array([lookup[c.lower()] for c in ch_names], dtype=np.float64)


# ---------------------------------------------------------------------------
# Conditions and seeded channel patterns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """One test condition: which electrodes are available at test time.

    ``param`` depends on ``kind``: ``montage`` -> tuple of kept channel names;
    ``random`` -> fraction of channels *removed*; ``cluster`` -> number of
    channels removed (a centre electrode plus its nearest neighbours);
    ``full`` -> ``None``.
    """

    name: str
    kind: Literal["full", "montage", "random", "cluster"]
    param: float | int | tuple[str, ...] | None = None


CONDITIONS: tuple[Condition, ...] = (
    Condition("full", "full"),
    Condition("iv2a", "montage", IV2A_CHANNELS),
    Condition("random25", "random", 0.25),
    Condition("tentwenty", "montage", TEN_TWENTY_CHANNELS),
    Condition("random10", "random", 0.10),
    Condition("random50", "random", 0.50),
    Condition("cluster8", "cluster", 8),
    Condition("cluster16", "cluster", 16),
)

#: The two conditions the decision rule is applied to.
PRIMARY: tuple[str, ...] = ("iv2a", "random25")

_MAX_ATTEMPTS = 1000


def _seed_hash(subject: str) -> int:
    """Stable 64-bit integer from a subject id.

    Python's built-in ``hash`` is salted per process, which would give a
    different channel pattern on every run; SHA-256 is the same everywhere.
    """
    return int.from_bytes(hashlib.sha256(subject.encode("utf-8")).digest()[:8], "big")


def _draw_mask(
    cond: Condition,
    n_channels: int,
    positions: np.ndarray | None,
    regions: np.ndarray,
    subject: str,
    fold: int,
    draw: int,
) -> np.ndarray:
    """One occupancy-valid keep-mask, redrawing with ``attempt + 1`` on rejection."""
    for attempt in range(_MAX_ATTEMPTS):
        rng = np.random.default_rng([_seed_hash(subject), fold, draw, attempt])
        keep = np.ones(n_channels, dtype=bool)
        if cond.kind == "random":
            n_remove = int(round(float(cond.param) * n_channels))
            keep[rng.permutation(n_channels)[:n_remove]] = False
        else:  # cluster
            k = int(cond.param)
            centre = int(rng.integers(n_channels))
            dist = np.linalg.norm(positions - positions[centre], axis=1)
            # The centre is at distance 0, so it heads the list: the k removed
            # are the centre plus its k - 1 nearest neighbours. Stable sort
            # makes ties deterministic.
            keep[np.argsort(dist, kind="stable")[:k]] = False
        if len(np.unique(regions[keep])) == len(REGION_NAMES):
            return keep
    raise RuntimeError(
        f"no pattern occupying all {len(REGION_NAMES)} regions in {_MAX_ATTEMPTS} "
        f"attempts for {cond.name} ({subject}, fold {fold}, draw {draw})"
    )


def available_masks(
    cond: Condition,
    ch_names: list[str],
    subject: str,
    fold: int,
    n_draws: int = 10,
) -> list[np.ndarray]:
    """Boolean keep-masks (length ``C``) for a condition.

    ``full`` and ``montage`` are fixed, so they yield one mask. ``random`` and
    ``cluster`` yield ``n_draws`` masks, each seeded by
    ``(subject, fold, draw, attempt)`` so the same call always returns the same
    patterns and different folds or subjects get different ones. The pattern is
    a function of identifiers only -- never of any EEG data -- so it cannot
    leak target-subject information.
    """
    n_channels = len(ch_names)
    if cond.kind == "full":
        return [np.ones(n_channels, dtype=bool)]
    if cond.kind == "montage":
        wanted = {c.lower() for c in cond.param}
        mask = np.array([c.lower() in wanted for c in ch_names], dtype=bool)
        if int(mask.sum()) != len(wanted):
            absent = sorted(wanted - {c.lower() for c in ch_names})
            raise ValueError(f"montage {cond.name!r} needs channels not in the data: {absent}")
        return [mask]
    regions = region_vector(ch_names)
    positions = positions_3d(ch_names) if cond.kind == "cluster" else None
    return [
        _draw_mask(cond, n_channels, positions, regions, subject, fold, d)
        for d in range(n_draws)
    ]
