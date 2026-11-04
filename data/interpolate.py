"""Spherical-spline repair of missing channels (the fair baseline for ShallowConvNet).

Why this exists (montage-robustness spec, section 4.1). ShallowConvNet has no
notion of electrode layout: if channels are absent at test time it cannot simply
run on fewer inputs, because its first layer expects all 64. The standard remedy
-- the one an EEG practitioner would use -- is to *interpolate* the missing
channels from the ones that remain, then feed the full 64-channel array to the
unchanged network. Comparing the graph model against ShallowConvNet *without*
this repair would be a straw man, so this module gives the baseline its best
fair shot.

Method: Perrin et al. (1989) spherical splines, the algorithm behind
``mne.Epochs.interpolate_bads``. The interpolation is linear in the kept
channels, so it reduces to a fixed matrix ``W`` of shape ``(n_missing, n_kept)``
that depends only on electrode positions (never on data), hence leakage-free.

This module calls the **private** MNE function
``mne.channels.interpolation._make_interpolation_matrix`` (no public API returns
the matrix itself). Private functions can change without notice, so
``tests/test_interpolate.py::test_matrix_matches_mne_interpolate_bads`` checks
the whole repair against the public ``interpolate_bads`` and fails if they
diverge.

Sphere convention. ``interpolate_bads`` works in MNE's *head* coordinate frame
(``set_montage`` re-expresses the montage relative to nasion and ear points) and
subtracts a fitted head-sphere origin before building the matrix: the spline
assumes electrodes on a sphere *centred on the head*, and the fitted centre is
roughly 4.5 cm above the frame's origin. The raw ``standard_1005`` coordinates
that ``data.montage.positions_3d`` returns are in a different (MRI) frame, so
mixing a head-frame origin with MRI-frame positions gives a wrong matrix. We
therefore take positions and origin together from one MNE ``Info`` with the
montage applied, and fit the sphere with the public
``mne.bem.fit_sphere_to_headshape`` as ``interpolate_bads(origin="auto")`` does.
The fit uses the whole 64-channel cap, not the kept subset, so it is identical
across conditions and uses no data.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from data.montage import common_average


@lru_cache(maxsize=8)
def _head_frame_geometry(ch_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Electrode positions ``(C, 3)`` and fitted sphere centre ``(3,)``, in metres.

    Both are in MNE's head frame, exactly as ``interpolate_bads`` sees them.
    Cached because the sphere fit is slow next to the matrix algebra and the cap
    is the same for every condition.
    """
    import mne

    info = mne.create_info(list(ch_names), 160.0, "eeg", verbose=False)
    info.set_montage(
        mne.channels.make_standard_montage("standard_1005"), match_case=False, verbose=False
    )
    pos = np.array([ch["loc"][:3] for ch in info["chs"]], dtype=np.float64)
    _, origin, _ = mne.bem.fit_sphere_to_headshape(info, units="m", verbose=False)
    return pos, np.asarray(origin, dtype=np.float64)


def interpolation_matrix(ch_names: list[str], keep: np.ndarray) -> np.ndarray:
    """Matrix ``W`` mapping kept-channel signals to missing-channel signals.

    Args:
        ch_names: all channel names, in the data's channel order.
        keep: boolean mask over ``ch_names``; ``True`` where the electrode is present.

    Returns:
        float64 array of shape ``(n_missing, n_kept)``: missing = ``W @ kept``.
        Rows follow the order of the ``False`` entries of ``keep``, columns the
        order of its ``True`` entries.
    """
    from mne.channels.interpolation import _make_interpolation_matrix

    keep = np.asarray(keep)
    if keep.dtype != bool or keep.shape != (len(ch_names),):
        raise ValueError(
            f"keep must be a boolean vector of length {len(ch_names)}; "
            f"got dtype {keep.dtype}, shape {keep.shape}"
        )
    if not keep.any():
        raise ValueError("keep selects no channels; nothing to interpolate from")

    pos, origin = _head_frame_geometry(tuple(ch_names))
    pos = pos - origin
    W = _make_interpolation_matrix(pos[keep], pos[~keep])
    return np.asarray(W, dtype=np.float64)


def repair_full_cap(X_noref: np.ndarray, keep: np.ndarray, W: np.ndarray | None) -> np.ndarray:
    """Return a full-cap array in which missing channels are interpolated.

    Order of operations is fixed (spec section 4.1), and each step has a reason:

    1. Re-reference the kept channels to *their own* average -- the only
       reference a real reduced cap could compute.
    2. Fill each missing channel with ``W @ kept``. This must follow step 1 so
       that the interpolated values live in the same referenced signal space as
       the kept channels they are built from.
    3. Re-reference all channels to their common average, so the output matches
       the full-cap convention the network was trained on (zero channel mean).
    4. Return float32.

    Args:
        X_noref: epochs before any re-referencing, ``(trials, channels, time)``.
        keep: boolean mask of length ``channels``.
        W: output of ``interpolation_matrix`` for this ``keep``; may be ``None``
            only if no channel is missing.
    """
    X_noref = np.asarray(X_noref)
    kept = common_average(X_noref, keep)  # validates X_noref and keep; step 1
    keep = np.asarray(keep)
    n_trials, n_channels, n_time = X_noref.shape
    n_missing = int((~keep).sum())

    full = np.empty((n_trials, n_channels, n_time), dtype=np.float64)
    full[:, keep, :] = kept
    if n_missing:
        if W is None:
            raise ValueError(f"{n_missing} channels are missing but W is None")
        W = np.asarray(W)
        expected = (n_missing, int(keep.sum()))
        if W.shape != expected:
            raise ValueError(f"W has shape {W.shape}; expected {expected} for this keep mask")
        # (n_missing, n_kept) x (trials, n_kept, time) -> (trials, n_missing, time)
        full[:, ~keep, :] = np.einsum("mk,tkn->tmn", W, kept.astype(np.float64))  # step 2

    return common_average(full)  # steps 3 and 4
