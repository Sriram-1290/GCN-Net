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
