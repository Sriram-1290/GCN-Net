"""Feature standardisation, fit on the training split only.

The reference mean-centres over the entire dataset *before* splitting
(``Stack_Dataset - mean(Stack_Dataset, 1)`` in ``make_dataset.m``), which leaks
test statistics into training. Here the statistics are an explicit object: fit it
on train, apply it to train/val/test. There is no code path that fits on data it
was not handed.

:class:`Standardizer` deliberately has no ``fit_transform`` over a whole dataset —
the missing convenience method is the point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Standardizer:
    """Per-channel mean/std, estimated over trials and time of the training split."""

    mean: np.ndarray  # (n_channels, 1)
    std: np.ndarray   # (n_channels, 1)
    n_trials_fit: int

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Standardise ``(n_trials, n_channels, n_samples)`` with the stored stats."""
        if X.ndim != 3:
            raise ValueError(f"expected (n_trials, n_channels, n_samples), got {X.shape}")
        if X.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"channel count {X.shape[1]} does not match the fitted "
                f"{self.mean.shape[0]}"
            )
        return ((X - self.mean) / self.std).astype(np.float32)


def fit_standardizer(X_train: np.ndarray, eps: float = 1e-8,
                     per_feature: bool = False) -> Standardizer:
    """Estimate statistics from the training split only.

    Per-channel rather than global: electrode variance differs by an order of
    magnitude across the scalp, and a single global scale would let high-variance
    channels dominate the correlation structure the graph is built on.

    Args:
        per_feature: reduce over samples only, giving one statistic per
            ``(channel, feature)`` pair rather than per channel. Required for band
            power, where the last axis is frequency bands whose scales differ by
            orders of magnitude — collapsing them would let the low-frequency
            bands set the scale for every band.
    """
    if X_train.ndim != 3:
        raise ValueError(f"expected (n_trials, n_channels, n_samples), got {X_train.shape}")
    if X_train.shape[0] == 0:
        raise ValueError("cannot fit a standardizer on zero trials")

    axes = (0,) if per_feature else (0, 2)
    mean = X_train.mean(axis=axes, keepdims=True)[0]
    std = X_train.std(axis=axes, keepdims=True)[0]

    dead = np.flatnonzero(std.ravel() < eps)
    if dead.size:
        raise ValueError(f"channels {dead.tolist()} have ~zero variance in the training split")

    return Standardizer(mean=mean, std=std, n_trials_fit=int(X_train.shape[0]))


def fit_leaky_standardizer(X_all: np.ndarray) -> Standardizer:
    """Fit over **all** data including test, reproducing the reference.

    Only for the leaky replication arm (project_documentation.md 7.2).
    """
    return fit_standardizer(X_all)
