"""Euclidean alignment (EA) with a rank-aware inverse square root.

EA (He & Wu, 2020) re-centres each subject's trials so their mean covariance
becomes the identity: with ``R = mean_i X_i X_i^T / T`` the aligned trial is
``R^{-1/2} X_i``. That removes most of the between-subject covariance shift
before a model sees the data.

Why not a plain inverse square root
-----------------------------------
The cache is common-average-referenced, so the 64 channels sum to zero at every
sample and ``R`` has rank 63 (research report ``docs/RESEARCH_DIRECTIONS.md``
section 9.4). The null direction has eigenvalue 0 in exact arithmetic but only
rounding noise in floating point, and ``lambda^{-1/2}`` turns that noise into a
huge gain: in a synthetic float32 test a plain inverse amplified the null
direction to 3.2 times the signal power. No published EA work covers this case.

The remedy is a pseudo-inverse square root: eigen-decompose the (symmetrised)
reference, keep only eigenvalues above ``rtol`` times the largest, and invert
those. The result whitens *within* the rank-r subspace (aligned covariance has
r eigenvalues equal to 1) and sends the null direction to zero rather than
amplifying it. All channels are kept; ``W`` is simply a rank-r projector-scaled
matrix. The fallback, if this ever proves inadequate, is Ledoit-Wolf shrinkage
as in MOABB.

Leakage: ``fit_ea`` only sees the trials it is handed. Fit it on training
trials (or, for the pseudo-online variant, on the subject's calibration run);
there is no code path that estimates a reference from other data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _check_trials(X: np.ndarray) -> None:
    if X.ndim != 3:
        raise ValueError(f"expected (n_trials, n_channels, n_samples), got shape {X.shape}")
    if X.shape[0] == 0:
        raise ValueError("need at least one trial")


def trial_covariances(X: np.ndarray) -> np.ndarray:
    """Per-trial covariance ``X X^T / T`` of ``(n, C, T)`` data, float64, ``(n, C, C)``.

    No mean subtraction: the data is band-passed and referenced, so it is already
    (near) zero-mean, and EA is defined on the plain second-moment matrix.
    """
    _check_trials(X)
    X64 = np.asarray(X, dtype=np.float64)
    return np.einsum("nct,ndt->ncd", X64, X64) / X64.shape[2]


def euclidean_reference(X: np.ndarray) -> np.ndarray:
    """Arithmetic mean of the trial covariances, ``(C, C)`` float64."""
    return trial_covariances(X).mean(axis=0)


def inv_sqrt_psd(R: np.ndarray, rtol: float = 1e-8) -> np.ndarray:
    """Pseudo-inverse square root of a symmetric PSD matrix.

    Eigenvalues not above ``rtol * max`` are treated as zero (the null space of a
    rank-deficient covariance) and dropped, so they are not amplified.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {R.shape}")
    vals, vecs = np.linalg.eigh((R + R.T) / 2.0)
    keep = vals > rtol * vals.max()
    scale = np.zeros_like(vals)
    scale[keep] = vals[keep] ** -0.5
    return (vecs * scale) @ vecs.T


def align(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply ``W @ X_i`` to every trial; float32, same shape as ``X``."""
    _check_trials(X)
    n_ch = X.shape[1]
    if W.shape != (n_ch, n_ch):
        raise ValueError(f"W has shape {W.shape}, expected ({n_ch}, {n_ch})")
    out = np.einsum("cd,ndt->nct", np.asarray(W, dtype=np.float64), np.asarray(X, dtype=np.float64))
    return out.astype(np.float32)


@dataclass(frozen=True)
class EAReference:
    """A fitted alignment: the whitening matrix, its rank and the trials behind it."""

    W: np.ndarray  # (C, C) pseudo-inverse square root of the reference
    rank: int      # number of eigenvalues kept
    n_trials: int  # trials the reference was estimated from


def fit_ea(X_train: np.ndarray, rtol: float = 1e-8) -> EAReference:
    """Estimate the reference from ``X_train`` only and return its whitening matrix."""
    R = euclidean_reference(X_train)
    W = inv_sqrt_psd(R, rtol)
    vals = np.linalg.eigvalsh((R + R.T) / 2.0)
    rank = int(np.sum(vals > rtol * vals.max()))
    return EAReference(W=W, rank=rank, n_trials=int(X_train.shape[0]))
