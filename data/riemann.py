"""Riemannian tangent-space helpers (covariances, Frechet mean, tangent projection).

Moved here unchanged from ``scripts/run_tangent_space.py`` so that every tangent-space
arm imports one implementation: the subject-dependent baseline (A3 of 9.11), the
group-transfer runner (``scripts/run_tangent_transfer.py``, 9.26 A1 and A6) and the
confound audit (9.26 A3 and A4). Behaviour, defaults and the shrinkage of 0.1 are
exactly those of the original private helpers, which that script still exposes under
their old names.

**Implemented here rather than with ``pyriemann``** because ``CLAUDE.md`` says not to
add dependencies casually and the whole method is ~40 lines of ``numpy.linalg``.
"""

from __future__ import annotations

import numpy as np


def covariances(X: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    """``(N, C, T) -> (N, C, C)`` shrunk sample covariances.

    Shrinkage toward a scaled identity keeps every matrix positive definite, which
    the matrix logarithm below requires. With T comparable to C the raw sample
    covariance is often singular, so this is not optional. (The common average
    reference also leaves the raw covariance rank C-1; shrinkage restores full rank.)
    """
    n, c, _ = X.shape
    out = np.empty((n, c, c))
    for i in range(n):
        s = np.cov(X[i].astype(np.float64))
        trace = np.trace(s) / c
        out[i] = (1 - shrinkage) * s + shrinkage * trace * np.eye(c)
    return out


def sqrtm_inv(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(C^{1/2}, C^{-1/2})`` of an SPD matrix, eigenvalues floored at 1e-10."""
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-10, None)
    return (V * np.sqrt(w)) @ V.T, (V * (1.0 / np.sqrt(w))) @ V.T


def logm_spd(C: np.ndarray) -> np.ndarray:
    """Matrix logarithm of an SPD matrix, eigenvalues floored at 1e-10."""
    w, V = np.linalg.eigh(C)
    return (V * np.log(np.clip(w, 1e-10, None))) @ V.T


def expm_spd(C: np.ndarray) -> np.ndarray:
    """Matrix exponential of a symmetric matrix."""
    w, V = np.linalg.eigh(C)
    return (V * np.exp(w)) @ V.T


def riemannian_mean(covs: np.ndarray, n_iter: int = 15, tol: float = 1e-6) -> np.ndarray:
    """Frechet mean under the affine-invariant metric, by the standard fixed point.

    Starts from the arithmetic mean and repeatedly steps along the average tangent
    direction. Converges in a handful of iterations for well-conditioned inputs.
    """
    M = covs.mean(axis=0)
    for _ in range(n_iter):
        Ms, Mis = sqrtm_inv(M)
        T = np.mean([logm_spd(Mis @ C @ Mis) for C in covs], axis=0)
        M = Ms @ expm_spd(T) @ Ms
        if np.linalg.norm(T) < tol:
            break
    return M


def tangent_space(covs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Project SPD matrices to the tangent space at ``ref`` and vectorise.

    Off-diagonal entries are scaled by sqrt(2) so the Euclidean norm of the vector
    equals the Frobenius norm of the matrix -- the standard convention, without
    which the diagonal is implicitly down-weighted.
    """
    _, Ris = sqrtm_inv(ref)
    c = ref.shape[0]
    iu = np.triu_indices(c)
    scale = np.full((c, c), np.sqrt(2.0))
    np.fill_diagonal(scale, 1.0)
    return np.stack([(logm_spd(Ris @ C @ Ris) * scale)[iu] for C in covs])
