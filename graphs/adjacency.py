"""Electrode graph construction from Pearson correlation.

The paper (arXiv:2006.08924) defines the electrode adjacency as

    A = |P| - I                                                    (paper, Sec. III)

where ``P`` is the 64x64 Pearson correlation matrix between electrode time
series. Subtracting the identity removes self-loops, since ``|P|`` has a unit
diagonal by construction.

**Deliberate deviation from the paper.** The paper computes ``P`` over "all the
signals of participants" — the entire pooled dataset, test data included. That is
the leakage CLAUDE.md forbids by name: the graph the model convolves over would
carry information about the test trials. :func:`pcc_adjacency` therefore takes
only the data it is given, and callers must pass the training split.
:func:`leaky_pcc_adjacency` reproduces the paper's version for the replication
arm, and is loud about what it is.
"""

from __future__ import annotations

import numpy as np


def _channel_time_matrix(X: np.ndarray) -> np.ndarray:
    """Flatten ``(n_trials, n_channels, n_samples)`` to ``(n_channels, n_trials*n_samples)``.

    Trials are concatenated along time so that every channel pair is correlated
    over the same timepoints. Note this treats trial boundaries as continuous,
    which is what a correlation over "overall signals" means.
    """
    if X.ndim != 3:
        raise ValueError(f"expected (n_trials, n_channels, n_samples), got {X.shape}")
    n_trials, n_channels, n_samples = X.shape
    if n_trials == 0:
        raise ValueError("cannot build an adjacency from zero trials")
    return X.transpose(1, 0, 2).reshape(n_channels, n_trials * n_samples)


def pcc_adjacency(X_train: np.ndarray) -> np.ndarray:
    """Absolute-Pearson adjacency ``A = |P| - I`` from **training data only**.

    Args:
        X_train: ``(n_trials, n_channels, n_samples)`` from the training split.

    Returns:
        ``(n_channels, n_channels)`` float64, symmetric, zero diagonal,
        entries in [0, 1].
    """
    flat = _channel_time_matrix(X_train)

    # A channel that is constant over the whole split has zero variance and would
    # make corrcoef emit NaN. Fail loudly rather than propagate NaN into the
    # Laplacian, where it would silently poison every eigenvalue.
    stds = flat.std(axis=1)
    dead = np.flatnonzero(stds == 0)
    if dead.size:
        raise ValueError(f"channels {dead.tolist()} are constant; cannot correlate")

    P = np.corrcoef(flat)
    if not np.all(np.isfinite(P)):
        raise ValueError("Pearson matrix contains non-finite values")

    A = np.abs(P) - np.eye(P.shape[0])
    A = np.clip(A, 0.0, 1.0)          # guard float error on the diagonal
    A = 0.5 * (A + A.T)               # enforce exact symmetry
    np.fill_diagonal(A, 0.0)
    return A


def leaky_pcc_adjacency(X_all: np.ndarray) -> np.ndarray:
    """The paper's adjacency, built over **all** data including test.

    Only for the leaky replication arm (project_documentation.md 7.2). Using this
    in a reported result would mean the graph carries test-set information.
    """
    return pcc_adjacency(X_all)


def threshold_adjacency(A: np.ndarray, keep_top_k: int | None = None,
                        min_weight: float | None = None) -> np.ndarray:
    """Optionally sparsify the (otherwise fully dense) correlation graph.

    ``A = |P| - I`` is dense: every electrode pair has a non-zero weight, so the
    "graph" carries little topological structure. The paper does not sparsify, so
    the default pipeline does not either — but sparsification is an obvious
    ablation, and Graclus coarsening behaves differently on a dense graph.

    Args:
        keep_top_k: keep each node's k strongest edges (symmetrised by union).
        min_weight: zero out any edge below this weight.
    """
    if keep_top_k is None and min_weight is None:
        return A

    out = A.copy()
    if min_weight is not None:
        out[out < min_weight] = 0.0
    if keep_top_k is not None:
        n = out.shape[0]
        if not 1 <= keep_top_k < n:
            raise ValueError(f"keep_top_k must be in [1, {n - 1}], got {keep_top_k}")
        mask = np.zeros_like(out, dtype=bool)
        # argpartition on -row puts the k largest first
        for i in range(n):
            top = np.argpartition(-out[i], keep_top_k)[:keep_top_k]
            mask[i, top] = True
        mask |= mask.T  # union keeps the result symmetric
        out = np.where(mask, out, 0.0)

    np.fill_diagonal(out, 0.0)
    return out
