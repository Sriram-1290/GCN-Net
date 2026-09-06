"""A CSP-derived spatial basis, as a drop-in replacement for the graph basis.

**Why this exists.** 9.13 established that the electrode graph is inert: a hard
Chebyshev constraint, a free operator and a Dirichlet graph prior span 0.0108 and
are mutually indistinguishable. The natural reading was "spatial priors do not
help here". CSP-Net (Jiang et al., arXiv:2411.11879) reports the opposite for a
*different* prior on the *same* operator -- initialising a CNN's spatial layer
from CSP filters gains ~6 points over a ShallowConvNet backbone across four MI
datasets, and their ablation shows it is the initialisation rather than the extra
layer that does the work (a randomly initialised variant barely beats the
backbone).

So the question 9.13 actually left open is not *whether* to constrain the spatial
operator but *with what*. This module answers it by building a basis of the same
shape and scaling as the Chebyshev one, from CSP instead of from the graph.

**The construction mirrors `graphs.laplacian` deliberately.** The Chebyshev basis
rescales the Laplacian to ``L~ = 2L/lambda_max - I`` so its spectrum lies in
[-1, 1], then applies the recurrence ``T0 = I``, ``T1 = L~``,
``Tk = 2 L~ T(k-1) - T(k-2)``. Here the same recurrence runs on a CSP operator
rescaled the same way. Everything downstream -- shapes, weight layout, the
``bmfk`` einsum ordering in :class:`models.free_spatial.FreeSpatialConv` -- is
untouched, so swapping the basis changes exactly one thing.

**Leakage.** CSP is supervised, so it must be fit on training windows only,
exactly as the correlation graph is. The caller passes the fold's training split;
this module never sees test data.
"""

from __future__ import annotations

import numpy as np


def _rescale(P: np.ndarray) -> np.ndarray:
    """Map a symmetric operator's spectrum into [-1, 1].

    Mirrors the ``2L/lambda_max - I`` rescaling the Chebyshev basis applies to the
    Laplacian, so the recurrence below is numerically comparable to the graph one
    and the operator norm at initialisation is of the same order.
    """
    lmax = float(np.abs(np.linalg.eigvalsh(P)).max())
    if lmax <= 0:
        raise ValueError("CSP operator is identically zero; cannot rescale")
    return 2.0 * P / lmax - np.eye(P.shape[0], dtype=P.dtype)


def csp_operator(X: np.ndarray, y: np.ndarray, n_components: int = 8) -> np.ndarray:
    """``(M, M)`` symmetric operator spanning the CSP-discriminative subspace.

    CSP yields filters ``W`` of shape ``(f, M)``; ``W.T @ W`` is the symmetric
    positive-semidefinite operator whose range is the subspace those filters
    select. Using the symmetric outer product rather than the filters themselves
    keeps the node dimension at ``M``, which is what the spatial layer requires --
    the layer mixes across nodes and must return one value per node.

    Args:
        X: ``(N, M, T)`` training windows, in the node ordering the model sees.
        y: ``(N,)`` binary labels for those windows.
        n_components: number of CSP filters to span the subspace with. CSP-Net's
            Figure 7 finds ~8 best for a ShallowConvNet backbone.

    Returns:
        ``(M, M)`` symmetric operator, rescaled to spectrum [-1, 1].
    """
    from mne.decoding import CSP

    if X.ndim != 3:
        raise ValueError(f"expected (N, M, T) windows, got {X.shape}")
    if len(np.unique(y)) != 2:
        raise ValueError(f"CSP needs exactly two classes, got {np.unique(y)}")

    n_nodes = X.shape[1]
    f = int(min(n_components, n_nodes))
    if f % 2:                      # CSP pairs filters from the two spectrum ends
        f -= 1
    if f < 2:
        raise ValueError(f"need at least 2 CSP components, got {f}")

    # Ledoit-Wolf shrinkage because a single 0.40 s window carries fewer samples
    # than channels; the class covariances are averaged over ~600 windows each, so
    # the average is full rank, but the shrinkage keeps individual estimates sane.
    csp = CSP(n_components=f, reg="ledoit_wolf", log=True,
              transform_into="average_power")
    csp.fit(X.astype(np.float64), y)

    W = np.asarray(csp.filters_[:f], dtype=np.float64)     # (f, M)
    P = W.T @ W                                            # (M, M), symmetric PSD
    return _rescale(P)


def csp_basis(X: np.ndarray, y: np.ndarray, order: int = 2,
              n_components: int = 8) -> np.ndarray:
    """``(order + 1, M, M)`` Chebyshev-style basis built on the CSP operator.

    Shape- and scale-compatible with ``graphs.chebyshev.build_bases``' layer-0
    output, so it can be handed to :class:`models.graph_bandpower.GraphBandPowerNet`
    with ``spatial="free"`` and nothing else changes: ``FreeSpatialConv`` copies
    whatever basis it is given into its learnable ``operators``.
    """
    P = csp_operator(X, y, n_components=n_components)
    m = P.shape[0]
    basis = [np.eye(m, dtype=np.float64), P]
    for _ in range(2, order + 1):
        basis.append(2.0 * P @ basis[-1] - basis[-2])
    return np.stack(basis[:order + 1]).astype(np.float32)
