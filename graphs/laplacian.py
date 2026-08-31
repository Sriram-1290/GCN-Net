"""Graph Laplacian and Chebyshev rescaling.

Follows Defferrard's ``cnn_graph`` (ChebNet, NIPS 2016), which the reference
implementation forks verbatim. Dense NumPy throughout: the electrode graph has 64
nodes, so sparse machinery would cost more than it saves and would obscure code
that has to be defended in a viva.

Symmetric normalised Laplacian:

    L = I - D^(-1/2) A D^(-1/2)

Its eigenvalues lie in [0, 2] for any non-negative symmetric ``A``, which is why
the reference can rescale with a hardcoded ``lmax = 2`` instead of computing the
true largest eigenvalue.
"""

from __future__ import annotations

import numpy as np


def _check_adjacency(A: np.ndarray) -> None:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"adjacency must be square, got {A.shape}")
    if not np.allclose(A, A.T, atol=1e-10):
        raise ValueError("adjacency is not symmetric")
    if np.any(A < 0):
        raise ValueError("adjacency has negative weights; normalisation assumes A >= 0")
    if np.any(np.diag(A) != 0):
        raise ValueError("adjacency has self-loops; expected a zero diagonal")


def laplacian(A: np.ndarray, normalized: bool = True) -> np.ndarray:
    """Graph Laplacian of a dense adjacency matrix.

    Args:
        A: ``(n, n)`` symmetric, non-negative, zero diagonal.
        normalized: symmetric normalisation ``I - D^-1/2 A D^-1/2`` if True,
            combinatorial ``D - A`` otherwise.
    """
    _check_adjacency(A)
    d = A.sum(axis=0)

    if not normalized:
        return np.diag(d) - A

    # Defferrard adds machine epsilon before the reciprocal so an isolated node
    # (degree 0) yields a finite row rather than inf. Keep the same guard.
    d = d + np.spacing(np.array(0, dtype=A.dtype))
    d_inv_sqrt = 1.0 / np.sqrt(d)
    D = np.diag(d_inv_sqrt)
    L = np.eye(A.shape[0], dtype=A.dtype) - D @ A @ D
    return 0.5 * (L + L.T)  # kill asymmetry from float error


def largest_eigenvalue(L: np.ndarray) -> float:
    """True lambda_max of a symmetric Laplacian."""
    return float(np.linalg.eigvalsh(L)[-1])


def rescale_laplacian(L: np.ndarray, lmax: float | None = 2.0) -> np.ndarray:
    """Map the Laplacian spectrum into [-1, 1] for the Chebyshev expansion.

        L_tilde = L / (lmax / 2) - I

    Args:
        lmax: ``2.0`` reproduces the reference exactly (a valid upper bound for a
            normalised Laplacian, though loose when the true lambda_max is
            smaller). Pass ``None`` to use the computed lambda_max instead, which
            uses the full [-1, 1] range and is the ablation worth running.
    """
    if lmax is None:
        lmax = largest_eigenvalue(L)
    if lmax <= 0:
        raise ValueError(f"lmax must be positive, got {lmax}")
    return L / (lmax / 2.0) - np.eye(L.shape[0], dtype=L.dtype)


def chebyshev_basis(L_tilde: np.ndarray, K: int) -> np.ndarray:
    """Chebyshev polynomials ``T_0..T_{K-1}`` of a rescaled Laplacian.

    ``T_0 = I``, ``T_1 = L_tilde``, ``T_k = 2 L_tilde T_{k-1} - T_{k-2}``.

    Precomputing these once is the whole reason the dense formulation is cheap:
    the graph is fixed for a given split, so the forward pass reduces to a stack
    of matmuls against a constant ``(K, n, n)`` tensor.

    Returns:
        ``(K, n, n)``.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    n = L_tilde.shape[0]
    T = np.empty((K, n, n), dtype=L_tilde.dtype)
    T[0] = np.eye(n, dtype=L_tilde.dtype)
    if K > 1:
        T[1] = L_tilde
    for k in range(2, K):
        T[k] = 2.0 * L_tilde @ T[k - 1] - T[k - 2]
    return T
