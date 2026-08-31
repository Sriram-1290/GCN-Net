"""A learnable electrode graph, initialised at the fixed PCC adjacency.

Motivation is the Phase 3a failure (project_documentation.md 9.6). Giving the
network band power explicitly changed nothing on the confound-free comparisons,
so the gap to CSP+LDA is most likely *spatial*: CSP fits a discriminative spatial
filter using the labels, while our network convolves over a correlation graph
built without ever seeing them. This module lets the graph be fit to the task.

Design choices that matter:

* **Initialised to zero delta**, so at step 0 the model is *bit-for-bit* the fixed
  PCC baseline. Any difference in the ablation is therefore attributable to
  learning the graph and nothing else — not a different initialisation, not a
  different parameter count at the start.
* **The coarsening hierarchy stays fixed.** Graclus is discrete and
  non-differentiable, and recomputing the pooling tree every step would change
  which electrodes are siblings mid-training. Only the *edge weights* used by the
  Chebyshev filters are learned; pooling still follows the PCC hierarchy.
* **Symmetry and non-negativity are enforced by construction**, not by a penalty.
  A normalised Laplacian is only well defined for a symmetric non-negative
  adjacency, so a soft constraint would allow invalid graphs during training.
* **Cheap.** 64+32+16+8+4+2 nodes gives 5,460 extra parameters against the
  model's 359,428 — about 1.5%. An improvement cannot be dismissed as extra
  capacity.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .cheb_conv import ChebConv


class LearnableAdjacency(nn.Module):
    """Adjacency parameterised as ``relu(A_pcc + delta)``, symmetric, zero diagonal.

    Args:
        base: ``(M, M)`` fixed PCC adjacency for this layer, used as the
            initialisation point and kept as a buffer for comparison.
        learn_scale: also learn a scalar on the base, letting the model damp or
            amplify the prior graph as a whole rather than edge by edge.
    """

    def __init__(self, base: np.ndarray, learn_scale: bool = False) -> None:
        super().__init__()
        if base.ndim != 2 or base.shape[0] != base.shape[1]:
            raise ValueError(f"base adjacency must be square, got {base.shape}")
        self.n_nodes = base.shape[0]
        self.register_buffer("base", torch.as_tensor(base, dtype=torch.float32))
        # Zero init: the graph starts exactly at the PCC prior.
        self.delta = nn.Parameter(torch.zeros(self.n_nodes, self.n_nodes))
        self.scale = nn.Parameter(torch.ones(())) if learn_scale else None

    def forward(self) -> torch.Tensor:
        base = self.base if self.scale is None else self.scale * self.base
        a = base + self.delta
        a = 0.5 * (a + a.t())                       # symmetric by construction
        a = torch.relu(a)                           # non-negative by construction
        return a - torch.diag(torch.diagonal(a))    # no self-loops

    def delta_l1(self) -> torch.Tensor:
        """L1 of the learned correction, for optional sparsity regularisation.

        Encourages the model to change few edges, which keeps the learned graph
        interpretable against the PCC prior rather than becoming an arbitrary
        dense matrix.
        """
        return self.delta.abs().sum()

    def extra_repr(self) -> str:
        return f"n_nodes={self.n_nodes}, learn_scale={self.scale is not None}"


def torch_laplacian(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric normalised Laplacian ``I - D^-1/2 A D^-1/2``, differentiable.

    Mirrors :func:`graphs.laplacian.laplacian`. The epsilon guards a node whose
    edges have all been driven to zero by the relu, which would otherwise divide
    by zero and put NaN through every downstream eigenvalue.
    """
    d = A.sum(dim=0) + eps
    d_inv_sqrt = d.pow(-0.5)
    n = A.shape[0]
    eye = torch.eye(n, device=A.device, dtype=A.dtype)
    L = eye - d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)
    return 0.5 * (L + L.t())


def torch_rescale_laplacian(L: torch.Tensor, lmax: float = 2.0) -> torch.Tensor:
    """Map the spectrum into [-1, 1]. Fixed ``lmax`` keeps this differentiable.

    The true lambda_max would need an eigendecomposition every forward pass;
    ``lmax = 2`` is a valid upper bound for a normalised Laplacian and is what the
    reference uses, so the baseline and the learnable variant stay comparable.
    """
    eye = torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
    return L / (lmax / 2.0) - eye


def torch_chebyshev_basis(L_tilde: torch.Tensor, K: int) -> torch.Tensor:
    """``T_0..T_{K-1}`` of a rescaled Laplacian, as a differentiable ``(K, M, M)``."""
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    n = L_tilde.shape[0]
    terms = [torch.eye(n, device=L_tilde.device, dtype=L_tilde.dtype)]
    if K > 1:
        terms.append(L_tilde)
    for _ in range(2, K):
        terms.append(2.0 * L_tilde @ terms[-1] - terms[-2])
    return torch.stack(terms, dim=0)


class DynamicChebConv(ChebConv):
    """Chebyshev convolution whose graph is learned rather than precomputed.

    Subclasses :class:`ChebConv` so the weight layout, initialisation and shape
    checks are shared with the verified baseline; only the basis changes from a
    fixed buffer to something recomputed each forward pass.
    """

    def __init__(self, in_features: int, out_features: int, base: np.ndarray, K: int,
                 weight_std: float = 0.1, learn_scale: bool = False,
                 lmax: float = 2.0) -> None:
        # Build the fixed basis first so the parent sets up weights and shapes,
        # then replace it with the learnable graph.
        from graphs.laplacian import chebyshev_basis, laplacian, rescale_laplacian
        fixed = chebyshev_basis(rescale_laplacian(laplacian(base), lmax=lmax), K)
        super().__init__(in_features, out_features, fixed.astype(np.float32),
                         weight_std=weight_std)
        self.adjacency = LearnableAdjacency(base, learn_scale=learn_scale)
        self.lmax = lmax

    def current_basis(self) -> torch.Tensor:
        A = self.adjacency()
        return torch_chebyshev_basis(
            torch_rescale_laplacian(torch_laplacian(A), self.lmax), self.K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (batch, nodes, features), got {tuple(x.shape)}")
        n_batch, n_nodes, n_feat = x.shape
        if n_nodes != self.n_nodes:
            raise ValueError(f"graph has {self.n_nodes} nodes but input has {n_nodes}")
        if n_feat != self.in_features:
            raise ValueError(f"expected {self.in_features} input features, got {n_feat}")

        tx = torch.einsum("kmn,bnf->bmfk", self.current_basis(), x)
        tx = tx.reshape(n_batch * n_nodes, self.in_features * self.K)
        return (tx @ self.weight).view(n_batch, n_nodes, self.out_features)
