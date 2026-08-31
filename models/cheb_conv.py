"""Dense Chebyshev graph convolution — the PyTorch equivalent of ``chebyshev5``.

The reference builds a sparse TF tensor from the Laplacian inside the filter and
runs ``sparse_tensor_dense_matmul`` K times per forward pass. With 64 nodes that
is slower than a dense matmul and much harder to read, so here the Chebyshev
basis ``T_0..T_{K-1}`` is precomputed once (the graph is fixed for a given split)
and registered as a buffer. The arithmetic is identical.

Weight layout matches the reference exactly. ``chebyshev5`` reshapes its stacked
basis to ``(N, M, Fin, K)`` and then to ``(N*M, Fin*K)``, so the weight rows are
ordered with ``K`` fastest: row index ``fin * K + k``. This module produces the
same ordering, so a weight matrix could in principle be transferred between the
two implementations.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ChebConv(nn.Module):
    """Order-K Chebyshev spectral convolution on a fixed graph.

    Args:
        in_features: ``Fin`` — features per node on input.
        out_features: ``Fout``.
        basis: ``(K, M, M)`` Chebyshev polynomials of the rescaled Laplacian,
            from :func:`graphs.laplacian.chebyshev_basis`.
        weight_std: standard deviation of the truncated-normal initialiser. The
            reference uses a fixed 0.1 rather than a fan-in-scaled value; kept
            as the default so the baseline matches, but overridable because a
            fixed std is a questionable choice at ``Fin*K = 512``.
    """

    def __init__(self, in_features: int, out_features: int, basis: np.ndarray,
                 weight_std: float = 0.1) -> None:
        super().__init__()
        if basis.ndim != 3 or basis.shape[1] != basis.shape[2]:
            raise ValueError(f"basis must be (K, M, M), got {basis.shape}")

        self.in_features = in_features
        self.out_features = out_features
        self.K, self.n_nodes, _ = basis.shape

        self.register_buffer("basis", torch.as_tensor(basis, dtype=torch.float32))
        self.weight = nn.Parameter(torch.empty(in_features * self.K, out_features))

        # tf.truncated_normal_initializer(0, 0.1): normal, resampled beyond 2 sigma.
        nn.init.trunc_normal_(self.weight, mean=0.0, std=weight_std,
                              a=-2 * weight_std, b=2 * weight_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, M, Fin) -> (N, M, Fout)``."""
        if x.dim() != 3:
            raise ValueError(f"expected (batch, nodes, features), got {tuple(x.shape)}")
        n_batch, n_nodes, n_feat = x.shape
        if n_nodes != self.n_nodes:
            raise ValueError(f"graph has {self.n_nodes} nodes but input has {n_nodes}")
        if n_feat != self.in_features:
            raise ValueError(f"expected {self.in_features} input features, got {n_feat}")

        # T_k @ x for every k, giving (N, M, Fin, K) with K last -- matching the
        # reference's reshape order so the weight rows line up.
        tx = torch.einsum("kmn,bnf->bmfk", self.basis, x)
        tx = tx.reshape(n_batch * n_nodes, self.in_features * self.K)
        out = tx @ self.weight
        return out.view(n_batch, n_nodes, self.out_features)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"K={self.K}, nodes={self.n_nodes}")


class VertexBiasNorm(nn.Module):
    """The reference's ``b2relu``: per-vertex-per-filter bias, BatchNorm, softplus.

    ``main-GCN.py`` configures ``b2relu``, not ``b1relu``, so the bias has shape
    ``[1, M, F]`` — one per node per filter — rather than ``[1, 1, F]``. This
    accounts for most of the model's parameters in the early layers.

    The one deliberate deviation: the reference hardcodes
    ``tf.layers.batch_normalization(..., training=True)``, so at test time it
    normalises with test-batch statistics. Here BatchNorm respects ``.eval()``.
    Set ``always_train_bn=True`` to reproduce the leaky behaviour for the
    replication arm.
    """

    def __init__(self, n_nodes: int, n_features: int, always_train_bn: bool = False) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.full((1, n_nodes, n_features), 0.1))
        self.bn = nn.BatchNorm1d(n_features)
        self.always_train_bn = always_train_bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.bias
        # tf.layers.batch_normalization on (N, M, F) normalises the last axis over
        # (N, M). PyTorch BatchNorm1d(F) wants (N, F, M), hence the transposes.
        x = x.transpose(1, 2)
        if self.always_train_bn and not self.bn.training:
            self.bn.train()
            x = self.bn(x)
            self.bn.eval()
        else:
            x = self.bn(x)
        x = x.transpose(1, 2)
        return nn.functional.softplus(x)


class GraphMaxPool(nn.Module):
    """Stride-p max pooling over the node axis (the reference's ``mpool1``).

    Valid only because :mod:`graphs.coarsening` has reordered nodes so that
    ``2i`` and ``2i+1`` are siblings in the coarsening hierarchy.
    """

    def __init__(self, p: int) -> None:
        super().__init__()
        if p < 1:
            raise ValueError(f"pool size must be >= 1, got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 1:
            return x
        n_nodes = x.shape[1]
        if n_nodes % self.p != 0:
            raise ValueError(f"{n_nodes} nodes is not divisible by pool size {self.p}")
        pooled = nn.functional.max_pool1d(x.transpose(1, 2), kernel_size=self.p, stride=self.p)
        return pooled.transpose(1, 2)

    def extra_repr(self) -> str:
        return f"p={self.p}"
