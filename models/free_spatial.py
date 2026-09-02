"""An unconstrained spatial operator, initialised at the Chebyshev basis.

This is the control for the claim in ``project_documentation.md`` 9.7.1, which is
currently an *inference* from two null results rather than a measurement. That
section argues the bottleneck is not *which* graph the network convolves over but
the graph-convolutional inductive bias itself:

    CSP fits an unconstrained discriminative spatial projection: any linear
    combination of all 64 channels, chosen to maximise the between-class variance
    ratio. A Chebyshev graph convolution cannot express that. Its spatial mixing
    is restricted to graph-local neighbourhoods with weights shared across nodes.
    Learning the adjacency changes *which* nodes are neighbours; it does not lift
    the locality or the weight sharing.

The three arms now available differ in exactly one thing — how constrained the
spatial operator is:

===========================  ==========================================
arm                          spatial operators ``S_k``
===========================  ==========================================
``ChebConv`` (baseline)      fixed ``T_k(L̃)``: polynomial in the PCC
                             Laplacian, symmetric, 1-hop at K=2
``DynamicChebConv`` (9.7)    polynomial in a *learned* Laplacian; still
                             symmetric, non-negative, graph-local
``FreeSpatialConv`` (here)   arbitrary dense ``(M, M)``; no locality, no
                             symmetry, no polynomial structure
===========================  ==========================================

If the free operator closes the gap to CSP, 9.7.1 is demonstrated rather than
argued. If it does not, 9.7.1 is wrong and the limitation lies elsewhere — most
likely sample size (17.1). Either outcome resolves something, which is why this
is the first experiment on the roadmap.

Design choices, following :mod:`models.dynamic_adjacency`:

* **Initialised at the Chebyshev basis**, so at step 0 the model is *bit-for-bit*
  the fixed-graph baseline. Any difference is attributable to lifting the
  constraint and nothing else — not a different initialisation, not a different
  parameter count at the start.
* **The coarsening hierarchy stays fixed.** Only the spatial operators inside the
  filters are freed; pooling still follows the PCC Graclus tree, exactly as in
  the learnable-adjacency arm. That keeps this a single-variable change against
  both the baseline and 9.7.
* **Cheap.** ``K * sum(M_i^2)`` over the 64/32/16/8/4/2 hierarchy is 10,920
  parameters against the model's ~359k — about 3%. A gain cannot be dismissed as
  extra capacity, and the arm can be re-run with ``weight_std`` unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class FreeSpatialConv(nn.Module):
    """``ChebConv`` with the fixed basis replaced by free dense operators.

    Drop-in for :class:`models.cheb_conv.ChebConv`: identical constructor
    signature, identical forward contract, identical weight layout. The only
    difference is that ``basis`` becomes a :class:`torch.nn.Parameter` rather
    than a buffer, so the ``K`` spatial mixing matrices are learned without any
    graph-structural constraint.

    Args:
        in_features: ``Fin`` — features per node on input.
        out_features: ``Fout``.
        basis: ``(K, M, M)`` Chebyshev polynomials of the rescaled Laplacian,
            used **only** as the initialisation point.
        weight_std: standard deviation of the truncated-normal initialiser for
            the channel-mixing weights, matching ``ChebConv``.
    """

    def __init__(self, in_features: int, out_features: int, basis: np.ndarray,
                 weight_std: float = 0.1) -> None:
        super().__init__()
        if basis.ndim != 3 or basis.shape[1] != basis.shape[2]:
            raise ValueError(f"basis must be (K, M, M), got {basis.shape}")

        self.in_features = in_features
        self.out_features = out_features
        self.K, self.n_nodes, _ = basis.shape

        # The one change from ChebConv: a Parameter, not a buffer. Initialised at
        # the Chebyshev basis so step 0 is bit-identical to the baseline.
        self.operators = nn.Parameter(torch.as_tensor(basis, dtype=torch.float32).clone())
        # Kept for the diagnostic in `drift_from_init`; not trained.
        self.register_buffer("basis_init",
                             torch.as_tensor(basis, dtype=torch.float32).clone())

        self.weight = nn.Parameter(torch.empty(in_features * self.K, out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=weight_std,
                              a=-2 * weight_std, b=2 * weight_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, M, Fin) -> (N, M, Fout)``.

        Identical algebra to ``ChebConv.forward``; only ``self.operators`` is
        learnable where ``self.basis`` was fixed. The ``bmfk`` einsum ordering is
        preserved so the weight rows still line up with the reference's
        ``fin * K + k`` layout.
        """
        if x.dim() != 3:
            raise ValueError(f"expected (batch, nodes, features), got {tuple(x.shape)}")
        n_batch, n_nodes, n_feat = x.shape
        if n_nodes != self.n_nodes:
            raise ValueError(f"graph has {self.n_nodes} nodes but input has {n_nodes}")
        if n_feat != self.in_features:
            raise ValueError(f"expected {self.in_features} input features, got {n_feat}")

        tx = torch.einsum("kmn,bnf->bmfk", self.operators, x)
        tx = tx.reshape(n_batch * n_nodes, self.in_features * self.K)
        out = tx @ self.weight
        return out.view(n_batch, n_nodes, self.out_features)

    @torch.no_grad()
    def drift_from_init(self) -> dict[str, float]:
        """How far the operators moved from the Chebyshev basis.

        The learnable-adjacency arm (9.7) was a null result, and the checkpoints
        had to be inspected to establish that the mechanism had *engaged* — a
        null because nothing trained is a different finding from a null because
        training does not help. This returns the same diagnostic without needing
        a separate checkpoint pass.
        """
        delta = self.operators - self.basis_init
        denom = self.basis_init.abs().mean().clamp_min(1e-12)
        return {
            "l2": float(delta.norm()),
            "max_abs": float(delta.abs().max()),
            "frac_changed_1pct": float((delta.abs() > 0.01 * denom).float().mean()),
        }

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"K={self.K}, nodes={self.n_nodes}, free=True")
