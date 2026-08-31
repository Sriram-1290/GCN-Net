"""GCNs-Net: the six-layer Chebyshev GCN from Hou et al. (IEEE TNNLS 2024).

Architecture, from `main-GCN.py`:

    F = [16, 32, 64, 128, 256, 512]   six graph-conv layers
    K = [2] * 6                       Chebyshev order 2
    p = [2] * 6                       six stride-2 max-pools
    M = [4]                           four-class output
    graphs: 64 -> 32 -> 16 -> 8 -> 4 -> 2 nodes, then a final pool to 1

Sample definition
-----------------
The reference treats **one timepoint as one sample**: `_inference` opens with
`tf.expand_dims(x, 2)`, so `Fin = 1`. We keep that, because it makes the
leaky-vs-correct comparison exact — same model, same sample definition, only the
*split* differs. Trial-level predictions come from aggregating the 640 timepoint
predictions of a trial (see `evaluation/`), not from changing the architecture.

`in_features` is nonetheless a parameter, because the temporal-attention
contribution will need a window rather than a single sample.

Two deliberate deviations from the reference, both switchable:

* BatchNorm respects `.eval()` (`always_train_bn=False` by default).
* `rescale_laplacian` may use the true lambda_max instead of a hardcoded 2.

One thing worth knowing: with `M = [4]` the reference's dropout is **never
applied**. `_inference` only drops between `self.M[:-1]` layers, and that list is
empty, so the configured `dropout = 0.50` is dead. We reproduce that by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .cheb_conv import ChebConv, GraphMaxPool, VertexBiasNorm


@dataclass(frozen=True)
class GCNsNetConfig:
    """Architecture hyperparameters. Defaults reproduce `main-GCN.py`."""

    filters: tuple[int, ...] = (16, 32, 64, 128, 256, 512)
    orders: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    pools: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    n_classes: int = 4
    in_features: int = 1
    weight_std: float = 0.1
    always_train_bn: bool = False
    dropout: float = 0.0  # dead in the reference; kept switchable
    fc_hidden: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not (len(self.filters) == len(self.orders) == len(self.pools)):
            raise ValueError("filters, orders and pools must be the same length")


class GCNsNet(nn.Module):
    """Six-layer Chebyshev GCN over a precomputed coarsening hierarchy.

    Args:
        bases: one Chebyshev basis per conv layer, each ``(K, M_i, M_i)``, built
            from the coarsened graphs. ``M_i`` must halve at each level.
        cfg: architecture configuration.
    """

    def __init__(self, bases: list[np.ndarray], cfg: GCNsNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GCNsNetConfig()
        n_layers = len(self.cfg.filters)
        if len(bases) != n_layers:
            raise ValueError(f"got {len(bases)} graph bases for {n_layers} conv layers")

        convs, norms, poolers = [], [], []
        n_in = self.cfg.in_features
        n_nodes = bases[0].shape[1]

        for i, (n_out, K, p) in enumerate(
            zip(self.cfg.filters, self.cfg.orders, self.cfg.pools)
        ):
            basis = bases[i]
            if basis.shape[0] != K:
                raise ValueError(f"layer {i}: basis has order {basis.shape[0]}, config says {K}")
            if basis.shape[1] != n_nodes:
                raise ValueError(
                    f"layer {i}: basis has {basis.shape[1]} nodes, "
                    f"but the previous layer produced {n_nodes}"
                )
            convs.append(ChebConv(n_in, n_out, basis, weight_std=self.cfg.weight_std))
            # Bias is applied before pooling, so it is sized on the pre-pool graph.
            norms.append(VertexBiasNorm(n_nodes, n_out, always_train_bn=self.cfg.always_train_bn))
            poolers.append(GraphMaxPool(p))
            n_in = n_out
            n_nodes //= p

        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.pools = nn.ModuleList(poolers)
        self.n_nodes_out = n_nodes

        flat = n_nodes * self.cfg.filters[-1]
        head: list[nn.Module] = []
        width = flat
        for hidden in self.cfg.fc_hidden:
            head += [nn.Linear(width, hidden), nn.BatchNorm1d(hidden), nn.Softplus()]
            if self.cfg.dropout > 0:
                head.append(nn.Dropout(self.cfg.dropout))
            width = hidden
        self.head = nn.Sequential(*head)

        # The reference's logits layer returns the raw matmul (relu=False returns x
        # before BN and softplus), so this stays a plain Linear.
        self.logits = nn.Linear(width, self.cfg.n_classes)
        nn.init.trunc_normal_(self.logits.weight, mean=0.0, std=self.cfg.weight_std,
                              a=-2 * self.cfg.weight_std, b=2 * self.cfg.weight_std)
        nn.init.constant_(self.logits.bias, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, M, Fin) -> (N, n_classes)``.

        A 2-D input ``(N, M)`` is treated as ``Fin = 1``, matching the reference's
        ``tf.expand_dims(x, 2)``.
        """
        if x.dim() == 2:
            x = x.unsqueeze(2)
        for conv, norm, pool in zip(self.convs, self.norms, self.pools):
            x = pool(norm(conv(x)))
        x = x.flatten(1)
        return self.logits(self.head(x))

    def l2_penalty(self) -> torch.Tensor:
        """L2 over the fully-connected parameters only.

        Matches the reference, where `_weight_variable(..., regularization=True)`
        is passed only in `fc`; conv weights and vertex biases are excluded. The
        reference accumulates `tf.nn.l2_loss`, which is ``sum(v**2) / 2``.
        """
        total = torch.zeros((), device=self.logits.weight.device)
        for module in list(self.head) + [self.logits]:
            if isinstance(module, nn.Linear):
                total = total + 0.5 * (module.weight ** 2).sum()
                if module.bias is not None:
                    total = total + 0.5 * (module.bias ** 2).sum()
        return total

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_bases(graphs, orders: tuple[int, ...], lmax: float | None = 2.0) -> list[np.ndarray]:
    """Chebyshev bases for each coarsened graph.

    Args:
        graphs: output of :func:`graphs.coarsening.coarsen`, one per conv layer.
        orders: Chebyshev order per layer.
        lmax: ``2.0`` reproduces the reference; ``None`` uses the true lambda_max
            per layer, which uses the full [-1, 1] Chebyshev domain (see
            project_documentation.md 6.5/6.6).
    """
    from graphs.laplacian import chebyshev_basis, laplacian, rescale_laplacian

    bases = []
    for G, K in zip(graphs, orders):
        A = np.asarray(G.todense(), dtype=np.float64)
        L = laplacian(A, normalized=True)
        bases.append(chebyshev_basis(rescale_laplacian(L, lmax=lmax), K).astype(np.float32))
    return bases
