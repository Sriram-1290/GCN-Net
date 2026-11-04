"""A graph spatial stage inside a band-power architecture (17.3 / 9.11.2 follow-up).

**What this is, and whose idea the shape is.** The macro-architecture — temporal
filter, spatial filter, square, average, log — is Schirrmeister et al.'s
ShallowConvNet [15], which was itself designed as a differentiable FBCSP. That
ordering is not ours and is not novel. What is ours is putting a *graph-constrained*
spatial stage where ShallowConvNet has a free one, and asking whether the electrode
graph earns its place when the rest of the pipeline is built correctly.

**Why the ordering matters, and why 9.6 did not test it.** CSP works by linearly
combining channels and *then* taking the variance. The combination works through
cancellation between channels, which needs phase-coherent voltage: square first and
the sign is gone, so no later linear combination can perform that cancellation.

9.6 supplied band power as node features — computed per channel by FFT, *before* the
network — and found nothing. That is power-then-spatial. CSP, FBCSP and
ShallowConvNet are all spatial-then-power. **The null result in 9.6 therefore says
nothing about this architecture**, and the ordering it tested is the one that cannot
work in principle.

Four changes from :class:`GCNsNet`, each of which was a separate finding:

1. **Power is computed after the spatial stage** (``square -> mean over time -> log``)
   rather than never. ``GCNsNet`` applies softplus to voltage; motor imagery is a
   variance phenomenon (ERD/ERS) [13].
2. **A learned temporal filter comes first**, replacing reliance on the fixed 8-30 Hz
   band-pass. Temporal attention was the only repair of three that worked (+0.0600,
   9.11), and it is a temporal operation — a learned band-pass is the stronger form
   of that same insight. It also keeps the time axis alive until power is taken,
   where attention pooled it away before the first convolution.
3. **Depth is 1 by default, not 6.** ShallowConvNet has one spatial operation. Six
   Chebyshev layers over a near-complete 64-node graph is a textbook over-smoothing
   configuration, and nothing has ever shown the depth helps here.
4. **No Graclus node pooling.** Collapsing 64 nodes to 2 by local max over a
   correlation hierarchy is close to the opposite of the global weighted contrast CSP
   fits. ``pool_nodes=True`` restores it so the choice stays an ablation rather than
   an assumption.

**The regulariser variant, which is the actual contribution on offer.** A1 measured
the graph *constraint* as recovering only 16 % of the gap — it bounds the hypothesis
class and mostly costs. ``spatial="free"`` with ``graph_smoothness > 0`` instead uses
the graph as a **prior**: the spatial operator is unconstrained, and a Dirichlet-energy
penalty over the electrode graph pulls each node's mixing pattern toward its
neighbours'. With ~45 trials per problem, a spatial prior that *regularises* is
plausible exactly where one that *constrains* is not. Testing it against
``graph_smoothness=0`` isolates the prior from the extra parameters.

Expected outcome, recorded before running, since 9.7.1 is a lesson about not doing
that: this should land near ShallowConvNet (0.6725) if changes 1-4 are what matter,
and the graph terms should make little difference either way. Reaching ShallowConvNet
with a graph-informed model would be a result; exceeding it would be surprising, since
0.6725 already exceeds the best of a published 216,714-configuration search (17.0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .cheb_conv import ChebConv
from .free_spatial import FreeSpatialConv


@dataclass(frozen=True)
class GraphBandPowerConfig:
    """Architecture hyperparameters.

    Defaults follow ShallowConvNet's filter counts so the comparison is like for
    like, with the temporal kernel derived from the window rather than hardcoded —
    our 0.40 s window is far outside the 2-4 s the published values assume.
    """

    n_classes: int = 2
    n_temporal: int = 40          # ShallowConvNet's 40 temporal filters
    n_spatial: int = 40
    temporal_kernel: int | None = None   # None -> min(25, T // 4), as in models.eegnet
    spatial: str = "graph"        # "graph" (Chebyshev) or "free" (unconstrained)
    order: int = 2                # Chebyshev order, unused when spatial="free"
    n_layers: int = 1             # change 3: one spatial operation, not six
    pool_nodes: bool = False      # change 4: no Graclus pooling
    graph_smoothness: float = 0.0  # the regulariser variant; 0 disables it
    dropout: float = 0.5
    weight_std: float = 0.1
    eps: float = 1e-6             # guards the log, as in ShallowConvNet
    n_domains: int = 0            # >0 swaps BatchNorm for per-domain statistics
                                  # (models.dsbn); the last slot is the target
    readout: str = "flat"         # "flat": Linear(M * F), welded to one montage;
                                  # "region": mean log-power per anatomical region
    n_regions: int = 9            # 3 columns x 3 rows, see data.montage.REGION_NAMES

    def __post_init__(self) -> None:
        if self.spatial not in ("graph", "free"):
            raise ValueError(f"spatial must be 'graph' or 'free', got {self.spatial!r}")
        if self.n_domains < 0:
            raise ValueError(f"n_domains must be >= 0, got {self.n_domains}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if self.graph_smoothness < 0:
            raise ValueError("graph_smoothness must be >= 0")
        if self.readout not in ("flat", "region"):
            raise ValueError(f"readout must be 'flat' or 'region', got {self.readout!r}")
        if self.readout == "region":
            if self.spatial != "graph" or self.n_layers != 1 or self.pool_nodes:
                raise ValueError(
                    "readout='region' needs spatial='graph', n_layers=1 and "
                    "pool_nodes=False: the region map is defined on the input nodes, "
                    "so deepening or coarsening the node axis would make it "
                    "meaningless, and a 'free' operator is welded to one node count."
                )
            if self.n_regions < 1:
                raise ValueError(f"n_regions must be >= 1, got {self.n_regions}")
        if self.graph_smoothness and self.spatial == "graph":
            raise ValueError(
                "graph_smoothness applies the graph as a *prior* on an unconstrained "
                "operator; with spatial='graph' the graph is already a hard constraint "
                "and the two would be confounded. Use spatial='free'."
            )


class GraphBandPowerNet(nn.Module):
    """``(N, M, T) -> (N, n_classes)`` via temporal filter, graph spatial, band power.

    Args:
        bases: Chebyshev bases from :func:`models.gcns_net.build_bases`. Only the
            first is used when ``n_layers=1`` and ``pool_nodes=False``; the deeper
            ones are needed only if pooling is switched back on.
        laplacian: ``(M, M)`` **combinatorial** Laplacian ``D - A`` of the layer-0
            graph, used solely by the smoothness penalty. Combinatorial rather than
            normalised because the penalty is exactly
            ``0.5 * sum_ij A_ij (v_i - v_j)^2`` in that form — "an electrode's weights
            should resemble its neighbours'" — whose null space is the constant
            vector. The normalised Laplacian's null vector is ``D^(1/2) 1``, so a
            uniform spatial pattern would be penalised, which is not the prior we
            mean. Required when ``graph_smoothness > 0``.
        n_samples: ``T``, needed to size the temporal kernel and validate input.
        node_regions: ``(M,)`` integer region id of each node, in the same order as
            the basis. Required when ``cfg.readout == "region"``, refused otherwise.

    **Why a region readout.** The Chebyshev convolution shares its weights across
    nodes and the BatchNorm normalises over spatial filters, so both already work
    for any node count; only ``Linear(M * F)`` ties the model to 64 channels. The
    region readout replaces it with the mean log band-power inside each anatomical
    region, so the classifier sees ``n_regions * F`` features whatever montage is
    present, and :meth:`set_graph` can then swap in a subset's basis at test time.
    Pooling is per region, not over all nodes, because left-vs-right imagery is a
    location signal: a global mean would discard exactly the contrast to be decoded.
    """

    def __init__(self, bases: list[np.ndarray], n_samples: int,
                 cfg: GraphBandPowerConfig | None = None,
                 laplacian: np.ndarray | None = None,
                 node_regions: np.ndarray | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GraphBandPowerConfig()
        if (node_regions is not None) != (self.cfg.readout == "region"):
            raise ValueError(
                "node_regions is required with readout='region' and meaningless "
                f"with readout={self.cfg.readout!r}"
            )
        if len(bases) < self.cfg.n_layers:
            raise ValueError(
                f"need {self.cfg.n_layers} graph bases, got {len(bases)}"
            )
        self.n_nodes = bases[0].shape[1]
        self.n_samples = n_samples

        kern = self.cfg.temporal_kernel or max(3, min(25, n_samples // 4))
        if kern > n_samples:
            raise ValueError(f"temporal kernel {kern} exceeds window {n_samples}")
        self.temporal_kernel = kern
        self.n_time_out = n_samples - kern + 1

        # Change 2: one temporal filter bank, shared across nodes -- exactly
        # ShallowConvNet's first layer, which is a Conv2d over (1, kernel).
        self.temporal = nn.Conv1d(1, self.cfg.n_temporal, kern, bias=False)

        spatial, norms = [], []
        n_in = self.cfg.n_temporal
        for i in range(self.cfg.n_layers):
            basis = bases[i]
            if self.cfg.spatial == "free":
                spatial.append(FreeSpatialConv(n_in, self.cfg.n_spatial, basis,
                                               weight_std=self.cfg.weight_std))
            else:
                spatial.append(ChebConv(n_in, self.cfg.n_spatial, basis,
                                        weight_std=self.cfg.weight_std))
            if self.cfg.n_domains:
                from models.dsbn import DomainSpecificBatchNorm1d
                norms.append(DomainSpecificBatchNorm1d(self.cfg.n_spatial,
                                                       self.cfg.n_domains))
            else:
                norms.append(nn.BatchNorm1d(self.cfg.n_spatial))
            n_in = self.cfg.n_spatial
        self.spatial = nn.ModuleList(spatial)
        self.norms = nn.ModuleList(norms)

        self.pool = nn.MaxPool1d(2) if self.cfg.pool_nodes else None
        n_nodes_out = self.n_nodes // (2 ** self.cfg.n_layers) if self.cfg.pool_nodes \
            else self.n_nodes

        self.drop = nn.Dropout(self.cfg.dropout)
        if self.cfg.readout == "region":
            self.register_buffer(
                "region_onehot", self._region_matrix(node_regions, self.n_nodes))
            self.classifier = nn.Linear(self.cfg.n_regions * self.cfg.n_spatial,
                                        self.cfg.n_classes)
        else:
            self.classifier = nn.Linear(n_nodes_out * self.cfg.n_spatial,
                                        self.cfg.n_classes)

        if self.cfg.graph_smoothness:
            if laplacian is None:
                raise ValueError(
                    "graph_smoothness > 0 needs the layer-0 Laplacian to penalise against"
                )
            self.register_buffer(
                "laplacian", torch.as_tensor(laplacian, dtype=torch.float32))
        else:
            self.laplacian = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (batch, nodes, samples), got {tuple(x.shape)}")
        n, m, t = x.shape
        if m != self.n_nodes or t != self.n_samples:
            raise ValueError(
                f"expected (N, {self.n_nodes}, {self.n_samples}), got {tuple(x.shape)}")

        # temporal filtering, shared across nodes: (N, M, T) -> (N, M, F_t, T')
        h = self.temporal(x.reshape(n * m, 1, t)).reshape(n, m, self.cfg.n_temporal, -1)

        # spatial mixing at every (filter, timepoint): the graph acts across nodes
        # while the time axis is carried along untouched, so power can be taken after.
        tp = h.shape[-1]
        h = h.permute(0, 3, 1, 2).reshape(n * tp, m, self.cfg.n_temporal)
        for conv, norm in zip(self.spatial, self.norms):
            h = conv(h)
            h = norm(h.transpose(1, 2)).transpose(1, 2)
            if self.pool is not None:
                h = self.pool(h.transpose(1, 2)).transpose(1, 2)
        h = h.reshape(n, tp, h.shape[1], h.shape[2])

        # Change 1: square -> mean over time -> log. A differentiable band-power
        # estimate, taken AFTER the spatial combination, which is the ordering CSP
        # and ShallowConvNet share and the one 9.6 did not test.
        h = (h * h).mean(dim=1).clamp_min(self.cfg.eps).log()
        if self.cfg.readout == "region":
            # (R, M) x (N, M, F) -> (N, R, F): mean log-power over each region's nodes.
            h = torch.einsum("rm,nmf->nrf", self.region_onehot, h)
        return self.classifier(self.drop(h.flatten(1)))

    # ---------------------------------------------------------- swapping graphs
    def _region_matrix(self, node_regions: np.ndarray, n_nodes: int) -> torch.Tensor:
        """``(R, M)`` averaging matrix: row r holds ``1 / count_r`` on region r's nodes.

        Counts are clamped to at least 1, so a region with no nodes in the present
        montage is an all-zero row and contributes zero features rather than NaN.
        """
        regions = np.asarray(node_regions)
        if regions.shape != (n_nodes,) or not np.issubdtype(regions.dtype, np.integer):
            raise ValueError(
                f"node_regions must be an integer vector of length {n_nodes}, "
                f"got shape {regions.shape} dtype {regions.dtype}")
        if regions.size and (regions.min() < 0 or regions.max() >= self.cfg.n_regions):
            raise ValueError(
                f"node_regions values must lie in [0, {self.cfg.n_regions}), "
                f"got range [{regions.min()}, {regions.max()}]")
        onehot = torch.zeros(self.cfg.n_regions, n_nodes)
        onehot[torch.as_tensor(regions, dtype=torch.long), torch.arange(n_nodes)] = 1.0
        return onehot / onehot.sum(dim=1, keepdim=True).clamp_min(1.0)

    def set_graph(self, basis: np.ndarray, node_regions: np.ndarray) -> None:
        """Swap in another montage's Chebyshev basis and region map, keeping all weights.

        Only the region readout can do this: every learned tensor (temporal filter,
        Chebyshev weights, BatchNorm, classifier) is independent of the node count,
        and the only montage-specific state is the basis and the region matrix. Both
        are replaced together so they cannot disagree about node order. The caller
        must restore the full graph before scoring another condition.

        Args:
            basis: ``(K, S, S)`` basis of the subset graph, with the same K as built.
            node_regions: ``(S,)`` region id per node, in the basis's node order.
        """
        if self.cfg.readout != "region":
            raise ValueError("set_graph is only defined for readout='region'")
        conv = self.spatial[0]
        basis = np.asarray(basis)
        if basis.ndim != 3 or basis.shape[1] != basis.shape[2] or basis.shape[0] != conv.K:
            raise ValueError(f"basis must be ({conv.K}, S, S), got {basis.shape}")
        n_nodes = basis.shape[1]
        region_onehot = self._region_matrix(node_regions, n_nodes)   # validate first
        device = conv.basis.device
        conv.basis = torch.as_tensor(basis, dtype=torch.float32, device=device)
        conv.n_nodes = n_nodes
        self.region_onehot = region_onehot.to(device)
        self.n_nodes = n_nodes

    # ------------------------------------------------------------- penalties
    def l2_penalty(self) -> torch.Tensor:
        """L2 over the classifier only, matching GCNsNet and models.eegnet."""
        total = 0.5 * (self.classifier.weight ** 2).sum()
        if self.classifier.bias is not None:
            total = total + 0.5 * (self.classifier.bias ** 2).sum()
        return total

    def graph_penalty(self) -> torch.Tensor:
        """Dirichlet energy of the spatial operators over the electrode graph.

        The regulariser variant. For a row ``v`` of a spatial operator — the mixing
        weights one node applies to all others — ``v L vᵀ`` with the combinatorial
        Laplacian is ``0.5 * sum_ij A_ij (v_i - v_j)^2``, which is small exactly when
        ``v`` varies smoothly across graph neighbours and zero when it is uniform. Summing it over rows and layers expresses
        "an electrode's spatial pattern should resemble its neighbours'" as a penalty
        rather than as a hard constraint, which is the distinction A1 showed matters:
        the constraint recovered 16 % of the gap, and a prior is free to be ignored
        where the data disagrees.
        """
        if self.laplacian is None:
            return torch.zeros((), device=self.classifier.weight.device)
        total = torch.zeros((), device=self.laplacian.device)
        for conv in self.spatial:
            ops = getattr(conv, "operators", None)
            if ops is None:                      # a fixed Chebyshev basis has none
                continue
            for k in range(ops.shape[0]):
                v = ops[k]                       # (M, M): row i is node i's mixing
                total = total + (v @ self.laplacian * v).sum()
        return total / max(1, len(self.spatial))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
