"""Attention over time within a sample, before the graph convolutions.

The second component of the contribution (H2 in the pre-registration, §10.3). The
first, the learnable adjacency, addresses *where* on the scalp the model looks;
this addresses *when* within a trial.

What it replaces
----------------
With a windowed sample the network already sees ``(N, M, T)`` and the first
Chebyshev filter mixes the ``T`` timepoints with a **fixed linear map** — the same
weights for every trial, so the model cannot emphasise the part of one trial where
the imagery actually happened. Motor imagery ERD/ERS is not uniform across a
trial: it builds after the cue and decays, and the onset varies by trial and by
subject. Attention lets the weighting be computed per trial from the data.

How the scores are computed
---------------------------
The score at time ``t`` is derived from the **whole 64-channel pattern** at that
instant, not from one channel, so the model asks "does this moment look like
motor imagery across the scalp?" rather than scoring each electrode separately.
Scores are shared across nodes by default: ERD timing is broadly common across
channels within a trial, a shared profile is far more interpretable (one curve per
trial rather than 64), and it costs fewer parameters. ``per_node=True`` relaxes
this if regional timing differences turn out to matter.

Honest note on the ablation
---------------------------
Unlike :mod:`models.dynamic_adjacency`, this is **not** initialisation-equivalent
to the baseline. Attention pools ``T`` timepoints into ``n_heads`` summaries, so
the first conv sees ``Fin = n_heads`` rather than ``Fin = T``; there is no setting
of the parameters that reproduces the fixed linear mix exactly. The comparison is
therefore "attention pooling versus linear mixing" and the parameter counts differ
slightly in both directions — attention adds its own weights but shrinks the first
conv. Both counts are logged so the difference cannot be mistaken for capacity.
"""

from __future__ import annotations

import torch
from torch import nn


class TemporalAttention(nn.Module):
    """Pool the time axis of ``(N, M, T)`` into ``(N, M, n_heads)`` by attention.

    Args:
        n_nodes: number of electrodes ``M``; the score network reads across them.
        hidden: width of the score network's hidden layer.
        n_heads: number of attention profiles. Several heads let the model keep
            more than one temporal summary — e.g. an early and a late window —
            instead of collapsing the trial to a single number.
        per_node: compute a separate temporal profile per electrode. Off by
            default; see the module docstring.
        dropout: applied to the attention weights, which is the usual place for it
            in an attention block.
    """

    def __init__(self, n_nodes: int, hidden: int = 32, n_heads: int = 4,
                 per_node: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        self.n_nodes = n_nodes
        self.n_heads = n_heads
        self.per_node = per_node

        out_dim = n_heads * n_nodes if per_node else n_heads
        self.score = nn.Sequential(
            nn.Linear(n_nodes, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else None
        self._last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, M, T) -> (N, M, n_heads)``."""
        if x.dim() != 3:
            raise ValueError(f"expected (batch, nodes, time), got {tuple(x.shape)}")
        n_batch, n_nodes, n_time = x.shape
        if n_nodes != self.n_nodes:
            raise ValueError(f"expected {self.n_nodes} nodes, got {n_nodes}")
        if n_time < 2:
            raise ValueError(
                f"temporal attention needs a time axis, got T={n_time}. Use a "
                f"windowed sample (--window > 1), not the reference's single timepoint."
            )

        xt = x.transpose(1, 2)                       # (N, T, M)
        scores = self.score(xt)                      # (N, T, heads) or (N, T, heads*M)

        if self.per_node:
            scores = scores.view(n_batch, n_time, self.n_heads, n_nodes)
            alpha = torch.softmax(scores, dim=1)     # normalise over time
            if self.drop is not None:
                alpha = self.drop(alpha)
            out = torch.einsum("ntm,nthm->nmh", xt, alpha)
        else:
            alpha = torch.softmax(scores, dim=1)     # (N, T, heads), over time
            if self.drop is not None:
                alpha = self.drop(alpha)
            out = torch.einsum("ntm,nth->nmh", xt, alpha)

        # Kept for the interpretability figures: which moments the model used.
        self._last_weights = alpha.detach()
        return out

    def last_attention(self) -> torch.Tensor | None:
        """Attention weights from the most recent forward pass, for Phase 6."""
        return self._last_weights

    def extra_repr(self) -> str:
        return (f"n_nodes={self.n_nodes}, n_heads={self.n_heads}, "
                f"per_node={self.per_node}")
