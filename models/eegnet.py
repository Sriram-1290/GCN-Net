"""EEGNet and ShallowConvNet — the A2 control (project_documentation.md 17.2).

**Why these two exist in this repository.** Every deep result so far comes from one
architecture, the one under audit. That leaves "the GCN loses to CSP+LDA"
ambiguous between two readings that imply very different conclusions:

* **the graph prior is wrong** — 9.7.1's claim, which A1 tests directly; or
* **any deep model is below its data threshold here** — 17.1, prompted by the
  MOABB benchmark's finding that deep learning "typically requires more than 150
  trials per class", and does well on an *easy* binary problem "with as few as 50
  trials per class", while Riemannian and raw pipelines succeed below 50. We have
  ~45 trials per binary problem and ~36 per training fold.

EEGNet contains no graph anywhere: its spatial stage is an unconstrained
depthwise convolution across all channels, which is precisely the operation
9.7.1 says a Chebyshev filter cannot express. So if EEGNet *also* loses to CSP by
a similar margin on identical folds, the limitation is sample size and 9.7.1 is
unidentified. If it closes the gap, 9.7.1 survives A2 and A1 should agree.

Both models take ``(N, C, T)`` — the same array the graph arms consume, minus the
Graclus permutation — so the entire ``Samples`` / ``Trainer`` / ``metrics`` path
is shared and the trial-level aggregation is identical. Nothing about the split,
the standardiser, the folds or the logging differs from the arms they are
compared against.

Window length, and why it is called out
---------------------------------------
**These architectures were designed for 2-4 second windows** (Lawhern: 500 samples
at 128 Hz; Schirrmeister: ~500+ at 250 Hz). Our confirmatory arms use
``window=64`` at 160 Hz, which is **0.40 s** — far outside that regime, and short
enough that the paper's absolute kernel and pool sizes do not fit at all.

Rather than silently truncating, every temporal hyperparameter here is **derived
from T**, with the papers' published values recovered when T reaches the length
they assumed. The derivation is in each class's docstring and the resulting sizes
are logged. This is a real caveat on A2 and it should be reported as one — it is
also independent motivation for B5 (17.3), the window-length sweep that has never
been run.

Regularisation
--------------
EEGNet regularises by max-norm constraints, not weight decay. Those are
implemented here (depthwise ``max_norm=1``, classifier ``max_norm=0.25``) and
applied in ``forward`` during training, so the shared :class:`Trainer` needs no
change — touching the trainer would risk altering the arms these are compared
against, which CLAUDE.md forbids. Both classes also expose ``l2_penalty`` with
the same convention as :class:`GCNsNet` (L2 over the classifier only, matching
``tf.nn.l2_loss``'s factor of a half), so the shared ``--regularization`` flag
means the same thing for every arm.
"""

from __future__ import annotations

import torch
from torch import nn


def _max_norm_(weight: torch.Tensor, max_norm: float, dim: int = 0) -> None:
    """Rescale slices of ``weight`` along ``dim`` to have norm at most ``max_norm``.

    Keras' ``MaxNorm`` constraint, which both reference implementations use. In
    place and under ``no_grad`` — this is a projection applied to the parameter,
    not a term in the loss.
    """
    with torch.no_grad():
        norm = weight.norm(2, dim=tuple(i for i in range(weight.dim()) if i != dim),
                           keepdim=True).clamp_min(1e-8)
        weight.mul_((norm.clamp_max(max_norm) / norm))


class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern et al. 2018), adapted to short windows.

    Args:
        n_channels: ``C``. 64 for this dataset.
        n_samples: ``T``. 64 for the confirmatory arms (0.40 s at 160 Hz).
        n_classes: output classes.
        sfreq: sampling rate, used only to derive ``kern_length``.
        f1: temporal filters. ``8`` is the published EEGNet-8,2.
        depth: spatial filters per temporal filter (``D``). ``2`` published.
        dropout: published value is 0.25 for within-subject work.

    Temporal sizing, derived rather than hardcoded:

    * ``kern_length = min(sfreq // 2, T // 2)``. The paper's heuristic is half the
      sampling rate (80 samples here); at ``T = 64`` that exceeds the window, so
      it falls back to half the window (32). At ``T >= 160`` the paper's value is
      recovered exactly.
    * The two average pools are the published ``(1, 4)`` and ``(1, 8)``, a 32x
      total reduction, so ``T`` must be at least 32. At ``T = 64`` the classifier
      sees ``F2 x 2`` features.
    """

    def __init__(self, n_channels: int, n_samples: int, n_classes: int,
                 sfreq: float = 160.0, f1: int = 8, depth: int = 2,
                 dropout: float = 0.25) -> None:
        super().__init__()
        if n_samples < 32:
            raise ValueError(
                f"EEGNet pools by 32x total; T={n_samples} leaves nothing. "
                "Use --window 32 or more."
            )
        f2 = f1 * depth
        self.n_channels, self.n_samples = n_channels, n_samples
        self.kern_length = max(2, min(int(sfreq // 2), n_samples // 2))
        # published separable kernel is 16, at the post-pool length of T // 4
        self.sep_length = max(2, min(16, n_samples // 4))

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, self.kern_length), padding="same", bias=False),
            nn.BatchNorm2d(f1),
        )
        # The spatial stage: one unconstrained filter over ALL channels per
        # temporal filter. This is the operation 9.7.1 says a Chebyshev
        # convolution cannot express, and it is why EEGNet is the right control.
        self.depthwise = nn.Conv2d(f1, f1 * depth, (n_channels, 1), groups=f1,
                                   bias=False)
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(f1 * depth), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(f1 * depth, f1 * depth, (1, self.sep_length),
                      padding="same", groups=f1 * depth, bias=False),
            nn.Conv2d(f1 * depth, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(f2 * (n_samples // 32), n_classes)

    def _constrain(self) -> None:
        _max_norm_(self.depthwise.weight, 1.0, dim=0)
        _max_norm_(self.classifier.weight, 0.25, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, C, T) -> (N, n_classes)``."""
        if x.dim() != 3:
            raise ValueError(f"expected (batch, channels, samples), got {tuple(x.shape)}")
        if x.shape[1] != self.n_channels or x.shape[2] != self.n_samples:
            raise ValueError(
                f"expected (N, {self.n_channels}, {self.n_samples}), got {tuple(x.shape)}"
            )
        if self.training:
            self._constrain()
        x = x.unsqueeze(1)                       # (N, 1, C, T)
        x = self.depthwise(self.block1(x))       # (N, F1*D, 1, T)
        x = self.separable(self.block2(x))       # (N, F2, 1, T//32)
        return self.classifier(x.flatten(1))

    def l2_penalty(self) -> torch.Tensor:
        """L2 over the classifier only — the same convention as ``GCNsNet``."""
        total = 0.5 * (self.classifier.weight ** 2).sum()
        if self.classifier.bias is not None:
            total = total + 0.5 * (self.classifier.bias ** 2).sum()
        return total

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ShallowConvNet(nn.Module):
    """ShallowConvNet (Schirrmeister et al. 2017), adapted to short windows.

    A deliberately FBCSP-shaped network: temporal convolution, then an
    unconstrained spatial combination across all channels, then square -> average
    pool -> log, which is a differentiable band-power estimate. That makes it the
    most direct deep analogue of the CSP+LDA baseline it is being compared with,
    and a useful second point alongside EEGNet.

    Temporal sizing, derived rather than hardcoded. The published values (25-sample
    convolution, 75-sample pool at stride 15) assume ~500 samples; here they scale
    with ``T`` and are recovered at that length. Sizes are asserted, not clamped
    silently.
    """

    def __init__(self, n_channels: int, n_samples: int, n_classes: int,
                 n_filters: int = 40, dropout: float = 0.5) -> None:
        super().__init__()
        self.n_channels, self.n_samples = n_channels, n_samples
        self.temporal_kernel = max(3, min(25, n_samples // 4))
        conv_out = n_samples - self.temporal_kernel + 1
        self.pool = max(2, min(75, conv_out // 3))
        self.pool_stride = max(1, self.pool // 5)
        pooled = (conv_out - self.pool) // self.pool_stride + 1
        if pooled < 1:
            raise ValueError(
                f"T={n_samples} is too short for ShallowConvNet: "
                f"kernel {self.temporal_kernel}, pool {self.pool}"
            )

        self.temporal = nn.Conv2d(1, n_filters, (1, self.temporal_kernel), bias=False)
        # Unconstrained spatial combination over all channels -- as with EEGNet,
        # this is the operation the graph formulation cannot express.
        self.spatial = nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.pooling = nn.AvgPool2d((1, self.pool), stride=(1, self.pool_stride))
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(n_filters * pooled, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, C, T) -> (N, n_classes)``."""
        if x.dim() != 3:
            raise ValueError(f"expected (batch, channels, samples), got {tuple(x.shape)}")
        if x.shape[1] != self.n_channels or x.shape[2] != self.n_samples:
            raise ValueError(
                f"expected (N, {self.n_channels}, {self.n_samples}), got {tuple(x.shape)}"
            )
        x = self.bn(self.spatial(self.temporal(x.unsqueeze(1))))
        # square -> average -> log: a differentiable log band-power estimate.
        # clamp_min guards the log; the reference uses eps = 1e-6.
        x = self.pooling(x * x).clamp_min(1e-6).log()
        return self.classifier(self.drop(x).flatten(1))

    def l2_penalty(self) -> torch.Tensor:
        total = 0.5 * (self.classifier.weight ** 2).sum()
        if self.classifier.bias is not None:
            total = total + 0.5 * (self.classifier.bias ** 2).sum()
        return total

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_baseline(name: str, n_channels: int, n_samples: int, n_classes: int,
                   sfreq: float = 160.0) -> nn.Module:
    """Factory used by ``run_subject_dependent.py --model {eegnet,shallow}``."""
    if name == "eegnet":
        return EEGNet(n_channels, n_samples, n_classes, sfreq=sfreq)
    if name == "shallow":
        return ShallowConvNet(n_channels, n_samples, n_classes)
    raise ValueError(f"unknown baseline {name!r}; expected 'eegnet' or 'shallow'")
