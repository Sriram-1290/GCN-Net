"""Domain-specific batch normalisation, for the cross-subject pre-training stage.

**Why.** 9.15's pre-training pools ~92 subjects through a *single* BatchNorm, so one
set of running statistics is asked to describe 92 people whose EEG amplitude and
covariance structure differ substantially. 9.10 measured how much of the outcome is
subject-level: a resting rhythm recorded before any modelling explains ~30 % of the
between-subject variance. Normalising them all together throws that structure away.

TSMNet (Kobler et al., arXiv:2206.01323) isolates this effect on 5 datasets and 158
subjects: removing domain-specific batch normalisation costs **2.4-3.9 %**, the second
largest term in their ablation. The idea is simple -- give each domain its own
normalisation statistics and share every other parameter.

**Why this is worth trying when 9.19 and 9.21 found nothing.** Those closed off
*spatial priors* and the *sample definition*, both of which act within a subject. This
acts between subjects, which is the axis 9.17 showed actually moves: pre-training is
the only intervention that broke the ~0.67 ceiling, and it did so by borrowing other
subjects' data. DSBN is a change to *how* that borrowing is done, not another
architectural prior.

**Batches must be domain-homogeneous.** A minibatch mixing domains has no single
correct set of statistics to use. :class:`evaluation.trainer.Trainer` therefore samples
within a domain when the fold supplies domain labels, exactly as TSMNet does.
"""

from __future__ import annotations

import torch
from torch import nn


class DomainSpecificBatchNorm1d(nn.Module):
    """``BatchNorm1d`` with one set of statistics per domain, weights shared.

    Drop-in for :class:`torch.nn.BatchNorm1d` in shape and dtype. The active domain is
    set on the module before the forward pass rather than passed as an argument, so the
    surrounding model's ``forward(x)`` signature is unchanged and the batch-norm swap
    stays invisible to every other component.

    Args:
        num_features: as ``BatchNorm1d``.
        n_domains: number of source domains (subjects) plus one slot at index
            ``n_domains - 1`` reserved for an unseen target, so a fine-tuned model has
            somewhere to accumulate its own statistics without disturbing the sources.

    The affine parameters are **shared** across domains: only the running statistics and
    the per-batch normalisation differ. Sharing them is what makes this a normalisation
    fix rather than 92 separate models.
    """

    def __init__(self, num_features: int, n_domains: int) -> None:
        super().__init__()
        if n_domains < 1:
            raise ValueError(f"n_domains must be >= 1, got {n_domains}")
        self.num_features = num_features
        self.n_domains = n_domains

        # affine=False: the shared scale and shift live here, applied after whichever
        # domain's normalisation ran, so every domain sees the same learned transform.
        self.norms = nn.ModuleList(
            [nn.BatchNorm1d(num_features, affine=False) for _ in range(n_domains)]
        )
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self._domain = 0

    def set_domain(self, domain: int) -> None:
        if not 0 <= domain < self.n_domains:
            raise ValueError(f"domain {domain} outside [0, {self.n_domains})")
        self._domain = int(domain)

    @property
    def domain(self) -> int:
        return self._domain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norms[self._domain](x)
        shape = [1, self.num_features] + [1] * (h.dim() - 2)
        return h * self.weight.view(shape) + self.bias.view(shape)


def set_domain(module: nn.Module, domain: int) -> bool:
    """Set the active domain on every DSBN layer inside ``module``.

    Returns whether any were found, so callers can tell a DSBN model from a plain one
    without inspecting its configuration.
    """
    found = False
    for m in module.modules():
        if isinstance(m, DomainSpecificBatchNorm1d):
            m.set_domain(domain)
            found = True
    return found


def target_domain(module: nn.Module) -> bool:
    """Select the reserved target slot on every DSBN layer inside ``module``.

    Anything being predicted on is an unseen domain -- a held-out source subject
    during pre-training, or the fine-tuning target -- so it must not borrow a
    particular source subject's statistics. The last slot is reserved for exactly
    this and is seeded from the source mean by :func:`seed_target_domain`.
    """
    found = False
    for m in module.modules():
        if isinstance(m, DomainSpecificBatchNorm1d):
            m.set_domain(m.n_domains - 1)
            found = True
    return found


def seed_target_domain(module: nn.Module, target: int) -> None:
    """Initialise the target domain's statistics from the mean of the source domains.

    A fine-tuning subject has no pre-trained statistics of its own. Starting from the
    average of the sources is strictly better than starting from BatchNorm's default
    (zero mean, unit variance), which would discard everything pre-training learned
    about the scale of this data at the first forward pass.
    """
    for m in module.modules():
        if not isinstance(m, DomainSpecificBatchNorm1d):
            continue
        srcs = [n for i, n in enumerate(m.norms) if i != target]
        if not srcs:
            continue
        with torch.no_grad():
            m.norms[target].running_mean.copy_(
                torch.stack([n.running_mean for n in srcs]).mean(0))
            m.norms[target].running_var.copy_(
                torch.stack([n.running_var for n in srcs]).mean(0))
