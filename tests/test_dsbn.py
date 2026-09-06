"""Does domain-specific batch norm keep the domains separate?

The failure this guards against is silent: if the domain index is ignored, DSBN
degrades to a plain BatchNorm and the arm measures nothing, while still training
and still producing a plausible number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.dsbn import (DomainSpecificBatchNorm1d, seed_target_domain,  # noqa: E402
                         set_domain)


def test_shape_matches_batchnorm1d():
    d = DomainSpecificBatchNorm1d(8, n_domains=3)
    x = torch.randn(16, 8, 5)
    assert d(x).shape == x.shape


def test_domains_accumulate_separate_statistics():
    """The whole point: two domains fed different data must not share running stats."""
    d = DomainSpecificBatchNorm1d(4, n_domains=2)
    d.train()
    d.set_domain(0)
    d(torch.randn(32, 4) * 1.0 + 10.0)
    d.set_domain(1)
    d(torch.randn(32, 4) * 1.0 - 10.0)
    m0 = d.norms[0].running_mean
    m1 = d.norms[1].running_mean
    assert (m0 - m1).abs().max() > 1.0, "domains share statistics; DSBN is inert"


def test_affine_parameters_are_shared_across_domains():
    """Only the statistics are per-domain. Sharing the affine transform is what makes
    this a normalisation fix rather than N separate models."""
    d = DomainSpecificBatchNorm1d(4, n_domains=3)
    names = {n for n, _ in d.named_parameters()}
    assert names == {"weight", "bias"}, f"unexpected parameters: {names}"
    assert d.weight.numel() == 4 and d.bias.numel() == 4


def test_eval_uses_the_selected_domains_statistics():
    d = DomainSpecificBatchNorm1d(4, n_domains=2)
    d.train()
    d.set_domain(0); d(torch.randn(64, 4) + 5.0)
    d.set_domain(1); d(torch.randn(64, 4) - 5.0)
    d.eval()
    x = torch.zeros(8, 4)
    d.set_domain(0); a = d(x)
    d.set_domain(1); b = d(x)
    assert not torch.allclose(a, b), "eval ignores the domain selection"


def test_set_domain_rejects_out_of_range():
    d = DomainSpecificBatchNorm1d(4, n_domains=2)
    for bad in (-1, 2, 99):
        with pytest.raises(ValueError):
            d.set_domain(bad)


def test_set_domain_helper_reaches_nested_layers_and_reports_absence():
    net = torch.nn.Sequential(torch.nn.Linear(4, 4),
                              DomainSpecificBatchNorm1d(4, n_domains=3))
    assert set_domain(net, 2) is True
    assert net[1].domain == 2
    plain = torch.nn.Sequential(torch.nn.BatchNorm1d(4))
    assert set_domain(plain, 1) is False, "must report when no DSBN is present"


def test_target_domain_is_seeded_from_the_source_mean():
    """A fine-tuning subject has no statistics; starting from BatchNorm's default would
    discard what pre-training learned about the scale of the data."""
    d = DomainSpecificBatchNorm1d(4, n_domains=3)
    with torch.no_grad():
        d.norms[0].running_mean.fill_(2.0)
        d.norms[1].running_mean.fill_(4.0)
    seed_target_domain(d, target=2)
    assert torch.allclose(d.norms[2].running_mean, torch.full((4,), 3.0))


def test_zero_domains_is_rejected():
    with pytest.raises(ValueError):
        DomainSpecificBatchNorm1d(4, n_domains=0)
