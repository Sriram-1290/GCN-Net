"""Is DSBN actually wired, and is the non-DSBN path still untouched?

Two failures matter here and both are silent. If domain labels never reach the
trainer, DSBN averages over subjects and the arm measures nothing. If the wiring
perturbs the default path, every previously logged arm becomes incomparable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.pipeline import Samples  # noqa: E402
from evaluation.trainer import TrainConfig, Trainer  # noqa: E402
from models.dsbn import DomainSpecificBatchNorm1d  # noqa: E402
from models.graph_bandpower import (GraphBandPowerConfig,  # noqa: E402
                                    GraphBandPowerNet)


def _trainer(n_domains=0):
    bases = [torch.stack([torch.eye(8)] * 3).numpy()]
    cfg = GraphBandPowerConfig(n_classes=2, spatial="free", n_domains=n_domains)
    m = GraphBandPowerNet(bases, 16, cfg)
    return Trainer(m, TrainConfig(epochs=1, batch_size=4, device="cpu", verbose=False),
                   n_classes=2)


def test_samples_defaults_to_no_domains():
    """Additive field: everything that never heard of domains keeps working."""
    s = Samples(X=np.zeros((4, 8, 1), np.float32), y=np.zeros(4, np.int64),
                trial_id=np.zeros(4, np.int64))
    assert s.domains is None


def test_batches_without_domains_are_the_old_behaviour():
    t = _trainer()
    g = torch.Generator().manual_seed(0)
    b = t._domain_batches(10, g, None)
    assert all(d is None for _, d in b)
    assert sum(len(i) for i, _ in b) == 10
    assert len(torch.cat([i for i, _ in b]).unique()) == 10, "every sample exactly once"


def test_every_batch_is_drawn_from_a_single_domain():
    """A batch mixing subjects has no one correct set of normalisation statistics."""
    t = _trainer(n_domains=4)
    dom = np.array([0] * 7 + [1] * 7 + [2] * 6)
    g = torch.Generator().manual_seed(0)
    for idx, d in t._domain_batches(len(dom), g, dom):
        assert len(set(dom[idx.numpy()])) == 1, "batch spans domains"
        assert dom[idx.numpy()][0] == d, "batch labelled with the wrong domain"


def test_domain_batching_covers_every_sample_exactly_once():
    t = _trainer(n_domains=4)
    dom = np.array([0] * 7 + [1] * 7 + [2] * 6)
    g = torch.Generator().manual_seed(0)
    seen = torch.cat([i for i, _ in t._domain_batches(len(dom), g, dom)])
    assert sorted(seen.tolist()) == list(range(len(dom)))


def test_model_uses_dsbn_only_when_asked():
    def count(m):
        return sum(1 for x in m.modules() if isinstance(x, DomainSpecificBatchNorm1d))
    assert count(_trainer(n_domains=5).model) == 1
    assert count(_trainer().model) == 0, "DSBN leaked into the default path"


def test_prediction_selects_the_reserved_target_slot():
    """Anything predicted on is an unseen domain and must not borrow a source's stats."""
    t = _trainer(n_domains=4)
    layer = next(x for x in t.model.modules()
                 if isinstance(x, DomainSpecificBatchNorm1d))
    layer.set_domain(1)
    t.predict_logits(Samples(X=np.zeros((4, 8, 16), np.float32),
                             y=np.zeros(4, np.int64), trial_id=np.arange(4)))
    assert layer.domain == layer.n_domains - 1


def test_dsbn_is_part_of_the_source_model_cache_key():
    """A DSBN source model must never be reused for a non-DSBN run, or the arm
    silently mixes two normalisation schemes."""
    src = (ROOT / "scripts" / "run_pretrain_finetune.py").read_text(encoding="utf-8")
    key = src[src.index("def source_cache_key"):src.index("def subset_samples")]
    assert '"dsbn"' in key, "cache key ignores --dsbn"
