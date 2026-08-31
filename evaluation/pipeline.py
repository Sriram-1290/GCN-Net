"""Assemble one fold end to end.

**This module exists so that every leakage-sensitive operation happens in one
auditable place, in the right order.** Scattering these steps across scripts is
how test statistics leak into training without anyone noticing. The order is:

1. Split trials (already done by the caller — this module never splits).
2. Build the adjacency from the **training trials only**.
3. Coarsen that graph and derive the node permutation.
4. Fit the standardiser on the **training trials only**.
5. Apply the standardiser, permutation and timepoint expansion to every split.

Steps 2 and 4 are the two that the reference gets wrong (adjacency over all data,
and BatchNorm aside, its own centring is CAR so that one is fine). Setting
``leaky=True`` reproduces the reference's version for the replication arm, and is
noisy about it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.normalize import Standardizer, fit_standardizer
from graphs.adjacency import pcc_adjacency
from graphs.coarsening import coarsen, perm_data
from models.gcns_net import build_bases


@dataclass(frozen=True)
class Trials:
    """Trial-level data for one split."""

    X: np.ndarray  # (n_trials, n_channels, n_samples)
    y: np.ndarray  # (n_trials,)

    def __post_init__(self) -> None:
        if self.X.ndim != 3:
            raise ValueError(f"expected (trials, channels, samples), got {self.X.shape}")
        if len(self.y) != len(self.X):
            raise ValueError(f"{len(self.y)} labels for {len(self.X)} trials")

    def __len__(self) -> int:
        return len(self.X)


@dataclass(frozen=True)
class Samples:
    """Timepoint-level samples, the form the model consumes."""

    X: np.ndarray        # (n_timepoints, n_nodes, 1) float32
    y: np.ndarray        # (n_timepoints,) int64
    trial_id: np.ndarray  # (n_timepoints,) int64 — for trial-level aggregation

    def __len__(self) -> int:
        return len(self.X)


@dataclass(frozen=True)
class Fold:
    train: Samples
    val: Samples | None
    test: Samples
    bases: list[np.ndarray]
    perm: list[int]
    adjacency: np.ndarray
    standardizer: Standardizer
    leaky: bool


def to_timepoint_samples(X: np.ndarray, y: np.ndarray, trial_offset: int = 0) -> Samples:
    """Expand ``(trials, nodes, time)`` into one sample per timepoint.

    This is the reference's sample definition (``Fin = 1``). Every timepoint of a
    trial inherits the trial's label, and carries its trial id so predictions can
    be aggregated back afterwards.
    """
    n_trials, n_nodes, n_time = X.shape
    # (trials, nodes, time) -> (trials, time, nodes) -> (trials*time, nodes, 1)
    flat = np.ascontiguousarray(X.transpose(0, 2, 1)).reshape(n_trials * n_time, n_nodes, 1)
    labels = np.repeat(y, n_time).astype(np.int64)
    trial_id = np.repeat(np.arange(n_trials, dtype=np.int64) + trial_offset, n_time)
    return Samples(X=flat.astype(np.float32), y=labels, trial_id=trial_id)


def prepare_fold(train: Trials, test: Trials, val: Trials | None = None, *,
                 levels: int = 5, orders: tuple[int, ...] = (2,) * 6,
                 lmax: float | None = 2.0, seed: int = 42,
                 leaky: bool = False) -> Fold:
    """Build graph, statistics and tensors for one fold.

    Args:
        train, test, val: trial-level splits. ``val`` may be ``None`` for the
            subject-dependent protocol where a k-fold already holds out test.
        lmax: ``2.0`` reproduces the reference; ``None`` uses the true lambda_max.
        leaky: reproduce the reference's protocol — build the adjacency and the
            standardiser from **all** splits pooled. Only for the replication arm.

    Returns:
        A :class:`Fold` with everything the trainer needs.
    """
    if leaky:
        # The reference computes the PCC over "all the signals of participants",
        # so the graph the model convolves over knows about the test trials.
        pool = [train.X, test.X] + ([val.X] if val is not None else [])
        fit_source = np.concatenate(pool, axis=0)
    else:
        fit_source = train.X

    adjacency = pcc_adjacency(fit_source)
    graphs, perm = coarsen(adjacency, levels=levels, self_connections=False, seed=seed)
    bases = build_bases(graphs, orders, lmax=lmax)
    standardizer = fit_standardizer(fit_source)

    def convert(split: Trials, offset: int) -> Samples:
        X = standardizer.apply(split.X)
        X = perm_data(X, perm)
        return to_timepoint_samples(X, split.y, trial_offset=offset)

    # Distinct trial-id ranges per split so aggregated ids never collide.
    train_s = convert(train, 0)
    val_s = convert(val, 100_000) if val is not None else None
    test_s = convert(test, 200_000)

    return Fold(train=train_s, val=val_s, test=test_s, bases=bases, perm=perm,
                adjacency=adjacency, standardizer=standardizer, leaky=leaky)


def concat_trials(parts: list[Trials]) -> Trials:
    """Pool several subjects' trials into one split (used by LOSO)."""
    if not parts:
        raise ValueError("cannot concatenate zero splits")
    return Trials(
        X=np.concatenate([p.X for p in parts], axis=0),
        y=np.concatenate([p.y for p in parts], axis=0),
    )


def prepare_leaky_fold(trials: Trials, test_frac: float = 0.1, *, levels: int = 5,
                       orders: tuple[int, ...] = (2,) * 6, lmax: float | None = 2.0,
                       seed: int = 42) -> Fold:
    """Reproduce the reference protocol exactly. **Never for a reported result.**

    Every leakage path at once, as `make_dataset.m` + `GCN_Model.py` implement them:

    1. adjacency `A = |P| - I` over *all* trials, including test;
    2. standardiser fit over all trials;
    3. the split is `randperm` over **timepoints**, so ~every trial contributes
       rows to both sides.

    (BatchNorm's train-mode-at-test is the fourth path and lives in the model,
    via ``GCNsNetConfig(always_train_bn=True)``.)

    This exists to *measure* the leakage rather than assert it: same model, same
    data, only the protocol differs from :func:`prepare_fold`.
    """
    from data.splits import leaky_timepoint_holdout

    adjacency = pcc_adjacency(trials.X)
    graphs, perm = coarsen(adjacency, levels=levels, self_connections=False, seed=seed)
    bases = build_bases(graphs, orders, lmax=lmax)
    standardizer = fit_standardizer(trials.X)

    X = perm_data(standardizer.apply(trials.X), perm)
    pooled = to_timepoint_samples(X, trials.y)

    n_trials, n_time = trials.X.shape[0], trials.X.shape[2]
    train_idx, test_idx = leaky_timepoint_holdout(n_trials, n_time, test_frac, seed)

    def take(idx):
        return Samples(X=pooled.X[idx], y=pooled.y[idx], trial_id=pooled.trial_id[idx])

    return Fold(train=take(train_idx), val=None, test=take(test_idx), bases=bases,
                perm=perm, adjacency=adjacency, standardizer=standardizer, leaky=True)
