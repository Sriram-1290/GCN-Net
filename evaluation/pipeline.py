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
from models.gcns_net import build_bases, dense_adjacencies


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
    # Which source subject each sample came from, when the model normalises per
    # domain (models.dsbn). None everywhere else, so nothing that ignores it changes.
    domains: np.ndarray | None = None  # (n_timepoints,) int64

    def __len__(self) -> int:
        return len(self.X)


@dataclass(frozen=True)
class Fold:
    train: Samples
    val: Samples | None
    test: Samples
    bases: list[np.ndarray]
    adjacencies: list[np.ndarray]  # per-layer dense graphs, to seed a learnable one
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


def to_window_samples(X: np.ndarray, y: np.ndarray, window: int, stride: int,
                      trial_offset: int = 0, start: int = 0) -> Samples:
    """Cut each trial into overlapping windows of ``window`` timepoints.

    The reference's ``window = 1`` gives one sample per timepoint and therefore no
    temporal or spectral information whatsoever — section 9.2 shows that
    formulation decodes nothing once the split is honest. Widening the window is
    the minimal change that gives the network something a band-power effect could
    actually be visible in, and it is also what a temporal-attention module needs.

    Windows never cross a trial boundary, so a window is always wholly inside one
    trial and inherits that trial's label. Overlap between windows of the *same*
    trial is fine — they stay on the same side of any trial-level split.

    ``start`` is the sample index of the first window (default 0, the original
    behaviour). A positive value drops the early samples of every trial, e.g. to
    exclude the cue-locked onset transient (Phase A, A4).
    """
    n_trials, n_nodes, n_time = X.shape
    if not 1 <= window <= n_time:
        raise ValueError(f"window {window} must be in [1, {n_time}]")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}")

    starts = np.arange(start, n_time - window + 1, stride)
    if len(starts) == 0:
        raise ValueError(
            f"window {window} / stride {stride} / start {start} yields no windows")

    # (trials, windows, nodes, window) -> (trials*windows, nodes, window)
    chunks = np.stack([X[:, :, s:s + window] for s in starts], axis=1)
    flat = chunks.reshape(n_trials * len(starts), n_nodes, window)
    labels = np.repeat(y, len(starts)).astype(np.int64)
    trial_id = np.repeat(np.arange(n_trials, dtype=np.int64) + trial_offset, len(starts))
    return Samples(X=np.ascontiguousarray(flat, dtype=np.float32), y=labels, trial_id=trial_id)


def prepare_fold(train: Trials, test: Trials, val: Trials | None = None, *,
                 levels: int = 5, orders: tuple[int, ...] = (2,) * 6,
                 lmax: float | None = 2.0, seed: int = 42, sfreq: float = 160.0,
                 window: int = 1, stride: int | None = None,
                 features: str = "raw",
                 bands: str | tuple[tuple[float, float], ...] = "mu_beta",
                 leaky: bool = False, build_graph: bool = True,
                 train_samples: bool = True, window_start: int = 0) -> Fold:
    """Build graph, statistics and tensors for one fold.

    Args:
        train, test, val: trial-level splits. ``val`` may be ``None`` for the
            subject-dependent protocol where a k-fold already holds out test.
        lmax: ``2.0`` reproduces the reference; ``None`` uses the true lambda_max.
        window: timepoints per sample. ``1`` is the reference's definition; larger
            values give the model temporal extent (see :func:`to_window_samples`).
        stride: hop between windows; defaults to ``window`` (non-overlapping).
        window_start: sample index of the first window, applied identically to the
            training, validation and test windows (all go through ``convert``).
            ``0`` is the original behaviour. Raw features only.
        features: ``"raw"`` uses voltage samples as node features; ``"bandpower"``
            uses log band power per channel per band (see :mod:`data.features`).
            Band power must be computed from an *unfiltered* cache.
        bands: named band set or explicit ``((lo, hi), ...)``, for band power only.
        leaky: reproduce the reference's protocol — build the adjacency and the
            standardiser from **all** splits pooled. Only for the replication arm.
        build_graph: ``False`` skips graph construction entirely — no adjacency,
            no coarsening, no node permutation — for models that do not use one
            (A2's EEGNet and ShallowConvNet, 17.2). Steps 1, 4 and 5 above are
            unchanged, which is the point of routing those arms through this
            function rather than a parallel one: the leakage-sensitive operations
            stay in a single audited place for every arm.
        train_samples: ``False`` still fits the graph and the standardiser on the
            training trials but does not window them: ``Fold.train`` is then an
            empty :class:`Samples` with the usual per-sample shape. For callers that
            only score a fitted model, such as the pre-train/fine-tune runner on a
            source-model cache hit, where the source windows (~2.2 GiB) feed only
            the pre-training that is skipped. Everything else is unchanged.

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

    if build_graph:
        adjacency = pcc_adjacency(fit_source)
        graphs, perm = coarsen(adjacency, levels=levels,
                               self_connections=False, seed=seed)
        bases = build_bases(graphs, orders, lmax=lmax)
        adjacencies = dense_adjacencies(graphs)
    else:
        if features != "raw":
            raise ValueError(
                "build_graph=False supports raw features only; band power is a"
                " graph-arm ablation (9.6)"
            )
        # Identity permutation: channels stay in montage order, which is what a
        # convolutional spatial filter expects. No fake nodes are added.
        adjacency = np.empty((0, 0), dtype=np.float32)
        bases, adjacencies = [], []
        perm = list(range(fit_source.shape[1]))
    standardizer = fit_standardizer(fit_source)

    step = stride if stride is not None else window
    if window_start and features != "raw":
        raise ValueError("window_start supports raw features only")
    if window_start and window == 1 and step == 1:
        raise ValueError("window_start needs window > 1 (timepoint samples)")

    if features == "bandpower":
        # The graph still comes from the raw time series, so this is a single
        # change of node features and nothing else.
        from data.features import band_power_features, resolve_bands
        band_tuple = resolve_bands(bands)

        def featurise(split: Trials) -> tuple[np.ndarray, np.ndarray]:
            return band_power_features(split.X, sfreq, band_tuple, window, step)

        train_feat, _ = featurise(train)
        standardizer = fit_standardizer(train_feat, per_feature=True)

        def convert(split: Trials, offset: int) -> Samples:
            feat, trial_index = featurise(split)
            feat = perm_data(standardizer.apply(feat), perm)
            return Samples(X=feat.astype(np.float32),
                           y=split.y[trial_index].astype(np.int64),
                           trial_id=trial_index + offset)
    elif features != "raw":
        raise ValueError(f"unknown features {features!r}; expected 'raw' or 'bandpower'")
    else:
        def convert(split: Trials, offset: int) -> Samples:
            X = standardizer.apply(split.X)
            if build_graph:
                X = perm_data(X, perm)
            if window == 1 and step == 1:
                return to_timepoint_samples(X, split.y, trial_offset=offset)
            return to_window_samples(X, split.y, window, step, trial_offset=offset,
                                     start=window_start)

    # Distinct trial-id ranges per split so aggregated ids never collide.
    if train_samples:
        train_s = convert(train, 0)
    else:
        if features != "raw":
            raise ValueError("train_samples=False supports raw features only")
        # Zero trials through the same conversion, so the per-sample shape that
        # model factories read is exactly what the full conversion would give.
        train_s = convert(Trials(X=train.X[:0], y=train.y[:0]), 0)
    val_s = convert(val, 100_000) if val is not None else None
    test_s = convert(test, 200_000)

    return Fold(train=train_s, val=val_s, test=test_s, bases=bases,
                adjacencies=adjacencies, perm=perm, adjacency=adjacency,
                standardizer=standardizer, leaky=leaky)


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
    adjacencies = dense_adjacencies(graphs)
    standardizer = fit_standardizer(trials.X)

    X = perm_data(standardizer.apply(trials.X), perm)
    pooled = to_timepoint_samples(X, trials.y)

    n_trials, n_time = trials.X.shape[0], trials.X.shape[2]
    train_idx, test_idx = leaky_timepoint_holdout(n_trials, n_time, test_frac, seed)

    def take(idx):
        return Samples(X=pooled.X[idx], y=pooled.y[idx], trial_id=pooled.trial_id[idx])

    return Fold(train=take(train_idx), val=None, test=take(test_idx), bases=bases,
                adjacencies=adjacencies, perm=perm, adjacency=adjacency,
                standardizer=standardizer, leaky=True)
