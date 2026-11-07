"""Tests for the A5 leak-ablation arms (docs 9.26.3).

The experiment is only interpretable if each toggle changes exactly one thing, so
these tests pin that on synthetic data: ``none`` must be the correct protocol
bit-for-bit, ``all`` must be the reference protocol bit-for-bit, and each single
toggle must differ from ``none`` in its own path and in nothing else.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.splits import leaky_timepoint_holdout, stratified_trial_kfold  # noqa: E402
from evaluation.pipeline import Trials, prepare_fold, prepare_leaky_fold  # noqa: E402
from graphs.adjacency import pcc_adjacency  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


abl = _load("run_leak_ablation")
sd = _load("run_subject_dependent")

SEED = 42


def _data(n=40, ch=64, t=32, seed=0):
    """Synthetic trials with channel-dependent scale and a per-trial drift, so a
    statistic fitted on a different subset of the data is visibly different."""
    rng = np.random.default_rng(seed)
    scale = rng.uniform(0.5, 3.0, size=(1, ch, 1))
    drift = np.linspace(0, 2, n)[:, None, None]
    X = rng.standard_normal((n, ch, t)) * scale + drift * rng.standard_normal((1, ch, 1))
    y = np.arange(n) % 4
    return X, y


def _same_samples(a, b):
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.y, b.y)
    assert np.array_equal(a.trial_id, b.trial_id)


def _same_fold(a, b):
    _same_samples(a.train, b.train)
    _same_samples(a.test, b.test)
    if a.val is None or b.val is None:
        assert a.val is None and b.val is None
    else:
        _same_samples(a.val, b.val)
    assert np.array_equal(a.adjacency, b.adjacency)
    assert a.perm == b.perm
    assert np.array_equal(a.standardizer.mean, b.standardizer.mean)
    assert np.array_equal(a.standardizer.std, b.standardizer.std)
    for ba, bb in zip(a.bases, b.bases):
        assert np.array_equal(ba, bb)


def _build(arm, X, y, fold_id=None):
    leaks = abl.ARMS[arm]
    if fold_id is None:
        fold_id = abl.fold_ids(leaks, 5)[0]
    return abl.build_fold(X, y, leaks, fold_id, n_folds=5, val_frac=0.2, seed=SEED,
                          lmax=2.0, test_frac=0.1, sfreq=160.0)


def _correct_path(X, y, fold_id):
    """What run_subject_dependent.py does for one fold with its default flags."""
    split = stratified_trial_kfold(y, n_splits=5, seed=SEED)[fold_id]
    tr_idx, val_idx = sd.split_off_validation(y, split.train, 0.2, SEED)
    return prepare_fold(Trials(X=X[tr_idx], y=y[tr_idx]),
                        Trials(X=X[split.test], y=y[split.test]),
                        Trials(X=X[val_idx], y=y[val_idx]),
                        lmax=2.0, seed=SEED, sfreq=160.0, window=1, stride=None,
                        features="raw", leaky=False, build_graph=True)


def test_arm_names_and_paths():
    assert set(abl.ARMS) == {"none", "split", "bn", "graph", "all"}
    assert abl.ARMS["none"].paths() == []
    assert abl.ARMS["bn"].paths() == ["batchnorm_train_at_test"]
    assert abl.ARMS["graph"].paths() == ["adjacency_over_all_data",
                                         "standardizer_over_all_data"]
    assert abl.ARMS["split"].paths() == ["timepoint_shuffled_split"]
    assert len(abl.ARMS["all"].paths()) == 4


def test_fold_ids_per_arm():
    for arm in ("none", "bn", "graph"):
        assert abl.fold_ids(abl.ARMS[arm], 5) == [0, 1, 2, 3, 4]
    for arm in ("split", "all"):
        assert abl.fold_ids(abl.ARMS[arm], 5) == [None]


def test_none_matches_correct_path():
    X, y = _data()
    for fold_id in (0, 3):
        built = _build("none", X, y, fold_id)
        _same_fold(built.fold, _correct_path(X, y, fold_id))
        assert built.fold.leaky is False
        assert not set(built.fold.train.trial_id) & set(built.fold.test.trial_id)
    assert abl.model_config(abl.ARMS["none"], 4).always_train_bn is False


def test_all_matches_reference_path():
    X, y = _data()
    built = _build("all", X, y)
    ref = prepare_leaky_fold(Trials(X=X, y=y), test_frac=0.1, lmax=2.0, seed=SEED)
    _same_fold(built.fold, ref)
    assert built.fold.leaky is True and built.fold_id is None
    assert abl.model_config(abl.ARMS["all"], 4).always_train_bn is True


def test_bn_alone_changes_only_the_batchnorm_mode():
    X, y = _data()
    _same_fold(_build("bn", X, y).fold, _build("none", X, y).fold)
    cfg_none = abl.model_config(abl.ARMS["none"], 4)
    cfg_bn = abl.model_config(abl.ARMS["bn"], 4)
    assert cfg_bn.always_train_bn is True and cfg_none.always_train_bn is False
    assert {**vars(cfg_bn), "always_train_bn": False} == vars(cfg_none)


def test_graph_alone_changes_only_the_fit_source():
    X, y = _data()
    none = _build("none", X, y).fold
    g = _build("graph", X, y).fold
    # the trial-level split is untouched
    for a, b in ((g.train, none.train), (g.val, none.val), (g.test, none.test)):
        assert np.array_equal(a.trial_id, b.trial_id) and np.array_equal(a.y, b.y)
    assert not set(g.train.trial_id) & set(g.test.trial_id)
    # adjacency and standardiser come from every trial of the subject
    assert np.allclose(g.adjacency, pcc_adjacency(X))
    assert not np.allclose(g.adjacency, none.adjacency)
    assert np.allclose(g.standardizer.mean, X.mean(axis=(0, 2), keepdims=True)[0])
    assert not np.allclose(g.standardizer.mean, none.standardizer.mean)
    assert abl.model_config(abl.ARMS["graph"], 4).always_train_bn is False


def test_split_alone_changes_only_the_split():
    X, y = _data()
    n, t = X.shape[0], X.shape[2]
    s = _build("split", X, y)
    assert s.fold_id is None and s.fold.val is None
    # timepoint holdout: trials sit on both sides
    assert set(s.fold.train.trial_id) & set(s.fold.test.trial_id)
    assert len(s.fold.train) + len(s.fold.test) == n * t

    # ... but the graph and the standardiser use the TRAINING timepoints only
    train_idx, _ = leaky_timepoint_holdout(n, t, 0.1, SEED)
    flat = X.transpose(1, 0, 2).reshape(X.shape[1], -1)
    train_flat = flat[:, train_idx]
    assert np.allclose(s.fold.adjacency, pcc_adjacency(train_flat[None]))
    assert not np.allclose(s.fold.adjacency, pcc_adjacency(X))
    assert np.allclose(s.fold.standardizer.mean[:, 0], train_flat.mean(axis=1))
    assert not np.allclose(s.fold.standardizer.mean[:, 0], flat.mean(axis=1))
    assert abl.model_config(abl.ARMS["split"], 4).always_train_bn is False

    # same rows as `all`; they differ only in the graph/standardiser fit source
    a = _build("all", X, y).fold
    assert np.array_equal(a.train.trial_id, s.fold.train.trial_id)
    assert np.array_equal(a.test.trial_id, s.fold.test.trial_id)
    assert np.array_equal(a.train.y, s.fold.train.y)
    assert not np.allclose(a.adjacency, s.fold.adjacency)


def test_seed_fold_makes_each_fold_independent_of_process_history():
    """A resumed run must initialise a fold exactly as an uninterrupted run does."""
    import torch
    abl.seed_fold(42, "S002", 3)
    first = torch.rand(4)
    torch.rand(1000)                      # unrelated work in between (earlier folds)
    abl.seed_fold(42, "S002", 3)
    assert torch.equal(first, torch.rand(4))
    abl.seed_fold(42, "S002", 4)
    assert not torch.equal(first, torch.rand(4))
    abl.seed_fold(42, "S002", None)       # timepoint-holdout arms
    assert not torch.equal(first, torch.rand(4))
