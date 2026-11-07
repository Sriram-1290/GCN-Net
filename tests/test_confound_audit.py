"""Run-family confound audit (A3 / A4): task selection, fold integrity, logged metrics.

Synthetic data only. Each class (three channels each) or rest family (half of the channels) raises the
variance of its own channels, so the problem is separable by covariance, which is what the tangent-space
model reads. The signal spans several channels because the classifier is strongly
penalised (C=0.01) and needs more than a couple of informative dimensions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_confound_audit as audit  # noqa: E402
from data.cache import write_manifest  # noqa: E402
from data.config import load_config  # noqa: E402
from evaluation.experiment_log import read_experiments  # noqa: E402

FIST, FEET = (4, 8, 12), (6, 10, 14)
N_CH, N_T, PER_RUN_CLASS = 12, 256, 16


def _noise(rng, n, chans):
    x = rng.standard_normal((n, N_CH, N_T)).astype(np.float32)
    x[:, list(chans)] *= 5.0
    return x


def _trial_subject(rng):
    """Class 0/1 from fist runs, class 2/3 from feet runs, PER_RUN_CLASS each."""
    X, y, run = [], [], []
    for r in FIST + FEET:
        for c in ((0, 1) if r in FIST else (2, 3)):
            X.append(_noise(rng, PER_RUN_CLASS, range(3 * c, 3 * c + 3)))
            y += [c] * PER_RUN_CLASS
            run += [r] * PER_RUN_CLASS
    return np.concatenate(X), np.array(y), np.array(run)


def _rest_subject(rng):
    X, run = [], []
    for r in FIST + FEET:
        X.append(_noise(rng, 2 * PER_RUN_CLASS, range(0, 6) if r in FIST else range(6, 12)))
        run += [r] * (2 * PER_RUN_CLASS)
    return np.concatenate(X), np.array(run)


@pytest.fixture
def workspace(tmp_path):
    """Config files and synthetic caches for two subjects."""
    base = (ROOT / "configs" / "bandpass_8_30.yaml").read_text(encoding="utf-8")
    cfgs = {}
    for name in ("trial", "rest"):
        cache = tmp_path / f"cache_{name}"
        cache.mkdir()
        text = base.replace("cache_dir: data/cache_bp830", f"cache_dir: {cache.as_posix()}")
        path = tmp_path / f"{name}.yaml"
        path.write_text(text, encoding="utf-8")
        write_manifest(load_config(path), cache, [], [])
        cfgs[name] = (path, cache)
    rng = np.random.default_rng(0)
    for s in ("S001", "S002"):
        X, y, run = _trial_subject(rng)
        np.savez(cfgs["trial"][1] / f"{s}.npz", X=X, y=y, run=run,
                 onset=np.arange(len(y), dtype=float),
                 ch_names=np.array([f"C{i}" for i in range(N_CH)], dtype=object),
                 sfreq=np.float64(160.0))
        X, run = _rest_subject(rng)   # same layout as scripts/build_rest_cache.py: no sfreq
        np.savez(cfgs["rest"][1] / f"{s}.npz", X=X, run=run,
                 onset=np.arange(len(run), dtype=float),
                 ch_names=np.array([f"C{i}" for i in range(N_CH)], dtype=object))
    return tmp_path, cfgs


def _cli(cfgs, tmp, experiment, split, task=None, model="tgsp", extra=()):
    key = "rest" if experiment == "rest-family" else "trial"
    log = tmp / "log.jsonl"
    argv = ["--experiment", experiment, "--model", model, "--split", split,
            "--config", str(cfgs[key][0]), "--tag", "T", "--log", str(log),
            "--subjects", "S001", "S002", *extra]
    if task:
        argv += ["--task", task]
    return argv, log


# ---------------------------------------------------------------------- the three
def test_rh_vs_feet_task_draws_from_both_families():
    _, y, run = _trial_subject(np.random.default_rng(1))
    assert set(run[y == 1]) == set(FIST) and set(run[y == 3]) == set(FEET)

    idx, y01 = audit.select_task(y, "rh_vs_feet")
    assert set(np.unique(y01)) == {0, 1}
    assert set(run[idx][y01 == 0]) == set(FIST)    # class 1, right fist, relabelled 0
    assert set(run[idx][y01 == 1]) == set(FEET)    # class 3, both feet, relabelled 1
    assert (y[idx][y01 == 0] == 1).all() and (y[idx][y01 == 1] == 3).all()

    # Controls stay inside one family.
    i_lr, lr = audit.select_task(y, "lr")
    i_bf, bf = audit.select_task(y, "bf")
    assert set(run[i_lr]) == set(FIST) and set(run[i_bf]) == set(FEET)
    assert (y[i_lr][lr == 0] == 0).all() and (y[i_lr][lr == 1] == 1).all()
    assert (y[i_bf][bf == 0] == 2).all() and (y[i_bf][bf == 1] == 3).all()


@pytest.mark.parametrize("experiment,task", [
    ("rest-family", None), ("cross-family", "rh_vs_feet"),
    ("cross-family", "lr"), ("cross-family", "bf")])
def test_run_split_never_tests_on_training_runs(workspace, monkeypatch, experiment, task):
    tmp, cfgs = workspace
    calls = []
    real = audit.run_disjoint_folds

    def spy(runs, y, fold_runs):
        folds = real(runs, y, fold_runs)
        for f in folds:
            calls.append((set(runs[f.train]), set(runs[f.test]), tuple(fold_runs)))
        return folds

    monkeypatch.setattr(audit, "run_disjoint_folds", spy)
    argv, log = _cli(cfgs, tmp, experiment, "run", task)
    assert audit.main(argv) == 0

    assert len(calls) == 2 * 3                     # 2 subjects x 3 folds
    for train_runs, test_runs, _ in calls:
        assert test_runs and not (train_runs & test_runs)
    expected = {"rest-family": audit.RUN_PAIR_FOLDS, "rh_vs_feet": audit.RUN_PAIR_FOLDS,
                "lr": [(4,), (8,), (12,)], "bf": [(6,), (10,), (14,)]}[task or "rest-family"]
    assert [tuple(sorted(t)) for _, t, _ in calls[:3]] == [tuple(e) for e in expected]
    rows = read_experiments(log)
    assert [tuple(r["fold_test_runs"]) for r in rows[:3]] == [tuple(e) for e in expected]


@pytest.mark.parametrize("experiment,task,split", [
    ("rest-family", None, "trial"), ("rest-family", None, "run"),
    ("cross-family", "rh_vs_feet", "trial"), ("cross-family", "rh_vs_feet", "run"),
    ("cross-family", "lr", "run"), ("cross-family", "bf", "trial")])
def test_auc_and_accuracy_logged(workspace, experiment, task, split):
    tmp, cfgs = workspace
    argv, log = _cli(cfgs, tmp, experiment, split, task)
    assert audit.main(argv) == 0
    rows = read_experiments(log)
    n_folds = 5 if split == "trial" else 3
    assert len(rows) == 2 * n_folds
    for r in rows:
        assert r["accuracy"] > 0.9 and r["roc_auc"] > 0.9
        assert r["experiment"] == "confound_audit" and r["audit"] == experiment
        assert r["task"] == (task or "rest_family")
        assert r["model"] == "tgsp" and r["split"] == split and r["n_folds"] == n_folds
        assert r["n_train_trials"] > 0 and r["n_test_trials"] > 0
        assert sum(map(sum, r["trial_confusion"])) == r["n_test_trials"]
        assert r["seed"] == 42 and "git_commit" in r and "timestamp_utc" in r


# --------------------------------------------------------------- guards and resume
def test_task_required_for_cross_family_and_forbidden_for_rest(workspace):
    tmp, cfgs = workspace
    argv, _ = _cli(cfgs, tmp, "cross-family", "run")
    with pytest.raises(SystemExit):
        audit.main(argv)
    argv, _ = _cli(cfgs, tmp, "rest-family", "run", task="lr")
    with pytest.raises(SystemExit):
        audit.main(argv)


def test_trial_folds_are_disjoint_and_run_folds_come_from_run_numbers_only():
    _, y, run = _trial_subject(np.random.default_rng(2))
    idx, y01 = audit.select_task(y, "rh_vs_feet")
    for f in audit.build_folds(run[idx], y01, "trial", [], 42):
        assert not set(f.train) & set(f.test)
    # Changing the data cannot change run-fold membership.
    a = audit.build_folds(run[idx], y01, "run", audit.RUN_PAIR_FOLDS, 42)
    b = audit.build_folds(run[idx], y01[::-1].copy()[::-1], "run", audit.RUN_PAIR_FOLDS, 7)
    for fa, fb in zip(a, b):
        assert np.array_equal(fa.test, fb.test)


def test_resume_skips_logged_folds(workspace):
    tmp, cfgs = workspace
    argv, log = _cli(cfgs, tmp, "cross-family", "run", "lr")
    audit.main(argv)
    n = len(read_experiments(log))
    audit.main(argv + ["--resume"])
    assert len(read_experiments(log)) == n


def test_shallow_smoke_logs_trial_level_metrics(workspace):
    tmp, cfgs = workspace
    argv, log = _cli(cfgs, tmp, "rest-family", "run", model="shallow",
                     extra=("--epochs", "2"))
    argv[argv.index("S002"):argv.index("S002") + 1] = []   # one subject is enough
    assert audit.main(argv) == 0
    rows = read_experiments(log)
    assert len(rows) == 3
    for r in rows:
        assert 0.0 <= r["roc_auc"] <= 1.0 and r["model"] == "shallow"
        assert r["window"] == 64 and r["stride"] == 16 and r["epochs"] == 2
        assert r["n_val_trials"] > 0
        assert sum(map(sum, r["trial_confusion"])) == r["n_test_trials"]
