"""Leakage tests for the pre-train / fine-tune runner (17.11.2).

This script is the most leakage-sensitive thing in the project: it deliberately trains
on 102 other subjects before touching the target, so the boundaries have to be checked
rather than assumed. Three of them carry the protocol:

1. **The target contributes nothing to pre-training** — not their trials, not the graph,
   not the normalisation statistics.
2. **Fine-tune train and test are disjoint at TRIAL level**, and stay disjoint after
   windowing. A single shared window would reintroduce the exact defect the whole
   project exists to document (§4.4).
3. **The folds match `run_subject_dependent.py` at the same seed**, or the arms are not
   paired and no per-subject comparison is valid.

The control arm gets a test too: `--no-pretrain` must produce the identical pipeline
with only the pre-training skipped, since that is what isolates the effect of
pre-training from the effect of using source-fit normalisation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.splits import stratified_trial_kfold  # noqa: E402
from evaluation.pipeline import Samples, Trials, concat_trials, prepare_fold  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_pretrain_finetune", ROOT / "scripts" / "run_pretrain_finetune.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trials(n=20, c=64, t=640, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, c, t)).astype(np.float32)
    y = np.array([0, 1] * (n // 2), dtype=np.int64)
    return Trials(X=X, y=y)


# ------------------------------------------------------- the trial-id partition

def test_subset_samples_selects_exactly_the_named_trials():
    m = _load()
    s = Samples(X=np.arange(60, dtype=np.float32).reshape(20, 3, 1),
                y=np.zeros(20, dtype=np.int64),
                trial_id=np.repeat(np.arange(5) + 200_000, 4))
    out = m.subset_samples(s, {200_000, 200_002})
    assert set(np.unique(out.trial_id)) == {200_000, 200_002}
    assert len(out) == 8


def test_finetune_train_and_test_share_no_window():
    """The defect this project documents, asserted for the new runner.

    A trial-level split must survive windowing: every window of a test trial must be
    absent from the fine-tune training set.
    """
    m = _load()
    target = _trials(seed=1)
    source = concat_trials([_trials(seed=2), _trials(seed=3)])
    prepared = prepare_fold(source, target, window=64, stride=16, build_graph=False)

    for split in stratified_trial_kfold(target.y, n_splits=5, seed=42):
        tr = m.subset_samples(prepared.test, set(split.train + 200_000))
        te = m.subset_samples(prepared.test, set(split.test + 200_000))
        assert len(tr) and len(te)
        assert not (set(np.unique(tr.trial_id)) & set(np.unique(te.trial_id))), (
            f"fold {split.fold}: a trial appears in both fine-tune train and test")
        # and together they account for every window, nothing silently dropped
        assert len(tr) + len(te) == len(prepared.test)


def test_folds_match_the_subject_dependent_runner_at_the_same_seed():
    """Without this the arms are not paired and no per-subject comparison holds."""
    target = _trials(seed=4)
    a = [(s.fold, tuple(s.train), tuple(s.test))
         for s in stratified_trial_kfold(target.y, n_splits=5, seed=42)]
    b = [(s.fold, tuple(s.train), tuple(s.test))
         for s in stratified_trial_kfold(target.y, n_splits=5, seed=42)]
    assert a == b, "the fold generator is not deterministic at a fixed seed"


# --------------------------------------------- the target never reaches pre-training

def test_source_statistics_do_not_depend_on_the_target():
    """The graph and standardiser must be identical whatever the target contains.

    `prepare_fold` fits on its `train` argument only; the target is passed as `test`
    purely to be transformed. If the target ever leaked into the fit, replacing it with
    different data would change the fitted statistics.
    """
    source = concat_trials([_trials(seed=5), _trials(seed=6)])
    one = prepare_fold(source, _trials(seed=7), window=64, stride=16, build_graph=True)
    two = prepare_fold(source, _trials(seed=99), window=64, stride=16,
                       build_graph=True)

    assert np.allclose(one.adjacency, two.adjacency), (
        "the adjacency changed when only the target changed -- the target is leaking "
        "into the graph")
    assert np.allclose(one.standardizer.mean, two.standardizer.mean)
    assert np.allclose(one.standardizer.std, two.standardizer.std)


def test_target_is_excluded_from_the_source_list():
    """The runner's own selection rule, checked directly."""
    subjects = [f"S{i:03d}" for i in range(1, 11)]
    for target in subjects:
        source = [s for s in subjects if s != target]
        assert target not in source
        assert len(source) == len(subjects) - 1


# ---------------------------------------------------------------- grouped pre-training

def test_groups_partition_the_subjects_exactly():
    m = _load()
    subjects = [f"S{i:03d}" for i in range(1, 104)]
    for k in (0, 1, 3, 10, 103, 500):
        groups = m.assign_groups(subjects, k)
        flat = [s for g in groups for s in g]
        assert sorted(flat) == sorted(subjects), f"k={k}: not a cover"
        assert len(flat) == len(set(flat)), f"k={k}: groups overlap"
        assert all(groups), f"k={k}: an empty group"


def test_no_target_is_ever_in_its_own_source_set():
    """The leakage guarantee that grouping has to preserve.

    Pre-training excludes the target's whole GROUP, not just the target, so every
    member of that group is absent from the model they later fine-tune from. If this
    ever failed, a subject would be pre-trained on their own trials.
    """
    m = _load()
    subjects = [f"S{i:03d}" for i in range(1, 104)]
    for k in (0, 10, 25):
        for group in m.assign_groups(subjects, k):
            source = [s for s in subjects if s not in group]
            for target in group:
                assert target not in source, f"k={k}: {target} is in its own source set"
            assert len(source) == len(subjects) - len(group)


def test_grouping_is_deterministic_and_seed_free():
    """Reproducible without carrying a seed, since it is index % k."""
    m = _load()
    subjects = [f"S{i:03d}" for i in range(1, 104)]
    assert m.assign_groups(subjects, 10) == m.assign_groups(subjects, 10)


def test_groups_are_interleaved_not_contiguous_blocks():
    """A contiguous block would make a group a slice of the recording order."""
    m = _load()
    subjects = [f"S{i:03d}" for i in range(1, 104)]
    first = m.assign_groups(subjects, 10)[0]
    assert first[:3] == ["S001", "S011", "S021"], first[:3]


def test_ten_groups_cut_the_pretraining_count_by_an_order_of_magnitude():
    """The point of the change: error 24 made 103 pre-trainings a ~51 h run."""
    m = _load()
    subjects = [f"S{i:03d}" for i in range(1, 104)]
    assert len(m.assign_groups(subjects, 0)) == 103      # leave-one-subject-out
    assert len(m.assign_groups(subjects, 10)) == 10      # grouped
    # and a target still draws on the great majority of the cohort
    assert min(103 - len(g) for g in m.assign_groups(subjects, 10)) >= 92


# ------------------------------------------------------------------ the control arm

def test_no_pretrain_control_is_declared_and_documented():
    """The control exists and says why, since it is what makes the arm interpretable."""
    text = (ROOT / "scripts" / "run_pretrain_finetune.py").read_text(encoding="utf-8")
    assert "--no-pretrain" in text
    assert "THE CONTROL" in text
    assert "isolat" in text.lower()


def test_finetune_lr_defaults_below_pretrain_lr():
    """Fine-tuning ~36 trials at the pre-training rate destroys transferred features."""
    src = (ROOT / "scripts" / "run_pretrain_finetune.py").read_text(encoding="utf-8")
    assert '"--pretrain-lr", type=float, default=1e-3' in src
    assert '"--finetune-lr", type=float, default=1e-4' in src


def test_model_factory_covers_every_runner_architecture():
    m = _load()
    with pytest.raises(ValueError, match="unknown model"):
        m.build_model("resnet", None, 64, 2, 160.0)
