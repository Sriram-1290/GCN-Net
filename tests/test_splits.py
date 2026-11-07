"""Tests for split correctness — the leakage guardrails.

Every test here corresponds to a rule in CLAUDE.md's Data Leakage section. The
leaky-split tests are inverted: they assert that the reference protocol *does*
leak, which is what makes the replication arm a measurement rather than a claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.splits import (
    RUN_PAIR_FOLDS,
    count_shared_trials,
    leaky_timepoint_holdout,
    loso_folds,
    run_disjoint_folds,
    stratified_trial_kfold,
)

SUBJECTS = [f"S{i:03d}" for i in range(1, 21)]


def _labels(n_per_class: int = 22, n_classes: int = 4) -> np.ndarray:
    return np.repeat(np.arange(n_classes), n_per_class)


# --------------------------------------------------------------------------
# trial-level, subject-dependent
# --------------------------------------------------------------------------

def test_trial_kfold_never_shares_a_trial():
    y = _labels()
    for split in stratified_trial_kfold(y, n_splits=5, seed=0):
        assert np.intersect1d(split.train, split.test).size == 0


def test_trial_kfold_covers_every_trial_exactly_once_across_folds():
    y = _labels()
    folds = stratified_trial_kfold(y, n_splits=5, seed=0)
    covered = np.concatenate([f.test for f in folds])
    assert np.array_equal(np.sort(covered), np.arange(len(y)))


def test_trial_kfold_preserves_class_balance():
    y = _labels()
    for split in stratified_trial_kfold(y, n_splits=4, seed=0):
        counts = np.bincount(y[split.test], minlength=4)
        assert counts.max() - counts.min() <= 1


def test_trial_kfold_is_reproducible_under_the_same_seed():
    y = _labels()
    a = stratified_trial_kfold(y, n_splits=5, seed=7)
    b = stratified_trial_kfold(y, n_splits=5, seed=7)
    for fa, fb in zip(a, b):
        assert np.array_equal(fa.test, fb.test)


def test_trial_kfold_rejects_2d_input():
    with pytest.raises(ValueError):
        stratified_trial_kfold(np.zeros((10, 2)))


# --------------------------------------------------------------------------
# LOSO
# --------------------------------------------------------------------------

def test_loso_has_one_fold_per_subject():
    folds = loso_folds(SUBJECTS, n_val_subjects=3)
    assert len(folds) == len(SUBJECTS)
    assert sorted(f.test[0] for f in folds) == sorted(SUBJECTS)


def test_loso_test_subject_appears_in_no_other_split():
    for fold in loso_folds(SUBJECTS, n_val_subjects=3):
        subject = fold.test[0]
        assert subject not in fold.train
        assert subject not in fold.val


def test_loso_validation_is_disjoint_from_training():
    for fold in loso_folds(SUBJECTS, n_val_subjects=4):
        assert not set(fold.train) & set(fold.val)


def test_loso_partitions_all_subjects():
    for fold in loso_folds(SUBJECTS, n_val_subjects=3):
        assert set(fold.train) | set(fold.val) | set(fold.test) == set(SUBJECTS)


def test_loso_validation_sets_differ_across_folds():
    """Identical val subjects everywhere would bias model selection."""
    folds = loso_folds(SUBJECTS, n_val_subjects=3, seed=1)
    assert len({f.val for f in folds}) > 1


def test_loso_rejects_impossible_validation_size():
    with pytest.raises(ValueError):
        loso_folds(SUBJECTS[:3], n_val_subjects=3)


# --------------------------------------------------------------------------
# the reference's leaky split — asserted to leak
# --------------------------------------------------------------------------

def test_leaky_split_puts_the_same_trial_on_both_sides():
    n_trials, n_samples = 90, 640
    train, test = leaky_timepoint_holdout(n_trials, n_samples, test_frac=0.1, seed=0)
    shared = count_shared_trials(train, test, n_samples)
    # With 640 timepoints per trial and a 10% test cut, essentially every trial
    # contributes to both sides. This is the reference protocol.
    assert shared == n_trials


def test_leaky_split_sizes_match_the_reference_ratio():
    n_trials, n_samples = 50, 640
    train, test = leaky_timepoint_holdout(n_trials, n_samples, test_frac=0.1, seed=0)
    total = n_trials * n_samples
    assert len(train) + len(test) == total
    assert abs(len(test) / total - 0.1) < 0.01


def test_correct_trial_split_shares_no_trials_when_expanded_to_timepoints():
    """The contrast: a trial-level split stays clean after expanding to timepoints."""
    n_samples = 640
    y = _labels()
    split = stratified_trial_kfold(y, n_splits=5, seed=0)[0]
    expand = lambda idx: (idx[:, None] * n_samples + np.arange(n_samples)).ravel()
    assert count_shared_trials(expand(split.train), expand(split.test), n_samples) == 0


# --------------------------------------------------------------------------
# LOSO cost estimation (guards against launching an infeasible run)
# --------------------------------------------------------------------------

def test_sample_estimate_matches_the_reference_formulation():
    from scripts.run_loso import estimate_samples
    # window=1 is the reference's one-timepoint sample: trials x timepoints
    assert estimate_samples(8730, 640, window=1, stride=1, features="raw") == 8730 * 640


def test_sample_estimate_counts_windows_not_timepoints():
    from scripts.run_loso import estimate_samples
    # (640 - 64) // 32 + 1 = 19 windows per trial
    assert estimate_samples(100, 640, window=64, stride=32, features="raw") == 1900


def test_windowing_brings_loso_inside_the_budget_and_timepoints_do_not():
    """The check that stops an infeasible LOSO run from looking like a hang."""
    from scripts.run_loso import SAMPLE_BUDGET, estimate_samples
    timepoint = estimate_samples(8730, 640, 1, 1, "raw")
    windowed = estimate_samples(8730, 640, 64, 64, "raw")
    assert timepoint > SAMPLE_BUDGET      # the reference formulation is hopeless
    assert windowed <= SAMPLE_BUDGET      # windowing makes it feasible


# --------------------------------------------------------------------------
# run-disjoint folds (Phase A)
# --------------------------------------------------------------------------

def _run_labels() -> tuple[np.ndarray, np.ndarray]:
    """Six imagery runs with 6 trials each, both classes in every run."""
    runs = np.repeat([4, 6, 8, 10, 12, 14], 6)
    y = np.tile([0, 1], 18)
    return runs, y


def test_run_disjoint_folds_are_disjoint():
    runs, y = _run_labels()
    folds = run_disjoint_folds(runs, y, RUN_PAIR_FOLDS)
    assert len(folds) == len(RUN_PAIR_FOLDS) == 3
    for fold, held_out in zip(folds, RUN_PAIR_FOLDS):
        test_runs, train_runs = set(runs[fold.test]), set(runs[fold.train])
        assert test_runs == set(held_out)
        assert not test_runs & train_runs
        assert len(fold.train) + len(fold.test) == len(y)
    # Every trial is tested exactly once across the folds.
    tested = np.concatenate([f.test for f in folds])
    assert sorted(tested.tolist()) == list(range(len(y)))
    assert [f.fold for f in folds] == [0, 1, 2]


def test_run_disjoint_folds_reject_missing_class():
    runs, y = _run_labels()
    y[runs == 12] = 0
    y[runs == 14] = 0          # the (12, 14) test set now holds one class only
    with pytest.raises(ValueError, match="class"):
        run_disjoint_folds(runs, y, RUN_PAIR_FOLDS)


def test_run_disjoint_folds_reject_class_missing_from_train():
    runs = np.repeat([4, 6, 8], 4)
    y = np.array([0, 1, 0, 1,  1, 1, 1, 1,  1, 1, 1, 1])
    # Class 0 lives only in run 4; holding run 4 out leaves train without it.
    with pytest.raises(ValueError, match="class"):
        run_disjoint_folds(runs, y, [(4,)])
