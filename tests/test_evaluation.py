"""Tests for metrics, fold assembly and the trial-aggregation logic."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import (
    aggregate_to_trials,
    cohen_kappa,
    confusion_matrix,
    evaluate,
    macro_f1,
    paired_bootstrap_ci,
)
from evaluation.pipeline import Trials, concat_trials, prepare_fold, to_timepoint_samples


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_perfect_prediction_scores_one():
    y = np.array([0, 1, 2, 3, 0, 1])
    r = evaluate(y, y, 4)
    assert r.accuracy == 1.0
    assert r.macro_f1 == pytest.approx(1.0)
    assert r.kappa == pytest.approx(1.0)


def test_confusion_rows_are_true_labels():
    cm = confusion_matrix(np.array([0, 0, 1]), np.array([0, 1, 1]), 2)
    assert cm.tolist() == [[1, 1], [0, 1]]


def test_kappa_is_zero_for_chance_agreement():
    # predict one class always, on a balanced 2-class problem
    y = np.array([0, 1] * 50)
    cm = confusion_matrix(y, np.zeros_like(y), 2)
    assert cohen_kappa(cm) == pytest.approx(0.0, abs=1e-9)


def test_per_class_accuracy_tracks_support():
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    r = evaluate(y_true, y_pred, 2)
    assert r.per_class_accuracy[0] == pytest.approx(2 / 3)
    assert r.per_class_accuracy[1] == pytest.approx(1.0)


def test_macro_f1_ignores_class_frequency():
    cm = np.array([[90, 10], [5, 5]])
    assert 0.0 < macro_f1(cm) < 1.0


def test_bootstrap_ci_brackets_a_real_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(0.70, 0.05, 40)
    b = a - 0.04
    mean, lo, hi = paired_bootstrap_ci(a, b, n_boot=2000, seed=1)
    assert mean == pytest.approx(0.04, abs=1e-9)
    assert lo > 0  # the improvement is significant


def test_bootstrap_ci_includes_zero_when_there_is_no_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(0.7, 0.05, 40)
    b = rng.normal(0.7, 0.05, 40)
    _, lo, hi = paired_bootstrap_ci(a, b, n_boot=2000, seed=1)
    assert lo < 0 < hi


# --------------------------------------------------------------------------
# trial aggregation
# --------------------------------------------------------------------------

def test_aggregation_recovers_the_trial_label_from_noisy_timepoints():
    """Most timepoints wrong individually, but the trial vote is still right."""
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 0.1, size=(100, 4))
    logits[:, 2] += 0.15  # a weak but consistent lean toward class 2
    trial_id = np.zeros(100, dtype=np.int64)
    _, pred = aggregate_to_trials(logits, trial_id)
    assert pred.tolist() == [2]


def test_majority_and_mean_logit_can_be_requested():
    logits = np.array([[5.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    tid = np.zeros(3, dtype=np.int64)
    # majority: class 1 wins 2 votes to 1
    assert aggregate_to_trials(logits, tid, method="majority")[1].tolist() == [1]
    # mean logit: class 0's single large logit dominates the average
    assert aggregate_to_trials(logits, tid, method="mean_logit")[1].tolist() == [0]


def test_aggregation_keeps_trials_separate():
    logits = np.array([[9.0, 0.0], [9.0, 0.0], [0.0, 9.0], [0.0, 9.0]])
    tid = np.array([7, 7, 9, 9])
    trials, pred = aggregate_to_trials(logits, tid)
    assert trials.tolist() == [7, 9]
    assert pred.tolist() == [0, 1]


def test_unknown_aggregation_method_raises():
    with pytest.raises(ValueError):
        aggregate_to_trials(np.zeros((2, 2)), np.zeros(2, dtype=np.int64), method="nope")


# --------------------------------------------------------------------------
# fold assembly
# --------------------------------------------------------------------------

def _trials(n=8, ch=64, t=64, seed=0) -> Trials:
    rng = np.random.default_rng(seed)
    return Trials(X=rng.normal(size=(n, ch, t)).astype(np.float32),
                  y=np.tile(np.arange(4), n // 4).astype(np.int64))


def test_timepoint_expansion_shapes_and_labels():
    tr = _trials(n=4, ch=8, t=10)
    s = to_timepoint_samples(tr.X, tr.y)
    assert s.X.shape == (40, 8, 1)
    assert len(s.y) == 40 and len(s.trial_id) == 40
    # every timepoint of trial 0 carries trial 0's label
    assert set(s.y[s.trial_id == 0]) == {tr.y[0]}


def test_expansion_preserves_the_channel_vector_at_each_timepoint():
    X = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)  # trials, ch, time
    s = to_timepoint_samples(X, np.array([0, 1]))
    # sample 0 is trial 0 at t=0: the channel vector X[0, :, 0]
    assert np.allclose(s.X[0, :, 0], X[0, :, 0])
    assert np.allclose(s.X[1, :, 0], X[0, :, 1])


def test_prepare_fold_builds_the_graph_from_training_data_only():
    train, test = _trials(seed=1), _trials(seed=2)
    honest = prepare_fold(train, test, leaky=False)
    leaky = prepare_fold(train, test, leaky=True)
    assert not np.allclose(honest.adjacency, leaky.adjacency)
    assert honest.leaky is False and leaky.leaky is True


def test_prepare_fold_standardiser_sees_only_training_trials():
    train, test = _trials(seed=1), _trials(seed=2)
    fold = prepare_fold(train, test)
    assert fold.standardizer.n_trials_fit == len(train)


def test_prepare_fold_produces_a_usable_hierarchy():
    fold = prepare_fold(_trials(seed=1), _trials(seed=2))
    assert len(fold.bases) == 6
    assert [b.shape[1] for b in fold.bases] == [64, 32, 16, 8, 4, 2]
    assert fold.train.X.shape[1] == 64


def test_train_and_test_trial_ids_never_collide():
    fold = prepare_fold(_trials(seed=1), _trials(seed=2), val=_trials(seed=3))
    assert not set(fold.train.trial_id) & set(fold.test.trial_id)
    assert not set(fold.train.trial_id) & set(fold.val.trial_id)
    assert not set(fold.val.trial_id) & set(fold.test.trial_id)


def test_concat_trials_pools_subjects():
    pooled = concat_trials([_trials(n=4, seed=1), _trials(n=8, seed=2)])
    assert len(pooled) == 12


def test_trials_rejects_mismatched_labels():
    with pytest.raises(ValueError):
        Trials(X=np.zeros((3, 4, 5)), y=np.zeros(2))


# --------------------------------------------------------------------------
# the leaky replication fold — asserted to leak
# --------------------------------------------------------------------------

def test_leaky_fold_puts_every_trial_on_both_sides():
    from evaluation.pipeline import prepare_leaky_fold
    tr = _trials(n=8, ch=64, t=64)
    fold = prepare_leaky_fold(tr, test_frac=0.1, seed=0)
    shared = set(fold.train.trial_id) & set(fold.test.trial_id)
    assert len(shared) == len(tr), "the reference protocol must leak every trial"
    assert fold.leaky is True
    assert fold.val is None


def test_leaky_fold_splits_timepoints_not_trials():
    from evaluation.pipeline import prepare_leaky_fold
    tr = _trials(n=8, ch=64, t=64)
    fold = prepare_leaky_fold(tr, test_frac=0.1, seed=0)
    total = 8 * 64
    assert len(fold.train) + len(fold.test) == total
    assert abs(len(fold.test) / total - 0.1) < 0.02


def test_leaky_fold_graph_uses_all_trials():
    from evaluation.pipeline import prepare_leaky_fold
    from graphs.adjacency import pcc_adjacency
    tr = _trials(n=8, ch=64, t=64)
    fold = prepare_leaky_fold(tr, seed=0)
    assert np.allclose(fold.adjacency, pcc_adjacency(tr.X))
