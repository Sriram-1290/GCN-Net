"""Tests for metrics, fold assembly and the trial-aggregation logic."""

from __future__ import annotations

from pathlib import Path

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


# --------------------------------------------------------------------------
# window-based samples
# --------------------------------------------------------------------------

def test_windows_never_cross_a_trial_boundary():
    from evaluation.pipeline import to_window_samples
    X = np.arange(3 * 2 * 10, dtype=np.float32).reshape(3, 2, 10)
    s = to_window_samples(X, np.array([0, 1, 2]), window=5, stride=5)
    assert s.X.shape == (6, 2, 5)          # 3 trials x 2 windows
    # window 0 is trial 0's first half; its content must come only from trial 0
    assert np.allclose(s.X[0], X[0, :, 0:5])
    assert np.allclose(s.X[1], X[0, :, 5:10])
    assert np.allclose(s.X[2], X[1, :, 0:5])
    assert s.trial_id.tolist() == [0, 0, 1, 1, 2, 2]


def test_window_samples_inherit_the_trial_label():
    from evaluation.pipeline import to_window_samples
    X = np.zeros((2, 3, 8), dtype=np.float32)
    s = to_window_samples(X, np.array([3, 1]), window=4, stride=2)
    assert set(s.y[s.trial_id == 0]) == {3}
    assert set(s.y[s.trial_id == 1]) == {1}


def test_overlapping_windows_produce_more_samples():
    from evaluation.pipeline import to_window_samples
    X = np.zeros((1, 2, 100), dtype=np.float32)
    y = np.array([0])
    assert len(to_window_samples(X, y, window=20, stride=20)) == 5
    assert len(to_window_samples(X, y, window=20, stride=10)) == 9


def test_window_start_offsets_windows():
    from evaluation.pipeline import to_window_samples
    T = 400
    X = np.random.default_rng(1).normal(size=(2, 3, T)).astype(np.float32)
    s = to_window_samples(X, np.array([0, 1]), 64, 16, start=160)
    n = len(range(160, T - 64 + 1, 16))
    assert s.X.shape == (2 * n, 3, 64)
    assert np.array_equal(s.X[0], X[0, :, 160:224])         # first window at sample 160
    assert np.array_equal(s.X[n], X[1, :, 160:224])         # trial 1 starts there too
    last = 160 + 16 * (n - 1)
    assert np.array_equal(s.X[n - 1], X[0, :, last:last + 64])


def test_window_start_default_unchanged():
    from evaluation.pipeline import to_window_samples
    X = np.random.default_rng(2).normal(size=(3, 4, 300)).astype(np.float32)
    y = np.array([0, 1, 0])
    a = to_window_samples(X, y, 64, 16)
    b = to_window_samples(X, y, 64, 16, start=0)
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.y, b.y)
    assert np.array_equal(a.trial_id, b.trial_id)
    assert len(a) == 3 * len(range(0, 300 - 64 + 1, 16))


def test_window_start_must_leave_room_for_a_window():
    from evaluation.pipeline import to_window_samples
    X = np.zeros((1, 2, 100), dtype=np.float32)
    with pytest.raises(ValueError):
        to_window_samples(X, np.array([0]), 64, 16, start=40)
    with pytest.raises(ValueError):
        to_window_samples(X, np.array([0]), 64, 16, start=-1)


def test_prepare_fold_window_start_reaches_train_val_and_test():
    rng = np.random.default_rng(3)

    def trials(n):
        return Trials(X=rng.normal(size=(n, 8, 200)).astype(np.float32),
                      y=np.arange(n) % 2)

    tr, va, te = trials(6), trials(4), trials(4)
    kw = dict(build_graph=False, window=64, stride=16)
    base = prepare_fold(tr, te, va, **kw)
    same = prepare_fold(tr, te, va, window_start=0, **kw)
    late = prepare_fold(tr, te, va, window_start=80, **kw)
    for a, b in ((base.train, same.train), (base.val, same.val), (base.test, same.test)):
        assert np.array_equal(a.X, b.X)
    n_late = len(range(80, 200 - 64 + 1, 16))
    for split, n_trials in ((late.train, 6), (late.val, 4), (late.test, 4)):
        assert len(split) == n_trials * n_late
    # The test windows begin at the offset too, not only the training windows.
    std = late.standardizer.apply(te.X)
    assert np.allclose(late.test.X[0], std[0, :, 80:144])


def test_window_of_one_matches_the_timepoint_expansion():
    from evaluation.pipeline import to_timepoint_samples, to_window_samples
    X = np.random.default_rng(0).normal(size=(2, 4, 6)).astype(np.float32)
    y = np.array([0, 1])
    a = to_timepoint_samples(X, y)
    b = to_window_samples(X, y, window=1, stride=1)
    assert np.allclose(a.X, b.X)
    assert np.array_equal(a.y, b.y) and np.array_equal(a.trial_id, b.trial_id)


def test_window_rejects_impossible_sizes():
    from evaluation.pipeline import to_window_samples
    X = np.zeros((1, 2, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        to_window_samples(X, np.array([0]), window=20, stride=1)
    with pytest.raises(ValueError):
        to_window_samples(X, np.array([0]), window=5, stride=0)


def test_prepare_fold_with_windows_sets_the_feature_dimension():
    fold = prepare_fold(_trials(n=8, ch=64, t=64), _trials(n=8, ch=64, t=64, seed=2),
                        window=16, stride=16)
    assert fold.train.X.shape[1:] == (64, 16)


# --------------------------------------------------------------------------
# multiple-comparison correction (pre-registration 10.4)
# --------------------------------------------------------------------------

def test_bootstrap_p_is_small_for_a_real_effect():
    from evaluation.metrics import bootstrap_p_value
    rng = np.random.default_rng(0)
    a = rng.normal(0.70, 0.03, 40)
    assert bootstrap_p_value(a, a - 0.05, "greater") < 0.01


def test_bootstrap_p_is_calibrated_under_the_null():
    """The property that matters: a controlled false-positive rate.

    Asserting one seed's p-value is large only tests that seed's luck -- two
    independent samples of 40 differ by ~1.8 SE often enough. What must hold is
    that across many nulls, about 5 % of p-values fall below 0.05.
    """
    from evaluation.metrics import bootstrap_p_value
    ps = []
    for s in range(200):
        r = np.random.default_rng(s)
        ps.append(bootstrap_p_value(r.normal(0, 1, 40), r.normal(0, 1, 40),
                                    "two-sided", n_boot=2000, seed=s))
    ps = np.array(ps)
    assert 0.01 < (ps < 0.05).mean() < 0.12, (ps < 0.05).mean()
    assert 0.35 < np.median(ps) < 0.65, np.median(ps)


def test_bootstrap_p_respects_direction():
    """A one-sided test in the wrong direction must not be significant."""
    from evaluation.metrics import bootstrap_p_value
    rng = np.random.default_rng(0)
    a = rng.normal(0.70, 0.03, 40)
    b = a + 0.05                      # b is genuinely larger
    assert bootstrap_p_value(a, b, "greater") > 0.9
    assert bootstrap_p_value(a, b, "less") < 0.01


def test_bootstrap_p_never_returns_zero():
    from evaluation.metrics import bootstrap_p_value
    a = np.full(30, 1.0)
    assert bootstrap_p_value(a, np.zeros(30), "greater", n_boot=1000) >= 1 / 1000


def test_holm_is_stricter_than_uncorrected():
    from evaluation.metrics import holm_bonferroni
    out = holm_bonferroni({"H1": 0.04, "H2": 0.04, "H3": 0.04, "H4": 0.04, "H5": 0.04})
    # each would pass alone at 0.05; none survives the correction
    assert not any(v["reject"] for v in out.values())


def test_holm_keeps_a_strong_result():
    from evaluation.metrics import holm_bonferroni
    out = holm_bonferroni({"H1": 0.0001, "H2": 0.4, "H3": 0.6})
    assert out["H1"]["reject"] and not out["H2"]["reject"]


def test_holm_steps_down_once_a_test_fails():
    """A later test cannot be rejected after an earlier one fails."""
    from evaluation.metrics import holm_bonferroni
    out = holm_bonferroni({"a": 0.001, "b": 0.30, "c": 0.31})
    assert out["a"]["reject"]
    assert not out["b"]["reject"] and not out["c"]["reject"]


def test_holm_thresholds_step_correctly():
    from evaluation.metrics import holm_bonferroni
    out = holm_bonferroni({"a": 0.001, "b": 0.002, "c": 0.003}, alpha=0.06)
    assert out["a"]["threshold"] == pytest.approx(0.02)
    assert out["b"]["threshold"] == pytest.approx(0.03)
    assert out["c"]["threshold"] == pytest.approx(0.06)


def test_holm_handles_an_empty_family():
    from evaluation.metrics import holm_bonferroni
    assert holm_bonferroni({}) == {}


# --------------------------------------------------------------------------
# resumability: the experiment log is its own checkpoint
# --------------------------------------------------------------------------

def test_completed_keys_reads_back_what_was_logged(tmp_path):
    from evaluation.experiment_log import completed_keys, log_experiment
    p = tmp_path / "log.jsonl"
    for fold in range(3):
        log_experiment({"tag": "t1", "subject": "S001", "fold": fold}, path=p)
    log_experiment({"tag": "other", "subject": "S002", "fold": 0}, path=p)
    assert completed_keys(p, "t1") == {("S001", 0), ("S001", 1), ("S001", 2)}


def test_completed_keys_isolates_by_tag(tmp_path):
    """Resuming one run must not skip folds belonging to a different run."""
    from evaluation.experiment_log import completed_keys, log_experiment
    p = tmp_path / "log.jsonl"
    log_experiment({"tag": "a", "subject": "S001", "fold": 0}, path=p)
    log_experiment({"tag": "b", "subject": "S001", "fold": 0}, path=p)
    assert completed_keys(p, "a") == {("S001", 0)}
    assert completed_keys(p, "c") == set()


def test_completed_keys_survives_a_truncated_final_line(tmp_path):
    """A crash mid-write must not make the whole run unresumable."""
    from evaluation.experiment_log import completed_keys, log_experiment
    p = tmp_path / "log.jsonl"
    log_experiment({"tag": "t", "subject": "S001", "fold": 0}, path=p)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write('{"tag": "t", "subject": "S00')      # torn write
    assert completed_keys(p, "t") == {("S001", 0)}


def test_completed_keys_on_a_missing_log_is_empty(tmp_path):
    from evaluation.experiment_log import completed_keys
    assert completed_keys(tmp_path / "nope.jsonl", "t") == set()


def test_describe_resume_says_what_it_found():
    from evaluation.experiment_log import describe_resume
    assert "starting from the beginning" in describe_resume(set(), 10, "t")
    assert "3 of 10" in describe_resume({("S001", 0), ("S001", 1), ("S002", 0)}, 10, "t")


def test_find_duplicates_is_empty_on_a_clean_log(tmp_path):
    from evaluation.experiment_log import find_duplicates, log_experiment
    p = tmp_path / "log.jsonl"
    for fold in range(3):
        log_experiment({"tag": "t", "subject": "S001", "fold": fold}, path=p)
    assert find_duplicates(p) == {}


def test_find_duplicates_catches_a_repeated_fold(tmp_path):
    from evaluation.experiment_log import find_duplicates, log_experiment
    p = tmp_path / "log.jsonl"
    log_experiment({"tag": "t", "subject": "S001", "fold": 0}, path=p)
    log_experiment({"tag": "t", "subject": "S001", "fold": 0}, path=p)
    assert find_duplicates(p) == {"t": 1}


def test_two_rows_per_subject_are_not_duplicates_when_the_problem_differs(tmp_path):
    """The CSP baseline writes one row per binary problem; that is correct."""
    from evaluation.experiment_log import find_duplicates, log_experiment
    p = tmp_path / "log.jsonl"
    log_experiment({"tag": "csp", "subject": "S001", "problem": "left_vs_right"}, path=p)
    log_experiment({"tag": "csp", "subject": "S001", "problem": "fists_vs_feet"}, path=p)
    assert find_duplicates(p) == {}


def test_a_sweep_over_learning_rates_is_not_duplicated(tmp_path):
    from evaluation.experiment_log import find_duplicates, log_experiment
    p = tmp_path / "log.jsonl"
    for lr in (1e-3, 1e-2):
        log_experiment({"tag": "sweep", "subject": "S001", "fold": 0,
                        "train_config": {"learning_rate": lr}}, path=p)
    assert find_duplicates(p) == {}


def test_concurrent_appends_never_tear_a_row(tmp_path):
    """Several runners append to one log at once; rows are ~3 KB, past any
    platform's atomic-write guarantee. This happened for real: two processes wrote
    in the same second and left a 16-byte orphan, losing a fold's row."""
    import json
    import subprocess
    import sys
    import textwrap

    log = tmp_path / "concurrent.jsonl"
    script = tmp_path / "writer.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, r"{Path(__file__).resolve().parent.parent}")
        from evaluation.experiment_log import log_experiment
        for i in range(25):
            log_experiment({{"tag": sys.argv[1], "subject": f"S{{i:03d}}", "fold": i,
                            "payload": list(range(200))}}, path=r"{log}")
    """), encoding="utf-8")

    procs = [subprocess.Popen([sys.executable, str(script), f"w{i}"]) for i in range(4)]
    for p in procs:
        assert p.wait() == 0

    lines = [l for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 100, f"expected 100 rows, got {len(lines)}"
    for l in lines:
        json.loads(l)          # raises if any row was torn


def test_completed_keys_separates_window_starts(tmp_path):
    """A resumed run at one window start must not skip another start's folds."""
    from evaluation.experiment_log import completed_keys, log_experiment
    p = tmp_path / "log.jsonl"
    log_experiment({"tag": "t", "subject": "S001", "fold": 0}, path=p)
    log_experiment({"tag": "t", "subject": "S002", "fold": 0, "window_start": 160}, path=p)
    assert completed_keys(p, "t") == {("S001", 0), ("S002", 0)}   # default call unchanged
    assert completed_keys(p, "t", window_start=0) == {("S001", 0)}
    assert completed_keys(p, "t", window_start=160) == {("S002", 0)}
