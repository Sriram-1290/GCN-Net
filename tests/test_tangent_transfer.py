"""Tests for subject-independent tangent-space transfer (9.26 A6 arm 2, and A1).

The classical control of A6 must get exactly the treatment the deep model gets: the
same groups, the same EA references, and the calibration run excluded from scoring in
both arms. These tests pin each of those on synthetic data, through the real ``main``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.alignment import align, fit_ea  # noqa: E402
from data.riemann import covariances  # noqa: E402

RUN_ORDER = (4, 6, 8, 10, 12, 14)
N_PER_RUN = 6
N_CH, N_T = 8, 160
SUBJECTS = ("S001", "S002", "S003", "S004")


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_tangent_transfer", ROOT / "scripts" / "run_tangent_transfer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_subject(subject: str, cache_dir=None):
    """36 trials, 6 per imagery run, labelled f(run, T-code); per-subject mixing.

    Each class raises the power of its own channel, so the task is decodable.
    """
    rng = np.random.default_rng(2000 + int(subject[1:]))
    run = np.repeat(RUN_ORDER, N_PER_RUN).astype(np.int64)
    alt = np.tile([0, 1], len(run) // 2)
    y = np.where(np.isin(run, (4, 8, 12)), alt, 2 + alt).astype(np.int64)
    X = rng.normal(size=(len(run), N_CH, N_T))
    for c in range(4):
        X[y == c, c] *= 3.0
    mix = np.eye(N_CH) + 0.1 * rng.normal(size=(N_CH, N_CH))
    X = np.einsum("cd,ndt->nct", mix, X)
    return SimpleNamespace(subject=subject, X=X.astype(np.float32), y=y, run=run,
                           onset=np.zeros(len(run), dtype=np.float32),
                           ch_names=[f"C{i}" for i in range(N_CH)], sfreq=160.0)


def _task(subject, classes):
    ep = _fake_subject(subject)
    keep = np.isin(ep.y, classes) if classes else np.ones(len(ep.y), dtype=bool)
    y = ep.y[keep]
    if classes:
        y = (y == classes[1]).astype(np.int64)
    return ep.X[keep], y, ep.run[keep]


def _run_main(m, monkeypatch, workdir, *extra, subjects=SUBJECTS):
    monkeypatch.setattr(m, "load_subject", _fake_subject)
    log = Path(workdir) / "log.jsonl"
    argv = ["run_tangent_transfer.py",
            "--config", str(ROOT / "configs" / "bandpass_8_30.yaml"),
            "--subjects", *subjects, "--groups", "2", "--log", str(log), "--quiet",
            "--tag", "T", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    assert m.main() == 0
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def _separable_subject(seed, gain=1.0, n=60):
    """The same spatial structure in every subject (class 0 raises channels 0-1,
    class 1 channels 2-3), independent noise, and an optional global gain.

    No per-subject random mixing: that makes the off-diagonal tangent features
    subject-specific in sign, and such data does not transfer by construction.
    """
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n // 2)
    X = rng.normal(size=(n, N_CH, 256))
    X[y == 0, 0:2] *= 4.0
    X[y == 1, 2:4] *= 4.0
    return (gain * X).astype(np.float32), y


@pytest.mark.parametrize("ea", [False, True])
def test_tangent_transfer_separable(ea):
    """Fit on one subject, score another: a separable task transfers above 0.9.

    With EA the target carries a 2x global gain, the between-subject shift EA exists
    to remove; unaligned, that shift alone sends transfer towards chance.
    """
    m = _load()
    Xs, ys = _separable_subject(1)
    Xt, yt = _separable_subject(2, gain=2.0 if ea else 1.0)
    if ea:
        Xs, Xt = align(Xs, fit_ea(Xs).W), align(Xt, fit_ea(Xt).W)
    model = m.fit_transfer(covariances(Xs), ys)
    pred = m.score_trials(model, covariances(Xt))
    assert pred.shape == yt.shape
    assert float(np.mean(pred == yt)) > 0.9


def _expected_target_covs(subject, classes, ea):
    """The target's covariances as the protocol defines them, computed here
    independently: reference from the calibration run under pseudo-online."""
    X, _, runs = _task(subject, classes)
    if ea == "pseudo-online":
        calib = 4 if tuple(classes) == (0, 1) else 6
        X = align(X, fit_ea(X[runs == calib]).W)
    return covariances(X), runs


@pytest.mark.parametrize("ea", ["none", "pseudo-online"])
@pytest.mark.parametrize("classes,calibration", [((0, 1), 4), ((2, 3), 6)])
def test_tangent_calibration_run_never_scored(monkeypatch, tmp_path, ea, classes,
                                              calibration):
    """Review Focus 2 for the classical arm: no trial of the calibration run is ever
    scored, with or without EA. Each scored covariance is matched back to its trial."""
    m = _load()
    seen = []
    real = m.score_trials

    def spy(model, covs):
        seen.append(np.array(covs, copy=True))
        return real(model, covs)

    monkeypatch.setattr(m, "score_trials", spy)
    rows = _run_main(m, monkeypatch, tmp_path, "--classes", *map(str, classes),
                     "--ea", ea)
    assert [r["subject"] for r in rows] == ["S001", "S003", "S002", "S004"]
    assert len(seen) == len(rows)
    for row, covs in zip(rows, seen):
        expect, runs = _expected_target_covs(row["subject"], classes, ea)
        scored = []
        for c in covs:
            dist = np.abs(expect - c).reshape(len(expect), -1).max(axis=1)
            i = int(np.argmin(dist))
            assert dist[i] < 1e-8 * np.abs(c).max(), "a scored trial matches no trial"
            scored.append(i)
        assert len(set(scored)) == len(scored)
        assert calibration not in set(runs[scored].tolist())
        assert sorted(scored) == np.flatnonzero(runs != calibration).tolist()
        assert row["calibration_run_excluded"] is True and row["ea"] == ea
        assert np.array(row["trial_confusion"]).sum() == len(scored)
        assert row["calibration_runs"] == [calibration]


def _fingerprints(classes):
    table = {}
    for s in SUBJECTS:
        X, _, _ = _task(s, classes)
        for i, x in enumerate(X):
            table[x.tobytes()] = (s, i)
    return table


def test_tangent_ea_references(monkeypatch, tmp_path):
    """Sources: each subject's own trials. Target: its calibration run only."""
    m = _load()
    calls = []
    real = m.fit_ea

    def spy(X, *a, **k):
        calls.append(np.array(X, copy=True))
        return real(X, *a, **k)

    monkeypatch.setattr(m, "fit_ea", spy)
    _run_main(m, monkeypatch, tmp_path, "--classes", "0", "1", "--ea", "pseudo-online",
              "--targets", "S001")
    table = _fingerprints((0, 1))
    found = []
    for X in calls:
        ids = [table[x.tobytes()] for x in X]
        owners = {s for s, _ in ids}
        assert len(owners) == 1, f"one EA reference mixed subjects {owners}"
        found.append((owners.pop(), sorted(i for _, i in ids)))
    _, _, runs = _task("S001", (0, 1))
    n = len(runs)
    assert [idx for s, idx in found if s == "S001"] == [np.flatnonzero(runs == 4).tolist()]
    sources = [(s, idx) for s, idx in found if s != "S001"]
    assert sorted(s for s, _ in sources) == ["S002", "S004"]
    assert all(idx == list(range(n)) for _, idx in sources)


def test_tangent_offline_ea_is_labelled_transductive(monkeypatch, tmp_path):
    m = _load()
    (row,) = _run_main(m, monkeypatch, tmp_path, "--classes", "0", "1", "--ea", "offline",
                       "--targets", "S001")
    assert row["ea"] == "offline" and row["transductive"] is True
    assert row["ea_reference"] == "all_target_trials"


def test_tangent_groups_are_the_pretraining_groups(monkeypatch, tmp_path):
    """Same groups as every other arm, assigned over all subjects (index mod k)."""
    m = _load()
    rows = _run_main(m, monkeypatch, tmp_path, "--classes", "0", "1")
    groups = {r["subject"]: r["group_id"] for r in rows}
    assert groups == {"S001": 1, "S003": 1, "S002": 2, "S004": 2}
    assert all(r["n_source_subjects"] == 2 and r["n_groups"] == 2 for r in rows)
    # a --targets run keeps the full assignment
    (row,) = _run_main(m, monkeypatch, tmp_path / "t", "--classes", "0", "1",
                       "--targets", "S004")
    assert row["group_id"] == 2


def test_tangent_four_class_scores_every_trial_and_logs_both_matrices(monkeypatch,
                                                                      tmp_path):
    """A1: 4-class, all trials scored, plus the matrix outside runs 4 and 6."""
    m = _load()
    (row,) = _run_main(m, monkeypatch, tmp_path, "--targets", "S001")
    _, _, runs = _task("S001", None)
    cm = np.array(row["trial_confusion"])
    ex = np.array(row["confusion_excl_calibration"])
    assert cm.shape == ex.shape == (4, 4) and row["n_classes"] == 4
    assert cm.sum() == len(runs)
    assert ex.sum() == int(np.sum(~np.isin(runs, (4, 6))))
    assert (ex <= cm).all()
    assert row["calibration_run_excluded"] is False and row["ea"] == "none"


def test_tangent_four_class_refuses_ea(monkeypatch, tmp_path):
    m = _load()
    with pytest.raises(SystemExit) as err:
        _run_main(m, monkeypatch, tmp_path, "--ea", "pseudo-online")
    assert err.value.code == 2


def test_tangent_resume_skips_logged_targets(monkeypatch, tmp_path):
    m = _load()
    args = ("--classes", "0", "1", "--targets", "S001", "S002", "--resume")
    assert len(_run_main(m, monkeypatch, tmp_path, *args)) == 2
    fits = []
    monkeypatch.setattr(m, "fit_transfer", lambda *a, **k: fits.append(1))
    assert len(_run_main(m, monkeypatch, tmp_path, *args)) == 2
    assert fits == [], "a resumed run with nothing left still fitted a model"


def test_tangent_classifier_matches_the_subject_dependent_runner():
    """No new hyperparameters: the same scaler and penalised logistic regression."""
    a = (ROOT / "scripts" / "run_tangent_space.py").read_text(encoding="utf-8")
    b = (ROOT / "scripts" / "run_tangent_transfer.py").read_text(encoding="utf-8")
    for snippet in ("RobustScaler()", "LogisticRegression(C=0.01, max_iter=2000)"):
        assert snippet in a and snippet in b
