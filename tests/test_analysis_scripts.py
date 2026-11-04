"""Tests for the analysis scripts whose outputs are quoted in the write-ups.

These three scripts compute numbers that appear in `PAPER.md`, `RESULTS.md` and
`project_documentation.md`, and until now none of them had a test. That is the wrong
way round: a bug in a training script shows up as a bad accuracy, but a bug in an
*analysis* script produces a plausible number that goes straight into a paper.

The concrete precedent is error 17. `posthoc_controls.py`'s verdict rule tested
statistical significance rather than effect size, and would have reported "9.7.1
supported" for an arm recovering 16 % of the gap. It was caught by running the tool on
real results — which only works if you already suspect the answer.

`smr_predictor` gets the most attention here because it is the newest, because it was
implemented from a published method description rather than a reference implementation,
and because its output (+0.547 / +0.548, ~30 % of variance) is now in the paper's
abstract. The test injects a sinusoid of known frequency and amplitude and checks the
predictor recovers it, and checks that it does *not* fire on 1/f noise alone.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(name: str):
    """Import a script from scripts/ as a module without executing main()."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------- SMR predictor

def _synthetic_raw(alpha_amplitude: float, sfreq: float = 160.0, seconds: float = 61.0,
                   n_channels: int = 3, alpha_hz: float = 10.0, seed: int = 0):
    """A Raw with 1/f-ish noise plus an optional alpha peak of known amplitude."""
    mne = pytest.importorskip("mne")
    rng = np.random.default_rng(seed)
    n = int(sfreq * seconds)
    t = np.arange(n) / sfreq

    # pink-ish noise: white noise shaped by 1/f in the frequency domain
    white = rng.normal(size=(n_channels, n))
    spec = np.fft.rfft(white, axis=1)
    freqs = np.fft.rfftfreq(n, 1 / sfreq)
    spec[:, 1:] /= freqs[1:] ** 0.5
    data = np.fft.irfft(spec, n=n, axis=1)
    data /= data.std(axis=1, keepdims=True)

    if alpha_amplitude:
        data = data + alpha_amplitude * np.sin(2 * np.pi * alpha_hz * t)

    info = mne.create_info(["C3", "Cz", "C4"][:n_channels], sfreq, "eeg")
    return mne.io.RawArray(data * 1e-6, info, verbose=False)


def test_predictor_recovers_an_injected_alpha_peak():
    """The whole method: a real mu peak must register above the 1/f background."""
    smr = _load("smr_predictor")
    quiet = smr.smr_predictor(_synthetic_raw(0.0), picks=[0, 1, 2])
    peaked = smr.smr_predictor(_synthetic_raw(2.0), picks=[0, 1, 2])
    assert peaked > quiet + 5.0, (
        f"injecting a 10 Hz peak moved the predictor only {peaked - quiet:.2f} dB "
        f"({quiet:.2f} -> {peaked:.2f}); it is not measuring what it claims to"
    )


def test_predictor_is_monotonic_in_peak_amplitude():
    smr = _load("smr_predictor")
    values = [smr.smr_predictor(_synthetic_raw(a), picks=[0, 1, 2])
              for a in (0.0, 0.5, 1.0, 2.0)]
    assert values == sorted(values), f"not monotonic in amplitude: {values}"


def test_predictor_ignores_overall_power():
    """Scaling a signal must not change it — that is the point of the 1/f fit.

    A subject with more power everywhere is not a subject with a stronger mu rhythm.
    Raw band power would conflate the two; an excess over a fitted background must not.
    """
    smr = _load("smr_predictor")
    raw = _synthetic_raw(1.0)
    base = smr.smr_predictor(raw, picks=[0, 1, 2])
    scaled = raw.copy()
    scaled._data *= 10.0
    assert smr.smr_predictor(scaled, picks=[0, 1, 2]) == pytest.approx(base, abs=0.2)


def test_predictor_is_small_for_pure_background():
    """No peak, no score. Guards against the fit itself manufacturing a bump."""
    smr = _load("smr_predictor")
    assert smr.smr_predictor(_synthetic_raw(0.0, seed=3), picks=[0, 1, 2]) < 6.0


# ------------------------------------------------------- per-subject statistics

def test_win_rate_counts_ties_separately():
    pc = _load("posthoc_controls")
    a = np.array([0.6, 0.5, 0.4, 0.5])
    b = np.array([0.5, 0.5, 0.5, 0.4])
    wins, ties, rate = pc.win_rate(a, b)
    assert (wins, ties) == (2, 1)
    assert rate == pytest.approx(0.5)


def test_per_subject_averages_over_folds_and_problems():
    het = _load("subject_heterogeneity")
    rows = [
        {"tag": "T-x", "subject": "S1", "trial_accuracy": 0.4},
        {"tag": "T-x", "subject": "S1", "trial_accuracy": 0.6},
        {"tag": "T-x", "subject": "S2", "trial_accuracy": 0.8},
        {"tag": "OTHER", "subject": "S1", "trial_accuracy": 0.0},   # wrong tag
        {"tag": "T-x", "subject": "S3"},                            # no accuracy
    ]
    out = het.per_subject(rows, "T-x", "trial_accuracy")
    assert out == {"S1": pytest.approx(0.5), "S2": pytest.approx(0.8)}


def test_bootstrap_ci_brackets_the_point_estimate():
    het = _load("subject_heterogeneity")
    rng = np.random.default_rng(0)
    d = rng.normal(0.05, 0.1, size=200)
    lo, hi = het.bootstrap_ci(d, lambda x: float(x.mean()))
    assert lo < d.mean() < hi


# ------------------------------------------------------------ the verdict rule

def test_gap_fraction_is_what_drives_the_verdict_not_significance():
    """Error 17, encoded.

    An arm that clears significance while recovering a sixth of the gap must not be
    read as vindicating 9.7.1. This checks the arithmetic the verdict rests on, using
    the real measured values.
    """
    fixed, csp, a1, shallow = 0.5438, 0.6239, 0.5566, 0.6725
    gap = csp - fixed
    assert gap == pytest.approx(0.0801, abs=5e-4)
    assert (a1 - fixed) / gap == pytest.approx(0.160, abs=0.01)      # minor
    assert (shallow - fixed) / gap > 1.0                              # closes it
    # the distinction the first version of the rule missed
    assert (a1 - fixed) > 0.0                                         # significant...
    assert (a1 - fixed) / gap < 0.5                                   # ...but not material


def test_completeness_guard_targets_match_the_protocol():
    """103 subjects x 5 folds x 2 binary problems; CSP logs per subject per problem."""
    pc = _load("posthoc_controls")
    assert pc.EXPECTED_FOLDS == 103 * 5 * 2 == 1030
    assert set(pc.ARMS) >= {"fixed graph", "CSP+LDA", "A1 free spatial",
                            "A2 EEGNet", "A2 ShallowNet"}
    for cid, arm, ref in pc.COMPARISONS:
        assert arm in pc.ARMS and ref in pc.ARMS, f"{cid} names an unknown arm"


# ------------------------------------------------- montage robustness (9.24.5)

_COND_NAMES = ["iv2a", "random25", "tentwenty", "random10", "random50",
               "cluster8", "cluster16"]
_SUBJ = tuple(f"S{i + 1:03d}" for i in range(6))
_M1, _M2 = "ROB1-chebrp", "ROB1-shallow"


def _robustness_rows(gain=None, full_gain=0.0, drop=(), subjects=_SUBJ):
    """Synthetic log. ``gain[cond]`` is M1 - M2: a float, or ``f(i, problem, fold)``.

    M2's base score differs by subject, by problem and by fold, so any aggregation that
    ignores a problem or a fold changes the answer. ``drop`` removes (tag, subject, fold)
    rows so the completeness guard can be hit.
    """
    gain = {"iv2a": 0.03, "random25": 0.0, **(gain or {})}
    rows = []
    for arm, prefix in (("m1", _M1), ("m2", _M2)):
        for pi, problem in enumerate(("binary-0-1", "binary-2-3")):
            base_tag = f"{prefix}-{problem}"
            for i, subj in enumerate(subjects):
                for fold in range(5):
                    m2 = 0.55 + 0.01 * i + 0.03 * pi + 0.004 * fold
                    for cond in [None] + _COND_NAMES:
                        g = full_gain if cond is None else gain.get(cond, 0.0)
                        g = g(i, pi, fold) if callable(g) else g
                        acc = m2 + (g if arm == "m1" else 0.0)
                        tag = base_tag if cond is None else f"{base_tag}.{cond}"
                        if (tag, subj, fold) in drop:
                            continue
                        rows.append({"tag": tag, "subject": subj, "fold": fold,
                                     "trial_accuracy": acc, "accuracy": acc})
    return rows


def _table(rr, **kw):
    return rr.robustness_table(_robustness_rows(**kw), _M1, _M2, _SUBJ)


def _crit(t, cond="iv2a"):
    return t["conditions"][cond]["criteria"]


def test_robustness_table_decision():
    rr = _load("robustness_results")

    t = _table(rr)
    assert t["conditions"]["iv2a"]["decision"] == "win"
    assert t["conditions"]["iv2a"]["diff"] == pytest.approx(0.03)
    assert all(_crit(t).values())
    assert t["conditions"]["iv2a"]["wins"] == t["n_subjects"] == 6
    assert t["conditions"]["random25"]["decision"] == "no effect"
    assert t["conditions"]["random25"]["direction"] == "level"
    assert not t["guard"]["fires"]
    # secondary conditions are labelled exploratory and get no decision
    assert t["conditions"]["tentwenty"]["decision"] is None
    assert t["conditions"]["tentwenty"]["role"] == "exploratory"
    md = rr.format_markdown(t).splitlines()
    assert next(l for l in md if l.startswith("| iv2a (primary)")).endswith("| win |")
    assert next(l for l in md if l.startswith("| random25 (primary)")
                ).endswith("| no effect (level) |")
    assert next(l for l in md if l.startswith("| tentwenty")).endswith(
        "| none (exploratory) |")

    # Full guard: M1 trails M2 by 0.03 on the full cap -> a would-be win is a trade-off
    t = _table(rr, full_gain=-0.03)
    assert t["guard"]["fires"]
    assert t["conditions"]["iv2a"]["decision"] == "trade-off"
    assert t["conditions"]["random25"]["decision"] == "no effect"
    # a deficit inside the 0.020 guard does not fire it
    t = _table(rr, full_gain=-0.015)
    assert not t["guard"]["fires"] and t["conditions"]["iv2a"]["decision"] == "win"


def test_robustness_each_criterion_fails_alone():
    """Each case violates exactly one of the four criteria, so none can be deleted."""
    rr = _load("robustness_results")

    # margin only: unanimous +0.015 is significant and on every subject, but < 0.020
    t = _table(rr, gain={"iv2a": 0.015})
    assert _crit(t) == {"margin": False, "ci_excludes_zero": True, "holm": True,
                        "majority": True}
    assert t["conditions"]["iv2a"]["decision"] == "no effect"
    assert t["conditions"]["iv2a"]["direction"] == "M1 ahead"

    # majority only: 3 of 6 subjects at +0.08, 3 tied -> mean 0.04, CI > 0, Holm passes,
    # but 3 wins is not a strict majority of 6
    t = _table(rr, gain={"iv2a": lambda i, pi, f: 0.08 if i < 3 else 0.0})
    c = t["conditions"]["iv2a"]
    assert c["diff"] == pytest.approx(0.04) and c["wins"] == 3
    assert _crit(t) == {"margin": True, "ci_excludes_zero": True, "holm": True,
                        "majority": False}
    assert c["decision"] == "no effect"

    # CI (and Holm) fail while margin and majority pass: four subjects +0.12, two -0.10
    t = _table(rr, gain={"iv2a": lambda i, pi, f: 0.12 if i < 4 else -0.10})
    c = t["conditions"]["iv2a"]
    assert c["diff"] == pytest.approx(0.28 / 6) and c["wins"] == 4
    assert _crit(t)["margin"] and _crit(t)["majority"]
    assert not _crit(t)["ci_excludes_zero"]
    assert c["decision"] == "no effect"


def test_robustness_measured_loss_is_labelled_a_loss():
    rr = _load("robustness_results")
    t = _table(rr, gain={"iv2a": -0.04})
    c = t["conditions"]["iv2a"]
    assert c["decision"] == "loss" and c["direction"] == "M1 behind"
    assert next(l for l in rr.format_markdown(t).splitlines()
                if l.startswith("| iv2a (primary)")).endswith("| loss (M1 behind) |")
    # a small deficit is not a detectable loss: no effect, direction reported
    c = _table(rr, gain={"iv2a": -0.015})["conditions"]["iv2a"]
    assert c["decision"] == "no effect" and c["direction"] == "M1 behind"
    # a beyond-margin deficit whose CI reaches zero is not a loss either
    c = _table(rr, gain={"iv2a": lambda i, pi, f: -0.12 if i < 4 else 0.10}
               )["conditions"]["iv2a"]
    assert c["diff"] == pytest.approx(-0.28 / 6) and c["ci_hi"] > 0
    assert c["decision"] == "no effect"


def test_robustness_aggregation_is_mean_of_both_problems_over_all_folds():
    """Problems and folds carry different gains; only the right aggregate gives 0.03.

    Problem 0 gains 0.10 on fold 0 only (problem mean 0.02); problem 1 gains 0.04 on
    every fold (mean 0.04). Mean of the two problem means = 0.03. Using one problem,
    one fold, or dropping a fold gives something else.
    """
    rr = _load("robustness_results")

    def gain(i, pi, fold):
        return (0.10 if fold == 0 else 0.0) if pi == 0 else 0.04

    t = _table(rr, gain={"iv2a": gain})
    c = t["conditions"]["iv2a"]
    assert c["diff"] == pytest.approx(0.03)
    exp_m2 = np.mean([0.55 + 0.01 * i + 0.03 * pi + 0.004 * f
                      for i in range(6) for pi in range(2) for f in range(5)])
    assert c["m2"] == pytest.approx(exp_m2)
    assert c["m1"] == pytest.approx(exp_m2 + 0.03)


def test_robustness_table_refuses_partial_and_mismatched_runs():
    rr = _load("robustness_results")
    # one fold of one condition tag missing
    rows = _robustness_rows(drop={(f"{_M2}-binary-2-3.random25", "S004", 4)})
    with pytest.raises(rr.IncompleteRun, match="random25"):
        rr.robustness_table(rows, _M1, _M2, _SUBJ)
    # M2 lacks a subject that M1 has
    rows = [r for r in _robustness_rows()
            if not (r["tag"].startswith(_M2) and r["subject"] == "S006")]
    with pytest.raises(rr.IncompleteRun, match="different subjects"):
        rr.robustness_table(rows, _M1, _M2, _SUBJ)
    # an unknown prefix is empty, not silently "complete"
    with pytest.raises(rr.IncompleteRun):
        rr.robustness_table(_robustness_rows(), "ROB1-nope", _M2, _SUBJ)


def test_robustness_refuses_when_both_arms_miss_the_expected_subjects():
    """M1 and M2 agreeing with each other is not completeness: the target is fixed."""
    rr = _load("robustness_results")
    rows = _robustness_rows()                                    # 6 subjects, both arms
    with pytest.raises(rr.IncompleteRun, match="not the expected 20"):
        rr.robustness_table(rows, _M1, _M2)                      # default S001..S020
    with pytest.raises(rr.IncompleteRun, match="unexpected"):
        rr.robustness_table(rows, _M1, _M2, _SUBJ[:5])           # a surplus subject
    assert rr.DEV_SUBJECTS[0] == "S001" and rr.DEV_SUBJECTS[-1] == "S020"
    assert len(rr.DEV_SUBJECTS) == 20


def test_robustness_condition_tag_matches_the_writer():
    """The script restates ``condition_tag`` to avoid importing torch; keep them equal.

    The writer's function is lifted from its source with ``ast`` rather than imported,
    because ``evaluation.robustness`` imports torch and this test must not need it.
    """
    import ast
    rr = _load("robustness_results")
    from data.montage import CONDITIONS
    src = (ROOT / "evaluation" / "robustness.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "condition_tag")
    ns: dict = {}
    exec(compile(ast.Module([fn], []), "robustness.py", "exec"), ns)
    condition_tag = ns["condition_tag"]
    for c in CONDITIONS:
        assert rr.condition_tag("ROB1-x-binary-0-1", c.name) == \
            condition_tag("ROB1-x-binary-0-1", c.name)
