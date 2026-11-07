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


def test_robustness_duplicate_rows_are_last_wins_and_warned_about():
    rr = _load("robustness_results")
    clean = _robustness_rows()
    assert rr.duplicate_warning(clean, _M1, _M2) is None
    assert rr.duplicate_counts(clean, _M1, _M2) == {}

    tag = f"{_M1}-binary-0-1"
    target = next(r for r in clean if r["tag"] == tag and r["subject"] == "S002"
                  and r["fold"] == 3)
    dup = dict(target, trial_accuracy=0.99, accuracy=0.99)
    other = f"{_M2}-binary-2-3.random25"
    dup2 = next(r for r in clean if r["tag"] == other and r["subject"] == "S001"
                and r["fold"] == 0)
    rows = clean + [dup, dup, dict(dup2)]       # 2 surplus rows on one tag, 1 on another

    assert rr.duplicate_counts(rows, _M1, _M2) == {tag: 2, other: 1}
    warning = rr.duplicate_warning(rows, _M1, _M2)
    assert warning.startswith("WARNING") and f"{tag}: 2" in warning and f"{other}: 1" in warning
    # last-wins: the later row replaces the earlier one, and the fold count stays 5
    assert rr._folds(rows, tag)[("S002", 3)] == 0.99
    assert len(rr._folds(rows, tag)) == len(rr._folds(clean, tag))
    rr.robustness_table(rows, _M1, _M2, _SUBJ)      # still complete, not refused


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


def test_robustness_drop_from_full_is_a_difference_of_differences():
    """POST HOC helper: with M1 +0.03 on iv2a and level on Full, M1 loses 0.03 less."""
    rr = _load("robustness_results")
    d = rr.drop_from_full_table(_robustness_rows(), _M1, _M2, _SUBJ)
    assert d["iv2a"]["did"] == pytest.approx(0.03)
    assert d["iv2a"]["wins"] == 6 and d["iv2a"]["ci_lo"] > 0
    assert d["random25"]["did"] == pytest.approx(0.0)
    # M1 trailing by 0.03 on Full shrinks its drop: level cond -> (0 - -0.03) - 0 = +0.03
    d = rr.drop_from_full_table(_robustness_rows(full_gain=-0.03), _M1, _M2, _SUBJ)
    assert d["random25"]["did"] == pytest.approx(0.03)


# --------------------------------------------------- Phase A analysis (9.26)
#
# Synthetic rows go through the module's own completeness guard and statistics, so every
# construction below has a known answer. ``_PA_SUBJ`` is a dozen subjects rather than 103:
# the script takes the expected subject set as an argument, and the real default
# (103, from configs/bandpass_8_30.yaml) is asserted separately.

_PA_SUBJ = tuple(f"S{i + 1:03d}" for i in range(12))
_FOLD_OFFSETS = {5: (-0.02, -0.01, 0.0, 0.01, 0.02), 3: (-0.01, 0.0, 0.01)}  # sum to 0
_CM4 = np.array([[8, 2, 1, 1], [3, 7, 1, 1], [1, 1, 5, 5], [2, 0, 4, 6]])
_CM_FLAT = np.array([[5, 5, 0, 0], [5, 5, 0, 0], [0, 0, 5, 5], [0, 0, 5, 5]])  # within 0.5


def _pa():
    return _load("phase_a_results")


def _pa_rows(tags, fields_fn, subjects=_PA_SUBJ, drop=()):
    """Rows for ``[(tag, folds), ...]``; ``fields_fn(tag, i, fold, folds) -> dict``.

    ``folds`` of None is a one-row-per-subject tag (zero-shot, timepoint holdout, A1).
    ``drop`` is a set of ``(tag, subject, fold)`` to leave out.
    """
    rows = []
    for tag, folds in tags:
        for i, subj in enumerate(subjects):
            for f in (range(folds) if folds else [None]):
                if (tag, subj, f) in drop:
                    continue
                rows.append({"tag": tag, "subject": subj, "fold": f,
                             **fields_fn(tag, i, f, folds)})
    return rows


def _acc_fields(acc, auc=None, cm=_CM4):
    return {"trial_accuracy": acc, "accuracy": acc, "timepoint_accuracy": acc,
            "roc_auc": acc if auc is None else auc,
            "trial_confusion": cm.tolist(), "timepoint_confusion": cm.tolist(),
            "confusion_excl_calibration": cm.tolist()}


def _base(i, folds, fold):
    off = _FOLD_OFFSETS[folds][fold] if folds else 0.0
    return 0.55 + 0.01 * i + off


def _plan_tags(exp, **kw):
    plan = _pa().plan(exp, **kw)
    return list(plan.primary) + list(plan.secondary)


def _a6_rows(d1, d2, d3, offline=0.05, subjects=_PA_SUBJ, drop=()):
    def fields(tag, i, fold, folds):
        shift = 0.0
        if "si-shallow-ea-offline" in tag or "si-tgsp-ea-offline" in tag:
            shift = offline
        elif "si-shallow-ea" in tag:
            shift = d1
        elif "si-tgsp-ea" in tag:
            shift = d2
        elif "ft-shallow-ea" in tag:
            shift = d3
        return _acc_fields(_base(i, folds, fold) + shift)
    return _pa_rows(_plan_tags("a6"), fields, subjects, drop)


def test_phase_a_default_subject_set_is_the_103():
    s = _pa().default_subjects()
    assert len(s) == 103 and "S001" in s and "S109" in s
    assert not {"S088", "S089", "S092", "S100", "S104", "S106"} & set(s)


def test_phase_a_a6_decides_effect_for_a_unanimous_gain_of_0_03():
    pa = _pa()
    res = pa.a6_results(pa.Log(_a6_rows(0.03, 0.03, 0.03), _PA_SUBJ))
    assert [c["id"] for c in res["contrasts"]] == ["C1", "C2", "C3"]
    for c in res["contrasts"]:
        assert c["diff"] == pytest.approx(0.03)
        assert c["verdict"] == "effect" and c["direction"] == "+"
        assert c["wins"] == c["n"] == 12 and c["ci_lo"] > 0
        assert c["holm_reject"] is True
    # predictions 1 and 2 hold as effects; 3 fails because 0.03 >= 0.020
    assert res["predictions"]["P1"].startswith("holds")
    assert res["predictions"]["P2"].startswith("holds")
    assert res["predictions"]["P3"].startswith("fails")


def test_phase_a_a6_decides_no_effect_for_0_01_and_reads_p3_as_equivalent():
    pa = _pa()
    res = pa.a6_results(pa.Log(_a6_rows(0.01, 0.01, 0.01), _PA_SUBJ))
    for c in res["contrasts"]:
        assert c["diff"] == pytest.approx(0.01) and c["verdict"] == "no effect"
    assert res["predictions"]["P1"].startswith("does not hold")
    assert res["predictions"]["P3"] == "holds: equivalence demonstrated"
    text = pa.analyse("a6", _a6_rows(0.01, 0.01, 0.01), _PA_SUBJ).text
    assert "no effect" in text and "TRANSDUCTIVE" in text


def test_phase_a_a6_effect_needs_the_margin_the_ci_and_the_right_sign():
    pa = _pa()
    # +0.03 on C1 only; C2 falls by 0.03 (an effect, but the wrong way for P2); C3 level
    res = pa.a6_results(pa.Log(_a6_rows(0.03, -0.03, 0.0), _PA_SUBJ))
    c1, c2, c3 = res["contrasts"]
    assert (c1["verdict"], c1["direction"]) == ("effect", "+")
    assert (c2["verdict"], c2["direction"]) == ("effect", "-")
    assert c3["verdict"] == "no effect"
    assert res["predictions"]["P2"].startswith("does not hold")
    assert res["predictions"]["P3"] == "holds: equivalence demonstrated"
    # a 0.03 mean carried by noise: CI reaches zero, so no effect
    rows = _a6_rows(0.0, 0.0, 0.0)
    for r in rows:
        if "si-shallow-ea" in r["tag"] and "offline" not in r["tag"]:
            i = _PA_SUBJ.index(r["subject"])
            r["trial_accuracy"] += 0.40 if i % 2 else -0.34     # mean +0.03, huge spread
    c1 = pa.a6_results(pa.Log(rows, _PA_SUBJ))["contrasts"][0]
    assert c1["diff"] == pytest.approx(0.03) and c1["ci_lo"] < 0
    assert c1["verdict"] == "no effect"


def test_phase_a_a6_subject_score_is_the_mean_of_the_two_problems():
    pa = _pa()
    rows = _a6_rows(0.0, 0.0, 0.0)
    for r in rows:      # +0.10 on problem 0-1 only: the mean of the two moves by 0.05
        if r["tag"] == "PA-A6-si-tgsp-ea-binary-0-1":
            r["trial_accuracy"] += 0.10
    c2 = pa.a6_results(pa.Log(rows, _PA_SUBJ))["contrasts"][1]
    assert c2["diff"] == pytest.approx(0.05)
    assert c2["per_problem"]["0-1"]["diff"] == pytest.approx(0.10)
    assert c2["per_problem"]["2-3"]["diff"] == pytest.approx(0.0)


def test_phase_a_a6_offline_arms_are_secondary_and_never_decide():
    pa = _pa()
    full = _a6_rows(0.0, 0.0, 0.0, offline=0.08)
    res = pa.a6_results(pa.Log(full, _PA_SUBJ))
    assert res["offline"]["shallow"]["diff"] == pytest.approx(0.08)
    assert all(c["verdict"] == "no effect" for c in res["contrasts"])   # offline ignored
    no_off = [r for r in full if "offline" not in r["tag"]]
    rep = pa.analyse("a6", no_off, _PA_SUBJ)               # secondary absent: still prints
    assert "not run" in rep.text and rep.exit_code == 0


def _a4_rows(subjects=_PA_SUBJ, drop=(), runopp=0.40):
    drops = {"rh_vs_feet": lambda i: 0.12 + 0.003 * i, "lr": lambda i: 0.04,
             "bf": lambda i: 0.06}
    rows = []
    for tag, folds in _plan_tags("a4"):
        _, _, task, model, split = tag.split("-")
        for i, subj in enumerate(subjects):
            for f in range(folds):
                if (tag, subj, f) in drop:
                    continue
                off = _FOLD_OFFSETS[folds][f]
                if split == "trial":
                    auc = 0.85 + 0.01 * i
                elif split == "run":
                    auc = 0.85 + 0.01 * i - drops[task](i)
                else:                                   # runopp: AUC below chance
                    auc = runopp
                rows.append({"tag": tag, "subject": subj, "fold": f,
                             "roc_auc": auc + off, "accuracy": auc + off - 0.1})
    return rows


def test_phase_a_a4_inflation_equals_the_constructed_value():
    """Inflation = (trial - run on rh_vs_feet) - mean of (trial - run) on lr and bf."""
    pa = _pa()
    res = pa.a4_results(pa.Log(_a4_rows(), _PA_SUBJ))
    expected = np.mean([(0.12 + 0.003 * i) - (0.04 + 0.06) / 2 for i in range(12)])
    for model in ("tgsp", "shallow"):
        est = res["models"][model]["auc"]
        assert est["estimate"] == pytest.approx(expected)            # 0.0865
        assert est["ci_lo"] > 0.020 and est["verdict"] == "inflation"
        # accuracy rides alongside: it was built as AUC - 0.1 everywhere, so same value
        assert res["models"][model]["accuracy"]["estimate"] == pytest.approx(expected)
        assert res["models"][model]["drops"]["lr"] == pytest.approx(0.04)
    assert res["models"]["tgsp"]["runopp"]["auc_runopp"] == pytest.approx(0.40)


def test_phase_a_a4_reads_a_level_estimate_as_no_extra_inflation():
    pa = _pa()
    rows = _a4_rows()
    for r in rows:      # controls lose exactly as much as rh_vs_feet: estimate 0
        if "-run" in r["tag"] and not r["tag"].endswith("runopp") \
                and ("-lr-" in r["tag"] or "-bf-" in r["tag"]):
            i = _PA_SUBJ.index(r["subject"])
            r["roc_auc"] = (0.85 + 0.01 * i - (0.12 + 0.003 * i)
                            + _FOLD_OFFSETS[3][r["fold"]])
    est = pa.a4_results(pa.Log(rows, _PA_SUBJ))["models"]["tgsp"]["auc"]
    assert est["estimate"] == pytest.approx(0.0, abs=1e-9)
    assert est["verdict"] == "no extra inflation"


def _a3_rows(trial, run, runopp, subjects=_PA_SUBJ, drop=()):
    def fields(tag, i, fold, folds):
        level = trial if "-trial-" in tag else run if "-run-" in tag else runopp
        return {"accuracy": level + 0.002 * i - 0.011 + _FOLD_OFFSETS[folds][fold]}
    return _pa_rows(_plan_tags("a3"), fields, subjects, drop)


def test_phase_a_a3_chance_reading_gap_and_drift_decomposition():
    pa = _pa()
    res = pa.a3_results(pa.Log(_a3_rows(0.70, 0.60, 0.40), _PA_SUBJ))
    cell = res["models"]["tgsp"]["bp830"]
    assert cell["trial"]["reading"] == "above chance"
    assert cell["run"]["reading"] == "above chance"
    assert cell["runopp"]["reading"] == "below chance"
    assert cell["gap"]["diff"] == pytest.approx(0.10)                 # (i) - (ii)
    assert cell["drift_free"]["mean"] == pytest.approx(0.50, abs=1e-9)
    assert cell["drift"]["mean"] == pytest.approx(0.10, abs=1e-9)     # half the difference
    # the drift decomposition is for tangent space only
    assert "runopp" not in res["models"]["shallow"]["bp830"]
    # within 0.020 of 0.5 is at chance
    near = pa.a3_results(pa.Log(_a3_rows(0.515, 0.50, 0.50), _PA_SUBJ))
    assert near["models"]["tgsp"]["car"]["trial"]["reading"] == "at chance"


def test_phase_a_a3_secondary_runopp_is_marked_not_run_when_absent_or_short():
    pa = _pa()
    rows = [r for r in _a3_rows(0.70, 0.60, 0.40) if "runopp" not in r["tag"]]
    rep = pa.analyse("a3", rows, _PA_SUBJ)
    assert rep.exit_code == 0 and "not run" in rep.text
    short = [r for r in _a3_rows(0.70, 0.60, 0.40)
             if not (r["tag"].endswith("runopp-car") and r["subject"] == "S003"
                     and r["fold"] == 1)]
    rep = pa.analyse("a3", short, _PA_SUBJ)           # incomplete secondary: not quoted
    assert "not run" in rep.text and "runopp-car" in rep.text


def _a5_rows(all_tp=0.90, subjects=_PA_SUBJ, drop=()):
    tp = {"none": 0.52, "split": 0.60, "bn": 0.55, "graph": 0.53, "all": all_tp}

    def fields(tag, i, fold, folds):
        arm = tag.rsplit("-", 1)[1]
        cm = _CM_FLAT if arm == "none" else _CM4
        return _acc_fields(tp[arm] + 0.001 * i, cm=cm)
    return _pa_rows(_plan_tags("a5"), fields, subjects, drop)


def test_phase_a_a5_gates_come_first_and_contrasts_follow_when_they_pass():
    pa = _pa()
    res = pa.a5_results(pa.Log(_a5_rows(), _PA_SUBJ))
    assert res["gates"]["all"]["pass"] and res["gates"]["none"]["pass"]
    assert res["gates"]["none"]["value"] == pytest.approx(0.5)        # the flat matrix
    c = res["contrasts"]["split"]["timepoint"]
    assert c["diff"] == pytest.approx(0.08) and c["verdict"] == "effect"
    assert res["contrasts"]["graph"]["timepoint"]["verdict"] == "no effect"
    rep = pa.analyse("a5", _a5_rows(), _PA_SUBJ)
    assert rep.exit_code == 0 and "Contrasts" in rep.text


def test_phase_a_a5_failed_gate_prints_the_failure_and_no_contrasts():
    pa = _pa()
    rep = pa.analyse("a5", _a5_rows(all_tp=0.70), _PA_SUBJ)
    assert rep.exit_code != 0
    assert "GATE FAILED" in rep.text and "Contrasts" not in rep.text
    assert "[0.8794, 0.9147]" in rep.text
    # the none gate: within-family must sit in [0.4984, 0.5199]
    rows = _a5_rows()
    for r in rows:
        if r["tag"].endswith("-none"):
            r["trial_confusion"] = _CM4.tolist()      # within-family 0.65: out of range
    rep = pa.analyse("a5", rows, _PA_SUBJ)
    assert rep.exit_code != 0 and "GATE FAILED" in rep.text and "Contrasts" not in rep.text


def test_phase_a_a2_reads_a_drop_when_the_ws160_arm_is_lower_by_0_03():
    pa = _pa()

    def fields(tag, i, fold, folds):
        shift = -0.03 if tag.startswith("PA-A2") else 0.0
        return _acc_fields(_base(i, folds, fold) + shift)
    rows = _pa_rows(_plan_tags("a2"), fields)
    res = pa.a2_results(pa.Log(rows, _PA_SUBJ))
    assert res["contrast"]["diff"] == pytest.approx(-0.03)            # (ii) - (i)
    assert res["verdict"].startswith("drop")
    assert res["contrast"]["wins"] == 0 and res["contrast"]["n"] == 12


def test_phase_a_a1_decomposition_of_a_known_matrix_matches_decompose():
    pa = _pa()
    fam = _load("run_family_analysis")
    mats = {i: _CM4 + i * np.eye(4, dtype=int) for i in range(12)}

    def fields(tag, i, fold, folds):
        return {"trial_confusion": mats[i].tolist(),
                "confusion_excl_calibration": (mats[i] - 1).clip(0).tolist()}
    rows = _pa_rows(list(pa.plan("a1").primary), fields)          # no A6 comparator rows
    res = pa.a1_results(pa.Log(rows, _PA_SUBJ))
    for model in ("shallow", "tgsp"):
        m = res["models"][model]
        pooled = fam.decompose(sum(mats.values()))
        assert m["pooled"] == pooled
        per = [fam.decompose(mats[i]) for i in range(12)]
        for key in ("accuracy_4class", "family_accuracy", "within_family_accuracy"):
            assert m["mean"][key] == pytest.approx(np.mean([d[key] for d in per]))
        excl = [fam.decompose((mats[i] - 1).clip(0)) for i in range(12)]
        assert m["excl"]["within_family_accuracy"] == pytest.approx(
            np.mean([d["within_family_accuracy"] for d in excl]))
    assert res["models"]["tgsp"]["vs_a6"] is None                  # no A6 rows: no claim


def test_phase_a_a1_compares_within_family_with_the_a6_noea_arm_of_the_same_model():
    pa = _pa()
    fam = _load("run_family_analysis")

    def fields(tag, i, fold, folds):
        if tag.startswith("PA-A6"):
            return _acc_fields(0.70)                 # the A6 binary baseline
        return {"trial_confusion": _CM4.tolist(),
                "confusion_excl_calibration": _CM4.tolist()}
    tags = _plan_tags("a1") + [(f"PA-A6-si-{m}-noea-binary-{p}", None)
                               for m in ("shallow", "tgsp") for p in ("0-1", "2-3")]
    res = pa.a1_results(pa.Log(_pa_rows(tags, fields), _PA_SUBJ))
    cmp_ = res["models"]["tgsp"]["vs_a6"]
    wf = fam.decompose(_CM4)["within_family_accuracy"]
    assert cmp_["a6"] == pytest.approx(0.70) and cmp_["a1"] == pytest.approx(wf)
    assert cmp_["diff"] == pytest.approx(wf - 0.70)


@pytest.mark.parametrize("experiment", ["a1", "a2", "a3", "a4", "a5", "a6"])
def test_phase_a_missing_row_makes_every_experiment_refuse(experiment):
    """One absent row in any primary tag is refused, naming the tag and the count."""
    pa = _pa()
    plan = pa.plan(experiment)
    every = list(plan.primary) + list(plan.secondary)

    def fields(tag, i, fold, folds):
        return _acc_fields(0.60 + 0.01 * i)
    pa.analyse(experiment, _pa_rows(every, fields), _PA_SUBJ)    # control: complete runs
    for tag, folds in plan.primary:
        last = (tag, _PA_SUBJ[-1], (folds - 1) if folds else None)
        rows = _pa_rows(every, fields, drop={last})
        need = len(_PA_SUBJ) * (folds or 1)
        with pytest.raises(pa.IncompleteRun) as e:
            pa.analyse(experiment, rows, _PA_SUBJ)
        assert tag in str(e.value) and f"{need - 1} of {need}" in str(e.value)


def test_phase_a_full_row_counts_are_the_pre_registered_ones():
    """5x103 = 515 for 5-fold tags, 3x103 = 309 for run tags, 103 for the rest."""
    pa = _pa()
    n = len(pa.default_subjects())
    by_tag = {t: n * (f or 1) for e in ("a1", "a2", "a3", "a4", "a5", "a6")
              for t, f in (list(pa.plan(e).primary) + list(pa.plan(e).secondary))}
    assert by_tag["A2-shallow-binary-0-1"] == 515
    assert by_tag["PA-A2-shallow-ws160-binary-0-1"] == 515
    assert by_tag["PA-A3-rest-tgsp-trial-bp830"] == 515
    assert by_tag["PA-A3-rest-shallow-run-car"] == 309
    assert by_tag["PA-A3-rest-tgsp-runopp-car"] == 309
    assert by_tag["PA-A4-lr-shallow-run"] == 309
    assert by_tag["PA-A4-rh_vs_feet-tgsp-runopp"] == 309
    assert by_tag["PA-A5-leak-none"] == by_tag["PA-A5-leak-bn"] == 515
    assert by_tag["PA-A5-leak-split"] == by_tag["PA-A5-leak-all"] == 103
    assert by_tag["PA-A6-si-tgsp-ea-binary-2-3"] == 103
    assert by_tag["PA-A6-ft-shallow-ea-binary-0-1"] == 515
    assert by_tag["PTG-shallow-binary-2-3"] == 515
    assert by_tag["PA-A6-si-shallow-ea-offline-binary-0-1"] == 103
    assert by_tag["PA-A1-shallow-4class"] == by_tag["PA-A1-tgsp-4class"] == 103


def test_phase_a_partial_mode_is_watermarked_and_never_decides():
    pa = _pa()
    rows = _a6_rows(0.03, 0.03, 0.03,
                    drop={("PA-A6-si-tgsp-ea-binary-0-1", "S012", None),
                          ("PA-A6-ft-shallow-ea-binary-0-1", "S012", 4)})
    with pytest.raises(pa.IncompleteRun):
        pa.analyse("a6", rows, _PA_SUBJ)
    rep = pa.analyse("a6", rows, _PA_SUBJ, allow_partial=True)
    assert rep.text.count("PARTIAL — NOT A RESULT") >= 2
    assert "| effect" not in rep.text and "no effect" not in rep.text
    assert "11 of 12" in rep.text            # progress is stated, S012 is left out


def test_phase_a_duplicate_rows_are_last_wins_and_noted():
    pa = _pa()
    rows = _a6_rows(0.03, 0.03, 0.03)
    dup = dict(next(r for r in rows if r["tag"] == "PA-A6-si-tgsp-ea-binary-0-1"
                    and r["subject"] == "S002"), trial_accuracy=0.99)
    rep = pa.analyse("a6", rows + [dup], _PA_SUBJ)
    assert "duplicate" in rep.text.lower()
    assert "PA-A6-si-tgsp-ea-binary-0-1: 1" in rep.text


def test_phase_a_cli_refuses_with_a_nonzero_exit(tmp_path):
    import json
    import subprocess
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in _a6_rows(0.03, 0.03, 0.03)[:50]),
                   encoding="utf-8")
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "phase_a_results.py"),
                        "a6", "--log", str(log)], capture_output=True, text=True,
                       encoding="utf-8")
    assert p.returncode != 0
    assert "PA-A6-" in p.stderr + p.stdout and "of 103" in p.stderr + p.stdout


# --- validation against the real log: the script must reproduce what was logged ---

@pytest.fixture(scope="module")
def _real_log():
    import json
    path = ROOT / "runs" / "experiments.jsonl"
    if not path.exists():
        pytest.skip("runs/experiments.jsonl is absent")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:      # a line being appended by a live job
                    pass
    pa = _pa()
    return pa, pa.Log(rows, pa.default_subjects())


def _need(log, tag, folds):
    have, need, complete = log.status(tag, folds)
    if not complete:
        pytest.skip(f"{tag} is not complete in this log ({have} of {need})")


def test_phase_a_reproduces_the_logged_a2_within_family_figure(_real_log):
    """A2-shallow-binary-*: 0.6725, the mean of the two problems' subject means."""
    pa, log = _real_log
    for p in ("0-1", "2-3"):
        _need(log, f"A2-shallow-binary-{p}", 5)
    arm = pa.binary_arm(log, "A2-shallow-binary-{p}")
    assert round(float(arm["mean"].mean()), 4) == 0.6725


def test_phase_a_reproduces_the_logged_ptg_shallow_figure(_real_log):
    """PTG-shallow-binary-*: 0.7369."""
    pa, log = _real_log
    for p in ("0-1", "2-3"):
        _need(log, f"PTG-shallow-binary-{p}", 5)
    arm = pa.binary_arm(log, "PTG-shallow-binary-{p}")
    assert round(float(arm["mean"].mean()), 4) == 0.7369


def test_phase_a_reproduces_the_a5_none_gate_procedure(_real_log):
    """Per-subject pooled trial_confusion -> decompose -> mean over subjects, per seed;
    the mean of the three seeds is the logged 0.5090. Likewise the leaky 0.8975."""
    pa, log = _real_log
    seeds = [f"P1-A-103-s{k}" for k in (1, 2, 3)]
    for t in seeds:
        _need(log, t, 5)
    per_seed = [float(np.mean(pa.within_family_by_subject(log, t))) for t in seeds]
    assert round(float(np.mean(per_seed)), 4) == 0.5090
    leaky = [f"P1-leaky-103-s{k}" for k in (1, 2, 3)]
    for t in leaky:
        _need(log, t, None)
    tp = [float(pa.per_subject(log, t, "timepoint_accuracy").mean()) for t in leaky]
    assert round(float(np.mean(tp)), 4) == 0.8975


# --------------------------------------------- --tgsp-only: a finished arm is a run

def _tgsp(rows):
    return [r for r in rows if "tgsp" in r["tag"]]


def test_phase_a_tgsp_only_plans_keep_the_tangent_space_tags_alone():
    pa = _pa()
    for exp in ("a1", "a3", "a4", "a6"):
        tags = [t for t, _ in pa.plan(exp, tgsp_only=True).primary]
        assert tags and all("tgsp" in t for t in tags), exp
        full = [t for t, _ in pa.plan(exp).primary]
        assert set(tags) < set(full)
    assert len(pa.plan("a3", tgsp_only=True).primary) == 4
    assert len(pa.plan("a4", tgsp_only=True).primary) == 6
    assert [t for t, _ in pa.plan("a6", tgsp_only=True).primary] == [
        f"PA-A6-si-tgsp-{a}-binary-{p}" for a in ("noea", "ea") for p in ("0-1", "2-3")]
    assert all(f is None for _, f in pa.plan("a6", tgsp_only=True).primary)
    for exp in ("a2", "a5"):
        with pytest.raises(ValueError):
            pa.plan(exp, tgsp_only=True)
        with pytest.raises(ValueError):
            pa.analyse(exp, [], _PA_SUBJ, tgsp_only=True)


def test_phase_a_a6_tgsp_only_never_prints_a_verdict_or_a_scorecard():
    pa = _pa()
    rows = _tgsp(_a6_rows(0.03, 0.03, 0.03, offline=0.07))     # no ShallowConvNet rows
    with pytest.raises(pa.IncompleteRun):
        pa.analyse("a6", rows, _PA_SUBJ)                       # the full run still refuses
    rep = pa.analyse("a6", rows, _PA_SUBJ, tgsp_only=True)
    assert rep.exit_code == 0
    text, low = rep.text, rep.text.lower()
    assert "Holm-adjusted decision pending: the three primary contrasts are adjusted " \
           "together, and arms 1 and 3 are not complete." in text
    for banned in ("effect", "scorecard", "p2", "holds", "does not hold", "verdict",
                   "meaning (fixed"):
        assert banned not in low, banned
    assert "+0.0300" in text and "12/12" in text and "raw p" in text
    assert "diff >= 0.020" in text and "CI excludes zero" in text
    assert "TRANSDUCTIVE" in text and "+0.0700" in text        # offline arm, labelled
    assert "si-shallow" not in text
    res = pa.a6_results(pa.Log(rows, _PA_SUBJ), tgsp_only=True)
    c = res["contrast"]
    assert c["diff"] == pytest.approx(0.03) and c["margin_met"] and c["ci_excludes_zero"]
    assert "verdict" not in c and "p_holm" not in c and "predictions" not in res
    # offline absent: still prints, marked not run
    no_off = [r for r in rows if "offline" not in r["tag"]]
    assert "not run" in pa.analyse("a6", no_off, _PA_SUBJ, tgsp_only=True).text
    # an incomplete tangent-space tag is refused
    short = _tgsp(_a6_rows(0.03, 0.03, 0.03,
                           drop={("PA-A6-si-tgsp-ea-binary-2-3", "S005", None)}))
    with pytest.raises(pa.IncompleteRun, match="PA-A6-si-tgsp-ea-binary-2-3: 11 of 12"):
        pa.analyse("a6", short, _PA_SUBJ, tgsp_only=True)


def test_phase_a_a3_tgsp_only_prints_tangent_space_with_drift_and_refuses_when_short():
    pa = _pa()
    rows = _tgsp(_a3_rows(0.70, 0.60, 0.40))
    with pytest.raises(pa.IncompleteRun):
        pa.analyse("a3", rows, _PA_SUBJ)                        # shallow tags missing
    rep = pa.analyse("a3", rows, _PA_SUBJ, tgsp_only=True)
    assert rep.exit_code == 0 and "shallow" not in rep.text.split("NOTE")[0]
    assert "Drift decomposition" in rep.text and "below chance" in rep.text
    assert "reported separately when complete" in rep.text
    res = pa.a3_results(pa.Log(rows, _PA_SUBJ), tgsp_only=True)
    assert list(res["models"]) == ["tgsp"]
    assert res["models"]["tgsp"]["bp830"]["drift"]["mean"] == pytest.approx(0.10, abs=1e-9)
    short = _tgsp(_a3_rows(0.70, 0.60, 0.40,
                           drop={("PA-A3-rest-tgsp-run-car", "S012", 2)}))
    with pytest.raises(pa.IncompleteRun, match="PA-A3-rest-tgsp-run-car: 35 of 36"):
        pa.analyse("a3", short, _PA_SUBJ, tgsp_only=True)


def test_phase_a_a4_tgsp_only_reports_the_inflation_for_tangent_space():
    pa = _pa()
    rows = _tgsp(_a4_rows())
    with pytest.raises(pa.IncompleteRun):
        pa.analyse("a4", rows, _PA_SUBJ)
    rep = pa.analyse("a4", rows, _PA_SUBJ, tgsp_only=True)
    assert rep.exit_code == 0 and "inflation" in rep.text
    assert "reported separately when complete" in rep.text
    res = pa.a4_results(pa.Log(rows, _PA_SUBJ), tgsp_only=True)
    assert list(res["models"]) == ["tgsp"]
    expected = np.mean([(0.12 + 0.003 * i) - 0.05 for i in range(12)])
    assert res["models"]["tgsp"]["auc"]["estimate"] == pytest.approx(expected)
    assert res["models"]["tgsp"]["runopp"]["auc_runopp"] == pytest.approx(0.40)
    short = _tgsp(_a4_rows(drop={("PA-A4-lr-tgsp-run", "S001", 0)}))
    with pytest.raises(pa.IncompleteRun, match="PA-A4-lr-tgsp-run: 35 of 36"):
        pa.analyse("a4", short, _PA_SUBJ, tgsp_only=True)


def test_phase_a_cli_tgsp_only_is_refused_for_a2_and_a5():
    import subprocess
    for exp in ("a2", "a5"):
        p = subprocess.run([sys.executable, str(ROOT / "scripts" / "phase_a_results.py"),
                            exp, "--tgsp-only"], capture_output=True, text=True,
                           encoding="utf-8")
        assert p.returncode != 0 and "tgsp-only" in p.stderr


def test_phase_a_refuses_a_tag_whose_rows_mix_arms():
    """A smoke row (or a mid-run flag change) under a real PA- tag must not be counted."""
    pa = _pa()
    rows = _a6_rows(0.03, 0.03, 0.0)
    target = next(r for r in rows if r["tag"] == "PA-A6-si-shallow-ea-binary-0-1")
    target["pretrain_epochs"] = 1                       # a smoke run's setting
    with pytest.raises(pa.IncompleteRun, match="pretrain_epochs"):
        pa.analyse("a6", rows, subjects=_PA_SUBJ)


def test_phase_a_notes_tags_that_span_commits():
    pa = _pa()
    rows = _a6_rows(0.03, 0.03, 0.0)
    for k, r in enumerate(rows):
        r["git_commit"] = "aaa" if k % 2 else "bbb"
    rep = pa.analyse("a6", rows, subjects=_PA_SUBJ)
    assert "commits" in rep.text
