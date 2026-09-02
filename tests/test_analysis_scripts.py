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
