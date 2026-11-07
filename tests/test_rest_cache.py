"""Rest-period (T0) cache: epoching and the run -> family mapping.

Synthetic data only. The ramp in channel 0 makes every epoch's first sample equal
to its source sample index, so a misplaced or shifted epoch cannot pass.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest

from data.config import load_config
from data.edf_loader import RunValidationError, epoch_rest, rest_family

ROOT = Path(__file__).resolve().parent.parent
SFREQ = 160.0
SEG_S = 4.2  # EEGMMIDB T0/T1/T2 segments are 4.1-4.2 s


def _cfg():
    return load_config(ROOT / "configs" / "bandpass_8_30.yaml")


def _raw(n_t0: int = 15, n_task: int = 15, dur: float = SEG_S, tail_s: float = 1.0,
         n_ch: int = 4, t0_last: bool = False):
    """T0 and T1/T2 segments alternating, then ``tail_s`` of padding.

    Starts with T0 (as the real runs do), or ends with T0 when ``t0_last``.
    """
    descr, onsets = [], []
    t = 0.0
    for i in range(max(n_t0, n_task)):
        pair = [("T0", i < n_t0), ("T1" if i % 2 == 0 else "T2", i < n_task)]
        for code, present in (pair[::-1] if t0_last else pair):
            if present:
                descr.append(code)
                onsets.append(t)
                t += dur
    n = int(round((t + tail_s) * SFREQ))
    data = np.zeros((n_ch, n))
    data[0] = np.arange(n) * 1e-6  # volts: sample index in microvolts
    info = mne.create_info([f"C{i}" for i in range(n_ch)], SFREQ, "eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_annotations(mne.Annotations(onsets, [dur] * len(onsets), descr))
    return raw, [o for o, d in zip(onsets, descr) if d == "T0"]


def test_rest_family_mapping():
    for run in (4, 8, 12):
        assert rest_family(run) == 0
    for run in (6, 10, 14):
        assert rest_family(run) == 1
    for run in (1, 2, 3, 5, 7, 9, 11, 13, 0, 15):
        with pytest.raises(ValueError):
            rest_family(run)


def test_epoch_rest_on_synthetic_raw():
    raw, t0_onsets = _raw()
    X, onsets = epoch_rest(raw, 4, _cfg())

    assert X.shape == (15, 4, 640)
    assert X.dtype == np.float32
    np.testing.assert_allclose(onsets, t0_onsets, atol=1e-4)
    for x, onset in zip(X, t0_onsets):
        start = int(round(onset * SFREQ))
        np.testing.assert_allclose(x[0], np.arange(start, start + 640), atol=1e-3)
        assert not x[1:].any()


@pytest.mark.filterwarnings("ignore:Limited .* annotation")
def test_epoch_rest_drops_only_an_epoch_that_runs_past_the_end():
    # Final T0 is the last segment and the recording stops 1 s into it
    # (15 T0 + 15 task = 30 annotations, so only the end-of-recording check fires).
    raw, t0_onsets = _raw(t0_last=True, tail_s=-SEG_S + 1.0)
    X, onsets = epoch_rest(raw, 4, _cfg())
    assert X.shape[0] == 14
    np.testing.assert_allclose(onsets, t0_onsets[:14], atol=1e-4)


def test_epoch_rest_rejects_bad_counts_and_short_segments():
    cfg = _cfg()
    with pytest.raises(RunValidationError):  # annotation count != 30
        epoch_rest(_raw(n_t0=15, n_task=10)[0], 4, cfg)
    with pytest.raises(RunValidationError):  # 30 annotations but not 15 rest
        epoch_rest(_raw(n_t0=10, n_task=20)[0], 4, cfg)
    with pytest.raises(RunValidationError):  # T0 shorter than the epoch
        epoch_rest(_raw(dur=3.0)[0], 4, cfg)
    with pytest.raises(ValueError):  # not an imagery run
        epoch_rest(_raw()[0], 3, cfg)
