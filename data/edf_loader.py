"""Load PhysioNet EEGMMIDB imagery runs into trial-level epochs.

Design notes that matter for the writeup:

* **Epochs are cut manually by annotation onset**, not with ``mne.Epochs``. Run
  lengths vary across subjects (123 / 124 / 125 s), so any fixed offset into the
  run is wrong; and ``mne.Epochs`` applies baseline correction by default, which
  with ``tmin=0`` would subtract a degenerate one-sample baseline. Explicit
  slicing keeps the operation obvious and unit-testable.

* **No normalisation happens here.** The cache holds raw microvolts. Mean/std are
  fit on the training split only, at training time. The TensorFlow reference
  mean-centres over the whole dataset before splitting, which is leakage.

* **Every shape and rate is asserted.** Silent broadcasting on EEG tensors is the
  classic way to produce a confident wrong answer, so a run that deviates aborts
  the build instead of being coerced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci

from .config import Config
from .labels import class_index

# MNE returns EEG in volts; the EDF files are in microvolts with a scale factor of
# exactly 1.0. Convert back so the cache is in the units the data was recorded in.
_VOLTS_TO_MICROVOLTS = 1e6


class RunValidationError(RuntimeError):
    """A run deviates from the expected recording parameters."""


@dataclass
class SubjectEpochs:
    """Trial-level epochs for one subject."""

    subject: str
    X: np.ndarray  # (n_trials, n_channels, n_samples) float32, microvolts
    y: np.ndarray  # (n_trials,) int64, index into CLASS_NAMES
    run: np.ndarray  # (n_trials,) int64, source run number
    onset: np.ndarray  # (n_trials,) float32, seconds into the run
    ch_names: list[str]
    sfreq: float

    def __post_init__(self) -> None:
        n = self.X.shape[0]
        assert self.y.shape == (n,), f"{self.y.shape} != ({n},)"
        assert self.run.shape == (n,), f"{self.run.shape} != ({n},)"
        assert self.onset.shape == (n,), f"{self.onset.shape} != ({n},)"
        assert self.X.ndim == 3, f"expected (trials, channels, samples), got {self.X.shape}"
        assert len(self.ch_names) == self.X.shape[1]


def run_path(root: Path, subject: str, run: int) -> Path:
    return root / subject / f"{subject}R{run:02d}.edf"


def load_run_raw(path: Path, cfg: Config) -> mne.io.BaseRaw:
    """Read one EDF, standardise channel names, validate, optionally filter."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose="error")

    # Strips the trailing dots and fixes capitalisation: 'Fc5.' -> 'FC5', 'C3..' -> 'C3'.
    # Required before any montage lookup or topographic plot.
    eegbci.standardize(raw)

    if abs(raw.info["sfreq"] - cfg.dataset.expect_sfreq) > 1e-6:
        raise RunValidationError(
            f"{path.name}: sampling rate {raw.info['sfreq']} Hz, "
            f"expected {cfg.dataset.expect_sfreq} Hz"
        )
    if len(raw.ch_names) != cfg.dataset.expect_n_channels:
        raise RunValidationError(
            f"{path.name}: {len(raw.ch_names)} channels after dropping the "
            f"'EDF Annotations' signal, expected {cfg.dataset.expect_n_channels}"
        )

    pp = cfg.preprocess
    if pp.notch is not None:
        raw.notch_filter(pp.notch, verbose="error")
    if pp.bandpass is not None:
        raw.filter(pp.bandpass[0], pp.bandpass[1], verbose="error")
    if pp.rereference is not None:
        raw.set_eeg_reference(pp.rereference, verbose="error")

    return raw


def epoch_run(raw: mne.io.BaseRaw, run: int, cfg: Config
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut one run into (X, y, onsets), dropping rest (T0) trials.

    Returns X as (n_trials, n_channels, n_samples) in microvolts.
    """
    sfreq = raw.info["sfreq"]
    n_samples = cfg.epoch.n_samples(sfreq)
    data = raw.get_data() * _VOLTS_TO_MICROVOLTS

    ann = raw.annotations
    if len(ann) != cfg.dataset.expect_events_per_run:
        raise RunValidationError(
            f"run {run}: {len(ann)} annotations, "
            f"expected {cfg.dataset.expect_events_per_run}"
        )

    epochs, labels, onsets = [], [], []
    for descr, onset in zip(ann.description, ann.onset):
        label = class_index(run, descr.strip())
        if label is None:  # T0 rest, discarded under scheme (a)
            continue

        start = int(round((onset + cfg.epoch.tmin) * sfreq))
        stop = start + n_samples
        if start < 0 or stop > data.shape[1]:
            raise RunValidationError(
                f"run {run}: trial at {onset:.2f}s needs samples [{start}, {stop}) "
                f"but the recording has {data.shape[1]}"
            )

        epochs.append(data[:, start:stop])
        labels.append(label)
        onsets.append(onset)

    X = np.asarray(epochs, dtype=np.float32)
    assert X.shape == (len(labels), data.shape[0], n_samples), X.shape
    return X, np.asarray(labels, dtype=np.int64), np.asarray(onsets, dtype=np.float32)


def load_subject(subject: str, cfg: Config) -> SubjectEpochs:
    """Load and epoch every imagery run for one subject."""
    Xs, ys, runs, onsets = [], [], [], []
    ch_names: list[str] | None = None
    sfreq: float | None = None

    for run in cfg.dataset.imagery_runs:
        path = run_path(cfg.dataset.root, subject, run)
        if not path.exists():
            raise FileNotFoundError(path)

        raw = load_run_raw(path, cfg)
        if ch_names is None:
            ch_names, sfreq = list(raw.ch_names), raw.info["sfreq"]
        elif list(raw.ch_names) != ch_names:
            raise RunValidationError(
                f"{subject} run {run}: channel names differ from run "
                f"{cfg.dataset.imagery_runs[0]}"
            )

        X, y, onset = epoch_run(raw, run, cfg)
        Xs.append(X)
        ys.append(y)
        onsets.append(onset)
        runs.append(np.full(len(y), run, dtype=np.int64))

    assert ch_names is not None and sfreq is not None
    return SubjectEpochs(
        subject=subject,
        X=np.concatenate(Xs, axis=0),
        y=np.concatenate(ys, axis=0),
        run=np.concatenate(runs, axis=0),
        onset=np.concatenate(onsets, axis=0),
        ch_names=ch_names,
        sfreq=sfreq,
    )
