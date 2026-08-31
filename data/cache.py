"""Disk cache of epoched trials, one compressed .npz per subject.

Preprocessing all 109 subjects takes minutes; experiments must not pay that cost
repeatedly (CLAUDE.md, Dataset Rules). One file per subject rather than one big
archive, so LOSO folds can load only the subjects they need.

The cache stores **raw microvolts with no normalisation applied**. Fitting
mean/std here would leak test statistics into training, since the cache is built
before any split exists.

A ``manifest.json`` records the config that produced the cache. If the config
changes, the manifest no longer matches and the cache must be rebuilt — that is
the mechanism preventing a silent preprocessing change between experiments.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import Config
from .edf_loader import SubjectEpochs

MANIFEST_NAME = "manifest.json"


def config_fingerprint(cfg: Config) -> str:
    """Stable hash of every config field that affects the cached arrays."""
    payload = {
        "imagery_runs": list(cfg.dataset.imagery_runs),
        "expect_sfreq": cfg.dataset.expect_sfreq,
        "expect_n_channels": cfg.dataset.expect_n_channels,
        "expect_events_per_run": cfg.dataset.expect_events_per_run,
        "tmin": cfg.epoch.tmin,
        "length_s": cfg.epoch.length_s,
        "bandpass": list(cfg.preprocess.bandpass) if cfg.preprocess.bandpass else None,
        "notch": cfg.preprocess.notch,
        "rereference": cfg.preprocess.rereference,
        "normalize": cfg.preprocess.normalize,
        "label_scheme": cfg.label_scheme,
        "class_names": list(cfg.class_names),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def subject_path(cache_dir: Path, subject: str) -> Path:
    return cache_dir / f"{subject}.npz"


def save_subject(ep: SubjectEpochs, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = subject_path(cache_dir, ep.subject)
    np.savez_compressed(
        path,
        X=ep.X,
        y=ep.y,
        run=ep.run,
        onset=ep.onset,
        ch_names=np.array(ep.ch_names, dtype=object),
        sfreq=np.float64(ep.sfreq),
    )
    return path


def load_subject(subject: str, cache_dir: Path) -> SubjectEpochs:
    path = subject_path(cache_dir, subject)
    if not path.exists():
        raise FileNotFoundError(f"{path} — build the cache first")
    with np.load(path, allow_pickle=True) as z:
        return SubjectEpochs(
            subject=subject,
            X=z["X"],
            y=z["y"],
            run=z["run"],
            onset=z["onset"],
            ch_names=[str(c) for c in z["ch_names"]],
            sfreq=float(z["sfreq"]),
        )


def write_manifest(cfg: Config, cache_dir: Path, entries: list[dict],
                   dropped: list[dict]) -> Path:
    """Record what was built, from which config, and what was dropped and why."""
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_source": str(cfg.source_path),
        "config_fingerprint": config_fingerprint(cfg),
        "label_scheme": cfg.label_scheme,
        "class_names": list(cfg.class_names),
        "imagery_runs": list(cfg.dataset.imagery_runs),
        "excluded_subjects": sorted(cfg.dataset.exclude_subjects),
        "preprocess": {
            "bandpass": list(cfg.preprocess.bandpass) if cfg.preprocess.bandpass else None,
            "notch": cfg.preprocess.notch,
            "rereference": cfg.preprocess.rereference,
            "normalize": cfg.preprocess.normalize,
        },
        "n_subjects": len(entries),
        "n_trials": sum(e["n_trials"] for e in entries),
        "subjects": entries,
        "dropped": dropped,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def read_manifest(cache_dir: Path) -> dict:
    path = cache_dir / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} — build the cache first")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_cache_matches(cfg: Config, cache_dir: Path) -> None:
    """Fail loudly if the cache on disk was built with a different config.

    Guards CLAUDE.md's "don't silently change preprocessing between experiments".
    """
    manifest = read_manifest(cache_dir)
    want = config_fingerprint(cfg)
    got = manifest.get("config_fingerprint")
    if got != want:
        raise RuntimeError(
            f"cache at {cache_dir} was built with config fingerprint {got}, "
            f"but the current config is {want}. Rebuild the cache — and re-run "
            f"the baseline, since prior results are no longer comparable."
        )
