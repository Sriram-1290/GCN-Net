"""Build the rest-period (T0) cache for the label-free run-family test.

    python scripts/build_rest_cache.py --config configs/rest_bp830.yaml
    python scripts/build_rest_cache.py --config configs/rest_car.yaml
    python scripts/build_rest_cache.py --config configs/rest_car.yaml --subjects S001

Same subject list and exclusions as ``build_cache.py``. Each ``S###.npz`` holds
``X`` (n_rest, 64, 640) float32 in microvolts, ``run`` (source run number, so the
family is ``rest_family(run)``), ``onset`` (seconds into the run) and
``ch_names``. Preprocessing is whatever the config says, applied by
``load_run_raw`` exactly as for the trial cache. Subjects whose file already
exists are skipped, so an interrupted build resumes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import MANIFEST_NAME, config_fingerprint, subject_path  # noqa: E402
from data.config import Config, all_subjects, load_config  # noqa: E402
from data.edf_loader import (RunValidationError, epoch_rest, load_run_raw,  # noqa: E402
                             rest_family, run_path)


def build_subject(subject: str, cfg: Config, cache_dir: Path) -> dict:
    """Epoch the rest segments of every imagery run for one subject and save them."""
    Xs, runs, onsets = [], [], []
    ch_names: list[str] | None = None
    for run in cfg.dataset.imagery_runs:
        path = run_path(cfg.dataset.root, subject, run)
        if not path.exists():
            raise FileNotFoundError(path)
        raw = load_run_raw(path, cfg)
        if ch_names is None:
            ch_names = list(raw.ch_names)
        elif list(raw.ch_names) != ch_names:
            raise RunValidationError(f"{subject} run {run}: channel names differ")
        X, onset = epoch_rest(raw, run, cfg)
        Xs.append(X)
        onsets.append(onset)
        runs.append(np.full(len(onset), run, dtype=np.int64))

    assert ch_names is not None
    X, run_arr = np.concatenate(Xs), np.concatenate(runs)
    assert X.dtype == np.float32 and X.shape[1:] == (len(ch_names), cfg.epoch.n_samples(
        cfg.dataset.expect_sfreq)), X.shape
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(subject_path(cache_dir, subject), X=X, run=run_arr,
                        onset=np.concatenate(onsets),
                        ch_names=np.array(ch_names, dtype=object))
    return _entry(subject, run_arr)


def _entry(subject: str, run_arr: np.ndarray) -> dict:
    fam = np.array([rest_family(int(r)) for r in run_arr])
    return {"subject": subject, "n_rest": int(len(run_arr)),
            "per_run": {int(r): int((run_arr == r).sum()) for r in np.unique(run_arr)},
            "per_family": [int((fam == 0).sum()), int((fam == 1).sum())]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="subject IDs to build; default is every non-excluded subject")
    ap.add_argument("--out", default=None, help="override dataset.cache_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache_dir = Path(args.out) if args.out else cfg.dataset.cache_dir
    subjects = args.subjects if args.subjects else all_subjects(cfg)

    print(f"config      {cfg.source_path}")
    print(f"cache       {cache_dir}")
    print(f"runs        {list(cfg.dataset.imagery_runs)}")
    print(f"bandpass    {cfg.preprocess.bandpass}   rereference {cfg.preprocess.rereference}")
    print(f"subjects    {len(subjects)}")
    print()

    entries: list[dict] = []
    dropped: list[dict] = []
    t0 = time.time()
    for i, subject in enumerate(subjects, 1):
        tag = f"[{i:3d}/{len(subjects)}] {subject}"
        existing = subject_path(cache_dir, subject)
        try:
            if existing.exists():
                with np.load(existing, allow_pickle=True) as z:
                    entry = _entry(subject, z["run"])
                print(f"{tag}  exists, skipped  n_rest {entry['n_rest']}")
            else:
                entry = build_subject(subject, cfg, cache_dir)
                print(f"{tag}  n_rest {entry['n_rest']}  per-run {entry['per_run']}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            dropped.append({"subject": subject, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"{tag}  DROPPED  {type(exc).__name__}: {exc}")
            continue
        entries.append(entry)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "rest_T0",
        "config_source": str(cfg.source_path),
        "config_fingerprint": config_fingerprint(cfg),
        "imagery_runs": list(cfg.dataset.imagery_runs),
        "family_of_run": {str(r): rest_family(r) for r in cfg.dataset.imagery_runs},
        "excluded_subjects": sorted(cfg.dataset.exclude_subjects),
        "preprocess": {
            "bandpass": list(cfg.preprocess.bandpass) if cfg.preprocess.bandpass else None,
            "notch": cfg.preprocess.notch,
            "rereference": cfg.preprocess.rereference,
        },
        "epoch": {"tmin": cfg.epoch.tmin, "length_s": cfg.epoch.length_s},
        "n_subjects": len(entries),
        "n_rest": sum(e["n_rest"] for e in entries),
        "subjects": entries,
        "dropped": dropped,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    counts = [e["n_rest"] for e in entries]
    print()
    print(f"built {len(entries)} subjects, {sum(counts)} rest epochs in {time.time() - t0:.1f}s")
    if counts:
        print(f"per-subject epoch count: min {min(counts)}  max {max(counts)}  "
              f"mean {np.mean(counts):.1f}")
    if dropped:
        print(f"dropped {len(dropped)}: {[d['subject'] for d in dropped]}")
    print(f"manifest -> {path}")
    return 1 if dropped else 0


if __name__ == "__main__":
    raise SystemExit(main())
