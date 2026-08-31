"""Build the epoched trial cache for every usable subject.

    python scripts/build_cache.py                       # all subjects
    python scripts/build_cache.py --subjects S001 S002  # a subset, for smoke tests
    python scripts/build_cache.py --config configs/other.yaml

Subjects that fail validation are reported and skipped rather than silently
coerced, and the reason is written into the manifest.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import save_subject, write_manifest  # noqa: E402
from data.config import all_subjects, load_config  # noqa: E402
from data.edf_loader import load_subject  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="subject IDs to build; default is every non-excluded subject")
    ap.add_argument("--out", default=None, help="override dataset.cache_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache_dir = Path(args.out) if args.out else cfg.dataset.cache_dir
    subjects = args.subjects if args.subjects else all_subjects(cfg)

    print(f"config      {cfg.source_path}")
    print(f"root        {cfg.dataset.root}")
    print(f"cache       {cache_dir}")
    print(f"runs        {list(cfg.dataset.imagery_runs)}")
    print(f"classes     {list(cfg.class_names)}")
    print(f"bandpass    {cfg.preprocess.bandpass}  (None = matches the TF reference)")
    print(f"subjects    {len(subjects)}")
    print()

    entries: list[dict] = []
    dropped: list[dict] = []
    t0 = time.time()

    for i, subject in enumerate(subjects, 1):
        try:
            ep = load_subject(subject, cfg)
        except Exception as exc:  # noqa: BLE001 - report and continue
            dropped.append({"subject": subject, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[{i:3d}/{len(subjects)}] {subject}  DROPPED  {type(exc).__name__}: {exc}")
            continue

        save_subject(ep, cache_dir)
        counts = [int((ep.y == c).sum()) for c in range(len(cfg.class_names))]
        entries.append({
            "subject": subject,
            "n_trials": int(ep.X.shape[0]),
            "per_class": counts,
            "shape": list(ep.X.shape),
        })
        print(f"[{i:3d}/{len(subjects)}] {subject}  {ep.X.shape}  per-class {counts}")

    path = write_manifest(cfg, cache_dir, entries, dropped)
    total = sum(e["n_trials"] for e in entries)
    print()
    print(f"built {len(entries)} subjects, {total} trials in {time.time() - t0:.1f}s")
    if dropped:
        print(f"dropped {len(dropped)}: {[d['subject'] for d in dropped]}")
    print(f"manifest -> {path}")
    return 1 if dropped else 0


if __name__ == "__main__":
    raise SystemExit(main())
