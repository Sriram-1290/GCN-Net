"""Verify the no-reference cache against the standard (common-average) cache.

    python scripts/check_noref_cache.py
    python scripts/check_noref_cache.py --ref data/cache_bp830 --noref data/cache_bp830_noref

Reduced-montage conditions score the no-reference cache, while training and the
published numbers use the referenced one. If the two caches disagree on trial
order, labels or channel order, the wrong trials get scored with no error, so this
checks, for every subject:

  1. the same subject set in both caches;
  2. identical ``ch_names``, ``y``, trial count (and, as a stricter order check,
     ``run`` and ``onset``);
  3. ``common_average(noref.X)`` equals the referenced ``X`` to within
     ``1e-3 * median |X|`` (max absolute difference). The tolerance is only float32
     storage noise: the common average is linear per sample, so re-referencing
     epochs is equivalent to re-referencing the continuous recording.

Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import load_subject, read_manifest  # noqa: E402
from data.montage import common_average  # noqa: E402

REL_TOL = 1e-3


def check_subject(subject: str, ref_dir: Path, noref_dir: Path) -> tuple[float, list[str]]:
    """Return (max abs diff / median |X|, list of failure messages) for one subject."""
    ref = load_subject(subject, ref_dir)
    nor = load_subject(subject, noref_dir)
    problems: list[str] = []

    if ref.ch_names != nor.ch_names:
        problems.append("ch_names differ")
    if ref.X.shape != nor.X.shape:
        problems.append(f"X shape {ref.X.shape} vs {nor.X.shape}")
        return float("inf"), problems
    if not np.array_equal(ref.y, nor.y):
        problems.append("y differs")
    if not np.array_equal(ref.run, nor.run):
        problems.append("run differs")
    if not np.array_equal(ref.onset, nor.onset):
        problems.append("onset differs")

    rebuilt = common_average(nor.X)
    scale = float(np.median(np.abs(ref.X)))
    worst = float(np.abs(rebuilt - ref.X).max()) / scale
    if worst > REL_TOL:
        problems.append(f"common_average(noref) differs from referenced X: "
                        f"{worst:.3e} x median|X| > {REL_TOL:g}")
    return worst, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="data/cache_bp830", help="common-average cache")
    ap.add_argument("--noref", default="data/cache_bp830_noref", help="no-reference cache")
    args = ap.parse_args()
    ref_dir, noref_dir = Path(args.ref), Path(args.noref)

    subjects = [e["subject"] for e in read_manifest(ref_dir)["subjects"]]
    noref_subjects = [e["subject"] for e in read_manifest(noref_dir)["subjects"]]
    if subjects != noref_subjects:
        print(f"FAIL  subject lists differ: only in ref "
              f"{sorted(set(subjects) - set(noref_subjects))}, only in noref "
              f"{sorted(set(noref_subjects) - set(subjects))}")
        return 1

    worst_subject, worst_rel = "", -1.0
    n_ok = 0
    for subject in subjects:
        rel, problems = check_subject(subject, ref_dir, noref_dir)
        if problems:
            print(f"FAIL  {subject}: {'; '.join(problems)}")
        else:
            n_ok += 1
        if rel > worst_rel:
            worst_subject, worst_rel = subject, rel

    print(f"worst subject {worst_subject}: max |diff| = {worst_rel:.3e} x median|X| "
          f"(tolerance {REL_TOL:g})")
    print(f"{n_ok}/{len(subjects)} subjects match")
    return 0 if n_ok == len(subjects) else 1


if __name__ == "__main__":
    raise SystemExit(main())
