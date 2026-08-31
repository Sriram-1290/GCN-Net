"""2x2 representation ablation: band-pass x sample definition.

    python scripts/ablation_summary.py

Reports, for each arm, the 4-class accuracy and the family / within-family
decomposition (see project_documentation.md 7.1, 9.2), with per-subject values so
the comparisons can be paired. Pairing on subject is essential here: between-subject
variance in EEG is large enough to swamp the effects being measured.

**Within-family accuracy is the headline number.** It is the confound-free measure
of motor-imagery decoding — left vs right fist, and both fists vs both feet —
whereas the 4-class number and the family number both include the run-family
signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import paired_bootstrap_ci  # noqa: E402
from scripts.run_family_analysis import decompose  # noqa: E402

ARMS = [
    ("honest-20subj", "A", "none", "1 tp"),
    ("B-bp-tp", "B", "8-30 Hz", "1 tp"),
    ("C-raw-win64", "C", "none", "64 tp"),
    ("D-bp-win64", "D", "8-30 Hz", "64 tp"),
]


def per_subject_decomposition(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Pool each subject's fold confusion matrices, then decompose."""
    by_subject: dict[str, list[np.ndarray]] = defaultdict(list)
    for r in rows:
        if "trial_confusion" in r:
            by_subject[r["subject"]].append(np.array(r["trial_confusion"]))
    return {s: decompose(np.sum(cms, axis=0)) for s, cms in by_subject.items()}


def bootstrap_vs_chance(values: np.ndarray, chance: float, n_boot: int = 10_000,
                        seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap CI for the mean, to compare against a chance level."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default="runs/experiments.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.log, encoding="utf-8") if l.strip()]
    arms: dict[str, dict[str, dict[str, float]]] = {}

    print(f"{'arm':<4} {'band-pass':<9} {'sample':<7} {'n':>4} {'4-class':>9} "
          f"{'family':>8} {'within':>8} {'within 95% CI':>20}")
    print("-" * 76)

    for tag, letter, bp, sample in ARMS:
        arm_rows = [r for r in rows if r.get("tag") == tag]
        if not arm_rows:
            print(f"{letter:<4} {bp:<9} {sample:<7} {'--':>4}")
            continue
        per_sub = per_subject_decomposition(arm_rows)
        arms[letter] = per_sub

        four = np.array([d["accuracy_4class"] for d in per_sub.values()])
        fam = np.array([d["family_accuracy"] for d in per_sub.values()])
        win = np.array([d["within_family_accuracy"] for d in per_sub.values()])
        m, lo, hi = bootstrap_vs_chance(win, 0.5)
        flag = "*" if lo > 0.5 else " "
        print(f"{letter:<4} {bp:<9} {sample:<7} {len(per_sub):>4} {four.mean():>9.4f} "
              f"{fam.mean():>8.4f} {m:>8.4f}{flag} [{lo:.4f}, {hi:.4f}]")

    print("-" * 76)
    print(f"{'chance':<21} {'':>7} {0.25:>9.4f} {0.50:>8.4f} {0.50:>8.4f}")
    print("* = 95% CI excludes chance (0.5000)")

    print("\n--- paired contrasts on within-family accuracy ---")
    contrasts = [("B", "A", "band-pass, at 1 timepoint"),
                 ("D", "C", "band-pass, at 64 tp"),
                 ("C", "A", "window, unfiltered"),
                 ("D", "B", "window, band-passed"),
                 ("D", "A", "both vs the published formulation")]
    for hi_arm, lo_arm, label in contrasts:
        if hi_arm not in arms or lo_arm not in arms:
            continue
        shared = sorted(set(arms[hi_arm]) & set(arms[lo_arm]))
        a = np.array([arms[hi_arm][s]["within_family_accuracy"] for s in shared])
        b = np.array([arms[lo_arm][s]["within_family_accuracy"] for s in shared])
        mean, lo, hi = paired_bootstrap_ci(a, b, n_boot=10_000, seed=0)
        sig = "significant" if lo > 0 else "not significant"
        print(f"  {hi_arm} - {lo_arm}  {label:<34} {mean:+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  {sig}")

    if "A" in arms and "D" in arms:
        shared = sorted(set(arms["A"]) & set(arms["D"]))
        a = np.array([arms["A"][s]["within_family_accuracy"] for s in shared])
        d = np.array([arms["D"][s]["within_family_accuracy"] for s in shared])
        print(f"\n  subjects above chance within-family: "
              f"A {int((a > 0.5).sum())}/{len(a)}   D {int((d > 0.5).sum())}/{len(d)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
