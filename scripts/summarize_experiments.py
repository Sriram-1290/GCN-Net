"""Summarise the experiment log into the leaky-vs-correct comparison.

    python scripts/summarize_experiments.py
    python scripts/summarize_experiments.py --leaky-tag leaky-20subj --honest-tag honest-20subj

Reports per-subject means, the paired difference across subjects with a bootstrap
confidence interval, and the pooled confusion matrix. CLAUDE.md requires paired
tests and CIs across subjects rather than bare accuracy means, and pairing on
subject is what makes the comparison fair: subject variance in EEG dwarfs the
difference between most models.
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


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no experiment log at {path}")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def by_subject(rows: list[dict], key: str) -> dict[str, float]:
    """Mean of `key` per subject, averaging over folds."""
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            acc[r["subject"]].append(float(r[key]))
    return {s: float(np.mean(v)) for s, v in acc.items()}


def describe(name: str, values: dict[str, float]) -> None:
    v = np.array(list(values.values()))
    if not len(v):
        print(f"{name}: no rows")
        return
    print(f"{name}: n={len(v)}  mean {v.mean():.4f}  sd {v.std(ddof=1) if len(v)>1 else 0:.4f}  "
          f"min {v.min():.4f}  max {v.max():.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default="runs/experiments.jsonl")
    ap.add_argument("--leaky-tag", default="leaky-20subj")
    ap.add_argument("--honest-tag", default="honest-20subj")
    args = ap.parse_args()

    from evaluation.experiment_log import warn_if_duplicated
    warn_if_duplicated(args.log)
    rows = load_rows(Path(args.log))
    leaky_rows = [r for r in rows if r.get("tag") == args.leaky_tag]
    honest_rows = [r for r in rows if r.get("tag") == args.honest_tag]

    print(f"log: {args.log}  ({len(rows)} rows total)")
    print(f"leaky  '{args.leaky_tag}': {len(leaky_rows)} rows")
    print(f"honest '{args.honest_tag}': {len(honest_rows)} rows\n")

    leaky_tp = by_subject(leaky_rows, "timepoint_accuracy")
    honest_tp = by_subject(honest_rows, "timepoint_accuracy")
    honest_trial = by_subject(honest_rows, "trial_accuracy")

    print("--- per-subject means ---")
    describe("leaky  timepoint ", leaky_tp)
    describe("honest timepoint ", honest_tp)
    describe("honest trial     ", honest_trial)

    shared = sorted(set(leaky_tp) & set(honest_tp))
    if shared:
        a = np.array([leaky_tp[s] for s in shared])
        b = np.array([honest_tp[s] for s in shared])
        mean, lo, hi = paired_bootstrap_ci(a, b, n_boot=10_000, seed=0)
        print(f"\n--- paired over {len(shared)} subjects (timepoint accuracy) ---")
        print(f"leaky - honest: {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"chance = {0.25:.4f}")
        n_above = int((b > 0.25).sum())
        print(f"subjects above chance under the correct protocol: {n_above}/{len(b)}")

        print("\n--- per subject ---")
        print(f"{'subject':<9} {'leaky':>8} {'honest':>8} {'trial':>8} {'gap':>8}")
        for s in shared:
            print(f"{s:<9} {leaky_tp[s]:8.4f} {honest_tp[s]:8.4f} "
                  f"{honest_trial.get(s, float('nan')):8.4f} "
                  f"{leaky_tp[s] - honest_tp[s]:8.4f}")

    cms = [np.array(r["trial_confusion"]) for r in honest_rows if "trial_confusion" in r]
    if cms:
        total = np.sum(cms, axis=0)
        print("\n--- pooled trial confusion, correct protocol (rows=true) ---")
        for i, row in enumerate(total):
            print(f"  class {i}: " + " ".join(f"{v:5d}" for v in row))
        diag = np.trace(total) / total.sum()
        print(f"  pooled trial accuracy {diag:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
