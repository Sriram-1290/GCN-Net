"""Decompose 4-class accuracy into run-family identification vs true MI decoding.

Under scheme (a), classes 0/1 (left, right fist) come *only* from runs 4/8/12 and
classes 2/3 (both fists, both feet) come *only* from runs 6/10/14. Any run-level
non-neural signal — electrode drift, impedance change, alertness — therefore
separates {0,1} from {2,3} for free, without decoding imagery at all.

project_documentation.md 7.1 recorded this confound in advance and said any
4-class number must be reported alongside two diagnostics. This script computes
them from the pooled confusion matrices in the experiment log:

* **run-family accuracy** — did the model identify which run family the trial came
  from? Chance is 50 %.
* **within-family accuracy** — given the family, did it get the movement right?
  Chance is 50 %. **This is the only number that reflects motor-imagery decoding.**

If within-family accuracy sits at chance while the 4-class number looks
respectable, the model has decoded the session, not the intention.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# classes 0,1 come from runs 4/8/12; classes 2,3 from runs 6/10/14
FAMILY = np.array([0, 0, 1, 1])


def decompose(cm: np.ndarray) -> dict[str, float]:
    """Split a 4x4 confusion matrix into family and within-family accuracy."""
    if cm.shape != (4, 4):
        raise ValueError(f"expected a 4x4 confusion matrix, got {cm.shape}")
    total = cm.sum()

    # family: did true family == predicted family?
    family_correct = 0
    for t in range(4):
        for p in range(4):
            if FAMILY[t] == FAMILY[p]:
                family_correct += cm[t, p]

    # within-family: among predictions that landed in the right family,
    # how often was the exact class right?
    within_correct = int(np.trace(cm))
    within_total = family_correct

    return {
        "accuracy_4class": float(np.trace(cm) / total),
        "family_accuracy": float(family_correct / total),
        "within_family_accuracy": float(within_correct / within_total) if within_total else float("nan"),
        "n": int(total),
    }


def per_family_detail(cm: np.ndarray) -> str:
    lr = cm[:2, :2]   # true left/right, predicted left/right
    bf = cm[2:, 2:]   # true fists/feet, predicted fists/feet
    lines = []
    for name, block in (("left vs right fist", lr), ("both fists vs feet", bf)):
        n = block.sum()
        acc = np.trace(block) / n if n else float("nan")
        lines.append(f"  {name:<22} {acc:.4f}  (n={n}, chance 0.5000)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default="runs/experiments.jsonl")
    ap.add_argument("--tag", default="honest-20subj")
    ap.add_argument("--key", default="trial_confusion")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.log, encoding="utf-8") if l.strip()]
    cms = [np.array(r[args.key]) for r in rows
           if r.get("tag") == args.tag and args.key in r]
    if not cms:
        raise SystemExit(f"no rows with tag={args.tag!r} and key={args.key!r}")

    pooled = np.sum(cms, axis=0)
    d = decompose(pooled)

    print(f"tag: {args.tag}   folds: {len(cms)}   trials: {d['n']}\n")
    print("pooled confusion (rows=true, cols=predicted):")
    names = ["left_fist", "right_fist", "both_fists", "both_feet"]
    print(f"{'':>12} " + " ".join(f"{n[:9]:>9}" for n in names))
    for name, row in zip(names, pooled):
        print(f"{name:>12} " + " ".join(f"{v:9d}" for v in row))

    print(f"\n4-class accuracy        {d['accuracy_4class']:.4f}   (chance 0.2500)")
    print(f"run-family accuracy     {d['family_accuracy']:.4f}   (chance 0.5000)")
    print(f"within-family accuracy  {d['within_family_accuracy']:.4f}   (chance 0.5000)")
    print("\nbinary sub-problems, the confound-free measures of MI decoding:")
    print(per_family_detail(pooled))

    print("\ninterpretation:")
    fam_lift = d["family_accuracy"] - 0.5
    win_lift = d["within_family_accuracy"] - 0.5
    print(f"  family identification is {fam_lift:+.4f} above chance")
    print(f"  within-family decoding is {win_lift:+.4f} above chance")
    if win_lift < 0.02:
        print("  -> the 4-class number is carried by run-family identification.")
        print("     Within a run family the model is at chance, i.e. it is not")
        print("     decoding motor imagery at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
