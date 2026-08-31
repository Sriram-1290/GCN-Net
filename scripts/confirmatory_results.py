"""The confirmatory results table, with Holm correction.

    python scripts/confirmatory_results.py

Evaluates the five pre-registered hypotheses (project_documentation.md 10.3)
exactly as specified: paired across subjects, primary outcome is mean
within-family accuracy on the two binary problems, Holm-Bonferroni across the
family at family-wise alpha 0.05, directions fixed in advance.

It will refuse to report a hypothesis as supported on the development set. Section
10.1 demotes the 20 subjects used so far to a development set and moves every
confirmatory test to the full 103, so a run tagged with development data is
labelled EXPLORATORY here no matter what its p-value is.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import (  # noqa: E402
    bootstrap_p_value,
    holm_bonferroni,
    paired_bootstrap_ci,
)

CONFIRMATION_MIN_SUBJECTS = 100  # 103 minus the excluded; anything less is development

# (id, description, tag_a, tag_b, alternative). tag_b None means "versus chance".
HYPOTHESES = [
    ("H1", "learnable adjacency > fixed PCC graph",
     "F-learngraph-binary", "P2-gcn-binary", "greater"),
    ("H2", "temporal attention > no attention",
     "G-attention-binary", "P2-gcn-binary", "greater"),
    ("H3", "attention + learnable graph > better of either alone",
     "H-both-binary", "F-learngraph-binary", "greater"),
    ("H4", "best configuration vs CSP+LDA",
     "BEST", "P2-csp", "two-sided"),
    ("H5", "LOSO within-family > chance",
     "loso-binary", None, "greater"),
]


def binary_within_family(rows: list[dict], prefix: str) -> dict[str, float]:
    """Mean accuracy over the two binary problems, per subject.

    The primary outcome. Tags are expected as ``<prefix>-0-1`` and ``<prefix>-2-3``;
    a subject counts only if it has both, so the mean is never taken over one.
    """
    per_problem: dict[str, dict[str, list[float]]] = {}
    for suffix in ("0-1", "2-3"):
        acc: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            tag = str(r.get("tag", ""))
            if tag.startswith(prefix) and tag.endswith(suffix) \
                    and r.get("trial_accuracy") is not None:
                acc[r["subject"]].append(r["trial_accuracy"])
        per_problem[suffix] = acc
    shared = set(per_problem["0-1"]) & set(per_problem["2-3"])
    return {s: float(np.mean([np.mean(per_problem["0-1"][s]),
                              np.mean(per_problem["2-3"][s])])) for s in shared}


def csp_within_family(rows: list[dict]) -> dict[str, float]:
    per: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if str(r.get("tag", "")).startswith("P2-csp") and "accuracy" in r:
            per[r["subject"]].append(r["accuracy"])
    return {s: float(np.mean(v)) for s, v in per.items() if len(v) == 2}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default="runs/experiments.jsonl")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    from evaluation.experiment_log import warn_if_duplicated
    warn_if_duplicated(args.log)
    rows = [json.loads(l) for l in open(args.log, encoding="utf-8") if l.strip()]

    print("CONFIRMATORY RESULTS -- pre-registered in project_documentation.md 10")
    print(f"primary outcome: mean within-family accuracy over the two binary problems")
    print(f"family-wise alpha {args.alpha}, Holm-Bonferroni over "
          f"{len(HYPOTHESES)} tests\n")

    measured: dict[str, dict] = {}
    p_values: dict[str, float] = {}

    for hid, label, tag_a, tag_b, alt in HYPOTHESES:
        a = csp_within_family(rows) if tag_a == "P2-csp" else binary_within_family(rows, tag_a)
        if tag_b is None:
            b = {s: 0.5 for s in a}          # H5 compares against chance
        elif tag_b == "P2-csp":
            b = csp_within_family(rows)
        else:
            b = binary_within_family(rows, tag_b)

        shared = sorted(set(a) & set(b))
        if len(shared) < 2:
            print(f"{hid}  {label:<48} NOT YET RUN")
            measured[hid] = {"status": "not run"}
            continue

        av = np.array([a[s] for s in shared])
        bv = np.array([b[s] for s in shared])
        p = bootstrap_p_value(av, bv, alternative=alt)
        mean, lo, hi = paired_bootstrap_ci(av, bv)
        p_values[hid] = p
        measured[hid] = {"label": label, "n": len(shared), "a": av.mean(),
                         "b": bv.mean(), "diff": mean, "lo": lo, "hi": hi,
                         "alt": alt}

    if not p_values:
        print("\nNo confirmatory arms have been run yet. Nothing to correct.")
        print("Development-set results live in project_documentation.md 9 and are "
              "explicitly exploratory.")
        return 0

    corrected = holm_bonferroni(p_values, alpha=args.alpha)

    print(f"{'':<4} {'hypothesis':<44} {'n':>4} {'diff':>8} {'95% CI':>20} "
          f"{'p':>8} {'p_adj':>8}  verdict")
    print("-" * 118)
    for hid, label, *_ in HYPOTHESES:
        m = measured.get(hid, {})
        if m.get("status") == "not run":
            print(f"{hid:<4} {label:<44} {'--':>4}   not yet run")
            continue
        c = corrected[hid]
        development = m["n"] < CONFIRMATION_MIN_SUBJECTS
        if development:
            verdict = f"EXPLORATORY (n={m['n']} < {CONFIRMATION_MIN_SUBJECTS})"
        else:
            verdict = "SUPPORTED" if c["reject"] else "not supported"
        print(f"{hid:<4} {label:<44} {m['n']:>4} {m['diff']:>+8.4f} "
              f"[{m['lo']:+.4f}, {m['hi']:+.4f}] {c['p']:>8.4f} "
              f"{c['p_adjusted']:>8.4f}  {verdict}")

    if any(m.get("n", 0) < CONFIRMATION_MIN_SUBJECTS
           for m in measured.values() if "n" in m):
        print(f"\nNOTE: rows marked EXPLORATORY ran on the development set. Section 10.1 "
              f"\nrequires confirmatory tests on all 103 subjects; those 83 unseen "
              f"subjects have\nnever influenced a design decision and are what make the "
              f"test confirmatory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
