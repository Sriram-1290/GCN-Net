"""A1 and A2: which explanation for the CSP gap survives? (17.1, 17.2)

    python scripts/posthoc_controls.py
    python scripts/posthoc_controls.py --allow-partial      # diagnostics only

Two explanations for the gap to CSP+LDA are live, and they are distinguishable:

* **the graph prior** — 9.7.1's claim that order-2 Chebyshev filters cannot express
  the unconstrained spatial projection CSP fits. **A1** tests it directly by
  freeing the spatial operators while changing nothing else.
* **sample size** — 17.1, from MOABB's finding that deep learning needs >150 trials
  per class in general and ~50 on an easy binary problem, while classical methods
  do well below 50. We have ~45 per binary problem. **A2** tests it with two
  graph-free CNNs on identical folds.

The interpretation grid this script prints:

===================  ====================  =======================================
A1 vs fixed graph    A2 vs CSP             reading
===================  ====================  =======================================
better               closes the gap        graph prior was the binding constraint;
                                           9.7.1 supported
no better            still loses           sample size; 9.7.1 unidentified, and the
                                           gap is not about graphs at all
better               still loses           the constraint costs something, but
                                           something else also binds
no better            closes the gap        the graph is not the issue but the
                                           *architecture* is -- 9.7.1 is wrong in
                                           its stated form
===================  ====================  =======================================

**Nothing here is confirmatory.** 10.3 fixed the hypothesis list in advance and
neither A1 nor A2 is on it; A1 additionally tests a claim (9.7.1) written *after*
the confirmatory runs. Holm correction is applied across this post-hoc family
anyway, because six comparisons uncorrected would be expected to throw a false
positive about one time in four -- but a corrected post-hoc test is still
post-hoc, and this script says so in its own output.

Following E1 (17.6), every comparison reports the **per-subject win rate**
alongside the mean. The H4 result is a thin mean difference over an interleaved
distribution, and reporting a mean without its spread is how that got missed.
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

DEFAULT_LOG = Path("runs/experiments.jsonl")
EXPECTED_FOLDS = 103 * 5 * 2   # subjects x folds x binary problems

# label -> (tag prefix, accuracy field)
ARMS = {
    "fixed graph":   ("C-fixed-binary", "trial_accuracy"),
    "+ attention":   ("C-attention-binary", "trial_accuracy"),
    "+ both":        ("C-both-binary", "trial_accuracy"),
    "CSP+LDA":       ("C-csp", "accuracy"),
    "A1 free spatial": ("A1-free-binary", "trial_accuracy"),
    "A2 EEGNet":     ("A2-eegnet-binary", "trial_accuracy"),
    "A2 ShallowNet": ("A2-shallow-binary", "trial_accuracy"),
}

# (id, arm, reference, alternative). All two-sided: no direction was pre-declared
# for any of these, and predicting our own method wins would be motivated
# reasoning -- the same reasoning that made H4 two-sided (10.3).
COMPARISONS = [
    ("A1-vs-fixed", "A1 free spatial", "fixed graph"),
    ("A1-vs-csp",   "A1 free spatial", "CSP+LDA"),
    ("EEGNet-vs-fixed", "A2 EEGNet", "fixed graph"),
    ("EEGNet-vs-csp",   "A2 EEGNet", "CSP+LDA"),
    ("Shallow-vs-fixed", "A2 ShallowNet", "fixed graph"),
    ("Shallow-vs-csp",   "A2 ShallowNet", "CSP+LDA"),
]


def load(log: Path) -> list[dict]:
    if not log.exists():
        raise SystemExit(f"no experiment log at {log}")
    return [json.loads(l) for l in log.open(encoding="utf-8") if l.strip()]


def per_subject(rows: list[dict], prefix: str, field: str
                ) -> tuple[dict[str, float], int]:
    """Per-subject mean over folds and both binary problems, plus the fold count."""
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if str(r.get("tag", "")).startswith(prefix) and r.get(field) is not None:
            acc[r["subject"]].append(float(r[field]))
    n_folds = sum(len(v) for v in acc.values())
    return {s: float(np.mean(v)) for s, v in acc.items()}, n_folds


def win_rate(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    d = a - b
    return int((d > 0).sum()), int(np.isclose(d, 0).sum()), float((d > 0).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--allow-partial", action="store_true",
                    help="report arms that have not finished. Error #10 was quoting "
                         "0.5774 from a run that read 0.5418 when complete, so this is "
                         "off by default and the output is marked PARTIAL when used")
    args = ap.parse_args()

    rows = load(args.log)
    arms, counts = {}, {}
    for label, (prefix, field) in ARMS.items():
        vals, n = per_subject(rows, prefix, field)
        if vals:
            arms[label], counts[label] = vals, n

    print("=" * 78)
    print("A1 / A2 POST-HOC CONTROLS  --  EXPLORATORY, NOT CONFIRMATORY")
    print("Neither is on the pre-registered list (10.3). A1 tests a claim (9.7.1)")
    print("written after the confirmatory runs. Do not report these as confirmatory.")
    print("=" * 78)

    print("\narm completeness")
    print("-" * 78)
    incomplete = []
    for label in ARMS:
        if label not in arms:
            print(f"  {label:<18} MISSING")
            incomplete.append(label)
            continue
        n, subj = counts[label], len(arms[label])
        # CSP logs one row per subject per problem, not per fold.
        target = EXPECTED_FOLDS if label != "CSP+LDA" else 103 * 2
        ok = n >= target
        print(f"  {label:<18} {n:>5} rows / {target} target, {subj:>3} subjects  "
              f"{'complete' if ok else 'INCOMPLETE'}")
        if not ok:
            incomplete.append(label)

    if incomplete and not args.allow_partial:
        print(f"\nrefusing to report: {', '.join(incomplete)} incomplete.")
        print("Re-run with --allow-partial for diagnostics only. Error #10 in 14 was")
        print("quoting a mean from a run at 35 of 100 folds; it moved by 0.036 when done.")
        return 1
    partial = " [PARTIAL -- DO NOT QUOTE]" if incomplete else ""

    print(f"\nper-subject means{partial}")
    print("-" * 78)
    for label, vals in arms.items():
        v = np.array(list(vals.values()))
        print(f"  {label:<18} {v.mean():.4f}   (n={len(v)} subjects, "
              f"SD {v.std(ddof=1):.4f})")

    results, p_values = {}, {}
    print(f"\npaired comparisons, two-sided{partial}")
    print("-" * 78)
    for cid, arm, ref in COMPARISONS:
        if arm not in arms or ref not in arms:
            continue
        subjects = sorted(set(arms[arm]) & set(arms[ref]))
        a = np.array([arms[arm][s] for s in subjects])
        b = np.array([arms[ref][s] for s in subjects])
        mean, lo, hi = paired_bootstrap_ci(a, b)
        p = bootstrap_p_value(a, b, alternative="two-sided")
        wins, ties, rate = win_rate(a, b)
        results[cid] = dict(arm=arm, ref=ref, mean=mean, lo=lo, hi=hi, p=p,
                            wins=wins, ties=ties, rate=rate, n=len(subjects))
        p_values[cid] = p
        print(f"  {cid:<18} {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}  "
              f"wins {wins}/{len(subjects)} ({rate:.0%})")

    if not results:
        print("  no A1/A2 arms in the log yet")
        return 1

    print(f"\nHolm-Bonferroni over the {len(p_values)} post-hoc comparisons{partial}")
    print("-" * 78)
    corrected = holm_bonferroni(p_values)
    for cid, info in sorted(corrected.items(), key=lambda kv: kv[1]["p_value"]):
        r = results[cid]
        verdict = "DIFFERENT" if info["reject"] else "not distinguishable"
        print(f"  {cid:<18} p={info['p_value']:.4f}  vs {info['threshold']:.4f}  "
              f"{verdict}")

    # ------------------------------------------------------------ the verdict
    print(f"\nwhat this says about 17.1{partial}")
    print("-" * 78)

    def sig(cid: str) -> bool:
        return cid in corrected and bool(corrected[cid]["reject"])

    a1_helps = sig("A1-vs-fixed") and results["A1-vs-fixed"]["mean"] > 0
    a1_closes = "A1-vs-csp" in results and not sig("A1-vs-csp")
    cnn_closes = any(not sig(c) for c in ("EEGNet-vs-csp", "Shallow-vs-csp")
                     if c in results)
    cnn_loses = all(sig(c) and results[c]["mean"] < 0
                    for c in ("EEGNet-vs-csp", "Shallow-vs-csp") if c in results)

    if a1_helps and (a1_closes or cnn_closes):
        print("  The graph prior was a binding constraint: lifting it helps, and an")
        print("  unconstrained spatial filter reaches CSP. 9.7.1 is SUPPORTED.")
    elif not a1_helps and cnn_loses:
        print("  Freeing the spatial operator changes nothing AND graph-free CNNs lose")
        print("  to CSP by a similar margin. The limitation is not the graph.")
        print("  9.7.1 is UNIDENTIFIED -- sample size (17.1) explains the gap at least")
        print("  as well, and 9.7.1 should be rewritten to say so.")
    elif not a1_helps and cnn_closes:
        print("  The graph is not the issue, but the wider architecture is: a CNN closes")
        print("  the gap while freeing our spatial operator does not. 9.7.1 is WRONG in")
        print("  its stated form -- the constraint it names is not what costs the gap.")
    elif a1_helps and cnn_loses:
        print("  Lifting the constraint helps but does not close the gap, and CNNs lose")
        print("  too. The graph prior costs something real; something else also binds.")
    else:
        print("  Mixed. Read the table above rather than taking a one-line verdict.")

    print("\n  Report this alongside the per-subject win rates, not instead of them:")
    print("  E1 (17.6) showed a mean difference of -0.0263 sitting inside a")
    print("  distribution running from -0.267 to +0.200.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
