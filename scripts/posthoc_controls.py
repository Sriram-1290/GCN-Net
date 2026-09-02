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

The interpretation grid this script prints. **The test is the fraction of the
fixed-graph-to-CSP gap an arm recovers, not whether it is significant.** With 103
subjects the design detects ~0.020 (17.2), so an arm can beat the fixed graph
significantly while recovering almost none of the gap — an earlier version of this
script called that "9.7.1 supported", which was wrong.

=====================  ====================  ====================================
A1 recovers            a CNN recovers        reading
=====================  ====================  ====================================
most of the gap        most of the gap       graph prior was binding; 9.7.1
                                             supported as written
a little (significant  most of the gap       SPLIT: sample size refuted, the
but < half)                                  limitation is architectural, but the
                                             mechanism 9.7.1 names is minor and
                                             most of the gap is elsewhere
nothing                most of the gap       the constraint 9.7.1 names is not the
                                             cause; 9.7.1 wrong as stated
nothing                nothing               sample size (17.1); 9.7.1
                                             unidentified
=====================  ====================  ====================================

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
    "G1 gbp graph":  ("G1-gbp-graph-binary", "trial_accuracy"),
    "G2 gbp free":   ("G2-gbp-free-binary", "trial_accuracy"),
    "G3 gbp + prior": ("G3-gbp-prior-binary", "trial_accuracy"),
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
    # The reordered architecture (17.3 / 9.12). Each comparison answers one question:
    # did reordering fix the graph model; does the graph constraint still cost; does
    # the graph earn its place as a prior; and does any of it reach ShallowConvNet.
    ("gbpGraph-vs-attention", "G1 gbp graph", "+ attention"),
    ("gbpGraph-vs-csp",       "G1 gbp graph", "CSP+LDA"),
    ("gbpFree-vs-gbpGraph",   "G2 gbp free", "G1 gbp graph"),
    ("gbpPrior-vs-gbpFree",   "G3 gbp + prior", "G2 gbp free"),
    ("gbpBest-vs-shallow",    "G3 gbp + prior", "A2 ShallowNet"),
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


def make_figure(arms: dict[str, dict[str, float]], out: Path) -> None:
    """Ladder of arms with per-subject spread, against the published ceiling.

    The ordering is the argument: every arm left of CSP is a graph model, every
    arm right of it is not, and the published ceiling line shows that the two
    graph-free CNNs are not merely beating our baseline but reaching the range a
    216,714-configuration search reports for this dataset (17.0).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["fixed graph", "A1 free spatial", "+ both", "+ attention",
             "CSP+LDA", "A2 EEGNet", "A2 ShallowNet",
             "G1 gbp graph", "G2 gbp free", "G3 gbp + prior"]
    order = [k for k in order if k in arms]
    vals = [np.array(list(arms[k].values())) for k in order]
    means = [v.mean() for v in vals]
    # red: the published graph formulation. green: graph-free. blue: the reordered
    # architecture, which is a graph model built the way CSP is.
    old_graph = {"fixed graph", "A1 free spatial", "+ both", "+ attention"}
    reordered = {"G1 gbp graph", "G2 gbp free", "G3 gbp + prior"}
    colors = ["#8A2F2A" if k in old_graph else
              "#1F4D8F" if k in reordered else
              "#4B4F5A" if k == "CSP+LDA" else "#1B6B4A" for k in order]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, (v, c) in enumerate(zip(vals, colors)):
        ax.scatter(np.random.default_rng(i).normal(i, .07, len(v)), v,
                   s=9, alpha=.28, color=c, edgecolor="none")
    ax.bar(range(len(order)), means, width=.55, color=colors, alpha=.85, zorder=3)
    for i, m in enumerate(means):
        ax.text(i, m + .012, f"{m:.4f}", ha="center", fontsize=9, zorder=4)

    ax.axhline(.5, color="#333", lw=.9, ls=":", label="chance")
    ax.axhline(.650, color="#1F4D8F", lw=1.2, ls="--",
               label="best of 216,714 pipelines on PhysionetMI [23]")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([k.replace("A2 ", "").replace("A1 ", "A1 ") for k in order],
                       fontsize=9)
    ax.set_ylabel("within-family accuracy")
    ax.set_ylim(.30, 1.0)
    ax.set_title("Red: the published graph formulation.  Green: graph-free.  "
                 "Blue: reordered, graph-informed\n"
                 "103 subjects, identical folds, dots are subjects", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "posthoc_controls.png", dpi=200, bbox_inches="tight")
    print(f"\nwrote {out / 'posthoc_controls.png'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--allow-partial", action="store_true",
                    help="report arms that have not finished. Error #10 was quoting "
                         "0.5774 from a run that read 0.5418 when complete, so this is "
                         "off by default and the output is marked PARTIAL when used")
    ap.add_argument("--figures", type=Path, default=None,
                    help="write posthoc_controls.png into this directory")
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
    for cid, info in sorted(corrected.items(), key=lambda kv: kv[1]["p"]):
        r = results[cid]
        verdict = "DIFFERENT" if info["reject"] else "not distinguishable"
        print(f"  {cid:<18} p={info['p']:.4f}  vs {info['threshold']:.4f}  "
              f"(adj {info['p_adjusted']:.4f})  "
              f"{verdict}")

    # ------------------------------------------------------------ the verdict
    print(f"\nwhat this says about 17.1{partial}")
    print("-" * 78)

    def sig(cid: str) -> bool:
        return cid in corrected and bool(corrected[cid]["reject"])

    # Significance alone is the wrong test here, and an earlier version of this
    # script got it wrong: with 103 subjects the design detects ~0.020 (17.2), so
    # an arm can beat the fixed graph significantly while recovering almost none
    # of the gap to CSP. What matters is the FRACTION OF THE GAP CLOSED.
    common = sorted(set(arms["fixed graph"]) & set(arms["CSP+LDA"]))
    gap = float(np.mean([arms["CSP+LDA"][s] for s in common])
                - np.mean([arms["fixed graph"][s] for s in common]))

    def closed(cid: str) -> float | None:
        """Fraction of the fixed-graph-to-CSP gap this arm recovers."""
        if cid not in results or gap <= 0:
            return None
        return results[cid]["mean"] / gap

    print(f"  gap to close (CSP - fixed graph): {gap:+.4f}")
    for cid in ("A1-vs-fixed", "EEGNet-vs-fixed", "Shallow-vs-fixed"):
        frac = closed(cid)
        if frac is not None:
            print(f"    {results[cid]['arm']:<18} recovers {frac:6.1%} of it"
                  f"  ({'significant' if sig(cid) else 'not significant'})")

    MATERIAL = 0.5   # an arm must recover at least half the gap to count as "closes it"
    a1 = closed("A1-vs-fixed")
    cnn = max([c for c in (closed("EEGNet-vs-fixed"), closed("Shallow-vs-fixed"))
               if c is not None], default=None)
    a1_material = a1 is not None and sig("A1-vs-fixed") and a1 >= MATERIAL
    a1_minor = a1 is not None and sig("A1-vs-fixed") and a1 < MATERIAL
    cnn_material = cnn is not None and cnn >= MATERIAL

    print()
    if a1_material and cnn_material:
        print("  The graph prior was the binding constraint: lifting it recovers most of")
        print("  the gap, and a graph-free CNN does too. 9.7.1 SUPPORTED as written.")
    elif a1_minor and cnn_material:
        print("  SPLIT VERDICT. Deep learning is NOT below its data threshold here -- a")
        print("  graph-free CNN on the same ~45 trials closes the gap -- so the sample-size")
        print("  explanation (17.1) is REFUTED and the limitation is architectural.")
        print("  But freeing the spatial operator inside the GCN recovers only a small")
        print("  fraction of the gap, so **the specific mechanism 9.7.1 names -- graph")
        print("  locality and node-shared weights -- is real but minor**. Most of the gap")
        print("  lies elsewhere in the architecture. 9.7.1 must be rewritten: right about")
        print("  where the problem is, wrong about what it is.")
    elif not sig("A1-vs-fixed") and cnn_material:
        print("  Freeing the spatial operator changes nothing, yet a graph-free CNN closes")
        print("  the gap. The constraint 9.7.1 names is NOT the cause; something else in")
        print("  the architecture is. 9.7.1 is WRONG in its stated form.")
    elif not sig("A1-vs-fixed") and not cnn_material:
        print("  Neither freeing the operator nor a graph-free CNN helps. The limitation is")
        print("  not architectural at all -- sample size (17.1) is the better explanation")
        print("  and 9.7.1 is UNIDENTIFIED.")
    elif a1_material and not cnn_material:
        print("  Lifting the constraint recovers much of the gap but CNNs do not, which is")
        print("  an odd combination -- check the arms before interpreting.")
    else:
        print("  Mixed. Read the table above rather than taking a one-line verdict.")

    print("\n  Report this alongside the per-subject win rates, not instead of them:")
    print("  E1 (17.6) showed a mean difference of -0.0263 sitting inside a")
    print("  distribution running from -0.267 to +0.200.")

    if args.figures:
        make_figure(arms, args.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
