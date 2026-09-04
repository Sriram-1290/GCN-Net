"""Is the mean the right statistic? (E1, project_documentation.md 17.6)

    python scripts/subject_heterogeneity.py
    python scripts/subject_heterogeneity.py --figures figures

Every headline in this project is a mean over 103 subjects, and the confirmatory
table compares those means. That framing assumes the choice of method is the
dominant source of variation. This script tests that assumption against the
experiment log, and it needs **no compute** — the rows already exist.

The three quantities that matter:

* **Win rate.** On how many subjects does our model actually beat CSP? A mean
  difference of −0.0263 is compatible with "CSP is uniformly better" and with
  "the two are interleaved and CSP wins the average by a margin on a few
  subjects". Those imply very different conclusions and the mean cannot separate
  them.
* **Correlation across methods.** If per-subject accuracies of two structurally
  unrelated methods correlate strongly, then what varies is the *subject*, not
  the algorithm — which is the signature of a subject-level ceiling (17.6 E2,
  and the BCI-illiteracy literature).
* **Variance ratio.** Between-subject SD against the method gap, as a blunt
  statement of which factor the study is actually measuring.

External context, on this exact dataset: a Friedman-Nemenyi benchmark of 1,056
configurations finds the single best pipeline optimal for only 35 % of
PhysionetMI participants, and per-subject matching worth ~7 accuracy points --
versus the 0.0263 under discussion here.

Nothing in this script is confirmatory. 10.3 fixed the hypothesis list in advance
and this is not on it; it is a descriptive re-analysis of a pre-registered
comparison, and it is labelled exploratory wherever it is reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_LOG = Path("runs/experiments.jsonl")

# (label, tag prefix, accuracy field). The network arms log both a timepoint and a
# trial accuracy; the trial one is what CSP is scored on, so it is the only fair
# comparison -- see evaluation/metrics.py and confirmatory_results.py.
ARMS = {
    "fixed": ("C-fixed-binary", "trial_accuracy"),
    "attention": ("C-attention-binary", "trial_accuracy"),
    "learngraph": ("C-learngraph-binary", "trial_accuracy"),
    "both": ("C-both-binary", "trial_accuracy"),
    "csp": ("C-csp", "accuracy"),
    "shallow": ("A2-shallow-binary", "trial_accuracy"),
    "gbp": ("G3-gbp-prior-binary", "trial_accuracy"),
    "ptg": ("PTG-gbp-binary", "trial_accuracy"),
    "ptc": ("PTC-gbp-binary", "trial_accuracy"),
}


def per_subject(rows: list[dict], prefix: str, field: str) -> dict[str, float]:
    """Mean accuracy per subject over every fold and both binary problems.

    The two within-family problems are averaged together because that is the
    pre-registered primary outcome (10.2); doing it per subject keeps the pairing
    unit the subject, as 10.4 requires.
    """
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if str(r.get("tag", "")).startswith(prefix) and r.get(field) is not None:
            acc[r["subject"]].append(float(r[field]))
    return {s: float(np.mean(v)) for s, v in acc.items()}


def bootstrap_ci(values: np.ndarray, statistic, n_boot: int = 10_000,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap over subjects, matching evaluation/metrics.py."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    draws = np.array([statistic(values[i]) for i in idx])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def compare(name_a: str, name_b: str, arms: dict[str, dict[str, float]]) -> dict | None:
    a, b = arms.get(name_a), arms.get(name_b)
    if not a or not b:
        return None
    subjects = sorted(set(a) & set(b))
    if not subjects:
        return None

    va = np.array([a[s] for s in subjects])
    vb = np.array([b[s] for s in subjects])
    d = va - vb

    wins = int((d > 0).sum())
    ties = int(np.isclose(d, 0).sum())
    win_rate = wins / len(subjects)
    lo, hi = bootstrap_ci(d, lambda x: float((x > 0).mean()))

    # Pearson r; both arms are per-subject means over the same folds.
    r = float(np.corrcoef(va, vb)[0, 1])

    return {
        "subjects": len(subjects), "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
        "gap": float(d.mean()), "wins": wins, "ties": ties, "win_rate": win_rate,
        "win_ci": (lo, hi), "r": r, "sd_between": float(va.std(ddof=1)),
        "quartiles": tuple(float(q) for q in np.percentile(d, [0, 25, 50, 75, 100])),
    }


EXPECTED_SUBJECTS = 103


def report(res: dict, label_a: str, label_b: str) -> None:
    # Error 10: an arm read 0.5774 at 35 of 100 folds and 0.5418 when finished. A
    # comparison over a partial arm is not a small version of the real one, so it is
    # labelled unmistakably rather than printed quietly beside the complete ones.
    partial = ("   [PARTIAL -- DO NOT QUOTE]"
               if res["subjects"] < EXPECTED_SUBJECTS else "")
    print(f"\n{label_a}  vs  {label_b}      ({res['subjects']} subjects){partial}")
    print("-" * 66)
    print(f"  mean accuracy            {res['mean_a']:.4f}  vs  {res['mean_b']:.4f}   "
          f"(gap {res['gap']:+.4f})")
    print(f"  {label_a} wins on         {res['wins']}/{res['subjects']} subjects "
          f"({res['win_rate']:.0%})  95% CI [{res['win_ci'][0]:.0%}, {res['win_ci'][1]:.0%}]"
          + (f", {res['ties']} ties" if res["ties"] else ""))
    print(f"  Pearson r across methods {res['r']:+.3f}"
          + ("   <- the subject dominates, not the method" if res["r"] > 0.6 else ""))
    print(f"  between-subject SD       {res['sd_between']:.4f}"
          f"   ({(res['sd_between'] / abs(res['gap'])) ** 2:.0f}x the variance of the gap)"
          if abs(res["gap"]) > 1e-9 else "")
    q = res["quartiles"]
    print(f"  per-subject difference   min {q[0]:+.3f}  q1 {q[1]:+.3f}  median {q[2]:+.3f}  "
          f"q3 {q[3]:+.3f}  max {q[4]:+.3f}")


def make_figure(arms: dict[str, dict[str, float]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    net_key = next(k for k in ("ptg", "gbp", "attention") if k in arms)
    net, csp = arms[net_key], arms["csp"]
    subjects = sorted(set(net) & set(csp))
    va = np.array([net[s] for s in subjects])
    vb = np.array([csp[s] for s in subjects])
    d = np.sort(va - vb)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    ax = axes[0]
    ax.scatter(vb, va, s=22, alpha=.75, edgecolor="none", color="#1F4D8F")
    lim = [min(va.min(), vb.min()) - .02, max(va.max(), vb.max()) + .02]
    ax.plot(lim, lim, color="#8A2F2A", lw=1, ls="--", label="equal accuracy")
    ax.axhline(.5, color="#999", lw=.7, ls=":")
    ax.axvline(.5, color="#999", lw=.7, ls=":")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("CSP+LDA, within-family accuracy")
    ax.set_ylabel("GCN + temporal attention")
    ax.set_title(f"Per-subject, r = {np.corrcoef(va, vb)[0, 1]:.3f}\n"
                 f"network wins on {int((va > vb).sum())}/{len(subjects)} subjects",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    colors = ["#1F4D8F" if x > 0 else "#8A2F2A" for x in d]
    ax.bar(range(len(d)), d, color=colors, width=1.0)
    ax.axhline(0, color="#333", lw=.8)
    ax.axhline(d.mean(), color="#1B6B4A", lw=1.2, ls="--",
               label=f"mean {d.mean():+.4f}")
    ax.set_xlabel("subjects, sorted by difference")
    ax.set_ylabel("network − CSP")
    ax.set_title("The mean hides the spread", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("Subject identity dominates method choice (exploratory, 17.6)",
                 fontsize=11)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "subject_heterogeneity.png", dpi=200, bbox_inches="tight")
    print(f"\nwrote {out / 'subject_heterogeneity.png'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--figures", type=Path, default=None,
                    help="directory to write subject_heterogeneity.png into")
    args = ap.parse_args()

    if not args.log.exists():
        print(f"no experiment log at {args.log}", file=sys.stderr)
        return 1

    rows = [json.loads(l) for l in args.log.open(encoding="utf-8") if l.strip()]
    arms = {name: per_subject(rows, prefix, field)
            for name, (prefix, field) in ARMS.items()}
    arms = {k: v for k, v in arms.items() if v}

    print("EXPLORATORY re-analysis of the confirmatory arms (17.6 E1).")
    print("Not a pre-registered test -- 10.3 fixed that list and this is not on it.")
    print(f"\narms found: {', '.join(f'{k} (n={len(v)})' for k, v in arms.items())}")
    short = [k for k, v in arms.items() if len(v) < EXPECTED_SUBJECTS]
    if short:
        print(f"INCOMPLETE, marked PARTIAL below: {', '.join(short)} "
              f"(expected {EXPECTED_SUBJECTS} subjects)")

    # Each pair answers one question, and every pair is reported whether or not it
    # flatters us. Arms absent from the log are skipped rather than faked.
    for a, b, la, lb in [("attention", "csp", "GCN + attention", "CSP+LDA"),
                         ("both", "csp", "GCN + both", "CSP+LDA"),
                         ("attention", "fixed", "GCN + attention", "fixed graph"),
                         ("gbp", "csp", "G3 reordered", "CSP+LDA"),
                         ("gbp", "attention", "G3 reordered", "GCN + attention"),
                         ("gbp", "shallow", "G3 reordered", "ShallowConvNet"),
                         ("ptg", "ptc", "pre-trained", "no-pretrain control"),
                         ("ptg", "shallow", "pre-trained", "ShallowConvNet")]:
        res = compare(a, b, arms)
        if res:
            report(res, la, lb)

    if "csp" in arms and any(k in arms for k in ("ptg", "gbp", "attention")):
        net_key = next(k for k in ("ptg", "gbp", "attention") if k in arms)
        net, csp = arms[net_key], arms["csp"]
        subjects = sorted(set(net) & set(csp))
        below = [s for s in subjects if max(net[s], csp[s]) <= 0.5]
        print(f"\nsubjects where BOTH methods are at or below chance: "
              f"{len(below)}/{len(subjects)} ({len(below)/len(subjects):.0%})")
        print("  consistent with the 15-30% BCI-illiteracy rate; a subject-level")
        print("  ceiling no architecture can lift. See 17.6 E2.")

    if args.figures:
        if "attention" in arms and "csp" in arms:
            make_figure(arms, args.figures)
        else:
            print("\nno figure: needs both the attention and CSP arms", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
