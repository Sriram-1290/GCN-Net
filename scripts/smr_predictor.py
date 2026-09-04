"""Does resting sensorimotor rhythm predict who is decodable? (E2, 17.6)

    python scripts/smr_predictor.py --figures figures

**Why this exists.** 17.6 found that per-subject accuracies of two structurally unrelated
methods — our repaired graph network and CSP+LDA — correlate at *r* = 0.825, and that
between-subject SD (0.1247) carries 23-38x the variance of any difference between methods.
Eleven of 103 subjects sit at or below chance under *both*. That pattern is the signature of
a **subject-level ceiling**: something about the person, not the algorithm.

The literature has a name and a measure for it. Between 15 % and 30 % of people do not
produce classifiable sensorimotor rhythms even after training, and the standard predictor is
the strength of the resting mu rhythm relative to the 1/f background — Blankertz et al.'s
SMR predictor. If it explains a meaningful share of our between-subject variance, then part
of what looks like model failure is a ceiling no architecture can lift, and the limitations
section can say so with a number instead of a citation.

**Data.** Run R01, the eyes-open baseline. It is downloaded, it has never been used by any
experiment in this project, and it contains no motor imagery — so nothing here can leak into
a decoding result. It is not in the cache (the cache holds imagery runs only), so this reads
the raw EDFs.

**Method.** Common average reference, Welch PSD, then a 1/f fit in log-log space over
3-40 Hz *excluding* 7-14 Hz, and the predictor is the largest excess of the true spectrum
over that fit within 8-13 Hz, in dB. Taking the excess over a fitted background rather than
raw band power is what separates a genuine mu peak from a subject who simply has more power
everywhere. Averaged over C3, Cz and C4 — the sensorimotor electrodes.

**This is exploratory.** It is not on the pre-registered list (10.3) and was devised after
seeing the per-subject correlations.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.config import all_subjects, load_config  # noqa: E402
from data.edf_loader import load_run_raw, run_path  # noqa: E402

BASELINE_RUN = 1          # R01, eyes open
SENSORIMOTOR = ("C3", "Cz", "C4")   # the loader normalises EEGMMIDB's padded names
MU = (8.0, 13.0)
FIT_BAND = (3.0, 40.0)
FIT_EXCLUDE = (7.0, 14.0)


def smr_predictor(raw, picks: list[int]) -> float:
    """Largest dB excess over a fitted 1/f background within the mu band.

    Args:
        raw: an MNE Raw for the baseline run, already re-referenced.
        picks: indices of the sensorimotor channels.

    Returns:
        The predictor in dB, averaged over ``picks``. Higher means a more
        prominent resting mu rhythm.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spectrum = raw.compute_psd(method="welch", fmin=1.0, fmax=45.0,
                                   n_fft=int(raw.info["sfreq"] * 2), verbose=False)
    psd, freqs = spectrum.get_data(return_freqs=True)

    fit_mask = ((freqs >= FIT_BAND[0]) & (freqs <= FIT_BAND[1])
                & ~((freqs >= FIT_EXCLUDE[0]) & (freqs <= FIT_EXCLUDE[1])))
    mu_mask = (freqs >= MU[0]) & (freqs <= MU[1])

    out = []
    for ch in picks:
        p = psd[ch]
        if not np.all(np.isfinite(p)) or np.any(p <= 0):
            continue
        # 1/f is linear in log-log; fit the background, then measure the peak above it.
        coef = np.polyfit(np.log10(freqs[fit_mask]), np.log10(p[fit_mask]), 1)
        background = 10 ** np.polyval(coef, np.log10(freqs))
        excess_db = 10 * np.log10(p[mu_mask] / background[mu_mask])
        out.append(float(excess_db.max()))
    return float(np.mean(out)) if out else float("nan")


def per_subject_accuracy(rows: list[dict], prefix: str, field: str) -> dict[str, float]:
    acc = defaultdict(list)
    for r in rows:
        if str(r.get("tag", "")).startswith(prefix) and r.get(field) is not None:
            acc[r["subject"]].append(float(r[field]))
    return {s: float(np.mean(v)) for s, v in acc.items()}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/bandpass_8_30.yaml"))
    ap.add_argument("--log", type=Path, default=Path("runs/experiments.jsonl"))
    ap.add_argument("--figures", type=Path, default=None)
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    subjects = args.subjects or all_subjects(cfg)
    print("EXPLORATORY (17.6 E2). Not pre-registered; devised after seeing 17.6's")
    print("per-subject correlations. Baseline run R01 only -- no imagery, so no leak.\n")

    scores: dict[str, float] = {}
    for i, subject in enumerate(subjects, 1):
        path = run_path(Path(cfg.dataset.root), subject, BASELINE_RUN)
        if not path.exists():
            print(f"  {subject}: no R01, skipped")
            continue
        try:
            raw = load_run_raw(path, cfg)
        except Exception as exc:                      # noqa: BLE001
            print(f"  {subject}: unreadable ({exc}), skipped")
            continue
        names = list(raw.ch_names)
        picks = [names.index(c) for c in SENSORIMOTOR if c in names]
        if not picks:
            print(f"  {subject}: no sensorimotor channels among {names[:4]}..., skipped")
            continue
        scores[subject] = smr_predictor(raw, picks)
        if i % 20 == 0:
            print(f"  ...{i}/{len(subjects)}")

    print(f"\nSMR predictor computed for {len(scores)} subjects")
    v = np.array(list(scores.values()))
    print(f"  range {v.min():.2f} to {v.max():.2f} dB, median {np.median(v):.2f}")

    rows = [json.loads(l) for l in args.log.open(encoding="utf-8") if l.strip()]
    # Every complete arm, so the correlation can be read across the whole ladder: if
    # the predictor tracks a subject-level ceiling it should hold for all of them, not
    # only the ones that existed when this was written.
    arms = {
        "GCN + attention": per_subject_accuracy(rows, "C-attention-binary", "trial_accuracy"),
        "CSP+LDA": per_subject_accuracy(rows, "C-csp", "accuracy"),
        "ShallowConvNet": per_subject_accuracy(rows, "A2-shallow-binary", "trial_accuracy"),
        "G3 reordered": per_subject_accuracy(rows, "G3-gbp-prior-binary", "trial_accuracy"),
        "pre-trained": per_subject_accuracy(rows, "PTG-gbp-binary", "trial_accuracy"),
    }
    arms = {k: v for k, v in arms.items() if len(v) >= 50}

    print("\ncorrelation of the resting SMR predictor with per-subject accuracy")
    print("-" * 66)
    results = {}
    for name, acc in arms.items():
        common = sorted(set(scores) & set(acc))
        if len(common) < 10:
            continue
        x = np.array([scores[s] for s in common])
        y = np.array([acc[s] for s in common])
        r = pearson(x, y)
        results[name] = (x, y, r)
        print(f"  {name:<18} r = {r:+.3f}   r^2 = {r*r:.3f}   "
              f"({r*r:.0%} of between-subject variance)  n = {len(common)}")

    # the subjects nobody can decode
    if "CSP+LDA" in arms and "GCN + attention" in arms:
        both = sorted(set(arms["CSP+LDA"]) & set(arms["GCN + attention"]) & set(scores))
        failing = [s for s in both
                   if max(arms["CSP+LDA"][s], arms["GCN + attention"][s]) <= 0.5]
        if failing:
            f_smr = np.mean([scores[s] for s in failing])
            o_smr = np.mean([scores[s] for s in both if s not in failing])
            print(f"\n  subjects at or below chance under BOTH methods: {len(failing)}/{len(both)}")
            print(f"    their mean SMR predictor {f_smr:.2f} dB vs {o_smr:.2f} dB for the rest")
            print(f"    difference {f_smr - o_smr:+.2f} dB")

    if args.figures and results:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4),
                                 squeeze=False)
        for ax, (name, (x, y, r)) in zip(axes[0], results.items()):
            ax.scatter(x, y, s=20, alpha=.7, color="#1F4D8F", edgecolor="none")
            if len(x) > 2:
                b = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, np.polyval(b, xs), color="#8A2F2A", lw=1.2)
            ax.axhline(.5, color="#999", lw=.8, ls=":")
            ax.set_xlabel("resting SMR predictor (dB)")
            ax.set_ylabel("within-family accuracy")
            ax.set_title(f"{name}\nr = {r:+.3f}", fontsize=10)
        fig.suptitle("Resting mu prominence against decodability (exploratory, 17.6 E2)",
                     fontsize=11)
        fig.tight_layout()
        args.figures.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figures / "smr_predictor.png", dpi=200, bbox_inches="tight")
        print(f"\nwrote {args.figures / 'smr_predictor.png'}")

    out = Path("runs/smr_predictor.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(scores, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
