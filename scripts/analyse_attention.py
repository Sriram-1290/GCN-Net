"""What did temporal attention actually learn?

    python scripts/analyse_attention.py --subjects S001 S002 ... --out figures

H2 is the project's one positive result: attention gains +0.0600 within-family
while a learnable graph gains nothing (project_documentation.md 9.9). A p-value
says it helped; it does not say *what* it learned. This produces the mechanistic
evidence.

Two questions, two panels:

1. **Within-window profile.** Attention weights over the 64 timepoints (0.4 s) of a
   sample. Uniform weights would mean attention is doing nothing a fixed average
   could not, and the H2 gain would need another explanation.
2. **Decodability across the trial.** Each trial is cut into overlapping windows at
   stride 16, so window *k* starts at 16k samples (100k ms) into the 4 s trial.
   Accuracy per window position says *when* in a trial the imagery is decodable.
   Motor imagery ERD builds after the cue and decays, so a peak in the first
   1-2 s is the neurophysiologically expected shape — and finding it would be
   independent evidence the model latched onto real signal rather than artefact.

Uses the confirmatory configuration throughout (8-30 Hz, window 64, stride 16,
50 epochs, Adam 1e-3, batch 256) so the figure describes the model that produced
the H2 result rather than a different one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import load_subject  # noqa: E402
from data.config import all_subjects, load_config  # noqa: E402
from data.splits import stratified_trial_kfold  # noqa: E402
from evaluation.experiment_log import set_seed  # noqa: E402
from evaluation.pipeline import Trials, prepare_fold  # noqa: E402
from evaluation.trainer import TrainConfig, Trainer  # noqa: E402
from models.gcns_net import GCNsNet, GCNsNetConfig  # noqa: E402

WINDOW, STRIDE, SFREQ = 64, 16, 160.0


def window_positions(n_time: int) -> np.ndarray:
    """Start time in seconds of each window within a trial."""
    return np.arange(0, n_time - WINDOW + 1, STRIDE) / SFREQ


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/bandpass_8_30.yaml")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--n-subjects", type=int, default=12)
    ap.add_argument("--classes", nargs=2, type=int, default=[2, 3])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    subjects = args.subjects or all_subjects(cfg)[:args.n_subjects]

    all_weights = []          # (samples, time, heads) per subject
    acc_by_position = []      # accuracy per window position per subject

    for i, subject in enumerate(subjects, 1):
        ep = load_subject(subject, cfg.dataset.cache_dir)
        mask = np.isin(ep.y, args.classes)
        X = ep.X[mask]
        y = (ep.y[mask] == args.classes[1]).astype(np.int64)

        split = stratified_trial_kfold(y, n_splits=5, seed=args.seed)[0]
        fold = prepare_fold(Trials(X=X[split.train], y=y[split.train]),
                            Trials(X=X[split.test], y=y[split.test]),
                            sfreq=ep.sfreq, window=WINDOW, stride=STRIDE, seed=args.seed)

        model = GCNsNet(fold.bases, GCNsNetConfig(
            n_classes=2, in_features=WINDOW, temporal_attention=True,
            attention_heads=args.heads))
        trainer = Trainer(model, TrainConfig(epochs=args.epochs, batch_size=256,
                                             learning_rate=1e-3, verbose=False),
                          n_classes=2)
        trainer.fit(fold)

        logits = trainer.predict_logits(fold.test)
        pred = logits.argmax(axis=1)

        # Attention weights from the final evaluation batch chain: re-run in eval
        # mode over the whole test split so every sample contributes.
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(fold.test.X).to(trainer.cfg.device)
            model(xb)
            w = model.attention.last_attention().cpu().numpy()
        all_weights.append(w)

        # Window position within the trial: windows are emitted in order per trial,
        # so the k-th window of a trial starts at k * STRIDE.
        n_windows = len(window_positions(X.shape[2]))
        pos = np.tile(np.arange(n_windows), len(fold.test.y) // n_windows)
        correct = (pred == fold.test.y).astype(float)
        acc_by_position.append([correct[pos == k].mean() for k in range(n_windows)])
        print(f"[{i:2d}/{len(subjects)}] {subject}: {len(fold.test)} test windows, "
              f"acc {correct.mean():.3f}", flush=True)

    W = np.concatenate(all_weights, axis=0)      # (samples, time, heads)
    A = np.array(acc_by_position)                # (subjects, n_windows)
    times = np.arange(WINDOW) / SFREQ * 1000     # ms within a window
    starts = window_positions(640)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    uniform = 1.0 / WINDOW
    # Plotting the population mean here would be actively misleading: individual
    # trials are strongly peaked but at *different* moments, so the average across
    # samples comes out nearly flat. Show individual profiles and the spread.
    rng = np.random.default_rng(0)
    for idx in rng.choice(len(W), size=min(40, len(W)), replace=False):
        axes[0].plot(times, W[idx, :, 0], lw=0.6, color="#3B44A8", alpha=0.28)
    axes[0].plot(times, W[:, :, 0].mean(axis=0), lw=2.4, color="#A63D34",
                 label="mean across samples")
    axes[0].axhline(uniform, ls="--", lw=1.2, color="0.35", label="uniform")
    pop_dev = float(np.abs(W.mean(axis=0) - uniform).max() / uniform)
    per_sample = np.abs(W - uniform).max(axis=1) / uniform
    ratio = float(np.median(W.max(axis=1) / W.min(axis=1)))
    axes[0].set_xlabel("time within the 0.4 s window (ms)")
    axes[0].set_ylabel("attention weight (head 0)")
    axes[0].set_title(
        f"Selective per trial, not on average\n"
        f"per-sample deviation {np.median(per_sample):.0%} (median), max/min "
        f"{ratio:.1f}x; population mean only {pop_dev:.0%}", fontsize=10)
    axes[0].legend(fontsize=7, frameon=False)

    m, sd = A.mean(axis=0), A.std(axis=0)
    axes[1].plot(starts, m, lw=1.8, color="#3B44A8")
    axes[1].fill_between(starts, m - sd / np.sqrt(len(A)), m + sd / np.sqrt(len(A)),
                         alpha=0.2, color="#3B44A8")
    axes[1].axhline(0.5, ls="--", lw=1.0, color="0.4")
    peak = starts[int(np.argmax(m))]
    axes[1].axvline(peak, ls=":", lw=1.0, color="#A63D34")
    axes[1].set_xlabel("window start within the 4 s trial (s)")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title(f"When in a trial the imagery is decodable\n"
                      f"peak at {peak:.2f} s  (n={len(A)} subjects, chance 0.5)",
                      fontsize=10)

    fig.suptitle("Temporal attention: what it learned and when the signal is",
                 fontsize=12)
    fig.tight_layout()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "attention_analysis.png", dpi=200)
    plt.close(fig)

    print(f"\nattention: population-mean deviation {pop_dev:.1%}, per-sample "
          f"median {np.median(per_sample):.1%}, max/min {ratio:.2f}x")
    print(f"decodability peaks at {peak:.2f} s into the trial "
          f"(range {m.min():.3f} - {m.max():.3f})")
    print(f"-> {out / 'attention_analysis.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
