"""Leave-One-Subject-Out evaluation — the subject-independent protocol.

    python scripts/run_loso.py --config configs/bandpass_8_30.yaml \
        --window 64 --stride 32 --epochs 30 --lr 1e-3 --batch-size 256

One fold per subject: train on all but one, hold that subject out entirely, and
draw a disjoint validation set of other subjects for checkpoint selection. Nothing
about the held-out subject touches training — not their trials, not the adjacency,
not the normalisation statistics.

**Why this matters here.** Neither GCNs-Net headline is subject-independent: the
93.06 % is within-subject and the 88.57 % "group level" pools subjects and then
splits *across the pool*, so the same subjects appear on both sides
(project_documentation.md 4.5). A correct LOSO evaluation of this architecture
appears to be genuinely new.

**Cost warning.** LOSO is ~100x the subject-dependent protocol: 103 folds, each
training on ~8,700 trials instead of ~58. With the reference's one-timepoint
samples that is 5.6M samples per fold and the run is infeasible — weeks. It is
only practical on a windowed or band-power representation, so this script
estimates the sample count up front and refuses obviously hopeless configurations
rather than appearing to hang.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import assert_cache_matches, load_subject  # noqa: E402
from data.config import all_subjects, load_config  # noqa: E402
from data.splits import loso_folds  # noqa: E402
from evaluation.experiment_log import (  # noqa: E402
    completed_keys, describe_resume, log_experiment, set_seed)
from evaluation.pipeline import Trials, concat_trials, prepare_fold  # noqa: E402
from evaluation.trainer import TrainConfig, Trainer  # noqa: E402
from models.gcns_net import GCNsNet, GCNsNetConfig  # noqa: E402

# Refuse above this many training samples per fold unless --force. At ~8 ms per
# 256-sample step this is already several hours for 103 folds.
SAMPLE_BUDGET = 400_000


def estimate_samples(n_trials: int, n_time: int, window: int, stride: int,
                     features: str) -> int:
    if features == "bandpower" or window > 1:
        n_windows = max(1, (n_time - window) // stride + 1)
        return n_trials * n_windows
    return n_trials * n_time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/bandpass_8_30.yaml")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="restrict the subject pool; folds are one per subject in it")
    ap.add_argument("--max-folds", type=int, default=None,
                    help="run only the first N folds, for a quick read")
    ap.add_argument("--fold-range", nargs=2, type=int, default=None, metavar=("LO", "HI"),
                    help="run folds [LO, HI) only. Folds are independent, so splitting "
                         "a run across processes changes nothing about the result -- it "
                         "is purely a scheduling device for a machine with spare GPU.")
    ap.add_argument("--val-subjects", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--features", choices=["raw", "bandpower"], default="raw")
    ap.add_argument("--bands", default="mu_beta")
    ap.add_argument("--classes", nargs=2, type=int, default=None, metavar=("A", "B"))
    ap.add_argument("--learnable-graph", action="store_true")
    ap.add_argument("--lmax", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log", default="runs/experiments.jsonl")
    ap.add_argument("--tag", default="loso")
    ap.add_argument("--resume", action="store_true",
                    help="skip subjects already logged under this tag. LOSO folds\n"
                         "cost minutes each, so an interrupted run must not restart\n"
                         "from fold 0 -- that would duplicate every completed fold.")
    ap.add_argument("--force", action="store_true",
                    help="run even if the estimated cost exceeds the budget")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    seed_info = set_seed(seed)
    assert_cache_matches(cfg, cfg.dataset.cache_dir)

    subjects = args.subjects or all_subjects(cfg)
    lmax = None if args.lmax == 0 else args.lmax

    # Load every subject once. Re-reading the cache inside 103 folds would mean
    # ~10,000 decompressions for no reason.
    print(f"loading {len(subjects)} subjects...", flush=True)
    trials: dict[str, Trials] = {}
    for s in subjects:
        ep = load_subject(s, cfg.dataset.cache_dir)
        X, y = ep.X, ep.y
        if args.classes:
            mask = np.isin(y, args.classes)
            X, y = X[mask], (y[mask] == args.classes[1]).astype(np.int64)
        trials[s] = Trials(X=X, y=y)
    sfreq = ep.sfreq
    n_time = ep.X.shape[2]

    n_classes = 2 if args.classes else len(cfg.class_names)
    if args.features == "bandpower":
        from data.features import resolve_bands
        n_features = len(resolve_bands(args.bands))
        if cfg.preprocess.bandpass is not None:
            raise SystemExit("--features bandpower needs an unfiltered cache")
    else:
        n_features = args.window

    folds = loso_folds(subjects, n_val_subjects=args.val_subjects, seed=seed)
    if args.fold_range:
        lo, hi = args.fold_range
        folds = [f for f in folds if lo <= f.fold < hi]
    if args.max_folds:
        folds = folds[:args.max_folds]

    n_train_trials = sum(len(t) for t in trials.values()) - len(trials[subjects[0]])
    est = estimate_samples(n_train_trials, n_time, args.window, args.stride, args.features)
    print(f"\nLOSO: {len(folds)} folds over {len(subjects)} subjects")
    print(f"  per fold: ~{n_train_trials:,} training trials -> ~{est:,} samples")
    print(f"  features {args.features}  window {args.window}  stride {args.stride}  "
          f"graph {'learnable' if args.learnable_graph else 'fixed'}")
    if est > SAMPLE_BUDGET and not args.force:
        raise SystemExit(
            f"\nestimated {est:,} training samples per fold exceeds the budget of "
            f"{SAMPLE_BUDGET:,}. At {len(folds)} folds this would run for days.\n"
            f"Use a larger --stride, a windowed or bandpower representation, or "
            f"--force to override."
        )

    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                            learning_rate=args.lr, verbose=False)
    model_cfg = GCNsNetConfig(n_classes=n_classes, in_features=n_features,
                              learnable_graph=args.learnable_graph)

    done = completed_keys(args.log, args.tag) if args.resume else set()
    if args.resume:
        print(describe_resume(done, len(folds), args.tag))
    done_subjects = {s for s, _ in done}

    accs, trial_accs = [], []
    t0 = time.time()

    for fold in folds:
        subject = fold.test[0]
        if subject in done_subjects:
            continue
        train = concat_trials([trials[s] for s in fold.train])
        val = concat_trials([trials[s] for s in fold.val])
        test = trials[subject]

        prepared = prepare_fold(train, test, val, lmax=lmax, seed=seed, sfreq=sfreq,
                                window=args.window, stride=args.stride,
                                features=args.features, bands=args.bands)
        model = GCNsNet(prepared.bases, model_cfg, adjacencies=prepared.adjacencies)
        trainer = Trainer(model, train_cfg, n_classes=n_classes)
        history = trainer.fit(prepared)
        result = trainer.evaluate_samples(prepared.test)

        accs.append(result["timepoint"].accuracy)
        trial_accs.append(result["trial"].accuracy)

        log_experiment({
            "experiment": "loso",
            "tag": args.tag,
            "subject": subject,
            "fold": fold.fold,
            "protocol": "leave_one_subject_out",
            "split_type": "loso_folds",
            "n_train_subjects": len(fold.train),
            "n_val_subjects": len(fold.val),
            "val_subjects": list(fold.val),
            **seed_info,
            "config_source": str(cfg.source_path),
            "preprocess": {"bandpass": cfg.preprocess.bandpass,
                           "rereference": cfg.preprocess.rereference},
            "lmax": lmax, "window": args.window, "stride": args.stride,
            "features": args.features,
            "bands": args.bands if args.features == "bandpower" else None,
            "classes": args.classes, "n_classes": n_classes,
            "learnable_graph": args.learnable_graph,
            "train_config": train_cfg, "model_config": model_cfg,
            "n_parameters": model.n_parameters(),
            "sample_accuracy": result["timepoint"].accuracy,
            "trial_accuracy": result["trial"].accuracy,
            "trial_macro_f1": result["trial"].macro_f1,
            "trial_kappa": result["trial"].kappa,
            "trial_per_class": result["trial"].per_class_accuracy,
            "trial_confusion": result["trial"].confusion,
            "history": history.as_dict(),
        }, path=args.log)

        elapsed = time.time() - t0
        eta = elapsed / len(accs) * (len(folds) - len(accs))
        print(f"[{len(accs):3d}/{len(folds)}] held out {subject}: "
              f"sample {result['timepoint'].accuracy:.4f}  "
              f"trial {result['trial'].accuracy:.4f}   "
              f"eta {eta/60:.0f} min", flush=True)

    if not trial_accs:
        print("nothing to do: every fold for this tag was already logged")
        return 0

    a = np.array(trial_accs)
    rng = np.random.default_rng(0)
    boot = a[rng.integers(0, len(a), size=(10_000, len(a)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    chance = 1.0 / n_classes
    print("\n" + "=" * 68)
    print(f"LOSO trial accuracy  {a.mean():.4f} +/- {a.std(ddof=1):.4f}  "
          f"95% CI [{lo:.4f}, {hi:.4f}]   chance {chance:.4f}")
    print(f"subjects above chance: {int((a > chance).sum())}/{len(a)}")
    print(f"{'ABOVE CHANCE' if lo > chance else 'NOT distinguishable from chance'}")
    print(f"total {(time.time()-t0)/60:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
