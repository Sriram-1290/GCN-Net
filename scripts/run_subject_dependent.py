"""Subject-dependent baseline: stratified trial-level k-fold, per subject.

    python scripts/run_subject_dependent.py --subjects S001 --epochs 20 --lr 1e-3
    python scripts/run_subject_dependent.py --folds 5            # every subject

This is the *correct* protocol: folds are cut at trial level, and the adjacency
and standardiser are fit on the training trials of each fold only. Pass --leaky
to reproduce the reference's protocol instead (graph and statistics over all
data, BatchNorm in train mode at test time) for the replication arm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import assert_cache_matches, load_subject  # noqa: E402
from data.config import all_subjects, load_config  # noqa: E402
from data.splits import stratified_trial_kfold  # noqa: E402
from evaluation.experiment_log import (  # noqa: E402
    completed_keys, describe_resume, log_experiment, set_seed)
from evaluation.pipeline import Trials, prepare_fold  # noqa: E402
from evaluation.trainer import TrainConfig, Trainer  # noqa: E402
from models.gcns_net import GCNsNet, GCNsNetConfig  # noqa: E402


def split_off_validation(y: np.ndarray, train_idx: np.ndarray, frac: float, seed: int
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Carve a stratified validation set out of the training trials.

    Trial level, so no timepoint of a validation trial is ever trained on.
    Needed because CLAUDE.md requires best-by-validation checkpointing.
    """
    if frac <= 0:
        return train_idx, np.array([], dtype=np.int64)
    n_folds = max(2, int(round(1.0 / frac)))
    inner = stratified_trial_kfold(y[train_idx], n_splits=n_folds, seed=seed)[0]
    return train_idx[inner.train], train_idx[inner.test]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=None,
                    help="default is the reference's 1e-6; the paper says 0.01")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--classes", nargs=2, type=int, default=None, metavar=("A", "B"),
                    help="restrict to a binary problem between two class indices, "
                         "e.g. --classes 0 1. Both classes must come from the same run "
                         "family or the run-family confound re-enters.")
    ap.add_argument("--temporal-attention", action="store_true",
                    help="attend over time before the convs; needs --window > 1")
    ap.add_argument("--attention-heads", type=int, default=4)
    ap.add_argument("--attention-per-node", action="store_true")
    ap.add_argument("--learnable-graph", action="store_true",
                    help="learn the adjacency, initialised at the PCC graph so the "
                         "model starts identical to the fixed baseline")
    ap.add_argument("--learn-graph-scale", action="store_true")
    ap.add_argument("--adjacency-l1", type=float, default=0.0,
                    help="L1 on the learned graph correction (sparsity)")
    ap.add_argument("--features", choices=["raw", "bandpower"], default="raw",
                    help="node features: raw voltage, or log band power per band. "
                         "bandpower requires an UNFILTERED cache")
    ap.add_argument("--bands", default="mu_beta",
                    help="named band set for --features bandpower "
                         "(fbcsp, mu_beta, mu_beta_fine, broad)")
    ap.add_argument("--window", type=int, default=1,
                    help="timepoints per sample; 1 is the reference's definition")
    ap.add_argument("--stride", type=int, default=None,
                    help="hop between windows; defaults to --window (no overlap)")
    ap.add_argument("--lmax", type=float, default=2.0,
                    help="2.0 reproduces the reference; pass 0 for the true lambda_max")
    ap.add_argument("--leaky", action="store_true", help="reproduce the reference protocol")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log", default="runs/experiments.jsonl")
    ap.add_argument("--tag", default="", help="free-text note stored with each row")
    ap.add_argument("--resume", action="store_true",
                    help="skip (subject, fold) pairs already logged under this tag, "
                         "so an interrupted run can be restarted with the same command")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    seed_info = set_seed(seed)
    assert_cache_matches(cfg, cfg.dataset.cache_dir)

    subjects = args.subjects or all_subjects(cfg)
    lmax = None if args.lmax == 0 else args.lmax
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr if args.lr is not None else TrainConfig.learning_rate,
        verbose=not args.quiet,
    )
    n_classes = 2 if args.classes else len(cfg.class_names)
    if args.features == "bandpower":
        from data.features import resolve_bands
        n_features = len(resolve_bands(args.bands))
        if cfg.preprocess.bandpass is not None:
            raise SystemExit(
                f"--features bandpower needs an unfiltered cache, but "
                f"{cfg.source_path} sets bandpass={cfg.preprocess.bandpass}. "
                f"A band-passed signal has no power outside its passband."
            )
    else:
        n_features = args.window
    model_cfg = GCNsNetConfig(n_classes=n_classes,
                              in_features=n_features,
                              temporal_attention=args.temporal_attention,
                              attention_heads=args.attention_heads,
                              attention_per_node=args.attention_per_node,
                              learnable_graph=args.learnable_graph,
                              learn_graph_scale=args.learn_graph_scale,
                              adjacency_l1=args.adjacency_l1,
                              always_train_bn=args.leaky)

    print(f"subjects {len(subjects)}  folds {args.folds}  epochs {args.epochs}  "
          f"lr {train_cfg.learning_rate}  lmax {lmax}  leaky {args.leaky}  seed {seed}")
    print(f"cache {cfg.dataset.cache_dir}  bandpass {cfg.preprocess.bandpass}  "
          f"window {args.window}  stride {args.stride or args.window}")
    print(f"attention {'ON (' + str(args.attention_heads) + ' heads)' if args.temporal_attention else 'off'}")
    print(f"graph {'LEARNABLE (init at PCC)' if args.learnable_graph else 'fixed PCC'}"
          + (f"  L1 {args.adjacency_l1}" if args.adjacency_l1 else ""))
    print(f"features {args.features}"
          + (f" ({args.bands}, {n_features} bands)" if args.features == "bandpower" else ""))
    print(f"device {train_cfg.device}\n")

    done = completed_keys(args.log, args.tag) if args.resume else set()
    if args.resume:
        print(describe_resume(done, len(subjects) * args.folds, args.tag))

    per_subject: list[float] = []
    per_subject_tp: list[float] = []

    for subject in subjects:
        ep = load_subject(subject, cfg.dataset.cache_dir)
        X_all, y_all = ep.X, ep.y
        if args.classes:
            # Relabel the chosen pair to {0, 1}; everything downstream is unchanged.
            mask = np.isin(y_all, args.classes)
            X_all = X_all[mask]
            y_all = (y_all[mask] == args.classes[1]).astype(np.int64)
        ep = type(ep)(subject=ep.subject, X=X_all, y=y_all, run=ep.run[:len(y_all)],
                      onset=ep.onset[:len(y_all)], ch_names=ep.ch_names, sfreq=ep.sfreq)
        fold_trial_acc, fold_tp_acc = [], []

        for split in stratified_trial_kfold(ep.y, n_splits=args.folds, seed=seed):
            if (subject, split.fold) in done:
                continue
            tr_idx, val_idx = split_off_validation(ep.y, split.train, args.val_frac, seed)
            train = Trials(X=ep.X[tr_idx], y=ep.y[tr_idx])
            val = Trials(X=ep.X[val_idx], y=ep.y[val_idx]) if len(val_idx) else None
            test = Trials(X=ep.X[split.test], y=ep.y[split.test])

            fold = prepare_fold(train, test, val, lmax=lmax, seed=seed,
                                sfreq=ep.sfreq, window=args.window, stride=args.stride,
                                features=args.features, bands=args.bands,
                                leaky=args.leaky)
            model = GCNsNet(fold.bases, model_cfg, adjacencies=fold.adjacencies)
            trainer = Trainer(model, train_cfg, n_classes=n_classes)

            history = trainer.fit(
                fold, checkpoint_path=f"checkpoints/{subject}_fold{split.fold}.pt")
            result = trainer.evaluate_samples(fold.test)

            fold_trial_acc.append(result["trial"].accuracy)
            fold_tp_acc.append(result["timepoint"].accuracy)

            log_experiment({
                "experiment": "subject_dependent",
                "tag": args.tag,
                "subject": subject,
                "fold": split.fold,
                "n_folds": args.folds,
                "protocol": "leaky_reference" if args.leaky else "trial_level",
                "split_type": "stratified_trial_kfold",
                **seed_info,
                "config_source": str(cfg.source_path),
                "preprocess": {"bandpass": cfg.preprocess.bandpass,
                               "rereference": cfg.preprocess.rereference},
                "lmax": lmax,
                "classes": args.classes,
                "n_classes": n_classes,
                "window": args.window,
                "stride": args.stride or args.window,
                "temporal_attention": args.temporal_attention,
                "attention_heads": args.attention_heads if args.temporal_attention else None,
                "learnable_graph": args.learnable_graph,
                "features": args.features,
                "bands": args.bands if args.features == "bandpower" else None,
                "n_features": n_features,
                "train_config": train_cfg,
                "model_config": model_cfg,
                "n_parameters": model.n_parameters(),
                "n_train_trials": len(train), "n_test_trials": len(test),
                "timepoint_accuracy": result["timepoint"].accuracy,
                "trial_accuracy": result["trial"].accuracy,
                "trial_macro_f1": result["trial"].macro_f1,
                "trial_kappa": result["trial"].kappa,
                "trial_per_class": result["trial"].per_class_accuracy,
                "trial_confusion": result["trial"].confusion,
                "history": history.as_dict(),
            }, path=args.log)

            print(f"{subject} fold {split.fold}: "
                  f"timepoint {result['timepoint'].accuracy:.4f}  "
                  f"trial {result['trial'].accuracy:.4f}  "
                  f"(selection: {history.selection})")

        if not fold_trial_acc:      # every fold for this subject was already done
            continue
        per_subject.append(float(np.mean(fold_trial_acc)))
        per_subject_tp.append(float(np.mean(fold_tp_acc)))
        print(f"{subject}: trial {per_subject[-1]:.4f}  timepoint {per_subject_tp[-1]:.4f}\n")

    print("=" * 64)
    if not per_subject:
        print("nothing to do: every fold for this tag was already logged")
        return 0
    print(f"subjects: {len(per_subject)}")
    print(f"mean trial accuracy     {np.mean(per_subject):.4f} "
          f"+/- {np.std(per_subject):.4f}")
    print(f"mean timepoint accuracy {np.mean(per_subject_tp):.4f} "
          f"+/- {np.std(per_subject_tp):.4f}")
    print(f"log -> {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
