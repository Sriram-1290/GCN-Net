"""Reproduce the reference protocol, to measure the leakage rather than assert it.

    python scripts/run_leaky_replication.py --subjects S001 S002 --epochs 25 --lr 1e-2

Runs the published protocol exactly: adjacency and standardiser over all trials,
`randperm` over timepoints then a 90/10 cut, and BatchNorm left in training mode
at test. **Its output is never our result.** It is the upper arm of the
comparison described in project_documentation.md 7.2; the lower arm is
`run_subject_dependent.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import assert_cache_matches, load_subject  # noqa: E402
from data.config import all_subjects, load_config  # noqa: E402
from data.splits import count_shared_trials  # noqa: E402
from evaluation.experiment_log import (  # noqa: E402
    completed_keys, describe_resume, log_experiment, set_seed)
from evaluation.pipeline import Trials, prepare_leaky_fold  # noqa: E402
from evaluation.trainer import TrainConfig, Trainer  # noqa: E402
from models.gcns_net import GCNsNet, GCNsNetConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--lmax", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log", default="runs/experiments.jsonl")
    ap.add_argument("--tag", default="leaky-replication")
    ap.add_argument("--resume", action="store_true",
                    help="skip subjects already logged under this tag")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    seed_info = set_seed(seed)
    assert_cache_matches(cfg, cfg.dataset.cache_dir)

    subjects = args.subjects or all_subjects(cfg)
    lmax = None if args.lmax == 0 else args.lmax
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                            learning_rate=args.lr, verbose=False)
    model_cfg = GCNsNetConfig(n_classes=len(cfg.class_names), always_train_bn=True)

    print("REFERENCE PROTOCOL REPLICATION -- these numbers are not our result.")
    print(f"subjects {len(subjects)}  epochs {args.epochs}  lr {args.lr}  seed {seed}")
    print(f"device {train_cfg.device}\n")

    done = completed_keys(args.log, args.tag) if args.resume else set()
    if args.resume:
        print(describe_resume(done, len(subjects), args.tag))

    accs: list[float] = []
    for i, subject in enumerate(subjects, 1):
        if any(s == subject for s, _ in done):
            continue
        ep = load_subject(subject, cfg.dataset.cache_dir)
        fold = prepare_leaky_fold(Trials(X=ep.X, y=ep.y), test_frac=args.test_frac,
                                  lmax=lmax, seed=seed)
        shared = count_shared_trials(np.unique(fold.train.trial_id),
                                     np.unique(fold.test.trial_id), 1)

        model = GCNsNet(fold.bases, model_cfg)
        trainer = Trainer(model, train_cfg, n_classes=len(cfg.class_names))
        history = trainer.fit(fold)
        result = trainer.evaluate_samples(fold.test)
        accs.append(result["timepoint"].accuracy)

        log_experiment({
            "experiment": "leaky_replication",
            "tag": args.tag,
            "subject": subject,
            "protocol": "leaky_reference",
            "split_type": "leaky_timepoint_holdout",
            "leakage_paths": ["timepoint_shuffled_split", "adjacency_over_all_data",
                              "standardizer_over_all_data", "batchnorm_train_at_test"],
            "trials_in_both_splits": int(shared),
            "n_trials": int(len(ep.X)),
            **seed_info,
            "lmax": lmax,
            "train_config": train_cfg,
            "model_config": model_cfg,
            "timepoint_accuracy": result["timepoint"].accuracy,
            "timepoint_macro_f1": result["timepoint"].macro_f1,
            "timepoint_kappa": result["timepoint"].kappa,
            "timepoint_per_class": result["timepoint"].per_class_accuracy,
            "timepoint_confusion": result["timepoint"].confusion,
            "final_train_acc": history.epochs[-1].train_acc,
        }, path=args.log)

        print(f"[{i:3d}/{len(subjects)}] {subject}: timepoint {result['timepoint'].accuracy:.4f} "
              f"(train {history.epochs[-1].train_acc:.4f}, "
              f"{shared}/{len(ep.X)} trials on both sides)")

    print("\n" + "=" * 64)
    if not accs:
        print("nothing to do: every subject for this tag was already logged")
        return 0
    print(f"mean timepoint accuracy {np.mean(accs):.4f} +/- {np.std(accs):.4f} "
          f"over {len(accs)} subjects")
    print("Reported as a replication of the published protocol, NOT as our result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
