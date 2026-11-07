"""A5: the three GCNs-Net leakage paths, switched on one at a time (docs 9.26.3).

    python scripts/run_leak_ablation.py --leaks none  --tag PA-A5-leak-none  --resume
    python scripts/run_leak_ablation.py --leaks split --tag PA-A5-leak-split --resume
    python scripts/run_leak_ablation.py --leaks bn    --tag PA-A5-leak-bn    --resume
    python scripts/run_leak_ablation.py --leaks graph --tag PA-A5-leak-graph --resume
    python scripts/run_leak_ablation.py --leaks all   --tag PA-A5-leak-all   --resume

Every arm is the published formulation (window 1, the reference GCNs-Net, 4-class,
``configs/default.yaml``: no band-pass, CAR), trained with the settings the logged
reference rows used (25 epochs, lr 0.01, batch 1024) and seed 42. Only the leak
toggles differ:

* ``split``: random timepoint holdout (test fraction 0.1) instead of the trial-level
  5-fold; graph and standardiser fitted on the training timepoints only.
* ``bn``: BatchNorm in training mode at test time (``always_train_bn``).
* ``graph``: adjacency **and** standardiser fitted on the subject's pooled train, val
  and test trials. Bundled on purpose; the experiment cannot separate the two.
* ``all``: all three, which must reproduce ``prepare_leaky_fold`` + ``always_train_bn``.
* ``none``: the correct protocol, exactly as ``run_subject_dependent.py`` runs it.

Rows: one per (subject, fold) for the trial-level arms (``none``, ``bn``, ``graph``),
one per subject for the timepoint-holdout arms (``split``, ``all``). Each row carries
``timepoint_confusion`` and ``trial_confusion``. The trial confusion matrix aggregates
each trial's test timepoints by the **mean logit** (the same rule that produced the
logged ``P1-A`` trial matrices); in the holdout arms a trial is scored on its ~10 %
held-out timepoints only. ``scripts/run_family_analysis.py`` decomposes either matrix.

**The `all` and `split` numbers are never our result.** They measure the leakage.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import assert_cache_matches, load_subject  # noqa: E402
from data.config import all_subjects, load_config  # noqa: E402
from data.splits import count_shared_trials, stratified_trial_kfold  # noqa: E402
from evaluation.experiment_log import (  # noqa: E402
    completed_keys, describe_resume, log_experiment, set_seed)
from evaluation.pipeline import (  # noqa: E402
    Fold, Trials, prepare_fold, prepare_timepoint_holdout_fold)
from evaluation.trainer import TrainConfig, Trainer  # noqa: E402
from models.gcns_net import GCNsNet, GCNsNetConfig  # noqa: E402


@dataclass(frozen=True)
class Leaks:
    """Which of the three leakage paths are switched on."""

    split: bool = False  # timepoint-level split instead of trial-level
    bn: bool = False     # BatchNorm in training mode at test
    graph: bool = False  # adjacency AND standardiser over train + test

    def paths(self) -> list[str]:
        out = []
        if self.split:
            out.append("timepoint_shuffled_split")
        if self.graph:
            out += ["adjacency_over_all_data", "standardizer_over_all_data"]
        if self.bn:
            out.append("batchnorm_train_at_test")
        return out


ARMS: dict[str, Leaks] = {
    "none": Leaks(),
    "split": Leaks(split=True),
    "bn": Leaks(bn=True),
    "graph": Leaks(graph=True),
    "all": Leaks(split=True, bn=True, graph=True),
}


@dataclass
class BuiltFold:
    fold: Fold
    fold_id: int | None
    n_train_trials: int
    n_test_trials: int


def seed_fold(seed: int, subject: str, fold_id: int | None) -> None:
    """Seed torch from the fold's own identity (seed, subject, fold).

    Model initialisation draws from the global torch generator. Seeding once per
    process would make a ``--resume``d run (the job queue retries after host-memory
    failures) initialise every later fold differently from an uninterrupted run;
    seeding per fold makes each fold a pure function of its configuration, as
    ``seed_stage`` does in ``run_pretrain_finetune.py``.
    """
    key = f"{seed}:{subject}:{'holdout' if fold_id is None else fold_id}"
    h = hashlib.sha256(key.encode()).digest()
    torch.manual_seed(int.from_bytes(h[:8], "big") % (2 ** 63))


def fold_ids(leaks: Leaks, n_folds: int) -> list[int | None]:
    """Trial-level arms run every fold; timepoint-holdout arms run once (``None``)."""
    return [None] if leaks.split else list(range(n_folds))


def model_config(leaks: Leaks, n_classes: int) -> GCNsNetConfig:
    """The reference GCNs-Net; only BatchNorm's test-time mode depends on the arm."""
    return GCNsNetConfig(n_classes=n_classes, always_train_bn=leaks.bn)


def split_off_validation(y: np.ndarray, train_idx: np.ndarray, frac: float, seed: int
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Same carve-out as ``run_subject_dependent.split_off_validation`` (trial level)."""
    if frac <= 0:
        return train_idx, np.array([], dtype=np.int64)
    n_folds = max(2, int(round(1.0 / frac)))
    inner = stratified_trial_kfold(y[train_idx], n_splits=n_folds, seed=seed)[0]
    return train_idx[inner.train], train_idx[inner.test]


def build_fold(X: np.ndarray, y: np.ndarray, leaks: Leaks, fold_id: int | None, *,
               n_folds: int, val_frac: float, seed: int, lmax: float | None,
               test_frac: float, sfreq: float) -> BuiltFold:
    """Assemble one fold for the arm. The only place the toggles act on the data.

    Trial-level arms go through ``prepare_fold`` exactly as the correct protocol does;
    ``graph`` flips its ``leaky`` flag (graph and standardiser over train+val+test) and
    changes nothing else. Holdout arms go through
    ``prepare_timepoint_holdout_fold``; ``graph`` there selects whether the graph and
    standardiser see the held-out timepoints.
    """
    if leaks.split:
        if fold_id is not None:
            raise ValueError("timepoint-holdout arms have a single fold (fold_id None)")
        fold = prepare_timepoint_holdout_fold(Trials(X=X, y=y), test_frac,
                                              leak_graph=leaks.graph, lmax=lmax, seed=seed)
        return BuiltFold(fold, None, int(len(X)), int(len(X)))

    split = stratified_trial_kfold(y, n_splits=n_folds, seed=seed)[fold_id]
    tr_idx, val_idx = split_off_validation(y, split.train, val_frac, seed)
    train = Trials(X=X[tr_idx], y=y[tr_idx])
    val = Trials(X=X[val_idx], y=y[val_idx]) if len(val_idx) else None
    test = Trials(X=X[split.test], y=y[split.test])
    fold = prepare_fold(train, test, val, lmax=lmax, seed=seed, sfreq=sfreq,
                        window=1, stride=None, features="raw",
                        leaky=leaks.graph, build_graph=True)
    return BuiltFold(fold, fold_id, len(train), len(test))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leaks", choices=list(ARMS), required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="validation carve-out of the trial-level arms' training trials "
                         "(the holdout arms have none, as in the reference)")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--lmax", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42,
                    help="pre-registered as 42 for every arm")
    ap.add_argument("--log", default="runs/experiments.jsonl")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="skip (subject, fold) keys already logged under this tag")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    leaks = ARMS[args.leaks]
    cfg = load_config(args.config)
    seed_info = set_seed(args.seed)
    assert_cache_matches(cfg, cfg.dataset.cache_dir)

    subjects = args.subjects or all_subjects(cfg)
    lmax = None if args.lmax == 0 else args.lmax
    n_classes = len(cfg.class_names)
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                            learning_rate=args.lr, verbose=False)
    model_cfg = model_config(leaks, n_classes)
    ids = fold_ids(leaks, args.folds)

    print(f"A5 leak ablation: arm {args.leaks}  paths {leaks.paths() or 'none'}")
    print(f"subjects {len(subjects)}  folds/subject {len(ids)}  epochs {args.epochs}  "
          f"lr {args.lr}  seed {args.seed}  device {train_cfg.device}\n")

    done = completed_keys(args.log, args.tag) if args.resume else set()
    if args.resume:
        print(describe_resume(done, len(subjects) * len(ids), args.tag))

    tp_accs: list[float] = []
    for i, subject in enumerate(subjects, 1):
        todo = [f for f in ids if (subject, f) not in done]
        if not todo:
            continue
        ep = load_subject(subject, cfg.dataset.cache_dir)
        fold_tp: list[float] = []

        for fold_id in todo:
            seed_fold(args.seed, subject, fold_id)
            built = build_fold(ep.X, ep.y, leaks, fold_id, n_folds=args.folds,
                               val_frac=args.val_frac, seed=args.seed, lmax=lmax,
                               test_frac=args.test_frac, sfreq=ep.sfreq)
            fold = built.fold
            model = GCNsNet(fold.bases, model_cfg, adjacencies=fold.adjacencies)
            trainer = Trainer(model, train_cfg, n_classes=n_classes)
            history = trainer.fit(fold)
            result = trainer.evaluate_samples(fold.test)
            tp, tr = result["timepoint"], result["trial"]
            fold_tp.append(tp.accuracy)

            shared = (int(count_shared_trials(np.unique(fold.train.trial_id),
                                              np.unique(fold.test.trial_id), 1))
                      if leaks.split else 0)
            log_experiment({
                "experiment": "leak_ablation",
                "tag": args.tag,
                "leaks": args.leaks,
                "leakage_paths": leaks.paths(),
                "subject": subject,
                "fold": built.fold_id,
                "n_folds": args.folds if built.fold_id is not None else None,
                "protocol": ("leaky_reference" if args.leaks == "all"
                             else "trial_level" if args.leaks == "none" else "ablation"),
                "split_type": ("leaky_timepoint_holdout" if leaks.split
                               else "stratified_trial_kfold"),
                "trials_in_both_splits": shared,
                "n_trials": int(len(ep.X)),
                **seed_info,
                "config_source": str(cfg.source_path),
                "preprocess": {"bandpass": cfg.preprocess.bandpass,
                               "rereference": cfg.preprocess.rereference},
                "lmax": lmax,
                "n_classes": n_classes,
                "window": 1, "stride": 1,
                "model": "gcn",
                "train_config": train_cfg,
                "model_config": model_cfg,
                "n_parameters": model.n_parameters(),
                "n_train_trials": built.n_train_trials,
                "n_test_trials": built.n_test_trials,
                "timepoint_accuracy": tp.accuracy,
                "timepoint_macro_f1": tp.macro_f1,
                "timepoint_kappa": tp.kappa,
                "timepoint_per_class": tp.per_class_accuracy,
                "timepoint_confusion": tp.confusion,
                "trial_aggregation": "mean_logit",
                "trial_accuracy": tr.accuracy,
                "trial_macro_f1": tr.macro_f1,
                "trial_kappa": tr.kappa,
                "trial_per_class": tr.per_class_accuracy,
                "trial_confusion": tr.confusion,
                "final_train_acc": history.epochs[-1].train_acc,
                "selection": history.selection,
            }, path=args.log)

            print(f"[{i:3d}/{len(subjects)}] {subject} fold {built.fold_id}: "
                  f"timepoint {tp.accuracy:.4f}  trial {tr.accuracy:.4f}  "
                  f"(train {history.epochs[-1].train_acc:.4f}, {history.selection})",
                  flush=True)

            # Per-fold state is a few hundred MB of samples plus the model; the host
            # has crashed on accumulated state before.
            del built, fold, model, trainer, history, result
            gc.collect()

        tp_accs.append(float(np.mean(fold_tp)))
        del ep
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    print("\n" + "=" * 64)
    if not tp_accs:
        print("nothing to do: every (subject, fold) for this tag was already logged")
        return 0
    print(f"arm {args.leaks}: mean timepoint accuracy {np.mean(tp_accs):.4f} "
          f"+/- {np.std(tp_accs):.4f} over {len(tp_accs)} subjects")
    print("Leak-ablation arms measure leakage; none of them is a reported model result.")
    print(f"log -> {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
