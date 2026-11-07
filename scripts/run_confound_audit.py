"""Run-family confound audit (Phase A, A3 and A4 of 9.26).

    # A3: can run family be read off REST epochs, which carry no class label?
    python scripts/run_confound_audit.py --experiment rest-family --model tgsp \
        --split run --config configs/rest_bp830.yaml --tag PA-A3-rest-tgsp-run-bp830

    # A4: the cross-family task MOABB benchmarks, and its within-family controls
    python scripts/run_confound_audit.py --experiment cross-family --task rh_vs_feet \
        --model shallow --split trial --config configs/bandpass_8_30.yaml \
        --tag PA-A4-rh_vs_feet-shallow-trial

**Why.** Under the 8-30 Hz pipeline run family (fist runs 4/8/12 versus feet runs
6/10/14) is identifiable well above chance (9.8, 0.6168). A cross-family task such
as right fist versus both feet therefore mixes a motor contrast with a run-family
contrast, and a trial-level k-fold, which lets train and test share runs, cannot
tell them apart. Two measurements separate them:

* **rest-family (A3):** classify run family from the rest (T0) epochs alone, so no
  trial label exists to carry a motor signal.
* **cross-family (A4):** score the same task under trial-disjoint folds and under
  run-disjoint folds. The drop from the first to the second, compared with the same
  drop on the within-family controls (``lr``, ``bf``), estimates the inflation.

**Splits.** ``trial``: 5-fold stratified, runs mixed. ``run``: whole runs held out
(``data.splits.run_disjoint_folds``); fold membership is decided by run numbers only.

**Models.** ``tgsp``: the in-house tangent-space classifier of
``scripts/run_tangent_space.py`` (shrunk covariances, Riemannian mean of the TRAINING
covariances as reference, RobustScaler, ``LogisticRegression(C=0.01)``).
``shallow``: ShallowConvNet with the A2 settings (window 64, stride 16, 50 epochs,
lr 1e-3, batch 256, seed 42, stratified 20 % trial-level validation carve-out for
best-by-validation checkpointing, standardiser fitted on the training trials),
trial predictions by mean logit over windows.

**Output.** One row per (subject, fold) with trial-level ``accuracy`` and ``roc_auc``
(sklearn, positive-class score: the logistic decision value for ``tgsp``, the
mean-over-windows logit difference for ``shallow``) and the trial confusion matrix.
Rows are appended only after a fold is finished, so ``--resume`` is exact.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import assert_cache_matches, load_subject, subject_path  # noqa: E402
from data.config import all_subjects, load_config  # noqa: E402
from data.edf_loader import rest_family  # noqa: E402
from data.riemann import covariances, riemannian_mean, tangent_space  # noqa: E402
from data.splits import (RUN_PAIR_FOLDS, TrialSplit, run_disjoint_folds,  # noqa: E402
                         stratified_trial_kfold)
from evaluation.experiment_log import (completed_keys, describe_resume,  # noqa: E402
                                       log_experiment, set_seed)
from evaluation.metrics import aggregate_to_trials, evaluate  # noqa: E402

# Cross-family tasks: (class kept as 0, class kept as 1). Class ids are those of the
# four-class trial cache: 0 left fist, 1 right fist, 2 both fists, 3 both feet.
TASK_CLASSES: dict[str, tuple[int, int]] = {
    "rh_vs_feet": (1, 3),   # right fist (runs 4/8/12) vs both feet (runs 6/10/14)
    "lr": (0, 1),           # within the fist family
    "bf": (2, 3),           # within the fists-vs-feet family
}
# Run-disjoint folds per task. Within one family a fold holds out a single run.
TASK_RUN_FOLDS: dict[str, list[tuple[int, ...]]] = {
    "rh_vs_feet": RUN_PAIR_FOLDS,
    "lr": [(4,), (8,), (12,)],
    "bf": [(6,), (10,), (14,)],
}
REST_TASK = "rest_family"
TRIAL_FOLDS = 5

# A2 settings (Global Constraints), copied from the logged A2-shallow-binary-* arm.
WINDOW, STRIDE, LR, BATCH, VAL_FRAC = 64, 16, 1e-3, 256, 0.2
DEFAULT_EPOCHS = 50
SFREQ_FALLBACK = 160.0


# ---------------------------------------------------------------- data selection
def select_task(y: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray]:
    """Trial indices of a cross-family task and their labels relabelled to 0/1."""
    neg, pos = TASK_CLASSES[task]
    idx = np.flatnonzero(np.isin(y, (neg, pos)))
    return idx, (y[idx] == pos).astype(np.int64)


def load_epochs(experiment: str, task: str | None, subject: str, cache_dir: Path
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(X, y, runs, sfreq)`` for one subject, labels binary 0/1."""
    if experiment == "rest-family":
        # The rest cache has no sfreq key (build_rest_cache.py); every epoch is 4.0 s
        # at the validated 160 Hz.
        path = subject_path(Path(cache_dir), subject)
        if not path.exists():
            raise FileNotFoundError(f"{path} - build the rest cache first")
        with np.load(path, allow_pickle=True) as z:
            X, runs = z["X"], np.asarray(z["run"])
        y = np.array([rest_family(int(r)) for r in runs], dtype=np.int64)
        return X, y, runs, SFREQ_FALLBACK
    assert task is not None
    ep = load_subject(subject, cache_dir)
    runs = np.asarray(ep.run)
    idx, y = select_task(np.asarray(ep.y), task)
    return ep.X[idx], y, runs[idx], ep.sfreq


def build_folds(runs: np.ndarray, y: np.ndarray, split: str,
                fold_runs: list[tuple[int, ...]], seed: int) -> list[TrialSplit]:
    """Folds for one subject. Run folds come from run numbers only."""
    if split == "trial":
        return stratified_trial_kfold(y, n_splits=TRIAL_FOLDS, seed=seed)
    folds = run_disjoint_folds(runs, y, fold_runs)
    for f in folds:
        shared = np.intersect1d(runs[f.train], runs[f.test])
        if shared.size:   # unreachable by construction; a leak would be silent
            raise RuntimeError(f"fold {f.fold}: runs {shared.tolist()} in train and test")
    return folds


# ------------------------------------------------------------------------ models
def tgsp_fold(covs: np.ndarray, y: np.ndarray, split: TrialSplit
              ) -> tuple[np.ndarray, np.ndarray]:
    """Tangent-space classifier fitted on the training trials only.

    Settings are those of ``run_tangent_space.tangent_accuracy``. Returns
    ``(predicted class, positive-class score)`` for the test trials.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler

    ctr, cte = covs[split.train], covs[split.test]
    ref = riemannian_mean(ctr)                 # reference from TRAINING covariances
    ztr, zte = tangent_space(ctr, ref), tangent_space(cte, ref)
    scaler = RobustScaler().fit(ztr)
    clf = LogisticRegression(C=0.01, max_iter=2000)
    clf.fit(scaler.transform(ztr), y[split.train])
    zte = scaler.transform(zte)
    return clf.predict(zte), clf.decision_function(zte)


def shallow_fold(X: np.ndarray, y: np.ndarray, split: TrialSplit, *, sfreq: float,
                 seed: int, epochs: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """ShallowConvNet, A2 settings. Returns ``(pred, score, info)`` per test trial."""
    import torch
    from evaluation.pipeline import Trials, prepare_fold
    from evaluation.trainer import TrainConfig, Trainer
    from models.eegnet import build_baseline

    set_seed(seed)   # per fold, so a resumed run reproduces an uninterrupted one
    n_folds = max(2, int(round(1.0 / VAL_FRAC)))
    inner = stratified_trial_kfold(y[split.train], n_splits=n_folds, seed=seed)[0]
    tr_idx, val_idx = split.train[inner.train], split.train[inner.test]
    fold = prepare_fold(Trials(X=X[tr_idx], y=y[tr_idx]),
                        Trials(X=X[split.test], y=y[split.test]),
                        Trials(X=X[val_idx], y=y[val_idx]),
                        seed=seed, sfreq=sfreq, window=WINDOW, stride=STRIDE,
                        build_graph=False)
    cfg = TrainConfig(epochs=epochs, batch_size=BATCH, learning_rate=LR, verbose=False)
    _, n_ch, n_tp = fold.train.X.shape
    model = build_baseline("shallow", n_ch, n_tp, 2, sfreq=sfreq)
    trainer = Trainer(model, cfg, n_classes=2)
    history = trainer.fit(fold)
    logits = trainer.predict_logits(fold.test)

    trial_ids, pred = aggregate_to_trials(logits, fold.test.trial_id)
    _, inverse = np.unique(fold.test.trial_id, return_inverse=True)
    mean_logit = np.zeros((len(trial_ids), 2))
    np.add.at(mean_logit, inverse, logits)
    mean_logit /= np.bincount(inverse)[:, None]
    # Windows are laid out trial by trial, so sorted trial ids are the test order.
    assert np.array_equal(trial_ids - trial_ids[0], np.arange(len(split.test)))
    info = {"train_config": cfg, "n_parameters": model.n_parameters(),
            "n_val_trials": int(len(val_idx)), "selection": history.selection,
            "best_epoch": history.best_epoch}
    del trainer, model, fold
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred, mean_logit[:, 1] - mean_logit[:, 0], info


# -------------------------------------------------------------------------- main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=["rest-family", "cross-family"], required=True)
    ap.add_argument("--model", choices=["tgsp", "shallow"], required=True)
    ap.add_argument("--split", choices=["trial", "run"], required=True)
    ap.add_argument("--config", required=True,
                    help="rest-family: configs/rest_bp830.yaml or rest_car.yaml; "
                         "cross-family: configs/bandpass_8_30.yaml")
    ap.add_argument("--task", choices=sorted(TASK_CLASSES), default=None,
                    help="cross-family only (required there, forbidden for rest-family)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--log", default="runs/experiments.jsonl")
    ap.add_argument("--resume", action="store_true",
                    help="skip (subject, fold) pairs already logged under this tag")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                    help="--model shallow only; 50 is the A2 setting, other values "
                         "are for smoke tests and are logged")
    args = ap.parse_args(argv)
    if args.experiment == "cross-family" and args.task is None:
        ap.error("--task is required for --experiment cross-family")
    if args.experiment == "rest-family" and args.task is not None:
        ap.error("--task is not allowed for --experiment rest-family")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    warnings.filterwarnings("ignore")
    cfg = load_config(args.config)
    cache_dir = cfg.dataset.cache_dir
    assert_cache_matches(cfg, cache_dir)
    seed_info = set_seed(args.seed)
    subjects = args.subjects or all_subjects(cfg)
    task_name = REST_TASK if args.experiment == "rest-family" else args.task
    fold_runs = (RUN_PAIR_FOLDS if args.experiment == "rest-family"
                 else TASK_RUN_FOLDS[args.task])
    n_folds = TRIAL_FOLDS if args.split == "trial" else len(fold_runs)

    done = completed_keys(args.log, args.tag) if args.resume else set()
    print(f"{args.experiment} / {task_name} / {args.model} / {args.split}  "
          f"{len(subjects)} subjects x {n_folds} folds  cache {cache_dir}  tag {args.tag}")
    if args.resume:
        print(describe_resume(done, len(subjects) * n_folds, args.tag))

    from sklearn.metrics import roc_auc_score

    accs, aucs = [], []
    for subject in subjects:
        if all((subject, f) in done for f in range(n_folds)):
            continue
        t0 = time.time()
        X, y, runs, sfreq = load_epochs(args.experiment, args.task, subject, cache_dir)
        covs = covariances(X) if args.model == "tgsp" else None
        s_acc, s_auc = [], []
        for split in build_folds(runs, y, args.split, fold_runs, args.seed):
            if (subject, split.fold) in done:
                continue
            info: dict = {}
            if args.model == "tgsp":
                pred, score = tgsp_fold(covs, y, split)
            else:
                pred, score, info = shallow_fold(X, y, split, sfreq=sfreq,
                                                 seed=args.seed, epochs=args.epochs)
            y_te = y[split.test]
            res = evaluate(y_te, pred, 2)
            auc = float(roc_auc_score(y_te, score))
            s_acc.append(res.accuracy)
            s_auc.append(auc)
            log_experiment({
                "experiment": "confound_audit",
                "audit": args.experiment,
                "tag": args.tag,
                "task": task_name,
                "model": args.model,
                "split": args.split,
                "split_type": ("stratified_trial_kfold" if args.split == "trial"
                               else "run_disjoint"),
                "subject": subject,
                "fold": split.fold,
                "n_folds": n_folds,
                "fold_test_runs": sorted(int(r) for r in np.unique(runs[split.test])),
                "n_train_trials": int(len(split.train)),
                "n_test_trials": int(len(split.test)),
                "n_classes": 2,
                "accuracy": res.accuracy,
                "roc_auc": auc,
                "trial_confusion": res.confusion,
                "config_source": str(cfg.source_path),
                "cache_dir": str(cache_dir),
                "preprocess": {"bandpass": cfg.preprocess.bandpass,
                               "rereference": cfg.preprocess.rereference},
                **({"window": WINDOW, "stride": STRIDE, "epochs": args.epochs}
                   if args.model == "shallow" else {}),
                **info,
                **seed_info,
            }, path=args.log)
        if s_acc:
            accs.append(float(np.mean(s_acc)))
            aucs.append(float(np.mean(s_auc)))
            print(f"{subject}: acc {accs[-1]:.4f}  auc {aucs[-1]:.4f}  "
                  f"({time.time() - t0:.1f}s)", flush=True)
        del X, covs
        gc.collect()
        if args.model == "shallow":
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("=" * 64)
    if not accs:
        print("nothing to do: every fold for this tag was already logged")
        return 0
    print(f"subjects: {len(accs)}  mean accuracy {np.mean(accs):.4f}  "
          f"mean roc_auc {np.mean(aucs):.4f}\nlog -> {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
