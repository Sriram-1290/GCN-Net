"""Cross-subject pre-training, then per-subject fine-tuning (17.11.2).

    python scripts/run_pretrain_finetune.py --config configs/bandpass_8_30.yaml \
        --classes 0 1 --model shallow --tag P1-ptft-shallow-binary-0-1 --resume

**Why this exists.** 9.9 measured two numbers that have never been combined:

* **LOSO 0.6040** — ~100x the training data, no subject-specific calibration;
* **subject-dependent 0.6038** — calibration, and ~36 training trials per fold.

They are statistically the same number reached from opposite directions, and *neither
uses the other's information*. 9.9's own conclusion was that the subject-dependent
protocol is data-starved. Pre-training on the other 102 subjects and then fine-tuning
on the target's own training trials uses both, and it needs no data we do not have.

17.11.1 is why this is the right lever rather than merely another one. Reaching a mean
of 0.700 needs +0.0275, and the deficit sits in the lower tail: 41 of 103 subjects are
already above 0.70, the median is 0.644, and seven are below chance. [27] reports
pre-training giving up to +7.76 % in general, +27.34 % where training data is limited,
and — the figure that matters here — a **75 % reduction in the variance of accuracy
across subjects**. Variance reduction *is* lifting the weak tail.

Leakage
-------
The target subject is excluded from pre-training **entirely** — not their trials, not
the graph, not the normalisation statistics. Fine-tuning sees only the target's
training folds, which are the same folds `run_subject_dependent.py` uses at the same
seed, so the comparison is paired subject by subject and fold by fold.

The target's *training* trials could in principle have joined pre-training too, but the
split differs per fold, so that would need a separate pre-training run per fold —
1,030 instead of 206. Excluding the subject outright is both cheaper and the stricter
choice.

One representation throughout
-----------------------------
The graph and the standardiser are fit on the **source** subjects and used for both
phases. A pre-trained model cannot have its input rescaled underneath it between
phases, and re-fitting on the target would do exactly that. This does mean the
normalisation differs from `run_subject_dependent.py`, which fits on the target's own
training trials — so **`--no-pretrain` runs the identical pipeline with the
pre-training skipped**, isolating the effect of pre-training from the effect of the
representation. Run both or neither; a comparison against the subject-dependent arm
alone would confound the two.

Cost
----
One pre-training per (target, problem), reused across that target's five folds: 206
pre-trainings at roughly the cost of a LOSO fold (~1.6 min measured), plus cheap
fine-tunes. Budget around 6-8 hours for both binary problems. `--resume` is keyed on
(subject, fold) as everywhere else.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import load_subject  # noqa: E402
from data.config import all_subjects, load_config  # noqa: E402
from data.splits import stratified_trial_kfold  # noqa: E402
from evaluation.experiment_log import (  # noqa: E402
    completed_keys, describe_resume, log_experiment, set_seed)
from evaluation.pipeline import Samples, Trials, concat_trials, prepare_fold  # noqa: E402
from evaluation.trainer import TrainConfig, Trainer  # noqa: E402


def build_model(name: str, prepared, n_samples: int, n_classes: int, sfreq: float,
                spatial: str = "graph", graph_smoothness: float = 0.0):
    """The same architectures the other runners expose, on a prepared fold."""
    if name == "gcn":
        from models.gcns_net import GCNsNet, GCNsNetConfig
        cfg = GCNsNetConfig(n_classes=n_classes, in_features=n_samples,
                            temporal_attention=True)
        return GCNsNet(prepared.bases, cfg, adjacencies=prepared.adjacencies)
    if name in ("eegnet", "shallow"):
        from models.eegnet import build_baseline
        _, n_ch, n_tp = prepared.train.X.shape
        return build_baseline(name, n_ch, n_tp, n_classes, sfreq=sfreq)
    if name == "graphbp":
        from graphs.laplacian import laplacian
        from models.graph_bandpower import GraphBandPowerConfig, GraphBandPowerNet
        cfg = GraphBandPowerConfig(n_classes=n_classes, spatial=spatial,
                                   graph_smoothness=graph_smoothness)
        lap = (laplacian(prepared.adjacencies[0], normalized=False)
               if graph_smoothness else None)
        return GraphBandPowerNet(prepared.bases, n_samples, cfg, laplacian=lap)
    raise ValueError(f"unknown model {name!r}")


def assign_groups(subjects: list[str], n_groups: int) -> list[list[str]]:
    """Partition subjects into groups that share one pre-trained model.

    Assignment is ``index % n_groups``, which is deterministic, needs no seed, and
    interleaves subjects so no group is a contiguous block of the recording order.
    ``n_groups <= 1`` gives one group per subject — strict leave-one-subject-out
    pre-training, the original and most expensive form.
    """
    if n_groups <= 1:
        return [[s] for s in subjects]
    n_groups = min(n_groups, len(subjects))
    groups: list[list[str]] = [[] for _ in range(n_groups)]
    for i, s in enumerate(subjects):
        groups[i % n_groups].append(s)
    return [g for g in groups if g]


def subset_samples(s: Samples, trial_ids: set[int]) -> Samples:
    """Take the windows belonging to a set of trials.

    ``Samples`` carries a ``trial_id`` per window, so a trial-level split stays a
    trial-level split after windowing — no window of a test trial can reach training.
    """
    mask = np.isin(s.trial_id, list(trial_ids))
    return Samples(X=s.X[mask], y=s.y[mask], trial_id=s.trial_id[mask])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/bandpass_8_30.yaml")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--model", choices=["gcn", "eegnet", "shallow", "graphbp"],
                    default="shallow")
    ap.add_argument("--spatial", choices=["graph", "free"], default="graph")
    ap.add_argument("--graph-smoothness", type=float, default=0.0)
    ap.add_argument("--classes", nargs=2, type=int, default=None, metavar=("A", "B"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pretrain-epochs", type=int, default=30)
    ap.add_argument("--finetune-epochs", type=int, default=30)
    ap.add_argument("--pretrain-lr", type=float, default=1e-3)
    ap.add_argument("--finetune-lr", type=float, default=1e-4,
                    help="lower than pre-training: fine-tuning on ~36 trials at the "
                         "pre-training rate destroys the transferred features")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--val-subjects", type=int, default=5,
                    help="source subjects held out for pre-training checkpoint selection")
    ap.add_argument("--groups", type=int, default=10,
                    help="pre-train once per GROUP of targets rather than once per "
                         "target. 103 targets need 103 pre-trainings; 10 groups need 10, "
                         "which is what makes this arm tractable (error 24). Still "
                         "leakage-free: a group's model excludes that whole group, so a "
                         "target is never in its own source set. Pass 0 for strict "
                         "leave-one-subject-out pre-training")
    ap.add_argument("--no-pretrain", action="store_true",
                    help="THE CONTROL. Identical pipeline and representation, "
                         "pre-training skipped. Isolates the effect of pre-training "
                         "from the effect of using source-fit normalisation")
    ap.add_argument("--lmax", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log", default="runs/experiments.jsonl")
    ap.add_argument("--tag", default="ptft")
    ap.add_argument("--stop-at", default=None, metavar="[YYYY-MM-DD ]HH:MM")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    import datetime as dt
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    seed_info = set_seed(seed)
    subjects = args.subjects or all_subjects(cfg)
    lmax = None if args.lmax == 0 else args.lmax
    n_classes = 2 if args.classes else len(cfg.class_names)

    deadline = None
    if args.stop_at:
        now = dt.datetime.now()
        # "YYYY-MM-DD HH:MM" names one instant and is unambiguous overnight; a bare
        # "HH:MM" means today, and a past one stops immediately (error 23).
        if " " in args.stop_at:
            deadline = dt.datetime.strptime(args.stop_at, "%Y-%m-%d %H:%M")
        else:
            hh, mm = (int(v) for v in args.stop_at.split(":"))
            deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if deadline <= now:
            print(f"--stop-at {args.stop_at} is already past; stopping immediately.")
            return 0
        print(f"stopping at {deadline:%Y-%m-%d %H:%M}, at a subject boundary")

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

    pre_cfg = TrainConfig(epochs=args.pretrain_epochs, batch_size=args.batch_size,
                          learning_rate=args.pretrain_lr, verbose=False)
    ft_cfg = TrainConfig(epochs=args.finetune_epochs, batch_size=args.batch_size,
                         learning_rate=args.finetune_lr, verbose=False)

    print(f"model {args.model}  classes {args.classes}  "
          f"window {args.window} stride {args.stride}")
    groups = assign_groups(subjects, args.groups)
    n_src = len(subjects) - max(len(g) for g in groups)
    print("PRE-TRAIN " + ("SKIPPED (control arm)" if args.no_pretrain else
                          f"{args.pretrain_epochs} epochs @ {args.pretrain_lr}, "
                          f"{len(groups)} group(s) of <= {max(len(g) for g in groups)} "
                          f"targets, ~{n_src} source subjects each"))
    print(f"FINE-TUNE {args.finetune_epochs} epochs @ {args.finetune_lr} "
          f"on the target's training folds\n")

    done = completed_keys(args.log, args.tag) if args.resume else set()
    if args.resume:
        print(describe_resume(done, len(subjects) * args.folds, args.tag))

    per_subject: list[float] = []
    rng = np.random.default_rng(seed)
    t0 = time.time()

    for gi, group in enumerate(groups, 1):
        if all((t, f) in done for t in group for f in range(args.folds)):
            continue
        if deadline is not None and dt.datetime.now() >= deadline:
            print(f"\nreached --stop-at {args.stop_at}; stopping cleanly before "
                  f"group {gi}. Re-run with --resume to continue.")
            break

        # ---------------------------------------------------------- pre-train
        # The whole GROUP is excluded, so every target in it is absent from its own
        # source set: not their trials, not the graph, not the standardiser.
        source = [s for s in subjects if s not in group]
        val_pick = list(rng.choice(source, size=min(args.val_subjects, len(source) - 1),
                                   replace=False))
        src_train = concat_trials([trials[s] for s in source if s not in val_pick])
        src_val = concat_trials([trials[s] for s in val_pick])

        # One representation for both phases: graph and standardiser fit on SOURCE.
        # The group's trials are the "test" argument purely so they are transformed by
        # the same statistics; nothing about them is fit. concat_trials preserves order,
        # so each target occupies a known contiguous span of trial ids.
        group_trials = [trials[t] for t in group]
        offsets = np.cumsum([0] + [len(t) for t in group_trials])
        prepared = prepare_fold(
            src_train, concat_trials(group_trials), src_val, lmax=lmax, seed=seed,
            sfreq=sfreq, window=args.window, stride=args.stride,
            build_graph=args.model in ("gcn", "graphbp"))

        pretrained_state = None
        if not args.no_pretrain:
            model = build_model(args.model, prepared, args.window, n_classes, sfreq,
                                spatial=args.spatial,
                                graph_smoothness=args.graph_smoothness)
            trainer = Trainer(model, pre_cfg, n_classes=n_classes)
            trainer.fit(prepared)
            pretrained_state = {k: v.detach().clone()
                                for k, v in trainer.model.state_dict().items()}
            if not args.quiet:
                print(f"group {gi}/{len(groups)}: pre-trained on {len(source)} subjects "
                      f"[{time.time() - t0:.0f}s elapsed]", flush=True)

        for j, target in enumerate(group):
            if all((target, f) in done for f in range(args.folds)):
                continue
            # This target's trials are a contiguous span of the concatenated group.
            base = 200_000 + int(offsets[j])

            # -------------------------------------------------------- fine-tune
            # The same trial-level folds run_subject_dependent.py uses at this
            # seed, so the arms are paired subject by subject and fold by fold.
            y_target = trials[target].y
            fold_accs = []
            for split in stratified_trial_kfold(y_target, n_splits=args.folds, seed=seed):
                if (target, split.fold) in done:
                    continue
                ft_train = subset_samples(prepared.test, set(split.train + base))
                ft_test = subset_samples(prepared.test, set(split.test + base))

                model = build_model(args.model, prepared, args.window, n_classes, sfreq,
                                    spatial=args.spatial,
                                    graph_smoothness=args.graph_smoothness)
                if pretrained_state is not None:
                    model.load_state_dict(pretrained_state)
                trainer = Trainer(model, ft_cfg, n_classes=n_classes)
                # A Fold whose train/test are the target's own split, same representation.
                ft_fold = type(prepared)(
                    train=ft_train, val=None, test=ft_test, bases=prepared.bases,
                    adjacencies=prepared.adjacencies, perm=prepared.perm,
                    adjacency=prepared.adjacency, standardizer=prepared.standardizer,
                    leaky=False)
                history = trainer.fit(ft_fold)
                result = trainer.evaluate_samples(ft_test)
                fold_accs.append(result["trial"].accuracy)

                log_experiment({
                    "experiment": "pretrain_finetune",
                    "tag": args.tag,
                    "subject": target,
                    "fold": split.fold,
                    "n_folds": args.folds,
                    "protocol": "pretrain_finetune" if not args.no_pretrain else "finetune_only",
                    "split_type": "stratified_trial_kfold",
                    **seed_info,
                    "config_source": str(cfg.source_path),
                    "preprocess": {"bandpass": cfg.preprocess.bandpass,
                                   "rereference": cfg.preprocess.rereference},
                    "classes": args.classes,
                    "n_classes": n_classes,
                    "window": args.window,
                    "stride": args.stride,
                    "model": args.model,
                    "pretrained": not args.no_pretrain,
                    "n_source_subjects": len(source),
                    "n_groups": len(groups),
                    "group_id": gi,
                    "group_size": len(group),
                    "pretrain_epochs": 0 if args.no_pretrain else args.pretrain_epochs,
                    "finetune_epochs": args.finetune_epochs,
                    "pretrain_lr": None if args.no_pretrain else args.pretrain_lr,
                    "finetune_lr": args.finetune_lr,
                    "train_config": {"epochs": args.finetune_epochs,
                                     "batch_size": args.batch_size,
                                     "learning_rate": args.finetune_lr,
                                     "device": ft_cfg.device},
                    "n_parameters": sum(p.numel() for p in model.parameters()),
                    "accuracy": result["trial"].accuracy,
                    "trial_accuracy": result["trial"].accuracy,
                    "timepoint_accuracy": result["timepoint"].accuracy,
                    "trial_confusion": result["trial"].confusion.tolist(),
                    "trial_per_class": list(result["trial"].per_class_accuracy),
                    "trial_kappa": result["trial"].kappa,
                    "trial_macro_f1": result["trial"].macro_f1,
                    "history": history.as_dict(),
                }, path=args.log)

            if fold_accs:
                per_subject.append(float(np.mean(fold_accs)))
                if not args.quiet:
                    print(f"  {target}: trial {per_subject[-1]:.4f}", flush=True)

    print("\n" + "=" * 64)
    if not per_subject:
        print("nothing to do; every subject already logged under this tag")
        return 0
    print(f"subjects: {len(per_subject)}")
    print(f"mean trial accuracy  {np.mean(per_subject):.4f} "
          f"+/- {np.std(per_subject):.4f}")
    print(f"log -> {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
