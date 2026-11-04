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
The target subject — and, under `--groups`, that target's whole group — is excluded from
pre-training **entirely**: not their trials, not the graph, not the normalisation
statistics. Fine-tuning sees only the target's
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

Cost, and the correction that produced `--groups`
-------------------------------------------------
One pre-training per *target* means 206 of them, measured at **14.9 min each** — about
51 h, not the 6-8 h originally projected. That projection took its per-fold cost from
the LOSO runs, which use `GCNsNet`: attention pools the time axis away before the
convolutions, whereas `graphbp` deliberately carries it *through* the spatial stage
(9.11.2), so the spatial op runs on `N x T'` rows. On ~36 training trials that is
invisible; on ~100 pooled subjects it dominates. Error 24 in 14.

`--groups 10` pre-trains once per **group** of targets instead: 20 pre-trainings, on the
order of 4-5 h for both problems. Each group's model excludes that entire group, so the
leakage guarantee is unchanged and a target still draws on ~92 source subjects.

The run prints each group's elapsed time. **Check the first one against the estimate
before walking away** — that is the check error 24 skipped. `--resume` is keyed on
(subject, fold) as everywhere else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import load_subject  # noqa: E402
from data.config import all_subjects, load_config  # noqa: E402
from data.montage import CONDITIONS, region_vector  # noqa: E402
from data.splits import stratified_trial_kfold  # noqa: E402
from evaluation.experiment_log import (  # noqa: E402
    completed_keys, describe_resume, log_experiment, set_seed)
from evaluation.pipeline import Samples, Trials, concat_trials, prepare_fold  # noqa: E402
from evaluation.robustness import (  # noqa: E402
    completed_with_conditions, condition_row, condition_tag, evaluate_conditions)
from evaluation.trainer import TrainConfig, Trainer  # noqa: E402
from graphs.subset import Moments, moments, subset_representation  # noqa: E402


def build_model(name: str, prepared, n_samples: int, n_classes: int, sfreq: float,
                n_domains: int = 0,
                spatial: str = "graph", graph_smoothness: float = 0.0,
                readout: str = "flat", region_graph=None):
    """The same architectures the other runners expose, on a prepared fold.

    ``readout="region"`` (``graphbp`` only) has no coarsened graph to read from the
    prepared fold, because the fold is built with ``build_graph=False`` so the data
    stay in montage order. ``region_graph`` is then the ``(basis, node_regions)``
    pair from :func:`graphs.subset.subset_representation`, in that same order.
    """
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
                                   graph_smoothness=graph_smoothness,
                                   n_domains=n_domains, readout=readout)
        lap = (laplacian(prepared.adjacencies[0], normalized=False)
               if graph_smoothness else None)
        if readout == "region":
            if region_graph is None:
                raise ValueError("readout='region' needs region_graph=(basis, regions)")
            basis, regions = region_graph
            return GraphBandPowerNet([basis], n_samples, cfg, laplacian=lap,
                                     node_regions=regions)
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


def plan_groups(subjects: list[str], n_groups: int,
                targets: list[str] | None) -> list[tuple[int, list[str], list[str]]]:
    """``(group index, group, source subjects)`` for every group to be run.

    Groups are always assigned over **all** ``subjects``, never over the targets
    alone: a group's source set and its index (which seeds the validation pick and
    every fine-tune stage) must not depend on which subjects were asked for, or a
    ``--targets`` run would not reproduce the full run's numbers. Groups holding no
    target are dropped entirely, pre-training included.
    """
    if targets is not None:
        unknown = sorted(set(targets) - set(subjects))
        if unknown:
            raise ValueError(f"--targets {unknown} are not among the {len(subjects)} "
                             f"subjects being run")
    plan = []
    for gi, group in enumerate(assign_groups(subjects, n_groups), 1):
        if targets is not None and not set(group) & set(targets):
            continue
        plan.append((gi, group, [s for s in subjects if s not in group]))
    return plan


def provenance_fields(args) -> dict:
    """Log-row fields for the options this run changed from their defaults.

    Added only when non-default so that default-path rows keep exactly their old keys.
    """
    extra = {}
    if args.readout != "flat":
        extra["readout"] = args.readout
        extra["noref_config"] = str(args.noref_config)
    if args.source_tag and args.source_tag != args.tag:
        extra["source_tag"] = args.source_tag
    return extra


def sum_moments(parts: list[Moments]) -> Moments:
    """Pool per-subject moments; the sum is the moments of the pooled data."""
    total = parts[0]
    for part in parts[1:]:
        total = total + part
    return total


#: Conditions scored by ``--robustness``. "full" is not among them: the clean row IS
#: the full-montage score, from the referenced cache exactly as in PTG.
ROBUSTNESS_CONDITIONS = tuple(c for c in CONDITIONS if c.name != "full")


def load_noref(subject: str, cache_dir, classes, ref: Trials,
               ch_names: list[str]) -> Trials:
    """One subject's no-reference trials, filtered and labelled like ``ref``.

    Review Focus 5: the two caches must hold the same trials in the same order with
    the same channels, or the wrong trials are scored. Channel order, trial count and
    every label are checked against the referenced trials before anything is used.
    """
    nr = load_subject(subject, Path(cache_dir))
    if list(nr.ch_names) != list(ch_names):
        raise RuntimeError(f"{subject}: no-reference cache channel order differs "
                           f"from the referenced cache")
    X, y = nr.X, nr.y
    if classes:
        mask = np.isin(y, classes)
        X, y = X[mask], (y[mask] == classes[1]).astype(np.int64)
    if len(X) != len(ref.y):
        raise RuntimeError(f"{subject}: no-reference cache holds {len(X)} trials, "
                           f"referenced cache {len(ref.y)}")
    if not np.array_equal(y, ref.y):
        raise RuntimeError(f"{subject}: no-reference labels differ from the referenced "
                           f"cache's; the trials are not the same or not in order")
    return Trials(X=X, y=y)


def resume_keys(log, tag: str, robustness: bool) -> set:
    """``(subject, fold)`` pairs a resumed run may skip.

    With ``--robustness`` a fold counts as done only when its clean row AND every
    condition row are logged (Review Focus 4), so a fold cut off mid-evaluation is
    re-run.
    """
    if robustness:
        return completed_with_conditions(log, tag,
                                         [c.name for c in ROBUSTNESS_CONDITIONS])
    return completed_keys(log, tag)


def pick_validation(source: list[str], seed: int, gi: int, n: int) -> list[str]:
    """Source subjects held out to select the pre-training checkpoint.

    Drawn from a generator seeded by this group alone: on a resumed run the
    skipped groups must not shift which subjects validate this one.
    """
    val_rng = np.random.default_rng([seed, gi])
    return list(val_rng.choice(source, size=min(n, len(source) - 1), replace=False))


def seed_stage(seed: int, *parts: int) -> None:
    """Seed every RNG a training stage touches, from the stage's own identity.

    Without this, a stage's result depends on how much work the *process* did
    before it: dropout draws from the global generator, so skipping a pre-training
    fit (a cache hit, or a resumed run picking up mid-cohort) shifts every
    subsequent fold. That made three power-cut resumes silently non-reproducible
    and made the source-model cache change results. Seeding per stage makes each
    fold a pure function of its own configuration.
    """
    h = hashlib.sha256(":".join(str(p) for p in (seed, *parts)).encode()).digest()
    torch.manual_seed(int.from_bytes(h[:8], "big") % (2 ** 63))


def source_cache_key(args, seed: int, group_idx: int,
                     source: list[str], val_pick: list[str]) -> str:
    """Hash everything the source model's weights depend on.

    Deliberately exhaustive rather than clever: a stale hit here would silently
    pre-train on the wrong subjects, which is the one failure this script exists to
    make impossible. Anything that changes the fitted weights must change the key,
    so the source and validation subject lists are hashed by name.
    """
    fields = {
        "model": args.model, "spatial": args.spatial,
        "graph_smoothness": args.graph_smoothness,
        "classes": args.classes, "window": args.window, "stride": args.stride,
        "lmax": args.lmax, "seed": seed, "n_groups": args.groups,
        "group": group_idx,
        "epochs": args.pretrain_epochs, "lr": args.pretrain_lr,
        "batch_size": args.batch_size, "config": str(args.config),
        "source": sorted(source), "val": sorted(val_pick),
    }
    # "dsbn" and "readout" are added only when set. The ShallowConvNet source models
    # (5 h of pre-training) were keyed before either option existed, and a key that
    # gained a field for the default value would silently miss every one of them
    # (commit b4f75fe did exactly that with `"dsbn": false`, which the pinned test
    # test_shallow_cache_key_unchanged caught). A dsbn run keeps its `"dsbn": true`.
    if args.dsbn:
        fields["dsbn"] = True
    if args.readout != "flat":
        fields["readout"] = args.readout
        # The region basis is built from this cache, so a different or regenerated
        # one is a different source model.
        fields["noref_config"] = str(args.noref_config)
    payload = json.dumps(fields, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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
    ap.add_argument("--targets", nargs="+", default=None, metavar="SUBJECT",
                    help="subjects to fine-tune and evaluate (default: all). Groups are "
                         "still assigned over every subject, so each target keeps the "
                         "same source set, validation pick and seeds as in a full run; "
                         "groups with no target are skipped, pre-training included")
    ap.add_argument("--source-tag", default=None,
                    help="tag used in the source-model cache FILENAME (default: --tag). "
                         "Lets a run with a new --tag reuse another run's pre-trained "
                         "models; the hashed key still guards that they are identical")
    ap.add_argument("--readout", choices=["flat", "region"], default="flat",
                    help="--model graphbp only. 'region' pools to 9 anatomical regions, "
                         "so the model does not depend on the electrode count. The graph "
                         "and regions then come from the no-reference cache "
                         "(--noref-config) over the source TRAINING subjects")
    ap.add_argument("--noref-config", default="configs/bandpass_8_30_noref.yaml",
                    help="config of the no-reference cache; read only with "
                         "--readout region")
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
    ap.add_argument("--dsbn", action="store_true",
                    help="--model graphbp only: give every source subject its own "
                         "batch-norm statistics during pre-training, sharing all other "
                         "parameters (models/dsbn.py). Batches become domain-homogeneous. "
                         "The target gets the reserved last slot, seeded from the source "
                         "mean. TSMNet measures this at 2.4-3.9%% on 158 subjects.")
    ap.add_argument("--source-cache", default="runs/source_models",
                    help="directory of cached source models; '' disables caching. "
                         "A source model is keyed by everything that determines it "
                         "(model, group, source subjects, validation pick, epochs, lr, "
                         "batch, window, stride, seed, config), so a hit is exact. "
                         "Pre-training is ~70%% of this script's cost and is otherwise "
                         "re-paid in full whenever a run is resumed.")
    ap.add_argument("--robustness", action="store_true",
                    help="after each fold's clean evaluation, score the fine-tuned model "
                         "on every montage condition (evaluation.robustness) and log one "
                         "row per condition under '<tag>.<condition>'. --model shallow "
                         "(interpolation repair) or --model graphbp --readout region "
                         "(subset graph). Reads the no-reference cache (--noref-config)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    import datetime as dt
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    seed_info = set_seed(seed)
    subjects = args.subjects or all_subjects(cfg)
    if args.readout == "region" and args.model != "graphbp":
        ap.error("--readout region applies to --model graphbp only")
    if args.readout == "region" and (args.spatial != "graph" or args.graph_smoothness
                                     or args.lmax == 0):
        ap.error("--readout region needs --spatial graph, no --graph-smoothness and "
                 "a fixed --lmax")
    if args.robustness and not (args.model == "shallow"
                                or (args.model == "graphbp" and args.readout == "region")):
        ap.error("--robustness needs --model shallow or --model graphbp --readout region")
    if args.robustness and args.dsbn:
        ap.error("--robustness is not defined for --dsbn")
    plan = plan_groups(subjects, args.groups, args.targets)
    target_set = set(args.targets) if args.targets is not None else set(subjects)
    source_tag = args.source_tag or args.tag
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

    # Region readout: the graph is fit on NO-REFERENCE data so that it can be rebuilt
    # for any electrode subset later (graphs.subset). Per-subject moments are taken
    # once, one subject at a time, and only the moments are kept: the whole
    # no-reference cache never sits in memory. Each group's graph is then the sum over
    # exactly the subjects whose data fit the graph on the existing path
    # (source minus the validation pick).
    subject_moments: dict[str, Moments] = {}
    ch_names = list(ep.ch_names)
    # --robustness: the no-reference trials of the TARGETS only, for the test-time
    # conditions. Source subjects enter only through their moments.
    noref_targets: dict[str, Trials] = {}
    if args.readout == "region" or args.robustness:
        noref_dir = load_config(args.noref_config).dataset.cache_dir
    if args.readout == "region":
        print(f"moments of {len(subjects)} subjects from {noref_dir}", flush=True)
        for s in subjects:
            nr_trials = load_noref(s, noref_dir, args.classes, trials[s], ch_names)
            subject_moments[s] = moments(nr_trials.X)
            if args.robustness and s in target_set:
                noref_targets[s] = nr_trials
    elif args.robustness:
        for s in sorted(target_set):
            noref_targets[s] = load_noref(s, noref_dir, args.classes, trials[s], ch_names)

    pre_cfg = TrainConfig(epochs=args.pretrain_epochs, batch_size=args.batch_size,
                          learning_rate=args.pretrain_lr, verbose=False)
    ft_cfg = TrainConfig(epochs=args.finetune_epochs, batch_size=args.batch_size,
                         learning_rate=args.finetune_lr, verbose=False)

    print(f"model {args.model}  classes {args.classes}  "
          f"window {args.window} stride {args.stride}")
    # The FULL assignment: the count that is logged and shown in progress lines is the
    # grouping used, not the number of groups a --targets run happens to visit.
    groups = assign_groups(subjects, args.groups)
    n_src = len(subjects) - max(len(g) for g in groups)
    print("PRE-TRAIN " + ("SKIPPED (control arm)" if args.no_pretrain else
                          f"{args.pretrain_epochs} epochs @ {args.pretrain_lr}, "
                          f"{len(groups)} group(s) of <= {max(len(g) for g in groups)} "
                          f"targets, ~{n_src} source subjects each"))
    print(f"FINE-TUNE {args.finetune_epochs} epochs @ {args.finetune_lr} "
          f"on the target's training folds\n")

    done = resume_keys(args.log, args.tag, args.robustness) if args.resume else set()
    # Per tag as well, so a fold re-run after a partial write logs only the rows that
    # are missing and never duplicates one (find_duplicates would flag it).
    clean_done = completed_keys(args.log, args.tag) if args.resume else set()
    cond_done = {c.name: (completed_keys(args.log, condition_tag(args.tag, c.name))
                          if args.resume else set())
                 for c in ROBUSTNESS_CONDITIONS} if args.robustness else {}
    if args.resume:
        print(describe_resume(done, len(target_set) * args.folds, args.tag))

    per_subject: list[float] = []
    t0 = time.time()

    for gi, group, source in plan:
        if all((t, f) in done for t in group if t in target_set
               for f in range(args.folds)):
            continue
        if deadline is not None and dt.datetime.now() >= deadline:
            print(f"\nreached --stop-at {args.stop_at}; stopping cleanly before "
                  f"group {gi}. Re-run with --resume to continue.")
            break

        # ---------------------------------------------------------- pre-train
        # The whole GROUP is excluded, so every target in it is absent from its own
        # source set: not their trials, not the graph, not the standardiser.
        # (`source` comes from plan_groups: every subject outside the group.)
        val_pick = pick_validation(source, seed, gi, args.val_subjects)
        src_train = concat_trials([trials[s] for s in source if s not in val_pick])
        src_val = concat_trials([trials[s] for s in val_pick])

        # One representation for both phases: graph and standardiser fit on SOURCE.
        # The group's trials are the "test" argument purely so they are transformed by
        # the same statistics; nothing about them is fit. concat_trials preserves order,
        # so each target occupies a known contiguous span of trial ids.
        group_trials = [trials[t] for t in group]
        offsets = np.cumsum([0] + [len(t) for t in group_trials])

        cache_path = None
        if not args.no_pretrain and args.source_cache:
            cache_dir = Path(args.source_cache)
            cache_dir.mkdir(parents=True, exist_ok=True)
            key = source_cache_key(args, seed, gi, source, list(val_pick))
            cache_path = cache_dir / f"src-{source_tag}-g{gi}-{key}.pt"
        # The source windows (~2.2 GiB for ~87 subjects) feed only pre-training. On a
        # cache hit nothing reads them, so they are not built: on a machine short of
        # host memory, building them was enough to make CUDA initialisation fail. The
        # graph and standardiser are still fit on the full source trials, so the target
        # windows are bit-identical. DSBN labels the source windows, so it keeps them.
        source_windows = not (cache_path is not None and cache_path.exists()
                              and not args.dsbn)
        prepared = prepare_fold(
            src_train, concat_trials(group_trials), src_val if source_windows else None,
            lmax=lmax, seed=seed, sfreq=sfreq, window=args.window, stride=args.stride,
            build_graph=(args.model == "gcn"
                         or (args.model == "graphbp" and args.readout == "flat")),
            train_samples=source_windows)
        del src_train, src_val

        # Region readout: basis and regions from the source TRAINING subjects' moments
        # (source minus the validation pick), in montage order -- the order the data
        # are in because build_graph=False left them unpermuted.
        region_graph = None
        m_src = rep_all = None
        if args.readout == "region":
            m_src = sum_moments([subject_moments[x] for x in source if x not in val_pick])
            rep_all = subset_representation(
                m_src, np.ones(len(ch_names), dtype=bool), ch_names, lmax=args.lmax)
            assert np.array_equal(rep_all.regions, region_vector(ch_names))
            region_graph = (rep_all.basis, rep_all.regions)

        # Domain labels for the source split. concat_trials preserves subject order and
        # trial ids run 0..N-1 over it, so a per-trial domain array indexes straight
        # through `trial_id`. The last slot (n_domains - 1) is reserved for the target.
        n_domains = 0
        if args.dsbn:
            src_subjects = [x for x in source if x not in val_pick]
            per_trial = np.repeat(np.arange(len(src_subjects)),
                                  [len(trials[x].y) for x in src_subjects])
            prepared = replace(prepared, train=replace(
                prepared.train, domains=per_trial[prepared.train.trial_id]))
            n_domains = len(src_subjects) + 1

        pretrained_state = None
        if not args.no_pretrain:
            if cache_path is not None and cache_path.exists():
                pretrained_state = torch.load(cache_path, map_location="cpu",
                                              weights_only=True)
                if not args.quiet:
                    print(f"group {gi}/{len(groups)}: reused cached source model "
                          f"({cache_path.name}) [{time.time() - t0:.0f}s elapsed]",
                          flush=True)
            else:
                seed_stage(seed, gi, 0, 0)
                model = build_model(args.model, prepared, args.window, n_classes, sfreq,
                                    n_domains=n_domains, spatial=args.spatial,
                                    graph_smoothness=args.graph_smoothness,
                                    readout=args.readout, region_graph=region_graph)
                trainer = Trainer(model, pre_cfg, n_classes=n_classes)
                trainer.fit(prepared)
                pretrained_state = {k: v.detach().clone()
                                    for k, v in trainer.model.state_dict().items()}
                if cache_path is not None:
                    # Write via a temporary file: a power cut mid-write would otherwise
                    # leave a truncated checkpoint that a later run would load as valid.
                    tmp = cache_path.with_suffix(".pt.partial")
                    torch.save({k: v.cpu() for k, v in pretrained_state.items()}, tmp)
                    tmp.replace(cache_path)
                if not args.quiet:
                    print(f"group {gi}/{len(groups)}: pre-trained on {len(source)} "
                          f"subjects [{time.time() - t0:.0f}s elapsed]", flush=True)

        for j, target in enumerate(group):
            # j stays the member's index in the FULL group: it seeds the fine-tune
            # stage and locates the target's span in the group's concatenation.
            if target not in target_set:
                continue
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

                seed_stage(seed, gi, j + 1, split.fold)
                model = build_model(args.model, prepared, args.window, n_classes, sfreq,
                                    n_domains=n_domains, spatial=args.spatial,
                                    graph_smoothness=args.graph_smoothness,
                                    readout=args.readout, region_graph=region_graph)
                if pretrained_state is not None:
                    model.load_state_dict(pretrained_state)
                if n_domains:
                    # The target is an unseen domain. Seed its statistics from the
                    # source mean, then label its own windows with that slot so
                    # fine-tuning adapts it rather than any source subject's.
                    from models.dsbn import seed_target_domain
                    seed_target_domain(model, n_domains - 1)
                    ft_train = replace(ft_train, domains=np.full(
                        len(ft_train.y), n_domains - 1, dtype=np.int64))
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

                row = {
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
                    "dsbn": bool(n_domains),
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
                    **provenance_fields(args),
                }

                if args.robustness:
                    # The same test trials, from the no-reference cache. Same order as
                    # the referenced cache (checked per subject at load), and the
                    # labels are checked again on exactly the trials being scored.
                    test_idx = np.asarray(split.test)
                    nr = noref_targets[target]
                    test_noref = Trials(X=nr.X[test_idx], y=nr.y[test_idx])
                    if not np.array_equal(test_noref.y, y_target[test_idx]):
                        raise RuntimeError(f"{target} fold {split.fold}: no-reference "
                                           f"test labels differ from the referenced ones")
                    pending = tuple(c for c in ROBUSTNESS_CONDITIONS
                                    if (target, split.fold) not in cond_done[c.name])
                    scores = evaluate_conditions(
                        trainer, "graph" if args.readout == "region" else "shallow",
                        test_noref, base, ch_names, m_src, prepared.standardizer,
                        rep_all, target, split.fold, pending, args.window, args.stride,
                        lmax=args.lmax)
                    for c in pending:
                        log_experiment(condition_row(row, c.name, scores[c.name]),
                                       path=args.log)
                    if not args.quiet:
                        print(f"    {target} fold {split.fold}: clean "
                              f"{result['trial'].accuracy:.4f}  " + "  ".join(
                                  f"{c.name} {scores[c.name]['accuracy']:.4f}"
                                  for c in pending), flush=True)

                # The clean row goes LAST: once it is in the log, every condition row
                # of this fold is too. It is not written twice when a fold is re-run
                # only because a condition row was missing.
                if (target, split.fold) not in clean_done:
                    log_experiment(row, path=args.log)

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
