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

Zero-shot scoring and Euclidean alignment (9.26 A1 and A6)
----------------------------------------------------------
``--zero-shot`` scores each target with its group's source model and fine-tunes
nothing: one row per target (``fold`` is ``None``), ``finetune_epochs`` 0. It is the
subject-independent arm of A6 and, with ``--classes`` omitted, A1's 4-class arm, whose
rows also carry ``confusion_excl_calibration`` (the trials outside runs 4 and 6) so A1
can be compared with A6 on the same trials.

``--calibration-run-excluded`` never scores the task's calibration run: run 4 for the
fist family (classes 0/1), run 6 for the fists-vs-feet family (classes 2/3). A6
excludes it in **both** arms, with and without EA, so the arms are paired (Review Focus
2); with ``--ea pseudo-online`` that run is the target's EA reference.

``--ea`` aligns the band-passed trials before windowing (``data/alignment.py``). Each
source subject is aligned to the reference of its own trials. The target's reference:

* ``pseudo-online`` (``--zero-shot``): its calibration run only;
* ``offline`` (``--zero-shot``): all its trials -- TRANSDUCTIVE, logged as such;
* ``train-fold`` (fine-tuning): the fold's training trials only; the fold's test trials
  are aligned with that reference and never contribute to it.

No test trial enters a reference anywhere except the labelled transductive variant.
None of the three options enters the source cache key except ``ea``, which records
how the *sources* were aligned (identical for every EA mode), and only when set: an
unaligned run keeps every existing key, so A6's no-EA arm reuses PTG-shallow's models.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.alignment import EAReference, align, fit_ea  # noqa: E402
from data.cache import load_subject, read_manifest  # noqa: E402
from data.config import all_subjects, load_config  # noqa: E402
from data.montage import CONDITIONS, common_average, region_vector  # noqa: E402
from data.splits import stratified_trial_kfold  # noqa: E402
from evaluation.experiment_log import (  # noqa: E402
    completed_keys, describe_resume, log_experiment, set_seed)
from evaluation.pipeline import (  # noqa: E402
    Samples, Trials, concat_trials, prepare_fold, to_timepoint_samples, to_window_samples)
from evaluation.robustness import (  # noqa: E402
    completed_with_conditions, condition_row, condition_tag, evaluate_conditions)
from evaluation.trainer import TrainConfig, Trainer  # noqa: E402
from graphs.coarsening import perm_data  # noqa: E402
from graphs.subset import Moments, moments, subset_representation  # noqa: E402


def release_host_memory() -> None:
    """Collect garbage and return cached GPU blocks; changes no computed value."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


# ------------------------------------------- zero-shot scoring and EA (9.26 A1, A6)

#: Run family of each class: classes 0/1 (left, right fist) come only from runs
#: 4/8/12, classes 2/3 (both fists, both feet) only from runs 6/10/14.
CLASS_FAMILY = (0, 0, 1, 1)
#: Earliest imagery run of each family: the pseudo-online calibration run.
FAMILY_CALIBRATION_RUN = (4, 6)
#: Where the target's EA reference comes from, per ``--ea`` mode (logged).
EA_REFERENCE = {"pseudo-online": "calibration_run", "offline": "all_target_trials",
                "train-fold": "training_fold"}


def calibration_runs(classes) -> tuple[int, ...]:
    """The task's calibration run(s): the earliest run of each family its classes
    come from. ``(4,)`` for classes 0/1, ``(6,)`` for 2/3, ``(4, 6)`` for 4-class."""
    wanted = range(len(CLASS_FAMILY)) if not classes else classes
    return tuple(sorted({FAMILY_CALIBRATION_RUN[CLASS_FAMILY[c]] for c in wanted}))


def scored_trials(runs: np.ndarray, excluded) -> np.ndarray:
    """Indices of the trials whose run is not in ``excluded``; these are scored."""
    keep = np.flatnonzero(~np.isin(runs, list(excluded)))
    if not len(keep):
        raise RuntimeError(f"no trial left to score once runs {list(excluded)} are "
                           f"excluded")
    return keep


def target_reference_indices(runs: np.ndarray, classes, mode: str,
                             train_idx=None) -> np.ndarray:
    """The ONE target's trials its EA reference is estimated from.

    ``pseudo-online``: the calibration run only, which is then never scored.
    ``train-fold``: the fold's training trials only (``train_idx``), never a test
    trial. ``offline``: every trial -- the labelled transductive variant, the single
    exception to "references from training data only" (9.26 A6).
    """
    if mode == "pseudo-online":
        idx = np.flatnonzero(np.isin(runs, calibration_runs(classes)))
    elif mode == "offline":
        idx = np.arange(len(runs))
    elif mode == "train-fold":
        if train_idx is None:
            raise ValueError("train-fold needs the fold's training indices")
        idx = np.asarray(train_idx)
    else:
        raise ValueError(f"no target EA reference for mode {mode!r}")
    if not len(idx):
        raise RuntimeError(f"no trial to estimate the {mode} EA reference from")
    return idx


def align_each(parts: list[Trials]) -> Trials:
    """Pool subjects, each aligned to the reference of its own trials (EA sources).

    Filled into one preallocated array rather than concatenating aligned copies, so
    the pooled source set (~0.7 GB) never exists twice in host memory.
    """
    if not parts:
        raise ValueError("cannot pool zero subjects")
    X = np.empty((sum(len(p) for p in parts), *parts[0].X.shape[1:]), dtype=np.float32)
    i = 0
    for p in parts:
        X[i:i + len(p)] = align(p.X, fit_ea(p.X).W)
        i += len(p)
    return Trials(X=X, y=np.concatenate([p.y for p in parts]))


def target_windows(prepared, X: np.ndarray, y: np.ndarray, offset: int,
                   window: int, stride: int) -> Samples:
    """Window one target's trials exactly as ``prepare_fold`` windows its test split.

    The zero-shot and EA paths transform each target on its own, because under EA
    each target has its own alignment. Only the fold's SOURCE-fitted standardiser
    (and node permutation, for graph folds) is applied; nothing is fitted here.
    test_target_windows_match_the_group_windows pins this against prepare_fold.
    """
    Z = prepared.standardizer.apply(X)
    if prepared.bases:
        Z = perm_data(Z, prepared.perm)
    if window == 1 and stride == 1:
        return to_timepoint_samples(Z, y, trial_offset=offset)
    return to_window_samples(Z, y, window, stride, trial_offset=offset)


def score_samples(trainer, samples: Samples) -> dict:
    """Score windows with a fitted model: the one place zero-shot targets are scored.

    A named function so the tests can see exactly which trials reach scoring
    (Review Focus 2: under --calibration-run-excluded, never one of run 4 or 6).
    """
    if len(samples) == 0:
        raise RuntimeError("asked to score zero windows")
    return trainer.evaluate_samples(samples)


def alignment_fields(args) -> dict:
    """Log-row fields of the Phase A options, on every row of a run that uses one.

    Empty on the default path, like :func:`provenance_fields`, so existing arms keep
    exactly their old keys.
    """
    if not (args.zero_shot or args.calibration_run_excluded or args.ea != "none"):
        return {}
    return {"ea": args.ea, "zero_shot": args.zero_shot,
            "calibration_run_excluded": args.calibration_run_excluded,
            "transductive": args.ea == "offline",
            "ea_reference": EA_REFERENCE.get(args.ea)}


def ea_fields(ref: EAReference | None) -> dict:
    """The fitted target reference's rank and size, when there is one."""
    return {} if ref is None else {"ea_rank": ref.rank, "ea_n_trials": ref.n_trials}


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


def check_same_trials(noref_X: np.ndarray, ref_X: np.ndarray, where: str) -> None:
    """Fail unless the no-reference trials, re-referenced, ARE the referenced trials.

    Review Focus 5. Equal labels prove little for a binary problem, so the data of
    exactly the trials being scored are compared: CAR over all 64 channels must
    reproduce the referenced cache to ``1e-3 * median |X|``, the tolerance
    ``scripts/check_noref_cache.py`` verified for every subject.
    """
    rebuilt = common_average(noref_X)
    if rebuilt.shape != ref_X.shape:
        raise RuntimeError(f"{where}: no-reference trials {rebuilt.shape} vs "
                           f"referenced {ref_X.shape}")
    scale = float(np.median(np.abs(ref_X)))
    worst = float(np.abs(rebuilt - ref_X).max())
    if not worst <= 1e-3 * scale:
        raise RuntimeError(f"{where}: re-referenced no-reference trials differ from the "
                           f"referenced ones by {worst:.3e} (> 1e-3 x median|X| = "
                           f"{1e-3 * scale:.3e}); not the same trials")


def noref_provenance(noref_config, noref_dir) -> dict:
    """Fields naming the no-reference cache a condition row was computed from."""
    return {"noref_config": str(noref_config),
            "noref_fingerprint": read_manifest(Path(noref_dir))["config_fingerprint"]}


def require_source_windows(prepared) -> None:
    """Refuse to pre-train on a fold built without source windows.

    A fold prepared with ``train_samples=False`` (source-cache hit) has an empty
    training set. Fitting on it and writing the result to the cache under a valid
    key would poison every later run that hits that key.
    """
    if len(prepared.train) == 0:
        raise RuntimeError("pre-training would run with no source windows (the fold "
                           "was prepared for a cache hit); refusing to fit and cache it")


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
        # The region basis is built from this cache. The key hashes only this *path
        # string*, not the cache's manifest fingerprint, so a regenerated cache at the
        # same path would NOT change the key. Changing the key (e.g. to hash the
        # fingerprint) would orphan the cached ROB1-chebrp source models, so it is left
        # as is; the fingerprint is logged on every row (`noref_fingerprint`) instead.
        fields["noref_config"] = str(args.noref_config)
    # "ea" likewise only when set (getattr: namespaces predating the option lack it).
    # It records how the SOURCES were aligned -- each subject to its own trials -- which
    # is the same for every EA mode, so the fine-tuned arm (train-fold) reuses the
    # zero-shot arm's (pseudo-online) EA source models. The target's reference never
    # touches the source weights, so it is not hashed; nor are --zero-shot and
    # --calibration-run-excluded, which only change what is scored.
    if getattr(args, "ea", "none") != "none":
        fields["ea"] = "each-source-subject-own-trials"
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
    ap.add_argument("--zero-shot", action="store_true",
                    help="score each target with its group's source model WITHOUT "
                         "fine-tuning (9.26 A1, A6): one row per target, "
                         "finetune_epochs 0. Not part of the source cache key, so "
                         "--source-tag can reuse another arm's models")
    ap.add_argument("--calibration-run-excluded", action="store_true",
                    help="--zero-shot only: never score the task's calibration run "
                         "(run 4 for classes 0/1, run 6 for classes 2/3, both without "
                         "--classes). A6 excludes it in both arms so they stay paired; "
                         "under --ea pseudo-online it is the target's EA reference")
    ap.add_argument("--ea", choices=["none", "pseudo-online", "offline", "train-fold"],
                    default="none",
                    help="Euclidean alignment of the band-passed trials, before "
                         "windowing. Every source subject is aligned to its own trials. "
                         "Target reference: pseudo-online = its calibration run "
                         "(--zero-shot --calibration-run-excluded); offline = all its "
                         "trials (--zero-shot; TRANSDUCTIVE, logged as such); "
                         "train-fold = the fold's training trials, test trials aligned "
                         "with it (fine-tuning)")
    ap.add_argument("--require-source-cache", action="store_true",
                    help="fail instead of pre-training when a group's source model is "
                         "not cached: for arms defined as reusing cached models (A6's "
                         "no-EA arm scores PTG-shallow's)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    # Phase A options (9.26 A1, A6), checked before any data is loaded.
    if args.zero_shot and args.no_pretrain:
        ap.error("--zero-shot scores the pre-trained source model; with --no-pretrain "
                 "there is nothing to score")
    if args.zero_shot and (args.robustness or args.dsbn):
        ap.error("--zero-shot is not defined with --robustness or --dsbn")
    if args.calibration_run_excluded and not args.zero_shot:
        ap.error("--calibration-run-excluded applies to --zero-shot only; fine-tuned "
                 "arms score every trial, as PTG-shallow did")
    if args.ea in ("pseudo-online", "offline") and not args.zero_shot:
        ap.error(f"--ea {args.ea} is a zero-shot target reference; fine-tuning takes "
                 f"its reference from the training fold (--ea train-fold)")
    if args.ea == "train-fold" and args.zero_shot:
        ap.error("--ea train-fold is for fine-tuning; a zero-shot target has no "
                 "training fold (use pseudo-online or offline)")
    if args.ea == "pseudo-online" and not args.calibration_run_excluded:
        ap.error("--ea pseudo-online takes the target's reference from its calibration "
                 "run, which must then never be scored: add --calibration-run-excluded")
    if args.ea != "none" and (args.dsbn or args.robustness or args.readout != "flat"):
        ap.error("--ea is not defined with --dsbn, --robustness or --readout region")
    if args.require_source_cache and (args.no_pretrain or not args.source_cache):
        ap.error("--require-source-cache needs pre-training and a --source-cache")

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
    # The run of every kept trial: what --calibration-run-excluded and the pseudo-online
    # EA reference select on, and what A1's second confusion matrix is computed over.
    runs: dict[str, np.ndarray] = {}
    for s in subjects:
        ep = load_subject(s, cfg.dataset.cache_dir)
        X, y, run = ep.X, ep.y, np.asarray(ep.run)
        if args.classes:
            mask = np.isin(y, args.classes)
            X, y = X[mask], (y[mask] == args.classes[1]).astype(np.int64)
            run = run[mask]
        trials[s] = Trials(X=X, y=y)
        runs[s] = run
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
    # Condition rows name the cache they were computed from (the clean row, for
    # ShallowConvNet, never reads it and so does not).
    noref_prov = (noref_provenance(args.noref_config, noref_dir)
                  if args.robustness else None)

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
    excluded_runs = calibration_runs(args.classes) if args.calibration_run_excluded else ()
    if args.zero_shot:
        print("ZERO-SHOT: each target scored with its group's source model, no "
              "fine-tuning" + (f"; run(s) {list(excluded_runs)} never scored"
                               if excluded_runs else "") + "\n")
    else:
        print(f"FINE-TUNE {args.finetune_epochs} epochs @ {args.finetune_lr} "
              f"on the target's training folds\n")
    if args.ea != "none":
        print(f"EA: each source subject aligned to its own trials; targets to "
              f"{EA_REFERENCE[args.ea]}"
              + (" (TRANSDUCTIVE)" if args.ea == "offline" else "") + "\n")
    # A zero-shot target has one row and no fold; its resume key is (subject, None).
    fold_ids = [None] if args.zero_shot else list(range(args.folds))
    # Zero-shot and EA targets are windowed one at a time (each may have its own
    # alignment), so the group is not windowed as one test split.
    per_target = args.zero_shot or args.ea != "none"

    done = resume_keys(args.log, args.tag, args.robustness) if args.resume else set()
    # Per tag as well, so a fold re-run after a partial write logs only the rows that
    # are missing and never duplicates one (find_duplicates would flag it).
    clean_done = completed_keys(args.log, args.tag) if args.resume else set()
    cond_done = {c.name: (completed_keys(args.log, condition_tag(args.tag, c.name))
                          if args.resume else set())
                 for c in ROBUSTNESS_CONDITIONS} if args.robustness else {}
    if args.resume:
        # Count only keys of the requested targets: under --targets the log may hold
        # other subjects' rows, which would read as "130 of 100".
        print(describe_resume({k for k in done if k[0] in target_set},
                              len(target_set) * len(fold_ids), args.tag))

    per_subject: list[float] = []
    t0 = time.time()

    for gi, group, source in plan:
        if all((t, f) in done for t in group if t in target_set
               for f in fold_ids):
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
        # Under --ea every source subject (training and validation alike) is aligned to
        # the reference of its own trials before anything is fitted on it.
        pool = align_each if args.ea != "none" else concat_trials
        src_train = pool([trials[s] for s in source if s not in val_pick])
        src_val = pool([trials[s] for s in val_pick])

        # One representation for both phases: graph and standardiser fit on SOURCE.
        # The group's trials are the "test" argument purely so they are transformed by
        # the same statistics; nothing about them is fit. concat_trials preserves order,
        # so each target occupies a known contiguous span of trial ids.
        group_trials = [trials[t] for t in group]
        offsets = np.cumsum([0] + [len(t) for t in group_trials])
        # Zero-shot and EA targets are windowed one at a time below (target_windows),
        # with the same ids, so the group's split is empty here and never windowed.
        # A fresh empty array, not src_train.X[:0]: a view would keep the whole source
        # array alive past the `del` below.
        group_test = (Trials(X=np.empty((0, *src_train.X.shape[1:]), np.float32),
                             y=np.empty(0, np.int64)) if per_target
                      else concat_trials(group_trials))

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
        # Decided ONCE: re-checking the file later could see it vanish in between and
        # pre-train on a fold built without source windows (guarded below too).
        cache_hit = cache_path is not None and cache_path.exists()
        if args.require_source_cache and not cache_hit:
            raise RuntimeError(f"group {gi}: {cache_path} is not in the source cache and "
                               f"--require-source-cache forbids pre-training a "
                               f"replacement; check --source-tag and the key options")
        source_windows = not (cache_hit and not args.dsbn)
        prepared = prepare_fold(
            src_train, group_test, src_val if source_windows else None,
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
            if not np.array_equal(rep_all.regions, region_vector(ch_names)):
                raise RuntimeError(
                    "region order mismatch: the subset representation's region vector "
                    "differs from region_vector(ch_names); the region readout would "
                    "average the wrong electrodes")
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
            if cache_hit:
                pretrained_state = torch.load(cache_path, map_location="cpu",
                                              weights_only=True)
                if not args.quiet:
                    print(f"group {gi}/{len(groups)}: reused cached source model "
                          f"({cache_path.name}) [{time.time() - t0:.0f}s elapsed]",
                          flush=True)
            else:
                require_source_windows(prepared)
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

        if args.zero_shot:
            # One scoring model per group: the source weights exactly as pre-trained
            # (or loaded from the cache). The Trainer only predicts; fit() is never
            # called on it, so no optimiser step touches the weights.
            model = build_model(args.model, prepared, args.window, n_classes, sfreq,
                                spatial=args.spatial,
                                graph_smoothness=args.graph_smoothness,
                                readout=args.readout, region_graph=region_graph)
            model.load_state_dict(pretrained_state)
            trainer = Trainer(model, ft_cfg, n_classes=n_classes)

        for j, target in enumerate(group):
            # j stays the member's index in the FULL group: it seeds the fine-tune
            # stage and locates the target's span in the group's concatenation.
            if target not in target_set:
                continue
            if all((target, f) in done for f in fold_ids):
                continue
            # This target's trials are a contiguous span of the concatenated group.
            base = 200_000 + int(offsets[j])

            if args.zero_shot:
                # ------------------------------------------------------ zero-shot
                X_t, y_t, runs_t = trials[target].X, trials[target].y, runs[target]
                ea_ref = None
                if args.ea != "none":
                    # pseudo-online: the calibration run only (never scored below);
                    # offline: every trial (transductive, labelled on the row).
                    ref_idx = target_reference_indices(runs_t, args.classes, args.ea)
                    ea_ref = fit_ea(X_t[ref_idx])
                    X_t = align(X_t, ea_ref.W)
                tgt = target_windows(prepared, X_t, y_t, base, args.window, args.stride)
                # Review Focus 2: the calibration run's trials are dropped HERE, before
                # scoring, by run number alone, in both arms (with and without EA).
                keep = scored_trials(runs_t, excluded_runs)
                result = score_samples(trainer,
                                       subset_samples(tgt, set((keep + base).tolist())))
                row = {
                    "experiment": "pretrain_finetune",
                    "tag": args.tag,
                    "subject": target,
                    "fold": None,
                    "n_folds": None,
                    "protocol": "zero_shot",
                    "split_type": "group_subject_disjoint",
                    **seed_info,
                    "config_source": str(cfg.source_path),
                    "preprocess": {"bandpass": cfg.preprocess.bandpass,
                                   "rereference": cfg.preprocess.rereference},
                    "classes": args.classes,
                    "n_classes": n_classes,
                    "window": args.window,
                    "stride": args.stride,
                    "model": args.model,
                    "pretrained": True,
                    "dsbn": False,
                    "n_source_subjects": len(source),
                    "n_groups": len(groups),
                    "group_id": gi,
                    "group_size": len(group),
                    "pretrain_epochs": args.pretrain_epochs,
                    "finetune_epochs": 0,
                    "pretrain_lr": args.pretrain_lr,
                    "finetune_lr": None,
                    "train_config": {"epochs": 0, "batch_size": args.batch_size,
                                     "learning_rate": None, "device": ft_cfg.device},
                    "n_parameters": sum(p.numel() for p in model.parameters()),
                    "accuracy": result["trial"].accuracy,
                    "trial_accuracy": result["trial"].accuracy,
                    "timepoint_accuracy": result["timepoint"].accuracy,
                    "trial_confusion": result["trial"].confusion.tolist(),
                    "trial_per_class": list(result["trial"].per_class_accuracy),
                    "trial_kappa": result["trial"].kappa,
                    "trial_macro_f1": result["trial"].macro_f1,
                    "n_trials_scored": int(len(keep)),
                    "scored_runs": sorted(set(runs_t[keep].tolist())),
                    "calibration_runs": list(calibration_runs(args.classes)),
                    **alignment_fields(args),
                    **ea_fields(ea_ref),
                    **provenance_fields(args),
                }
                if not args.classes:
                    # A1: the same predictions restricted to the trials outside runs 4
                    # and 6, the trial set A6's subject-independent arms are scored on.
                    outside = scored_trials(runs_t, calibration_runs(None))
                    excl = score_samples(trainer,
                                         subset_samples(tgt, set((outside + base).tolist())))
                    row["confusion_excl_calibration"] = excl["trial"].confusion.tolist()
                    row["n_trials_excl_calibration"] = int(len(outside))
                log_experiment(row, path=args.log)
                per_subject.append(float(result["trial"].accuracy))
                if not args.quiet:
                    print(f"  {target}: zero-shot trial {per_subject[-1]:.4f} "
                          f"({len(keep)} trials scored)", flush=True)
                continue

            # -------------------------------------------------------- fine-tune
            # The same trial-level folds run_subject_dependent.py uses at this
            # seed, so the arms are paired subject by subject and fold by fold.
            y_target = trials[target].y
            fold_accs = []
            for split in stratified_trial_kfold(y_target, n_splits=args.folds, seed=seed):
                if (target, split.fold) in done:
                    continue
                ea_ref = None
                tgt = prepared.test
                if args.ea == "train-fold":
                    # The target's reference from this fold's TRAINING trials only; its
                    # test trials are aligned with that reference, never part of it.
                    ref_idx = target_reference_indices(runs[target], args.classes,
                                                       args.ea, train_idx=split.train)
                    ea_ref = fit_ea(trials[target].X[ref_idx])
                    tgt = target_windows(prepared, align(trials[target].X, ea_ref.W),
                                         y_target, base, args.window, args.stride)
                ft_train = subset_samples(tgt, set(split.train + base))
                ft_test = subset_samples(tgt, set(split.test + base))

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
                    **alignment_fields(args),
                    **ea_fields(ea_ref),
                    **provenance_fields(args),
                }

                if args.robustness:
                    # The same test trials, from the no-reference cache. Same order as
                    # the referenced cache (checked per subject at load), and labels
                    # AND data are checked again on exactly the trials being scored.
                    test_idx = np.asarray(split.test)
                    nr = noref_targets[target]
                    test_noref = Trials(X=nr.X[test_idx], y=nr.y[test_idx])
                    if not np.array_equal(test_noref.y, y_target[test_idx]):
                        raise RuntimeError(f"{target} fold {split.fold}: no-reference "
                                           f"test labels differ from the referenced ones")
                    check_same_trials(test_noref.X, trials[target].X[test_idx],
                                      f"{target} fold {split.fold}")
                    pending = tuple(c for c in ROBUSTNESS_CONDITIONS
                                    if (target, split.fold) not in cond_done[c.name])
                    scores = evaluate_conditions(
                        trainer, "graph" if args.readout == "region" else "shallow",
                        test_noref, base, ch_names, m_src, prepared.standardizer,
                        rep_all, target, split.fold, pending, args.window, args.stride,
                        lmax=args.lmax)
                    for c in pending:
                        log_experiment(condition_row(row, c.name, scores[c.name],
                                                     provenance=noref_prov),
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

        # Release this group's data BEFORE the next group builds its own. Python frees
        # these only when the names are rebound, which otherwise happens after the next
        # group's windows already exist, so the peak was two groups and the second group
        # of a process died allocating in fit_standardizer. Rebinding (not del) because
        # some names are unbound when a group's targets were all skipped. Nothing here
        # is read again: every value is rebuilt from its inputs at the top of the loop.
        prepared = model = trainer = pretrained_state = None
        ft_train = ft_test = ft_fold = history = result = None
        scores = test_noref = nr = m_src = rep_all = region_graph = None
        group_test = tgt = excl = X_t = None
        release_host_memory()

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
