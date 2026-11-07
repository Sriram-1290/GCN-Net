"""Subject-independent tangent-space transfer by group (9.26 A6 arm 2, and A1).

    python scripts/run_tangent_transfer.py --classes 0 1 --ea pseudo-online \
        --tag PA-A6-si-tgsp-ea-binary-0-1 --resume

**Why this exists.** A6 tests Euclidean alignment (EA) on the deep model, and giving EA
to the deep model but not to the classical baseline is an anti-goal of the design: EA
was built for covariance-based classifiers, so the tangent-space arm is the control
that says whether any EA effect is a data-alignment effect or an architecture effect.
With ``--classes`` omitted the same machinery runs A1's 4-class tangent-space arm.

Protocol
--------
* **Groups:** ``plan_groups`` from ``run_pretrain_finetune.py``, i.e.
  ``assign_groups(all subjects, 10)``, index mod 10 -- the same groups as every
  pre-training arm, so the results pair with them subject by subject.
* **Per group:** shrunk covariances (shrinkage 0.1) of every source subject's trials;
  the tangent point is the Riemannian mean of the POOLED source covariances; a
  ``RobustScaler`` and ``LogisticRegression(C=0.01, max_iter=2000)`` are fitted on the
  sources -- the settings of ``run_tangent_space.tangent_accuracy``, unchanged, since
  nothing in Phase A is tuned. Each target of the group is then scored.
* **Binary** (``--classes A B``): the task's calibration run (run 4 for classes 0/1,
  run 6 for classes 2/3) is never scored, with or without EA, so this arm pairs with
  the deep model's (Review Focus 2).
* **4-class** (no ``--classes``, A1): every trial is scored; each row also carries
  ``confusion_excl_calibration``, the same predictions over the trials outside runs 4
  and 6, for the comparison with A6. ``--ea`` must be ``none`` (A1 measures the
  unaligned protocol).

Leakage
-------
Nothing of a group's subjects enters its fit: not their trials, not the tangent point,
not the scaler. Under ``--ea`` each source subject is aligned to the reference of its own
trials; the target's reference is its calibration run (``pseudo-online``, which is then
never scored) or all its trials (``offline``, TRANSDUCTIVE and logged as such).

Memory
------
Subjects are loaded one at a time and only their covariances are kept (~1.5 MB per
subject for a binary task), so the raw trials of all 103 subjects never sit in memory.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.alignment import EAReference, align, fit_ea  # noqa: E402
from data.cache import load_subject  # noqa: E402
from data.config import all_subjects, load_config  # noqa: E402
from data.riemann import covariances, riemannian_mean, tangent_space  # noqa: E402
from evaluation.experiment_log import (  # noqa: E402
    completed_keys, describe_resume, log_experiment, set_seed)
from evaluation.metrics import evaluate  # noqa: E402
from scripts.run_pretrain_finetune import (  # noqa: E402
    EA_REFERENCE, assign_groups, calibration_runs, ea_fields, plan_groups, scored_trials,
    target_reference_indices)

#: Covariance shrinkage toward the scaled identity: data/riemann.py's default.
SHRINKAGE = 0.1


class TangentModel(NamedTuple):
    """A fitted transfer model: tangent point, scaler and classifier."""

    ref: np.ndarray   # (C, C) Riemannian mean of the pooled source covariances
    scaler: object
    clf: object


def fit_transfer(covs: np.ndarray, y: np.ndarray) -> TangentModel:
    """Fit on source covariances only: tangent point, scaler and classifier.

    The classifier settings are ``run_tangent_space.tangent_accuracy``'s, unchanged.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler

    ref = riemannian_mean(covs)
    z = tangent_space(covs, ref)
    scaler = RobustScaler().fit(z)
    clf = LogisticRegression(C=0.01, max_iter=2000)
    clf.fit(scaler.transform(z), y)
    return TangentModel(ref=ref, scaler=scaler, clf=clf)


def score_trials(model: TangentModel, covs: np.ndarray) -> np.ndarray:
    """Predict one class per trial: the one place targets are scored.

    A named function so the tests can see exactly which trials reach scoring
    (Review Focus 2: in the binary arms, never one of the calibration run).
    """
    if not len(covs):
        raise RuntimeError("asked to score zero trials")
    return model.clf.predict(model.scaler.transform(tangent_space(covs, model.ref)))


def load_task(subject: str, cache_dir, classes) -> tuple[np.ndarray, np.ndarray,
                                                        np.ndarray]:
    """One subject's trials, labels and runs for the task (binary labels 0/1)."""
    ep = load_subject(subject, cache_dir)
    X, y, run = ep.X, ep.y, np.asarray(ep.run)
    if classes:
        mask = np.isin(y, classes)
        X, y, run = X[mask], (y[mask] == classes[1]).astype(np.int64), run[mask]
    return X, y, run


def source_covariances(X: np.ndarray, ea: bool) -> np.ndarray:
    """A source subject's covariances; under EA aligned to its own trials first."""
    if ea:
        X = align(X, fit_ea(X).W)
    return covariances(X, shrinkage=SHRINKAGE)


def target_covariances(X: np.ndarray, runs: np.ndarray, classes,
                       mode: str) -> tuple[np.ndarray, EAReference | None]:
    """A target's covariances, aligned to the reference ``mode`` names (or not)."""
    ref = None
    if mode != "none":
        ref = fit_ea(X[target_reference_indices(runs, classes, mode)])
        X = align(X, ref.W)
    return covariances(X, shrinkage=SHRINKAGE), ref


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/bandpass_8_30.yaml")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--targets", nargs="+", default=None, metavar="SUBJECT",
                    help="subjects to score (default: all). Groups are still assigned "
                         "over every subject, so each target keeps its full-run sources")
    ap.add_argument("--groups", type=int, default=10)
    ap.add_argument("--classes", nargs=2, type=int, default=None, metavar=("A", "B"),
                    help="binary task, calibration run never scored; omitted: 4-class "
                         "(A1), every trial scored")
    ap.add_argument("--ea", choices=["none", "pseudo-online", "offline"], default="none",
                    help="Euclidean alignment. Sources: each to its own trials. Target: "
                         "pseudo-online = its calibration run; offline = all its trials "
                         "(TRANSDUCTIVE, logged as such)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log", default="runs/experiments.jsonl")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.classes is None and args.ea != "none":
        ap.error("the 4-class task (A1) measures the unaligned protocol: --ea must be "
                 "none without --classes")

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    seed_info = set_seed(seed)
    subjects = args.subjects or all_subjects(cfg)
    plan = plan_groups(subjects, args.groups, args.targets)
    groups = assign_groups(subjects, args.groups)
    target_set = set(args.targets) if args.targets is not None else set(subjects)
    binary = bool(args.classes)
    n_classes = 2 if binary else len(cfg.class_names)
    excluded = calibration_runs(args.classes) if binary else ()

    done = completed_keys(args.log, args.tag) if args.resume else set()
    if args.resume:
        print(describe_resume({k for k in done if k[0] in target_set},
                              len(target_set), args.tag))
    pending = {t for _, group, _ in plan for t in group
               if t in target_set and (t, None) not in done}
    todo = [(gi, group, source) for gi, group, source in plan if set(group) & pending]
    if not todo:
        print("nothing to do; every target already logged under this tag")
        return 0

    print(f"tangent-space transfer  classes {args.classes or 'all 4'}  ea {args.ea}  "
          f"{len(groups)} groups" + (f"; run(s) {list(excluded)} never scored"
                                     if excluded else ""), flush=True)
    if args.ea == "offline":
        print("EA offline: the target reference uses all its trials (TRANSDUCTIVE)")

    # One subject at a time; only covariances are kept.
    need_src = set().union(*(set(source) for _, _, source in todo))
    src_cov: dict[str, np.ndarray] = {}
    src_y: dict[str, np.ndarray] = {}
    tgt: dict[str, tuple] = {}
    t0 = time.time()
    for s in subjects:
        if s not in need_src and s not in pending:
            continue
        X, y, run = load_task(s, cfg.dataset.cache_dir, args.classes)
        if s in need_src:
            src_cov[s], src_y[s] = source_covariances(X, args.ea != "none"), y
        if s in pending:
            if args.ea == "none" and s in src_cov:
                covs, ref = src_cov[s], None      # unaligned: the same matrices
            else:
                covs, ref = target_covariances(X, run, args.classes, args.ea)
            tgt[s] = (covs, y, run, ref)
        del X
    if not args.quiet:
        print(f"covariances of {len(set(src_cov) | set(tgt))} subjects "
              f"[{time.time() - t0:.0f}s]", flush=True)

    per_subject: list[float] = []
    for gi, group, source in todo:
        C = np.concatenate([src_cov[s] for s in source])
        Y = np.concatenate([src_y[s] for s in source])
        model = fit_transfer(C, Y)
        if not args.quiet:
            print(f"group {gi}/{len(groups)}: fitted on {len(source)} subjects, "
                  f"{len(Y)} trials [{time.time() - t0:.0f}s]", flush=True)
        for t in group:
            if t not in pending:
                continue
            covs, y, run, ref = tgt.pop(t)
            # Review Focus 2: the calibration run is dropped HERE, by run number alone.
            keep = scored_trials(run, excluded)
            pred = score_trials(model, covs[keep])
            res = evaluate(y[keep], pred, n_classes)
            row = {
                "experiment": "tangent_transfer",
                "tag": args.tag,
                "subject": t,
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
                "model": "cov-tgsp",
                "shrinkage": SHRINKAGE,
                "tangent_point": "riemannian_mean_of_pooled_source_covariances",
                "classifier": "RobustScaler + LogisticRegression(C=0.01)",
                "n_source_subjects": len(source),
                "n_source_trials": int(len(Y)),
                "n_groups": len(groups),
                "group_id": gi,
                "group_size": len(group),
                "finetune_epochs": 0,
                "accuracy": res.accuracy,
                "trial_accuracy": res.accuracy,
                "trial_confusion": res.confusion.tolist(),
                "trial_per_class": list(res.per_class_accuracy),
                "trial_kappa": res.kappa,
                "trial_macro_f1": res.macro_f1,
                "n_trials_scored": int(len(keep)),
                "scored_runs": sorted(set(run[keep].tolist())),
                "calibration_runs": list(calibration_runs(args.classes)),
                "ea": args.ea,
                "zero_shot": True,
                "calibration_run_excluded": binary,
                "transductive": args.ea == "offline",
                "ea_reference": EA_REFERENCE.get(args.ea),
                **ea_fields(ref),
            }
            if not binary:
                # A1: the same predictions over the trials outside runs 4 and 6.
                outside = ~np.isin(run[keep], calibration_runs(None))
                excl = evaluate(y[keep][outside], pred[outside], n_classes)
                row["confusion_excl_calibration"] = excl.confusion.tolist()
                row["n_trials_excl_calibration"] = int(outside.sum())
            log_experiment(row, path=args.log)
            per_subject.append(float(res.accuracy))
            if not args.quiet:
                print(f"  {t}: {res.accuracy:.4f} ({len(keep)} trials scored)",
                      flush=True)
        # Release this group's pooled covariances and model before the next group.
        C = Y = model = None
        gc.collect()

    print("\n" + "=" * 64)
    print(f"subjects: {len(per_subject)}")
    print(f"mean trial accuracy  {np.mean(per_subject):.4f} "
          f"+/- {np.std(per_subject):.4f}")
    print(f"log -> {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
