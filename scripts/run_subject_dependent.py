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
import datetime as dt
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
from models.eegnet import build_baseline  # noqa: E402
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
    ap.add_argument("--temporal-pool", choices=["segments"], default=None,
                    help="fixed mean pool instead of attention: the H2 control")
    ap.add_argument("--attention-per-node", action="store_true")
    ap.add_argument("--learnable-graph", action="store_true",
                    help="learn the adjacency, initialised at the PCC graph so the "
                         "model starts identical to the fixed baseline")
    ap.add_argument("--learn-graph-scale", action="store_true")
    ap.add_argument("--csp-init", action="store_true",
                    help="--model graphbp with --spatial free: initialise the "
                         "spatial operators from CSP filters fitted on the fold's "
                         "TRAINING windows instead of from the electrode graph. "
                         "Same shapes, same trainability -- the only change is the "
                         "prior the operator starts from (9.13 showed the graph "
                         "prior is inert; see models/csp_prior.py).")
    ap.add_argument("--model", choices=["gcn", "eegnet", "shallow", "graphbp"],
                    default="gcn",
                    help="A2 (17.2): run a graph-free CNN baseline on the SAME folds, "
                         "cache, seeds and metric. eegnet = EEGNet-8,2; shallow = "
                         "ShallowConvNet. Discriminates the graph-prior explanation "
                         "from the sample-size one (17.1)")
    ap.add_argument("--free-spatial", action="store_true",
                    help="A1 (17.2): replace the Chebyshev basis with free dense "
                         "(K, M, M) operators, initialised at that basis so step 0 is "
                         "bit-identical. Lifts graph locality and the polynomial "
                         "constraint -- the direct test of 9.7.1. Not combinable with "
                         "--learnable-graph")
    ap.add_argument("--adjacency-l1", type=float, default=0.0,
                    help="L1 on the learned graph correction (sparsity)")
    ap.add_argument("--spatial", choices=["graph", "free"], default="graph",
                    help="--model graphbp only: Chebyshev graph convolution, or an "
                         "unconstrained operator the graph only regularises")
    ap.add_argument("--graph-smoothness", type=float, default=0.0,
                    help="--model graphbp with --spatial free: weight on the "
                         "Dirichlet-energy prior over the electrode graph. The "
                         "regulariser variant; 0 disables it")
    ap.add_argument("--gbp-layers", type=int, default=1,
                    help="--model graphbp only: spatial layers (default 1, against "
                         "GCNs-Net's 6)")
    ap.add_argument("--pool-nodes", action="store_true",
                    help="--model graphbp only: restore Graclus node pooling, so "
                         "dropping it stays an ablation rather than an assumption")
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
    ap.add_argument("--window-start", type=int, default=0,
                    help="sample index of the first window in every trial (train, val "
                         "and test alike); 0 keeps the original behaviour. Non-zero "
                         "values are logged as window_start")
    ap.add_argument("--lmax", type=float, default=2.0,
                    help="2.0 reproduces the reference; pass 0 for the true lambda_max")
    ap.add_argument("--leaky", action="store_true", help="reproduce the reference protocol")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log", default="runs/experiments.jsonl")
    ap.add_argument("--tag", default="", help="free-text note stored with each row")
    ap.add_argument("--stop-at", default=None, metavar="[YYYY-MM-DD ]HH:MM",
                    help="exit cleanly at this local time, at a FOLD BOUNDARY. "
                         "Use this rather than killing the process: a kill can orphan "
                         "the python child (error #11) and can interrupt a fold "
                         "mid-write. A bare HH:MM means TODAY and a past one stops "
                         "immediately rather than rolling to tomorrow, so the flag is "
                         "safe to repeat inside a loop (error 23). Use "
                         "'YYYY-MM-DD HH:MM' for an overnight deadline. Combine with "
                         "--resume to continue later, losing nothing")
    ap.add_argument("--resume", action="store_true",
                    help="skip (subject, fold) pairs already logged under this tag, "
                         "so an interrupted run can be restarted with the same command")
    ap.add_argument("--checkpoints", action="store_true",
                    help="save best-by-validation weights under checkpoints/<tag>/. "
                         "Off by default: 42 runs previously overwrote one set of 515 "
                         "untagged files, 763 MB that identified nothing.")
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

    # A2's baselines have no graph and no attention module. Silently ignoring a
    # flag would produce a log row claiming a configuration the run did not have,
    # which is how the tag pollution in error #12 happened. Fail instead.
    # GCNsNet-only flags. graphbp uses a graph but none of these modules, so it
    # belongs here too: silently logging a row that claims a configuration the run
    # did not have is error #12.
    if args.model != "gcn":
        ignored = [name for name, on in [
            ("--temporal-attention", args.temporal_attention),
            ("--temporal-pool", bool(args.temporal_pool)),
            ("--learnable-graph", args.learnable_graph),
            ("--free-spatial", args.free_spatial),
            ("--adjacency-l1", bool(args.adjacency_l1)),
        ] if on]
        if ignored:
            raise SystemExit(
                f"--model {args.model} does not use {', '.join(ignored)}; they would "
                "be ignored. Drop the flag, or use --model gcn."
            )
    # ...and the converse: graphbp-only flags must not be silently dropped either.
    if args.model != "graphbp":
        ignored = [name for name, on in [
            ("--spatial free", args.spatial != "graph"),
            ("--graph-smoothness", bool(args.graph_smoothness)),
            ("--gbp-layers", args.gbp_layers != 1),
            ("--pool-nodes", args.pool_nodes),
            ("--csp-init", args.csp_init),
        ] if on]
        if ignored:
            raise SystemExit(
                f"{', '.join(ignored)} applies only to --model graphbp; got "
                f"--model {args.model}."
            )
    # The CSP basis replaces the graph basis as an *initialisation*, which only
    # means anything if the operators are learnable. With spatial="graph" the
    # basis is a fixed buffer, so a CSP basis there would silently become a hard
    # CSP constraint -- a different experiment, and not the one this flag names.
    if args.csp_init and args.spatial != "free":
        raise SystemExit(
            "--csp-init initialises a *learnable* operator and needs --spatial free; "
            "with --spatial graph the basis is fixed, which would make it a hard "
            "constraint rather than a prior."
        )
    if args.model in ("eegnet", "shallow"):
        if args.window < 32:
            raise SystemExit(
                f"--model {args.model} is a windowed CNN and needs --window 32 "
                f"or more; got {args.window}. The reference's single-timepoint "
                "sample (--window 1) has no temporal extent for it to convolve."
            )
        if args.leaky:
            raise SystemExit("--leaky reproduces the reference GCN protocol; "
                             "it has no meaning for --model " + args.model)
    if args.temporal_pool or args.temporal_attention:
        pass  # in_features stays the window; the pool reduces it inside the model
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
                              temporal_pool=args.temporal_pool,
                              attention_heads=args.attention_heads,
                              attention_per_node=args.attention_per_node,
                              learnable_graph=args.learnable_graph,
                              free_spatial=args.free_spatial,
                              learn_graph_scale=args.learn_graph_scale,
                              adjacency_l1=args.adjacency_l1,
                              always_train_bn=args.leaky)

    print(f"subjects {len(subjects)}  folds {args.folds}  epochs {args.epochs}  "
          f"lr {train_cfg.learning_rate}  lmax {lmax}  leaky {args.leaky}  seed {seed}")
    print(f"cache {cfg.dataset.cache_dir}  bandpass {cfg.preprocess.bandpass}  "
          f"window {args.window}  stride {args.stride or args.window}")
    print(f"attention {'ON (' + str(args.attention_heads) + ' heads)' if args.temporal_attention else 'off'}")
    print(f"model {args.model}"
          + ("   (no graph; A2 control for 17.1)" if args.model != "gcn" else ""))
    print("graph " + ("n/a" if args.model != "gcn"
                     else "FREE SPATIAL (init at Chebyshev basis)" if args.free_spatial
                     else "LEARNABLE (init at PCC)" if args.learnable_graph
                     else "fixed PCC")
          + (f"  L1 {args.adjacency_l1}" if args.adjacency_l1 else ""))
    print(f"features {args.features}"
          + (f" ({args.bands}, {n_features} bands)" if args.features == "bandpower" else ""))
    print(f"device {train_cfg.device}\n")

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
        # A time already past means stop NOW, never "tomorrow". The rollover was the
        # obvious convenience and it is actively wrong here (error 23): these runs are
        # driven by a shell loop over the two binary problems, so when the first
        # invocation stops at 03:00 the second starts at 03:00:0x, reads its own
        # deadline as already past, and schedules itself for the *following* day --
        # a 24-hour run from a flag meant to bound one to fifty minutes. Stopping
        # immediately is the only reading that composes correctly under repetition.
        if deadline <= now:
            print(f"--stop-at {args.stop_at} is already past; stopping immediately. "
                  "Nothing was run. Use --resume with a later time to continue.")
            return 0
        print(f"stopping at {deadline:%Y-%m-%d %H:%M} "
              f"({(deadline - now).total_seconds() / 60:.0f} min from now), "
              "at a fold boundary")

    done = (completed_keys(args.log, args.tag, window_start=args.window_start)
            if args.resume else set())
    if args.resume:
        print(describe_resume(done, len(subjects) * args.folds, args.tag))

    per_subject: list[float] = []
    per_subject_tp: list[float] = []
    stopped_early = False

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
            # Checked before starting a fold, never during one, so the log never
            # gains a partial row and --resume picks up exactly here.
            if deadline is not None and dt.datetime.now() >= deadline:
                print(f"\nreached --stop-at {args.stop_at}; stopping cleanly "
                      f"before {subject} fold {split.fold}. "
                      "Re-run the same command with --resume to continue.")
                stopped_early = True
                break
            tr_idx, val_idx = split_off_validation(ep.y, split.train, args.val_frac, seed)
            train = Trials(X=ep.X[tr_idx], y=ep.y[tr_idx])
            val = Trials(X=ep.X[val_idx], y=ep.y[val_idx]) if len(val_idx) else None
            test = Trials(X=ep.X[split.test], y=ep.y[split.test])

            fold = prepare_fold(train, test, val, lmax=lmax, seed=seed,
                                sfreq=ep.sfreq, window=args.window, stride=args.stride,
                                window_start=args.window_start,
                                features=args.features, bands=args.bands,
                                leaky=args.leaky,
                                build_graph=args.model in ("gcn", "graphbp"))
            if args.model == "gcn":
                model = GCNsNet(fold.bases, model_cfg, adjacencies=fold.adjacencies)
            elif args.model == "graphbp":
                from graphs.laplacian import laplacian
                from models.graph_bandpower import (GraphBandPowerConfig,
                                                    GraphBandPowerNet)
                gbp_cfg = GraphBandPowerConfig(
                    n_classes=n_classes, spatial=args.spatial,
                    n_layers=args.gbp_layers, pool_nodes=args.pool_nodes,
                    graph_smoothness=args.graph_smoothness)
                # The layer-0 coarsened graph, which matches the node ordering
                # the model sees -- not the raw 64x64 adjacency.
                lap = (laplacian(fold.adjacencies[0], normalized=False)
                       if args.graph_smoothness else None)
                bases = fold.bases
                if args.csp_init:
                    # Fitted on the fold's TRAINING windows only, in the node
                    # ordering the model sees, so it leaks no more than the graph.
                    from models.csp_prior import csp_basis
                    bases = [csp_basis(fold.train.X, fold.train.y,
                                       order=gbp_cfg.order)] + list(fold.bases[1:])
                model = GraphBandPowerNet(bases, args.window, gbp_cfg,
                                          laplacian=lap)
            else:
                # Shape comes from the fold, not from the config, so a windowing
                # change cannot silently desynchronise the model from the data.
                _, n_ch, n_tp = fold.train.X.shape
                model = build_baseline(args.model, n_ch, n_tp, n_classes,
                                       sfreq=ep.sfreq)
            trainer = Trainer(model, train_cfg, n_classes=n_classes)

            history = trainer.fit(
                fold, checkpoint_path=(
                    f"checkpoints/{args.tag or 'untagged'}/"
                    f"{subject}_fold{split.fold}.pt" if args.checkpoints else None))
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
                **({"window_start": args.window_start} if args.window_start else {}),
                "temporal_attention": args.temporal_attention,
                "attention_heads": args.attention_heads if args.temporal_attention else None,
                "model": args.model,
                "spatial": args.spatial if args.model == "graphbp" else None,
                "graph_smoothness": (args.graph_smoothness
                                     if args.model == "graphbp" else None),
                "csp_init": args.csp_init if args.model == "graphbp" else None,
                "gbp_layers": args.gbp_layers if args.model == "graphbp" else None,
                "pool_nodes": args.pool_nodes if args.model == "graphbp" else None,
                "learnable_graph": args.learnable_graph,
                "free_spatial": args.free_spatial,
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

        if fold_trial_acc:
            per_subject.append(float(np.mean(fold_trial_acc)))
            per_subject_tp.append(float(np.mean(fold_tp_acc)))
            print(f"{subject}: trial {per_subject[-1]:.4f}  "
                  f"timepoint {per_subject_tp[-1]:.4f}\n")
        if stopped_early:
            break

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
