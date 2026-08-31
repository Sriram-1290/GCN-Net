"""Classical baseline: Common Spatial Patterns + LDA, on identical splits.

    python scripts/run_csp_baseline.py --config configs/bandpass_8_30.yaml

**This is the experiment that calibrates every other number in the project.**
The GCN reaches 0.5418 within-family under a correct protocol
(project_documentation.md 9.4). Whether that is a weak *model* or a hard *task*
is not answerable from deep-learning results alone. CSP+LDA is the standard motor
imagery baseline and has been for two decades; running it on the same subjects,
the same folds and the same metric settles it.

Two binary problems are run, matching the within-family decomposition:

* left fist vs right fist   (classes 0/1, from runs 4/8/12)
* both fists vs both feet   (classes 2/3, from runs 6/10/14)

These are the confound-free comparisons — both classes in each problem come from
the *same* run family, so the run-family artefact cannot contribute (7.1). Their
mean is directly comparable to the network's within-family accuracy.

CSP is fit on the training trials of each fold only. It is a supervised spatial
filter, so fitting it on all data would leak exactly as the reference's adjacency
does.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import load_subject  # noqa: E402
from data.config import all_subjects, load_config  # noqa: E402
from data.splits import stratified_trial_kfold  # noqa: E402
from evaluation.experiment_log import log_experiment, set_seed  # noqa: E402
from evaluation.metrics import evaluate  # noqa: E402

# (name, class_a, class_b) — both drawn from one run family, so no family confound
PROBLEMS = [
    ("left_vs_right_fist", 0, 1),
    ("both_fists_vs_feet", 2, 3),
]


def csp_lda_accuracy(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int,
                     n_components: int) -> tuple[float, np.ndarray]:
    """Cross-validated CSP+LDA accuracy on one binary problem.

    Args:
        X: ``(n_trials, n_channels, n_samples)`` for the two classes only.
        y: ``(n_trials,)`` with two distinct values.

    Returns:
        ``(accuracy, confusion_matrix)``.
    """
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"expected exactly two classes, got {classes}")
    binary = (y == classes[1]).astype(np.int64)

    preds = np.zeros_like(binary)
    for split in stratified_trial_kfold(binary, n_splits=n_splits, seed=seed):
        # CSP is supervised: fit on training trials only, exactly as the graph is.
        csp = CSP(n_components=n_components, reg="ledoit_wolf", log=True,
                  norm_trace=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_train = csp.fit_transform(X[split.train].astype(np.float64),
                                        binary[split.train])
            f_test = csp.transform(X[split.test].astype(np.float64))
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            clf.fit(f_train, binary[split.train])
            preds[split.test] = clf.predict(f_test)

    result = evaluate(binary, preds, 2)
    return result.accuracy, result.confusion


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/bandpass_8_30.yaml",
                    help="CSP needs band-limited input; the 8-30 Hz cache is the "
                         "right default and matches the network's arm D")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--components", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log", default="runs/experiments.jsonl")
    ap.add_argument("--tag", default="P2-csp-lda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    seed_info = set_seed(seed)
    subjects = args.subjects or all_subjects(cfg)

    print("CSP + LDA classical baseline")
    print(f"cache {cfg.dataset.cache_dir}  bandpass {cfg.preprocess.bandpass}")
    print(f"subjects {len(subjects)}  folds {args.folds}  "
          f"components {args.components}  seed {seed}\n")

    per_problem: dict[str, list[float]] = {name: [] for name, _, _ in PROBLEMS}
    within: list[float] = []

    for i, subject in enumerate(subjects, 1):
        ep = load_subject(subject, cfg.dataset.cache_dir)
        accs = {}
        for name, ca, cb in PROBLEMS:
            mask = np.isin(ep.y, [ca, cb])
            acc, cm = csp_lda_accuracy(ep.X[mask], ep.y[mask], args.folds, seed,
                                       args.components)
            accs[name] = acc
            per_problem[name].append(acc)

            log_experiment({
                "experiment": "csp_lda_baseline",
                "tag": args.tag,
                "subject": subject,
                "problem": name,
                "protocol": "trial_level",
                "split_type": "stratified_trial_kfold",
                "n_folds": args.folds,
                "n_components": args.components,
                **seed_info,
                "config_source": str(cfg.source_path),
                "preprocess": {"bandpass": cfg.preprocess.bandpass,
                               "rereference": cfg.preprocess.rereference},
                "n_trials": int(mask.sum()),
                "accuracy": acc,
                "confusion": cm,
            }, path=args.log)

        subject_within = float(np.mean(list(accs.values())))
        within.append(subject_within)
        print(f"[{i:3d}/{len(subjects)}] {subject}: "
              + "  ".join(f"{n.split('_')[0]}-{n.split('_')[-1]} {a:.4f}"
                          for n, a in accs.items())
              + f"  | within-family {subject_within:.4f}", flush=True)

    print("\n" + "=" * 68)
    for name, _, _ in PROBLEMS:
        v = np.array(per_problem[name])
        print(f"{name:<22} {v.mean():.4f} +/- {v.std(ddof=1):.4f}  (chance 0.5000)")
    w = np.array(within)

    # Bootstrap CI so this is comparable to the network's reported interval.
    rng = np.random.default_rng(0)
    boot = w[rng.integers(0, len(w), size=(10_000, len(w)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nwithin-family accuracy {w.mean():.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"subjects above chance: {int((w > 0.5).sum())}/{len(w)}")
    print("\ncompare: GCNs-Net arm D reached 0.5418, CI [0.5050, 0.5817] "
          "(20 subjects, project_documentation.md 9.4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
