"""Riemannian tangent-space baseline (A3), on the same folds as every other arm.

**Why this exists.** 13 records that CSP was left untuned, which makes H4's negative
result conservative. [23]'s benchmark of 216,714 configurations on PhysionetMI is
more specific than that: **covariance tangent-space projection (cov-tgsp) and CSP
are the two leading families**, nearly tied at 0.605 and 0.604 in 8-30 Hz, and the
best single pipeline overall was cov-tgsp with a RobustScaler and a linear SVM at
0.650. Coherence-based and nonlinear descriptors ranked substantially lower.

We have CSP. We have never had cov-tgsp. It is the single strongest family on this
dataset that is missing from our portfolio, and at ~1 s/fold it costs minutes.

**Implemented here rather than with `pyriemann`** because `CLAUDE.md` says not to
add dependencies casually and the whole method is ~40 lines of `scipy.linalg`.

**Leakage.** The covariance reference, the scaler and the classifier are all fitted
on training trials only, exactly as CSP is. A tangent-space projection whose
reference point saw the test trials would leak the test distribution into every
feature.

**On dimensionality.** 64 channels gives tangent vectors of 64*65/2 = 2080
dimensions against ~36 training trials per fold. That is hostile, and the
regularisation is doing most of the work -- shrinkage on the covariances and a
strongly penalised linear classifier. This is a fair-conditions baseline, not a
tuned one; if it underperforms, that is a statement about the protocol's sample
size and not evidence against the family.
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


def _covariances(X: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    """``(N, C, T) -> (N, C, C)`` shrunk sample covariances.

    Shrinkage toward a scaled identity keeps every matrix positive definite, which
    the matrix logarithm below requires. With T comparable to C the raw sample
    covariance is often singular, so this is not optional.
    """
    n, c, _ = X.shape
    out = np.empty((n, c, c))
    for i in range(n):
        s = np.cov(X[i].astype(np.float64))
        trace = np.trace(s) / c
        out[i] = (1 - shrinkage) * s + shrinkage * trace * np.eye(c)
    return out


def _sqrtm_inv(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-10, None)
    return (V * np.sqrt(w)) @ V.T, (V * (1.0 / np.sqrt(w))) @ V.T


def _logm_spd(C: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    return (V * np.log(np.clip(w, 1e-10, None))) @ V.T


def _expm_spd(C: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    return (V * np.exp(w)) @ V.T


def riemannian_mean(covs: np.ndarray, n_iter: int = 15, tol: float = 1e-6) -> np.ndarray:
    """Frechet mean under the affine-invariant metric, by the standard fixed point.

    Starts from the arithmetic mean and repeatedly steps along the average tangent
    direction. Converges in a handful of iterations for well-conditioned inputs.
    """
    M = covs.mean(axis=0)
    for _ in range(n_iter):
        Ms, Mis = _sqrtm_inv(M)
        T = np.mean([_logm_spd(Mis @ C @ Mis) for C in covs], axis=0)
        M = Ms @ _expm_spd(T) @ Ms
        if np.linalg.norm(T) < tol:
            break
    return M


def tangent_space(covs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Project SPD matrices to the tangent space at ``ref`` and vectorise.

    Off-diagonal entries are scaled by sqrt(2) so the Euclidean norm of the vector
    equals the Frobenius norm of the matrix -- the standard convention, without
    which the diagonal is implicitly down-weighted.
    """
    _, Ris = _sqrtm_inv(ref)
    c = ref.shape[0]
    iu = np.triu_indices(c)
    scale = np.full((c, c), np.sqrt(2.0))
    np.fill_diagonal(scale, 1.0)
    return np.stack([(_logm_spd(Ris @ C @ Ris) * scale)[iu] for C in covs])


def tangent_accuracy(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> float:
    """Cross-validated cov-tgsp accuracy on one binary problem."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler

    accs = []
    for split in stratified_trial_kfold(y, n_splits=n_splits, seed=seed):
        ctr = _covariances(X[split.train])
        cte = _covariances(X[split.test])
        # Reference fitted on TRAINING covariances only.
        ref = riemannian_mean(ctr)
        ztr, zte = tangent_space(ctr, ref), tangent_space(cte, ref)

        scaler = RobustScaler().fit(ztr)          # [23]'s best pipeline used this
        clf = LogisticRegression(C=0.01, max_iter=2000)   # 2080 dims, ~36 trials
        clf.fit(scaler.transform(ztr), y[split.train])
        accs.append(float((clf.predict(scaler.transform(zte)) == y[split.test]).mean()))
    return float(np.mean(accs)), accs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/bandpass_8_30.yaml")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", default="runs/experiments.jsonl")
    ap.add_argument("--tag", default="A3-tgsp")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    cfg = load_config(args.config)
    set_seed(args.seed)
    subjects = args.subjects or all_subjects(cfg)
    print(f"Riemannian tangent space + logistic regression, {len(subjects)} subjects")

    per_problem = {}
    for classes in ([0, 1], [2, 3]):
        means = []
        for s in subjects:
            ep = load_subject(s, cfg.dataset.cache_dir)
            mask = np.isin(ep.y, classes)
            X, y = ep.X[mask], (ep.y[mask] == classes[1]).astype(np.int64)
            acc, fold_accs = tangent_accuracy(X, y, args.folds, args.seed)
            means.append(acc)
            for f, a in enumerate(fold_accs):
                log_experiment({
                    "experiment": "tangent_space", "tag": f"{args.tag}-{classes[0]}-{classes[1]}",
                    "subject": s, "fold": f, "n_folds": args.folds,
                    "protocol": "subject_dependent", "split_type": "stratified_trial_kfold",
                    "seed": args.seed, "classes": classes, "n_classes": 2,
                    "config_source": args.config, "model": "cov-tgsp",
                    "accuracy": a, "trial_accuracy": a,
                }, args.log)
            print(f"  {s} {classes}: {acc:.4f}", flush=True)
        per_problem[tuple(classes)] = float(np.mean(means))
        print(f"{classes}: {per_problem[tuple(classes)]:.4f}")

    wf = float(np.mean(list(per_problem.values())))
    print(f"\nwithin-family: {wf:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
