"""Train/validation/test splits.

Three split families live here, and the difference between them is the central
methodological point of this project:

* :func:`stratified_trial_kfold` — subject-dependent, split at **trial** level.
  Correct: no timepoint from a test trial is ever seen in training.
* :func:`loso_folds` — subject-independent. Correct: a held-out subject is
  entirely unseen, and the validation subjects are disjoint from both.
* :func:`leaky_timepoint_holdout` — **deliberately wrong**, reproducing the
  reference protocol (shuffle every timepoint, then cut 90/10). It exists so the
  leakage can be *measured* rather than asserted. It must never be used for a
  reported result; see project_documentation.md 7.2.

Splits are returned as index arrays. Nothing here touches the feature data, so a
split can be logged, replayed and audited independently of the arrays it indexes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class TrialSplit:
    """Indices into a trial axis of length ``n_trials``."""

    train: np.ndarray
    test: np.ndarray
    fold: int

    def __post_init__(self) -> None:
        overlap = np.intersect1d(self.train, self.test)
        if overlap.size:
            raise ValueError(f"fold {self.fold}: {overlap.size} trials in both splits")


@dataclass(frozen=True)
class SubjectSplit:
    """Subject IDs for one LOSO fold."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    fold: int

    def __post_init__(self) -> None:
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = set(getattr(self, a)) & set(getattr(self, b))
            if shared:
                raise ValueError(f"fold {self.fold}: {sorted(shared)} in both {a} and {b}")


def stratified_trial_kfold(y: np.ndarray, n_splits: int = 5,
                           seed: int = 42) -> list[TrialSplit]:
    """Stratified k-fold over trials, for the subject-dependent protocol.

    Splitting at trial level is the minimum correct granularity: GCNs-Net treats
    every timepoint as a sample, so a shuffled split over samples would place
    timepoints from the same trial on both sides (see leaky_timepoint_holdout).
    """
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D trial labels, got shape {y.shape}")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        TrialSplit(train=tr.astype(np.int64), test=te.astype(np.int64), fold=i)
        for i, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y))
    ]


def loso_folds(subjects: list[str], n_val_subjects: int = 5,
               seed: int = 42) -> list[SubjectSplit]:
    """Leave-One-Subject-Out folds, with a disjoint validation set per fold.

    One fold per subject. ``n_val_subjects`` are drawn from the remaining
    subjects for early stopping / best-checkpoint selection, so model selection
    never sees the test subject. Drawing validation subjects rather than
    validation *trials* keeps the whole fold subject-independent.
    """
    if n_val_subjects < 1:
        raise ValueError("n_val_subjects must be >= 1")
    if n_val_subjects >= len(subjects) - 1:
        raise ValueError(
            f"n_val_subjects={n_val_subjects} leaves no training subjects "
            f"out of {len(subjects)}"
        )

    subjects = sorted(subjects)
    folds: list[SubjectSplit] = []
    for i, test_subject in enumerate(subjects):
        pool = [s for s in subjects if s != test_subject]
        # Per-fold seed so the validation draw is reproducible but not identical
        # across folds — identical val subjects everywhere would bias selection
        # toward whatever those particular subjects reward.
        rng = np.random.default_rng(seed + i)
        val = tuple(sorted(rng.choice(pool, size=n_val_subjects, replace=False).tolist()))
        train = tuple(s for s in pool if s not in set(val))
        folds.append(SubjectSplit(train=train, val=val, test=(test_subject,), fold=i))
    return folds


def leaky_timepoint_holdout(n_trials: int, n_samples: int, test_frac: float = 0.1,
                            seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the reference's leaky split. **Never use for a reported result.**

    The reference flattens every trial into ``n_samples`` independent rows, calls
    ``randperm`` over all of them, and cuts 90/10. Timepoints from the same trial
    therefore land on both sides, and since neighbouring timepoints of one trial
    are near-identical, the test set is effectively visible during training.

    Returns index pairs into a flattened ``(n_trials * n_samples,)`` axis, where
    index ``t * n_samples + s`` is timepoint ``s`` of trial ``t``.
    """
    total = n_trials * n_samples
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)
    n_train = int(total * (1.0 - test_frac))
    return perm[:n_train], perm[n_train:]


def trial_of(flat_index: np.ndarray, n_samples: int) -> np.ndarray:
    """Trial number owning each flattened timepoint index. Used to audit leakage."""
    return flat_index // n_samples


def count_shared_trials(train_idx: np.ndarray, test_idx: np.ndarray,
                        n_samples: int) -> int:
    """How many trials contribute timepoints to *both* splits.

    Zero for a correct split. Near ``n_trials`` for the reference protocol — this
    is the number that makes the leakage concrete in the writeup.
    """
    return len(
        np.intersect1d(trial_of(train_idx, n_samples), trial_of(test_idx, n_samples))
    )
