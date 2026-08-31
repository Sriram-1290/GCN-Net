"""Evaluation metrics, including the timepoint -> trial aggregation.

The model classifies **one timepoint at a time**, following the reference. A trial
is 640 timepoints, so a trial-level prediction is an aggregate over 640 votes.
Both numbers matter and they mean different things:

* **Timepoint accuracy** is what the reference reports. It is what the model
  actually optimises, but it treats 640 near-identical samples as independent, so
  it is not a per-decision accuracy in any BCI sense.
* **Trial accuracy** is the number a BCI would care about — one decision per
  imagined movement — and it is the one that can be compared across papers that
  use different sample definitions.

CLAUDE.md requires confusion matrices and per-class accuracy alongside aggregate
numbers, so those are returned by default rather than being opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationResult:
    """Metrics for one evaluation pass."""

    accuracy: float
    per_class_accuracy: tuple[float, ...]
    confusion: np.ndarray  # (n_classes, n_classes), rows = true, cols = predicted
    macro_f1: float
    kappa: float
    n_samples: int

    def summary(self, class_names: tuple[str, ...] | None = None) -> str:
        names = class_names or tuple(f"c{i}" for i in range(len(self.per_class_accuracy)))
        lines = [
            f"accuracy {self.accuracy:.4f}  macro-F1 {self.macro_f1:.4f}  "
            f"kappa {self.kappa:.4f}  (n={self.n_samples})",
            "per-class: " + "  ".join(
                f"{n}={a:.3f}" for n, a in zip(names, self.per_class_accuracy)),
            "confusion (rows=true):",
        ]
        for name, row in zip(names, self.confusion):
            lines.append(f"  {name:>12} " + " ".join(f"{v:5d}" for v in row))
        return "\n".join(lines)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true.astype(np.int64), y_pred.astype(np.int64)), 1)
    return cm


def cohen_kappa(cm: np.ndarray) -> float:
    """Chance-corrected agreement. Reported by the paper, so we report it too."""
    n = cm.sum()
    if n == 0:
        return float("nan")
    observed = np.trace(cm) / n
    expected = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (n * n)
    if np.isclose(expected, 1.0):
        return float("nan")
    return float((observed - expected) / (1.0 - expected))


def macro_f1(cm: np.ndarray) -> float:
    """Unweighted mean F1. Robust to the mild class imbalance in this dataset."""
    scores = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> ClassificationResult:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    support = cm.sum(axis=1)
    per_class = tuple(
        float(cm[c, c] / support[c]) if support[c] else float("nan") for c in range(n_classes)
    )
    total = cm.sum()
    return ClassificationResult(
        accuracy=float(np.trace(cm) / total) if total else float("nan"),
        per_class_accuracy=per_class,
        confusion=cm,
        macro_f1=macro_f1(cm),
        kappa=cohen_kappa(cm),
        n_samples=int(total),
    )


def aggregate_to_trials(logits: np.ndarray, trial_id: np.ndarray,
                        method: str = "mean_logit") -> tuple[np.ndarray, np.ndarray]:
    """Collapse per-timepoint predictions into one prediction per trial.

    Args:
        logits: ``(n_timepoints, n_classes)``.
        trial_id: ``(n_timepoints,)`` — which trial each timepoint belongs to.
        method: ``"mean_logit"`` averages the logits (soft vote, the default —
            it uses the model's confidence); ``"majority"`` counts argmax votes.

    Returns:
        ``(trial_ids_sorted, predicted_class_per_trial)``.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected (n_timepoints, n_classes), got {logits.shape}")
    if len(trial_id) != len(logits):
        raise ValueError(f"{len(trial_id)} trial ids for {len(logits)} timepoints")

    trials = np.unique(trial_id)
    index = {t: i for i, t in enumerate(trials)}
    n_classes = logits.shape[1]
    scores = np.zeros((len(trials), n_classes), dtype=np.float64)
    rows = np.array([index[t] for t in trial_id])

    if method == "mean_logit":
        np.add.at(scores, rows, logits)
        counts = np.bincount(rows, minlength=len(trials))[:, None]
        scores /= np.maximum(counts, 1)
    elif method == "majority":
        np.add.at(scores, (rows, logits.argmax(axis=1)), 1.0)
    else:
        raise ValueError(f"unknown aggregation method {method!r}")

    return trials, scores.argmax(axis=1)


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000,
                        seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap CI for the paired difference ``a - b`` across subjects/folds.

    CLAUDE.md requires confidence intervals and paired tests across subjects, not
    just means. Pair on subject, since subject variance dwarfs model differences.

    Returns:
        ``(mean_difference, ci_low, ci_high)`` at 95 %.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must match: {a.shape} vs {b.shape}")
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
