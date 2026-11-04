"""Score a fine-tuned model on every test-time montage condition (spec sections 4, 4.1).

Both models are trained on the full 64-channel cap. After each fine-tuning fold,
the fold's held-out trials are presented again under each condition -- a named
reduced montage, or random / clustered electrode losses -- and the model is
scored without any further training. What each model is given is fixed by
spec section 4.1 and is its best fair treatment:

* **graph** (M1, ``graphbp`` with the region readout): the kept electrodes,
  re-referenced to their own average (all a real reduced cap could record), then
  standardised by the subset standardiser, and the model's graph swapped for the
  subset's graph. Graph and standardiser come from the **source** subjects'
  no-reference moments re-referenced over the same subset
  (:func:`graphs.subset.subset_representation`), so no target data enters either.
* **shallow** (M2, ShallowConvNet): the missing electrodes are reconstructed by
  spherical-spline interpolation (:func:`data.interpolate.repair_full_cap`), which
  restores the 64 inputs the network needs, then the fold's full 64-channel source
  standardiser is applied, exactly as on the clean path.

Pairing: the channel patterns come from :func:`data.montage.available_masks`,
seeded by ``(subject, fold, draw)`` only, so M1 and M2 are scored on identical
patterns.

The one silent failure this module guards against (Review Focus 3): the region
model's basis and region map are persistent buffers. A subset graph left in place
after the loop would make every later score -- the Full condition, the next fold,
a saved state -- silently wrong. The full graph is therefore restored in a
``finally`` block, so it is back even when a condition raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from data.interpolate import interpolation_matrix, repair_full_cap
from data.montage import Condition, available_masks, common_average
from data.normalize import Standardizer
from evaluation.experiment_log import completed_keys
from evaluation.pipeline import Trials, to_window_samples
from graphs.subset import Moments, SubsetRepresentation, subset_representation


def condition_tag(tag: str, condition: str) -> str:
    """Log tag of a condition's rows: ``"<tag>.<condition>"``."""
    return f"{tag}.{condition}"


def evaluate_conditions(trainer, kind: Literal["graph", "shallow"], test_noref: Trials,
                        trial_offset: int, ch_names: list[str],
                        src_moments: Moments | None, full_standardizer: Standardizer,
                        full_rep: SubsetRepresentation | None, subject: str, fold: int,
                        conditions: tuple[Condition, ...], window: int, stride: int,
                        lmax: float = 2.0) -> dict[str, dict]:
    """Accuracy of ``trainer.model`` on each condition, mean over its draws.

    Args:
        trainer: the fine-tuned :class:`~evaluation.trainer.Trainer`.
        kind: ``"graph"`` (region-readout ``graphbp``) or ``"shallow"``.
        test_noref: the fold's test trials from the **no-reference** cache, in the
            same order and with the same labels as the referenced test trials.
        trial_offset: first trial id given to the windows.
        ch_names: the 64 channel names, in the data's channel order.
        src_moments: no-reference moments of the source TRAINING subjects, the
            ones ``full_rep`` was built from. Required for ``graph``, unused for
            ``shallow``.
        full_standardizer: the fold's 64-channel source standardiser (``shallow``).
        full_rep: the model's full-montage representation; required for ``graph``
            and restored after the loop.
        conditions: the conditions to score.
        window, stride: windowing, as on the clean path.
        lmax: the fixed lambda_max of the Chebyshev basis, as for ``full_rep``.

    Returns:
        ``{condition name: {"accuracy", "per_draw", "n_draws", "timepoint_accuracy",
        "confusion", "per_class", "kappa", "macro_f1"}}``. ``accuracy`` is trial
        accuracy averaged over the condition's draws (spec section 4), ``confusion``
        is summed over draws, and the other metrics are draw means.
    """
    if kind not in ("graph", "shallow"):
        raise ValueError(f"kind must be 'graph' or 'shallow', got {kind!r}")
    X_all = np.asarray(test_noref.X)
    if X_all.ndim != 3 or X_all.shape[1] != len(ch_names):
        raise ValueError(f"test_noref.X must be (trials, {len(ch_names)}, time); "
                         f"got {X_all.shape}")

    model = trainer.model
    if kind == "graph":
        if full_rep is None or src_moments is None:
            raise ValueError("kind='graph' needs full_rep and src_moments")
        K = model.spatial[0].K
        # The subset graphs are derived from src_moments, so those must be the moments
        # the model's own graph was built from; anything else (another group, target
        # data) would score the model on a representation it never had.
        check = subset_representation(src_moments, np.ones(len(ch_names), dtype=bool),
                                      ch_names, K=K, lmax=lmax)
        if not (np.array_equal(check.basis, full_rep.basis)
                and np.array_equal(check.regions, full_rep.regions)):
            raise ValueError("src_moments do not reproduce full_rep's graph; pass the "
                             "moments the model's full graph was built from")

    scores: dict[str, dict] = {}
    try:
        for cond in conditions:
            results = []
            for keep in available_masks(cond, ch_names, subject, fold):
                if kind == "graph":
                    rep = subset_representation(src_moments, keep, ch_names, K=K, lmax=lmax)
                    X = rep.standardizer.apply(common_average(X_all, keep))
                    model.set_graph(rep.basis, rep.regions)
                else:
                    # W is rebuilt from THIS keep and THIS channel order every time:
                    # repair_full_cap validates W by shape only, so a W from another
                    # mask of the same size would be accepted and silently wrong.
                    W = interpolation_matrix(ch_names, keep) if not keep.all() else None
                    X = full_standardizer.apply(repair_full_cap(X_all, keep, W))
                samples = to_window_samples(X, test_noref.y, window, stride,
                                            trial_offset=trial_offset)
                results.append(trainer.evaluate_samples(samples))
            scores[cond.name] = _summarise(results)
    finally:
        if kind == "graph":
            model.set_graph(full_rep.basis, full_rep.regions)
    return scores


def _summarise(results: list[dict]) -> dict:
    trial = [r["trial"] for r in results]
    per_draw = [float(t.accuracy) for t in trial]
    return {
        "accuracy": float(np.mean(per_draw)),
        "per_draw": per_draw,
        "n_draws": len(per_draw),
        "timepoint_accuracy": float(np.mean([r["timepoint"].accuracy for r in results])),
        "confusion": np.sum([t.confusion for t in trial], axis=0).tolist(),
        "per_class": np.mean([t.per_class_accuracy for t in trial], axis=0).tolist(),
        "kappa": float(np.mean([t.kappa for t in trial])),
        "macro_f1": float(np.mean([t.macro_f1 for t in trial])),
    }


def condition_row(clean_row: dict, condition: str, score: dict,
                  provenance: dict | None = None) -> dict:
    """The log row of one condition: every field of the clean row, metrics replaced.

    The fold's identity, configuration and fine-tuning history are the clean row's,
    because it is the same fine-tuned model. Every metric is replaced by the
    condition's, so no clean number can be read off a condition row by mistake.

    ``provenance`` is added to the condition row only -- the runner passes the
    no-reference cache it read (config path and manifest fingerprint), which every
    condition is computed from but the clean row, for ShallowConvNet, never names.
    """
    row = dict(clean_row)
    row.update(provenance or {})
    row.update({
        "tag": condition_tag(clean_row["tag"], condition),
        "condition": condition,
        "accuracy": score["accuracy"],
        "trial_accuracy": score["accuracy"],
        "timepoint_accuracy": score["timepoint_accuracy"],
        "trial_confusion": score["confusion"],
        "trial_per_class": score["per_class"],
        "trial_kappa": score["kappa"],
        "trial_macro_f1": score["macro_f1"],
        "n_draws": score["n_draws"],
        "per_draw": score["per_draw"],
    })
    return row


def completed_with_conditions(path: Path | str, tag: str,
                              condition_names) -> set[tuple[str, int | None]]:
    """``(subject, fold)`` logged under ``tag`` AND under every condition tag.

    Review Focus 4: a power cut between the clean row and the condition rows must
    leave the fold *not done*, so it is re-run rather than skipped with conditions
    missing. (The runner also logs the clean row last, so this is belt and braces.)
    """
    done = completed_keys(path, tag)
    for name in condition_names:
        done &= completed_keys(path, condition_tag(tag, name))
    return done
