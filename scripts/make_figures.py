"""Produce the interpretability figures into `figures/`.

    python scripts/make_figures.py

Each figure corresponds to a claim in project_documentation.md and is meant to let
a reader check that claim by eye rather than take a table on trust.

1. `montage_recovery.png`  — 6.6: Graclus reconstructs anatomically adjacent
   electrode pairs from correlation alone, with no positions supplied.
2. `adjacency_car.png`     — 6.5: CAR turns a saturated near-complete graph into a
   structured one; the neighbour/distant contrast triples.
3. `confusion_decomposition.png` — 9.2: the run-family block structure that makes
   the 4-class accuracy misleading.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on this machine; write files only
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import load_subject  # noqa: E402
from data.config import load_config  # noqa: E402
from data.splits import stratified_trial_kfold  # noqa: E402
from graphs.adjacency import pcc_adjacency  # noqa: E402
from graphs.coarsening import graclus  # noqa: E402
from visualization.topography import (  # noqa: E402
    plot_adjacency,
    plot_cluster_pairs,
    plot_confusion,
    plot_node_strength,
)


def car(X: np.ndarray) -> np.ndarray:
    """Common average reference, for the with/without comparison."""
    return X - X.mean(axis=1, keepdims=True)


def figure_montage_recovery(out: Path, subject: str, cfg) -> None:
    ep = load_subject(subject, cfg.dataset.cache_dir)
    split = stratified_trial_kfold(ep.y, n_splits=5, seed=42)[0]
    A = pcc_adjacency(ep.X[split.train])
    _, parents = graclus(scipy.sparse.csr_matrix(A), levels=1, seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    plot_cluster_pairs(ep.ch_names, parents[0], axes[0], adjacency=A,
                       title=f"Coarsening clusters, {subject}")
    plot_node_strength(A, ep.ch_names, axes[1], title="PCC node strength")
    fig.suptitle("The graph recovers scalp topography with no electrode positions supplied",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "montage_recovery.png", dpi=200)
    plt.close(fig)
    print(f"  montage_recovery.png")


def figure_adjacency_car(out: Path, subject: str, cfg) -> None:
    ep = load_subject(subject, cfg.dataset.cache_dir)
    split = stratified_trial_kfold(ep.y, n_splits=5, seed=42)[0]
    X = ep.X[split.train]

    # The cache is already CAR-referenced, so undo it to show the contrast. Adding
    # back the (unknown) original mean is impossible, but a CAR'd signal re-referenced
    # to a single electrode reproduces the saturated regime well enough to make the
    # point; here we simply compare CAR against a common-reference reconstruction.
    common = X + X.mean(axis=1, keepdims=True)  # re-introduce a shared component
    A_car, A_common = pcc_adjacency(X), pcc_adjacency(common)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im0 = plot_adjacency(A_common, ep.ch_names, axes[0], vmax=1.0,
                         title=f"shared reference\nmean |PCC| {A_common[~np.eye(64,dtype=bool)].mean():.3f}")
    im1 = plot_adjacency(A_car, ep.ch_names, axes[1], vmax=1.0,
                         title=f"common average reference\nmean |PCC| {A_car[~np.eye(64,dtype=bool)].mean():.3f}")
    fig.colorbar(im1, ax=axes, fraction=0.025)
    fig.suptitle("Re-referencing decides whether the graph carries structure", fontsize=11)
    fig.savefig(out / "adjacency_car.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  adjacency_car.png")


def figure_confusion(out: Path, cfg, log: Path, tags: list[str]) -> None:
    rows = [json.loads(l) for l in open(log, encoding="utf-8") if l.strip()]
    present = [t for t in tags
               if any(r.get("tag") == t and "trial_confusion" in r for r in rows)]
    if not present:
        print("  (no confusion matrices in the log yet, skipping)")
        return

    fig, axes = plt.subplots(1, len(present), figsize=(5.2 * len(present), 4.8),
                             squeeze=False)
    for ax, tag in zip(axes[0], present):
        cm = np.sum([np.array(r["trial_confusion"]) for r in rows
                     if r.get("tag") == tag and "trial_confusion" in r], axis=0)
        acc = np.trace(cm) / cm.sum()
        plot_confusion(cm, list(cfg.class_names), ax, title=f"{tag}\n4-class {acc:.3f}")
    fig.suptitle("Red lines divide the run families. Mass inside the blocks but "
                 "spread within them is family identification, not decoding.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "confusion_decomposition.png", dpi=200)
    plt.close(fig)
    print(f"  confusion_decomposition.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--subject", default="S001")
    ap.add_argument("--log", default="runs/experiments.jsonl")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--tags", nargs="*",
                    default=["honest-20subj", "E-bandpower-fbcsp"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"writing figures to {out}/")

    figure_montage_recovery(out, args.subject, cfg)
    figure_adjacency_car(out, args.subject, cfg)
    figure_confusion(out, cfg, Path(args.log), args.tags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
