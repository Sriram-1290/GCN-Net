"""Figures that put the graph back on the scalp.

The point of every plot here is to make a claim checkable by eye. The montage
figure in particular carries a real result: Graclus coarsening reconstructs
anatomically adjacent electrode pairs from signal correlation alone, with no
electrode positions supplied anywhere in the pipeline
(project_documentation.md 6.6). A reader can verify that by looking, which no
table of numbers achieves.

All functions take an ``ax`` and draw into it, so callers compose multi-panel
figures without this module owning any layout decisions.
"""

from __future__ import annotations

import numpy as np

_MONTAGE = "standard_1005"


def electrode_positions(ch_names: list[str]) -> np.ndarray:
    """2-D scalp positions for our channel names, from a standard montage.

    Returns ``(n_channels, 2)`` in the montage's own units. The dataset's dotted
    Sharbrough labels (``Fc5.``, ``C3..``) must already have been normalised by
    ``eegbci.standardize`` — every resulting name is present in ``standard_1005``.
    """
    import mne

    pos = mne.channels.make_standard_montage(_MONTAGE).get_positions()["ch_pos"]
    lookup = {k.lower(): v for k, v in pos.items()}
    missing = [c for c in ch_names if c.lower() not in lookup]
    if missing:
        raise ValueError(f"channels absent from {_MONTAGE}: {missing}")
    # Drop z; a top-down view is what makes adjacency legible.
    return np.array([lookup[c.lower()][:2] for c in ch_names])


def _draw_head(ax, radius: float) -> None:
    """Head outline, nose and ears, so the orientation is unambiguous."""
    import matplotlib.patches as patches

    ax.add_patch(patches.Circle((0, 0), radius, fill=False, lw=1.2, color="0.35"))
    ax.plot([-radius * .12, 0, radius * .12],
            [radius * .99, radius * 1.13, radius * .99], color="0.35", lw=1.2)
    for side in (-1, 1):
        ax.add_patch(patches.Ellipse((side * radius, 0), radius * .12, radius * .3,
                                     fill=False, lw=1.2, color="0.35"))
    ax.set_aspect("equal")
    ax.axis("off")


def plot_cluster_pairs(ch_names: list[str], parents: np.ndarray, ax,
                       adjacency: np.ndarray | None = None,
                       title: str = "Graclus level-0 clusters") -> None:
    """Draw each level-0 coarsening cluster as a line between its two electrodes.

    If the clusters are anatomically sensible the lines are short and local. If
    the coarsening were arbitrary they would criss-cross the scalp — which is
    exactly the failure this figure is meant to expose or rule out.
    """
    xy = electrode_positions(ch_names)
    radius = np.abs(xy).max() * 1.15
    _draw_head(ax, radius)

    lengths = []
    for cluster in np.unique(parents):
        members = np.flatnonzero(parents == cluster)
        if len(members) != 2:
            continue
        a, b = members
        w = adjacency[a, b] if adjacency is not None else 1.0
        ax.plot(xy[[a, b], 0], xy[[a, b], 1], "-", lw=0.8 + 2.4 * float(w),
                color="#3B44A8", alpha=0.75, zorder=1)
        lengths.append(float(np.linalg.norm(xy[a] - xy[b])))

    ax.scatter(xy[:, 0], xy[:, 1], s=16, color="0.2", zorder=2)
    for name, (x, y) in zip(ch_names, xy):
        ax.annotate(name, (x, y), fontsize=4.5, ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points", color="0.35")

    if lengths:
        # A random pairing of 64 electrodes averages far longer edges; quoting the
        # mean makes "short and local" a number rather than an impression.
        ax.set_title(f"{title}\nmean edge {np.mean(lengths):.3f} "
                     f"(random pairing ~{_expected_random_length(xy):.3f})",
                     fontsize=9)
    else:
        ax.set_title(title, fontsize=9)


def _expected_random_length(xy: np.ndarray, n_draws: int = 20_000,
                            seed: int = 0) -> float:
    """Mean distance between two electrodes picked at random — the null baseline."""
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(xy), n_draws)
    j = rng.integers(0, len(xy), n_draws)
    keep = i != j
    return float(np.linalg.norm(xy[i[keep]] - xy[j[keep]], axis=1).mean())


def plot_adjacency(A: np.ndarray, ch_names: list[str], ax, title: str = "",
                   vmax: float | None = None) -> None:
    """Adjacency as a heatmap, with sparse tick labels so it stays readable."""
    im = ax.imshow(A, cmap="magma", vmin=0, vmax=vmax or float(A.max()),
                   interpolation="nearest")
    step = max(1, len(ch_names) // 16)
    ticks = np.arange(0, len(ch_names), step)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([ch_names[i] for i in ticks], fontsize=5, rotation=90)
    ax.set_yticklabels([ch_names[i] for i in ticks], fontsize=5)
    ax.set_title(title, fontsize=9)
    return im


def plot_adjacency_difference(learned: np.ndarray, prior: np.ndarray,
                              ch_names: list[str], ax,
                              title: str = "learned - PCC prior") -> None:
    """Where learning moved the graph away from the correlation prior.

    Diverging around zero because the sign is the whole point: red edges were
    strengthened by training, blue ones weakened.
    """
    d = learned - prior
    lim = float(np.abs(d).max()) or 1.0
    im = ax.imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    step = max(1, len(ch_names) // 16)
    ticks = np.arange(0, len(ch_names), step)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([ch_names[i] for i in ticks], fontsize=5, rotation=90)
    ax.set_yticklabels([ch_names[i] for i in ticks], fontsize=5)
    ax.set_title(f"{title}\nmax |change| {lim:.3f}", fontsize=9)
    return im


def plot_node_strength(A: np.ndarray, ch_names: list[str], ax,
                       title: str = "node strength") -> None:
    """Total edge weight per electrode, drawn on the scalp.

    For a learned graph this shows which electrodes the model came to rely on —
    the sensorimotor strip is the region to look for.
    """
    xy = electrode_positions(ch_names)
    radius = np.abs(xy).max() * 1.15
    _draw_head(ax, radius)
    strength = A.sum(axis=1)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=strength, s=90, cmap="viridis",
                    edgecolors="0.25", linewidths=0.4, zorder=2)
    for name, (x, y) in zip(ch_names, xy):
        ax.annotate(name, (x, y), fontsize=4.5, ha="center", va="center",
                    color="white", zorder=3)
    ax.set_title(title, fontsize=9)
    return sc


def plot_attention(weights: np.ndarray, ax, sfreq: float = 160.0,
                   title: str = "temporal attention") -> None:
    """Mean attention over time, with a spread band across samples.

    Args:
        weights: ``(n_samples, n_time, n_heads)`` from
            :meth:`models.temporal_attention.TemporalAttention.last_attention`.
    """
    if weights.ndim != 3:
        raise ValueError(f"expected (samples, time, heads), got {weights.shape}")
    t = np.arange(weights.shape[1]) / sfreq
    for h in range(weights.shape[2]):
        w = weights[:, :, h]
        mean, sd = w.mean(axis=0), w.std(axis=0)
        ax.plot(t, mean, lw=1.4, label=f"head {h}")
        ax.fill_between(t, mean - sd, mean + sd, alpha=0.15)
    ax.axhline(1.0 / weights.shape[1], ls="--", lw=0.8, color="0.5",
               label="uniform")
    ax.set_xlabel("time within window (s)", fontsize=8)
    ax.set_ylabel("attention weight", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, frameon=False)


def plot_confusion(cm: np.ndarray, class_names: list[str], ax,
                   title: str = "confusion", normalize: bool = True) -> None:
    """Confusion matrix with the run-family block structure made visible.

    The dividing lines separate classes 0/1 (runs 4/8/12) from 2/3 (runs 6/10/14).
    Mass concentrated inside the blocks but spread within them is the signature of
    a model decoding run family rather than motor imagery (9.2).
    """
    m = cm.astype(float)
    if normalize:
        m = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=7, rotation=30, ha="right")
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("predicted", fontsize=8); ax.set_ylabel("true", fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.2f}" if normalize else f"{int(m[i, j])}",
                    ha="center", va="center", fontsize=7,
                    color="white" if m[i, j] > 0.5 else "0.2")
    if len(class_names) == 4:
        ax.axhline(1.5, color="#A63D34", lw=1.2)
        ax.axvline(1.5, color="#A63D34", lw=1.2)
    ax.set_title(title, fontsize=9)
    return im
