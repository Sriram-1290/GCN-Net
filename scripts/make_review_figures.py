"""Figures for the review paper, drawn from the 27 surveyed studies.

Everything here is original and derived from our own survey table -- the review
guidelines prohibit reusing figures from published papers, so the sample review's
bibliometric maps are deliberately not imitated. No figure invents data: the
taxonomy and year counts come straight from LITERATURE_SURVEY.md, and the two
schematics are conceptual diagrams of pipelines the text describes.

    python scripts/make_review_figures.py      # writes figures/review/*.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures" / "review"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.linewidth": .6})
INK, SOFT, ACCENT, FILL = "#1b1b1f", "#6b6b73", "#8a2d2d", "#f2eeee"

# Category -> (short label, [(author, year), ...]) -- the review's classification.
CATS = [
    ("Graph convolutional\nnetworks", [("Hou", 2024), ("Hou", 2022), ("Deng", 2025),
                                       ("Meng", 2026), ("Defferrard", 2016), ("Dhillon", 2007)]),
    ("Classical spatial\nfiltering", [("Ramoser", 2000), ("Blankertz", 2008), ("Ang", 2008)]),
    ("Graph-free deep\nnetworks", [("Lawhern", 2018), ("Schirrmeister", 2017), ("Mane", 2021)]),
    ("Riemannian and\nlearned spatial priors", [("Kobler", 2022), ("Jiang", 2024),
                                                ("Eder", 2024)]),
    ("Cross-subject transfer\nand augmentation", [("Xie", 2023), ("He", 2020),
                                                          ("Wu", 2025), ("Rommel", 2022),
                                                          ("Jiang", 2024)]),
    ("Benchmarks and\nevaluation methodology", [("Kapoor", 2023), ("Chevallier", 2024),
                                                ("Vasques", 2026)]),
    ("Neurophysiology\nand datasets", [("Pfurtscheller", 1999), ("Blankertz", 2010),
                                       ("Schalk", 2004), ("Goldberger", 2000)]),
]
assert sum(len(p) for _, p in CATS) == 27, "the survey reviewed 27 studies"


def box(ax, x, y, w, h, text, *, fc="white", ec=INK, lw=.7, weight="normal", size=8.2,
        color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            weight=weight, color=color, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color=INK):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=.8, color=color, mutation_scale=9))


# ------------------------------------------------------------- Fig 1: taxonomy
def fig_taxonomy():
    """Root on the left, categories stacked, studies to the right.

    A single row of seven category boxes cannot hold labels like "Cross-subject
    transfer, alignment, augmentation" at a readable size in a 7-inch figure, so the
    tree runs sideways, where width is plentiful.
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(-.01, 1.01); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0, .40, .17, .20, "Reviewed studies\non EEG motor\nimagery decoding\n(n = 27)",
        fc=FILL, ec=ACCENT, weight="bold", size=8)
    n = len(CATS); h = .105; step = (1 - n * h) / (n + 1)
    for i, (label, papers) in enumerate(CATS):
        y = 1 - step - (i + 1) * h - i * step
        arrow(ax, .17, .50, .25, y + h / 2)
        box(ax, .25, y, .38, h, f"{label.replace(chr(10), ' ')}  ({len(papers)})",
            weight="bold", size=6.9)
        ax.text(.655, y + h / 2, ",  ".join(f"{a} {yr}" for a, yr in papers),
                va="center", fontsize=6.8, color=SOFT)
    fig.savefig(OUT / "fig1_taxonomy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------- Fig 2: years by category
def fig_years():
    bins = [("≤2009", 0, 2009), ("2010–15", 2010, 2015), ("2016–19", 2016, 2019),
            ("2020–22", 2020, 2022), ("2023–26", 2023, 2026)]
    greys = ["#8a2d2d", "#b86b5f", "#d8a99a", "#5b5b66", "#8c8c96", "#b9b9c2", "#e1e1e6"]
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    bottom = [0] * len(bins)
    for (label, papers), c in zip(CATS, greys):
        counts = [sum(lo <= y <= hi for _, y in papers) for _, lo, hi in bins]
        ax.bar([b[0] for b in bins], counts, bottom=bottom, color=c, edgecolor="white",
               lw=.6, label=label.replace("\n", " "), width=.62)
        bottom = [b + c_ for b, c_ in zip(bottom, counts)]
    for i, t in enumerate(bottom):
        ax.text(i, t + .25, str(t), ha="center", fontsize=8, color=INK)
    ax.set_ylabel("Number of studies")
    ax.set_ylim(0, max(bottom) + 2.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.4, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.savefig(OUT / "fig2_years.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------- Fig 3: where band power is taken, schematic
def fig_ordering():
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.set_xlim(-.01, 1.01); ax.set_ylim(0, 1); ax.axis("off")
    rows = [
        (.70, "(a) Power first, then spatial", ["Multichannel\nEEG", "Band power\nper electrode",
          "Graph / spatial\nmixing", "Classifier"], 1,
         "Sign and phase between electrodes are lost before channels are combined;\n"
         "typical of graph models that use PSD or differential-entropy node features"),
        (.20, "(b) Spatial first, then power", ["Multichannel\nEEG", "Temporal\nfilter",
          "Spatial\ncombination", "Square ·\nmean · log", "Classifier"], 2,
         "Channels are combined while still signed, so opposing sources can cancel;\n"
         "the ordering used by CSP, FBCSP, ShallowConvNet, FBCNet — and GBP-Net"),
    ]
    for y, title, stages, hi, note in rows:
        ax.text(0, y + .19, title, fontsize=8.6, weight="bold", color=INK)
        # Fit n boxes and n-1 gaps into the 0.70 left portion; a fixed box width
        # overflowed on the five-stage row and the boxes overlapped.
        n = len(stages); gap = .025; w = (0.70 - (n - 1) * gap) / n
        for i, s in enumerate(stages):
            x = i * (w + gap)
            emph = (i == hi)
            box(ax, x, y - .06, w, .16, s, fc=FILL if emph else "white",
                ec=ACCENT if emph else INK, lw=1.0 if emph else .7, size=7.2,
                weight="bold" if emph else "normal")
            if i < n - 1:
                arrow(ax, x + w, y + .02, x + w + gap, y + .02)
        ax.text(.73, y + .02, note, fontsize=6.9, color=SOFT, va="center", linespacing=1.35)
    fig.savefig(OUT / "fig3_ordering.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------- Fig 4: proposed GBP-Net
def fig_gbpnet():
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(-.01, 1.01); ax.set_ylim(0, 1); ax.axis("off")
    # stage 1: pre-training
    ax.text(0, .95, "Stage 1 — cross-subject pre-training", fontsize=8.4, weight="bold")
    box(ax, 0, .66, .20, .2, "Source subjects\n(all groups except\nthe target's own)", size=7)
    box(ax, .27, .66, .46, .2, "GBP-Net trained on pooled source trials\n"
        "graph and normalisation fitted on source data only", fc=FILL, ec=ACCENT, size=7.2)
    box(ax, .80, .66, .20, .2, "Pre-trained\nweights", size=7.2)
    arrow(ax, .20, .76, .27, .76); arrow(ax, .73, .76, .80, .76)
    # stage 2: architecture
    ax.text(0, .52, "Stage 2 — per-subject fine-tuning (architecture)", fontsize=8.4,
            weight="bold")
    stages = ["EEG window\n(64 electrodes)", "Learned temporal\nfilter bank",
              "Graph convolution\nacross electrodes", "Square · mean\nover time · log",
              "Linear\nclassifier"]
    w = .165; gap = (1 - 5 * w) / 4
    for i, s in enumerate(stages):
        x = i * (w + gap)
        box(ax, x, .18, w, .22, s, fc=FILL if i in (2, 3) else "white",
            ec=ACCENT if i in (2, 3) else INK, size=7.1)
        if i < 4:
            arrow(ax, x + w, .29, x + w + gap, .29)
    arrow(ax, .90, .66, .59, .40, color=ACCENT)
    ax.text(.5, .04, "Target subject's own training folds only · evaluated on held-out trials "
            "of that subject", ha="center", fontsize=7, color=SOFT)
    fig.savefig(OUT / "fig4_gbpnet.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_taxonomy(); fig_years(); fig_ordering(); fig_gbpnet()
    for p in sorted(OUT.glob("*.png")):
        print(f"wrote {p.relative_to(OUT.parent.parent)}")
