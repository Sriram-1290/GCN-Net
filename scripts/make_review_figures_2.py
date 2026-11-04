"""Figures for Review Paper 2, drawn from 36 surveyed studies.

Generates:
  figures/review_2/fig1_taxonomy.png  - Systematic taxonomy of 36 studies across 7 categories
  figures/review_2/fig2_years.png     - Chronological distribution across categories (2000-2026)
  figures/review_2/fig3_ordering.png  - Operational ordering: power-then-spatial vs spatial-then-power
  figures/review_2/fig4_gbpnet.png    - Proposed GBP-Net: cross-subject pre-training & spatial-then-power graph pipeline

Run:
  python scripts/make_review_figures_2.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "figures" / "review_2"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.6
})
INK, SOFT, ACCENT, FILL, HIGHLIGHT = "#1b1b1f", "#55555c", "#8a2d2d", "#f4f1f1", "#2d5a8a"

# 36 studies grouped into 7 categories
CATS_2 = [
    ("Graph convolutional\nnetworks", [
        ("Hou", 2024), ("Hou", 2022), ("Song", 2020), ("Zhong", 2020),
        ("Musallam", 2021), ("Sun", 2021), ("Defferrard", 2016), ("Dhillon", 2007),
        ("Deng", 2025), ("Meng", 2026)
    ]),
    ("Classical spatial\nfiltering", [
        ("Ramoser", 2000), ("Blankertz", 2008), ("Ang", 2008)
    ]),
    ("Graph-free deep\nnetworks", [
        ("Lawhern", 2018), ("Schirrmeister", 2017), ("Mane", 2021)
    ]),
    ("Riemannian and\nlearned spatial priors", [
        ("Kobler", 2022), ("Jiang", 2024), ("Eder", 2024)
    ]),
    ("Cross-subject transfer\nand augmentation", [
        ("Xie", 2023), ("He", 2020), ("Wu", 2025), ("Rommel", 2022),
        ("Wu", 2022), ("Jiang", 2024)
    ]),
    ("Benchmarks and\nevaluation methodology", [
        ("Kapoor", 2023), ("Chevallier", 2024), ("Vasques", 2026),
        ("Klepl", 2024), ("Saibene", 2024), ("Page", 2021)
    ]),
    ("Neurophysiology,\ndatasets and tools", [
        ("Pfurtscheller", 1999), ("Blankertz", 2010), ("Schalk", 2004),
        ("Goldberger", 2000), ("Jia", 2021)
    ]),
]
TOTAL_PAPERS = sum(len(p) for _, p in CATS_2)
assert TOTAL_PAPERS == 36, f"Total papers count is {TOTAL_PAPERS}"


def box(ax, x, y, w, h, text, *, fc="white", ec=INK, lw=0.7, weight="normal", size=8.2, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            weight=weight, color=color, linespacing=1.22)


def arrow(ax, x0, y0, x1, y1, color=INK):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=color, mutation_scale=9))


def fig_taxonomy():
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 1)
    ax.axis("off")
    
    box(ax, 0, 0.38, 0.17, 0.24, "Reviewed studies\non EEG motor\nimagery decoding\n(PRISMA-guided,\n n = 36)",
        fc=FILL, ec=ACCENT, weight="bold", size=8)
    
    n = len(CATS_2)
    h = 0.108
    step = (1 - n * h) / (n + 1)
    
    for i, (label, papers) in enumerate(CATS_2):
        y = 1 - step - (i + 1) * h - i * step
        arrow(ax, 0.17, 0.50, 0.25, y + h / 2)
        box(ax, 0.25, y, 0.36, h, f"{label.replace(chr(10), ' ')} ({len(papers)})",
            weight="bold", size=6.9)
        # Format list of papers
        paper_str = ", ".join(f"{a} {yr}" for a, yr in papers)
        ax.text(0.63, y + h / 2, paper_str, va="center", fontsize=6.3, color=SOFT,
                bbox=dict(boxstyle="square,pad=0", fc="none", ec="none"))
        
    fig.savefig(OUT / "fig1_taxonomy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_years():
    bins = [("≤2009", 0, 2009), ("2010–15", 2010, 2015), ("2016–19", 2016, 2019),
            ("2020–22", 2020, 2022), ("2023–26", 2023, 2026)]
    colors = ["#8a2d2d", "#b86b5f", "#d8a99a", "#5b5b66", "#8c8c96", "#b9b9c2", "#3b5c7d"]
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    bottom = [0] * len(bins)
    for (label, papers), c in zip(CATS_2, colors):
        counts = [sum(lo <= y <= hi for _, y in papers) for _, lo, hi in bins]
        ax.bar([b[0] for b in bins], counts, bottom=bottom, color=c, edgecolor="white",
               lw=0.6, label=label.replace("\n", " "), width=0.62)
        bottom = [b + c_ for b, c_ in zip(bottom, counts)]
    for i, t in enumerate(bottom):
        ax.text(i, t + 0.3, str(t), ha="center", fontsize=8, color=INK)
    ax.set_ylabel("Number of studies")
    ax.set_ylim(0, max(bottom) + 3.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.2, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.savefig(OUT / "fig2_years.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_ordering():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 2.6))
    for ax in (ax1, ax2):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 2)
        ax.axis("off")

    # (a) Power first, then spatial
    ax1.text(0, 1.7, "(a) Inverted ordering: Power first, then spatial combination (loses phase & destructive cancellation)",
             fontsize=7.8, weight="bold", color=ACCENT)
    box(ax1, 0.0, 0.2, 1.8, 1.2, "Multichannel EEG\n$X \\in \\mathbb{R}^{C \\times T}$", size=7.2)
    arrow(ax1, 1.8, 0.8, 2.4, 0.8)
    box(ax1, 2.4, 0.2, 2.1, 1.2, "Band Power\nper Electrode\n(PSD / DE / Hilbert)", size=7.2, fc=FILL, ec=ACCENT)
    arrow(ax1, 4.5, 0.8, 5.1, 0.8)
    box(ax1, 5.1, 0.2, 2.1, 1.2, "Graph / Spatial\nMixing\n($\\tilde{A} X_{pow}$)", size=7.2)
    arrow(ax1, 7.2, 0.8, 7.8, 0.8)
    box(ax1, 7.8, 0.2, 1.8, 1.2, "Classifier\n(Softmax / MLP)", size=7.2)

    # (b) Spatial first, then power
    ax2.text(0, 1.7, "(b) Principled ordering: Spatial combination first, then band power (preserves signed cancellation)",
             fontsize=7.8, weight="bold", color=HIGHLIGHT)
    box(ax2, 0.0, 0.2, 1.6, 1.2, "Multichannel EEG\n$X \\in \\mathbb{R}^{C \\times T}$", size=7.2)
    arrow(ax2, 1.6, 0.8, 2.1, 0.8)
    box(ax2, 2.1, 0.2, 1.8, 1.2, "Learned Temporal\nFilter Bank\n($1 \\times K_t$ conv)", size=7.2)
    arrow(ax2, 3.9, 0.8, 4.4, 0.8)
    box(ax2, 4.4, 0.2, 2.0, 1.2, "Spatial Graph\nCombination\n(Signed Chebyshev)", size=7.2, fc="#eef4fa", ec=HIGHLIGHT)
    arrow(ax2, 6.4, 0.8, 6.9, 0.8)
    box(ax2, 6.9, 0.2, 1.6, 1.2, "Band Power\n$\\log(\\mathrm{mean}(x^2))$", size=7.2, fc="#eef4fa", ec=HIGHLIGHT)
    arrow(ax2, 8.5, 0.8, 8.9, 0.8)
    box(ax2, 8.9, 0.2, 1.1, 1.2, "Linear\nClassifier", size=7.2)

    fig.tight_layout()
    fig.savefig(OUT / "fig3_ordering.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_gbpnet():
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    # Title & Stage 1
    ax.text(0, 3.9, "Stage 1 — Cross-Subject Pre-Training (Conquering Data Scarcity)", fontsize=8.2, weight="bold", color=INK)
    box(ax, 0.2, 2.6, 2.5, 1.05, "Source Subjects\n(All groups except target's;\n10-fold split)", size=7.0, fc=FILL)
    arrow(ax, 2.7, 3.12, 3.3, 3.12)
    box(ax, 3.3, 2.6, 3.5, 1.05, "GBP-Net Pre-Training\n(Graph & standardizer fitted strictly\non pooled source data)", size=7.0, fc="#eef4fa", ec=HIGHLIGHT)
    arrow(ax, 6.8, 3.12, 7.5, 3.12)
    box(ax, 7.5, 2.6, 2.1, 1.05, "Pre-Trained\nBase Weights", size=7.2, weight="bold", fc=FILL, ec=ACCENT)

    # Transition Arrow
    arrow(ax, 8.55, 2.6, 8.55, 1.85, color=ACCENT)
    ax.text(8.7, 2.2, "Transfer", fontsize=7.0, color=ACCENT, weight="bold")

    # Stage 2
    ax.text(0, 1.85, "Stage 2 — Target Subject Fine-Tuning (Leak-Free Evaluation Architecture)", fontsize=8.2, weight="bold", color=INK)
    box(ax, 0.2, 0.3, 1.5, 1.15, "Target EEG Window\n(64 channels, 0.4s)", size=6.8)
    arrow(ax, 1.7, 0.87, 2.1, 0.87)
    box(ax, 2.1, 0.3, 1.7, 1.15, "Learned Temporal\nFilter Bank\n(40 filters)", size=6.8)
    arrow(ax, 3.8, 0.87, 4.2, 0.87)
    box(ax, 4.2, 0.3, 1.9, 1.15, "Chebyshev Spectral\nGraph Convolution\n(Order K=2, 1 layer)", size=6.8, fc="#eef4fa", ec=HIGHLIGHT, weight="bold")
    arrow(ax, 6.1, 0.87, 6.5, 0.87)
    box(ax, 6.5, 0.3, 1.7, 1.15, "Band Power\n$\\log(\\mathrm{mean}(x^2))$", size=6.8, fc="#eef4fa", ec=HIGHLIGHT)
    arrow(ax, 8.2, 0.87, 8.6, 0.87)
    box(ax, 8.6, 0.3, 1.2, 1.15, "Linear\nClassifier", size=6.8)

    ax.text(5.0, 0.05, "Target subject's own training trials only · Evaluated on held-out trials under stratified trial split",
            ha="center", fontsize=6.8, color=SOFT, style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "fig4_gbpnet.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig5_gbpnet.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_leakage_audit():
    fig = plt.figure(figsize=(7.6, 3.5))
    
    # Left subplot: Schematic of 3 leakage paths
    ax1 = fig.add_axes([0.01, 0.05, 0.47, 0.88])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    
    ax1.text(0.1, 9.4, "(a) Three Compounding Leakage Paths in GCNs-Net (SuperBruceJia/EEG-DL)",
             fontsize=7.8, weight="bold", color=ACCENT)
    
    # Path 1
    box(ax1, 0.1, 6.4, 9.7, 2.6,
        "1. Timepoint Shuffling Across Trials (randperm)\n"
        "• Sample unit defined as 1 isolated timepoint (6.25 ms, C=64, T=1)\n"
        "• Random permutation splits pooled timepoints across all trials\n"
        "• Result: Adjacent timepoints from same trial split into train & test\n"
        "  -> Model memorizes trial DC offset & phase (99.8% train acc)",
        fc=FILL, ec=ACCENT, size=6.3)
    
    # Path 2
    box(ax1, 0.1, 3.5, 9.7, 2.6,
        "2. Test-Time BatchNorm in Training Mode\n"
        "• In b2relu and fc blocks: BatchNorm hardcoded to training=True during eval\n"
        "• Test samples normalized using instantaneous test-batch statistics (μ, σ)\n"
        "• Result: Model peeks into collective test-batch distributions",
        fc=FILL, ec=ACCENT, size=6.3)
    
    # Path 3
    box(ax1, 0.1, 0.6, 9.7, 2.6,
        "3. Transductive Full-Dataset Graph Construction\n"
        "• Pearson correlation matrix A = |P| - I computed on entire dataset\n"
        "• Rescaled Laplacian L̃ fitted on pooled train, val, and test trials\n"
        "• Result: Test-set covariance structure leaks directly into spectral filters",
        fc=FILL, ec=ACCENT, size=6.3)

    # Right subplot: Empirical Accuracy comparison
    ax2 = fig.add_axes([0.64, 0.10, 0.34, 0.80])
    
    models = [
        "BiLSTM-GCN (Leaky)",
        "GCNs-Net (Leaky)",
        "GCNs-Net (Honest Audit)",
        "CSP Baseline (Linear)",
        "EEGNet (Compact CNN)",
        "Tangent Space (Vasques)",
        "ShallowConvNet (CNN)",
        "GBP-Net (Control)",
        "GBP-Net (Pre-Trained)"
    ]
    
    accs = [94.20, 93.06, 50.90, 62.39, 63.00, 65.00, 67.25, 67.44, 73.01]
    colors = [ACCENT, ACCENT, "#888890", "#55555c", "#55555c", "#3b5c7d", "#3b5c7d", HIGHLIGHT, "#166534"]
    
    y_pos = range(len(models))
    bars = ax2.barh(y_pos, accs, color=colors, height=0.62, edgecolor="none")
    
    # Chance level dashed line
    ax2.axvline(50.0, color="#b03030", linestyle="--", linewidth=0.8, alpha=0.75)
    ax2.text(50.0, 8.85, "Chance: 50%", fontsize=6.0, color="#b03030", ha="center", va="bottom", weight="bold")
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(models, fontsize=6.6)
    ax2.set_xlabel("Accuracy on PhysioNet EEGMMIDB (%)", fontsize=7.2)
    ax2.set_xlim(35, 102)
    ax2.set_ylim(-0.6, 9.1)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="x", labelsize=6.6)
    
    # Annotate bar values
    for bar, acc, model in zip(bars, accs, models):
        w = bar.get_width()
        if "Pre-Trained" in model:
            va_col = "#166534"
            fw = "bold"
        elif "Control" in model:
            va_col = HIGHLIGHT
            fw = "bold"
        elif acc in (50.90, 93.06, 94.20):
            va_col = ACCENT if acc > 90 else INK
            fw = "bold"
        else:
            va_col = INK
            fw = "normal"
        ax2.text(w + 1.2, bar.get_y() + bar.get_height()/2, f"{acc:.1f}%",
                 va="center", fontsize=6.3, color=va_col, weight=fw)
    
    ax2.set_title("(b) 40+ % Reality Gap on Benchmark", fontsize=7.8, weight="bold", color=INK, pad=10)
    
    fig.savefig(OUT / "fig4_leakage_audit.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig5_leakage_audit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_leakage_audit_nip():
    fig = plt.figure(figsize=(7.6, 3.6))
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.92])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(5.0, 9.6, "Methodological Audit: Three Compounding Data Leakage Pathways in EEG-GNN Literature",
            fontsize=8.5, weight="bold", color=ACCENT, ha="center")
    ax.text(5.0, 9.15, "Systematic deconstruction of validation pitfalls identified in reference codebase (SuperBruceJia/EEG-DL; Hou et al., 2024)",
            fontsize=7.0, style="italic", color=SOFT, ha="center")

    # Column 1: Timepoint Shuffling
    box(ax, 0.1, 0.2, 3.1, 8.6,
        "Pathway 1: Inter-Trial Timepoint Shuffling\n\n"
        "• Formulation Flaw:\n"
        "  Dataset partitioned at the isolated timepoint\n"
        "  level (6.25 ms frames, C=64, T=1) using\n"
        "  global random permutation (randperm).\n\n"
        "• Consequence:\n"
        "  Adjacent timepoints (t and t+1) from the\n"
        "  same continuous 4-second motor imagery\n"
        "  trial split across training and test folds.\n\n"
        "• Failure Mechanism:\n"
        "  EEG exhibits strong continuous temporal\n"
        "  autocorrelation and trial-specific DC offsets.\n"
        "  The network interpolates unseen test labels\n"
        "  via memorization rather than learning true\n"
        "  sensorimotor rhythm dynamics.\n\n"
        "• Rigorous Reformulation:\n"
        "  Strict trial-level (epoch-level) stratified\n"
        "  splitting with continuous temporal windows.",
        fc=FILL, ec=ACCENT, size=6.2)

    # Column 2: Test-Time BatchNorm
    box(ax, 3.45, 0.2, 3.1, 8.6,
        "Pathway 2: Test-Time BatchNorm in Training Mode\n\n"
        "• Formulation Flaw:\n"
        "  Normalization layers (b2relu, fc blocks)\n"
        "  hardcoded to training=True during evaluation;\n"
        "  running statistics tracking disabled.\n\n"
        "• Consequence:\n"
        "  Inference mini-batches are standardized using\n"
        "  instantaneous test-batch statistics:\n"
        "  μ_test = mean(X_test), σ_test = std(X_test)\n"
        "  instead of frozen training statistics (μ_train, σ_train).\n\n"
        "• Failure Mechanism:\n"
        "  The model peeks into collective test-set\n"
        "  class distributions during inference, artificially\n"
        "  aligning test representations and masking\n"
        "  domain shift and covariate drift.\n\n"
        "• Rigorous Reformulation:\n"
        "  Strict eval mode; batch statistics frozen\n"
        "  from training data; or Domain-Specific BN (DSBN).",
        fc=FILL, ec=ACCENT, size=6.2)

    # Column 3: Transductive Graph
    box(ax, 6.8, 0.2, 3.1, 8.6,
        "Pathway 3: Transductive Graph Estimation\n\n"
        "• Formulation Flaw:\n"
        "  Electrode connectivity matrix A = |Corr(X)| - I\n"
        "  and rescaled Laplacian L̃ computed across the\n"
        "  entire pooled dataset before train/test split.\n\n"
        "• Consequence:\n"
        "  Test-trial covariance structures leak directly\n"
        "  into Chebyshev spectral polynomial filters:\n"
        "  g_θ(L̃) = ∑ θ_k T_k(L̃).\n\n"
        "• Failure Mechanism:\n"
        "  Spectral filters are pre-adapted to the geometric\n"
        "  relationships of test data, violating inductive\n"
        "  generalization to unseen recordings.\n\n"
        "• Rigorous Reformulation:\n"
        "  Adjacency and Laplacian estimated strictly on\n"
        "  training trials, or computed dynamically per\n"
        "  individual trial window without leakage.",
        fc=FILL, ec=ACCENT, size=6.2)

    fig.savefig(OUT / "fig4_leakage_audit_nip.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_taxonomy()
    fig_years()
    fig_ordering()
    fig_gbpnet()
    fig_leakage_audit()
    fig_leakage_audit_nip()
    print("Review Paper 2 figures successfully created in", OUT)
