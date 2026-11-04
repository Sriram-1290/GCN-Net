"""Helper script to construct scripts/build_presentation_nip.py cleanly."""

with open("scripts/build_presentation.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Output file path and header comments
code = code.replace('output_path = "docs/PW-1_Review-1_Presentation.pptx"', 'output_path = "docs/PW-1_Review_Presentation-NIP.pptx"')
code = code.replace('Building PW-1 Review-1 Presentation (.pptx)', 'Building PW-1 Review-1 Presentation - NIP (.pptx)')

# 2. Slide 2 Abstract outcomes and speaker notes
code = code.replace(
    'GBP-Net resolves historical GCN failure modes and demonstrates +5.57% pre-training gain (73.01% vs 67.44%), beating all classical and CNN baselines.',
    'GBP-Net is theoretically designed to resolve historical GCN failure modes, surpass classical CSP and deep CNN baselines under leak-free protocols, and relieve BCI illiteracy via cross-subject pre-training.'
)

code = code.replace(
    '("Leak-Free SOTA Performance: ", "Achieve 73.01% cross-subject accuracy with 92-subject pre-training (67.44% without pre-training), exceeding the 65.0% Vasques ceiling."),',
    '("Target Decoding Performance: ", "Formulated to surpass classical CSP and compact CNN baselines under leak-free protocols, targeting performance beyond the 65.0% Vasques ceiling."),'
)

code = code.replace(
    'A verified leak-free model achieving 73.01% accuracy (67.44% without pre-training), physiologically interpretable topographic scalp maps, and an open-source evaluation suite.',
    'A theoretically sound, leak-free graph model designed to surpass classical baselines, physiologically interpretable topographic scalp maps, and a reproducible evaluation suite for PW-2.'
)

# 3. Slide 8 G5 text and card 6
code = code.replace(
    'Addressed via GBP-Net Stage 1 group pre-training (+5.57% gain to 73.01%).',
    'Addressed via proposed GBP-Net Stage 1 cross-subject pre-training framework.'
)
code = code.replace(
    'These 5 gaps directly justify our approved title and the design of GBP-Net (+5.57% empirical gain to 73.01%).',
    'These 5 gaps directly justify our approved title and motivate the theoretical formulation of GBP-Net.'
)
code = code.replace(
    '"Project Goal: ", "Deliver GBP-Net achieving 73.01% pre-trained leak-free accuracy (67.44% control), surpassing the 65.0% Vasques ceiling."',
    '"Project Goal: ", "Formulate GBP-Net with cross-subject pre-training to conquer sample scarcity and surpass the 65.0% Vasques ceiling under strict trial-level CV."'
)

# 4. Slide 10 replacement: fig4 image and metric card
code = code.replace('fig4_leakage_audit.png', 'fig4_leakage_audit_nip.png')
code = code.replace('THE QUANTITATIVE REALITY GAP & PRE-TRAINING BREAKTHROUGH ON PHYSIONET:', 'CRITICAL TAKEAWAYS FROM LITERATURE AUDIT & UNBIASED BENCHMARKS:')

old_metrics_block = """    metrics_str = [
        ("Published GCNs-Net (Leaky Baseline): ", "93.06%"),
        ("Audited Clean Re-implementation: ", "50.90%"),
        ("Vasques et al. 2026 Empirical Ceiling: ", "65.00%"),
        ("GBP-Net Control (No Pre-Training, PTC): ", "67.44%"),
        ("GBP-Net Pre-Trained (92-Subj Grouped, PTG): ", "73.01% (+5.57% gain, 98/103 above chance)")
    ]"""

new_metrics_block = """    metrics_str = [
        ("Published Leaky Claims in Literature: ", "93.06% (GCNs-Net) / 94.20% (BiLSTM-GCN)"),
        ("Underlying Methodological Cause: ", "Timepoint shuffling + eval BatchNorm + transductive graphs"),
        ("True Empirical Literature Ceiling: ", "65.00% across 216,714 pipelines (Vasques et al., 2026)"),
        ("Classical Spatial Filtering Baseline: ", "62.39% (CSP+LDA) on continuous PhysioNet trials"),
        ("Proposed GBP-Net Design Target: ", "Conquer data scarcity via cross-subject pre-training without leakage")
    ]"""

code = code.replace(old_metrics_block, new_metrics_block)

old_s10_notes = '''Slide 10 highlights one of our most significant critical findings: The Data Leakage Audit.
In Panel A on the left, we dissect the 3 compounding leakage pathways in the official GCNs-Net code:
1. Timepoint Shuffling: Randomly shuffling 6.25ms timepoints leaks adjacent samples from the same continuous trial across splits.
2. Test-Time BatchNorm in Train Mode: Recalculates batch statistics over test data during inference.
3. Transductive Pearson Adjacency: Calculates graph connectivity across all trials at once.
In Panel B, we show the quantitative reality gap: The published 93.06% is an artifact of leakage. Under leak-free trial splits, GCNs-Net drops to 50.90%. Vasques et al. (2026) confirmed that the true within-session ceiling across 216,000 pipelines is 65.0%. Crucially, our GBP-Net architecture achieves 67.44% without pre-training (PTC) and reaches a breakthrough 73.01% with 92-subject cross-subject pre-training (PTG), delivering a +5.57% boost with 98 of 103 subjects performing above chance!'''

new_s10_notes = '''Slide 10 highlights one of the most vital contributions of our literature review: Our Methodological Data Leakage Audit.
In Figure 4 on the left, we deconstruct the three compounding leakage pathways identified in the reference codebase of GCNs-Net:
1. Inter-Trial Timepoint Shuffling: Randomly permuting 6.25ms timepoints destroys epoch boundaries, allowing models to interpolate test labels via continuous voltage autocorrelation.
2. Test-Time BatchNorm in Train Mode: Batch normalization recalibrates mean and variance over test mini-batches, leaking test-set class distributions.
3. Transductive Graph Estimation: Computing the Pearson correlation graph on pooled train and test sets pre-adapts spectral filters to test geometry.
As summarized in the lower card, published claims of >90% are artifacts of this leakage. Independent benchmarks (Vasques et al., 2026 across 216,714 pipelines) prove that the true literature ceiling is 65.0%. Our proposed GBP-Net is formulated specifically to eliminate these three leakage vectors while using cross-subject pre-training to break the 65% ceiling legitimately!'''

code = code.replace(old_s10_notes, new_s10_notes)

# 5. Slide 11: Remove 337 unit tests mention
code = code.replace('("Reproducibility: ", "Deterministic seed control, automated pytest test suite (337 passing unit tests), Git versioning.")',
                    '("Planned Reproducibility: ", "Deterministic seed control, modular pipeline architecture, and Git version control for PW-2.")')

# 6. Slide 12: Deliverables Card 1 & Card 4
old_d1 = """        (
            "1. Verified Leak-Free SOTA Architecture",
            NAVY,
            [
                ("Validated Pre-Training Breakthrough: ", "Deliver GBP-Net achieving 73.01% accuracy with 92-subject cross-subject pre-training (+5.57% over non-pretrained control 67.44%), with 98/103 subjects above chance."),
                ("Exceeding Empirical Ceilings: ", "Surpass the 65.00% Vasques et al. (2026) empirical ceiling and classical CSP+LDA (62.39%)."),
                ("Clinical Generalizability: ", "Demonstrate effective zero-calibration transfer to unseen subjects via population pre-training, lifting weak cohorts (+7.20% in lowest quartile).")
            ]
        ),"""

new_d1 = """        (
            "1. Anticipated Decoding Breakthrough",
            NAVY,
            [
                ("Resolving the Classical Deficit: ", "Proposed spatial-then-power sequence is hypothesized to surpass classical CSP and compact CNN baselines without data leakage."),
                ("Breaking the Literature Ceiling: ", "Targeting statistically significant gains over the 65.0% Vasques benchmark via 92-subject cross-subject pre-training."),
                ("Relieving BCI-Illiteracy: ", "Hypothesized to disproportionately uplift low-performing subjects and reduce inter-subject decoding variance.")
            ]
        ),"""

code = code.replace(old_d1, new_d1)

old_d4 = """        (
            "4. Academic Publications & Open Deliverables",
            GOLD,
            [
                ("Open-Source Benchmark Suite: ", "Complete modular PyTorch/PyG codebase with automated test suite (337 passing unit tests) and reproducible pipelines."),
                ("Publication-Ready Review Paper: ", "Review Paper-2 (14 pages, IEEE format) and 1-Page Review Analysis conforming strictly to NGIT college guidelines."),
                ("Project Work Milestone: ", "Comprehensive PW-1 report and readiness for Phase-2 experimental deployment and defense.")
            ]
        )"""

new_d4 = """        (
            "4. Project Work Phase-Wise Roadmap",
            GOLD,
            [
                ("Phase-1 (PW-1 Milestone): ", "Comprehensive 36-paper literature review, PRISMA synthesis, problem formulation, title defense, and GBP-Net architectural formulation."),
                ("Phase-2 (PW-2 Execution): ", "Full modular PyTorch/PyG implementation, cross-subject pre-training on 92 subjects, controlled ablation benchmarking, and confirmatory statistical tests."),
                ("Final Deliverables: ", "Comprehensive PW-2 Major Project dissertation, open-source benchmark repository, and peer-reviewed conference/journal submission.")
            ]
        )"""

code = code.replace(old_d4, new_d4)

# Slide 12 Speaker notes
old_s12_notes = """In conclusion, Slide 12 summarizes our expected outcomes and deliverables:
1. A verified leak-free architecture reaching 73.01% cross-subject accuracy with 92-subject pre-training (67.44% control), beating standard CNN and Riemannian baselines.
2. A definitive scientific answer on whether graph convolutions provide genuine inductive bias for EEG.
3. Neurophysiological interpretability via scalp topographic maps aligning with C3/Cz/C4 sensorimotor activations.
4. Open academic deliverables: an open-source PyTorch/PyG benchmark repository, 337 unit tests, Review-Paper-2, and complete PW-1 documentation.
Thank you, respected evaluators. We are now open for questions and feedback."""

new_s12_notes = """In conclusion, Slide 12 outlines our expected outcomes and project work execution roadmap:
1. Anticipated Decoding Breakthrough: Formulated to resolve the classical deficit without data leakage, break past the 65.0% literature ceiling, and alleviate BCI illiteracy.
2. Controlled Ablation Suite: Planned for PW-2 to definitively isolate whether spectral graph convolutions provide genuine inductive bias compared to unconstrained spatial filters.
3. Neurophysiological Interpretability: Topographical scalp maps confirming activation over C3, Cz, and C4 sensorimotor regions.
4. Clear Phase-Wise Roadmap: Phase-1 (PW-1) successfully establishes the problem statement, literature taxonomy, research gaps, title defense, and architectural design; Phase-2 (PW-2) will execute the full implementation, training, and empirical validation.
Thank you, respected panel members. We are now open for your questions and suggestions."""

code = code.replace(old_s12_notes, new_s12_notes)

with open("scripts/build_presentation_nip.py", "w", encoding="utf-8") as f:
    f.write(code)

print("scripts/build_presentation_nip.py written successfully!")
