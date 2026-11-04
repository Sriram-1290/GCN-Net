"""Refine build_presentation_nip.py to fix all visual, mathematical, and NIP phrasing defects."""

def main():
    with open("scripts/build_presentation_nip.py", "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Slide 2
    text = text.replace(
        '("Data Leakage Reality: ", "Our audit reveals 3 compounding leakage paths (timepoint shuffling, test-mode BatchNorm). Under genuine leak-free trials, accuracy collapses to 50.9% (near chance)."),',
        '("Data Leakage Reality: ", "Our audit reveals 3 compounding leakage paths (timepoint shuffling, test-mode BatchNorm). Under genuine leak-free trials, published models fail to generalize."),'
    )
    text = text.replace(
        '("Resolving the Reality Gap: ", "Bridge the 42-point literature gap between reported 93% and actual 50.9% chance performance on PhysioNet."),',
        '("Resolving the Reality Gap: ", "Bridge the gap between published leaky claims (>93%) and legitimate literature ceilings (65.0% Vasques benchmark; 62.4% CSP)."),'
    )
    text = text.replace(
        '1. Problem Statement: GCNs-Net claims 93% accuracy on PhysioNet, but processes 6.25ms timepoints in isolation. When audited, timepoint shuffling leaked information across train and test splits. On clean trials, it drops to 50.9%.',
        '1. Problem Statement: GCNs-Net claims 93% accuracy on PhysioNet, but processes 6.25ms timepoints in isolation. When audited, timepoint shuffling leaked information across train and test splits. Under genuine continuous trials, published models fail to generalize.'
    )

    # 2. Slide 3
    text = text.replace(
        '("The 42-Point Collapse: ", "When audited on strictly separated trials, accuracy collapses from 93.06% down to 50.90% (near chance)."),',
        '("The Literature Reality Gap: ", "When evaluated on continuous trials with zero timepoint leakage, published models fail, while extensive benchmarks confirm the true clean ceiling is 65.0% (Vasques et al., 2026)."),'
    )
    text = text.replace(
        'Third: The Data Leakage Crisis. Our thorough audit of the published codebase (SuperBruceJia/EEG-DL) revealed that published 93% accuracies rely on shuffling 6.25ms timepoints. When evaluated properly on continuous trials, accuracy drops by 42 points to 50.9%.',
        'Third: The Data Leakage Crisis. Our thorough audit of the published codebase (SuperBruceJia/EEG-DL) revealed that published 93% accuracies rely on shuffling 6.25ms timepoints. When evaluated on clean continuous trials, published claims prove unreplicable, falling short of classical literature baselines.'
    )

    # 3. Slide 4
    text = text.replace(
        '("Faithful Reproduction: ", "Implement GCNs-Net exactly on PhysioNet EEGMMIDB (64 channels, 109 subjects) to validate the baseline pipeline."),',
        '("Methodological Baseline Audit: ", "Replicate and deconstruct the published GCNs-Net pipeline on PhysioNet EEGMMIDB to isolate literature leakage pathways."),'
    )

    # 4. Slide 6
    text = text.replace(
        '"Samples are single 6.25ms timepoints; split shuffles timepoints across train/test, causing severe data leakage. Drops to 50.9% on clean trials."',
        '"Samples are single 6.25ms timepoints; split shuffles timepoints across train/test, causing severe data leakage. Fails under continuous trial splits with zero sample overlap."'
    )

    # 5. Slide 8
    text = text.replace(
        '"Our Resolution: ", "Establish strict trial-level partitioning with zero sample overlap, exposing the 42% accuracy collapse and creating a clean benchmark."',
        '"Our Resolution: ", "Establish strict trial-level partitioning with zero sample overlap, eliminating timepoint leakage and creating an uncorrupted benchmark standard."'
    )
    text = text.replace(
        '(Inches(6.84), gap_y[2], "Impact Summary: Bridging the 42% Reality Gap", CRIMSON,',
        '(Inches(6.84), gap_y[2], "Impact Summary: Overcoming Literature Reality Gap", CRIMSON,'
    )
    text = text.replace(
        '"Scientific Impact: ", "Published 93.06% SOTA collapses to 50.90% under clean evaluation; true ceiling on PhysioNet is 65.0%.",',
        '"Scientific Impact: ", "Published 93.06% SOTA relies on timepoint leakage; true literature benchmark ceiling on PhysioNet is 65.00% across 216k pipelines (Vasques et al., 2026).",'
    )

    # 6. Slide 9
    text = text.replace(
        '("3. Signed ChebConv (K=2): ", "Chebyshev polynomial filtering: T_0(x)=x, T_1(x)=L~x, T_2(x)=2L~T_1 - T_0. Captures negative inter-hemispheric inhibition without exploding gradients."),',
        '("3. Signed ChebConv (K=2): ", "Chebyshev recurrence: T₀(x) = x, T₁(x) = L̃x, T₂(x) = 2L̃T₁(x) − T₀(x). Captures negative inter-hemispheric inhibition without numerical instability."),'
    )
    text = text.replace(
        '("4. Band-Power Log-Variance: ", "Temporal aggregation: P = log(1/T ∑_{t=1}^T x_t²). Converts oscillatory dynamics into stable power features, mirroring CSP physics."),',
        '("4. Band-Power Log-Variance: ", "Temporal aggregation: P = log( (1/T) ∑ xₜ² + ε ). Converts continuous dynamics into stable power features, mirroring CSP physics."),'
    )

    # 7. Slide 10
    text = text.replace(
        'add_header(slide, "Empirical Reality Gap: Data Leakage Audit", "Dissecting the 93.06% to 50.90% Collapse on PhysioNet EEGMMIDB", 10)',
        'add_header(slide, "Methodological Audit: Published Data Leakage Crisis", "Systematic Deconstruction of Literature Evaluation Pitfalls on PhysioNet", 10)'
    )
    text = text.replace(
        '("The 42% Reality Collapse: ", "When these 3 leaks are eliminated and evaluation is performed on continuous, trial-pure folds, GCNs-Net accuracy drops from 93.06% down to 50.90% (barely above the 25% chance level).")',
        '("Literature Reality Gap: ", "When evaluated on continuous, leak-free trials, published models cannot generalize, whereas extensive benchmarks prove the true clean ceiling across 216k pipelines is 65.00% (Vasques et al., 2026).")'
    )

    with open("scripts/build_presentation_nip.py", "w", encoding="utf-8") as f:
        f.write(text)

    print("Successfully updated scripts/build_presentation_nip.py!")

if __name__ == "__main__":
    main()
