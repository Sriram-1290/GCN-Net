"""Create PRESENTATION_SPEECH_NIP.md from PRESENTATION_SPEECH.md."""

with open("docs/PRESENTATION_SPEECH.md", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update Title and Header
text = text.replace(
    "# PW-1 Review-1 Presentation Defense Speech Guide",
    "# PW-1 Review-1 Presentation Defense Speech Guide (No-Implementation Phase / NIP)"
)

# 2. Slide 2: Abstract Spoken Script
old_s2 = """3. **Expected Outcomes:** GBP-Net resolves historical GCN failure modes and demonstrates a **+5.57% pre-training gain** (73.01% vs 67.44%), beating all classical and CNN baselines on PhysioNet."""
new_s2 = """3. **Expected Outcomes:** GBP-Net is formulated to resolve historical GCN failure modes, surpass classical CSP and deep CNN baselines under leak-free protocols, and relieve BCI illiteracy via cross-subject pre-training in PW-2."""
text = text.replace(old_s2, new_s2)

# 3. Slide 10: Spoken Script & Comeback
old_s10_script = """#### Spoken Script:
> "Slide 10 presents one of the most critical contributions of our Major Project: **Our Empirical Data Leakage Audit**, shown in **Figure 4**.
> 
> In Panel (a) on the left, we dissect the three compounding leakage vectors we uncovered in the official codebase of GCNs-Net:
> 1. *Timepoint-Level Shuffling:* The code shuffled 6.25ms slices. Because EEG has continuous temporal autocorrelation, having timepoint <em>t</em> in train and <em>t</em> + 1 in test allows the model to interpolate test labels with 99% accuracy through trivial 1D smoothing.
> 2. *Test-Time BatchNorm in Training Mode:* During inference, `model.train()` was left active. This recalculated batch mean and variance over the test mini-batch, directly leaking test-set target distributions into the output predictions.
> 3. *Transductive Adjacency:* The Pearson correlation graph was computed across the entire dataset before splitting.
> 
> **Look at Panel (b) for the quantitative reality gap:**
> - The published leaky baseline reported **93.06%**.
> - When we re-implemented the model under strict, leak-free trial splits, accuracy collapsed to **50.90%**&mdash;barely above the 25% chance level.
> - Furthermore, the extensive 2026 benchmark by Vasques et al. confirmed that the true within-session ceiling across 216,000 pipelines is **65.00%**.
> 
> Crucially, our proposed GBP-Net achieves **67.44%** without pre-training (PTC) and reaches a breakthrough **73.01%** with 92-subject cross-subject pre-training (PTG), delivering a +5.57% boost with 98 of 103 subjects performing above chance!"

#### Visual Cues:
- Point directly at the red bar at 50.90%, the 65.00% Vasques ceiling, and the green 73.01% pre-trained GBP-Net result.
- Deliver this with scientific confidence: you have audited the code, reproduced the baseline, and proved the flaw.

#### Anticipated Viva Question & Rapid Comeback:
- **Panel Question:** *"Is a 50.9% accuracy on 4 classes considered good or bad?"*
- **Defense Comeback:** *"Sir, for a 4-class problem, pure chance is 25%. While 50.9% is above random chance, it is 42 percentage points lower than what the authors claimed in IEEE Transactions (93.06%). It shows that their published model cannot generalize to unseen trials. That is why our pre-training and band-power architecture is essential, taking clean performance up to 73.01%."*"""

new_s10_script = """#### Spoken Script:
> "Slide 10 presents one of the most vital scientific contributions of our literature review: **Our Methodological Data Leakage Audit**, illustrated in **Figure 4**.
> 
> In Figure 4 on the left, we systematically deconstruct the three compounding leakage pathways identified in the reference codebase of GCNs-Net (SuperBruceJia/EEG-DL):
> 1. *Inter-Trial Timepoint Shuffling:* Shuffling isolated 6.25ms timepoint vectors destroys continuous trial boundaries. Because EEG possesses strong temporal autocorrelation, having timepoint <em>t</em> in train and <em>t</em> + 1 in test allows the model to interpolate test labels through trivial 1D continuity rather than learning neural patterns.
> 2. *Test-Time BatchNorm in Training Mode:* During evaluation, batch normalization was kept in training mode, calculating statistics over the test mini-batch and directly leaking test-set class distributions.
> 3. *Transductive Graph Estimation:* The Pearson correlation graph was computed across the pooled dataset before partitioning, pre-adapting spectral filters to unseen test geometries.
> 
> **Look at the bottom card for the literature reality gap:**
> - While published leaky pipelines claim **93.06%** or **94.20%**, independent large-scale benchmarks in literature (e.g., Vasques et al., 2026 across 216,714 pipelines) prove that the genuine literature ceiling sits at **65.00%**.
> - Our proposed GBP-Net is formulated specifically to eliminate all three leakage pathways, establishing a clean methodology designed to conquer sample scarcity legitimately via cross-subject pre-training in PW-2!"

#### Visual Cues:
- Point directly at the three leakage columns in Figure 4, then gesture toward the 65.00% Vasques empirical ceiling benchmark card below.
- Deliver this with scientific authority: you are presenting an architectural deconstruction grounded in literature provenance.

#### Anticipated Viva Question & Rapid Comeback:
- **Panel Question:** *"Why do you emphasize data leakage so strongly in Review-1?"*
- **Defense Comeback:** *"Respected sir, in machine learning for healthcare and BCIs, reproducibility is paramount. Several recent papers in top IEEE journals claimed >90% accuracy on PhysioNet, but our code audit revealed this was caused by data leakage. Identifying this gap in Review-1 ensures that our proposed GBP-Net architecture and upcoming PW-2 implementation are built on a leak-free, scientifically valid foundation."*"""

text = text.replace(old_s10_script, new_s10_script)

# 4. Slide 11: Spoken Script & Visual Cues
old_s11_script = """Our system is implemented using **PyTorch 2.4** and **PyTorch Geometric (PyG)** for graph operations, with **MNE-Python** for EEG signal preprocessing. Training is accelerated using an **NVIDIA GeForce RTX 5060 GPU with 8GB VRAM** and CUDA 12.4. Our codebase contains 337 unit tests, all currently passing with zero failures."""
new_s11_script = """Our planned system is designed in **PyTorch 2.4** and **PyTorch Geometric (PyG)** for spectral graph operations, with **MNE-Python** for EEG preprocessing and montage handling. Computation is targeted for an **NVIDIA GeForce RTX 5060 GPU with 8GB VRAM** and CUDA 12.4. The implementation pipeline is structured modularly for rigorous test-driven validation in PW-2."""
text = text.replace(old_s11_script, new_s11_script)

text = text.replace('Emphasize *"NVIDIA RTX 5060 GPU"* and *"337 unit tests passing"*.',
                    'Emphasize *"NVIDIA RTX 5060 GPU"* and *"Modular test-driven design planned for PW-2"*.' )

# 5. Slide 12: Spoken Script
old_s12_script = """1. **A Verified Leak-Free SOTA Architecture:** We will deliver GBP-Net achieving **73.01% accuracy** with 92-subject cross-subject pre-training (67.44% control), surpassing the 65.0% Vasques empirical ceiling and outperforming unconstrained CNN and Riemannian baselines.
> 2. **Definitive Resolution of the Graph Hypothesis:** We will provide the first controlled ablation study proving conclusively whether spectral graph convolutions provide genuine inductive bias over spatial filters when data leakage is eliminated.
> 3. **Neurophysiological Interpretability:** We will produce 2D topographical scalp heatmaps demonstrating that learned dynamic attention weights peak over C3 and C4 for hand imagery and Cz for foot imagery, validating model trustworthiness for clinical neurologists.
> 4. **Academic & Open Deliverables:** A fully documented, open-source PyTorch/PyG repository with 337 passing unit tests, our completed 14-page IEEE Review Paper, and complete PW-1 project documentation."""

new_s12_script = """1. **Anticipated Decoding Breakthrough:** Formulated to resolve the classical deficit without data leakage, break past the 65.0% literature ceiling, and alleviate BCI illiteracy via cross-subject pre-training.
> 2. **Controlled Ablation Suite:** Planned for PW-2 to definitively isolate whether spectral graph convolutions provide genuine inductive bias compared to unconstrained spatial filters.
> 3. **Neurophysiological Interpretability:** Planned topographical scalp maps confirming attention activation peaks over C3, Cz, and C4 sensorimotor regions.
> 4. **Phase-Wise Roadmap:** Complete Phase-1 deliverables today (Review Paper, 1-Page Analysis, Problem Formulation, Title Defense), followed by full empirical coding, pre-training execution, and thesis defense in Phase-2."""

text = text.replace(old_s12_script, new_s12_script)

# 6. Q37 update
old_q37 = """Crucially, GBP-Net reaches 67.44% without pre-training and achieves 73.01% under 92-subject pre-training, surpassing the 65.0% Vasques empirical ceiling."*"""
new_q37 = """Our proposed GBP-Net is specifically formulated to eliminate all three leakage pathways, targeting performance beyond the 65.0% Vasques empirical ceiling through cross-subject pre-training in PW-2."*"""
text = text.replace(old_q37, new_q37)

# 7. Q41 update
old_q41 = """Empirically, this grouped pre-training yields 73.01% within-family accuracy (PTG)&mdash;a +5.57 percentage point boost over the non-pretrained control (PTC: 67.44%), with 98 of 103 subjects performing above chance."*"""
new_q41 = """This grouped pre-training design is formulated to provide an uncorrupted, highly generalized spatial representation for target fine-tuning, directly addressing the per-subject sample starvation identified across literature."*"""
text = text.replace(old_q41, new_q41)

# 8. Q54 update (unit tests)
old_q54 = """#### Q54: "What is the role of your repository's 337 passing unit tests?"
- **The Underlying Trap:** Software engineering quality assurance in research.
- **Master Defense Answer:**
  > *"Respected sir, in computational neuroscience and deep learning, silent bugs (such as off-by-one temporal shifts or eval-mode toggles) produce confident but completely wrong results. Our 337 automated unit tests cover:  
  > 1. Zero sample leakage across train/test splits (`test_splits.py`).  
  > 2. Exact Chebyshev recurrence relation convergence (`test_model.py`).  
  > 3. Strict subject group disjointness during pre-training (`test_pretrain_finetune.py`).  
  > 4. Verified numerical equivalence between our PyTorch filter bank and classical Butterworth filtering (`test_features.py`).  
  > This guarantees senior-engineer level reproducibility."*"""

new_q54 = """#### Q54: "How will you ensure software quality and prevent silent bugs during the upcoming PW-2 implementation?"
- **The Underlying Trap:** Testing your engineering planning and scientific reproducibility for Phase-2.
- **Master Defense Answer:**
  > *"Respected sir, in computational neuroscience and deep learning, silent bugs (such as off-by-one temporal window slicing or incorrect evaluation modes) can produce confident but scientifically invalid numbers. For our Phase-2 implementation, we have designed a strict test-driven development plan that will verify:  
  > 1. Zero sample leakage across train and test partitions.  
  > 2. Numerical convergence of truncated Chebyshev spectral recurrences.  
  > 3. Complete group disjointness during cross-subject pre-training.  
  > 4. Equivalence between learnable temporal filter banks and classical frequency sub-bands.  
  > This ensures our empirical results in PW-2 will be unassailable and fully reproducible."*"""

text = text.replace(old_q54, new_q54)

# 9. Insert new Question on Review-1 scope at top of Q&A
q_intro = """### Domain 1: Project Scope, Approved Title & Architectural Grounding (Q1 &ndash; Q5)

#### Q1: "Why is there no implementation, source code, or experimental result presented in this Review-1 presentation?"
- **The Underlying Trap:** The reviewer is testing whether you understand the official syllabus, academic circular, and review schedule.
- **Master Defense Answer:**
  > *"Respected panel members, as per the official departmental circular issued on 26-09-2026 and the Osmania University curriculum for B.E. Major Project Work-1 (PW-1), Review-1 is strictly defined as the **Problem Formulation, Literature Survey, Title Defense, and Proposed Architectural Design** phase.  
  > Full software implementation, model training, and empirical benchmarking are formally scheduled for Review-2 and Major Project Work-2 (PW-2).  
  > In Review-1, our objective is to comprehensively synthesize the 36 literature papers, deconstruct published data leakage pitfalls, and formulate a mathematically sound architecture (GBP-Net) so that our upcoming implementation in PW-2 is built on a solid scientific foundation."*

#### Q2: """

text = text.replace("### Domain 1: Project Scope, Approved Title & Architectural Grounding (Q1 &ndash; Q5)\n\n#### Q1: ", q_intro)

with open("docs/PRESENTATION_SPEECH_NIP.md", "w", encoding="utf-8") as f:
    f.write(text)

print("PRESENTATION_SPEECH_NIP.md written successfully!")
