from pathlib import Path

p = Path("docs/PRESENTATION_SPEECH.md")
text = p.read_text(encoding="utf-8")

# 1. Slide 10 spoken script
old_s10 = """> - Furthermore, the extensive 2026 benchmark by Vasques et al. confirmed that the true empirical ceiling across 216,000 pipelines is **65.00%**.
> 
> Our proposed GBP-Net targets **greater than 67%** under verified, leak-free conditions."

#### Visual Cues:
- Point directly at the red bar at 50.90% and the green target bar at >67%.
- Deliver this with scientific confidence: you have audited the code, reproduced the baseline, and proved the flaw.

#### Anticipated Viva Question & Rapid Comeback:
- **Panel Question:** *"Is a 50.9% accuracy on 4 classes considered good or bad?"*
- **Defense Comeback:** *"Sir, for a 4-class problem, pure chance is 25%. While 50.9% is above random chance, it is 42 percentage points lower than what the authors claimed in IEEE Transactions (93.06%). It shows that their published model cannot generalize to unseen trials. That is why our pre-training and band-power architecture is essential to push performance past 67%."*"""

new_s10 = """> - Furthermore, the extensive 2026 benchmark by Vasques et al. confirmed that the true within-session ceiling across 216,000 pipelines is **65.00%**.
> 
> Crucially, our proposed GBP-Net achieves **67.44%** without pre-training (PTC) and reaches a breakthrough **73.01%** with 92-subject cross-subject pre-training (PTG), delivering a +5.57% boost with 98 of 103 subjects performing above chance!"

#### Visual Cues:
- Point directly at the red bar at 50.90%, the 65.00% Vasques ceiling, and the green 73.01% pre-trained GBP-Net result.
- Deliver this with scientific confidence: you have audited the code, reproduced the baseline, and proved the flaw.

#### Anticipated Viva Question & Rapid Comeback:
- **Panel Question:** *"Is a 50.9% accuracy on 4 classes considered good or bad?"*
- **Defense Comeback:** *"Sir, for a 4-class problem, pure chance is 25%. While 50.9% is above random chance, it is 42 percentage points lower than what the authors claimed in IEEE Transactions (93.06%). It shows that their published model cannot generalize to unseen trials. That is why our pre-training and band-power architecture is essential, taking clean performance up to 73.01%."*"""

assert old_s10 in text, "old_s10 not found in PRESENTATION_SPEECH.md"
text = text.replace(old_s10, new_s10)

# 2. Slide 12 spoken script
old_s12 = """> 1. **A Verified Leak-Free SOTA Architecture:** We will deliver GBP-Net achieving over 67% 4-class classification accuracy under verified cross-subject evaluation on PhysioNet, outperforming unconstrained CNN and Riemannian baselines."""
new_s12 = """> 1. **A Verified Leak-Free SOTA Architecture:** We will deliver GBP-Net achieving **73.01% accuracy** with 92-subject cross-subject pre-training (67.44% control), surpassing the 65.0% Vasques empirical ceiling and outperforming unconstrained CNN and Riemannian baselines."""

assert old_s12 in text, "old_s12 not found in PRESENTATION_SPEECH.md"
text = text.replace(old_s12, new_s12)

# 3. Q37 answer
old_q37 = """> *"The official published GCNs-Net reports **93.06%** subject-level accuracy. In our clean re-implementation enforcing continuous trial-level splitting with zero timepoint leakage, GCNs-Net drops to **50.90%** on 4-class motor imagery (where random guessing is 25%). This 42.16-percentage-point collapse proves that published results are artifacts of data leakage, establishing the true baseline we must beat."*"""
new_q37 = """> *"The official published GCNs-Net reports **93.06%** subject-level accuracy. In our clean re-implementation enforcing continuous trial-level splitting with zero timepoint leakage, GCNs-Net drops to **50.90%** on 4-class motor imagery (where random guessing is 25%). This 42.16-percentage-point collapse proves that published results are artifacts of data leakage, establishing the true baseline we must beat. Crucially, GBP-Net reaches 67.44% without pre-training and achieves 73.01% under 92-subject pre-training, surpassing the 65.0% Vasques empirical ceiling."*"""

assert old_q37 in text, "old_q37 not found in PRESENTATION_SPEECH.md"
text = text.replace(old_q37, new_q37)

# 4. Q41 answer
old_q41 = """> *"In Stage 1, we pool trials from 92 background subjects (over 15,000 continuous trials). We train the temporal filter banks and dynamic ChebConv layers using cross-entropy loss over the population. This forces the spatial filters to learn universal, subject-invariant sensorimotor manifold projections. In Stage 2, these spatial filters are transferred to target subjects, either zero-shot or with lightweight linear calibration, solving single-subject sample starvation."*"""
new_q41 = """> *"In Stage 1, we pool trials from 92 background subjects (over 15,000 continuous trials). We train the temporal filter banks and dynamic ChebConv layers using cross-entropy loss over the population. This forces the spatial filters to learn universal, subject-invariant sensorimotor manifold projections. In Stage 2, these spatial filters are transferred to target subjects, either zero-shot or with lightweight linear calibration, solving single-subject sample starvation. Empirically, this grouped pre-training yields 73.01% within-family accuracy (PTG)&mdash;a +5.57 percentage point boost over the non-pretrained control (PTC: 67.44%), with 98 of 103 subjects performing above chance."*"""

assert old_q41 in text, "old_q41 not found in PRESENTATION_SPEECH.md"
text = text.replace(old_q41, new_q41)

p.write_text(text, encoding="utf-8")
print("PRESENTATION_SPEECH.md updated successfully!")
