---
title: "Where Next: Evidence Review and Future Work"
subtitle: "Two fact-checked literature research rounds on EEG motor-imagery decoding, and what they mean for this project"
project: "GBP-Net / GCN-Net — EEG Motor Imagery Decoding (Neil Gogte Institute of Technology)"
date: "7 November 2026"
---

## Summary

The team asked two questions: what would give this project a *truly superior* decoding approach, and what would make it publishable in a Q1 journal. Two rounds of automated literature research tried to answer them. Every claim was checked by three independent verifiers, and any claim that two of them refuted was dropped. In total, 42 sources were read, 200 claims were extracted, 50 were verified, 43 survived and 7 were refuted.

**Five conclusions:**

1. **No motor-imagery architecture has an independently replicated advantage under fair evaluation.** Attention add-ons gain 0.4–1.5 points with no significance test. Many superiority claims beat an under-tuned baseline. Large benchmarks put classical covariance pipelines level with or ahead of CNNs. Twelve EEG foundation models are statistically tied with EEGNet and ShallowConvNet. A "superior architecture" is the weakest route to a Q1 paper, and it fits the project's own record: five ideas aimed at the ~0.73 result have measured nothing.
2. **What reliably improves cross-subject decoding is alignment and training recipe, not architecture.** Euclidean alignment (EA) gives +4.3 to +5.6 points under leave-one-subject-out (LOSO) in a peer-reviewed study, and about +10 points in a model pre-trained on 60 subjects. The project has not tried it. It is most likely to help the subject-independent LOSO arm (0.604). Whether it helps after full per-subject fine-tuning (the 0.73 arm) has never been tested.
3. **The project's strongest Q1 asset is evaluation rigour.** Published results on PhysioNet EEGMMIDB are exposed to three inflation routes the literature confirms: foundation models pre-trained on the test dataset, 4-class scores that pool the two run families, and non-neural confounds. The project already holds a fourth: the GCNs-Net collapse from 0.898 to chance.
4. **Novelty is partial and must be stated precisely.** The general block-design confound (Li et al., *IEEE TPAMI* 2021) and segment-level leakage in clinical EEG (Brookshire et al. 2024) are published. Still open and specific to this project:
    - the EEGMMIDB run-family confound, which survives band-pass filtering where Li et al.'s did not;
    - MOABB's own PhysioNet right-hand-vs-feet benchmark, which crosses run families without saying so.

    Foundation-model contamination, the GCNs-Net re-evaluation, train-mode BatchNorm and pooled-data graphs are **unchecked**: neither confirmed open nor found published.

5. **A Q1 submission needs 2–4 datasets, NERVE-ML compliance and released code and models.** The project already meets most NERVE-ML items, and its pre-registration is rarer than any precedent found.

**Recommended direction:** one evaluation paper with a real result inside it. Working title: *"What inflates and what improves motor-imagery decoding on PhysioNet: leakage, confounds, contamination, and alignment."* It would combine an audit, a contamination-controlled foundation-model comparison, alignment ablations and 2–3 datasets. Target journals: *Journal of Neural Engineering*, *IEEE TNSRE*, *IEEE JBHI* or *Patterns*. This is PW-2 work. The Review-2 review-paper submission (before 20 November 2026) comes first.

## 1. Purpose and method

### 1.1 Why this review

The project's measured results, recorded in `docs/project_documentation.md` §9 and §12, show that on PhysioNet EEGMMIDB the decoder's architecture barely matters once evaluation is correct. Cross-subject pre-training was the only large lever, and it helped every architecture equally. The team wants a stronger contribution, so this review asks what the wider literature says about two things: genuine improvements, and what Q1 journals accept.

### 1.2 How the research was done

Each round followed the same automated, adversarial process:

| Stage | What happened |
|---|---|
| Scope | The question was split into five search angles |
| Search | Five parallel web-search agents, one per angle |
| Fetch | Top sources fetched and de-duplicated; falsifiable claims extracted |
| Verify | Each claim checked by three independent verifiers; killed if two of three refuted it |
| Synthesise | Duplicates merged; claims ranked by confidence; sources cited |

| | Round 1 (broad survey) | Round 2 (filling the gaps) | Total |
|---|---|---|---|
| Sources fetched | 21 | 21 | 42 |
| Claims extracted | 105 | 95 | 200 |
| Claims verified | 25 | 25 | 50 |
| Confirmed | 21 | 22 | 43 |
| Refuted | 4 | 3 | 7 |
| Agents run | 103 | 103 | 206 |

**Confidence labels:**

- **High:** confirmed 3–0 by the verifiers, from a primary source.
- **Medium:** confirmed, but from a preprint, a single group or a small sample, or merged with a caveat.
- **Low:** a synthesis or an inference from absence, not a positive finding.

### 1.3 Limits of the evidence

- **Many key sources are recent, unreviewed preprints.** Examples are NeuroAtlas, Vasques et al. 2026, the MOABB benchmark, Del Pup et al. 2025 and Wimpff et al. 2025. Figures may change in later versions.
- **Almost every effect size comes from other datasets**, mainly BCI Competition IV-2a/2b (9 subjects each), not EEGMMIDB. Gains may be smaller here.
- **The alignment evidence comes from overlapping author groups**, so it is consistent but not independently replicated.
- **Three questions found no surviving evidence in either direction:** whether foundation models fit an 8 GB GPU, dataset licences and sizes, and novelty for four of the audit items. These are listed as open questions (Section 8), not as findings.

## 2. Starting point: what the project has measured

All figures are within-family accuracy (chance 0.50), over 103 subjects with paired statistics unless stated (`CLAUDE.md`, `docs/project_documentation.md`).

| Arm | Accuracy | Note |
|---|---|---|
| GCNs-Net, published (leaky) protocol | 0.898 | Reproduces the paper's 0.884 (timepoint accuracy) |
| GCNs-Net, correct split | **0.509** | At chance; the public code leaks three ways |
| Correctly split model: run-family identification | 0.617 | Chance 0.50, **under the 8–30 Hz band-pass** (§9) |
| CSP + LDA | 0.624 | Classical baseline |
| Tangent space (covariance) | 0.669 | Strongest classical baseline (§9.19) |
| ShallowConvNet | 0.673 | Beats CSP by +0.049 |
| Graph models, reordered (G1–G3) | 0.663–0.674 | Graph prior contributes nothing |
| Cross-subject pre-training, graph model (PTG) | 0.730 | +0.056 over its control |
| Cross-subject pre-training, ShallowConvNet | 0.737 | **Tied** with PTG: a data effect, not architecture |
| + domain-specific batch norm | 0.722 | No gain |
| LOSO (subject-independent, binary) | 0.604 | 4-class LOSO never measured |
| Montage robustness (§9.25, 20 subjects) | −0.112 vs ShallowConvNet | Loss on every cap; Phases 2–3 gated out |

## 3. Findings

### 3.1 Architectures: no replicated advantage

**No architecture has an independently replicated gain under fair evaluation (high confidence).**

- **Attention add-ons are small.** In one shared framework, channel attention added only +0.40 to +1.53 points across 8 dataset and protocol settings, with no significance test, and with the attention settings chosen on test accuracy [1].
- **Baselines are often under-tuned.** The same paper's ShallowConvNet scored 58.8 % on BCI IV-2a, against 73.7 % in the original publication. Against the original, the new model's margin shrinks from about 17 points to about 2–3 [1]. The authors themselves warn that comparing complex models to untuned EEGNet or ShallowConvNet "raises concerns about the validity of the comparison".
- **A tuned ShallowConvNet is a top performer.** On BNCI2014-001 left vs right, ShallowConvNet (0.776) is the strongest of 12 non-foundation baselines, ahead of Conformer, IFNet, FBCNet and Deformer [2]. A 13-dataset benchmark places ShallowConvNet, EEGNet, Conformer, Deformer and LMDA in one statistically tied top group [3].
- **Classical covariance pipelines are level with or ahead of CNNs.** In MOABB's benchmark (14 motor-imagery datasets, within-session), tangent space + SVM averages 71.1 against ShallowConvNet's 63.9 over the six multi-class datasets. On PhysioNet the figures are 58.5 against 41.9 [4]. In Vasques et al. (PhysioNet, 109 subjects, left vs right), tangent space (0.644) and CSP (0.615) are statistically indistinguishable [5].

**Implication.** The project's ShallowConvNet, which beats CSP, is a properly tuned baseline, and its tangent-space arm (0.669) already matches the strongest classical result. Any new architecture would have to clear both with paired per-subject statistics, which most of the literature does not provide. A paper framed around architectural superiority is the riskiest kind, with reviewers who check baselines.

### 3.2 EEG foundation models

**Foundation models do not reliably beat compact CNNs on motor imagery (high confidence).**

- **Fine-tuned, they're statistically tied.** Twelve foundation models are tied with EEGNet and ShallowConvNet, and EEGNet (about 2,000 parameters) ranked second overall, ahead of every other foundation model [3].
- **Frozen, they barely beat chance.** With the backbone frozen and only a linear classifier trained, foundation models "struggle to reach motor imagery performance significantly above chance". An end-to-end EEGNet beat the best of them on 13 of 18 datasets [6].
- **Size doesn't predict performance.** EEGPT, at 25 million parameters, ranked 12th [3].
- **A strict audit agrees.** One audit chose every setting from training data only and found supervised models (ATCNet, EEG Conformer, ShallowConvNet) beat every LaBraM and CBraMod configuration on 4-class IV-2a (n = 9, a preprint) [7].

**Frozen foundation models may rely on non-neural confounds (medium confidence).** Under a "confound filter" (4 Hz high-pass, 40 Hz low-pass, first second after the cue removed), frozen foundation-model performance fell below significance on 7 datasets [6].

- **Limits:** this covers frozen backbones only; the three filter steps were not tested separately; and it was not shown to be specific to foundation models.
- **For this project:** the 8–30 Hz band-pass already removes slow eye-movement activity, but the position of the 64-sample window relative to the cue has not been checked.

**Several foundation models were pre-trained on EEGMMIDB itself (high confidence).**

- **LaBraM** lists the full EEG Motor Movement/Imagery dataset (109 subjects, 47.3 hours) in its pre-training data [8]. Meta's NeuralBench model card warns that its motor-imagery results "may be inflated because the model has seen the same recordings (though not the same labels)" [9].
- **EEGPT** [3] and **AFPM** [2] also included it.
- **CBraMod** was pre-trained only on the Temple University Hospital clinical corpus [10], so it is the clean choice for a fair test.
- **The size of this inflation has never been measured** anywhere found.

**The best published foundation-model figure on EEGMMIDB has the flaws this project documented (high confidence).** CBraMod reports a 4-class balanced accuracy of 0.6417 [10]. It pools both run families, comes from one fixed 20-subject test split with no per-subject statistics, and an independent group using CBraMod's own code reached 61.2 %, below LaBraM's reported 0.617. *Never compare 0.6417 with this project's binary LOSO 0.604.*

### 3.3 Alignment and training recipe: the most likely real gain

**Alignment and training recipe matter far more than the backbone (high confidence).** In CBraMod's own appendix (4-class LOSO, IV-2a), a bundle of four ingredients adds +0.14 to +0.16 to *either* model, while the backbone difference is only +0.02 to +0.03 [10]. The ingredients are Euclidean alignment, session statistics, Mixup and subject-wise regularisation. They were tested only as a bundle, on 9 subjects, with no significance test.

**Euclidean alignment helps models trained on other subjects (high confidence).**

- **+4.3 to +5.6 points under LOSO.** Online EA (reference from the first 24 target trials) raised LOSO accuracy from 70.5 % to 74.8 % (+4.3 points); offline EA reached 76.1 % (+5.6) [11]. That was 2 datasets, 23 subjects, 3 CNNs, significant by permutation test, and convergence needed over 70 % fewer iterations.
- **Similar gains elsewhere.** Online alignment added +5.1 points (EA) and about +6.3 (Riemannian alignment) on IV-2a [12]. Disabling online EA in a model pre-trained on 60 subjects cost about 10 points (p < 0.001; a preprint) [13].

**Whether alignment still helps after full per-subject fine-tuning is unknown (medium confidence).**

- **The closest test is inconclusive.** After EA, light fine-tuning (only the last layer, on 24 trials) added nothing (74.81 → 74.79 %), but EA plus fine-tuning still beat fine-tuning without EA by about 2.9 points [11].
- **No study tests full fine-tuning.** None tests the project's PTG setting: the whole network fine-tuned on about 1,665 windows per subject.
- **The project's own result is a warning.** Domain-specific batch norm was absorbed by pre-training. Expect a smaller EA gain on the pre-trained arm than on the LOSO arm.

**Test-time normalisation is a cheap, label-free add-on for LOSO (medium confidence).** Re-estimating BatchNorm statistics on the unlabelled target subject matched or beat alignment on IV-2a/2b, and combining the two was better still [12]. That was a single group with 9 subjects, and the margins were within one standard deviation.

**Open practical problem:** the common average reference makes the covariance matrix rank-deficient (rank 63 of 64), so EA's inverse square root needs a stated remedy: regularisation, a pseudo-inverse, or dropping one channel. No verified source specifies which.

### 3.4 Data augmentation: weak evidence

**Augmentation gains come from a single group, not replication (medium confidence).** On BCI IV-2a, within-subject:

- Fourier-phase surrogates gave up to a 45 % *relative* gain, but only with one-eighth of the training data [14].
- Channel dropout gave up to 25 % relative on the full data, against a weak baseline (balanced accuracy about 0.60).

Neither was tested under LOSO or with cross-subject pre-training. Treat any expected gain in this project as low, and generate augmentations inside each fold, from training trials only.

### 3.5 Cross-dataset transfer

**Transferring a pre-trained model between datasets is cheap, but EEGMMIDB is a hard target (medium confidence).** EEGNet pre-trained on one dataset, frozen, with only a linear layer trained on the target, reached ROC-AUC of about 0.75 (left vs right) and about 0.70 (feet vs right hand) on PhysioNet with 16 trials per class, using 3 channels [15]. Code and models are public. PhysioNet's feet-vs-right-hand task crosses run families, so it shares the confound this project found.

### 3.6 Evaluation protocol

**Nested cross-validation (medium confidence).** A 2025 preprint argues that non-nested LOSO leaks when the same validation set picks the early-stopping point and is also reported, and that nested schemes are more reliable [16].

**Status in this project (checked):** compliant. The test fold is never used for model selection:

- subject-dependent runs carve their validation set out of each fold's training trials;
- LOSO and pre-training validate on separate held-out subjects;
- fine-tuning uses the final epoch.

## 4. Novelty landscape

| Contribution idea | Prior work found | What remains open | Status |
|---|---|---|---|
| Block or run-level confound in EEG classification | **Li et al., *IEEE TPAMI* 2021** [17]: block-only labels gave about 100 % without filtering, but **collapsed to chance after band-pass filtering** (40-class visual task) | The EEGMMIDB form: two classes interleaved within runs, classes confined to run families. The project measures run-family identification at 0.617 **under 8–30 Hz**, which Li et al.'s result does not predict | **Partly published; EEGMMIDB form open** |
| Run-family confound in the standard benchmark | MOABB reports PhysioNet right hand vs feet at AUC 0.94 against 0.68 for left vs right, **without noting the task crosses run families** [4] | How much of the 0.94 is the confound: never measured | **Open (strong hook)** |
| Segment-level leakage in EEG deep learning | **Brookshire et al. 2024** [18]: segment splits 99.8 % vs subject splits 53.0 %; only 17 of 63 clinical papers clearly avoided leakage. Clinical EEG only | Motor imagery and BCI, EEGMMIDB, train-mode BatchNorm, pooled-data graphs and foundation models: none mentioned | **General mechanism published; MI specifics open** |
| Foundation-model pre-training contamination | A model card warns of inflation [9]; one benchmark's "leakage-aware" ranking was unchanged [3] | How much contamination inflates results, measured by fine-tuning: not found | **Unchecked** |
| GCNs-Net or graph MI models re-evaluated under correct splits | None found | — | **Unchecked** |
| Train-mode BatchNorm and pooled-data graph as leakage paths | Not named in NERVE-ML, Brookshire or Li | — | **Unchecked** |

"Unchecked" means the searches produced no surviving evidence either way. Before claiming any of these as novel, a targeted check is needed: the forward citations of GCNs-Net, the related-work sections of EEG-FM-Compass, and the LaBraM and EEGPT evaluation papers.

## 5. What a Q1 submission needs

### 5.1 The NERVE-ML checklist

NERVE-ML [19] is a reviewer-facing checklist for machine learning in neural engineering (Carlson, Chavarriaga, Liu, Lotte and Lu; *J. Neural Eng.* 22(2):021002, 2025). It has five categories and about 40 items. It is non-prescriptive and not a mandatory submission requirement, but it describes what reviewers check.

| NERVE-ML requirement | GCNs-Net public code | This project |
|---|---|---|
| Statistically independent train and test data; test epochs not overlapping or adjacent to training epochs | **Violated:** timepoint shuffling | Met: trial- and subject-disjoint splits |
| Normalisation parameters chosen on training data only | **Violated:** train-mode BatchNorm at test time | Met: training-only standardiser, eval-mode BatchNorm |
| Parameters chosen on training data, not on all data | **Violated:** graph built on pooled data | Met: training-split graph |
| Hyperparameters and architecture chosen without test data | — | Met: validation from training data; test never used |
| Chance level; uncertainty; statistics | — | Met: chance levels, bootstrap CIs, paired tests, Holm correction |
| Full code, trained models, software versions released | — | **To do:** code exists; release of code and trained models must be committed to |

### 5.2 Precedents in the target venues

| Paper | Venue | Datasets | Notes |
|---|---|---|---|
| Li et al. 2021 [17] | *IEEE TPAMI* | 1 re-analysed study | Block-design critique |
| Brookshire et al. 2024 [18] | *Frontiers in Neuroscience* | 63-paper review + 2 datasets | Leakage audit |
| Rommel et al. 2022 [14] | *J. Neural Eng.* | 2 (one motor imagery, 9 subjects) | Augmentation benchmark |
| Junqueira et al. 2024 [11] | *J. Neural Eng.* | 2 (23 subjects), 3 models, LOSO | Alignment |
| Wimpff et al. 2024 [1] | *J. Neural Eng.* | 4, 5 seeds, within- and cross-subject | Attention benchmark |
| MOABB benchmark [4] | Preprint (submitted to JNE) | 36 (14 motor imagery) | Within-session only |
| Vasques et al. 2026 [5] | Preprint | 3, more than 340,000 fits | Friedman–Nemenyi statistics |

**Takeaways:**

- Precedents use **2–4 datasets**, with careful statistics.
- **Within-session-only benchmarks are criticised as "the easiest regime"**, including by their own authors, so cross-subject results matter.
- **None of the precedents reports pre-registration**, which this project has for every experiment.
- **No journal statement on negative results in BCI was found.**

## 6. Claims not to cite, and corrections

### 6.1 Refuted claims (do not cite)

1. "Calibration-free Riemannian re-centring raises cross-dataset accuracy in all 8 datasets." Refuted 0–3.
2. "Aligned models trained on other datasets beat within-subject training on PhysioNet (above 60 % vs 59 %)." Refuted 0–3.
3. "EA helps deep networks only after temporal filtering" (the 49.33 / 48.00 / 66.33 % figures). Refuted 0–3.
4. "The foundation-model deficit is dataset-dependent, with no gap on BNCI2014-004", as framed. Refuted 0–3.
5. Kapoor & Narayanan's survey summarised as covering specific EEG and BCI work. Refuted 0–3; cite it only as a general leakage framework.
6. "Del Pup et al. used the 109-subject EEGMMIDB." Refuted 1–2.
7. "Eder, Xu & Grosse-Wentrup (*J. Neural Eng.* 2024) found no significant CNN-vs-Riemannian difference across within-session, cross-session and cross-subject evaluation." Refuted 0–3.

### 6.2 Corrections to apply when citing

- **NERVE-ML:** about **40** items, not 38. It is non-prescriptive.
- **Wimpff et al. 2024:** tested **16** attention variants. Its third dataset is "BCIC III IVa".
- **Junqueira et al.:** +4.33 points is the **online** EA figure; offline EA is +5.59 points (table) or +5.55 (text).
- **Brookshire et al.:** write "17 of 63 papers clearly avoided leakage", not "46 leaked".
- **GCNs-Net vs NERVE-ML:** the mapping of GCNs-Net's leakage paths to NERVE-ML items is this project's own analysis. Write "GCNs-Net's code violates NERVE-ML items X, Y, Z", not "NERVE-ML identifies these paths".

## 7. Future work

### 7.1 The proposed paper

**Working title:** *What inflates and what improves motor-imagery decoding on PhysioNet: leakage, confounds, contamination, and alignment.*

**Thesis:** published motor-imagery results on PhysioNet EEGMMIDB are inflated by identifiable mechanisms. Under strict evaluation, data and alignment, not architecture, are what improve decoding.

**Contributions:**

1. **An audit of how results get inflated:**
    - the three leakage paths in GCNs-Net's public code, each mapped to a NERVE-ML item, with the 0.898 → 0.509 collapse;
    - the run-family confound measured directly, including how much of MOABB's 0.94 right-hand-vs-feet score survives run-disjoint evaluation.
2. **Foundation-model contamination:** fine-tuned CBraMod (never saw EEGMMIDB) against LaBraM (did) and the project's pre-trained ShallowConvNet, on 103 subjects with paired statistics. This would be the first measurement of contamination inflation.
3. **What genuinely helps:** Euclidean alignment, test-time normalisation and augmentation, each added one at a time to the LOSO and pre-training pipelines, with pre-registered decision rules.
4. **Generalisation:** a second and third dataset, and cross-dataset transfer.

**Target journals:** *Journal of Neural Engineering* (where most precedents sit), *IEEE TNSRE*, *IEEE JBHI*, *Patterns*.

### 7.2 Work packages

Effort estimates are rough person-weeks for a team of three, on one RTX 5060 (8 GB) and about 16 GB of RAM.

**WP0. Checks before committing (about 1 week)**

- A targeted novelty search for the four unchecked items in Section 4.
- Feasibility lookups: parameter counts, input length, memory use and public checkpoints for CBraMod and LaBraM; licences, sizes and channel overlap of the candidate datasets.
- Decide the paper's exact claims from the results.

**WP1. Prerequisites (about 2 weeks)**

- **4-class LOSO with the run-family decomposition,** never measured and needed before any 4-class comparison.
- **Confound check:** where the 64-sample window sits relative to the cue, and whether results hold when the first second after the cue is excluded.

**WP2. The audit (about 3 weeks)**

- Tabulate the three GCNs-Net leakage paths against NERVE-ML items, using existing results.
- **Measure the run-family confound without labels:** decode run family (runs 4/8/12 vs 6/10/14) from the **rest periods (T0)**, which carry no class information, under trial- and run-disjoint splits, with and without the 8–30 Hz band-pass. This directly tests Li et al.'s band-pass result on motor-imagery data.
- **Re-score MOABB's PhysioNet right-hand-vs-feet task** with run-disjoint splits, and report how much of the 0.94 survives.
- Pre-register predictions before running.

**WP3. Alignment and recipe (about 3–4 weeks)**

- **EA in the LOSO pipeline**, applied after the band-pass, with a stated remedy for the common-average-reference rank deficiency. Apply EA equally to the CSP and tangent-space baselines, or it manufactures a win.
- **EA in the pre-training pipeline**, using the same groups and seeds as PTG and PTG-shallow, so results are paired.
- **Test-time BatchNorm adaptation for LOSO**, then augmentation (channel dropout, Fourier surrogates), **one at a time**, as the project rules require.
- **Decision rule:** the existing 0.020 minimum detectable difference, with paired bootstrap CIs and Holm correction.

**WP4. Foundation models (about 3–4 weeks)**

- Fine-tune **CBraMod** and **LaBraM** on EEGMMIDB under the project's splits, against PTG-shallow, with within-family scoring and 4-class scoring with the family decomposition.
- Disclose LaBraM's pre-training exposure. The CBraMod–LaBraM gap is the contamination estimate. Never use a checkpoint already fine-tuned on PhysioNet subjects.
- **Gate:** if fine-tuning doesn't fit in 8 GB (WP0), either use a smaller input or fall back to frozen-backbone evaluation and say so.

**WP5. Second and third datasets (about 3–4 weeks)**

- **BCI Competition IV-2a** (9 subjects, 22 channels) is already allowed by the project rules (roadmap D3). Candidates with more subjects are **Cho2017** (52 subjects) and **Stieger 2021** (about 60 subjects, many sessions each).
- Run the strongest arms (CSP, tangent space, ShallowConvNet, with and without EA) on each, plus cross-dataset transfer.
- **Project rule:** don't download OpenBMI or High-Gamma unless explicitly approved. Note that High-Gamma is motor *execution*, not imagery.

**WP6. Writing and release (about 3 weeks)**

- Write against the NERVE-ML checklist, and include the completed checklist as supplementary material.
- Release code, trained models, the experiment log and the pre-registration documents.
- Keep the forbidden comparisons out: pre-trained arms against the 0.650 ceiling; 4-class figures without the decomposition; any partial run.

### 7.3 Suggested order and timeline

| Period | Work |
|---|---|
| Now to 20 November 2026 | **Review-2:** submit the review paper; prepare the design-phase presentation (unrelated to this plan) |
| PW-2 month 1 | WP0 checks; WP1 prerequisites; WP2 pre-registration |
| Month 2 | WP2 audit runs; WP3 alignment in LOSO |
| Month 3 | WP3 alignment in pre-training and recipe ablations; WP4 foundation models start |
| Month 4 | WP4 complete; WP5 second dataset |
| Month 5 | WP5 third dataset and cross-dataset transfer; analysis |
| Month 6 | WP6 writing, internal review, submission |

The order matters. WP1 and WP2 reuse existing data and protect every later claim. WP3 is the cheapest route to a genuine accuracy gain. WP4 and WP5 carry the most technical risk.

### 7.4 What not to do

- **Don't chase a "superior architecture" as the headline.** Any architectural change must be one ablated component inside WP3–WP5.
- **Don't compare** any pre-trained arm with the 0.650 within-session ceiling, CBraMod's 0.6417 with the project's binary results, or any 4-class figure without its run-family breakdown.
- **Don't evaluate LaBraM, EEGPT or AFPM on EEGMMIDB** without stating their pre-training exposure.
- **Don't bundle improvements.** Add one component at a time, pre-register it, and report nulls.
- **Don't tune on test results**, and don't quote partial runs.

### 7.5 Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| An audit item turns out to be already published | Medium | WP0 novelty search first; frame contributions around what is verified open (run-family confound under band-pass, the MOABB exposure, contamination measured by fine-tuning) |
| EA gives no gain on the pre-trained arm (absorbed, like domain-specific batch norm) | Medium | Report it honestly. The LOSO gain is the main target, and a well-analysed null is still a contribution |
| Foundation models don't fit the 8 GB GPU | Medium | WP0 check; reduce input length or batch size; frozen-backbone fallback, disclosed |
| Host memory crashes during long runs | High (happened twice) | Runs are resumable, retry automatically and release memory per group; avoid running other heavy processes alongside |
| Second-dataset pipeline takes longer than planned | Medium | Start with BCI IV-2a, which MOABB supports, before larger datasets |
| Reviewers ask for more datasets | Medium | Plan for 3 datasets; precedents show 2–4 is accepted |

## 8. Open questions

1. **Novelty:** has any paper measured foundation-model contamination by fine-tuning, re-evaluated GCNs-Net or other graph models under correct splits, or named train-mode BatchNorm or pooled-data graphs as leakage paths?
2. **Alignment after pre-training:** does EA add accuracy once a network pre-trained on about 92 subjects is fully fine-tuned per subject, and which remedy works best for the common-average-reference rank deficiency?
3. **Size of the confound:** how large is the EEGMMIDB run-family confound under the 8–30 Hz band-pass, and how much of MOABB's 0.94 right-hand-vs-feet AUC survives run-disjoint evaluation?
4. **Feasibility:** can CBraMod and LaBraM be fine-tuned within 8 GB of VRAM and 16 GB of RAM, and which second and third datasets are licensed, small enough to download, and compatible with the project's pipeline?
5. **Confounds in this project's own models:** do the 0.67–0.73 results hold when the first second after the cue is excluded?

## 9. Work package 0 results (7 November 2026)

Work package 0 (Section 7.2) ran as three targeted checks against primary sources. Full notes are kept outside the repository. The verdicts below **supersede the "Unchecked" entries in Section 4** and answer open questions 1 and 4.

### 9.1 Novelty

| Contribution | Verdict | Evidence |
|---|---|---|
| (a) GCNs-Net re-evaluated under correct splits | **Open** for GCNs-Net and its public code (medium–high confidence). The general "collapse under a subject-disjoint split" pattern on PhysioNet is **partly pre-empted** | None of GCNs-Net's 214 citing papers (OpenAlex) re-evaluates it. Two 2026 papers show the general pattern for other models: Lv et al., BIOENG'26 (Zenodo 21970353), a functional-connectivity GCN at 4-class LOSO 48.68 %, kappa 0.004; and Chang et al., *Brain Sciences* 16(8):873, a segment-vs-subject gap for a non-graph model. Cite both |
| (b) Train-mode BatchNorm and pooled-data graphs as leakage paths | **Open** (medium–high) | Nothing in EEG names either. Generic precedents: Kinahan et al., FAccT 2024 (leakage through feature and channel selection, on EEGBCI); Kapoor & Narayanan 2023. EEG test-time-adaptation papers use test-batch statistics *on purpose*, so frame (b) as **undeclared** use of test data |
| (c) Foundation-model pre-training contamination | **Documenting the overlap: pre-empted. Measuring the inflation in a controlled way: open** (medium–high) | EEG-FM-Bench (ICML 2026), HEAR, NeuralBench and EEG-FM-Compass flag and exclude overlap. NeuralBench reports "no clear trend" of leakage helping. An unreviewed GitHub repository (YihengLi-1) attempts LaBraM vs EEGNet on PhysioNet. EEG-FM-Bench lists LaBraM as overlapping only SEED, but LaBraM's own appendix lists EEGMMIDB |
| (d) EEGMMIDB run-family confound | **Open** (medium) | No paper found; the PhysioNet page only documents the run-to-label mapping |

**Read before submission:** Liu, SSRN 7197508 (leakage-audited PhysioNet LOSO with test-time alignment), and Lv & Akbari, Preprints.org 202604.1922.

### 9.2 Foundation-model feasibility

GPU-memory figures are analytic estimates, not measurements.

| | CBraMod | LaBraM |
|---|---|---|
| Verdict | **Feasible** | **Feasible with changes** |
| Code and weights | Official repo MIT; weights 19.8 MB on Hugging Face (Apache-2.0) | Official stack pins torch 2.0.1, deepspeed 0.4.0, mne 1.4.2 and NCCL; won't run on this Windows / Python 3.14 setup. Use braindecode 2.0.0 instead |
| Pre-training data | TUH only; **no EEGMMIDB exposure** | **Includes EEGMMIDB**, apparently all 14 runs of all 109 subjects (inferred); a GitHub issue asking which subjects is unanswered |
| Size | Backbone about 4.0 M parameters. The official PhysioNet-MI head adds about 41 M, which would overfit; use a pooled or one-layer head | Base only, 5.8 M parameters. All 64 EEGMMIDB channel names are in its vocabulary |
| Input | Resample to 200 Hz, 1-s patches, 64-channel input, µV/100. Channel order matters | Channel-name based |
| Memory (batch 64, fp32) | About 2–4 GB (estimate) | About 6–7 GB, tight. Use batch 32; batch 64 would need bf16 mixed precision, which needs approval |
| Warnings | Its PhysioNet recipe does not exclude the project's six bad subjects. Checkpoint `oraculumai/CBraMod-MI-CoreML` was fine-tuned on all 109 PhysioNet subjects: **do not use** | braindecode would install a torchaudio version that does not match torch 2.12.1 (untested) |

### 9.3 Second and third datasets

| | BNCI2014-001 (BCI IV-2a) | Cho2017 |
|---|---|---|
| Size | About 0.8 GB (measured) | 10.7 GB (measured) |
| Licence | CC BY-ND 4.0, no registration | CC0 per DataCite (MOABB metadata says CC-BY-4.0; credit as CC-BY) |
| Channels | 22 EEG at 250 Hz, **all within EEGMMIDB's 64**, including C3, Cz and C4 | 64 at 512 Hz, 62 shared with EEGMMIDB (missing P9, P10) |
| Classes | Left hand, right hand, **feet**, tongue | Left and right hand only |
| Subjects | 9 | 52 |
| Notes | The only candidate with a feet class | Open MOABB issue #1340: a class-dependent pre-cue baseline when epochs start before the cue with a low high-pass. **Unaffected** with 8–30 Hz and a 0–3 s window |

**Alternative:** Dreyer2023, with 87 subjects, CC-BY, 5.5 GB zipped, but only 27 channels. **Rejected:** Stieger2021 (377 GB) and Weibo2014 (10 subjects). All of these are in MOABB 1.7.2.

**Disk:** drive D: has about 22 GB free. IV-2a plus Cho2017 would use about 11.5 GB.

### 9.4 Euclidean alignment under the common average reference

- **No published work addresses the problem.** After common average referencing, the 64-channel covariance has rank 63, and no paper covers EA in that case. The original He & Wu code uses a plain inverse square root with no regularisation. MOABB 1.7.2 shrinks each trial covariance with Ledoit–Wolf, and pyRiemann's `invsqrtm` has no eigenvalue floor.
- **The plain inverse square root is unsafe here.** In a synthetic float32 test it amplified rounding noise in the null direction to 3.2 times the signal power.
- **Adopted remedy: a pseudo-inverse square root.** It keeps eigenvalues above a small relative tolerance (1e-6 to 1e-10), so it whitens within the rank-63 subspace and keeps all 64 channels. This follows the rank-aware whitening guidance in MNE's `compute_rank` documentation and EEGLAB's preprocessing guidance. The fallback is MOABB's Ledoit–Wolf shrinkage.
- **Placement: band-pass first, then EA**, as in the original code and MOABB's example. The contrary claim (Wu 2025) lacks the necessary control, and its reported average (66.33 %) does not match its own per-subject values (69.47 %).

### 9.5 What this changes

- **All four audit contributions are publishable as novel**, with the precise framing above: GCNs-Net-specific; undeclared use of test statistics; controlled quantification of contamination rather than documentation of overlap; and the EEGMMIDB-specific run-family confound.
- **Phase A needs no new dependencies or downloads**, because it uses the existing cache and pipeline:
    - WP1: 4-class LOSO with the family decomposition, and the window-position check;
    - WP2: the audit experiments;
    - WP3: EA (implemented in-house with the pseudo-inverse) in LOSO and pre-training.
- **Phase B needs three decisions:**
    - new dependencies: MOABB for the datasets, and braindecode for LaBraM (with the torchaudio risk);
    - about 11.5 GB of downloads;
    - for LaBraM, batch 32 in fp32, or approval for bf16 mixed precision.

    It covers WP4 (foundation models) and WP5 (datasets).

## References

1. M. Wimpff et al., "EEG motor imagery decoding: A framework for comparative analysis with channel attention mechanisms," *J. Neural Eng.* 21(3):036020, 2024. arXiv:2310.11198. https://arxiv.org/abs/2310.11198
2. AFPM, foundation model pre-trained on motor-imagery/execution datasets including PhysionetMI (preprint). https://arxiv.org/html/2507.11911v1
3. EEG-FM-Compass: 12 EEG foundation models vs 8 specialist models across 13 datasets (arXiv:2601.17883; publication status unconfirmed). https://arxiv.org/abs/2601.17883
4. S. Chevallier et al., "The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark," arXiv:2404.15319 (v1). https://arxiv.org/abs/2404.15319
5. X. Vasques, P. Barbaste, O. Oullier, "Subject-level heterogeneity in EEG motor imagery decoding: a large-scale benchmark and portfolio-based reduction of the search space," arXiv:2606.24394, 2026. https://papers.cool/arxiv/2606.24394
6. Kontras et al., NeuroAtlas (frozen-backbone evaluation of EEG foundation models), arXiv:2605.14698 v2, 2026 (preprint). https://www.alphaxiv.org/abs/2605.14698.md
7. Zhou & Roy, validation-locked audit of foundation models on BCI IV-2a, arXiv:2609.23924, 2026 (preprint). https://papers.cool/arxiv/2609.23924
8. W.-B. Jiang, L.-M. Zhao, B.-L. Lu, "Large brain model for learning generic representations with tremendous EEG data in BCI" (LaBraM), ICLR 2024; arXiv:2405.18765.
9. Meta FAIR NeuralBench, LaBraM model card. https://facebookresearch.github.io/neuroai/neuralbench/models/labram.html
10. CBraMod: a criss-cross brain foundation model for EEG decoding, ICLR 2025; arXiv:2412.07236. https://arxiv.org/html/2412.07236v3
11. Junqueira et al., Euclidean alignment for cross-subject deep-learning decoding, *J. Neural Eng.* 21(3):036038, 2024; arXiv:2401.10746. https://arxiv.org/html/2401.10746v4
12. Wimpff et al., online alignment and test-time adaptation for cross-subject MI decoding, BCI conference 2024; arXiv:2311.18520. https://arxiv.org/html/2311.18520v2
13. Wimpff et al., pre-trained source model with online EA and AdaBN, arXiv:2502.06828, 2025 (preprint). https://arxiv.org/html/2502.06828v1
14. C. Rommel, J. Paillard, T. Moreau, A. Gramfort, "Data augmentation for learning predictive models on EEG: a systematic comparison," *J. Neural Eng.* 2022, doi:10.1088/1741-2552/aca220; arXiv:2206.14483.
15. P. Guetschel, M. Tangermann, cross-dataset motor-imagery embedding benchmark, arXiv:2311.16109 (preprint). https://arxiv.org/abs/2311.16109
16. Del Pup et al., nested subject-based cross-validation for EEG deep learning, arXiv:2505.13021, 2025 (preprint). https://arxiv.org/abs/2505.13021
17. R. Li et al., "The perils and pitfalls of block design for EEG classification experiments," *IEEE TPAMI* 43(1):316–333, 2021, doi:10.1109/TPAMI.2020.2973153. https://arxiv.org/html/1812.07697
18. G. Brookshire et al., "Data leakage in deep learning studies of translational EEG," *Frontiers in Neuroscience* 18:1373515, 2024. https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1373515/full
19. D. Carlson, R. Chavarriaga, Y. Liu, F. Lotte, B.-L. Lu, NERVE-ML checklist for machine learning in neural engineering, *J. Neural Eng.* 22(2):021002, 2025, doi:10.1088/1741-2552/adbfbd. https://pmc.ncbi.nlm.nih.gov/articles/PMC11948487/

*Reference details are as reported by the verified sources. Re-check titles, author lists and publication status against the publisher before citing any of them in a manuscript; several are preprints that may since have changed.*
