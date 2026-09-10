# LITERATURE SURVEY REPORT

**ON**

## Reassessing Graph Convolutional Networks for EEG Motor Imagery Decoding: Evaluation Protocol, Architecture, and the Limits of a Benchmark

**BACHELOR OF ENGINEERING IN COMPUTER SCIENCE**

Submitted by

| Student name | Roll number |
|---|---|
| Saligram Sriram Shravan | `<2453-23-753-0__>` |
| `<STUDENT NAME>` | `<2453-23-753-0__>` |
| `<STUDENT NAME>` | `<2453-23-753-0__>` |

Under the guidance of
**`<GUIDE NAME>`**, `<DESIGNATION>`

Department of Computer Science
**NEIL GOGTE INSTITUTE OF TECHNOLOGY**
Kachavanisingaram Village, Hyderabad, Telangana 500058.

---

## 1. Literature Survey Table

Twenty-seven papers, all directly related to the project domain: EEG motor imagery decoding,
graph neural networks for EEG, classical spatial-filtering baselines, cross-subject transfer,
and evaluation methodology. Venue and year are as recorded by Crossref or arXiv.

| S.No | Author(s) | Year | Paper Title | Methodology / Algorithm | Dataset / Data Used | Key Findings | Limitations / Research Gap |
|---|---|---|---|---|---|---|---|
| 1 | Hou, Jia, Lun, Hao, Shi, Li, Zeng, Lv | 2024 | GCNs-Net: A Graph Convolutional Neural Network Approach for Decoding Time-Resolved EEG Motor Imagery Signals (*IEEE TNNLS* 35(6)) | Spectral GCN: Pearson-correlation adjacency → normalised Laplacian → Chebyshev filters (K=2) → Graclus coarsening | PhysioNet EEGMMIDB, 4-class | Reports 93.06 % subject-level and 88.57 % group-level accuracy | **The direct baseline of this project.** Each sample is a single 6.25 ms timepoint and the split is a shuffled permutation over timepoints, so trials appear on both sides; the sample definition and split are not analysed |
| 2 | Hou, Jia, Lun, Zhang, Chen, Wang, Lv | 2022 | Deep Feature Mining via the Attention-Based Bidirectional LSTM Graph Convolutional Neural Network | Attention + BiLSTM over a graph convolutional backbone | PhysioNet EEGMMIDB | Adds temporal modelling to the GCN lineage | Same protocol lineage as [1]; the evaluation split is inherited rather than re-examined |
| 3 | Schalk, McFarland, Hinterberger, Birbaumer, Wolpaw | 2004 | BCI2000: A General-Purpose Brain-Computer Interface System | Data-acquisition platform and paradigm definition | — | Defines the recording protocol behind EEGMMIDB | Not a decoding method; establishes the run structure that later causes a class-run confound |
| 4 | Deng, Xu, Huo, Hong, Liu | 2025 | Multi-level Hierarchical Dynamic Graph Convolutional Networks for Motor Imagery EEG Analysis (*Neurocomputing*) | Spatiotemporal attention + hierarchical global/local graph topologies by community clustering; multi-level wavelet blocks | BCI Competition datasets; PhysioNet | Learns multi-level dynamic inter-channel structure | **Read from abstract only — full text not accessible.** Evaluation protocol not verifiable by us |
| 5 | Meng, Yu, Zhang, Fang, Ma, She, Zhang | 2026 | SAMGCN: A Self-Adaptive Multilevel Graph Convolutional Network for Motor Imagery Classification (*Biomed. Signal Process. Control*) | Channel position encoding + self-attention graph embedding; multi-level GCN to mitigate over-smoothing | BCI Competition IV-2a, OpenBMI | Explicitly motivated by GCN over-smoothing | **Read from abstract only — full text not accessible.** Over-smoothing is asserted as the failure mode but not isolated against a non-graph control |
| 6 | Defferrard, Bresson, Vandergheynst | 2016 | Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (*NeurIPS*) | Chebyshev polynomial approximation of spectral graph convolution | MNIST, 20NEWS | The formulation every EEG spectral GCN inherits | Not EEG-specific; assumes a meaningful graph exists |
| 7 | Goldberger, Amaral, Glass, Hausdorff, Ivanov et al. | 2000 | PhysioBank, PhysioToolkit, and PhysioNet (*Circulation*) | Public physiological data resource | EEGMMIDB and others | Provides the 109-subject dataset | Resource paper; no decoding methodology |
| 8 | Kapoor, Narayanan | 2023 | Leakage and the Reproducibility Crisis in Machine-Learning-Based Science (*Patterns*) | Taxonomy of leakage; survey of affected fields | 294 papers across 17 fields | Data leakage is pervasive and typically produces inflated, irreproducible accuracy | Does not cover EEG/BCI specifically — **the gap this project fills for one BCI benchmark** |
| 9 | Dhillon, Guan, Kulis | 2007 | Weighted Graph Cuts without Eigenvectors: A Multilevel Approach (*IEEE TPAMI*) | Graclus multilevel graph clustering | Graph benchmarks | The coarsening used for graph pooling in [1] | Discrete and non-differentiable; pooling hierarchy is fixed before training |
| 10 | Ramoser, Müller-Gerking, Pfurtscheller | 2000 | Optimal Spatial Filtering of Single-Trial EEG During Imagined Hand Movement (*IEEE TRE*) | Common Spatial Patterns (CSP) | Motor imagery EEG | The standard classical spatial filter; unconstrained discriminative projection | Binary by construction; sensitive to noise and to covariance estimation with few trials |
| 11 | Blankertz, Tomioka, Lemm, Kawanabe, Müller | 2008 | Optimizing Spatial Filters for Robust EEG Single-Trial Analysis (*IEEE Signal Process. Mag.*) | CSP theory, regularisation, practical guidance | Motor imagery EEG | Establishes shrinkage and regularisation as necessary for small-sample CSP | Does not address deep architectures or cross-subject transfer |
| 12 | Ang, Chin, Zhang, Guan | 2008 | Filter Bank Common Spatial Pattern (FBCSP) in BCI Competition IV (*IJCNN*) | CSP applied per frequency sub-band with feature selection | BCI Competition IV | Multi-band decomposition materially improves on single-band CSP | Hand-designed bands; no learned representation |
| 13 | Pfurtscheller, Lopes da Silva | 1999 | Event-Related EEG/MEG Synchronization and Desynchronization: Basic Principles (*Clin. Neurophysiol.*) | Neurophysiological review of ERD/ERS | Human EEG/MEG | ERD/ERS in mu (8–13 Hz) and beta (14–30 Hz) develops over 1–2 s | Establishes that motor imagery is a band-power phenomenon on a multi-second timescale — a constraint most deep pipelines do not check against their window length |
| 14 | Lawhern, Solon, Waytowich, Gordon, Hung, Lance | 2018 | EEGNet: A Compact Convolutional Neural Network for EEG-Based BCIs (*J. Neural Eng.*) | Depthwise + separable temporal/spatial convolutions | Four BCI paradigms | A compact graph-free CNN generalising across paradigms | Designed for ~2–4 s windows; performance under very short windows is not characterised |
| 15 | Schirrmeister, Springenberg, Fiederer, Glasstetter et al. | 2017 | Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization (*Human Brain Mapping*) | ShallowConvNet / DeepConvNet: temporal conv → spatial conv → square → mean → log | High-Gamma, BCI Competition IV-2a | Shallow spatial-then-power CNN matches or beats FBCSP | Same 2–4 s design regime assumption; no graph structure considered |
| 16 | Chevallier, Carrara, Aristimunha, Guetschel, Sedlar, Lopes et al. | 2024 | The Largest EEG-Based BCI Reproducibility Study for Open Science: The MOABB Benchmark | Standardised cross-dataset benchmarking of many pipelines | 36 datasets | Within-session, CNNs and Riemannian methods are statistically indistinguishable | Standardises comparison but does not audit individual published protocols for leakage |
| 17 | Vasques, Barbaste, Oullier | 2026 | Subject-Level Heterogeneity in EEG Motor Imagery Decoding: A Large-Scale Benchmark and Portfolio-Based Reduction of the Search Space | 216,714 pipeline configurations; portfolio selection | PhysionetMI, Cho2017, Zhou2016 | On PhysionetMI the best single pipeline reaches 0.650; covariance tangent-space and CSP lead; 93 distinct winners across 109 subjects | Establishes a realistic ceiling but does not test graph-based deep models |
| 18 | Blankertz, Sannelli, Halder, Hammer, Kübler, Müller, Curio, Dickhaus | 2010 | Neurophysiological Predictor of SMR-Based BCI Performance (*NeuroImage*) | Resting mu-rhythm prominence above a fitted 1/f background | Resting EEG + BCI performance | ~37 % of between-subject performance variance is predictable before any modelling | Predictor is correlational; does not prescribe an architectural remedy |
| 19 | Xie, Wang, Meng, Yue, Meng, Yi, Jung, Xu, Ming | 2023 | Cross-Dataset Transfer Learning for Motor Imagery Signal Classification via Multi-Task Learning and Pre-Training (*J. Neural Eng.*) | Multi-task pre-training across datasets, then fine-tuning | Multiple MI datasets | Pre-training improves accuracy and reduces between-subject variance | Does not isolate pre-training from the representation changes accompanying it |
| 20 | Jiang, Meng, Chen, Xu, Wu | 2024 | CSP-Net: Common Spatial Pattern Empowered Neural Networks for EEG-Based Motor Imagery Classification | CSP filters used to initialise (or replace) a CNN spatial layer | Four public MI datasets | +6.0 points average over a ShallowConvNet backbone; ablation shows the *initialisation*, not the extra capacity, drives it | Not evaluated on EEGMMIDB; transfer of the gain to other datasets untested |
| 21 | Kobler, Hirayama, Zhao, Kawanabe | 2022 | SPD Domain-Specific Batch Normalization to Crack Interpretable Unsupervised Domain Adaptation in EEG (*NeurIPS*) | TSMNet: SPD-manifold features + domain-specific batch normalisation | 5 datasets, 603 sessions, 158 subjects | Domain-specific normalisation worth 2.4–3.9 %; SPD manifold 3–4.5 % | Gains measured in inter-session / inter-subject **transfer**, where the target is unlabelled; behaviour with a labelled target is not characterised |
| 22 | Rommel, Paillard, Moreau, Gramfort | 2022 | Data Augmentation for Learning Predictive Models on EEG: A Systematic Comparison (*J. Neural Eng.*) | 13 augmentations compared across time, frequency and spatial domains | BCI IV-2a, Sleep-Physionet | Fourier-transform surrogate is the strongest for motor imagery (up to 45 % relative gain in low-data regimes); channel shuffling and hemispheric symmetry are harmful for lateralised MI | Augmentations evaluated against un-pretrained baselines only; composition with transfer learning untested |
| 23 | He, Wu | 2020 | Transfer Learning for Brain-Computer Interfaces: A Euclidean Space Data Alignment Approach (*IEEE TBME*) | Whiten each subject's trials by the inverse square root of their mean covariance | MI and ERP datasets | Unsupervised, label-free alignment improving cross-subject transfer | Aligns second-order statistics only; interaction with deep pre-training not examined |
| 24 | Wu | 2025 | Revisiting Euclidean Alignment for Transfer Learning in EEG-Based Brain-Computer Interfaces (*J. Neural Eng.*) | Review of EA usage, correct application, and extensions | 13 BCI paradigms | Consolidates EA as a low-cost standard preprocessing step | A review; contributes no new controlled comparison |
| 25 | Eder, Xu, Grosse-Wentrup | 2024 | Benchmarking Brain–Computer Interface Algorithms: Riemannian Approaches vs Convolutional Neural Networks (*J. Neural Eng.*) | Head-to-head comparison of Riemannian pipelines and CNNs | Multiple MI datasets | No significant difference within-session between CNNs and Riemannian methods | Does not include graph-based models |
| 26 | Mane, Chew, Chua, Ang | 2021 | FBCNet: A Multi-View Convolutional Neural Network for Brain-Computer Interface | Filter bank → spatial convolution per band → variance layer | BCI Competition datasets | Recovers FBCSP's two missing ingredients — multi-view spectral input and variance-based aggregation — inside a CNN | Fixed hand-designed filter bank; no graph structure |
| 27 | Jiang, Zhao, Lu | 2024 | Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI (LaBraM, *ICLR*) | Vector-quantised neural spectrum prediction; transformer pre-trained on ~2,500 h of EEG | ~20 datasets | Large-scale self-supervised pre-training transfers across EEG tasks | Very large parameter count; recent evaluations question whether such models beat compact baselines on MI |

---

## 2. Comparative Literature Analysis

| Technique | No. of Papers | Advantages | Limitations |
|---|---|---|---|
| Graph convolutional networks (spectral / adaptive) | 5 — [1], [2], [4], [5], [6] | Encodes inter-channel structure explicitly; localised, parameter-efficient filters; adjacency can be learned | Requires a meaningful graph; prone to over-smoothing at depth; **none of the EEG papers isolates the graph's contribution against a graph-free control on identical folds** |
| Classical spatial filtering (CSP family) | 4 — [10], [11], [12], [26] | Strong, interpretable, effective with few trials; filter banks add spectral resolution | Binary by construction; hand-designed bands; covariance estimation degrades with short windows |
| Riemannian / covariance-manifold methods | 3 — [21], [25], [17] | Among the strongest families on MI benchmarks; principled geometry; robust to amplitude scaling | High-dimensional tangent vectors demand heavy regularisation with few trials |
| Graph-free CNNs | 3 — [14], [15], [26] | Spatial-then-power ordering matches the physics of CSP; compact; strong empirical baselines | Designed for 2–4 s windows; behaviour outside that regime is rarely characterised |
| Transfer learning, alignment, pre-training | 4 — [19], [21], [23], [24] | Attacks the real bottleneck — per-subject data scarcity; largely architecture-independent | Rarely separated from the representation changes accompanying it; controls often absent |
| Data augmentation | 1 — [22] | Cheap; large gains in low-data regimes; no extra subjects required | No single best strategy; several standard augmentations are actively harmful for lateralised MI |
| Large-scale / foundation models | 1 — [27] | Learns generic representations from very large unlabelled corpora | Heavy; advantage over compact baselines on MI is contested |
| Neurophysiological grounding | 2 — [13], [18] | Constrains what a decoder can plausibly extract; predicts per-subject ceiling | Correlational; prescribes no architecture |
| Benchmarks and evaluation methodology | 3 — [8], [16], [17] | Standardises comparison; documents leakage as a systemic risk | Benchmarks compare pipelines but do not audit individual published protocols |
| Datasets and infrastructure | 2 — [3], [7] | Enables reproducible, comparable work | The EEGMMIDB run structure introduces a class–run confound that most users do not control for |

*Papers appear in more than one row where they span techniques, so the counts sum to more
than 27.*

---

## 3. Research Gap

From the reviewed literature, four gaps are identified. Each is grounded in what the papers
above do and do not report, not asserted in the abstract.

**3.1 Reported accuracies on EEGMMIDB are not comparable, because the sample definition and
split are rarely stated or audited.** GCNs-Net [1] reports 93.06 %, while the largest
systematic benchmark of this dataset [17] finds that the *best of 216,714 configurations*
reaches 0.650 on the same data. A gap of that size is not a difference in modelling skill;
it indicates that the two numbers are measuring different quantities. Kapoor and Narayanan
[8] document exactly this pattern across seventeen fields but do not cover BCI. **No paper in
this survey re-evaluates a published EEG graph model under a corrected protocol.** That is the
first gap this project addresses.

**3.2 The dataset carries a structural confound that the literature does not control for.**
In EEGMMIDB, left/right fist occurs only in runs 4/8/12 and both-fists/both-feet only in runs
6/10/14 [3, 7]. Any run-level non-neural drift therefore separates two of the four classes for
free, so a four-class accuracy can rise without any motor imagery being decoded. None of the
reviewed papers reports a decomposition that would distinguish the two.

**3.3 The contribution of the electrode graph is asserted, not isolated.** The lineage moves
steadily toward adaptive and learnable adjacencies [1, 2, 4, 5], and SAMGCN [5] is explicitly
motivated by over-smoothing. But no paper compares a graph-constrained spatial operator
against an unconstrained one, or against a graph-free CNN, **on identical folds with every
other factor held fixed**. Without that control, an improvement attributed to a better graph
may simply be an improvement in the surrounding pipeline.

**3.4 Interventions reported to help are not validated on this dataset under a correct
protocol.** CSP-based initialisation [20], domain-specific normalisation [21], filter banks
[12, 26], augmentation [22] and alignment [23, 24] each report gains of roughly two to six
points — but on other datasets, other protocols, or against baselines that differ in more than
one respect. Whether any of them transfers to EEGMMIDB under a leakage-free evaluation is
untested.

### What this project does about it

The project reimplements GCNs-Net faithfully, reproduces its published protocol, then
re-evaluates the identical model under a trial-level split with the graph, normalisation and
all statistics fitted on the training split alone. It adds the within-family decomposition
that separates run-family identification from genuine decoding, and it evaluates the graph's
contribution against unconstrained and graph-free controls on identical folds. Every
comparison is paired on subject, with pre-registered predictions and decision rules fixed
before each run, and a complete per-fold experiment log recording the git commit, seed and
configuration behind every number reported.

---

## Notes for the submitting team

- **Entries [4] and [5] were read from abstracts and metadata only**; the full texts were not
  accessible. They are the two papers closest to this project's direction, and the survey
  says so rather than implying a deeper reading than took place.
- Author lists, years, venues and DOIs were verified against Crossref and the arXiv API
  rather than reconstructed from memory.
- Fields in `<angle brackets>` must be completed before submission: the two remaining student
  names and all three roll numbers, and the guide's name and designation.
