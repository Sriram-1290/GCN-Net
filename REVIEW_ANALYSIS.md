# One-Page Review-Paper Analysis

**Project:** GBP-Net: A Graph Band-Power Network with Cross-Subject Pre-Training for EEG Motor Imagery Decoding · **Batch No.** 22 · **Guide:** Dr. Gagan Kumar

| Item | Details |
|---|---|
| **Title of published review paper** | Graph Neural Network-Based EEG Classification: A Survey |
| **Authors** | Dominik Klepl, Min Wu, Fei He |
| **Journal / conference** | *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 32, pp. 493–503. DOI: 10.1109/TNSRE.2024.3355750 |
| **Publication year** | 2024 |
| **Main review topic** | Graph neural network (GNN) models for classifying EEG — emotion recognition, epilepsy, motor imagery, sleep staging and neurological disorders — organised by the architectural module each paper varies rather than by application |
| **Number of papers reviewed** | 63 |
| **Databases used for searching** | Google Scholar and ScienceDirect, queried on 1 November 2022; both peer-reviewed articles and preprints included |
| **Search keywords** | ("Graph neural network" OR "Graph convolutional network") AND ("Electroencephalography" OR "EEG"), screened by title and abstract |
| **Classification of reviewed studies** | Six categories following the GNN pipeline: (1) brain-graph definition — structural/distance-based vs functional, predefined vs learnable; (2) node features — time, frequency or graph domain; (3) node-feature preprocessing; (4) graph-convolution type — spatial, spectral or spatio-temporal; (5) node pooling; (6) formation of the graph embedding |
| **Important comparison tables** | Table I: brain-graph construction methods; Table II: node-feature definitions; Table III: node-feature preprocessing; Table IV: graph-convolution layers; Table V: node pooling; Table VI: graph-embedding methods. Figure 3 gives the distribution of tasks — emotion recognition dominates; motor imagery accounts for only about eight papers |
| **Main findings** | Chebyshev spectral convolution (ChebConv) is used by most surveyed papers, often with no stated reason for choosing it over a spatial layer. Raw EEG and differential entropy are the most common node features. The least innovation appears in the choice of graph-convolution layer. Distance-based graphs are noise-free but short-range; functional graphs capture long-range coupling but are noise-sensitive |
| **Research gaps identified** | Models do not transfer to datasets with a different number of electrodes, because ChebConv and node-concatenation readouts fix the node count; transfer-learning experiments are largely absent; frequency-band and cross-frequency information is under-used; no study compares graph-convolution layer types systematically |
| **Future research directions** | A controlled comparison of spatial, spectral and attention-based graph layers; pre-training on large open EEG datasets and transferring to small ones; cross-frequency coupling as graph structure; pooling and readout that tolerate variable electrode counts |
| **Reference style used** | Numbered citations in order of first appearance, IEEE style in the published version |
| **How this paper is useful for our project** | **(1)** It confirms that ChebConv — the operator in our baseline, GCNs-Net — is the field's default and is rarely justified, which motivates our controlled comparison of a graph-constrained spatial operator against an unconstrained one on identical folds. **(2)** Its call for transfer learning is exactly what GBP-Net's cross-subject pre-training implements. **(3)** Its module-by-module taxonomy shapes the classification scheme of our own review. **(4)** It analyses *architectures* but never examines *evaluation protocols* — how samples are defined or how data are split — which is the gap our review adds. The authors also state their review is not systematic and omits explainability |
