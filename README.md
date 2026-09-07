# GCNs-Net under honest evaluation

A PyTorch reimplementation and audit of **GCNs-Net** (Hou et al., *IEEE TNNLS* 35(6):7312–7323, 2024)
on the PhysioNet EEG Motor Movement/Imagery dataset, plus an attempt to build a version
that decodes motor imagery when evaluated correctly.

**Start here:** [`PAPER.md`](PAPER.md) is the full research paper — abstract, related
work, methods, results, discussion and references; the typeset version is
[`GCNs-Net-Reassessed.pdf`](GCNs-Net-Reassessed.pdf) (27 pages, rebuilt with
`scripts/build_paper_pdf.py`). [`RESULTS.md`](RESULTS.md) is the
shorter writeup: the findings in publication order with figures. [`project_documentation.md`](project_documentation.md) is the living
record: every measurement, decision and correction, plus full methods, a reproduction
recipe, and the ranked roadmap of what to do next (§17). [`CLAUDE.md`](CLAUDE.md) holds the project rules.

---

## The short version

The published protocol reports ~88–93 % on this dataset. Reimplementing it faithfully
reproduces that number. Evaluating the same model correctly does not.

| | |
|---|---|
| Published protocol, reproduced (103 subjects, 3 seeds) | **0.8975** [0.8794, 0.9147] |
| Correct trial-level split, 4-class | 0.3097 |
| …of which is actual motor-imagery decoding | **none** — within-family 0.5090 [0.4984, 0.5199], chance 0.5 |
| Best representation (within-family) | 0.5470 [0.5291, 0.5667] |
| **CSP+LDA on identical splits** | **0.6239** [0.5977, 0.6521] |
| **Temporal attention (confirmatory, H2)** | **0.6038** [0.5805, 0.6278] — +0.0600 over baseline, p<0.001 |
| Learnable adjacency (confirmatory, H1) | no effect: +0.0031, p=0.39 |
| **LOSO, subject-independent (H5)** | **0.6040** [0.5823, 0.6268], 88/103 above chance |
| **EEGNet, no graph (post-hoc)** | **0.6296** — indistinguishable from CSP, p=0.49 |
| **ShallowConvNet, no graph (post-hoc)** | **0.6725** — beats CSP by +0.0485, 94/103 above chance |
| **Reordered graph model, G3 (post-hoc)** | **0.6737** — beats CSP by +0.0389; ties ShallowConvNet, p=0.99 |
| **Tangent space, tuned classical (post-hoc)** | **0.6694** — beats CSP+LDA by +0.0454; ties the deep arms |
| CSP-initialised spatial operator (post-hoc) | 0.6686 — **−0.0008** against the graph-initialised control |
| **+ cross-subject pre-training, PTG (post-hoc)** | **0.7301** — +0.0557 over its own control, 98/103 above chance |
| **Same pre-training on ShallowConvNet (post-hoc)** | **0.7369** — tied with PTG; gains differ by −0.0073 |
| Pre-training + per-subject batch norm (post-hoc) | 0.7224 — **−0.0077** against PTG; no gain |

All rows except the post-hoc ones are 103 subjects x 3 seeds; the post-hoc arms are 103
subjects at one seed on identical folds. The paired gap between the
published and correct protocols is **+0.6096, 95 % CI [+0.5767, +0.6411]**.

Three findings, in order of how much they matter:

1. **The published number is produced by leakage.** Each sample is a single 6.25 ms
   timepoint and the split is `randperm` over timepoints, so every trial contributes rows
   to both train and test. Training accuracy reaches 99.85 % while validation sits at
   chance — the model memorises trial-specific voltages. Two further leaks compound it:
   BatchNorm is hardcoded to training mode, and the correlation graph is built over all
   data including test.
2. **What survives is an artefact, not decoding.** Classes 0/1 occur only in runs 4/8/12
   and classes 2/3 only in runs 6/10/14, so run-level drift separates the pairs for free.
   Under a correct split the model identifies the *run family* 12 points above chance and
   discriminates *within* a family at chance.
3. **A 25-year-old linear method beats every graph model we built.** CSP+LDA on the same
   subjects, folds and metrics wins on both binary problems. Neither explicit band power
   nor a learnable electrode graph closes that gap.
4. **Deep learning is not the problem.** Two graph-free CNNs on identical folds and the
   same ~45 trials per problem reach and pass CSP: EEGNet 0.6296 and ShallowConvNet 0.6725,
   the latter putting 94/103 subjects above chance. So sample size is not the explanation
   either. Freeing the spatial operator *inside* the graph model recovers only 16 % of the
   gap. Post-hoc; see §9.11.
5. **The graph was not the problem either — the architecture around it was.** Taking power
   *after* the spatial stage, with a learned temporal filter, one spatial layer and no node
   pooling, and the Chebyshev convolution left exactly as published, lifts the graph model
   to 0.6629 and past CSP by 0.0389 on 70 % of subjects. A hard constraint, a free operator
   and a Dirichlet graph prior then span 0.0108 and are indistinguishable — the graph is
   inert, not harmful. This overturns finding 3 and revises the 16 % above, which measured
   slack in a misordered network. Post-hoc; see §9.13.
6. **The largest single gain is data, not architecture.** Pre-training on other subjects —
   no subject ever pre-trained on their own data — adds 0.0557 over an otherwise identical
   control on 90 % of subjects, reaching 0.7301, and helps the weakest quartile twice as much
   as the strongest. Post-hoc; §9.15.
7. **And that gain is architecture-agnostic, by measurement.** Run unchanged on
   ShallowConvNet it adds 0.0483 and reaches 0.7369; the difference between the two gains is
   −0.0073 [−0.0204, +0.0056] on 47/103. The two pre-trained arms are **tied, not ranked**.
   This supports "this architecture *plus* pre-training reaches 0.7301", never "our
   architecture reaches 0.7301". Post-hoc; §9.17.
8. **~0.67 is a ceiling of the dataset, not of any model.** Four spatial priors on the same
   operator — hard graph, graph-init, Dirichlet penalty, CSP subspace — plus a graph-free
   CNN and a Riemannian tangent-space classifier with no network at all span **0.0108**.
   **Four interventions, each chosen because published work reported two to six points from
   it, each measured nothing here**: a CSP-initialised operator (−0.0008), a fourfold window
   change (best window is the 0.40 s we already had), per-subject batch normalisation
   (−0.0077), and the architecture-agnosticism control (−0.0073). Post-hoc; §9.17, §9.19,
   §9.21, §9.23.

## Reproducing it

```bash
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
pytest -q                                    # 337 tests
```

The raw dataset (3.4 GB, 109 subjects) is not in the repository. Download the PhysioNet
archive, extract it to `data/raw/eegmmidb/`, verify it, then build the caches:

```bash
cd data/raw/eegmmidb && sha256sum -c SHA256SUMS.txt      # must be 3052 OK
python scripts/build_cache.py                            # raw, CAR-referenced
python scripts/build_cache.py --config configs/bandpass_8_30.yaml
```

Then the two arms of the audit:

```bash
python scripts/run_leaky_replication.py --epochs 25 --lr 1e-2      # published protocol
python scripts/run_subject_dependent.py --folds 5 --epochs 25 --lr 1e-2   # correct
python scripts/run_family_analysis.py --tag <tag>                  # the decomposition
python scripts/run_csp_baseline.py                                 # classical baseline
```

Every run appends one row per fold to `runs/experiments.jsonl` with the git commit, seed,
full config and confusion matrix. That file is versioned, so any number in the
documentation can be traced to the run that produced it.

## Layout

```
configs/         default.yaml (raw) and bandpass_8_30.yaml; one cache each
data/            loader, epoching, splits, normalisation, band-power features
graphs/          PCC adjacency, Laplacian, Chebyshev basis, Graclus coarsening
models/          Chebyshev conv, GCNs-Net, learnable adjacency, temporal attention,
                 free spatial operator, EEGNet/ShallowConvNet, reordered graph block,
                 CSP prior, domain-specific batch norm
evaluation/      metrics, fold assembly, trainer, experiment log
visualization/   scalp topography, adjacency and confusion figures
scripts/         cache building, experiment runners, analysis
tests/           337 tests
```

## Things that will bite you

These cost real time to find. They are documented in full in `project_documentation.md`.

- **The class label is `f(run_number, T-code)`, never the T-code alone.** T1 means *left
  fist* in runs 4/8/12 and *both fists* in runs 6/10/14. Mapping `T1 -> class` silently
  merges two classes.
- **EDF files have 65 signals, not 64.** The 65th is `EDF Annotations`.
- **S090 is clean; S089 is the defective subject.** The commonly circulated exclusion list
  has these transposed. Verified against all 1,526 files.
- **The reference applies Common Average Reference**, not the dataset-wide mean centring it
  looks like. `mean(X, 1)` in `make_dataset.m` operates on a channels × time matrix. CAR is
  not leakage, and it matters — it triples the graph's neighbour/distant contrast.
- **The reference's shipped code does not build the paper's four classes.** It discards
  both-fists and substitutes resting-state windows from the baseline run. The paper's own
  99.42 % single-class accuracy is the tell.
- **Never quote a 4-class accuracy without the family decomposition.** It has twice
  improved while the improvement was entirely better run-family identification.
- **Never quote a partial run.** One arm read 0.5774 at 35 folds and 0.5418 completed.

## Status

**All five pre-registered hypotheses are resolved** (§9.9). Temporal attention works
(+0.0600, Holm-corrected p<0.001); the learnable adjacency does not (+0.0031); combining
them adds nothing over attention alone; CSP+LDA remains significantly ahead by 0.0263;
and LOSO reaches 0.6040 with 88/103 subjects above chance — *higher* than the
subject-dependent figure, because a LOSO fold trains on ~100x more data.

Nine post-hoc lines of work then went beyond the pre-registered set, each with its
predictions and decision rule fixed before any aggregate was read (§9.11, §9.13, §9.15,
§9.17, §9.19, §9.21, §9.23). Together they establish that neither sample size, nor the graph
formulation, nor any competing spatial prior, nor the window length is the limitation; that
~0.67 is a ceiling of the dataset under this protocol, reached independently by six methods
sharing almost no components; and that the only intervention which moves it is other
subjects' data, which works equally well with the graph removed. They are labelled
exploratory wherever they appear.

**Four of those nine refuted our own predictions**, including three interventions selected
from the literature as the most promising available at the time. Each was pre-registered with a decision
rule before it ran, which is the only reason they read as results rather than as narratives
fitted after the fact.

The audit is complete and reproducible.

`python scripts/confirmatory_results.py` prints that table and refuses to call anything
supported on development data.
