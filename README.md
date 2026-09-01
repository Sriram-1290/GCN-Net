# GCNs-Net under honest evaluation

A PyTorch reimplementation and audit of **GCNs-Net** (Hou et al., *IEEE TNNLS* 35(6):7312–7323, 2024)
on the PhysioNet EEG Motor Movement/Imagery dataset, plus an attempt to build a version
that decodes motor imagery when evaluated correctly.

**Start here:** [`PAPER.md`](PAPER.md) is the full research paper — abstract, related
work, methods, results, discussion and references; the typeset version is
[`GCNs-Net-Reassessed.pdf`](GCNs-Net-Reassessed.pdf) (12 pages, rebuilt with
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
| **CSP+LDA on identical splits** | **0.6217** |
| **Temporal attention (confirmatory, H2)** | **0.6038** [0.5805, 0.6278] — +0.0600 over baseline, p<0.001 |
| Learnable adjacency (confirmatory, H1) | no effect: +0.0031, p=0.39 |
| **LOSO, subject-independent (H5)** | **0.6040** [0.5823, 0.6268], 88/103 above chance |

All rows except the last two are 103 subjects x 3 seeds. The paired gap between the
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
3. **A 25-year-old linear method beats it.** CSP+LDA on the same subjects, folds and
   metrics wins on both binary problems and on 16 of 20 subjects. Neither giving the
   network explicit band power nor letting it learn the electrode graph closes that gap,
   which points at the graph-convolutional inductive bias itself rather than at any
   particular graph.

## Reproducing it

```bash
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
pytest -q                                    # 166 tests
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
models/          Chebyshev conv, GCNs-Net, learnable adjacency, temporal attention
evaluation/      metrics, fold assembly, trainer, experiment log
visualization/   scalp topography, adjacency and confusion figures
scripts/         cache building, experiment runners, analysis
tests/           166 tests
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

The audit is complete and reproducible. The remaining work — a learnable adjacency and
temporal attention — is **pre-registered** in `project_documentation.md` §10: five
hypotheses with fixed directions, Holm correction, and the 20 subjects used during
development demoted to a development set so all confirmatory tests run on the full 103.

`python scripts/confirmatory_results.py` prints that table and refuses to call anything
supported on development data.
