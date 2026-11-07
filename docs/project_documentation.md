# Project Documentation — GCNs-Net Improvement (EEG Motor Imagery)

Living record of what has been done, what was decided, and why. Append to this as
work proceeds; do not silently rewrite history — add dated entries to the changelog.

**Project:** improved graph convolutional network for EEG motor imagery decoding,
based on GCNs-Net (Hou et al., IEEE TNNLS 2022).
**Contribution:** dynamic/learnable adjacency + temporal attention, evaluated under
Leave-One-Subject-Out (LOSO), plus scalp-topography interpretability.
**Rules of engagement:** see `CLAUDE.md`.

---

## 1. Status

| Item | State | Date |
|---|---|---|
| Dataset acquired and verified | Done | 2026-08-31 |
| Environment / GPU verified | Done | 2026-08-31 |
| Dataset quality audit (all 109 subjects) | Done | 2026-08-31 |
| Reference implementation analysed | Done | 2026-08-31 |
| Data loader + epoching + cache | Done | 2026-08-31 |
| Splits (trial-level, LOSO, leaky) | Done | 2026-08-31 |
| PCC adjacency + Laplacian + Chebyshev basis | Done | 2026-08-31 |
| Graclus coarsening + binary-tree permutation | Done | 2026-08-31 |
| Git repository initialised | Done | 2026-08-31 |
| GCNs-Net model port (PyTorch) | Done | 2026-08-31 |
| Training loop, metrics, experiment log | Done | 2026-08-31 |
| Leaky replication of the published protocol | Done (20 subj) | 2026-08-31 |
| Correct-protocol subject-dependent baseline | Done (20 subj) | 2026-08-31 |
| Representation ablation (band-pass x window) | Done (20 subj) | 2026-08-31 |
| Classical baseline (CSP+LDA) — Gate A | Done (20 subj) | 2026-08-31 |
| Phase 1 anchor runs (103 subj x 3 seeds) | **Done** | 2026-09-01 |
| Phase 3a: band power as node features | Done (20 subj) | 2026-09-01 |
| Learnable adjacency implemented | Done | 2026-09-01 |
| Temporal attention implemented | Done | 2026-09-01 |
| LOSO runner | Done | 2026-09-01 |
| Interpretability figures | Done | 2026-09-01 |
| Holm correction + confirmatory results script | Done | 2026-09-01 |
| README + pinned requirements | Done | 2026-09-01 |
| Learnable-graph ablation (20 subj, exploratory) | Done | 2026-09-01 |
| Confirmatory runs (103 subj) | **Done — all five** | 2026-09-02 |
| Attention mechanism analysis | Done | 2026-09-02 |
| `CLAUDE.md` Expected Results rewritten | Done | 2026-09-02 |
| Confirmatory experimental work | **COMPLETE** | 2026-09-02 |
| LOSO evaluation (H5) | Done | 2026-09-02 |
| Contribution (learnable adjacency + temporal attention) | Done — attention supported, adjacency not | 2026-09-02 |
| Figures regenerated on confirmatory data | Done | 2026-09-02 |
| `RESULTS.md` | Done | 2026-09-02 |
| `PAPER.md` + PDF + artifact | Done — author list outstanding | 2026-09-02 |
| Literature review, ranked roadmap (17) | Done | 2026-09-02 |
| **A1** free spatial operator (post-hoc) | **Done — 16 % of the gap** | 2026-09-02 |
| **A2** EEGNet + ShallowConvNet (post-hoc) | **Done — both pass CSP** | 2026-09-02 |
| **E1** per-subject decomposition (12.6) | Done | 2026-09-02 |
| **E2** resting SMR predictor (12.7) | Done — ~30 % of variance | 2026-09-02 |
| Paper revised for A1/A2/E1/E2 | Done | 2026-09-02 |
| Documentation-consistency test suite | Done | 2026-09-02 |
| **Open — needs compute** | B5 window sweep, B1 Euclidean alignment, B3 CSD, B4 Dirichlet energy, D3 BCI IV-2a | — |
| **Open — needs institutional access** | Read [4] MHD-GCN and [5] SAMGCN; 8.1's novelty claim still rests on abstracts | — |
| **Open — needs the authors** | Author list and affiliations in `PAPER.md` | — |

---

## 2. Environment

Verified on 2026-08-31 by direct execution, not assumed.

| Component | Version / value |
|---|---|
| OS | Windows 11 Pro 26200 |
| Python | 3.14.6 |
| PyTorch | 2.12.1+cu132 |
| CUDA | 13.2 |
| GPU | NVIDIA GeForce RTX 5060, 8.55 GB |
| Compute capability | sm_120 (Blackwell) |
| `torch.cuda.is_available()` | True — device matmul confirmed |

`sm_120` is present in `torch.cuda.get_arch_list()`, so the Blackwell support concern
raised in `CLAUDE.md` is resolved.

**Installed:** numpy 2.4.4, scipy 1.18.0, scikit-learn 1.9.0, matplotlib 3.11.1,
pandas 3.0.5, torch 2.12.1.

**Also installed** (2026-08-31): `mne 1.12.1`, `h5py 3.16.0`, `torch_geometric 2.8.0.post1`,
`pyyaml 6.0.3`, `pytest`. PyG 2.8 is pure-Python, so no compiled `torch-scatter`
wheel is needed on Python 3.14.

```bash
pip install mne h5py torch_geometric pyyaml pytest
```

---

## 3. Dataset

### 3.1 Acquisition

Two acquisition attempts:

1. A recursive `wget` mirror into `physionet.org/` **stalled after ~8 hours** with only
   18 complete subjects (S001–S018) plus a truncated `S019R10.edf` — 594 MB of ~3.4 GB.
   Superseded; this tree is dead and can be deleted.
2. A user-supplied zip, `eeg-motor-movementimagery-dataset-1.0.0 (1).zip` (2.00 GB),
   which was complete.

### 3.2 Location and integrity

Extracted (stripping the archive's `files/` prefix) to:

```
data/raw/eegmmidb/
  S001/ ... S109/        14 .edf + 14 .edf.event each
  SHA256SUMS.txt  RECORDS  ANNOTATORS  wfdbcal
  64_channel_sharbrough.{png,pdf}
```

- **109 subjects x 14 runs = 1,526 EDF + 1,526 event files, 3.4 GB.**
- **All 3,052 files verified** against the bundled `SHA256SUMS.txt` — zero mismatches.
- Zero truncated files (header-declared size == on-disk size for every EDF).

### 3.3 File format facts (verified from headers, not assumed)

- **65 signals per file, not 64.** The 65th is `EDF Annotations` (80 samples/record vs
  160 for EEG). MNE drops it; a hand-rolled reader must not. Assert 64 after load.
- **160 Hz** on all EEG channels, except the bad subjects in §3.5.
- **Units are microvolts directly** — physical dimension `uV`, physical range ==
  digital range (+/-8092), so the EDF scale factor is exactly 1.0.
- **Channel labels carry trailing dots** in the Sharbrough montage: `Fc5.`, `C3..`,
  `Cz..`, `Iz..`. Normalise to standard 10-10 names before any montage lookup or
  topographic plot.
- **Run durations:** baselines (R01/R02) 60–61 s; task runs 123 s, 125 s, or 124 s
  depending on subject. This variation is normal — **epoch by annotation onset, never
  by fixed offsets into the run.**

### 3.4 Run to class mapping (confirmed against MNE `eegbci` documentation)

| Runs | Task | T1 | T2 |
|---|---|---|---|
| 1, 2 | baseline, eyes open / closed | — | — |
| 3, 7, 11 | execution: left vs right fist | left fist | right fist |
| **4, 8, 12** | **imagery: left vs right fist** | **left fist** | **right fist** |
| 5, 9, 13 | execution: hands vs feet | both fists | both feet |
| **6, 10, 14** | **imagery: hands vs feet** | **both fists** | **both feet** |

**Critical:** the class label is a function of `(run_number, T-code)`, never the T-code
alone. A loader that maps `T1 -> class` without consulting the run number silently
merges *left fist* with *both fists*. Assert on this.

T0 = rest, in all runs.

### 3.5 Data quality audit — all 109 subjects

Every EDF header and every imagery-run annotation channel was parsed. Findings:

| Subject | Rate | Anomaly | Hits imagery runs? |
|---|---|---|---|
| S088 | **128 Hz** (all task runs) | 38 events/run -> 114 trials | **Yes — all 6** |
| S092 | **128 Hz** (all task runs) | 38 events/run -> 114 trials | **Yes — all 6** |
| S100 | **128 Hz** (all task runs) | 24 events/run -> 72 trials | **Yes — all 6** |
| S104 | 160 Hz | R08 is 106 s, 26 events | Yes — R08 only (−4 trials) |
| S089 | 160 Hz | **R03 is 181 s** (vs 123 s) | No — R03 is *execution* |
| S106 | 160 Hz | **R05 is 37 s** (vs 125 s) | No — R05 is *execution* |
| S090 | 160 Hz | **none** | **No — nothing wrong** |

**Correction to the documented exclusion list.** `CLAUDE.md` originally listed
`S088, S090, S092, S100, S104, S106`. Empirically **S090 is clean** — all 14 runs at
160 Hz, 61/61/123x12 s, 30 annotations in every imagery run, a profile identical to its
neighbour S091. The subject that actually carries a defect is **S089** (a 181 s R03).
This is a transcription slip: **S089 was written as S090**. `CLAUDE.md` has been
corrected. Both spellings leave N=103, so no headline number changes, but the writeup
must name the right subject.

**Note on strictness.** For an *imagery-only* protocol, only **S088, S092, S100** are
genuinely unusable (wrong sampling rate *and* wrong trial structure). S089 and S106 are
defective only in execution runs we never load; S104 loses 4 trials from one run. The
6-subject exclusion is retained anyway for comparability with the paper and prior work
— but it is a *convention*, not a data-driven necessity, and should be described as
such when defending the work.

**External corroboration, found 2026-09-02.** The published curation of this dataset
— **[14]** Z. Shuqfa, A. Lakas and A. N. Belkacem, *Data in Brief* 54:110181, 2024
(verified against Crossref) — independently excludes **exactly S088, S089, S092, S100, S104 and S106**, citing
zero-length trials and short runs, and reports N = 103. That is a peer-reviewed,
independent confirmation of the S089-not-S090 correction made here from the EDF
header audit, arrived at by a different route. It converts the exclusion list from
our judgement call into the field's convention, which is a materially stronger
position in a viva. **Cite it in the paper's dataset section.**

### 3.6 Trial census

- **105 of 109 subjects have exactly 30 annotations in all 6 imagery runs**
  -> 45 T1 + 45 T2 = **90 motor trials per subject**, ~22–23 per class under scheme (a).
- The only deviations are the four subjects above. Median 90, min 72 (S100),
  max 114 (S088 / S092).

### 3.7 Disk

`data/raw/eegmmidb/` is 3.4 GB. Reclaimable at any time: the source zip (1.9 GB,
present twice — project root and `~/Downloads`) and the dead `physionet.org/` partial
mirror (594 MB). ~4.4 GB total. Not deleted — user's call.

---

## 4. Reference implementation — `SuperBruceJia/EEG-DL`

TensorFlow 1.13 / Python 3.6. We are porting to PyTorch.

### 4.1 What it actually is

`Models/Network/lib_for_GCN/` is **a fork of Michael Defferrard's `cnn_graph`**
(ChebNet, NIPS 2016) — `graph.py` and `coarsening.py` retain his exact function names
(`metis_one_level`, `compute_perm`, `perm_data`, `rescale_L`, `distance_scipy_spatial`).
The real port target is therefore a well-documented 2016 architecture, and PyG's
`ChebConv` implements the same operator.

Relevant files:

```
Models/main-GCN.py                        hyperparameters, adjacency load, coarsening
Models/Network/lib_for_GCN/GCN_Model.py   chebyshev5, mpool1, b1relu, fc, _inference
Models/Network/lib_for_GCN/graph.py       laplacian, rescale_L, adjacency (kNN)
Models/Network/lib_for_GCN/coarsening.py  Graclus coarsening, perm_data
Models/DatasetAPI/DataLoader.py           loader
Preprocess_EEG_Data/For-GCN-based-Models/make_dataset.m   MATLAB preprocessing
```

### 4.2 Architecture and hyperparameters (from `main-GCN.py`)

```
F = [16, 32, 64, 128, 256, 512]   6 graph-conv layers
K = [2, 2, 2, 2, 2, 2]            Chebyshev polynomial order 2
p = [2, 2, 2, 2, 2, 2]            6 max-pool layers, stride 2
M = [4]                           4-class output
coarsen(A, levels=5)  -> graphs of 64, 32, 16, 8, 4, 2 nodes
64 nodes -> 6 poolings -> 1 node x 512 features -> FC -> 4

num_epochs=100  batch_size=1024  learning_rate=1e-6
dropout=0.50    regularization=1e-3    decay_rate=1
```

Easy to miss on a skim: `b1relu` applies **softplus**, not ReLU, despite its name; and
`rescale_L` hardcodes `lmax=2` instead of computing the true lambda_max.

### 4.3 Three things NOT to port faithfully

1. **Each sample is a single timepoint.** `_inference` opens with
   `tf.expand_dims(x, 2)` -> `[N, 64, 1]`, so `Fin = 1`. A "sample" is one 64-channel
   voltage vector and **the model has no temporal dimension at all.** This is exactly
   the leakage trap described in `CLAUDE.md`. It also has a design consequence:
   *temporal attention cannot be bolted onto this architecture* — adding it requires
   redefining the input as `[N, 64, T]`, which changes the sample definition, the
   graph-conv input width, and the trial count. Our contribution modifies the base
   model's data model, not just its layer stack. State this explicitly in the writeup.

2. **BatchNorm is hardcoded to training mode.** `b1relu`, `b2relu` and `fc` all call
   `tf.layers.batch_normalization(..., training=True)` with no train/eval switch, so at
   test time it normalises using *test batch* statistics. With `batch_size=1024` of
   shuffled timepoints that is a second, independent leakage path stacked on the first.
   In PyTorch this is just `model.eval()` — but it means the repo's published numbers
   are not a trustworthy baseline.

   Note `main-GCN.py` sets `params['brelu'] = 'b2relu'`, not `b1relu`. The difference
   is the bias shape: `b2relu` uses `[1, M, F]` (one bias per vertex per filter),
   `b1relu` uses `[1, 1, F]` (one per filter). The port must use `b2relu`, which adds
   `M x F` bias parameters per conv layer rather than `F`.

3. **The adjacency is built over all data, including test.** `main-GCN.py` only does
   `pd.read_csv('Adjacency_Matrix.csv')`, and `adjacency()` in `graph.py` is
   Defferrard's kNN-Gaussian builder for *grid* graphs — but the PCC construction does
   exist, in `make_dataset.m`:

   ```matlab
   Pearson_matrix = corrcoef(Stack_Dataset);
   Absolute_Pearson_matrix = abs(Pearson_matrix);
   Adjacency_Matrix = Absolute_Pearson_matrix - eye(64, 64);
   ```

   Exactly `A = |P| - I`, matching `graphs/adjacency.py`. It is computed on the pooled
   `Stack_Dataset` before any split, so the graph carries test-set information — the
   leakage `CLAUDE.md` forbids by name. Our version takes only the split it is handed.

### 4.4 Preprocessing in the reference — read from the local checkout

Verified against `D:\projects\EEG-DL`, superseding an earlier reading of this section
that was based on a web summary and was wrong in one important respect (see the
2026-08-31 changelog entry).

**Filtering: none.** `Extract-Raw-Data-Into-Matlab-Files.py` defines
`butter_bandpass_filter` (5-30 Hz) and a 60 Hz `iirnotch`, but both call sites are
commented out. The shipped pipeline applies no filter.

**Re-referencing: yes — Common Average Reference.** `make_dataset.m` line 22 reads

```matlab
Stack_Dataset = Stack_Dataset - mean(Stack_Dataset, 1);
```

At that point `Stack_Dataset` is **64 channels x N timepoints**, so `mean(...,1)` is the
mean *across channels* at each timepoint. Subtracting it is Common Average Reference,
**not** dataset-wide mean centring. CAR uses no cross-sample statistics and is therefore
**not leakage**. Our config now sets `rereference: average` to match.

**Trial window.** `load_physionet_data` centres a 640-sample (4.0 s) window on the
annotated task period: `pad = (length - (t - start)) / 2` trims the ~4.1 s period by
about 8 samples each side. The module constants `MOVEMENT_START = 1*160` and
`MOVEMENT_END = 5*160` are **dead code** — declared, never referenced. Our `tmin = 0.0,
length_s = 4.0` is within ~50 ms of theirs.

**Split: timepoint-level and shuffled.** Confirmed:

```matlab
All_Data = [Stack_Dataset, Labels];       % rows are timepoints, cols 1:64 are channels
rowrank  = randperm(size(All_Data, 1));
training_set = All_Dataset(1:fix(row/10*9), 1:64);
```

Labels are replicated 640x per trial, so each row is one timepoint across 64 channels.
Shuffling then cutting 90/10 puts timepoints from the same trial on both sides.

**Scale.** The active configuration is **20 subjects** (`subject = range(1, 21)`); the
100-subject block is commented out. `NUM_TRIALS = 21 * num_classes = 84` per subject.

**Leakage paths in the reference: three, not four.**

1. Timepoint-level shuffled split (here).
2. BatchNorm on test-batch statistics (4.3.2).
3. Adjacency computed over all pooled data including test (4.3.3).

The earlier count of four wrongly included the CAR line. Three is still enough that the
published headline is not a trustworthy baseline, and `CLAUDE.md`'s ~90-93 % target is
very likely not reachable without them.

### 4.5 The paper itself (arXiv:2006.08924 / IEEE TNNLS 2022)

Read directly, because the repo could not answer these. Four things are now settled:

**1. The paper names four classes L / R / B / F — but the code does not build them.**
The paper says "Four MI tasks are termed as L (image left fist), R (image right fist),
B (image both fists), and F (image both feet)". The shipped extraction code builds
something else. In `load_physionet_data`:

```python
if (current_run - 6) % 4 != 0 or current_event[1] == 2:
    if (current_run - 6) % 4 == 0:
        current_event[1] = 3
```

For runs 6/10/14 only `action == 2` (both feet) is kept — **T1, both fists, is
discarded**. The fourth class is then filled from the *baseline eyes-open run* at random
offsets:

```python
path = base_path % (subject_id, BASELINE_RUN)   # run 1
offset = np.random.randint(0, signals.shape[0] - SAMPLES)
trials[i, :, :] = signals[offset: offset+SAMPLES, :]
labels[i, -1] = 1.
```

So the code's four classes are **left fist / right fist / both feet / resting-state**,
with the rest trials drawn as overlapping random windows from a single 60 s run. The
docstring agrees with the code, not the abstract: "num_classes: number of classes
(2, 3 or 4) for L/R, L/R/0, L/R/0/F".

This is corroborated by the paper's own per-class numbers (Sec. IV-C): 83.45 %, 86.72 %,
83.96 % and **99.42 %**. A near-perfect fourth class is what resting-state-vs-imagery
looks like; it is not what "both feet vs both fists" looks like. Those four average to
88.39 %, matching the reported 88.35 % group-level accuracy — so the easy class carries
roughly +3.6 points of the headline.

**Consequence: open decision 1 is reopened.** The paper claims a four-way motor-imagery
problem; the code solves a three-way MI problem plus an easy rest class.

**2. A sample is one timepoint.** "every time point is recognized as a sample to map the
brain state to the corresponding MI task" gives 640 samples per trial. Confirms
section 4.3.1 from the paper itself, not just from the code.

**3. The adjacency is A = |P| - I**, the absolute Pearson correlation matrix, computed
over "all the signals of participants" - the entire pooled dataset, test data included.
This is a **fourth leakage path**, and it is precisely the one CLAUDE.md names: "NEVER
compute the correlation/adjacency matrix using test data." Our implementation must build
the PCC from the training split only, which is a deliberate, documented deviation from
the paper.

**4. "Group level" (88.57 %) is NOT subject-independent.** It pools 20/50/100 subjects
and then splits 90/10 **across the pooled data**, so the same subjects appear in train
and test. That is subject-dependent training with more data, not LOSO.

Consequence: **there is no published LOSO baseline for GCNs-Net.** Neither headline
number (93.06 % subject level, 88.57 % group level) is subject-independent. CLAUDE.md's
expectation of 60-70 % under LOSO stands, and the gap to 93 % is not a failure to
reproduce - it is the difference between four leakage paths and none. This also
strengthens the contribution: a correct LOSO evaluation of this architecture appears to
be genuinely new.

**Leakage paths: three** — see the list in 4.4. (An earlier revision of this document
counted four, wrongly including the CAR line.)

**Paper/code hyperparameter mismatches**, worth noting before tuning anything: the paper
states Adam at learning rate 0.01 and weight decay 1e-2; `main-GCN.py` sets
`learning_rate = 1e-6` and `regularization = 1e-3`.

---

## 5. Port plan

### 5.1 Decision: hand-rolled dense Chebyshev conv, not `ChebConv`

With 64 nodes the graph is tiny, so dense `L @ x` is faster than sparse scatter, and
~20 lines reproduce `chebyshev5` exactly — including the `lmax=2` rescale, which
`ChebConv` computes differently by default. It is also unit-testable against a
hand-computed 4-node example, which matters for the viva. PyG is installed anyway as a
cross-check oracle.

### 5.2 Module layout

```
configs/        default.yaml — paths, runs, exclusions, class map, epoching, seed
data/           config.py, labels.py, edf_loader.py, cache.py  (+ splits, to come)
data/raw/       raw EDFs (gitignored payload, not code)
data/cache/     epoched .npz per subject + manifest.json (gitignored)
graphs/         pcc_adjacency, laplacian, coarsening (Graclus port)   [empty]
models/         cheb_conv, gcns_net baseline, dynamic_adj + temporal_attn  [empty]
evaluation/     train loop, LOSO runner, metrics, experiment log (jsonl) [empty]
visualization/  topomaps, confusion matrices, learned-adjacency plots [empty]
scripts/        build_cache.py
tests/          test_labels.py
```

`data/` is deliberately both a Python package and the home of the gitignored `raw/`
and `cache/` payload directories. They do not collide, and it keeps the layout
matching the module split named in `CLAUDE.md`.

### 5.3 Build order

1. Loader -> epoching -> disk cache
2. End-to-end loop with a **random** adjacency, trial-level split (proves the pipeline)
3. PCC adjacency (train split only)
4. Graclus coarsening port
5. Reproduce subject-dependent accuracy (target ~90–93 %)
6. LOSO evaluation (expect 60–70 % — this gap is the point, not a bug)
7. Contribution: dynamic adjacency, then temporal attention — **one at a time, each
   with an ablation**

The Graclus port is the fiddliest piece: the fake-node padding in `compute_perm`
(singletons get 1 fake node, childless singletons get 2, zero-filled so max-pool selects
the real node) must be exact or the pooling hierarchy silently misaligns.

### 5.4 Runtime validators (belt-and-braces over the exclusion list)

A hardcoded subject list is exactly the fragile thing that let the S089/S090 error
persist. The loader asserts independently:

```
sfreq == 160.0                per run
n_eeg_channels == 64          after dropping 'EDF Annotations'
n_annotations == 30           per imagery run
label = f(run_number, T-code) never the T-code alone
```

Any subject tripping these is dropped with a logged reason, so the empirical check and
the declared list must agree or the run stops.

---

## 6. What is built

### 6.1 Modules

| File | Purpose |
|---|---|
| `configs/default.yaml` | paths, imagery runs, exclusions, class scheme, epoching, preprocessing, seed |
| `data/config.py` | frozen dataclasses loaded from YAML; passed explicitly, no global singleton |
| `data/labels.py` | `(run, T-code) -> class`; raises rather than guessing |
| `data/edf_loader.py` | EDF read, channel standardisation, validation, manual epoching |
| `data/cache.py` | per-subject `.npz`, `manifest.json`, config-fingerprint guard |
| `scripts/build_cache.py` | CLI cache builder |
| `data/splits.py` | trial-level k-fold, LOSO with disjoint val subjects, and the leaky reference split |
| `data/normalize.py` | per-channel standardiser, fit on train only |
| `graphs/adjacency.py` | `A = \|P\| - I` from a given split; optional sparsification |
| `graphs/laplacian.py` | normalised Laplacian, Chebyshev rescaling, `T_0..T_{K-1}` basis |
| `graphs/coarsening.py` | Graclus matching, `compute_perm`, `perm_data`, `perm_adjacency` |
| `models/dynamic_adjacency.py` | learnable graph, torch Laplacian/Chebyshev, `DynamicChebConv` |
| `models/temporal_attention.py` | attention over the sample's time axis |
| `data/features.py` | log band power per channel per band (Hann-tapered rFFT) |
| `visualization/topography.py` | scalp figures, adjacency heatmaps, confusion with family blocks |
| `scripts/run_loso.py` | LOSO runner with a per-fold cost guard |
| `scripts/run_csp_baseline.py` | CSP+LDA classical baseline |
| `scripts/confirmatory_results.py` | the pre-registered table, Holm-corrected |
| `scripts/make_figures.py` | writes `figures/` |
| `README.md`, `requirements.txt` | reproduction instructions and pinned versions |
| `models/cheb_conv.py` | dense `chebyshev5` equivalent, `b2relu`, `mpool1` |
| `models/gcns_net.py` | the six-layer stack, `GCNsNetConfig`, `build_bases` |
| `tests/` | 86 tests, all passing |

### 6.2 Design decisions and why

- **Manual epoching by annotation onset**, not `mne.Epochs`. Run lengths vary
  (123/124/125 s) so fixed offsets are wrong, and `mne.Epochs` applies baseline
  correction by default, which at `tmin=0` would subtract a degenerate one-sample
  baseline.
- **`eegbci.standardize()`** normalises the dotted Sharbrough labels to standard
  names (`Fc5.` -> `FC5`, `Cz..` -> `CZ`/`Cz`), required before montage lookup.
- **MNE returns volts; the cache stores microvolts** (x1e6), the units the data was
  recorded in.
- **No normalisation in the cache.** Mean/std are fit on the training split at
  training time. The reference mean-centres the whole dataset pre-split (§4.4).
- **`bandpass: null` by default**, reproducing the reference's complete absence of
  filtering. Changing it changes the config fingerprint, which invalidates the cache.
- **Config fingerprint guard.** `assert_cache_matches()` refuses to run against a
  cache built under different preprocessing — the mechanism enforcing CLAUDE.md's
  "don't silently change preprocessing between experiments".
- **Validation aborts, never coerces.** Wrong sampling rate, wrong channel count,
  wrong annotation count, or an epoch running off the end of a recording all raise.

### 6.3 Cache built 2026-08-31

```
103 subjects, 9,270 trials, 0 dropped, 59.4 s, 447 MB
every subject exactly 90 trials, every epoch exactly (64, 640)
class totals: left_fist 2336, right_fist 2299, both_fists 2319, both_feet 2316
```

Balanced to within 1.6 % across classes. Verified on read-back: correct channel
names, 15 trials per run x 6 runs, plausible microvolt range, zero duplicate
(run, onset) pairs.

**Not yet built:** training loop, evaluation/metrics, experiment log, visualisation.

### 6.4 Splits and graph construction

`data/splits.py` provides three split families, and the distinction between them is
the methodological core of the project:

- `stratified_trial_kfold` - subject-dependent, split at **trial** level. Correct.
- `loso_folds` - one fold per subject, with `n_val_subjects` drawn from the remaining
  subjects for checkpoint selection so model selection never sees the test subject.
  Per-fold seeds vary the validation draw; identical validation subjects across all
  folds would bias selection toward whatever those subjects happen to reward.
- `leaky_timepoint_holdout` - **deliberately wrong**, reproducing the reference
  protocol. `count_shared_trials()` measures the damage: with 640 timepoints per trial
  and a 10 % cut, **all 90 trials contribute to both sides**. This is what makes the
  replication arm a measurement rather than an assertion.

`data/normalize.py` deliberately offers no `fit_transform` over a whole dataset. The
missing convenience method is the point.

### 6.5 First observations from real data (S001, fold 0)

Verified end to end on cached EEG. Two findings worth carrying forward.

**1. The paper's graph is nearly complete, and barely a graph at all.**

```
A = |P| - I    range [0.000, 0.999]    mean edge weight 0.675
density (>0.01) 0.984
```

**Superseded in part — see the CAR correction below.** 98.4 % of all electrode pairs
carry a non-trivial edge. Spatial structure *is* present and in the right direction -

```
|PCC|(C3, C1) 0.973   (C3, C5) 0.912   (C3, C4) 0.851   (C3, Oz) 0.718   (C3, Fp1) 0.638
mean neighbours 0.950  vs  mean distant 0.645
```

- but compressed into a narrow band.

**Correction (2026-08-31, after reading the local checkout).** That measurement was
taken *without* re-referencing, which we then believed matched the reference. It does
not: the reference applies CAR (4.4). With CAR the graph is considerably more
structured:

| | no re-reference | CAR (the reference) |
|---|---|---|
| mean edge weight | 0.686 | **0.353** |
| density(>0.5) | 0.840 | **0.271** |
| p5..p95 spread | 0.408 .. 0.933 | **0.032 .. 0.748** |
| C3 neighbours vs distant | 0.950 vs 0.645 (**1.47x**) | 0.656 vs 0.148 (**4.45x**) |
| lambda_max | 1.027 | 1.091 |

So the "near-complete graph with little topology" claim was substantially an artefact of
the missing re-reference. With CAR the weight distribution is far more informative -
the neighbour/distant contrast triples. This **weakens** the motivation for a learnable
adjacency somewhat, and that should be stated honestly rather than quietly dropped: the
fixed PCC graph is a more reasonable baseline than the first measurement suggested. The
contribution now has to earn its keep against a real graph.

`threshold_adjacency()` remains available so sparsification can still be ablated.

**2. The hardcoded `lmax = 2` wastes half the Chebyshev domain.**

```
true lambda_max 1.0269   (the reference hardcodes 2.0)
rescaled spectrum with lmax=2.0  ->  [-1.0000, 0.0269]
```

`lmax = 2` is a valid upper bound for a normalised Laplacian, but the actual spectrum
here reaches only 1.03, so `L_tilde` occupies just over half of [-1, 1]. The Chebyshev
polynomials are therefore evaluated over a compressed range and lose expressive power.
`rescale_laplacian(L, lmax=None)` uses the computed lambda_max instead. This is a cheap,
well-motivated ablation and a possible free improvement over the baseline - and it is
independent of our headline contribution, so it must be ablated separately rather than
bundled in.

**3. Adjacency leakage is real but modest at subject level.** Train-only vs all-data
adjacency for S001: `max|diff| 0.051`, `mean|diff| 0.0086`. Small, but the graph is
shared across every layer, so the effect is systematic rather than noise. At group
level, where the paper pools subjects, the comparison will differ.

**Normalisation check.** After fitting on the training split only, train standardises to
exactly 0.000 / 1.000 while test lands at 0.056 / 0.903 - not forced to 0/1, which is
the observable signature that the statistics never saw it.

**LOSO shape.** 103 folds; each is 97 train / 5 validation / 1 test subject, i.e. 8,730
training trials against 90 test trials.

### 6.6 Graclus coarsening

Ported from Defferrard's `cnn_graph`. Repeated heavy-edge matching under the
normalised-cut criterion `w(i,j) * (1/d_i + 1/d_j)`, then `compute_perm` reorders every
layer so nodes `2i` and `2i+1` are the two children of coarse node `i` - after which
`mpool1` is literally a stride-2 max over the node axis.

**Deliberate bug fix.** The reference's `metis_one_level` computes row lengths with an
off-by-one: `rowlength[count]` is incremented before `count` advances, so every row but
the last is one too long and the neighbour scan reads the first entry of the *next* row,
considering a node that is not a neighbour. Row offsets here are computed directly from
the sorted row indices. Same algorithm, valid matching, slightly different clustering.

**Results on the real graph (S001, fold 0 training split).**

```
layer sizes  64 -> 32 -> 16 -> 8 -> 4 -> 2      fake nodes: 0
every level: a perfect matching, zero singletons
```

The dense PCC graph always leaves an unmarked neighbour available, so matching is
perfect at every level and the fake-node machinery is never exercised. Sparsify and it
is: `keep_top_k=8` (density 0.16) gives `64 -> 34 -> 18 -> 9 -> 5 -> 3` with singletons
at four of five levels. So the fake-node path matters for the sparsification ablation,
and it is tested directly.

**The coarsening recovers scalp topography without being told about electrode
positions.** Every level-0 cluster is a genuinely adjacent electrode pair:

```
AF7+F7 0.911   T7+T9 0.921   CP2+CP4 0.982   FC5+C5  0.924
CP1+CPz 0.975  AF4+AF8 0.916  P8+PO8 0.941   Oz+Iz   0.872
TP7+P7 0.928   FC2+FC4 0.969  Fp1+AF3 0.958  F1+Fz   0.980
```

This is a useful result in its own right: although the correlation graph is near-complete
(6.5), the *ordering* of its weights encodes spatial adjacency well enough that
normalised-cut matching reconstructs the montage. It supports the interpretability strand
of the project and gives the fixed-graph baseline a fair defence.

**lambda_max grows as the graph coarsens**, which refines the finding in 6.5:

```
layer   0      1      2      3      4      5
nodes  64     32     16      8      4      2
lmax  1.027  1.050  1.096  1.189  1.396  2.000
```

The reference's hardcoded `lmax = 2` is therefore worst at layer 0 - where the spectrum
occupies barely half of [-1, 1] - and exactly right at the coarsest layer, where a
two-node graph genuinely has `lambda_max = 2`. The wasted Chebyshev range is an
early-layer problem, not a uniform one.

### 6.7 Model port

`models/gcns_net.py` reproduces the reference architecture at
`F = [16,32,64,128,256,512]`, `K = 2`, `p = [2]x6`, `M = [4]`.

**Parameter count: 359,428** — matching the ~350K stated in `CLAUDE.md`. This is the
sharpest single check that the port is faithful, because it is sensitive to the
`b1relu`/`b2relu` distinction: `b2relu` sizes its bias `[1, M, F]` per vertex per filter,
and using `b1relu` by mistake would change the count by tens of thousands. There is a test
asserting the range.

```
layer     nodes in   Fin  Fout  nodes out     params
conv1           64     1    16         32      1,088
conv2           32    16    32         16      2,112
conv3           16    32    64          8      5,248
conv4            8    64   128          4     17,664
conv5            4   128   256          2     67,072
conv6            2   256   512          1    264,192
logits           1   512     4          -      2,052
```

Note `conv6` alone is 73 % of the parameters (`256 x 2 x 512`), on a two-node graph. A
half-megabyte of weights operating on two nodes is worth flagging as an efficiency
ablation later; it is not obviously earning its place.

**Sample definition.** One timepoint is one sample, `Fin = 1`, exactly as the reference
defines it (`tf.expand_dims(x, 2)`). We keep this deliberately: it makes the leaky-vs-correct
comparison exact, since the model and the sample definition are identical and only the
*split* differs. Trial-level predictions come from aggregating a trial's 640 timepoint
predictions, not from changing the architecture. `in_features` is a parameter because the
temporal-attention contribution will need a window.

**Deliberate deviations, all switchable:**

- BatchNorm respects `.eval()`. `always_train_bn=True` reproduces the reference's
  hardcoded `training=True` for the replication arm.
- `build_bases(..., lmax=None)` uses the true lambda_max instead of the hardcoded 2.
- `l2_penalty()` covers the fully-connected parameters only, matching the reference,
  where `regularization=True` is passed only inside `fc`.

**Discovered while porting: the reference's dropout is dead code.** `_inference` applies
dropout only between `self.M[:-1]` layers; with `M = [4]` that list is empty, so the
configured `dropout = 0.50` is never applied. Reproduced by default (`dropout=0.0`,
`fc_hidden=()`).

**Throughput** (RTX 5060, real data): 2.2 ms per 1024-sample batch, 54 MB VRAM. One
subject is 46,080 timepoint samples, so ~0.10 s per epoch forward-only. Confirms
`CLAUDE.md`'s expectation that VRAM and speed are not constraints.

---

### 6.8 Contribution components (implemented, not yet evaluated)

**Learnable adjacency** (`models/dynamic_adjacency.py`). `A = relu(A_pcc + delta)`,
symmetrised and zero-diagonal *by construction* — a normalised Laplacian is only defined
for a symmetric non-negative adjacency, so a soft penalty would permit invalid graphs
mid-training. `delta` initialises to zero, so the model starts **bit-for-bit identical to
the fixed baseline**; a test asserts both produce the same output at init. Without that
property the ablation would confound "learning the graph" with "a different starting
point". The coarsening hierarchy stays fixed: Graclus is discrete and non-differentiable,
and recomputing the pooling tree each step would change which electrodes are siblings
during training. Cost is **5,460 parameters against 359,428 (1.5 %)**, pinned by a test so
a gain cannot be attributed to capacity. The Laplacian, rescaling and Chebyshev recurrence
are reimplemented in torch and tested against the numpy versions the baseline uses.

*Measured cost:* the dynamic convolution recomputes the basis every forward pass at every
layer, which is unavoidable since the graph changes as `delta` updates. This runs at
**20.6 s/fold against 3.1 s/fold** for the fixed graph — partly GPU contention, partly a
real ~2-3x training cost that must be reported alongside any benefit.

**Temporal attention** (`models/temporal_attention.py`). Scores are computed from the whole
64-channel pattern at each instant and shared across nodes by default, pooling `T`
timepoints into `n_heads` summaries. **Honest limitation:** unlike the adjacency this is
*not* initialisation-equivalent to the baseline — the first conv sees `Fin = n_heads`
rather than `Fin = T`, and no parameter setting reproduces the fixed linear mix. The
comparison is "attention pooling versus linear mixing"; parameter counts move in both
directions (attention adds weights but shrinks conv1 from 64 to 4 input features) and both
are logged.

### 6.9 LOSO runner

`scripts/run_loso.py`. `loso_folds` had been written and tested but nothing called it, so
Phase 4 was blocked on a script that did not exist. Two things it must get right that the
subject-dependent runner did not: subjects are loaded **once** rather than re-read inside
each fold (~10,000 cache decompressions otherwise), and it **estimates per-fold sample
count and refuses infeasible configurations** instead of appearing to hang.

That guard produced a finding. LOSO is ~100x the subject-dependent protocol — ~8,700
training trials per fold instead of ~58 — so the reference's one-timepoint sample
definition needs **5.9M samples per fold** and would run for days across 103 folds.
Windowing brings it inside budget. **This is a third way the published sample definition
fails**: beyond decoding nothing under an honest split, it makes subject-independent
evaluation computationally impossible, which may be part of why no LOSO number exists in
the literature for this architecture.

### 6.10 Figures, statistics and reproducibility

**Figures** (`visualization/topography.py`, `scripts/make_figures.py`). Each makes a claim
checkable by eye. `montage_recovery.png` quantifies 6.6: **mean cluster edge 0.045 against
0.109 for a random pairing**, so clusters are ~2.4x more local than chance. The
random-pairing baseline is drawn on the figure so "short and local" is a number rather than
an impression, and the handful of long edges that do cross are visible rather than hidden.

**Statistics.** `bootstrap_p_value` (directional, re-centred null, never returns zero from
a finite resample) and `holm_bonferroni` (step-down). `scripts/confirmatory_results.py`
evaluates the five pre-registered hypotheses and labels anything run on fewer than 100
subjects **EXPLORATORY regardless of its p-value**, enforcing 10.1 in code rather than by
discipline.

*A test of ours was wrong rather than the code:* asserting a single null seed gives a large
p-value tests that seed's luck, not the estimator. Replaced with a calibration test over
200 nulls — **4.5 % fall below 0.05 against a nominal 5 %, median p 0.456**.

**Reproducibility.** `README.md` and `requirements.txt`. torch is called out separately
since the sm_120 build is not on PyPI. The README carries a "things that will bite you"
section listing every trap this project actually hit.

---

## 7. Open decisions

| # | Question | Status |
|---|---|---|
| 1 | **4-class definition.** (a) left/right fist + both fists + both feet, all MI. (b) a rest class substituted for one MI class. | **CLOSED 2026-09-02.** Scheme (a) was used throughout and every reported number rests on it. The paper *names* L/R/B/F while the shipped code substitutes resting-state windows (4.5); we follow the paper's stated task, which is also the harder and more honest one. The discrepancy is reported as a finding rather than resolved in the code's favour. Full comparison in 7.1 |
| 2 | Whether to apply the S089/S090 correction or keep `CLAUDE.md` literal | Correction applied 2026-08-31; revertible |
| 3 | Delete the 4.4 GB of redundant zip + dead mirror | Awaiting user |
| 5 | **`CLAUDE.md`'s Expected Results section** stated ~90-93 % subject-dependent and 60-70 % LOSO as targets, both from the leaky literature. | **RESOLVED 2026-09-02**, on the user's instruction. Rewritten to carry measured values with CIs instead of targets, to record that LOSO came out *above* subject-dependent, to note 4-class LOSO is unmeasured, and to add the two reporting rules learned the hard way (always decompose; never quote a partial run) |
| 4 | **What "reproduce the baseline" means.** (i) Re-implement the leaky protocol, then show it collapses under a correct split. (ii) Only ever run the correct protocol. | **RESOLVED to (i) and now EXECUTED.** The leaky arm reproduces 0.8975 against the published 0.8835; the correct arm gives within-family 0.5090, at chance. The paired gap is +0.6096 [+0.5767, +0.6411] over 103 subjects (9.8). Note the leakage-path count was later corrected from four to three (4.4) |

---

### 7.1 Class scheme (a) vs (b) - the comparison, for the record

**(a) L / R / B / F** - left fist, right fist (runs 4/8/12), both fists, both feet
(runs 6/10/14). T0 discarded. ~22-23 trials per class, 90 per subject. **This is what
the paper used, and what we use.**

**(b) rest / L / R / F** - substitutes T0 rest for one motor class (usually B).
~90 rest trials against ~22 per motor class per subject.

| | (a) L/R/B/F | (b) rest/L/R/F |
|---|---|---|
| Matches the paper | **Yes** | No - breaks comparability |
| Class balance | Natural, ~22-23 each | ~90 rest vs ~22 each; needs subsampling or weighting |
| Data used | All 6 imagery runs, nothing wasted | Discards one motor class; subsampling discards rest trials |
| Difficulty | Harder - four active imagery states | Easier - rest-vs-task ERD/ERS is a large, well-localised effect |
| Headline accuracy | Lower, honest | Inflated relative to (a); not comparable to a no-rest 4-class number |
| Class purity | All four are motor imagery | "Rest" T0 sits between task periods, contaminated by preparation and after-effects |
| BCI realism | A 4-way command decoder | Includes an idle state, closer to an asynchronous BCI |

**Known confound affecting (a) - must be reported.** L and R come *only* from runs
4/8/12; B and F come *only* from runs 6/10/14. Any run-level non-neural signal
(electrode drift, impedance change, alertness) separates {L,R} from {B,F} for free,
without decoding imagery at all. This is inherent to the dataset, not to our
implementation, and it is not a train/test leak - but it means a 4-class accuracy can
look respectable while the model decodes almost no imagery.

Mitigation, to be reported alongside the 4-class number:

- the **confusion matrix** - near-zero between-family confusion combined with high
  within-family confusion is the signature of a model exploiting run identity;
- the **within-family binary accuracies** (L vs R, and B vs F), which are confound-free
  and are the honest measure of imagery decoding.

Scheme (b) partially breaks the confound for the rest class (T0 occurs in every run) but
leaves it intact for the motor classes.

### 7.2 Baseline reproduction protocol (decision 4, option (i))

Two runs of the same architecture, reported side by side:

1. **Leaky replication** - timepoint samples, shuffled 90/10 split, whole-dataset
   mean-centring, BatchNorm in train mode at test, adjacency from all data. Expected to
   land near the published 93 %. Reported explicitly as a reproduction of the published
   protocol, never as our result.
2. **Correct protocol** - trial-level split, train-split-only normalisation and
   adjacency, proper `model.eval()`. Then LOSO.

The difference between them is the headline finding. Where feasible, ablate the four
paths one at a time so the contribution of each is quantified rather than asserted.

---

## 8. References (verified 2026-08-31)

Checked against Crossref and, for [1], the arXiv PDF title page. **Two entries in the
draft list were wrong — see the notes.**

**[1]** Y. Hou, S. Jia, X. Lun, **Z. Hao, Y. Shi, Y. Li, R. Zeng**, and J. Lv, "GCNs-Net:
A Graph Convolutional Neural Network Approach for Decoding Time-Resolved EEG Motor
Imagery Signals," *IEEE Transactions on Neural Networks and Learning Systems*,
vol. 35, no. 6, pp. 7312-7323, 2024. DOI: 10.1109/TNNLS.2022.3202569.

> **Corrected.** The draft gave this paper the author list of [2]. The real authors are
> Yimin Hou\*, Shuyue Jia\*, Xiangmin Lun, **Ziqian Hao, Yan Shi, Yang Li, Rui Zeng**,
> Jinglei Lv (\* = co-first authorship, per the arXiv title page). S. Zhang, T. Chen and
> F. Wang are **not** authors of this paper.
> **Year corrected 2022 -> 2024.** Early access September 2022 (hence the 2022 DOI and the
> 2022 PubMed date), final issue vol. 35, no. 6, 2024. Cite 2024, or "early access 2022"
> explicitly if the 2022 date matters.

**[2]** Y. Hou, S. Jia, X. Lun, S. Zhang, T. Chen, F. Wang, and J. Lv, "Deep Feature
Mining via the Attention-Based Bidirectional Long Short Term Memory Graph Convolutional
Neural Network for Human Motor Imagery Recognition," *Frontiers in Bioengineering and
Biotechnology*, vol. 9, art. 706229, 2022. DOI: 10.3389/fbioe.2021.706229. — **correct as
drafted**; add volume and article number.

**[3]** G. Schalk, D. J. McFarland, T. Hinterberger, N. Birbaumer, and J. R. Wolpaw,
"BCI2000: A General-Purpose Brain-Computer Interface (BCI) System," *IEEE Transactions on
Biomedical Engineering*, vol. 51, no. 6, pp. 1034-1043, 2004.
DOI: 10.1109/TBME.2004.827072. — **correct as drafted.**

> PhysioNet's terms require citing the resource paper **in addition** to [3]:
> A. L. Goldberger *et al.*, "PhysioBank, PhysioToolkit, and PhysioNet: Components of a
> New Research Resource for Complex Physiologic Signals," *Circulation*, vol. 101, no. 23,
> pp. e215-e220, 2000. Add this as a separate entry.

**[4]** X. Deng, D. Xu, H. Huo, X. Hong, and B. Liu, "Multi-level hierarchical dynamic
graph convolutional network**s** for motor imagery EEG analysis," *Neurocomputing*,
vol. 626, art. 129594, 2025. DOI: 10.1016/j.neucom.2025.129594. — title takes
"Network**s**" (plural); "MHD-GCN" is the in-paper acronym, not part of the title.

**[5]** M. Meng, P. Yu, S. Zhang, F. Fang, Y. Ma, Q. She, and Y. Zhang, "SAMGCN: A
self-adaptive multilevel graph convolutional network for motor imagery classification,"
*Biomedical Signal Processing and Control*, vol. 116, art. 109506, 2026.
DOI: 10.1016/j.bspc.2026.109506. — the draft had **no authors**; added.

### 8.1 Positioning note — [4] and [5] are direct prior art

Both are **adaptive/dynamic graph learning for motor imagery**, which is the same space
as our stated contribution ("dynamic/learnable adjacency + temporal attention"):

- **[4] MHD-GCN** combines a spatiotemporal attention mechanism with hierarchical
  global/local dynamic topologies built by community clustering.
- **[5] SAMGCN** learns inter-channel relationships with channel position encoding and a
  channel self-attention graph-embedding module, over a multilevel GCN.

So "learnable adjacency + attention on a GCN for MI" is **not novel in itself as of 2026**.
The defensible novelty has to be elsewhere, and on current evidence the strongest candidates
are:

1. **Subject-independent LOSO evaluation.** Neither GCNs-Net headline number is
   subject-independent (4.5), and this needs checking for [4] and [5] too — if they also
   report only within-subject or pooled splits, a rigorous LOSO benchmark across these
   architectures is a genuine contribution.
2. **The leakage audit itself.** Quantifying how much of the published GCNs-Net accuracy
   comes from the three leakage paths (4.4) is a result no one else has published.
3. **Scalp-topography interpretability**, given the finding that the PCC graph recovers
   the montage unprompted (6.6).

**Action:** check whether [4] and [5] evaluate under LOSO and on which datasets, before
finalising the novelty claim. If either already reports LOSO on PhysioNet, the framing
needs to change.

### 8.2 Research directions in the reference list, and what we take from them

Sources: [1] read in full; [2] abstract + published record; [4] and [5] **abstracts only**
— both are paywalled on ScienceDirect (403) and the SSRN preprint of [4] is also blocked.
Claims below about [4] and [5] are therefore abstract-level and must be re-checked against
the PDFs before anything is written up.

**The arc.**

| | Paper | Graph | Temporal / spectral | Evaluation |
|---|---|---|---|---|
| 2022/24 | [1] GCNs-Net | **static** `\|PCC\| - I` over all signals | none — one timepoint per sample | within-subject + pooled |
| 2022 | [2] BiLSTM-GCN *(same group)* | static PCC | **BiLSTM + attention**, 0.4 s windows | within-subject + pooled |
| 2025 | [4] MHD-GCN | **dynamic**, hierarchical global/local topologies from community clustering | spatiotemporal attention + **multi-level wavelet** bands | PhysioNet **and** BCI Competition |
| 2026 | [5] SAMGCN | **adaptive**, adjacency updated during training; channel position encoding + channel self-attention | dual branch: **differential entropy** (frequency) + temporal | BCI IV-2a and OpenBMI |

**Five directions the field has converged on since [1]:**

1. **The adjacency is learned, not fixed.** Both [4] and [5] update it during training.
2. **Frequency decomposition replaced the raw broadband signal.** [4] uses multi-level
   wavelet blocks for clinical bands; [5] uses differential entropy features. [1] uses
   unfiltered broadband, which is now clearly the outlier.
3. **Attention over channels and/or time**, rather than pure convolution.
4. **Hierarchical / multi-level topology** instead of a single graph — [4] builds separate
   global and local functional spaces by community clustering.
5. **Over-smoothing is named as the failure mode** to design against ([5] explicitly).

**What we take, ranked by value per unit of work.**

1. **Band-pass filtering to the mu/beta range.** The single biggest lever we are missing.
   Our pipeline follows [1] with *no filtering at all* — broadband 0-80 Hz — while motor
   imagery lives in mu (8-13 Hz) and beta (13-30 Hz). Both modern papers work in the
   frequency domain. Cost: one config line, `bandpass: [8, 30]`, plus a cache rebuild
   (~45 s). Likely a larger gain than a learnable adjacency, and it must be ablated
   separately or it will contaminate attribution of every later result.
2. **Per-band adjacency.** One PCC graph per band rather than one broadband graph — the
   idea behind [4]'s "multispectral representations". Our `pcc_adjacency` already takes
   arbitrary data, so this is a loop, not new machinery.
3. **Measure over-smoothing before designing against it.** With six conv layers on a
   64-node near-complete graph, the baseline is a prime candidate. Node-feature similarity
   per layer is a cheap diagnostic and a good figure, and it would *justify* the
   contribution rather than assert it.
4. **Community clustering as an alternative pooling hierarchy** ([4]) — a direct ablation
   against our Graclus port, which already produces anatomically sensible clusters (6.6).

**Positioning consequence — [2] matters more than we assumed.** [2] is the *same group's*
follow-up and already pairs a GCN with temporal attention (BiLSTM + attention). Our stated
contribution must therefore be positioned against [2], not only [1], or the temporal-attention
half is straightforwardly pre-empted by the base paper's own authors.

**What still looks defensible:**

- **LOSO on PhysioNet.** [5] does not use PhysioNet at all (BCI IV-2a and OpenBMI), so it
  does not compete here. **[4] does use PhysioNet** — whether under LOSO is unknown and is
  the single most important thing to check.
- **The leakage audit** (4.4). No one appears to have published it.
- **Interpretability**, supported by the montage-recovery result (6.6).

**Open action:** obtain the PDFs of [4] and [5] and confirm (i) [4]'s evaluation protocol on
PhysioNet, (ii) whether either reports LOSO, (iii) their exact adjacency-learning formulation
so ours is not a re-derivation. Useful survey for the lit review:
*Graph Neural Network-based EEG Classification: A Survey*, arXiv:2310.02152.
---

## 9. Results

### 9.1 Leaky replication vs correct protocol (20 subjects)

Identical model, identical data, identical train/test sizes. Only the protocol differs.
Leaky arm: `scripts/run_leaky_replication.py`. Correct arm:
`scripts/run_subject_dependent.py`, stratified trial-level 5-fold. 25 epochs, Adam at
lr 1e-2, seed 42.

| | Published protocol | Correct protocol |
|---|---|---|
| trials on both sides of the split | **90 of 90** | **0** |
| adjacency / standardiser fit on | all data | training trials only |
| BatchNorm at test | train mode | eval mode |
| **timepoint accuracy** | **0.8824 +/- 0.1541** | **0.2773 +/- 0.0262** |
| trial accuracy | — | 0.3000 +/- 0.0500 |

**Paired over the 20 subjects: +0.6051, 95 % bootstrap CI [+0.5364, +0.6691].**

The leaky arm reproduces the paper's published **88.35 %** group-level figure almost
exactly (we get 88.24 %), which is the evidence that the port is faithful and that the
gap below is a protocol effect and not an implementation failure.

**Mechanism, from the training curve** (S001, correct protocol):

```
epoch  1   train 0.2806   val 0.2283
epoch 10   train 0.9001   val 0.1634
epoch 25   train 0.9985   val 0.3403
```

Train accuracy reaches 99.85 % while validation stays at chance. The model is not
failing to learn — it memorises trial-specific voltage patterns completely. Under the
published protocol ~90 % of every trial's timepoints are in the training set, so the
held-out 10 % are trivially recovered. A learning-rate sweep (1e-6, 1e-4, 1e-3, 1e-2)
leaves the held-out number at chance throughout, so this is not an optimisation artefact.
The reference's own lr of 1e-6 additionally does not train at all within 25 epochs.

### 9.2 The 30 % contains no motor-imagery decoding

Section 7.1 recorded the run-family confound **in advance** and specified the diagnostics
that must accompany any 4-class number. Running them (`scripts/run_family_analysis.py`)
on the pooled 1,800 test trials:

| | Correct protocol | Published protocol |
|---|---|---|
| 4-class accuracy (chance 0.25) | 0.3000 | 0.8824 |
| **run-family** accuracy (chance 0.50) | **0.6239** | 0.9355 |
| **within-family** accuracy (chance 0.50) | **0.4809** | 0.9432 |
| left vs right fist (chance 0.50) | 0.4937 | 0.9427 |
| both fists vs both feet (chance 0.50) | 0.4642 | 0.9437 |

Under the correct protocol the model identifies **which run family a trial came from**
at 12.4 points above chance, while discriminating *within* a family at 1.9 points
**below** chance. Classes 0/1 occur only in runs 4/8/12 and classes 2/3 only in runs
6/10/14, so family identification is available from any run-level non-neural signal —
electrode drift, impedance change, alertness — with no imagery decoding whatsoever.

**Conclusion: the entire above-chance 4-class performance is the run-family artefact.
Evaluated correctly, the published architecture and sample definition decode no motor
imagery at all.** Under leakage the same decomposition shows 94 % within-family, which
is memorisation of trials, not decoding.

This is the strongest single result in the project so far, and it is strengthened by the
confound having been predicted and written down before it was measured.

### 9.3 Caveats

- 20 subjects of 103, one seed. Extend before writing up.
- This is a verdict on the **published formulation** — one timepoint per sample — not on
  graph approaches to motor imagery in general. A single 6.25 ms sample carries no
  spectral content, and motor imagery is a band-power phenomenon, so it is unsurprising
  in hindsight that the formulation cannot generalise across trials.
- 17/20 subjects sit above chance on the 4-class number, but 9.2 shows that lift is
  family identification, so per-subject counts above chance are not evidence of decoding.

---

### 9.4 Representation ablation: band-pass x sample definition

A 2x2 over the same 20 subjects and the same correct protocol (stratified trial-level
5-fold). Band-pass arms use `configs/bandpass_8_30.yaml` and a separate cache. Window
arms use 64 timepoints (0.4 s, the length used by [2]) at stride 16.
Run with `scripts/ablation_summary.py`.

| arm | band-pass | sample | 4-class | family | **within-family** | within 95 % CI |
|---|---|---|---|---|---|---|
| A | none | 1 tp | 0.3000 | 0.6239 | 0.4859 | [0.4533, 0.5193] |
| B | 8-30 Hz | 1 tp | 0.3472 | 0.6778 | 0.5110 | [0.4810, 0.5439] |
| C | none | 64 tp | 0.3311 | 0.6544 | 0.5051 | [0.4655, 0.5489] |
| **D** | **8-30 Hz** | **64 tp** | 0.3300 | 0.6100 | **0.5418** | **[0.5050, 0.5817]** |
| chance | | | 0.2500 | 0.5000 | 0.5000 | |

**Only arm D's within-family CI excludes chance**, and only barely — the lower bound is
0.5050. Paired contrasts across subjects:

```
B - A  band-pass at 1 timepoint       +0.0250  [-0.0108, +0.0600]  not significant
D - C  band-pass at 64 tp             +0.0368  [-0.0079, +0.0808]  not significant
C - A  window, unfiltered             +0.0191  [-0.0130, +0.0540]  not significant
D - B  window, band-passed            +0.0309  [+0.0126, +0.0486]  significant
D - A  both vs published formulation  +0.0559  [+0.0184, +0.0921]  significant
```

Neither factor is significant alone; only the combination is. That is consistent with the
mechanism — a window supplies time extent, the filter supplies the right band, and a
band-power effect needs both to be visible — but with five uncorrected contrasts this
should be treated as suggestive rather than established.

**The 4-class number is a poor guide, and this is the useful part.** Arm D's 4-class
accuracy (0.3300) is no better than B or C, yet its *composition* changed: family accuracy
**fell** (0.6239 -> 0.6100) while within-family accuracy **rose** (0.4859 -> 0.5418). The
model shifted from exploiting the run-family artefact toward weak genuine decoding, and a
headline accuracy would have hidden that entirely. Subjects above chance within-family:
**A 8/20, D 12/20**.

**Honest reading.** Fixing the representation produces a small, statistically detectable,
practically weak improvement. Within-family 0.5418 on a binary problem is barely above
chance. So the published formulation's failure is *not* only preprocessing: 4-class motor
imagery on PhysioNet under a correct trial-level split is genuinely hard for this
architecture, and better inputs recover only a fraction of what the leakage was supplying.

**A prediction that failed, recorded because it was made in advance.** Section 9.2's
mechanism suggested band-passing would push *family* accuracy toward chance, on the
grounds that sub-8 Hz drift (61.5 % of unfiltered power, versus 3.5 % after filtering) is
the run-level artefact. At the timepoint level it went the other way (0.6239 -> 0.6778).
The diagnostic in 7.1 was also over-stated: family accuracy conflates run-level artefact
with a *genuine* neural difference, since bilateral and unilateral imagery differ
topographically. **Within-family remains the clean measure; family accuracy is not purely
confound.**

**Caution on partial runs.** At 35/100 folds arm D showed within-family 0.5774; the
completed 100 folds give 0.5418. Partial results overstated the effect, so interim numbers
should not be quoted.

---

### 9.5 Gate A — the classical baseline beats the network

The question 9.4 could not answer: is 0.54 within-family a limit of *this architecture*
or of *this task under honest evaluation*? Common Spatial Patterns + LDA settles it.

`scripts/run_csp_baseline.py` runs CSP (4 components, Ledoit-Wolf regularised) with
shrinkage LDA on the same subjects, the same `stratified_trial_kfold` splits, the same
8-30 Hz cache and the same metrics module. CSP is fit on training trials only — it is a
supervised spatial filter, so fitting it on all data would leak exactly as the reference's
adjacency does.

**Making the comparison exact.** The first CSP run was not strictly comparable: the
network's "within-family" accuracy is conditioned on it *predicting* the right family,
whereas CSP is handed the true family. A `--classes A B` flag was added to
`run_subject_dependent.py` so the network solves the identical two binary problems. Both
classes in each problem come from the same run family, so the run-family confound cannot
contribute.

| problem | GCNs-Net | CSP+LDA | paired GCN − CSP |
|---|---|---|---|
| left vs right fist | 0.5178 | **0.5711** | −0.0533 [−0.1000, −0.0044] |
| both fists vs both feet | 0.5767 | **0.6722** | −0.0956 [−0.1511, −0.0367] |
| **mean within-family** | **0.5472** | **0.6217** | **−0.0744 [−0.1111, −0.0367]** |

**CSP+LDA beats the network on both problems and on 16 of 20 subjects.** Both CIs exclude
zero.

**Gate A verdict: continue to Phase 3.** The ceiling is not the task — 0.67 is reachable
on fists-vs-feet where the network manages 0.58. The architecture, or our representation
of its input, is genuinely underperforming a linear method that predates it by two
decades. There is real headroom, so representation work is justified rather than
speculative.

**This sharpens the thesis.** It is no longer only "the published result was leakage" but
"the published result was leakage, and under honest evaluation the architecture is beaten
by CSP+LDA". That is a more pointed finding and it makes Phase 3 the obvious next step.

**A neurophysiological sanity check that passes.** Both methods find fists-vs-feet
(0.67 / 0.58) much easier than left-vs-right (0.57 / 0.52). Hand and foot motor cortex are
well separated on the homunculus, while left-vs-right hand requires finer lateralisation.
The ordering is what the anatomy predicts, which is mild evidence that both pipelines are
measuring something real rather than noise.

**Direct implication for Phase 3a.** CSP is explicitly a spatial-filter-plus-log-band-power
method. The network is currently being handed raw microvolts and asked to infer band power
from a waveform. Making band power an explicit node feature is therefore not a generic
tuning idea — it is giving the network the representation its winning competitor uses.

---

### 9.6 Phase 3a — band power as node features

Gate A's implication was that the network should be given the representation CSP
uses. `data/features.py` computes log band power per channel per window (Hann-tapered
rFFT, `fbcsp` 4 Hz bins from 4-40 Hz, 1 s window at stride 32 for 1 Hz resolution).
The graph is still built from the raw time series, so node features are the only change.

**4-class, 20 subjects:**

| arm | node features | 4-class | family | within-family |
|---|---|---|---|---|
| A | raw voltage, 1 timepoint | 0.3000 | 0.6239 | 0.4859 |
| D | raw voltage, 64 tp window, 8-30 Hz | 0.3300 | 0.6100 | 0.5418 |
| **E** | **log band power, fbcsp** | **0.4222** | **0.7378** | **0.5716** |

E − D within-family: **+0.0298, 95 % CI [+0.0051, +0.0532]** — significant.

**But read the decomposition, not the headline.** The 4-class number jumped 0.3300 →
0.4222, which looks like a large win. Family accuracy jumped 0.6100 → 0.7378 (+12.8
points) while within-family rose only +3.0 points. **Roughly four fifths of the apparent
gain is better run-family identification, not better motor-imagery decoding.** This is
the second time the 4-class number has been misleading in the direction the §7.1
diagnostic predicted.

**Against the matched binary problems, band power did not help at all:**

| problem | raw-voltage GCN | band-power GCN | CSP+LDA | band power − CSP |
|---|---|---|---|---|
| left vs right fist | 0.5178 | 0.4944 | **0.5711** | −0.0767 [−0.1378, −0.0178] |
| both fists vs feet | 0.5767 | 0.5967 | **0.6722** | −0.0756 [−0.1367, −0.0167] |

Band power versus raw voltage is **not significantly different on either problem**
(−0.0233 [−0.0611, +0.0144] and +0.0200 [−0.0178, +0.0589]), and band-power
left-vs-right sits **at chance**: 0.4944, CI [0.4644, 0.5222].

**Verdict: Phase 3a did not close the gap to CSP.** The hypothesis — that the network
was losing because it had to infer band power from waveforms — is **not supported**.
Handing it band power explicitly changed nothing on the confound-free comparisons.

**What this points to instead.** CSP is not merely a band-power method; it is a
*discriminative spatial filter* fit to maximise the between-class variance ratio. Our
network convolves over a **fixed** correlation graph built without reference to the
labels. The remaining gap is therefore most likely spatial, not spectral.

That is a direct, evidence-based argument for the learnable-adjacency contribution —
one now motivated by a measured failure rather than by the literature being fashionable.
It also revises §6.5's reading: the fixed PCC graph looked like a reasonable baseline
once CAR was applied, but a graph built without label information cannot do what CSP
does regardless of how well it captures resting correlation structure.

**Revised priority for Phase 3.** 3b (window length) and 3c (band edges) now look
low-yield — they tune a spectral representation that has just been shown not to be the
bottleneck. The learnable adjacency (Phase 5) should be brought forward.

---

### 9.7 Learnable adjacency — no effect (EXPLORATORY, development set)

**This is a development-set result on 20 subjects and is therefore exploratory
(10.1). The confirmatory H1 test requires all 103 subjects.**

Identical to the fixed-graph runs in every respect except `--learnable-graph`.
Because `delta` initialises to zero, the two models are bit-for-bit identical at
step 0, so any difference is attributable to learning the graph and nothing else.

| problem | fixed PCC | learned | CSP+LDA | learned − fixed |
|---|---|---|---|---|
| left vs right fist | 0.5178 | 0.5189 | **0.5711** | +0.0011 [−0.0167, +0.0200] p=0.46 |
| both fists vs both feet | 0.5767 | 0.5689 | **0.6722** | −0.0078 [−0.0344, +0.0211] p=0.69 |
| **mean within-family** | **0.5472** | **0.5439** | **0.6217** | **−0.0033 [−0.0200, +0.0161] p=0.62** |

The learned graph beats the fixed one on **7 of 20 subjects** — worse than a coin
flip. The 4-class arm is likewise unchanged: 0.3294 against 0.3300, within-family
0.5368 against 0.5418.

**The mechanism did engage.** A null result because nothing trained is a different
finding from a null result because training does not help, so the checkpoints were
inspected directly:

```
checkpoint      layer   ||delta||   max|d|   edges changed >1%
S001_fold0      conv0      0.9997   0.0485        81.3 %
S001_fold1      conv0      1.4237   0.0704        85.6 %
S001_fold2      conv0      1.9733   0.1078        90.4 %
```

80-90 % of first-layer edges moved. The graph was learned; it simply did not
matter.

### 9.7.1 What this means, taken with 9.6

> **SUPERSEDED IN PART, 2026-09-02, by the A1/A2 controls in 9.11.** The reasoning below
> was written before either control existed and it reaches the right *conclusion* by
> partly wrong *reasoning*. Kept verbatim because the project appends rather than
> rewrites, and because the gap between what was inferred here and what was later
> measured is itself the finding.
>
> * **What survived.** The limitation is architectural, and it is not sample size —
>   A2 shows graph-free CNNs reaching and passing CSP on the same ~45 trials.
> * **What did not.** This section names the mechanism as graph-locality plus
>   node-shared weights. A1 lifted exactly those constraints and recovered only
>   **16 %** of the gap (+0.0128 of +0.0802). The mechanism is real but minor; roughly
>   84 % of the gap lies elsewhere in the architecture.
> * **The lesson.** Two null results were read as identifying a cause. They bounded it
>   instead. An inference from absences is not a measurement, and this one was 84 %
>   wrong about magnitude while pointing the right way.


Two mechanisms have now been ruled out on the development set:

- **Spectral (9.6).** Giving the network explicit band power changed nothing on the
  confound-free comparisons.
- **Spatial graph (here).** Letting the network reshape the electrode graph
  discriminatively changed nothing either.

Yet CSP+LDA remains ahead by ~0.075. The remaining explanation is not *which* graph
the network convolves over, but **the graph-convolutional inductive bias itself**.

CSP fits an unconstrained discriminative spatial projection: any linear combination
of all 64 channels, chosen to maximise the between-class variance ratio. A Chebyshev
graph convolution cannot express that. Its spatial mixing is restricted to
graph-local neighbourhoods with weights *shared across nodes* — order-2 Chebyshev
means each node mixes only with its one-hop neighbourhood, using the same filter
everywhere. Learning the adjacency changes *which* nodes are neighbours; it does not
lift the locality or the weight sharing.

For motor imagery, where the discriminative signal is a specific weighted contrast
across the whole scalp (C3 against C4 for lateralisation, central against
peripheral for hand-versus-foot), that restriction appears to be a handicap rather
than a useful prior. **This is a defensible and interesting conclusion**: the graph
prior that motivates the entire architecture is, on this task, worse than no prior
at all.

It should be stated carefully. It is not "GCNs do not work on EEG" — it is that on
*this* dataset, under a correct protocol, with this Chebyshev formulation, the
graph constraint costs more than it contributes relative to an unconstrained
spatial filter.

**Consequence for H2/H3.** Temporal attention addresses *when*, not *where*, so it
is not ruled out by this. But if the spatial bottleneck is the inductive bias, the
honest expectation for H3 is modest. That expectation is recorded here, before the
confirmatory runs.

---

### 9.8 Phase 1 anchor results — full scale, 103 subjects, 3 seeds

Everything in 9.1-9.7 was measured on 20 subjects. These are the definitive numbers.
Zero duplicate rows; all nine tags complete.

**Published protocol, reproduced.**

```
seed 1  0.8978      seed 2  0.8970      seed 3  0.8975
POOLED  0.8975   95% CI [0.8794, 0.9147]   seed spread 0.0008
paper reports 0.8835 group-level / 0.9306 subject-level
```

**Correct trial-level split.**

| | 4-class | family | within-family | 95 % CI | above chance |
|---|---|---|---|---|---|
| published formulation (1 timepoint) | 0.3097 | 0.6106 | **0.5090** | [0.4984, 0.5199] | 53/103 |
| best representation (8-30 Hz, 64 tp) | 0.3395 | 0.6168 | **0.5470** | [0.5291, 0.5667] | 60/103 |

**The headline gap, paired over 103 subjects: +0.6096, 95 % CI [+0.5767, +0.6411].**

**Three things this settles.**

1. **The published formulation is at chance within-family at full scale.** Its CI
   [0.4984, 0.5199] contains 0.5. The 20-subject estimate was 0.4859; the full-scale
   value is 0.5090, and both are indistinguishable from chance. The conclusion in 9.2 —
   that the published architecture and sample definition decode no motor imagery once
   the split is honest — holds on all 103 subjects.
2. **The best representation is genuinely above chance, and modestly so.** 0.5470,
   CI [0.5291, 0.5667], excluding 0.5. The 20-subject estimate was 0.5418. Real, small.
3. **The development-set estimates were not optimistic.** Every 20-subject figure landed
   within its full-scale CI, and seed spread is tiny (0.0008 for the leaky arm, 0.0113
   across arm A's within-family). That matters for 10.1: it is evidence the development
   set was not silently overfitted by repeated inspection, which is the risk that
   demotion was meant to guard against.

**Reproducibility note.** These runs survived a full stop and restart mid-flight. Every
tag reached its exact target count with zero duplicates, which is the practical
validation of the `--resume` mechanism described in 6.11.

---

### 9.9 Confirmatory results — all five hypotheses, 103 subjects

Run once over the complete family, exactly as pre-registered in 10. Holm-Bonferroni at
family-wise alpha 0.05, directions fixed in advance, primary outcome mean within-family
accuracy over the two binary problems.

**Within-family accuracy (chance 0.5), 103 subjects:**

| configuration | mean | 95 % CI | above chance |
|---|---|---|---|
| fixed graph, no attention (baseline) | 0.5438 | [0.5288, 0.5598] | 64/103 |
| + learnable adjacency (H1) | 0.5469 | [0.5317, 0.5632] | 70/103 |
| **+ temporal attention (H2)** | **0.6038** | [0.5805, 0.6278] | 83/103 |
| + both (H3/H4) | 0.5976 | [0.5745, 0.6220] | 79/103 |
| CSP+LDA (H4 reference) | **0.6239** | [0.5977, 0.6521] | 81/103 |
| **LOSO, attention config (H5)** | **0.6040** | [0.5823, 0.6268] | 88/103 |

**Verdicts:**

| | hypothesis | difference | 95 % CI | p | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| H1 | learnable adjacency > fixed graph | +0.0031 | [−0.0038, +0.0102] | 0.196 | 0.392 | **not supported** |
| H2 | temporal attention > none | **+0.0600** | [+0.0463, +0.0745] | 0.0001 | 0.0005 | **SUPPORTED** |
| H3 | both > better of either alone | −0.0061 | [−0.0154, +0.0031] | 0.899 | 0.899 | **not supported** |
| H4 | full contribution vs CSP+LDA | **−0.0263** | [−0.0413, −0.0112] | 0.0004 | 0.0012 | **SUPPORTED** |
| H5 | LOSO within-family > chance | **+0.1040** | [+0.0826, +0.1269] | 0.0001 | 0.0005 | **SUPPORTED** |

**H4's verdict must be read with its sign.** The test was pre-specified two-sided, so
"supported" means only that a difference exists — and the difference is **negative**.
CSP+LDA still beats the full contribution by 0.0263 [0.0112, 0.0413]. Reporting H4 as
"supported" without the direction would be actively misleading; the honest statement is
that the architecture remains significantly *behind* a linear method from 2000.

### 9.9.1 What the four outcomes mean together

**Temporal attention works; the learnable graph does not.** H2 gains +0.0600 while H1
gains +0.0031, and H3 shows the combination is no better than attention alone (in fact
marginally worse). The confirmatory result reproduces the exploratory one in 9.7 on 83
unseen subjects, so this is not a development-set artefact.

That pattern is coherent with 9.7.1's argument. The deficit was never *which* electrodes
the graph connects — it was that the architecture had no way to weight *when* within a
trial. Attention supplies exactly that. Reshaping the graph supplies something the model
did not need.

**The gap to CSP narrowed but did not close.** Baseline 0.5438 -> 0.6038 with attention,
against CSP's 0.6239. The contribution recovers roughly three quarters of the deficit,
and the remainder is still statistically real. On the reading in 9.7.1, what is left is
the graph-convolutional inductive bias itself, which no component tested here removes.

**H5 is the strongest positive result in the project, and it is a surprise.** LOSO
within-family reaches 0.6040 [0.5823, 0.6268] with 88/103 subjects above chance —
**higher than the subject-dependent number of 0.6038**, not lower. Subject-independent
decoding is normally harder, and the expectation recorded in `CLAUDE.md` was that LOSO
would sit well below within-subject performance.

The explanation is data volume. A LOSO fold trains on ~4,600 trials from 102 subjects;
a subject-dependent fold trains on ~36 trials from one. Roughly 100x more training data
outweighs the cost of never having seen the test subject. It also implies the
subject-dependent protocol is *data-starved*, not that between-subject transfer is easy.

This is, as far as we can tell, the first correct subject-independent evaluation of this
architecture: neither published headline is subject-independent (4.5).

**Nothing here rescues the published claim.** The audit stands: 0.8975 under the
published protocol, chance within-family under a correct one (9.8). What the
contribution buys is a model that decodes *something* — 0.60 on a binary problem — which
is well short of the 0.93 the paper reports and still behind CSP+LDA.

---

### 9.10 What temporal attention actually learned (EXPLORATORY)

H2 is the project's only positive component result, so it is worth knowing *what*
attention learned rather than only that it helped. `scripts/analyse_attention.py`,
12 development subjects, confirmatory configuration.

**A hypothesis, tested and refuted.** The first reading of the attention profile was
that the weights are uniform — the population-mean profile deviates from uniform by
only 2.1 %. That suggested the H2 gain was not attention at all but two side effects:
temporal averaging (denoising) and a 16x narrower first conv layer (2,048 -> 128
parameters), i.e. regularisation.

`SegmentMeanPool` tests this directly. It produces the **same output shape** and leaves
the rest of the network identical, but has **no learned temporal parameters** — uniform
weights within each of 4 fixed segments.

| | within-family |
|---|---|
| fixed graph, no pooling | 0.5472 |
| **temporal attention** | **0.5939** |
| fixed mean pool (control) | 0.5367 |

```
mean pool  - baseline    -0.0106 [-0.0272, +0.0061]  p=0.885   no difference
attention  - baseline    +0.0467 [+0.0183, +0.0728]  p=0.0001  differs
attention  - mean pool   +0.0572 [+0.0233, +0.0900]  p=0.0009  differs
```

**Averaging gains nothing.** The hypothesis is refuted: attention beats an identical
architecture with fixed uniform weights by +0.0572. Neither denoising nor reduced width
explains H2.

**The figure was misleading, and so was our reading of it.** Plotting the
population-mean profile hides the effect entirely:

```
population-mean deviation from uniform        2.1 %
per-sample deviation from uniform (median)  104.5 %
per-sample deviation (90th percentile)      171.7 %
max/min weight within one sample (median)     4.9x
argmax timepoint across samples: std 18.8 of 64, spanning 0-63
```

Individual trials are **strongly peaked** — a ~5x ratio between the most and least
weighted timepoint — but they peak at *different* moments, so averaging across samples
flattens the profile to near-uniform. The figure now plots individual sample profiles
against the population mean, because the mean alone supports exactly the wrong
conclusion.

This is what the module was designed for and what the docstring predicted: ERD onset
varies by trial, so a per-trial weighting is the point. It is worth noting we reached
the correct interpretation only after a control refuted the incorrect one.

**Independent corroboration from the second panel.** Decodability across the trial
peaks at **0.60 s** after cue onset (range 0.519-0.667 across window positions). A rise
shortly after the cue is the neurophysiologically expected ERD shape, and it is
measured here without any of it being built into the model — evidence the network is
tracking real motor-imagery dynamics rather than an artefact.

**Caveat.** All of 9.10 is exploratory (10.6) and was not pre-registered. It explains
a confirmatory result; it does not add one. The mean-pool control in particular is a
new comparison and is reported as exploratory no matter how large its effect.

---

### 9.11 A1 and A2 — the controls that identified the limitation (EXPLORATORY)

**Run 2026-09-02. 103 subjects, 5 folds, both binary problems, identical cache, splits,
seeds, standardiser, windowing, trial aggregation and metric as the confirmatory arms —
`--model` and `--free-spatial` select the architecture inside the same runner, so the
comparison is matched by construction rather than by careful duplication.**

**Neither arm is on the pre-registered list (10.3), and A1 tests a claim (9.7.1) written
after the confirmatory runs. Both are exploratory. Holm correction is applied across the
six comparisons anyway, but a corrected post-hoc test is still post-hoc.**

#### 9.11.1 Results

| arm | within-family | vs fixed graph | % of gap recovered |
|---|---|---|---|
| fixed PCC graph | 0.5438 | — | — |
| learnable adjacency (9.7) | 0.5469 | +0.0031, p = 0.39 | 3.9 % |
| **A1 free spatial operator** | **0.5566** | **+0.0128** [+0.0042, +0.0221], p = 0.0046 | **16.0 %** |
| + both (H4 arm) | 0.5976 | +0.0538 | 67.1 % |
| + temporal attention | 0.6038 | +0.0600 | 74.8 % |
| **CSP+LDA** | **0.6239** | +0.0802 | 100 % (definitionally) |
| **A2 EEGNet** | **0.6296** | **+0.0858** [+0.0697, +0.1024] | **107.0 %** |
| **A2 ShallowConvNet** | **0.6725** | **+0.1287** [+0.1118, +0.1454] | **160.6 %** |

Against CSP+LDA, two-sided, paired on subject:

| comparison | difference | p (Holm) | wins |
|---|---|---|---|
| A1 free spatial vs CSP | −0.0673 [−0.0843, −0.0506] | 0.0006 | 24/103 (23 %) |
| **EEGNet vs CSP** | **+0.0056** [−0.0107, +0.0220] | **0.49 — not distinguishable** | 56/103 (54 %) |
| **ShallowConvNet vs CSP** | **+0.0485** [+0.0318, +0.0650] | **0.0006** | 67/103 (65 %) |

Per problem, and against the external ceiling:

| arm | left vs right fist | both fists vs feet | above chance |
|---|---|---|---|
| fixed graph | 0.5148 | 0.5728 | 64/103 |
| A1 free spatial | 0.5288 | 0.5845 | — |
| GCN + attention | 0.5877 | 0.6198 | 83/103 |
| CSP+LDA | 0.5942 | 0.6537 | 81/103 |
| EEGNet | 0.6058 | 0.6533 | 83/103 |
| **ShallowConvNet** | **0.6608** | **0.6841** | **94/103** |
| *[23] best of 216,714 configurations* | *0.650* | *n/a* | |
| *[23] cov-tgsp family / CSP family* | *0.605 / 0.604* | *n/a* | |

**ShallowConvNet's 0.6608 on left-vs-right exceeds the best single configuration [23]
reports from a 216,714-pipeline search on this dataset.** That is the strongest result the
project has produced, and it lands above the ceiling 17.0 established.

#### 9.11.2 What the two controls settle

**Sample size is refuted as the explanation.** 17.1 raised it on the strength of MOABB's
finding [15] that deep learning "typically requires more than 150 trials per class". We
have ~45 per binary problem and ~36 per training fold, so the concern was reasonable. It is
simply wrong here: two graph-free CNNs trained on *exactly those trials, those folds and
that cache* reach and pass CSP. Deep learning is not below its data threshold on this
dataset — graph-convolutional deep learning is below something else.

**The limitation is architectural, and 9.7.1 pointed the right way.** Every graph arm
plateaus between 0.5438 and 0.6038; both graph-free CNNs clear 0.6296. The separation is
not subtle: ShallowConvNet beats the fixed graph on 95/103 subjects.

**But the mechanism 9.7.1 named is minor.** A1 lifted graph locality, symmetry and
non-negativity — the exact constraints 9.7.1 identifies — while holding everything else
fixed and starting bit-identical to the baseline. It recovered **16 %** of the gap. Real
(p = 0.0046, and four times the learnable adjacency's effect, so the extra freedom does
buy something the learned graph could not) but small. **~84 % of the gap is elsewhere.**

Candidates, in rough order of how much they differ from what the CNNs do, none tested:

* **No temporal convolution.** Both CNNs convolve over time *first*; the GCN's first
  operation is spatial, and temporal structure only enters through attention. Note that
  attention is worth +0.0600 — by far the largest single gain any graph arm achieved —
  which is consistent with temporal modelling being the missing ingredient.
* **Depth and pooling.** Six conv layers with six Graclus poolings against EEGNet's two
  stages. This is B4's over-smoothing hypothesis, still unmeasured.
* **The band-power nonlinearity.** ShallowConvNet's square → average → log is a
  differentiable band-power estimate, and it is the best arm. 9.6 gave the *graph* explicit
  band power and it changed nothing — but that was band power as node features, not as a
  learned spectral filter after a learned spatial one.

#### 9.11.3 A correction this forced, and one it forced in the analysis code

**`CLAUDE.md`'s "a linear method from 2000 still wins" is false and has been corrected.**
It was true of every graph arm and was generalised too far. The honest statement is that
CSP+LDA beats every *graph* formulation tried, and loses to both graph-free CNNs.

**The first version of `scripts/posthoc_controls.py` would have reported the wrong
verdict.** Its decision rule tested whether A1 significantly beat the fixed graph. With 103
subjects the design detects ~0.020 (17.2), so an arm can clear significance while
recovering almost none of the gap — which is exactly what A1 does. It would have printed
"9.7.1 supported". The rule now tests the **fraction of the gap recovered**, with
significance as a precondition rather than the criterion. Caught by running the tool on
real results, not by re-reading it — the same lesson as 14's recurring theme.

#### 9.11.4 Consequences for the paper

`PAPER.md` and `RESULTS.md` predate these runs and **both now contain claims contradicted
by 9.11**, principally the framing that a classical linear method remains unbeaten. The
central finding is untouched and arguably strengthened — the published protocol still
collapses to chance, and the graph formulation still fails — but the surrounding argument
needs revision:

* the "a linear method from 2000 still wins" framing must go;
* the paper gains a positive result it did not have (ShallowConvNet above the published
  ceiling for this dataset), which changes it from a purely negative paper to a negative
  finding plus a constructive one;
* 9.7.1's mechanistic claim, quoted in the discussion, must be softened to the 16 % figure.

---

### 9.12 The reordered architecture — pre-registration (written before any result)

**Written 2026-09-02 at 23:15 local, commit `58415f4`, with the three arms launched and
zero rows aggregated.** 9.7.1 is this project's cautionary tale about interpreting results
after seeing them; this section exists so the G-arm interpretation is fixed in advance.

*Epistemic state.* `GraphBandPowerNet` is implemented, tested (16 tests) and smoke-run on
two subjects. **No aggregate over any `G-` tag has been computed.** Everything in 9.11 is
known; nothing about these arms is.

#### The arms

| tag | spatial stage | graph role |
|---|---|---|
| `G1-gbp-graph-binary-*` | Chebyshev, order 2 | hard constraint |
| `G2-gbp-free-binary-*` | unconstrained dense | none — the control |
| `G3-gbp-prior-binary-*` | unconstrained + Dirichlet penalty, weight 1e-4 | soft prior |

All three share the reordering (temporal filter -> spatial -> square -> mean -> log),
depth 1, and no node pooling. **The only thing that varies across the three is the graph's
role**, which is what makes the comparison interpretable.

#### The smoothness weight is measured, not guessed

At initialisation on S001 fold 0, cross-entropy is 0.786 and the graph penalty is 1470.
Weight 1e-4 therefore contributes **18.7 %** of the loss — real regularisation without
domination (1e-3 would be 187 %, 1e-5 only 1.9 %). Recorded because "the prior did not help"
is uninterpretable if the weight was arbitrary. **Only one weight is being run**, which is a
limitation: a null result for G3 will not distinguish "the prior does not help" from "this
weight is wrong", and a sweep is the follow-up if G3 is null.

#### Predictions, fixed now

1. **G1 and G2 both land near ShallowConvNet (0.6725), well above `+ attention` (0.6038).**
   This is the main prediction. If the reordering is what matters, all three G arms should
   clear 0.65 regardless of the graph's role.
2. **G2 >= G1 by a small margin**, mirroring A1: the graph as a hard constraint costs a
   little. If G1 clearly beats G2, A1's finding does not transfer and 9.11.2 needs revising.
3. **G3 - G2 is near zero.** The honest expectation is that the prior does not help.
   A positive, replicable G3 - G2 would be the project's first evidence that the electrode
   graph carries usable information, and would be the actual contribution.
4. **None of them exceeds ShallowConvNet meaningfully.** 0.6725 already exceeds the best of
   a published 216,714-configuration search (17.0), and the design detects ~0.020.

**Decision rules.** Primary outcome is mean within-family accuracy over both binary
problems, as in 10.2. Comparisons are paired on subject, two-sided, Holm-corrected across
the family in `posthoc_controls.py`. **A difference below 0.020 is not reportable as an
effect** whatever its p-value — that is the minimum detectable difference at 80 % power
(17.2), and it is the trap error 17 encoded. Per-subject win rates accompany every mean.

**These arms are exploratory.** They are not on the 10.3 list, and they test a claim
(9.11.2's ordering argument) formed after the confirmatory runs. They must not be reported
as confirmatory, and if they succeed the honest next step is a pre-registered replication,
not promotion.

---

### 9.13 The reordered architecture — results (EXPLORATORY)

**Run 2026-09-02/03 across two power cuts, 3,090 folds, 103 subjects, identical cache,
splits, seeds and metric as every other arm. Pre-registered in 9.12 at commit `58415f4`
with zero rows aggregated. All four predictions were confirmed.**

#### 9.13.1 Results

| arm | L vs R | B vs F | within-family | above chance |
|---|---|---|---|---|
| fixed PCC graph | 0.5148 | 0.5728 | 0.5438 | 65/103 |
| GCN + temporal attention | 0.5877 | 0.6198 | 0.6038 | 83/103 |
| CSP+LDA | 0.5942 | 0.6537 | 0.6239 | 81/103 |
| EEGNet | 0.6058 | 0.6533 | 0.6296 | 83/103 |
| ShallowConvNet | **0.6608** | 0.6841 | 0.6725 | **94/103** |
| **G1 gbp, graph constraint** | 0.6408 | 0.6850 | **0.6629** | 88/103 |
| **G2 gbp, free operator** | 0.6494 | 0.6893 | **0.6694** | 93/103 |
| **G3 gbp, graph as prior** | 0.6520 | **0.6954** | **0.6737** | 91/103 |

Paired, two-sided, Holm-corrected across eleven post-hoc comparisons:

| comparison | difference | p (Holm) | wins | verdict |
|---|---|---|---|---|
| **G1 vs GCN + attention** | **+0.0591** [+0.0458, +0.0722] | 0.0011 | 82/103 (80 %) | **different** |
| **G1 vs CSP+LDA** | **+0.0389** [+0.0224, +0.0552] | 0.0011 | 72/103 (70 %) | **different** |
| G2 vs G1 (constraint lifted) | +0.0065 [−0.0018, +0.0148] | 0.52 | 53/103 (51 %) | not distinguishable |
| G3 vs G2 (the prior) | +0.0043 [−0.0020, +0.0107] | 0.54 | 51/103 (50 %) | not distinguishable |
| G3 vs ShallowConvNet | +0.0012 [−0.0099, +0.0121] | 0.99 | 46/103 (45 %) | not distinguishable |

#### 9.13.2 Prediction scorecard

9.12 fixed four predictions before any row was aggregated. **All four hold.**

1. *"G1 and G2 both land near ShallowConvNet, well above `+ attention`."* — **Confirmed.**
   0.6629 and 0.6694 against 0.6725, and +0.0591 over the attention arm.
2. *"G2 ≥ G1 by a small margin, mirroring A1."* — **Direction confirmed, magnitude below
   detectability.** +0.0065, under the 0.020 rule 9.12 fixed in advance.
3. *"G3 − G2 is near zero; the prior does not help."* — **Confirmed.** +0.0043, and a
   **51/103 win rate is a coin flip.**
4. *"None exceeds ShallowConvNet meaningfully."* — **Confirmed.** +0.0012, p = 0.99.

Getting four for four is worth stating plainly, because 9.7.1 is the counter-example: the
difference is that these were written down first.

#### 9.13.3 What this changes

**A graph-based model now beats CSP+LDA.** G1 — the *hard* Chebyshev constraint, the same
graph formulation the paper proposes — reaches 0.6629 and beats CSP by **+0.0389** on
72/103 subjects. Every previous graph arm lost to CSP. The paper's central negative result
about the published *protocol* is untouched; what has changed is that the graph formulation
is no longer the thing that fails.

**The reordering was the entire story.** The four changes in 9.11.2 — power after the
spatial stage, a learned temporal filter, depth 1, no node pooling — lift the graph model
from 0.6038 to 0.6629, **+0.0591**, on 80 % of subjects. Nothing about the graph changed.

**And the graph itself contributes nothing.** Hard constraint (G1), free operator (G2) and
soft prior (G3) are mutually indistinguishable, spanning 0.0108 in total. The prior wins on
exactly half the subjects.

**The sharpest observation, and it revises 9.11 again.** In the *wrong* architecture,
freeing the spatial operator was significant: A1 gained +0.0128 (p = 0.0046) over the fixed
graph. In the *right* architecture the same relaxation gains +0.0065 and is not
distinguishable. **The graph constraint only cost anything while the rest of the
architecture was broken.** A1's 16 % was not measuring a graph problem; it was measuring
how much slack a badly-ordered network had. That is a different claim from either "the
graph helps" or "the graph hurts", and it is better supported than both.

#### 9.13.4 On the 0.70 target

**G3 at 0.6737 is the project's best model and is statistically identical to
ShallowConvNet.** 17.11 set out that *reaching* ShallowConvNet with a graph-informed model
would be a result and *exceeding* it would be surprising. The first happened; the second did
not, and the +0.0012 margin is a rounding artefact rather than a lead.

0.70 remains **+0.0263** away — almost exactly the gap CSP once held over us, now pointing
the other way. Nothing in 9.13 suggests architecture alone will close it: three
architecturally distinct spatial stages landed within 0.011 of each other and of
ShallowConvNet, which looks like a ceiling for this representation rather than a plateau
between designs. 17.11.2's pre-training route is untouched by these results and remains the
strongest untested lever.

**These arms are exploratory.** Not on the 10.3 list, and they test a claim (9.11.2) formed
after the confirmatory runs. Four-for-four on pre-registered predictions is evidence they
should be taken seriously; it is not a substitute for confirmatory status.

---

### 9.14 Pre-training — pre-registration (written before any result)

**Written 2026-09-03 22:05 local, commit `4246b74`, with both arms about to launch and
zero rows aggregated.** Same discipline as 9.12, which scored four for four.

**Arms.** `PT-gbp-binary-*` (pre-trained) and `PTC-gbp-binary-*` (the `--no-pretrain`
control). Architecture: **G3** — reordered band-power net, free spatial operator, graph
smoothness 1e-4. Chosen because 17.11.6 committed in advance to "whichever architecture
the G arms name", and G3 is the project's best measured arm at 0.6737. **The choice is
not load-bearing**: G1/G2/G3 span 0.0108 and are mutually indistinguishable (9.13), so
what is being tested is pre-training, not a variant.

**Why the control is mandatory.** Pre-training and source-fit normalisation change
together; only PT − PTC isolates pre-training. A comparison of PT against the
subject-dependent or G arms alone would confound the two.

**Predictions, fixed now:**

1. **PT > PTC.** The direction the whole idea rests on. [27] reports up to +7.76 %
   generally and +27.34 % on limited data; at ~36 fine-tuning trials we are in the
   limited regime.
2. **The gain is larger in the lower tail than the upper.** 17.11.1 showed the deficit
   sits below the median, and [27]'s 75 % variance reduction is the mechanism. Concretely:
   **between-subject SD should fall**, and the bottom-quartile mean should rise more than
   the top-quartile mean.
3. **PT beats ShallowConvNet's 0.6725.** Weaker confidence than 1 or 2 — this is the
   first thing in the project predicted to clear it.
4. **PT does not reach 0.70.** +0.0263 is more than a single lever has yet delivered here.

**Decision rules.** Primary outcome mean within-family accuracy over both binary problems;
paired on subject, two-sided, Holm-corrected in `posthoc_controls.py`. **A difference
below 0.020 is not an effect** whatever its p-value (17.2). Per-subject win rates reported
with every mean.

**Exploratory.** Not on the 10.3 list. Four-for-four on 9.12 is a reason to take these
predictions seriously, not a promotion to confirmatory.

#### 9.14.1 Amendment, 2026-09-04: grouped pre-training

**Made before any PT result was read.** The pre-trained arm reached 55 of 515 folds on one
problem — far below the completeness guard — and no aggregate over any `PT-` tag has been
computed. The four predictions above are unchanged; this amends only how the source model
is trained.

*Why.* Error 24: leave-one-subject-out pre-training costs one pre-training per target, 206
in total, measured at 14.9 min each — about 51 h. That is not a run we can afford.

*The change.* Pre-train once per **group** of targets. With `--groups 10` a target's model
excludes that target's whole group, so **no target is ever in its own source set** and the
leakage guarantee is unchanged; the only cost is that a target draws on ~92 source subjects
instead of 102. Twenty pre-trainings instead of 206.

*Tag hygiene.* The 55 existing `PT-*` rows were produced under leave-one-subject-out
pre-training and **must not be mixed with grouped rows under one tag** — that is error 12.
The grouped arm therefore uses a **new tag, `PTG-*`**. `PT-*` is abandoned at 55 folds,
is unreportable on its own, and is kept only as a record of the configuration that proved
too expensive.

*What is unaffected.* The control arm `PTC-*` is complete at 0.6744 and does not involve
pre-training at all, so grouping does not touch it. PTG − PTC remains the contrast that
isolates pre-training.

---

### 9.15 Cross-subject pre-training — results (EXPLORATORY)

**Run 2026-09-03/04 across a power cut, 2,060 folds over both arms, 103 subjects,
identical folds to every other arm. Pre-registered in 9.14 at commit `ce867df`, amended
for tractability in 9.14.1 before any result was read. Three of four predictions hold;
the fourth is refuted in the favourable direction.**

#### 9.15.1 Results

| arm | L vs R | B vs F | within-family | above chance |
|---|---|---|---|---|
| CSP+LDA | 0.5942 | 0.6537 | 0.6239 | 81/103 |
| ShallowConvNet | 0.6608 | 0.6841 | 0.6725 | 94/103 |
| G3 reordered | 0.6520 | 0.6954 | 0.6737 | 91/103 |
| **PTC** — control, no pre-training | 0.6533 | 0.6956 | **0.6744** | 89/103 |
| **PTG** — grouped pre-training | **0.7206** | **0.7396** | **0.7301** | **98/103** |

Paired, two-sided, Holm-corrected over fourteen post-hoc comparisons:

| comparison | difference | p (Holm) | wins |
|---|---|---|---|
| **PTG vs PTC** (isolates pre-training) | **+0.0557** [+0.0464, +0.0650] | 0.0014 | **93/103 (90 %)** |
| PTG vs ShallowConvNet | +0.0576 [+0.0454, +0.0693] | 0.0014 | 85/103 (83 %) |
| PTG vs G3 | +0.0564 [+0.0461, +0.0666] | 0.0014 | 87/103 (84 %) |

59/103 subjects at or above 0.70; 36/103 at or above 0.80.

#### 9.15.2 Prediction scorecard

1. *"PTG > PTC."* — **Confirmed.** +0.0557 on 90 % of subjects.
2. *"The gain is larger in the lower tail; between-subject SD should fall."* —
   **Confirmed.** By baseline quartile: bottom **+0.0720**, second +0.0491, third +0.0671,
   top **+0.0346**. The weakest quartile gains twice what the strongest does, and SD falls
   0.1548 → 0.1490. This is the mechanism [27] reports and the one 17.11.1 argued was
   needed, since the deficit sits below the median.
3. *"PTG beats ShallowConvNet's 0.6725."* — **Confirmed.** +0.0576.
4. *"PTG does not reach 0.70."* — **REFUTED.** 0.7301.

#### 9.15.3 The leakage audit, run before any of the above was reported

`CLAUDE.md` says that an unexpectedly high number means suspecting leakage first. 0.7301
is the highest figure this project has produced under a correct protocol, so eight checks
were run before anything was written down:

1. every row's logged `group_id` matches the deterministic `index % 10` assignment;
2. group sizes are 10-11 and source-set sizes 92-93, and every source set is exactly
   `103 − group_size`;
3. the groups cover all 103 subjects and are disjoint;
4. **no target appears in its own source set**, for every group;
5. `pretrained` is true on all 1,030 rows;
6. exactly 5 folds per subject per problem;
7. **all 1,030 folds report `selection = final_epoch`** — the fine-tune stage has no
   validation set, so nothing was ever selected using the target's test trials;
8. the gain is spread across groups (+0.0322 to +0.0933, SD 0.0189) rather than
   concentrated in a few, which is what a group-local leak would look like.

The shape of the effect is also the wrong shape for leakage: prediction 2 held, and a leak
of test information would be expected to help uniformly or to help strong subjects, not to
help the weakest quartile twice as much as the strongest.

#### 9.15.4 The comparison this number must NOT be given

**PTG's 0.7206 on left-versus-right exceeds the 0.650 that [23] reports as the best of
216,714 configurations on PhysionetMI. That is not a like-for-like comparison and must
never be presented as one.**

That benchmark evaluates within-session pipelines that see only the target subject's own
data. PTG additionally consumes ~92 other subjects' recordings during pre-training. It is
not a better pipeline under their protocol; it is a different protocol with access to two
orders of magnitude more data. The honest statement is that **cross-subject pre-training
lifts this dataset above the within-session ceiling**, which is a claim about the value of
the extra data, not about the architecture.

Every arm in 12.8 *except* PTG is directly comparable to that 0.650. PTG is not.

#### 9.15.5 What this does and does not settle

**Settles.** Pre-training is the largest single intervention in the project: +0.0557 over
its own control, more than the reordering (+0.0591 from a much lower base) and far more
than anything the graph contributed. 9.9's reading — that the subject-dependent protocol
is data-starved — is now directly confirmed rather than inferred.

**Does not settle.** The gain is a *data* effect, not an architectural one. It would very
likely transfer to ShallowConvNet or EEGNet unchanged, and **we have not tested that**. The
claim "our architecture reaches 0.73" is therefore not supported; the supported claim is
"this architecture, plus pre-training that any of these architectures could use, reaches
0.73". Running PTG on ShallowConvNet is the obvious control and it has not been run.

**Still exploratory.** Not on the 10.3 list. Three-for-four here and four-for-four on 9.12
is a reason to take the pre-registration seriously, not a promotion to confirmatory.

---

### 9.16 Is the pre-training gain architecture-agnostic? (PRE-REGISTERED 2026-09-05)

**Written before the run, as 9.12 and 9.14 were. Predictions and decision rules below are
fixed; nothing here may be revised after a number is read.**

#### 9.16.1 The claim under test

`CLAUDE.md` currently forbids "our architecture reaches 0.73" and mandates "this
architecture **plus pre-training that any of these architectures could use** reaches 0.73".
The second clause is an **assertion, not a measurement**. 9.15.5 states it plainly: the gain
"would very likely transfer to ShallowConvNet or EEGNet unchanged, and we have not tested
that". This is that test.

Let

* **Δ_gbp = PTG − PTC = +0.0557** [+0.0464, +0.0650], measured, n = 103, SD 0.0487
* **Δ_shallow = PTG-shallow − PTC-shallow**, to be measured on the same subjects and folds

The primary quantity is the **difference of differences, Δ_shallow − Δ_gbp**, paired on
subject. It is not PTG-shallow's absolute accuracy, which is uninformative on its own.

#### 9.16.2 Why the design is clean, and what would make it dirty

**The two architectures start from the same place.** PTC = 0.6744 and ShallowConvNet =
0.6725 differ by 0.0019, far inside the 0.020 minimum detectable difference fixed in 9.12.
A difference in transfer gain therefore cannot be dismissed as a ceiling or floor artefact —
the usual and usually correct objection to comparing gains across models that start at
different accuracies.

**PTC-shallow is mandatory and is also a validity check.** Pre-training and source-fit
normalisation change together, so only the within-architecture difference isolates
pre-training; this is the same reason PTC exists. Additionally, PTC-shallow should land near
ShallowConvNet's 0.6725, exactly as PTC landed near G3's 0.6737. **If it does not, the
pipeline change is not neutral for this architecture and the difference of differences is
confounded — stop and diagnose rather than reporting a gain.**

#### 9.16.3 Power

Per-subject gain SD is 0.0487 over 103 subjects, so SE = 0.0048 and a single arm-vs-arm gain
has a minimum detectable difference of **0.0134** at 80 % power, two-sided α = 0.05. For the
difference of differences, with the two gains measured on the same subjects and correlating
at ρ:

| ρ between the two architectures' per-subject gains | MDD |
|---|---|
| 0.0 | 0.0190 |
| 0.3 | 0.0159 |
| 0.5 | 0.0134 |
| 0.7 | 0.0104 |

ρ should be **substantially positive**: 9.15 established that the gain is driven by
subject-level data hunger (bottom quartile +0.0720, top +0.0346), which is a property of the
subject, not of the model. So the realistic MDD is **~0.013–0.016** — enough to separate
"transfers fully" from "transfers about half", not enough to resolve differences below 0.010.
**Any |Δ_shallow − Δ_gbp| below 0.010 must be reported as indistinguishable, not as a
direction.**

#### 9.16.4 Predictions, fixed in advance

1. **PTC-shallow lands within 0.020 of ShallowConvNet's 0.6725.** Validity check, not a
   result. Failure halts the analysis.
2. **Δ_shallow > 0.** Pre-training helps ShallowConvNet too.
3. **|Δ_shallow − Δ_gbp| < 0.020** — the gains are equivalent, i.e. the transfer is
   architecture-agnostic. *This is the prediction the paper currently assumes to be true.*
4. **PTG-shallow > 0.70.**

#### 9.16.5 Decision rules, fixed in advance

**A — gains equivalent (|Δ_shallow − Δ_gbp| < 0.020).** The 9.15.5 caveat is confirmed by
measurement instead of asserted. `PAPER.md` contribution 10 and its Limitations entry stand
as written, with "untested" replaced by the measurement. Nothing about the reordered
architecture is special for transfer, and we say so.

**B — Δ_shallow reliably *smaller*.** The reordered graph model benefits more from
pre-training than ShallowConvNet does. This would be a genuine *architectural* claim and the
only outcome in which "our architecture reaches 0.73" becomes defensible — but only if
prediction 1 held, so that it is not a floor effect. It would materially strengthen
contribution 10.

**C — Δ_shallow reliably *larger*.** ShallowConvNet is the better vehicle for the project's
largest effect, and PTG-shallow becomes the project's best number — a model with no graph in
it at all. **This outcome is recorded here in advance precisely because it is the
uncomfortable one**, and the rule is that it gets reported with the same prominence as A or
B. `CLAUDE.md`'s standing instruction not to write "our architecture reaches 0.73" would
then be permanent rather than provisional.

#### 9.16.6 Conditions on reporting

* The eight-point leakage audit of 9.15.3 is re-run on PTG-shallow **before** any number is
  written down. 0.70+ is still the region where leakage is the first hypothesis.
* Both arms must be complete — 1,030 folds each — before anything is quoted. 9.15's own rule.
* **Exploratory, not confirmatory.** Not on the 10.3 list. Same status as 9.13 and 9.15.
* PTG-shallow must never be compared to the 0.650 within-session ceiling, for the reason in
  9.15.4. It consumes ~92 other subjects.

#### 9.16.7 Cost, measured rather than estimated

Error 24 was an 8x cost miss from carrying one architecture's timing to another, and the
first estimate for *this* run repeated it in miniature — it scaled ShallowConvNet's
subject-dependent per-fold time (13.2 s against `graphbp`'s 17.0 s) and concluded shallow was
0.78x cheaper. Measured at pre-training scale it is **0.68x the throughput, i.e. slower**:
the subject-dependent figure was dominated by `graphbp`'s graph construction, which
ShallowConvNet does not do, not by compute.

| | windows | epochs | sample-passes | measured |
|---|---|---|---|---|
| one source model | 144,855 (87 subjects) | 30 | 4.35 M | 8.0 min `graphbp`, ~11.8 min shallow |
| x 20 (10 groups x 2 problems) | | | 87.0 M | 2.67 h / **3.92 h** |
| one fine-tune fold | 1,332 (4/5 of one subject) | 30 | 0.04 M | 4.3 s `graphbp`, ~6.8 s shallow |
| x 1,030 folds | | | 41.2 M | 1.21 h / **1.94 h** |

Source models are **68 % of the sample-passes and ~70 % of the wall clock**. PTG-shallow
≈ 5.9 h, PTC-shallow ≈ 1.9 h, **total ≈ 7.8 h**. Cheaper alternatives were considered and
rejected: dropping to 5 groups would save 2.2 h but give ShallowConvNet's source models 82
subjects where `graphbp`'s had 92, confounding the one comparison this run exists to make.

---

### 9.17 Is the pre-training gain architecture-agnostic? — results (EXPLORATORY)

**Run 2026-09-05/06, 2,060 folds over two arms, 103 subjects, identical folds to every
other arm. Pre-registered in 9.16 at commit `4f9f819` before any number existed.
Four predictions of four hold. The eight-point audit of 9.15.3 passed on all eight
before any figure below was read.**

#### 9.17.1 Results

| arm | L vs R | B vs F | within-family | above chance |
|---|---|---|---|---|
| PTC-gbp — control | 0.6533 | 0.6956 | 0.6744 | 89/103 |
| **PTG-gbp** | 0.7206 | 0.7396 | **0.7301** | 98/103 |
| PTC-shallow — control | 0.6714 | 0.7057 | 0.6886 | 92/103 |
| **PTG-shallow** | **0.7387** | 0.7351 | **0.7369** | **101/103** |

| comparison | difference | 95 % CI | wins |
|---|---|---|---|
| gain, graphbp (PTG − PTC) | **+0.0557** | [+0.0464, +0.0650] | 93/103 |
| gain, shallow (PTG − PTC) | **+0.0483** | [+0.0379, +0.0590] | 82/103 |
| **DiD: shallow − graphbp** | **−0.0073** | [−0.0204, +0.0056] | 47/103 |
| head to head: PTG-shallow − PTG-gbp | +0.0068 | [−0.0022, +0.0155] | 59/103 |

#### 9.17.2 Prediction scorecard — four of four

1. *"PTC-shallow within 0.020 of ShallowConvNet's 0.6725."* — **Holds.** 0.6886, +0.0161.
2. *"Δ_shallow > 0."* — **Holds.** +0.0483 on 82/103.
3. *"|Δ_shallow − Δ_gbp| < 0.020."* — **Holds.** 0.0073.
4. *"PTG-shallow > 0.70."* — **Holds.** 0.7369.

#### 9.17.3 Decision rule A fires: the gains are equivalent

9.16.3 fixed a floor before the run: **any |DiD| below 0.010 is reported as
indistinguishable, not as a direction.** The DiD is **−0.0073**, its CI contains zero, and
it wins on 47 of 103 subjects — a coin flip. **Pre-training transfers to ShallowConvNet at
the same magnitude it transfers to the reordered graph model.**

The 9.15.5 caveat is therefore **confirmed by measurement rather than asserted**. The gain
is a property of the data, not of the architecture, and `CLAUDE.md`'s standing instruction —
write "this architecture plus pre-training", never "our architecture reaches 0.73" — is now
permanent rather than provisional.

**The head-to-head is also a tie, and must be reported as one.** PTG-shallow's 0.7369
exceeds PTG-gbp's 0.7301 by 0.0068, which is below the same 0.010 floor and carries a CI
containing zero. **Do not write that ShallowConvNet is the project's best model**; the
supported claim is that the two are indistinguishable at the top of the table. The split is
informative: shallow wins left-versus-right (0.7387 vs 0.7206) and loses both-fists-versus-feet
(0.7351 vs 0.7396), netting to a wash.

#### 9.17.4 Two things the analysis got wrong in advance

**The power estimate was optimistic.** 9.16.3 argued the two architectures' per-subject
gains would correlate "substantially", putting the MDD at 0.013–0.016. Measured
**ρ = +0.160** — weakly positive, so the effective MDD is near the uncorrelated 0.0190. The
design was therefore *less* able to resolve a small interaction than pre-registered. The
conclusion is unaffected, since a DiD of −0.0073 is inside even the optimistic floor, but a
DiD of 0.015 would have been reported as indistinguishable when it might have been real.

**The pre-run projection expected outcome C and got A.** From PTC-shallow's completed 0.6886
it was argued that shallow needed only +0.0415 to pass PTG-gbp, against graphbp's +0.0557,
and would therefore surpass it. Shallow did gain +0.0483 — more than the 0.0415 — and did
land nominally ahead. But the projection treated a nominal lead as a win, and 0.0068 is not
one. **Predicting the point estimate correctly and the verdict wrongly is exactly what the
pre-registered floor exists to catch.**

#### 9.17.5 What this settles and what it does not

**Settles.** Pre-training is architecture-agnostic on this dataset, measured on two
architectures that differ in every component except the folds they train on. The largest
effect in this project belongs to the data, not to any model in it.

**Does not settle.** Two architectures are not "any architecture" — EEGNet is untested, and
both arms here take power after a spatial stage. Nor does it explain *why* shallow gains
slightly less from a slightly higher baseline; the regression of gain on baseline (slope
−0.086) is too weak to carry that, and is biased by shared noise.

**Still exploratory.** Not on the 10.3 list.

**The 9.15.4 prohibition extends to this arm.** PTG-shallow's 0.7387 on left-versus-right
must never be set against [23]'s 0.650. It consumes ~92 other subjects' recordings; that
benchmark is within-session.

---

### 9.18 Is the graph the wrong prior, rather than priors being useless? (PRE-REGISTERED 2026-09-06)

**Written and committed before the run. Predictions and decision rules are fixed.**

#### 9.18.1 The claim under test

9.13 found the electrode graph inert: hard constraint (G1 0.6629), free operator
(G2 0.6694) and Dirichlet prior (G3 0.6737) span 0.0108 and are mutually
indistinguishable. The natural reading is "a spatial prior buys nothing here".

CSP-Net (arXiv:2411.11879) reports the opposite for a *different* prior on the *same*
operator: initialising a CNN's spatial layer from CSP filters gains **+6.0 points** over a
ShallowConvNet backbone averaged across four MI datasets (70.29 -> 76.26), and their
ablation separates prior from capacity -- a randomly initialised variant barely beats the
backbone. If that transfers, 9.13's conclusion should be narrowed from "spatial priors do
not help" to **"the electrode graph is the wrong prior"**, which is a sharper and more
useful claim.

#### 9.18.2 Design

**One variable.** `models/csp_prior.py` builds a basis of identical shape and scaling to
the Chebyshev one -- CSP operator `P = W.T @ W` rescaled to spectrum [-1, 1], then the same
recurrence `T0 = I`, `T1 = P`, `T2 = 2P^2 - I`. `FreeSpatialConv` copies whatever basis it
is handed into its learnable `operators`, so **no model code changes at all**; the arms
differ only in what the operator is initialised from.

| arm | spatial operator | initialised from |
|---|---|---|
| **G2** (existing control) | free, learnable | the electrode graph's Chebyshev basis |
| **CI** (new) | free, learnable | CSP filters on the fold's training windows |

CSP is supervised and is fitted on **training windows only**, in the node ordering the model
sees, exactly as the correlation graph is. `--csp-init` is rejected with `--spatial graph`,
because a fixed CSP basis would be a hard constraint rather than a prior -- a different
experiment.

#### 9.18.3 Predictions, fixed in advance

1. **CI > G2.** Directional. The prior is the only difference.
2. **The gain exceeds the 0.020 MDD** fixed in 9.12, i.e. this is not another
   graph-sized non-effect.
3. **CI > CSP+LDA's 0.6239** by more than G2 does.
4. **CI does not reach PTG's 0.7301**, since that arm additionally consumes ~92 other
   subjects and this one does not.

#### 9.18.4 Decision rules, fixed in advance

**A -- CI beats G2 by more than 0.020.** 9.13's conclusion is narrowed: the graph is the
wrong prior, not evidence that priors are useless. This becomes the paper's constructive
finding and the natural successor experiment is CSP-init *plus* pre-training.

**B -- CI and G2 are indistinguishable (|diff| < 0.020).** The spatial prior really does buy
nothing on this dataset regardless of its source, and CSP-Net's gain does not transfer to
EEGMMIDB under a correct protocol. 9.13 stands as written and is strengthened.

**C -- CI is reliably worse.** The CSP subspace is a worse starting point than the graph,
which would be surprising and would need a diagnostic before anything is claimed.

#### 9.18.5 Conditions on reporting

* Both problems complete at 1,030 folds before any figure is quoted.
* Paired on subject against G2 at the same seed and folds; Holm-corrected within this
  family; bootstrap CI over 10,000 resamples.
* **Exploratory, not confirmatory.** Not on the 10.3 list.
* The 9.15.4 prohibition applies: no comparison to the 0.650 within-session ceiling for any
  arm that consumes other subjects. CI does not, so CI *is* comparable to it.

---

### 9.19 The CSP prior, and the tangent-space baseline — results (EXPLORATORY)

**Run 2026-09-06. A3 is 1,030 folds, CI is 1,030 folds, both on the same 103 subjects and
the same folds as every other arm. CI was pre-registered in 9.18 at commit `d6f21d4`
before any number existed. One of four predictions holds, and it is the trivial one.**

#### 9.19.1 Results

| arm | L vs R | B vs F | within-family | above chance |
|---|---|---|---|---|
| CSP+LDA | 0.5942 | 0.6537 | 0.6239 | 81/103 |
| G1 — hard graph constraint | 0.6408 | 0.6850 | 0.6629 | 88/103 |
| **CI — CSP-initialised operator** | 0.6509 | 0.6863 | **0.6686** | 92/103 |
| G2 — graph-initialised operator (control) | 0.6494 | 0.6893 | 0.6694 | 93/103 |
| **A3 — cov-tgsp (new)** | 0.6550 | 0.6837 | **0.6694** | 89/103 |
| ShallowConvNet | 0.6608 | 0.6841 | 0.6725 | 93/103 |
| G3 — graph as Dirichlet prior | 0.6520 | 0.6954 | 0.6737 | 90/103 |

| comparison | difference | 95 % CI | wins |
|---|---|---|---|
| **CI vs G2** (isolates the prior) | **−0.0008** | [−0.0094, +0.0076] | 49/103 |
| CI vs G3 | −0.0051 | [−0.0134, +0.0030] | 46/103 |
| CI vs ShallowConvNet | −0.0039 | [−0.0150, +0.0072] | 50/103 |
| CI vs CSP+LDA | +0.0447 | [+0.0282, +0.0612] | 68/103 |
| **A3 vs CSP+LDA** | **+0.0454** | [+0.0310, +0.0594] | 74/103 |
| A3 vs ShallowConvNet | −0.0031 | [−0.0164, +0.0101] | 47/103 |

#### 9.19.2 Prediction scorecard — one of four, and it is the trivial one

1. *"CI > G2."* — **REFUTED.** −0.0008 on 49/103, a coin flip.
2. *"The gain exceeds the 0.020 MDD."* — **REFUTED.**
3. *"CI beats CSP+LDA by more than G2 does."* — **REFUTED.** G2 leads CSP by +0.0455,
   CI by +0.0447.
4. *"CI does not reach PTG's 0.7301."* — Holds, at 0.6686. Trivially.

**Decision rule B fires.** CSP-Net reports +6.0 points on a ShallowConvNet backbone across
four MI datasets from exactly this intervention, with an ablation showing it is the
initialisation and not the added capacity. **It does not transfer to EEGMMIDB under a
correct protocol.** 9.13 stands as written and is strengthened rather than narrowed.

#### 9.19.3 What this actually establishes: a ceiling, not a prior

The interesting reading is not CI alone but the column it joins. **Seven methods that share
almost nothing now land between 0.6629 and 0.6737 — a span of 0.0108, under the 0.020 MDD:**

* four different priors on the *same* learnable spatial operator — hard Chebyshev graph
  (G1), graph initialisation (G2), Dirichlet graph penalty (G3), CSP subspace (CI);
* a graph-free CNN (ShallowConvNet, 0.6725);
* a Riemannian classical method with no network at all (A3, 0.6694).

9.13 concluded "the graph contributes nothing". The stronger and better-supported claim is
now that **the spatial prior does not matter at all on this data, whatever it is, and
~0.67 is the within-subject ceiling.** Only cross-subject pre-training breaks it
(PTG 0.7301, PTG-shallow 0.7369), and 9.17 established that is a data effect rather than an
architectural one. Both findings point the same way: on this dataset the binding constraint
is how much data a subject-dependent fold has, not what is built on top of it.

#### 9.19.4 A3 puts a number on a caveat that has been unquantified since 14

14 has recorded from the start that CSP was left untuned, making H4's negative result
conservative, but never said by how much. **cov-tgsp beats CSP+LDA by +0.0454 on 74/103
subjects.** This makes the paper's central comparison harder on our own methods, which is
the correct direction for a critical paper, and it independently reproduces [23]'s family
ranking on PhysionetMI (cov-tgsp and CSP the top two, cov-tgsp ahead) under a protocol we
control.

`scripts/run_tangent_space.py` implements the affine-invariant Frechet mean and tangent-space
projection directly rather than adding `pyriemann`, per `CLAUDE.md`'s rule on dependencies.
The reference point, scaler and classifier are all fitted on training trials only.

#### 9.19.5 Provenance notes

* **A smoke test read 0.7926 on three subjects**, above every arm in the project. Checked
  before use: on those same subjects CSP reaches 0.700, ShallowConvNet 0.785 and PTG-shallow
  0.830. S001 and S002 are simply easy subjects. Small-sample selection, not leakage.
* **Nine duplicate rows** were removed from `CI-cspinit-binary-2-3`. Two processes ran the
  same tag concurrently during a botched relaunch and, sharing a seed and folds, wrote
  byte-identical rows — verified identical in value before deduplicating, so the removal is
  lossless. See error 26.
* **Exploratory, not confirmatory.** Neither arm is on the 10.3 list.

---

### 9.20 Window length (B5) — pre-registered 2026-09-06, dev set

**Written and committed before the run.**

#### 9.20.1 Why this is now the most interesting axis left

Every raw arm in this project uses `window=64` = **0.40 s**, and that value has never been
varied for any arm. It was inherited, not chosen. Two things make it the obvious next test:

* **Neurophysiology.** ERD/ERS develops over 1-2 s and the literature analyses windows
  around 0.5-2.5 s post-cue. 0.40 s may simply be shorter than the phenomenon. Note the
  countervailing evidence: brief (2 s) motor imagery has been reported to yield *more*
  informative ERS features than sustained (4 s), so longer is not automatically better --
  which is exactly why this needs measuring rather than assuming.
* **9.19 removed every other candidate.** Seven methods sharing almost nothing land within
  0.0108 of each other. When four different spatial priors, a graph-free CNN and a
  Riemannian classical method all agree on ~0.67, the thing they *share* becomes the
  suspect -- and what they share is the 0.40 s window and the 8-30 Hz band.

It also bears directly on a stated limitation: 7 records that EEGNet and ShallowConvNet were
run far outside the 2-4 s regime their published hyperparameters assume, making their
advantage conservative. Lengthening the window moves them toward that regime.

#### 9.20.2 Design

**ShallowConvNet on the development set (S001-S020), `stride=16` held fixed.** ShallowConvNet
because 9.19 established the architecture does not matter here, so the cheapest arm is the
right probe; the development set because this is a *sweep* to choose a value, which is
exactly what 10.1 demoted those subjects for. `window=64` is not re-run -- the existing
`A2-shallow-binary` rows restricted to those 20 subjects are the baseline (0.6733, against
0.6725 on all 103, so the subset is representative).

| window | seconds | windows/trial | relative cost |
|---|---|---|---|
| 64 (baseline, already run) | 0.40 | 37 | 1.00x |
| **128** | 0.80 | 33 | 1.78x |
| **256** | 1.60 | 25 | 2.70x |

**Power.** The paired SD on these 20 subjects is 0.0524, giving SE 0.0117 and a
**minimum detectable difference of 0.0328** at 80 % power -- considerably coarser than the
0.020 available at 103 subjects. **Any dev-set difference below 0.033 must be reported as
indistinguishable**, and this sweep can only justify a confirmatory run, never conclude one.

#### 9.20.3 Predictions, fixed in advance

1. **A longer window beats 0.40 s.** Directional; the mechanism is that 0.40 s truncates
   ERD/ERS and gives ~2.5 Hz frequency resolution, ~9 bins inside 8-30 Hz.
2. **The best window is 128 or 256, not the longest available.** The brief-MI literature
   argues against monotonic improvement.
3. **The gain exceeds the 0.0328 dev-set MDD**, i.e. it is not another 9.19-style tie.

#### 9.20.4 Decision rules, fixed in advance

**A -- a window beats 64 by more than 0.0328.** Run that window confirmatorily on all 103
subjects, and treat every existing arm's window as a limitation rather than a choice. This
would mean the ~0.67 ceiling in 9.19.3 is partly an artefact of the sample definition, which
would be the most consequential finding since 9.13.

**B -- all windows within 0.0328 of each other.** The 0.40 s window is not the constraint,
the 9.19 ceiling stands, and 7's caveat about ShallowConvNet's design regime can be
retired as measured rather than assumed.

**C -- longer windows are reliably worse.** Would suggest the short window is doing useful
implicit augmentation (37 overlapping views per trial) and that trial-level aggregation is
carrying more than assumed.

**Reporting.** Exploratory; dev-set only; no number from this sweep may be quoted as a
result for the paper, only as a reason to run a confirmatory arm.

---

### 9.21 Window length — results (EXPLORATORY, development set)

**Run 2026-09-06, 400 folds over two new windows, ShallowConvNet, S001-S020, stride 16.
Pre-registered in 9.20 at commit `7a30d35`. Zero of three predictions hold.**

#### 9.21.1 Results

| window | L vs R | B vs F | within-family |
|---|---|---|---|
| **64 (0.40 s)** — the inherited value | 0.6522 | 0.6944 | **0.6733** |
| 128 (0.80 s) | 0.6356 | 0.6822 | 0.6589 |
| 256 (1.60 s) | 0.6144 | 0.6822 | 0.6483 |

| comparison | difference | 95 % CI | wins |
|---|---|---|---|
| 128 vs 64 | −0.0144 | [−0.0300, +0.0011] | 9/20 |
| 256 vs 64 | −0.0250 | [−0.0439, −0.0067] | 5/20 |

Total spread across a **fourfold** window range: **0.0250**.

#### 9.21.2 Prediction scorecard — zero of three

1. *"A longer window beats 0.40 s."* — **REFUTED, and in the opposite direction.** Both
   longer windows are numerically worse and the trend is monotone.
2. *"The best window is 128 or 256, not the longest."* — **REFUTED.** The best is 64.
3. *"The gain exceeds the 0.0328 dev-set MDD."* — **REFUTED.** There is no gain.

**Decision rule B fires.** Every window sits within the 0.0328 threshold 9.20.2 fixed in
advance, so all three must be reported as indistinguishable. The 0.40 s window is not what
holds this dataset at ~0.67.

**The honest tension.** The 256 comparison's CI excludes zero, and the decline is monotone
across three points. Under the pre-registered rule that is still "indistinguishable", and it
is reported as such — but a confirmatory run at 103 subjects, where the MDD falls to 0.020,
might well find 1.60 s reliably *worse*. That would be decision rule C, not B.

#### 9.21.3 A confound in this sweep, stated rather than buried

**Stride was held at 16, so a longer window means fewer training windows per trial: 37 at
64 samples, 33 at 128, 25 at 256 — a 32 % cut in training samples between the extremes.**
This project has established repeatedly that the subject-dependent protocol is data-starved
(9.9, and pre-training's +0.0557 in 9.15). A 32 % reduction in samples is therefore a
plausible sole explanation for a 0.0250 decline, and **this design cannot separate "longer
windows are worse" from "fewer samples are worse".**

The headline is unaffected -- no window *improves* on 0.40 s, which is what the sweep was
run to find out -- but rule C's mechanism cannot be established from these arms. Holding
windows-per-trial constant by scaling stride with window would disentangle it, and has not
been run.

#### 9.21.4 What this settles

**Settles.** The window is not the hidden constraint. Combined with 9.19, the ~0.67
within-subject ceiling now survives: four spatial priors, three architectures, a Riemannian
classical method, *and* a fourfold change in the sample definition. That is a much stronger
ceiling claim than any single arm supports.

**Retires a limitation.** 7 records that EEGNet and ShallowConvNet ran far outside the 2-4 s
regime their published hyperparameters assume, and calls their advantage "conservative" on
that basis. At 1.60 s -- inside that regime -- ShallowConvNet is *worse*, not better. The
caveat can now be stated as measured rather than assumed, and it was not costing them
anything.

**Does not settle.** Dev set only, one architecture, and the sample-count confound above.
No number here may be quoted as a paper result; 9.20.4 restricted this sweep to justifying a
confirmatory run, and it does not justify one.

---

### 9.22 Domain-specific batch normalisation (PRE-REGISTERED 2026-09-07)

**Written and committed before the run.**

#### 9.22.1 Why this, after 9.19 and 9.21 found nothing

Those two closed off *spatial priors* and the *sample definition*. Both act **within** a
subject, and both landed inside the noise. This acts **between** subjects, which is the only
axis that has ever moved the ceiling: 9.17 showed pre-training breaks 0.67 by borrowing
other subjects' data, and does so architecture-agnostically.

9.15's pre-training pools ~92 subjects through a **single** BatchNorm — one set of running
statistics describing 92 people whose amplitude and covariance structure differ
substantially — while 9.10 measured that ~30 % of the between-subject variance is fixed
before any modelling. TSMNet (arXiv:2206.01323) isolates exactly this on 5 datasets and 158
subjects and finds removing domain-specific normalisation costs **2.4–3.9 %**, the second
largest term in their ablation.

This is a change to *how* the borrowing is done, not another prior on top of it.

#### 9.22.2 Design

**One variable against the existing PTG control.** Every source subject gets its own
batch-norm statistics; **all other parameters, including the affine scale and shift, are
shared** — which is what makes this a normalisation fix rather than 92 models. Batches
become domain-homogeneous, since a batch mixing subjects has no single correct set of
statistics. The target gets a reserved slot seeded from the source mean, then adapts it
during fine-tuning.

| arm | source normalisation |
|---|---|
| **PTG** (existing control) | one BatchNorm shared across ~92 subjects |
| **PTD** (new) | one set of statistics per source subject |

**The default path is verified untouched.** Re-running the non-DSBN smoke after the wiring
reproduces 0.5520 ± 0.0240, bit-identical to the value measured before any of it, so every
previously logged arm remains comparable. `--dsbn` is part of the source-model cache key, so
a DSBN source model can never be reused for a non-DSBN run.

#### 9.22.3 Predictions, fixed in advance

1. **PTD > PTG.** Directional.
2. **The gain exceeds the 0.020 MDD** of 9.12.
3. **The gain is larger in the weakest quartile**, as pre-training's was (9.15) — if the
   mechanism is better handling of between-subject variability, it should act where that
   variability hurts most.
4. **PTD stays below 0.80.** A larger jump would be the signature of a leak, not an effect.

#### 9.22.4 Decision rules, fixed in advance

**A — PTD beats PTG by more than 0.020.** The first architectural-side intervention in this
project to move anything since the reordering, and it does so on the between-subject axis
rather than the within-subject one. It would also strengthen 9.17: the pre-training effect
would have identifiable structure rather than being a black-box data gain.

**B — indistinguishable (|diff| < 0.020).** Then the ceiling survives its fourth challenger,
and the pre-training gain is not obviously attributable to normalisation. Given 9.19 and
9.21, this is the outcome to expect.

**C — PTD reliably worse.** Would suggest 92 separate statistics fragment an estimate that
pooling makes more robust — plausible at ~1,665 windows per subject, and worth reporting
because it bounds how far per-domain modelling can be pushed at this sample size.

#### 9.22.5 Conditions on reporting

* Both problems complete at 1,030 folds before anything is quoted.
* The eight-point leakage audit of 9.15.3 re-runs on PTD first.
* Paired against PTG at the same seed and folds; bootstrap CI over 10,000 resamples.
* **Exploratory, not confirmatory.** Not on the 10.3 list.
* The 9.15.4 prohibition applies: **never compare** PTD to the 0.650 within-session
  ceiling — it is a different protocol with ~92 extra subjects, not a better pipeline.

---

### 9.23 Domain-specific batch normalisation — results (EXPLORATORY)

**Run 2026-09-07, 1,030 folds, 103 subjects, identical folds to PTG. Pre-registered in 9.22
at commit `b4f75fe` before any number existed. The eight-point audit passed on all eight
before anything below was read. One of four predictions holds, and it is the trivial one.**

#### 9.23.1 Results

| arm | L vs R | B vs F | within-family | above chance |
|---|---|---|---|---|
| PTC — no pre-training | 0.6533 | 0.6956 | 0.6744 | 89/103 |
| **PTG — shared BatchNorm (control)** | 0.7206 | 0.7396 | **0.7301** | 98/103 |
| **PTD — per-subject BatchNorm** | 0.7172 | 0.7277 | **0.7224** | 95/103 |

| comparison | difference | 95 % CI | wins |
|---|---|---|---|
| **PTD vs PTG** (isolates DSBN) | **−0.0077** | [−0.0164, +0.0013] | **39/103** |

#### 9.23.2 Prediction scorecard — one of four

1. *"PTD > PTG."* — **REFUTED.** The point estimate is negative.
2. *"The gain exceeds the 0.020 MDD."* — **REFUTED.** There is no gain.
3. *"The gain is larger in the weakest quartile."* — **Not evaluable**; there is no gain to
   distribute. For the record the by-quartile differences are +0.0073, −0.0175, −0.0062,
   −0.0141, so the weakest quartile is the only one that is not negative — weakly consistent
   with the mechanism, on an effect that does not exist.
4. *"PTD stays below 0.80."* — Holds, at 0.7224. Trivially.

**Decision rule B fires.** |−0.0077| is inside the 0.020 threshold 9.12 fixed, so DSBN is
reported as indistinguishable from a shared BatchNorm. TSMNet's 2.4–3.9 % does not transfer
to this pipeline.

#### 9.23.3 The tension, stated rather than smoothed

**The magnitude says "indistinguishable" and the direction says "slightly worse", and these
disagree.** The mean difference is well inside the MDD, but PTD wins on only **39 of 103**
subjects — a sign test gives **p = 0.0176**. A consistent small decrement is exactly what
decision rule C anticipated: *"92 separate statistics fragment an estimate that pooling makes
more robust — plausible at ~1,665 windows per subject."* Each source subject supplies roughly
1,665 windows to estimate its own running statistics, against ~145,000 pooled; per-domain
estimates are correspondingly noisier.

The pre-registered rule is stated on effect size, so **B is what we report**. But the honest
summary is that DSBN is not merely neutral here — it is very slightly and quite consistently
harmful, and a confirmatory run designed around the sign rather than the mean would probably
detect it.

#### 9.23.4 What this settles

**The ceiling survives its fourth challenger.** 9.19 closed off spatial priors, 9.21 the
sample definition, and this closes off per-subject normalisation. All three were selected
because published work reported gains of 2–6 points from them, and all three measured
nothing here.

**It also sharpens 9.17.** Pre-training's +0.0557 is not obviously attributable to better
handling of between-subject variability — the most direct intervention on that axis does not
reproduce it. Whatever pre-training buys, it is not normalisation.

**Does not settle.** One architecture, one pipeline, one dataset. TSMNet's result is obtained
in *inter-session and inter-subject transfer*, where a model is evaluated on an unseen
domain; our fine-tuning stage gives every target its own labelled data, which is a friendlier
setting and may leave less for domain-specific normalisation to fix.

**Still exploratory.** Not on the 10.3 list. The 9.15.4 prohibition applies: **never
compare** PTD to the 0.650 within-session ceiling — it consumes ~92 other subjects and is
a different protocol, not a better pipeline.

---

### 9.24 Montage robustness — pre-registration (written before any result)

**Written 2026-11-04, about 23:25 local, with zero `ROB` rows in `runs/experiments.jsonl`
(`grep -c '"tag": "ROB' runs/experiments.jsonl` prints 0). The commit timestamp of the
pre-registration commit is authoritative; the clock time here is approximate.** The commit hash recorded here,
`9d1ba9c`, is the HEAD the section was written on top of; the commit that *contains* this
section (message "Pre-register the montage-robustness experiment") is the pre-registration
commit, and every `ROB` run starts after it. Design: `docs/superpowers/specs/2026-11-04-montage-robustness-design.md`.
**Phase 1 is EXPLORATORY.** Not on the 10.3 list.

#### 9.24.1 Question

On the full 64-electrode task the graph model and ShallowConvNet tie once both are
pre-trained (9.17). A Chebyshev graph convolution shares its weights across nodes, so the
same filter can run on any electrode subset for which a graph exists; ShallowConvNet's
spatial layer has one weight per electrode and only runs on the 64 it was trained on. **If
that flexibility is real, a graph model trained once on the full cap should degrade more
gracefully than ShallowConvNet when electrodes are missing at test time.** This tests that,
under rules fixed here, against the strongest fair baseline (interpolation, then
re-referencing). It is not an attempt to raise full-montage accuracy and nothing in it is
tuned.

#### 9.24.2 Region table (generated by `scripts/print_region_table.py`, never changed)

The region-pooled readout (spec 3.1) pools log band power within nine fixed regions,
{left, midline, right} x {frontal, central, posterior}, assigned from the electrode *name*
by `data.montage.region_of`: suffix `z` -> midline, odd -> left, even -> right; prefixes Fp,
AF, F, FC, FT -> frontal; C, T, CP, TP -> central; P, PO, O, I -> posterior. The output
below is pasted verbatim from the script, run against `data/cache_bp830` channel names.

| # | channel | region |
|---|---|---|
| 0 | FC5 | left-frontal |
| 1 | FC3 | left-frontal |
| 2 | FC1 | left-frontal |
| 3 | FCz | midline-frontal |
| 4 | FC2 | right-frontal |
| 5 | FC4 | right-frontal |
| 6 | FC6 | right-frontal |
| 7 | C5 | left-central |
| 8 | C3 | left-central |
| 9 | C1 | left-central |
| 10 | Cz | midline-central |
| 11 | C2 | right-central |
| 12 | C4 | right-central |
| 13 | C6 | right-central |
| 14 | CP5 | left-central |
| 15 | CP3 | left-central |
| 16 | CP1 | left-central |
| 17 | CPz | midline-central |
| 18 | CP2 | right-central |
| 19 | CP4 | right-central |
| 20 | CP6 | right-central |
| 21 | Fp1 | left-frontal |
| 22 | Fpz | midline-frontal |
| 23 | Fp2 | right-frontal |
| 24 | AF7 | left-frontal |
| 25 | AF3 | left-frontal |
| 26 | AFz | midline-frontal |
| 27 | AF4 | right-frontal |
| 28 | AF8 | right-frontal |
| 29 | F7 | left-frontal |
| 30 | F5 | left-frontal |
| 31 | F3 | left-frontal |
| 32 | F1 | left-frontal |
| 33 | Fz | midline-frontal |
| 34 | F2 | right-frontal |
| 35 | F4 | right-frontal |
| 36 | F6 | right-frontal |
| 37 | F8 | right-frontal |
| 38 | FT7 | left-frontal |
| 39 | FT8 | right-frontal |
| 40 | T7 | left-central |
| 41 | T8 | right-central |
| 42 | T9 | left-central |
| 43 | T10 | right-central |
| 44 | TP7 | left-central |
| 45 | TP8 | right-central |
| 46 | P7 | left-posterior |
| 47 | P5 | left-posterior |
| 48 | P3 | left-posterior |
| 49 | P1 | left-posterior |
| 50 | Pz | midline-posterior |
| 51 | P2 | right-posterior |
| 52 | P4 | right-posterior |
| 53 | P6 | right-posterior |
| 54 | P8 | right-posterior |
| 55 | PO7 | left-posterior |
| 56 | PO3 | left-posterior |
| 57 | POz | midline-posterior |
| 58 | PO4 | right-posterior |
| 59 | PO8 | right-posterior |
| 60 | O1 | left-posterior |
| 61 | Oz | midline-posterior |
| 62 | O2 | right-posterior |
| 63 | Iz | midline-posterior |

**Full cap** (64 channels)

| region | n | channels |
|---|---|---|
| left-frontal | 11 | FC5, FC3, FC1, Fp1, AF7, AF3, F7, F5, F3, F1, FT7 |
| midline-frontal | 4 | FCz, Fpz, AFz, Fz |
| right-frontal | 11 | FC2, FC4, FC6, Fp2, AF4, AF8, F2, F4, F6, F8, FT8 |
| left-central | 9 | C5, C3, C1, CP5, CP3, CP1, T7, T9, TP7 |
| midline-central | 2 | Cz, CPz |
| right-central | 9 | C2, C4, C6, CP2, CP4, CP6, T8, T10, TP8 |
| left-posterior | 7 | P7, P5, P3, P1, PO7, PO3, O1 |
| midline-posterior | 4 | Pz, POz, Oz, Iz |
| right-posterior | 7 | P2, P4, P6, P8, PO4, PO8, O2 |

**IV-2a montage** (22 channels)

| region | n | channels |
|---|---|---|
| left-frontal | 2 | FC3, FC1 |
| midline-frontal | 2 | Fz, FCz |
| right-frontal | 2 | FC2, FC4 |
| left-central | 5 | C5, C3, C1, CP3, CP1 |
| midline-central | 2 | Cz, CPz |
| right-central | 5 | C2, C4, C6, CP2, CP4 |
| left-posterior | 1 | P1 |
| midline-posterior | 2 | Pz, POz |
| right-posterior | 1 | P2 |

**10-20 montage** (19 channels)

| region | n | channels |
|---|---|---|
| left-frontal | 3 | Fp1, F7, F3 |
| midline-frontal | 1 | Fz |
| right-frontal | 3 | Fp2, F4, F8 |
| left-central | 2 | T7, C3 |
| midline-central | 1 | Cz |
| right-central | 2 | C4, T8 |
| left-posterior | 3 | P7, P3, O1 |
| midline-posterior | 1 | Pz |
| right-posterior | 3 | P4, P8, O2 |

#### 9.24.3 Conditions and the two primaries

Both models are pre-trained and fine-tuned on all 64 channels. Every condition is applied
**only at test time**, to the held-out fold of each subject, with the same channel pattern
for both models.

| condition (log name) | electrodes available | role |
|---|---|---|
| **Full** (no `<tag>.full` row) | all 64 | reference; measures what the region readout costs; scored by the tag's clean row, from the referenced cache exactly as in PTG |
| **IV-2a montage** (`iv2a`) | the 22 channels listed above | **Primary 1** |
| **Random 25 %** (`random25`) | 48 of 64 (16 removed) | **Primary 2** |
| 10-20 montage (`tentwenty`) | the 19 channels listed above | secondary |
| Random 10 % (`random10`) | 58 of 64 (6 removed) | secondary |
| Random 50 % (`random50`) | 32 of 64 (32 removed) | secondary |
| Clustered 8 (`cluster8`) | 8 removed | secondary |
| Clustered 16 (`cluster16`) | 16 removed | secondary |

Secondary conditions are exploratory and labelled so wherever they appear; they never enter
the win decision.

**Pattern generation.** Random conditions remove a fraction (10 % -> 6 removed / 58 kept,
25 % -> 16 / 48, 50 % -> 32 / 32). Clustered conditions remove the given count (8 or 16): a
random centre plus its nearest neighbours by 3-D `standard_1005` distance. Ten draws per
(subject, fold), seeded from (subject, fold, draw, attempt); a draw that empties any of the
nine regions is rejected and redrawn. The rejection depends only on the seed and the region
table, never on the data, and the same accepted mask is used for M1 and M2. Per-fold
accuracy is the mean over the ten draws.

*Disclosed consequence of the seed.* The seed does not include the condition name, so within
one draw the random10 / random25 / random50 masks are nested, and cluster8 / cluster16 share
a centre. That makes the degradation curve across fractions paired. The two primaries are
unaffected: each is scored on its own masks against the other model's identical masks.

**Re-referencing.** The standard cache is average-referenced over all 64 channels, so masking
afterwards would leave every surviving channel carrying the absent ones through the
reference. Every non-Full condition therefore re-references the test trials over the
electrodes present, from a no-reference cache (`data/cache_bp830_noref`, fingerprint
`881699d2fa46d9e3`). Full is not re-scored from that cache: it is the tag's clean row, from
the referenced cache, and there is no `<tag>.full` row. Average referencing and band-pass
filtering commute, and this is checked: re-referencing the no-reference cache over all 64
channels reproduces the referenced cache for **103 of 103** subjects
(`scripts/check_noref_cache.py`: same trials, order, labels and channel names), which is
the equivalence for Full.

#### 9.24.4 Arms

| arm | model | tags (condition rows `"<tag>.<condition>"`) | handling of missing channels |
|---|---|---|---|
| **M1** | graph-convolution GBP: `--model graphbp --spatial graph --readout region` (Chebyshev, K = 2, one layer, no node pooling, region-pooled readout, 9 x 40 = 360 features) | `ROB1-chebrp-binary-0-1`, `ROB1-chebrp-binary-2-3` | drops the missing nodes; the graph and per-channel standardiser are **recomputed from source-training subjects re-referenced over the same electrode subset** (exact, from moments) |
| **M2** | ShallowConvNet, reusing the cached PTG-shallow source models: `--source-tag PTG-shallow-binary-0-1` / `-2-3` | `ROB1-shallow-binary-0-1`, `ROB1-shallow-binary-2-3` | re-reference the present electrodes, reconstruct the missing ones by spherical-spline interpolation exactly as MNE `interpolate_bads` (head-frame positions, sphere origin fitted on the full 64-channel cap), common average over all 64, then the source standardiser |

Development subjects are **S001-S020**; groups are always assigned over all 103 subjects
(`index % 10`), so a development target never sits in its own source set. Settings are
those of the PTG arms and are not tuned: window 64, stride 16, batch 256, pre-train 30
epochs at 1e-3, fine-tune 30 epochs at 1e-4, 10 groups, 5 folds, seed 42, lmax 2.0, K = 2,
`n_temporal` = `n_spatial` = 40, dropout 0.5. M1 needs a new grouped pre-training (20 source
models); M2 reuses its cached ones and is fine-tuned again.

**Clean-path integrity.** M2's clean (Full) scores were verified **bit-identical** to the
logged PTG-shallow rows for S001 (5 folds), so the evaluation hook does not disturb the
published path. M2's Full score must keep reproducing PTG-shallow's logged per-fold values
for every development subject.

Secondary arms M1-interp (M1 with interpolation instead of node dropping) and M2-zero (M2
with zero-filled channels), if run (the spec leaves their timing open, and M2-zero is needed
in Phase 2), are context only in Phase 1 and never part of the Phase 1 win decision. In
Phase 2 M2's better repair, interpolation or zero-fill, counts (9.24.7).

#### 9.24.5 Decision rule (verbatim from the spec, section 6)

**Clear win** for M1 over M2 on a primary condition requires all four:
1. paired mean difference in within-family accuracy **>= 0.020** (the design's minimum
   detectable difference, 17.2);
2. 95 % bootstrap CI excluding zero;
3. Holm-corrected *p* < 0.05 across the **two** primary conditions;
4. M1 ahead on a **majority of subjects**.

**And one guard on the full montage**: if M1 trails M2 on the Full condition by more than
0.020, which would be a detectable loss, any robustness advantage is reported as a
**trade-off**, not as a clear win. A model that is more robust but worse on a complete cap
does not meet "a clear win". The comparison with PTG-gbp's 0.7301 is reported for context and
is not part of the rule.

Everything else is reported as measured, including a loss. Secondary conditions are
exploratory and labelled so.

#### 9.24.6 Predictions, fixed now

1. **Full: M1 is within 0.020 of M2.**
2. **Random 10 %: no difference greater than 0.020.**
3. **IV-2a: M1 beats M2 by at least 0.020.**
4. **Random 50 %: M1 beats M2 by at least 0.020.**
5. **Random 25 %: weaker confidence; M1 is ahead, but possibly by less than 0.020.**

Predictions 1 and 2 are the expected nulls (interpolation is strong when few channels are
missing); 3 and 4 are where a win is most likely; 5 is the primary the design is least sure
of. Random 25 % is nevertheless Primary 2, so a null there is a null on a primary and is
reported as one. Prediction 4 is on a secondary condition and is exploratory.

#### 9.24.7 Phase 2 settings (verbatim from the spec, section 7), fixed before Phase 1 starts

**Phase 1 — development (20 subjects, S001-S020).** Implement 3.1 and the test-time
conditions, pre-register, run M1 and M2 on the development subjects. Exploratory. Report the
result whatever it is.

**Phase 2 — the robustness-training objection (development set).** Runs only if Phase 1 shows
M1 ahead on a primary condition by at least the 0.020 margin. All settings below are my
calls, fixed in 9.24 before Phase 1 starts, so they cannot be chosen in the light of
Phase 1:

- **Both models train with channel dropout in pre-training *and* fine-tuning.** Dropout during
  fine-tuning alone, on ~36 trials, would be a weak version of the baseline. This costs two new
  sets of 20 source models, about 5 h each.
- **Rate:** each training window independently loses 25 % of its electrodes, matching
  Primary 2. Fixed, not tuned.
- **Each model trains the way it handles missing channels at test time.** M1 drops the nodes,
  with the graph rebuilt for that subset; the Chebyshev basis with the fixed lambda_max = 2
  is a few 64 x 64 products per batch. M2 zero-fills, the standard form of channel dropout.
- **M2 is tested both with interpolation and with zeros, and its better score counts.** This
  biases the comparison towards the baseline, deliberately.
- **"Survives"** means M1-with-dropout beats M2-with-dropout (at its better repair) on the
  same primary condition, by the same four criteria of 9.24.5.

**Phase 3 — confirmation.** Runs only if a win survives Phase 2, using the Phase 2
configuration (both models trained with dropout), with no changes, on the 83 subjects outside
the development set. This is the result that goes in the paper. The 103-subject figure may be
shown alongside it.

If Phase 1 or Phase 2 fails the bar, the experiment stops and is written up as a null, as the
four previous challengers were.

#### 9.24.8 Conditions on reporting

* Within-family accuracy is the outcome; no 4-class figure is quoted.
* No tag is aggregated below its fold target (completeness check), and a partial run is never
  quoted. Development targets: 20 subjects x 5 folds = 100 folds per tag.
* A fold whose clean row was logged but whose condition rows were not is re-run, not skipped.
* Paired on subject, bootstrap CI over 10,000 resamples, win counts reported with every mean.
* **Phase 1 is exploratory and a win on 20 subjects is not a paper result.** The 9.15.4
  prohibition applies in full: M1 and M2 are both pre-trained on ~92 other subjects, and
  **neither may be compared to the 0.650 within-session ceiling**; that is a different
  protocol, not a better pipeline. Comparisons are M1 against M2 only. PTG-gbp's 0.7301 is
  context, not part of the rule.
* A null is an acceptable outcome and would be the fifth measured challenger to the ~0.73
  picture (9.19, 9.21, 9.23, 9.17); it is written up as such.

*Clarifications made on 2026-11-04, before any `ROB` row existed (the log still held
zero): the scope of "never part of the win decision" for M2-zero and M1-interp (9.24.4),
the scoring of Full by the clean row rather than a `full` condition row (9.24.3), the
"if run" status of the secondary arms (9.24.4), and the authority of the commit
timestamp over the clock time above. None changes a prediction, a threshold or a rule.*

---

### 9.25 Montage robustness — results (EXPLORATORY, development set)

**Run 2026-11-04 to 2026-11-07, 20 development subjects (S001-S020), 100 clean rows and
700 condition rows per tag, four tags (3,200 `ROB` rows in `runs/experiments.jsonl`).
Pre-registered in 9.24 (commit `ada4780`, clarifications `b976a43`) before any `ROB` row
existed. Every tag was complete (100 rows) before anything below was read; the
completeness guard in `scripts/robustness_results.py` refuses a partial run. None of the
five predictions holds, the Full-montage guard fires, both primaries are measured losses,
and the Phase 2 gate is not met. The experiment stops here.**

#### 9.25.1 Results

Produced by `scripts/robustness_results.py --m1 ROB1-chebrp --m2 ROB1-shallow` and pasted
verbatim. Within-family accuracy, subject score = mean over the two problems (left/right
fist, both fists/both feet) and the 5 folds, paired on subject; the *p* column is
one-sided in the direction M1 > M2 fixed in advance, so a loss reads 1.0000.

```
M1 = ROB1-chebrp   M2 = ROB1-shallow   20 subjects (within-family accuracy, paired on subject; one-sided p, M1 > M2)

| condition | M1 | M2 | diff | 95 % CI | p | Holm p | wins/n | decision |
|---|---|---|---|---|---|---|---|---|
| Full (guard) | 0.6344 | 0.7461 | -0.1117 | [-0.1356, -0.0867] | 1.0000 | - | 1/20 | guard fires |
| iv2a (primary) | 0.5911 | 0.6544 | -0.0633 | [-0.0939, -0.0333] | 1.0000 | 1.0000 | 3/20 | loss (M1 behind) |
| random25 (primary) | 0.6186 | 0.7126 | -0.0940 | [-0.1150, -0.0726] | 1.0000 | 1.0000 | 1/20 | loss (M1 behind) |
| tentwenty (EXPLORATORY) | 0.5756 | 0.6722 | -0.0967 | [-0.1322, -0.0639] | 1.0000 | - | 0/20 | none (exploratory) |
| random10 (EXPLORATORY) | 0.6232 | 0.7289 | -0.1057 | [-0.1257, -0.0848] | 1.0000 | - | 1/20 | none (exploratory) |
| random50 (EXPLORATORY) | 0.6008 | 0.6752 | -0.0744 | [-0.0968, -0.0540] | 1.0000 | - | 1/20 | none (exploratory) |
| cluster8 (EXPLORATORY) | 0.6198 | 0.7123 | -0.0925 | [-0.1149, -0.0703] | 1.0000 | - | 1/20 | none (exploratory) |
| cluster16 (EXPLORATORY) | 0.6024 | 0.6698 | -0.0674 | [-0.0911, -0.0465] | 1.0000 | - | 1/20 | none (exploratory) |

Full-montage guard: M1 - M2 = -0.1117 against a threshold of -0.020: FIRES -- a would-be win is reported as a trade-off.
```

Integrity checks that ran before the table was read. M2's clean score reproduces the
logged PTG-shallow value for the same 20 subjects exactly (0.7500 on left/right fist, 0.7422
on both fists/both feet, from both tags), so the evaluation hook did not disturb the
published path (9.24.4). Both arms share identical masks per (subject, fold, draw).

#### 9.25.2 Prediction scorecard — zero of five

1. *"Full: M1 is within 0.020 of M2."* — **REFUTED.** M1 trails by **0.1117** [-0.1356,
   -0.0867], five and a half times the margin; M1 ahead on 1/20. The guard fires.
2. *"Random 10 %: no difference greater than 0.020."* — **REFUTED.** M1 trails by 0.1057,
   essentially the Full deficit. Interpolation was expected to make this an even match
   because few channels are missing; it does, in the sense that M2 loses only 0.0172 from
   Full, but the gap that was already there at Full is untouched. (Exploratory condition.)
3. *"IV-2a: M1 beats M2 by at least 0.020."* — **REFUTED, in the wrong direction.** M1
   trails by 0.0633 [-0.0939, -0.0333]; M1 ahead on 3/20. (Primary 1.)
4. *"Random 50 %: M1 beats M2 by at least 0.020."* — **REFUTED, in the wrong direction.** M1
   trails by 0.0744 [-0.0968, -0.0540]; M1 ahead on 1/20. (Exploratory condition.)
5. *"Random 25 %: M1 is ahead, but possibly by less than 0.020."* — **REFUTED.** M1 is not
   ahead at all: it trails by 0.0940 [-0.1150, -0.0726]; M1 ahead on 1/20. (Primary 2.)
   The "weaker confidence" in this prediction was not weak enough.

#### 9.25.3 Decision (9.24.5, applied as written)

- **Full guard fires:** M1 - M2 = -0.1117 against the -0.020 threshold. A would-be win would
  have been reported as a trade-off; there is none to report.
- **Primary 1 (IV-2a): loss.** Diff -0.0633, CI wholly below zero, M1 ahead on 3/20.
- **Primary 2 (random 25 %): loss.** Diff -0.0940, CI wholly below zero, M1 ahead on 1/20.
- **Clear win: no.** Every one of the four criteria fails on both primaries (the margin, the
  CI, Holm-corrected *p* of 1.0000, and the subject majority).
- **Phase 2 gate (9.24.7): not met.** It required M1 ahead on a primary by at least 0.020;
  M1 is behind by 0.063 and 0.094. **Phases 2 and 3 are not run.** The experiment stops and
  is written up as a null, in fact as a measured loss, exactly as 9.24.7 committed to.

**This is the fifth challenger to the ~0.67 -> 0.73 picture to measure nothing in M1's
favour** (9.17's null difference of differences, 9.19, 9.21, 9.23, and now this).

#### 9.25.4 Why M1 is behind — EXPLORATORY, the likely mechanism, not proven

M1's pre-trained weights do transfer. Fine-tuning starts well above chance (train loss
0.675 and accuracy 0.58, against 0.87 and 0.53 for the no-pre-training control), so this
is not a loading bug. But M1 **underfits**: its training accuracy goes from 0.58 to only
0.64 over 30 fine-tuning epochs, where the full-electrode PTG-gbp goes from 0.65 to 0.88.
The most plausible cause is the very property under test. A Chebyshev convolution with
node-shared weights, followed by region-averaged band power into nine regions (9 x 40 =
360 features), leaves nothing that can act as an electrode-specific spatial filter, and
electrode-specific spatial filtering is what ShallowConvNet's per-electrode spatial layer
and CSP both do. The same property is what lets M1 run on any cap. **We state this as the
likely mechanism, not as established**: no ablation separating node sharing from region
pooling was run, and none is planned. For context only, on these 20 subjects M1's
within-family accuracy on left/right fist is 0.6211, below the full-electrode PTC-gbp
(no pre-training, per-electrode readout) at 0.6344, and 0.6478 on both fists/both feet
against PTC-gbp's 0.6967. No region-readout arm without pre-training was run, so M1's own
pre-training gain is unmeasured.

#### 9.25.5 Post hoc observation — NOT part of the decision, EXPLORATORY, POST HOC

Added after the decision above was fixed, on a controller ruling, because it is in the data
and is honest to show. **It was not pre-registered, it plays no role in the verdict, and
it is reported together with the fact that M1 is behind in every condition.** For each
subject, each model's drop from its own Full score; then the difference of differences,
`(M1_cond - M1_full) - (M2_cond - M2_full)`, with a paired bootstrap CI
(`evaluation.metrics.paired_bootstrap_ci`, 10,000 resamples). Positive means M1 loses *less*
than M2. Produced by `scripts/robustness_results.py ... --post-hoc` (a small addition to
the script, with one unit test).

```
| condition | M1 drop | M2 drop | difference of differences | 95 % CI | M1 loses less on |
|---|---|---|---|---|---|
| iv2a | -0.0433 | -0.0917 | +0.0483 | [+0.0139, +0.0828] | 16/20 |
| random25 | -0.0159 | -0.0336 | +0.0177 | [+0.0059, +0.0309] | 14/20 |
| tentwenty | -0.0589 | -0.0739 | +0.0150 | [-0.0222, +0.0522] | 11/20 |
| random10 | -0.0112 | -0.0172 | +0.0060 | [-0.0021, +0.0144] | 11/20 |
| random50 | -0.0337 | -0.0709 | +0.0372 | [+0.0154, +0.0601] | 14/20 |
| cluster8 | -0.0146 | -0.0338 | +0.0192 | [+0.0071, +0.0307] | 15/20 |
| cluster16 | -0.0320 | -0.0763 | +0.0443 | [+0.0212, +0.0667] | 17/20 |
```

**Both facts together.** M1 does lose less than M2 when electrodes vanish: on IV-2a by
+0.0483 [+0.0139, +0.0828] (16/20 subjects), on random 25 % by +0.0177 [+0.0059, +0.0309]
(14/20). Yet M1 is behind M2 in every condition, including both primaries, by 0.063 and
0.094. The two statements are the same data: the IV-2a drop advantage of 0.0483 cancels
less than half of the 0.1117 gap that already existed on the full cap (-0.1117 + 0.0483 =
-0.0634). **This does not make M1 the better model on any cap.** A model with a lower
starting point and an underfit readout has less to lose, and these data cannot separate
"node-shared weights degrade gracefully" from "a model that uses little electrode-specific
information has little electrode-specific information to lose". The CIs are over 20
subjects, are not multiplicity-corrected, and cover a quantity chosen after the result
was known.

#### 9.25.6 What this does and does not establish

**Establishes.** In this pipeline, the node-agnostic region-pooled graph model does not
match the strongest fair baseline on a complete 64-channel cap (-0.1117 on 20 subjects,
guard fired), and that deficit carries to every reduced montage tried. Robustness
bought by dropping electrode-specific spatial filtering costs more on the full cap than it
saves under channel loss. Under 9.24's own rule, the premise "the flexibility is real, so M1
degrades more gracefully" has not produced a usable advantage on any condition.

**Does not establish.**
- That a graph model cannot be made robust. This is one design (Chebyshev K = 2, region
  readout, nine regions) with one pre-training scheme. The attribution to node sharing and
  region pooling is untested.
- Anything about the robustness-training objection. Phase 2 (both models trained with
  25 % channel dropout) was gated on a primary win and did not run; nothing here says what
  channel dropout would do to either model. M1-interp and M2-zero, the secondary arms,
  were not run.
- Anything beyond 20 development subjects. Phase 3 (the 83 remaining subjects) does not run.

**Provenance.** The run was interrupted twice by host-memory failures
(`ArrayMemoryError` while preparing the second and third pre-training groups of M1),
paused on 2026-11-05 at the user's request, and resumed on 2026-11-07 with `--resume`
after a behaviour-neutral change that releases each group's arrays before the next group is
built (commit `8f47b73`). The log holds 160 rows (group 1 of `ROB1-chebrp-binary-0-1`,
subjects S001 and S011) stamped `3562b23` and the remaining 3,040 stamped `8f47b73`; the
change was reviewed as behaviour-neutral, so the two differ only in the recorded commit
hash, but this is a review finding and not something re-measured. M1 pre-training was
redone for any group that had not finished.

**Still exploratory.** Not on the 10.3 list. The 9.15.4 prohibition applies in full: both
arms are pre-trained on ~92 other subjects, **neither may be compared to the 0.650
within-session ceiling**, and the only comparison here is M1 against M2. PTG-gbp's full-cap
0.7301 (103 subjects) is context, not part of the rule; on these 20 subjects it is
0.7300 (mean of 0.7078 and 0.7522), against M1's 0.6344.

---

## 10. Pre-registration of the remaining ablations

**Written 2026-09-01 at commit `3347aa0`, before the learnable-graph result was known.**

*Epistemic state at time of writing.* The learnable-graph ablation is running and has
written 28 of 100 fold rows for one of its three tags to the experiment log. **No
aggregate over any `F-` tag has been computed or inspected.** Everything in 9.1-9.6 is
known; nothing about the learnable graph is. The commit above timestamps this against the
run.

*Why this exists.* §"If the results stay weak" in the plan names the failure mode: with
enough configurations, one will cross a significance threshold and will not replicate.
Fixing the list, the outcome measure and the decision rules in advance is the defence, and
it is only worth anything if written before the results are seen.

### 10.1 Development set versus confirmation set

Every result so far (9.1-9.6) used the **same 20 subjects**, and they have been looked at
repeatedly while choosing what to try next. Those 20 are therefore a **development set**,
and results on them are exploratory no matter how clean the statistics look.

**All confirmatory tests below run on the full 103 subjects.** The 83 unseen subjects have
never influenced any design decision. Where a confirmatory result is reported, it is the
103-subject number; the 20-subject number may be shown alongside as the exploratory
estimate that motivated the test.

### 10.2 Primary outcome

**Mean within-family accuracy across the two binary problems** (left vs right fist; both
fists vs both feet), computed per subject, chance 0.5.

Chosen because it is the only confound-free measure available: both classes in each
problem come from the same run family, so the artefact quantified in 9.2 cannot
contribute. The 4-class number is *not* the primary outcome — it has already misled twice
(9.4, 9.6), improving while the improvement was mostly better run-family identification.

**Secondary outcomes**, reported always but not used for decisions: 4-class accuracy with
the family / within-family decomposition, Cohen's kappa, macro-F1, per-class accuracy and
the confusion matrix.

### 10.3 Confirmatory hypotheses

Fixed list. No test will be added after seeing any result.

| # | Hypothesis | Direction | Comparison |
|---|---|---|---|
| H1 | A learnable adjacency beats the fixed PCC graph | one-sided, learnable higher | paired, same folds |
| H2 | Temporal attention beats no attention | one-sided, attention higher | paired, same folds |
| H3 | Learnable adjacency + temporal attention beats the better of either alone | one-sided | paired |
| H4 | The best configuration differs from CSP+LDA | **two-sided** — no prediction | paired, same folds |
| H5 | LOSO within-family accuracy exceeds chance | one-sided | bootstrap CI vs 0.5 |

H4 is two-sided deliberately. On the evidence so far CSP is ahead, and predicting our own
method wins would be motivated reasoning.

### 10.4 Analysis plan

- Paired across subjects; the pairing unit is the subject, since between-subject variance
  dwarfs every effect measured so far.
- 95 % bootstrap CI on the paired difference, 10,000 resamples, seed 0.
- **Holm-Bonferroni correction across the five confirmatory tests**, family-wise alpha 0.05.
- A hypothesis is **supported** only if its Holm-corrected interval excludes zero in the
  predicted direction. Anything else is reported as **not supported** — never as a trend,
  and never re-tested with different hyperparameters.
- All five outcomes are reported whichever way they fall, including in the writeup.

### 10.4.1 Two clarifications, made before the confirmatory runs started

Recorded 2026-09-01, before any `C-` tag existed, because both change how a result
would be read and neither should be settled after seeing one.

**Tag namespace.** Confirmatory runs use a dedicated `C-` prefix on all 103 subjects.
The existing `F-`, `P2-` and `E-` tags hold 20-subject development data and are never
mixed into a confirmatory comparison. The analysis script enforces this by pointing
only at `C-` tags.

**H4's "best configuration" was ambiguous.** As written it invites choosing the best
arm after seeing the results, which inflates the comparison. It is fixed here as the
**combined arm** (`C-both-binary`) — attention plus learnable graph, the full stated
contribution — regardless of whether some other arm scores higher. If a single
component happens to beat the combination, that is reported as an H3 outcome, not
promoted into H4.

**H3's baseline is resolved from data, but by a rule fixed in advance:** the better of
the two single-component arms by mean within-family accuracy. Which one that is cannot
be known in advance; the rule for choosing it can be, and is.

### 10.4.2 H5's configuration, fixed before the LOSO run

10.5 fixed the arm configuration from the development phase, but LOSO was never given
one. Recorded now, before the run starts.

LOSO trains on ~4,600 trials per fold against ~45 for the subject-dependent protocol —
about 100x the data. **Window 64 at stride 16 is kept identical to the other arms**, so
the representation does not change between hypotheses. **Epochs drop from 50 to 30**,
because with 100x the samples per epoch, 30 epochs is already ~60x the gradient steps
the subject-dependent arms receive; holding epochs at 50 would compare a far more
heavily trained model rather than the same one on more data. Learning rate, batch size,
optimiser and the 5 disjoint validation subjects are unchanged.

This is a deviation from a literal reading of 10.5 and is flagged as such rather than
folded in silently. The alternative — matching epochs and letting training length differ
by 60x — would be the larger deviation.

### 10.5 Stopping rules

- **One configuration per hypothesis.** Each arm gets the hyperparameters already fixed by
  the development phase (8-30 Hz, 64-timepoint window, stride 16, 50 epochs, Adam at 1e-3,
  batch 256). A losing arm is **not** re-run with different settings and the best reported.
- Three seeds per confirmatory arm; the reported value is the mean across seeds.
- If an arm fails, it is recorded as a negative result. The audit (9.1-9.2) already stands
  as a complete finding, so nothing depends on any of these succeeding.

### 10.6 Exploratory analyses — explicitly not confirmatory

These may inform future work and will be **labelled exploratory** wherever they appear. No
correction is applied, no claim is made from them, and none may be promoted to a
confirmatory test after the fact:

- window length (0.4 s / 1 s / 2 s / full trial) and band edges (3b, 3c)
- true lambda_max versus the reference's hardcoded 2 (6.5, 6.6)
- adjacency L1 sparsity levels, and graph sparsification via `threshold_adjacency`
- the over-smoothing measurement across the six conv layers
- learned-versus-PCC graph topography (Phase 6 interpretability)

### 10.7 What gets reported regardless of outcome

The leakage audit and its decomposition (9.1, 9.2); the full-scale leaky replication
(0.8975 across 103 subjects and 3 seeds); the CSP+LDA comparison (9.5); the Phase 3a
negative result (9.6); and all five confirmatory outcomes above. A well-characterised
negative result is the stated acceptable outcome in `CLAUDE.md`, and on current evidence it
is the likely one.

---

## 11. Methods — complete specification

Everything needed to reproduce or defend any number in section 9, in one place. Where a
choice deviates from the reference it is marked **[deviation]** with the reason.

### 11.1 Data pipeline

```
PhysioNet EEGMMIDB, 109 subjects x 14 runs, 3,052 files, SHA256-verified
  |
  exclude S088, S089, S092, S100, S104, S106            -> 103 subjects
  keep imagery runs 4, 6, 8, 10, 12, 14 only            -> 6 runs/subject
  |
  read EDF (MNE), drop the 65th 'EDF Annotations' signal -> 64 channels
  eegbci.standardize()  'Fc5.' -> 'FC5'                  -> 10-05 names
  assert sfreq == 160.0, n_channels == 64, 30 annotations/run
  |
  Common Average Reference                               [matches make_dataset.m]
  band-pass 8-30 Hz (4th-order Butterworth, MNE default) [deviation: reference filters nothing]
  |
  epoch by annotation onset, tmin=0.0, 4.0 s = 640 samples
  label = f(run_number, T-code); T0 discarded
  |
  cache: one .npz per subject, (90, 64, 640) float32, microvolts, NO normalisation
```

**Why no normalisation in the cache.** Mean/std are fit on the training split at training
time. The cache is built before any split exists, so normalising here would leak test
statistics into every experiment that reads it.

**Two caches exist.** `data/cache` (unfiltered, CAR only) and `data/cache_bp830`
(8-30 Hz). Band power must be computed from the unfiltered one — a band-passed signal has
no power outside its passband — and the runner refuses the wrong combination.

**Cache fingerprint.** `manifest.json` records a hash of every preprocessing field.
`assert_cache_matches()` refuses to run against a cache built under a different config,
which is the mechanism enforcing "don't silently change preprocessing between experiments".

### 11.2 Architecture

Reproduces `main-GCN.py` exactly unless marked.

```
input     (N, 64, Fin)        Fin = 1 (reference), 64 (window), or n_bands (band power)
  [optional] TemporalAttention: (N, 64, T) -> (N, 64, n_heads)   [our addition, H2]
conv1     ChebConv  Fin -> 16    K=2, 64 nodes    b2relu[1,64,16]  maxpool/2 -> 32
conv2     ChebConv   16 -> 32    K=2, 32 nodes    b2relu[1,32,32]  maxpool/2 -> 16
conv3     ChebConv   32 -> 64    K=2, 16 nodes    b2relu[1,16,64]  maxpool/2 ->  8
conv4     ChebConv   64 ->128    K=2,  8 nodes    b2relu[1, 8,128] maxpool/2 ->  4
conv5     ChebConv  128 ->256    K=2,  4 nodes    b2relu[1, 4,256] maxpool/2 ->  2
conv6     ChebConv  256 ->512    K=2,  2 nodes    b2relu[1, 2,512] maxpool/2 ->  1
flatten   512
logits    Linear 512 -> n_classes        (plain linear; the reference's relu=False path)
```

**359,428 parameters** at `Fin=1`. `conv6` alone is 264,192 of them (73 %), operating on a
two-node graph.

- `b2relu` = per-vertex-per-filter bias `[1, M, F]` init 0.1, then BatchNorm, then
  **softplus** (not ReLU, despite the reference's name). `main-GCN.py` selects `b2relu`,
  not `b1relu`.
- Weights: truncated normal, sigma 0.1, matching `_weight_variable`.
- **[deviation]** BatchNorm respects `.eval()`. The reference hardcodes `training=True`;
  `always_train_bn=True` restores that for the replication arm.
- Chebyshev basis is dense and precomputed per split — 64 nodes make sparse scatter
  slower, and the basis is constant unless the graph is learnable.
- Graph: `A = |PCC| - I` from the **training split only** **[deviation]**; the reference
  computes it over all data including test.
- Coarsening: Graclus heavy-edge matching, `levels=5`, giving 64/32/16/8/4/2 nodes, with
  `compute_perm` reordering so nodes 2i and 2i+1 are siblings. **[deviation]** an
  off-by-one in the reference's row-length scan is fixed (6.6).
- `rescale_laplacian(lmax=2.0)` matches the reference. True lambda_max is 1.03 at layer 0
  rising to 2.00 at layer 5, so the fixed bound wastes Chebyshev range in early layers
  (6.6) — logged as an untested exploratory ablation.

### 11.3 Training

| | leaky replication | correct protocol | LOSO |
|---|---|---|---|
| optimiser | Adam | Adam | Adam |
| learning rate | 1e-2 | 1e-3 | 1e-3 |
| batch size | 1024 | 256 | 256 |
| epochs | 25 | 50 | 30 |
| L2 (FC params only) | 1e-3 | 1e-3 | 1e-3 |
| model selection | final epoch | best validation | best validation |
| validation | none | 20 % of training trials | 5 disjoint subjects |

The reference's own learning rate of 1e-6 does not train at all within 25 epochs; the
paper states 0.01, and the two disagree (4.5). A sweep over 1e-6, 1e-4, 1e-3, 1e-2 left
held-out accuracy at chance under a correct split regardless, so the choice is not what
produces the audit result.

LOSO's 30 epochs is a deviation recorded in 10.4.2: a LOSO fold sees ~100x the data, so
30 epochs is already ~60x the gradient steps a subject-dependent fold receives.

### 11.4 Splits — four protocols, and what each is for

| protocol | unit | leaks? | used for |
|---|---|---|---|
| `leaky_timepoint_holdout` | timepoint, shuffled 90/10 | **yes, by design** | reproducing the published number |
| `stratified_trial_kfold` | trial, stratified 5-fold | no | subject-dependent results |
| `loso_folds` | subject | no | subject-independent results |
| inner split of training trials | trial | no | validation / checkpoint selection |

The leaky protocol is asserted **to leak** in the test suite: at 640 timepoints per trial
and a 10 % cut, all 90 trials contribute rows to both sides. That makes the replication a
measurement rather than a claim.

### 11.5 Metrics and statistics

- **Primary outcome: within-family accuracy.** Mean over the two binary problems (left vs
  right fist; both fists vs both feet). Both classes in each problem come from the same
  run family, so the run-family artefact cannot contribute. Chance 0.5.
- **Always reported alongside:** 4-class accuracy, run-family accuracy, per-class
  accuracy, confusion matrix, Cohen's kappa, macro-F1.
- **Trial-level aggregation:** a trial's prediction is the argmax of its mean logit across
  its windows/timepoints. Timepoint accuracy is also logged because that is what the
  reference reports.
- **Pairing:** all comparisons are paired on subject. Between-subject variance in EEG
  dwarfs every effect measured here.
- **Intervals:** 95 % bootstrap, 10,000 resamples, seed 0.
- **p-values:** bootstrap under a re-centred null; directional where pre-registered;
  never reported as 0 (floor 1/n_boot).
- **Multiplicity:** Holm-Bonferroni across the five confirmatory hypotheses, family-wise
  alpha 0.05.

---

## 12. Consolidated results

Every headline number, in one place. Within-family accuracy unless stated; chance 0.5 for
within-family and binary problems, 0.25 for 4-class.

### 12.1 The audit — 103 subjects, 3 seeds

| | 4-class | run-family | within-family | 95 % CI |
|---|---|---|---|---|
| published protocol (leaky), timepoint acc | — | — | **0.8975** | [0.8794, 0.9147] |
| correct split, published formulation | 0.3097 | 0.6106 | **0.5090** | [0.4984, 0.5199] |
| correct split, 8-30 Hz + 64-tp window | 0.3395 | 0.6168 | **0.5470** | [0.5291, 0.5667] |

Paired gap, published vs correct protocol: **+0.6096, [+0.5767, +0.6411]**, n=103.
Seed spread on the leaky arm: 0.0008.

### 12.2 Representation ablation — 20 development subjects, exploratory

| arm | band-pass | sample | 4-class | family | within-family |
|---|---|---|---|---|---|
| A | none | 1 tp | 0.3000 | 0.6239 | 0.4859 |
| B | 8-30 Hz | 1 tp | 0.3472 | 0.6778 | 0.5110 |
| C | none | 64 tp | 0.3311 | 0.6544 | 0.5051 |
| D | 8-30 Hz | 64 tp | 0.3300 | 0.6100 | **0.5418** |
| E | band power (fbcsp) | 1 s | 0.4222 | 0.7378 | 0.5716 |

Only D's within-family CI excluded chance. E's large 4-class gain is ~4/5 run-family
identification, not decoding (9.6).

### 12.3 Confirmatory results — 103 subjects, Holm-corrected

| configuration | within-family | 95 % CI | above chance |
|---|---|---|---|
| fixed graph, no attention | 0.5438 | [0.5288, 0.5598] | 64/103 |
| + learnable adjacency | 0.5469 | [0.5317, 0.5632] | 70/103 |
| **+ temporal attention** | **0.6038** | [0.5805, 0.6278] | 83/103 |
| + both | 0.5976 | [0.5745, 0.6220] | 79/103 |
| CSP+LDA | **0.6239** | [0.5977, 0.6521] | 81/103 |
| LOSO (attention config) | **0.6040** | [0.5823, 0.6268] | 88/103 |

| | hypothesis | diff | 95 % CI | p (Holm) | verdict |
|---|---|---|---|---|---|
| H1 | learnable adjacency > fixed | +0.0031 | [−0.0038, +0.0102] | 0.392 | not supported |
| H2 | attention > none | +0.0600 | [+0.0463, +0.0745] | 0.0005 | **SUPPORTED** |
| H3 | both > better single | −0.0061 | [−0.0154, +0.0031] | 0.899 | not supported |
| H4 | contribution vs CSP+LDA | −0.0263 | [−0.0413, −0.0112] | 0.0012 | **SUPPORTED (negative)** |
| H5 | LOSO > chance | +0.1040 | [+0.0826, +0.1269] | 0.0005 | **SUPPORTED** |

### 12.4 Classical baseline, per problem — 103 subjects

*Corrected 2026-09-02. This table previously carried **20-subject development** values
(0.5711 / 0.6722 against 0.5178 / 0.5767) with no cohort stated, and the same mixed pair
had propagated into `PAPER.md` 5.4 and `RESULTS.md` R4 alongside a 103-subject mean —
error 20 in 14.*

| | CSP+LDA | fixed graph | paired difference |
|---|---|---|---|
| left vs right fist | 0.5942 | 0.5148 | −0.0794 [−0.1068, −0.0535] |
| both fists vs both feet | 0.6537 | 0.5728 | −0.0809 [−0.1031, −0.0591] |
| **mean within-family** | **0.6239** | **0.5438** | **−0.0802 [−0.0985, −0.0621]** |

Both methods find fists-vs-feet easier than left-vs-right, which is what the homunculus
predicts and is mild evidence both pipelines track real signal.

### 12.5 A1 / A2 post-hoc controls — 103 subjects (9.11)

**Exploratory. Not on the pre-registered list; A1 tests a claim written after the
confirmatory runs.**

| arm | within-family | vs fixed graph | % of gap to CSP |
|---|---|---|---|
| fixed PCC graph | 0.5438 | — | — |
| learnable adjacency | 0.5469 | +0.0031, p = 0.39 | 3.9 % |
| **A1 free spatial operator** | **0.5566** | +0.0128 [+0.0042, +0.0221], p = 0.0046 | **16.0 %** |
| + temporal attention | 0.6038 | +0.0600 | 74.8 % |
| CSP+LDA | 0.6239 | +0.0802 | 100 % |
| **A2 EEGNet** | **0.6296** | +0.0858 [+0.0697, +0.1024] | **107.0 %** |
| **A2 ShallowConvNet** | **0.6725** | +0.1287 [+0.1118, +0.1454] | **160.6 %** |

vs CSP+LDA, two-sided, Holm-corrected: A1 −0.0673 [−0.0843, −0.0506] p = 0.0006 (24/103);
EEGNet +0.0056 [−0.0107, +0.0220] **p = 0.49, not distinguishable** (56/103);
ShallowConvNet **+0.0485** [+0.0318, +0.0650] p = 0.0006 (67/103).

Per problem: ShallowConvNet 0.6608 (L/R) and 0.6841 (B/F), 94/103 above chance against
CSP's 81. **0.6608 exceeds the 0.650 that [23] reports as the best single configuration
from a 216,714-pipeline search on this dataset.**

### 12.6 Per-subject decomposition — 103 subjects (17.6 E1)

**Exploratory re-analysis of the confirmatory arms; not a pre-registered test.**

| comparison | wins | Pearson *r* | mean | median |
|---|---|---|---|---|
| attention vs CSP | 37/103 (36 %) | +0.825 | −0.0202 | −0.011 |
| both vs CSP (H4) | 34/103 (33 %) | +0.831 | −0.0263 | −0.033 |
| attention vs fixed graph (H2) | **82/103 (80 %)** | +0.823 | +0.0600 | +0.056 |

Between-subject SD 0.1247 against a method gap of ~0.02-0.03: subject identity carries
23-38x the variance of the method choice. 11/103 subjects at or below chance under *both*
methods. **H2 is a consistent per-subject effect; H4 is a thin mean over an interleaved
distribution.** Report both means with their spreads.

### 12.8 Reordered architecture and the pre-training control — 103 subjects (9.13)

**Exploratory (9.12 pre-registered, 9.13 scored). All four predictions held.**

| arm | L vs R | B vs F | within-family | above chance |
|---|---|---|---|---|
| **G1 gbp, graph constraint** | 0.6408 | 0.6850 | **0.6629** | 88/103 |
| **G2 gbp, free operator** | 0.6494 | 0.6893 | **0.6694** | 93/103 |
| **G3 gbp, graph as prior** | 0.6520 | 0.6954 | **0.6737** | 91/103 |
| **PTC, pre-training control** | 0.6533 | 0.6956 | **0.6744** | 89/103 |
| **PTG, grouped pre-training** | **0.7206** | **0.7396** | **0.7301** | **98/103** |

Paired, two-sided, Holm-corrected: **G1 vs CSP +0.0389** (p = 0.0011, 72/103);
**G1 vs attention +0.0591** (p = 0.0011, 82/103); G2 vs G1 +0.0065 (p = 0.52);
G3 vs G2 +0.0043 (p = 0.54, **51/103 — a coin flip**); G3 vs ShallowConvNet +0.0012
(p = 0.99, 46/103).

**PTC is the pre-training control, not a result in itself.** It runs the identical
pipeline with `--no-pretrain`, and landing at 0.6744 — indistinguishable from G3's
0.6737 — is what makes the eventual PTG − PTC contrast interpretable: it shows the
source-fit representation is neutral, so any difference is pre-training.

### 12.7 Resting SMR predicts a third of the between-subject variance (17.6 E2)

**Exploratory. Devised after seeing 12.6; not pre-registered.**

12.6 shows subject identity dominating method choice, with *r* ≈ 0.83 between two unrelated
methods — the signature of a subject-level ceiling. The standard measure of that ceiling is
Blankertz et al.'s SMR predictor [26]: the prominence of the resting mu rhythm above the
fitted 1/f background. `scripts/smr_predictor.py` computes it from **run R01, the 61-second
eyes-open baseline** — downloaded from the start, never used by any experiment here, and
containing no motor imagery, so it cannot leak into a decoding result.

| arm | *r* with the resting predictor | variance explained |
|---|---|---|
| GCN + temporal attention | **+0.547** | **30 %** |
| CSP+LDA | **+0.548** | **30 %** |
| ShallowConvNet | +0.480 | 23 % |

Predictor range 16.32-30.10 dB, median 21.14, over all 103 subjects.

**The two headline numbers are +0.547 and +0.548.** A graph network and a linear method from
2000, sharing nothing, are predicted equally well by a property of the *subject* measured
before either ran. That is about as direct a demonstration of a subject-level ceiling as
this dataset can supply. The 11 subjects at or below chance under both methods average
1.70 dB less mu prominence than the rest.

For external calibration, [26] reports r = 0.53 between the predictor and SMR-BCI feedback
accuracy in 80 participants, i.e. about 28 % of variance; our 30 % on a different task and
cohort is consistent with it. *(Corrected 2026-09-20: earlier text said ~37 %, which does not
match the paper; see error 32.)*

**Consequence.** Roughly a third of what looks like model failure on this dataset is a
property of the participants, fixed before any modelling choice. 13 now says so with a
number rather than a citation. It does not excuse the graph formulation — the 70 % that
remains is still where ShallowConvNet beats it by 0.1287 — but it bounds how much any
architecture could ever recover.

---

## 13. Limitations and threats to validity

Stated plainly because a viva will ask, and because several of these bound the
conclusions more tightly than the confidence intervals do.

**One dataset, one architecture.** Every conclusion is about GCNs-Net on PhysioNet
EEGMMIDB. "The graph inductive bias costs more than it contributes" (9.7.1) is supported
*here*; it is not a claim about graph networks for EEG in general. BCI IV-2a was scoped
and never run (decision 6).

**The 4-class problem is structurally confounded and cannot be de-confounded.** Classes
0/1 occur only in runs 4/8/12 and 2/3 only in 6/10/14. No analysis can separate
run-family signal from neural signal in a 4-class number; only the within-family
decomposition avoids it, at the cost of answering a binary question instead of the
4-class one the paper poses.

**Within-family accuracy is not what the paper reports.** Our primary outcome is a
binary problem; the paper's is 4-class. The comparison to 0.9306 is therefore a
comparison of protocols, not of directly commensurable numbers. This is stated wherever
the two appear together.

**The development set was inspected repeatedly.** Sections 9.1-9.7 used the same 20
subjects while choosing what to try next. 10.1 demoted them and moved all confirmatory
tests to 103 subjects, and 9.8 showed every 20-subject estimate fell inside its
full-scale CI — but exploratory results remain exploratory.

**Single seed for the confirmatory arms.** Phase 1 used three seeds and showed a spread
of 0.0008, so seed variance is small for this architecture. The confirmatory arms used
one seed each, which 10.5 permits but which leaves seed variance unmeasured for those
specific numbers.

**Post-hoc analyses are not confirmatory.** 9.10's mean-pool control was devised after
seeing H2 and is labelled exploratory throughout. It is strong evidence about mechanism;
it is not a pre-registered test.

**CSP was given the band-passed cache.** CSP requires band-limited input, so it ran on
8-30 Hz data — the same cache as the network arms it is compared against. It was not
tuned: 4 components, Ledoit-Wolf, shrinkage LDA, no search. A tuned CSP would likely be
stronger, which makes H4's negative result conservative rather than optimistic.

**LOSO used one configuration.** The attention config only. Whether a learnable graph or
band power behaves differently under LOSO is unmeasured.

**A1 and A2 are post-hoc (added 2026-09-02).** Neither is on the pre-registered list (10.3),
and A1 tests a claim (9.7.1) formed after the confirmatory runs. They are strong evidence
about mechanism; they are not confirmatory tests, and 12.3's verdicts are unaffected by them.

**The A2 baselines run outside their design regime.** EEGNet assumes ~500 samples at 128 Hz
and ShallowConvNet ~500+ at 250 Hz; our confirmatory window is 64 samples at 160 Hz = 0.40 s,
so their published kernel and pool sizes do not fit and every temporal hyperparameter is
derived from T. This makes their advantage **conservative** — they beat CSP under conditions
unfavourable to them — but a matched comparison at a longer window has not been run, and
window length has never been swept for any arm (B5, 17.3).

**9.11 does not identify the remaining 84 % of the gap.** It bounds the graph-locality
contribution at 16 % and excludes sample size. The temporal-ordering hypothesis offered in
9.11.2 is untested.

**About 30 % of the between-subject variance is not ours to fix (12.7).** The resting SMR
predictor, computed from a baseline run recorded before any task, correlates at +0.547 with
our best model and +0.548 with CSP. Part of what looks like model failure is a subject-level
ceiling. This bounds the headroom any architecture has on this dataset, and it is a reason to
prefer per-subject reporting (12.6) over means.

**The classical baseline we report is weaker than the field's.** [23] puts a tuned
tangent-space or CSP pipeline near 0.605 on left-vs-right where our untuned CSP reaches
0.5942, so every "loses to CSP" statement here understates the true classical bar.

**Trial counts are small.** ~22-23 trials per class per subject, ~45 per binary problem.
Subject-dependent folds train on ~36 trials. 9.9 argues this is why LOSO outperforms
subject-dependent, which also means subject-dependent numbers here are noisier than the
CIs alone suggest.

**We did not read [4] or [5].** Both are paywalled. The novelty claim rests on their
abstracts and on the observation that neither GCNs-Net headline is subject-independent.

---

## 14. Errors made and corrected

Every substantive error in this project and how it was caught. Recorded because the
pattern is instructive: **most were caught by an independent check disagreeing with a
process that reported success.**

| # | Error | How it was caught | Fix |
|---|---|---|---|
| 1 | Exclusion list named S090; S090 is clean and S089 is defective | audited all 1,526 EDF headers instead of trusting the list | 3.5, `CLAUDE.md` corrected |
| 2 | Claimed `make_dataset.m` mean-centres over the dataset (leakage) | read the local source; `mean(X,1)` on a channels x time matrix is CAR | 4.4, leakage count 4 -> 3, config set to CAR |
| 3 | Claimed the PCC adjacency construction was absent from the repo | it is in `make_dataset.m`; only `main-GCN.py` lacks it | 4.3 |
| 4 | Used `b1relu` in the analysis; `main-GCN.py` configures `b2relu` | read `main-GCN.py` directly | 4.3, parameter count now matches |
| 5 | Said the paper's four classes are L/R/B/F | the shipped code discards both-fists and substitutes rest | 4.5, decision 1 |
| 6 | Predicted band-passing would lower run-family accuracy | it rose (0.6239 -> 0.6778) | 9.4, prediction recorded as failed |
| 7 | Over-stated the family diagnostic as pure confound | bilateral vs unilateral imagery genuinely differ topographically | 9.4 |
| 8 | Hypothesised H2's gain was averaging plus regularisation | built a fixed mean-pool control; it gained nothing | 9.10 |
| 9 | Read the attention figure as showing uniform weights | per-sample deviation is 104 %; the population mean hides it | 9.10, figure corrected |
| 10 | Quoted 0.5774 from a partial run; completed value 0.5418 | fold count checked against target | rule added to `CLAUDE.md` |
| 11 | Left an orphaned driver running for hours after "stopping" it | user observed the machine was not idle | kill the process **tree** |
| 12 | Polluted a tag with 431 rows from a verification run without `--subjects` | row count 531 vs expected 100 | rows removed; runners now take `--subjects` explicitly |
| 13 | Concurrent writers tore two log rows | JSON parse failure, then a fold-count check | locked single-syscall append |
| 14 | Documentation drifted for two commits | user asked whether it was being maintained | caught up; drift recorded rather than hidden |
| 15 | Wrote a **fabricated DOI and wrong author list** into 3.5 for the curation paper | ran the Crossref check instead of re-reading the draft | corrected to Shuqfa et al., *Data in Brief* 54:110181 |
| 16 | Claimed B4/C2 could run on "checkpoints that already exist" | checked the filesystem before recommending the work; the cleanup had removed them | 17.3 corrected, cost changed from near-zero to small |
| 17 | `posthoc_controls.py`'s verdict rule tested **significance instead of effect size**, and would have printed "9.7.1 supported" for an arm recovering 16 % of the gap | ran the tool on real results rather than re-reading it | rule now tests fraction of gap recovered; 9.11.3 |
| 18 | Predicted sample size (17.1) would explain the CSP gap, and said so twice | A2: graph-free CNNs on the *same* ~45 trials reach and pass CSP | 9.11.2, prediction recorded as failed |
| 19 | 9.7.1 inferred a *mechanism* from two null results; it was ~84 % wrong about magnitude | A1 measured it directly at 16 % of the gap | 9.7.1 superseded in part, original kept verbatim |
| 20 | `PAPER.md` 5.4 mixed a **20-subject** per-problem breakdown with the **103-subject** mean in one table | its own two rows did not average to its own mean (0.6217 vs 0.6239) | recomputed on 103 subjects: −0.0802 [−0.0985, −0.0621] |
| 21 | The 5.9 edit **deleted the `## 6. Discussion` heading**; the prose survived, so the paper read correctly but ran 5 -> 7 for three commits, including in a released PDF | listing headings during an audit, not reading the prose | restored; now covered by `test_section_numbering_is_contiguous` |
| 24 | **Under-costed the pre-training arm by ~8x** — projected 6-8 h, measured ~51 h — by carrying the LOSO timing (`GCNsNet`, which pools time before the convs) over to `graphbp` (which deliberately carries the time axis *through* the spatial stage, so its cost scales with pooled sample count far more steeply) | the arm reached 50 folds in 149 min when it should have reached ~450 | grouped pre-training reduces 206 pre-trainings to ~20; a timing probe on the actual arm would have cost three minutes |
| 23 | `--stop-at` rolled a past time forward to **tomorrow**. Each process honoured 03:00 correctly, but the shell loop then started the next binary problem at 03:00:0x, which read its own deadline as past and scheduled itself for the *following day* — a 24-hour run from a flag meant to bound one to 50 minutes | the user asked for status at 03:02 and the processes were still alive with `-2-3` tags gaining rows | a past time now stops immediately, which is the only reading that composes under repetition |
| 25 | **Fine-tuning results depended on how much work the process had already done.** Dropout draws from the global torch generator, and `val_pick` came from one generator advanced once per group, so skipping a completed group on resume shifted every remaining fold. The three power-cut resumes were therefore not reproducible, and the logged PTG/PTC numbers cannot be re-derived exactly by a clean run | adding a source-model cache and finding the cached run scored 0.5188 where the cold run scored 0.5110 — the cache was supposed to be result-neutral | each stage now seeds itself from its own identity (`seed_stage`), and `val_pick` is seeded per group; a resumed run now reproduces an uninterrupted one fold for fold (verified on 12 folds) |
| 26 | **Launched the CSP-prior arm on the script's defaults** -- 100 epochs, batch 1024, lr 1e-6 -- instead of the G-arm protocol it had to match. At that learning rate the model sat at chance, and batch 1024 is the VRAM-overflow condition documented in `CLAUDE.md` the day before | the arm was running at ~4.5 min/subject against a 100 s/subject expectation, so the logged `train_config` was compared against G2's | 24 rows purged; the flags are now in `scripts/run_tonight.sh` with a comment saying they must not be dropped |
| 27 | **Purged the bad rows by tag, and six came back.** A bash wrapper from an earlier launch was still looping and re-logging under the stale configuration; killing Python alone had simply let it spawn the next invocation | the retained rows' `train_config` was checked before they were trusted, and showed two distinct configurations | purged by configuration rather than tag; all arms now run from one sequential driver that can never hold two processes, and whose header says to kill the wrapper first |
| 28 | **Filled the GPU by accumulating orphan processes.** Four Python processes at ~1.7 GB of CUDA reserve each filled an 8 GB card; the user reported 7.6/8.0 GB before it was noticed. The suspected cause -- batch size -- was wrong | measured allocated/reserved across five real folds: 0.07 / 1.74 GB, flat, no leak. The process list then showed the orphans | one driver, one process; a blanket `taskkill /IM python.exe` also killed an unrelated running job and is not the way to stop these |
| 29 | **My own literature-driven recommendation did not transfer.** CSP-Net's +6.0 points was presented as the highest-value available change; measured here it is -0.0008 | 9.18 pre-registered it with a decision rule for exactly this outcome | reported as decision rule B; the null is more informative than the predicted positive would have been (9.19.3) |
| 30 | **Committed over 19 failing tests.** The suite was run as `pytest -q | tail -2`, so the shell reported `tail`'s exit code rather than pytest's, and the `&&` chain committed anyway. Adding `"dsbn"` to the source-model cache key had broken every test whose `_Args` stub predates the flag | the next full run, read without a pipe | fixed in `4dbdee1`; pytest is now redirected to a file and its exit code checked explicitly, and `dsbn` joined the parametrised list of key-changing hyperparameters |
| 31 | **Three bugs in one wiring pass**, each caught by the next command rather than by review: `Samples` and `Fold` are frozen dataclasses so the domain array needed `dataclasses.replace`; a patch inserted `"dsbn"` into `source_cache_key`, where `n_domains` is out of scope; and that same patch's *second* assert aborted the write, leaving the first fix unapplied and the `NameError` latent | a smoke run on 6 subjects before the 4-hour arm | all three fixed before launch. The lesson is the third one: an aborted multi-edit script leaves no trace, so a failed assert must be re-read as "nothing was written", not "one edit failed" |
| 32 | **Misquoted a cited effect size across four documents.** Blankertz et al. 2010 was cited as explaining "~37 %" of between-subject BCI performance variance. The abstract reports r = 0.53 in N = 80, i.e. r² ≈ 0.28. The wrong figure reached the paper, RESULTS, this document and the submitted literature survey | checking the claim against the abstract via NCBI E-utilities before reusing it in a fourth document, the review paper | corrected everywhere to r = 0.53 / ~28 %; the conclusion drawn from it (our 30 % is consistent) is unchanged and in fact closer. The submitted survey PDF carries the old figure |
| 22 | The log's exclusive lock seeked to **EOF** before locking one byte, but `msvcrt.locking` locks relative to the *current position* — so once two processes saw different EOFs they locked different bytes and the mutual exclusion lapsed | its own regression test (for error 13) went intermittently flaky under full-suite load | lock byte fixed at offset 0, a rendezvous every process agrees on; 10/10 clean afterwards |

**The recurring lesson.** A process exiting with code 0 and printing a plausible mean is
not evidence it did what it claimed. Job Q reported COMPLETE having silently lost a fold;
the orphaned driver kept working after being "stopped"; a torn write loses a row without
raising. Every one of these was found by comparing an *independent count* against an
expected target. That check is now part of how any run is verified.

---

## 15. Reproduction recipe

Exact commands for every reported number. Assumes the caches are built (README).

```bash
# 12.1 the audit, 103 subjects x 3 seeds
for s in 1 2 3; do
  python scripts/run_leaky_replication.py --epochs 25 --lr 1e-2 --seed $s \
      --resume --tag "P1-leaky-103-s$s"
  python scripts/run_subject_dependent.py --folds 5 --epochs 25 --lr 1e-2 \
      --batch-size 1024 --window 1 --seed $s --resume --tag "P1-A-103-s$s"
  python scripts/run_subject_dependent.py --config configs/bandpass_8_30.yaml \
      --folds 5 --epochs 50 --lr 1e-3 --batch-size 256 --window 64 --stride 16 \
      --seed $s --resume --tag "P1-D-103-s$s"
done

# 12.3 confirmatory arms (add --classes 0 1 and --classes 2 3 for each)
python scripts/run_csp_baseline.py --tag C-csp
python scripts/run_subject_dependent.py --config configs/bandpass_8_30.yaml \
    --folds 5 --epochs 50 --lr 1e-3 --batch-size 256 --window 64 --stride 16 \
    --classes 0 1 --resume --tag C-fixed-binary-0-1            # + --learnable-graph
                                                               # + --temporal-attention
                                                               # + both, for H3/H4
python scripts/run_loso.py --config configs/bandpass_8_30.yaml --classes 0 1 \
    --window 64 --stride 16 --epochs 30 --lr 1e-3 --batch-size 256 --resume \
    --tag C-loso-binary-0-1

# analysis
python scripts/confirmatory_results.py      # 12.3 verdict table, Holm-corrected
python scripts/run_family_analysis.py --tag <tag>   # 4-class -> family / within-family
python scripts/ablation_summary.py          # 12.2
python scripts/analyse_attention.py         # 9.10
python scripts/make_figures.py              # montage, CAR, confusion

# 12.5 post-hoc controls (A1 / A2, 9.11) -- identical folds to the confirmatory arms
for c in "0 1" "2 3"; do
  python scripts/run_subject_dependent.py --config configs/bandpass_8_30.yaml \
      --folds 5 --epochs 50 --lr 1e-3 --batch-size 256 --window 64 --stride 16 \
      --classes $c --free-spatial --resume --tag "A1-free-binary-${c// /-}"
  for m in eegnet shallow; do
    python scripts/run_subject_dependent.py --config configs/bandpass_8_30.yaml \
        --folds 5 --epochs 50 --lr 1e-3 --batch-size 256 --window 64 --stride 16 \
        --classes $c --model $m --resume --tag "A2-$m-binary-${c// /-}"
  done
done
python scripts/posthoc_controls.py --figures figures   # 12.5/12.8 verdict + figure

# 12.8 the reordered architecture (9.12 pre-registered, 9.13 scored)
for c in "0 1" "2 3"; do
  for v in "graph:--spatial graph" "free:--spatial free"            "prior:--spatial free --graph-smoothness 1e-4"; do
    n=${v%%:*}; flags=${v#*:}
    python scripts/run_subject_dependent.py --config configs/bandpass_8_30.yaml         --folds 5 --epochs 50 --lr 1e-3 --batch-size 256 --window 64 --stride 16         --classes $c --model graphbp $flags --resume         --tag "G-gbp-$n-binary-${c// /-}"
  done
done

# 17.11.2 pre-training, grouped (9.14 + 9.14.1). --groups 10 is what makes this
# tractable: 20 pre-trainings rather than 206 (error 24).
for c in "0 1" "2 3"; do
  python scripts/run_pretrain_finetune.py --config configs/bandpass_8_30.yaml       --model graphbp --spatial free --graph-smoothness 1e-4 --classes $c       --groups 10 --folds 5 --pretrain-epochs 30 --finetune-epochs 30       --window 64 --stride 16 --resume --tag "PTG-gbp-binary-${c// /-}"
  # the control: identical pipeline, pre-training skipped. Run both or neither.
  python scripts/run_pretrain_finetune.py --config configs/bandpass_8_30.yaml       --model graphbp --spatial free --graph-smoothness 1e-4 --classes $c       --groups 10 --folds 5 --finetune-epochs 30 --no-pretrain       --window 64 --stride 16 --resume --tag "PTC-gbp-binary-${c// /-}"
done
python scripts/subject_heterogeneity.py --figures figures   # 12.6, no compute

# the typeset paper (PAPER.md -> GCNs-Net-Reassessed.pdf)
python scripts/build_paper_pdf.py           # -> build/paper_print.html
msedge --headless=new --disable-gpu --no-pdf-header-footer \
    --virtual-time-budget=25000 \
    --print-to-pdf=GCNs-Net-Reassessed.pdf build/paper_print.html
```

Every run appends one row per fold to `runs/experiments.jsonl` with the git commit, seed
and full config. All runners take `--resume`; re-running a command after an interruption
picks up exactly where it stopped and cannot duplicate a row.

---

## 16. Glossary

| term | meaning |
|---|---|
| **run family** | which pair of runs a trial came from: 4/8/12 (left/right fist) or 6/10/14 (both fists/both feet) |
| **run-family accuracy** | whether the model identified the family, ignoring which class within it. Chance 0.5. Contaminated by run-level drift *and* by a genuine bilateral-vs-unilateral difference |
| **within-family accuracy** | given the family, whether the exact class was right. Chance 0.5. **The confound-free measure and the project's primary outcome** |
| **leaky protocol** | the published one: timepoint samples, `randperm` split, adjacency and statistics over all data, BatchNorm in train mode at test |
| **correct protocol** | trial-level split, train-split-only adjacency and statistics, BatchNorm respecting `.eval()` |
| **timepoint accuracy** | accuracy over individual 6.25 ms samples — what the reference reports |
| **trial accuracy** | one decision per imagined movement, from the mean logit across a trial's samples |
| **development set** | the 20 subjects inspected repeatedly while choosing what to try. Results on them are exploratory (10.1) |
| **confirmatory** | pre-registered, run on all 103 subjects, Holm-corrected (10) |
| **ERD/ERS** | event-related de/synchronisation — the mu/beta power change that motor imagery produces |

---

## 17. Future directions

*Written 2026-09-02 and substantially revised the same day after a second, deeper
literature pass plus four analyses of our own log. The revision changed the conclusions,
not just the detail — see 17.1 and 17.6. Provenance and reading depth in 17.9. Published
as an artifact alongside this section.*

Ranked by **what each direction would establish**, not by how much accuracy it might add.
That ordering is deliberate: the project's value rests on a negative result, which
controls strengthen and tuning destroys.

**Status, 2026-09-02 evening.** Two items are no longer proposals:

| item | state |
|---|---|
| **A1** unconstrained spatial operator | **RUN. Results in 9.11.** 0.5566 — significant (+0.0128, p = 0.0046) but recovers only **16 %** of the gap. 9.7.1 superseded in part |
| **E1** per-subject distribution | **DONE.** `scripts/subject_heterogeneity.py` + `figures/subject_heterogeneity.png`. Results in 17.6 |
| **A2** graph-free CNN baselines | **RUN. Results in 9.11.** EEGNet 0.6296 (indistinguishable from CSP), ShallowConvNet **0.6725** (beats CSP by +0.0485) |

The command for A1, matching the confirmatory arms exactly so it is comparable to
`C-fixed-binary` fold for fold:

```bash
for c in "0 1" "2 3"; do
  python scripts/run_subject_dependent.py --config configs/bandpass_8_30.yaml       --folds 5 --epochs 50 --lr 1e-3 --batch-size 256 --window 64 --stride 16       --classes $c --free-spatial --resume --tag "A1-free-binary-${c// /-}"
done
python scripts/confirmatory_results.py
```

And for A2, on the same folds:

```bash
for m in eegnet shallow; do
  for c in "0 1" "2 3"; do
    python scripts/run_subject_dependent.py --config configs/bandpass_8_30.yaml         --folds 5 --epochs 50 --lr 1e-3 --batch-size 256 --window 64 --stride 16         --classes $c --model $m --resume --tag "A2-$m-binary-${c// /-}"
  done
done
```

**Design notes on A2, because two of them are caveats rather than details.**

* **It shares the runner, not just the protocol.** `--model` selects the architecture
  inside `run_subject_dependent.py`, so the folds, seeds, cache, standardiser, windowing,
  trial aggregation, resume logic and log schema are identical *by construction* rather
  than by careful duplication. `prepare_fold(build_graph=False)` skips only the graph —
  adjacency, coarsening, node permutation — leaving every leakage-sensitive step in the
  same audited place (11's steps 1, 4, 5 are untouched). Channels reach the convolution in
  montage order, with no fake-node padding.
* **Caveat: these architectures are being run far outside their design regime.** EEGNet
  assumes ~500 samples at 128 Hz and ShallowConvNet ~500+ at 250 Hz; our confirmatory
  window is 64 samples at 160 Hz, i.e. **0.40 s**. The papers' absolute kernel and pool
  sizes do not fit at all. Every temporal hyperparameter here is therefore *derived from
  T*, recovering the published values once T is long enough (EEGNet's `kern_length` is 32
  at T=64 and the paper's 80 at T=640). **This is a real limitation of A2 and must be
  reported as one** — and it is independent motivation for B5, since a fair A2 arguably
  needs a longer window than the one the GCN arms were run at.
* **Guards rather than silent ignoring.** `--model eegnet` with `--temporal-attention`,
  `--learnable-graph`, `--free-spatial`, `--adjacency-l1` or `--leaky` exits with an error
  instead of logging a row that claims a configuration the run did not have. That failure
  mode is error #12.
* **Regularisation is matched.** EEGNet's own max-norm constraints are implemented
  (depthwise 1.0, classifier 0.25) and applied inside `forward` so the shared `Trainer` is
  untouched — changing it would alter the arms these are compared against. Both baselines
  also expose `l2_penalty` with `GCNsNet`'s convention, so `--regularization` means the
  same thing for every arm.

**Neither A1 nor A2 is on the pre-registered list (10.3), and neither may be reported as
confirmatory.**
It tests a claim in 9.7.1 that was itself written after the confirmatory runs, so it is a
post-hoc mechanistic test. Label it exploratory, and pre-register it separately if it is to
carry weight.

### 17.0 What "good" looks like on this dataset

**This section did not exist in the first draft, and it should have.** Every judgement
below depends on knowing the achievable ceiling, and we had never established it.

A 2026 benchmark [23] evaluated **216,714 pipeline configurations** on PhysionetMI
(132,762 subject-level observations) under MOABB's `LeftRightImagery` protocol — binary
left-vs-right hand, chance 0.5, the same problem as our `left_vs_right_fist`:

| pipeline family (PhysionetMI, 8-30 Hz) | accuracy |
|---|---|
| best single configuration (cov-tgsp + RobustScaler + linear SVM) | **0.650 ± 0.183** |
| covariance tangent-space projection, family mean | 0.605 ± 0.174 |
| CSP, family mean | 0.604 ± 0.178 |
| CSP, 8-15 Hz | 0.596 ± 0.172 |
| **our CSP+LDA, left vs right fist** (untuned, 4 components) | **0.5711** |
| **our best model, mean within-family** | **0.6038** |

The authors describe PhysionetMI as **the most challenging and most heterogeneous dataset
in the benchmark**, with "lower absolute performance and reduced separation between the
top-ranking families" than the others.

Three consequences, and they matter more than anything in the first draft:

1. **Our numbers are in the normal range.** Our untuned CSP sits a little under the
   published tuned family means, which is what "untuned" should look like. The pipeline is
   not broken and the task really is this hard.
2. **The ceiling is roughly 0.65 after searching 216,714 configurations.** The headroom
   between our best model and the best known pipeline on this dataset is around 0.05, not
   the 0.30 that the published 0.9306 implies.
3. **The 0.0263 gap we are discussing is small relative to the spread between top
   families** (0.596 to 0.650 spans 0.054). Chasing it is close to noise-mining.

### 17.1 Two competing explanations for the CSP gap

9.7.1 concluded that the bottleneck is **the graph-convolutional inductive bias** — order-2
Chebyshev filters mix each node only with its one-hop neighbourhood using node-shared
weights, whereas CSP fits an unconstrained discriminative projection over all 64 channels.
That remains an inference from two null results (9.6, 9.7), not a measurement.

**The second pass found a competing explanation that 9.7.1 does not consider, and it is
well supported.** The MOABB benchmark [15] reports that deep learning "typically requires
more than 150 trials per class" for satisfactory results, that ShallowConvNet, EEGNet-8.2
and DeepConvNet reach satisfactory performance on an easy binary problem "with as few as
50 trials per class", and that Riemannian and raw-signal pipelines do well "even with
fewer than 50 trials".

**We have ~45 trials per binary problem, and subject-dependent folds train on ~36.** That
is below even the classical threshold and far below the deep-learning one. On this
account the GCN loses to CSP not because the graph prior is wrong but because *any* deep
model is below its data threshold while CSP is comfortably within its own — and 9.9
already found the corroborating fact independently, since LOSO (~100x more training data)
marginally beats subject-dependent.

**Two honest caveats, in both directions:**

- MOABB's deep-learning arm used published hyperparameters with no per-dataset tuning,
  while its Riemannian and raw arms were tuned by nested cross-validation. Their
  DL-underperforms result is therefore confounded by a tuning asymmetry. **Ours is
  confounded in the opposite direction** — our network was tuned and our CSP was not — so
  the two biases do not compound, they oppose.
- 50 trials per class is the figure for an *easy* binary problem. Left-vs-right fist is
  not easy. Our ~45 is close enough to the boundary that sample size is a plausible
  competing explanation, not a refutation.

**These two explanations are distinguishable, and A2 distinguishes them.** If EEGNet — an
unconstrained spatial filter followed by temporal convolution, no graph anywhere — also
loses to CSP by a similar margin on identical folds, the limitation is sample size and
9.7.1's central claim is unidentified. If EEGNet closes the gap, 9.7.1 survives. **This
was the third-priority control in the first draft. It is now joint-first with A1.**

### 17.2 Tier A — controls that must exist first

Not improvements. These decide whether any improvement is attributable, and two of them
could overturn 9.7.1. **Nothing in tiers B-E is worth starting before A1 and A2 have run.**

| # | Direction | Effort | What it settles |
|---|---|---|---|
| **A1** | **Replace the Chebyshev stack with an unconstrained `64 -> F` spatial mixing layer**, holding cache, folds, seeds, attention and training loop constant | Small | Whether the locality-and-weight-sharing constraint *specifically* costs the gap. Converts 9.7.1 from inference to measurement — or falsifies it |
| **A2** | **EEGNet and ShallowConvNet on identical splits** | Small-medium | **Discriminates the two explanations in 17.1.** Now joint-first, not third |
| **A3** | **Tangent-space LR and FBCSP**, tuned | Small | 13 records that CSP was untuned. [23] shows tuned cov-tgsp reaches 0.605-0.650 on this dataset, so our 0.5711 is beatable and the true gap is probably larger than 0.0263 — which makes the negative result *stronger*, not weaker |

A3 will not change the direction of H4, and that is the point.

**On statistical power, since a viva will ask.** Recomputed from the log: the paired H2
difference over 103 subjects has mean +0.0600 and SD 0.0741 (Cohen's dz = 0.810), giving
SE 0.0073 and a **minimum detectable difference of about 0.020 at 80 % power**. The design
was adequately powered for the 0.0263 it detected. It is *not* powered for anything
much smaller, which is a further argument against chasing residual differences.

### 17.3 Tier B — cheap levers with strong priors

| # | Direction | Effort | Note |
|---|---|---|---|
| **B1** | **Euclidean Alignment on the LOSO arm** [17,18] — whiten each subject's trials by `R̄^(-1/2)`; unsupervised, no target labels | Very small | The LOSO runs apply *no* cross-subject alignment at all. **Apply it to CSP too** — EA helps covariance methods at least as much, and using it on one arm only would manufacture a win |
| **B2** | **Per-run alignment as a de-confounding probe** | Small | See below — novel; searches found no published treatment of the run-family confound |
| **B3** | **Surface Laplacian / CSD preprocessing** | Small | `A = \|P\| - I` is near-dense because scalp channels are correlated by volume conduction, not connectivity. CSD is a spatial high-pass that suppresses exactly that. The one change that would make the graph *mean something* |
| **B4** | **Measure Dirichlet energy per layer; sweep graph sparsity and depth** | Small | Six conv layers over a near-complete 64-node graph is a textbook over-smoothing configuration. [22] derives Graph Dirichlet Energy bounds showing GCNs act as low-pass filters and that volume conduction *accelerates* the decay. **Correction: this is not free.** The cleanup removed `checkpoints/` and made checkpointing opt-in (`--checkpoints`), so no weights are on disk; one arm must be re-run with that flag before the forward hooks can be attached |
| **B5** | **Sweep the raw-voltage window length** | Very small | **Never done.** Every confirmatory raw run uses `window=64` = **0.40 s**; the only `window=160` rows in the log are band-power runs. ERD/ERS develops over 1-2 s, so 0.4 s may be structurally too short — and attention currently operates over 37 such windows, so window length changes what attention is even *for* |

**B2 in detail.** 13 states the 4-class problem "cannot be de-confounded." That is true of
the *labels* — classes 0/1 occur only in runs 4/8/12 — but may not be true of the *signal*.
The confound is hypothesised to be run-level non-neural drift (impedance, electrode
settling, alertness). Whitening each run by its own covariance would remove exactly that
kind of stationary run-level structure. Clean prediction:

- **If run-family accuracy (0.6100) falls toward 0.5 while within-family is unchanged, the
  family signal was an artefact.**
- **If both fall, the family separation was partly genuine** — which error #7 already warns
  is likely, since bilateral and unilateral imagery genuinely differ.

Either outcome is reportable and neither can be spun. **Pre-register it before running.**

**B4 has independent support from the direct prior art.** [5] SAMGCN — one of the two
papers 8.1 identifies as closest to our stated contribution — is *explicitly motivated* by
GCN over-smoothing: it states that deeper GCNs are prone to over-smoothing and that this
significantly reduces EEG recognition accuracy, and its multilevel design exists to
mitigate it. Over-smoothing is a recognised failure mode inside this exact lineage, which
makes measuring it on our own six-layer model cheap and well-motivated rather than
speculative.

### 17.4 Tier C — architectural redirection

Only if tier A confirms the diagnosis. Both preserve the project's framing — a structured,
geometry-aware spatial prior — rather than abandoning it for a generic CNN.

- **C1. Move from the graph to the SPD manifold.** The spatial covariance matrix *is* the
  electrode relationship structure, and it lives on a Riemannian manifold. Tensor-CSPNet
  [19] extends SPDNet to time-frequency blocks. The viva argument is clean: *we kept the
  geometric prior and fixed the geometry* — a Chebyshev filter on a thresholded correlation
  graph is a lossy, sparsified, Euclidean approximation to the covariance structure
  Riemannian methods use exactly. **Caveat: closer to "different project" than "next
  ablation."** And note 17.0 — the realistic prize is a few points, not a transformation.
- **C2. Build on attention, the one component that worked.** H2 is the only supported
  hypothesis (+0.0600, p = 0.0005) and 9.10 established the mechanism: per-sample deviation
  from uniform has median 104.5 %, max/min 4.9x, but at different moments per trial.
  **Does the learned peak track a measurable ERD onset?** Correlating per-trial argmax
  against per-trial mu-band desynchronisation latency would convert a mechanism story into
  evidence. **Same correction as B4:** the checkpoints 9.10 used were removed in the
  cleanup, so this needs one arm re-run with `--checkpoints` first.
  **Do this jointly with B5** — attention
  over 37 windows of 0.4 s and attention over 8 windows of 1 s are different mechanisms.

### 17.5 Tier D — attack the sample-size constraint

Given 17.1, this tier is more important than the first draft implied: sample size is now a
live competing explanation, not just a background constraint.

- **D1. MI foundation-model pretraining.** MIRepNet [20] pretrains on ~50,000 trials from
  seven datasets and reports +19.4 points over EEGNet on BNCI2014001 while fine-tuning on
  fewer than 30 trials per class.
  > **Leakage warning.** Several MI pretraining corpora include EEGMMIDB. Fine-tuning a
  > checkpoint that saw our test subjects during pretraining is subject-level leakage — the
  > exact failure this project exists to document. Audit the pretraining manifest; if
  > EEGMMIDB is in it, pretrain from scratch with the 103 subjects held out.
- **D2. Augmentation.** Cheaper, lower ceiling — reported MI gains cluster around 14 % of
  unaugmented accuracy. **Must be generated from training trials only, inside the fold,
  asserted in code.**
- **D3. A second dataset — BCI IV-2a.** Scoped at the start (decision 6), never run. Now
  the strongest single addition to the *paper*: 13's first limitation is "one dataset, one
  architecture." It also disambiguates a real alternative — EEGMMIDB subjects performed a
  screening protocol with no feedback training and [23] rates it the hardest and most
  heterogeneous MI dataset it benchmarked, whereas IV-2a subjects were trained with
  feedback and its ceiling is known.

### 17.6 Tier E — reframe what gets reported

**Promoted after the second pass. This is no longer a tidy-up tier; on the evidence below
it may be the most defensible contribution left.**

**E1 IS NOW DONE.** `scripts/subject_heterogeneity.py` (added 2026-09-02) produces the
table below and `figures/subject_heterogeneity.png` from the existing log, with no compute.
It is labelled EXPLORATORY in its own output: 10.3 fixed the confirmatory list in advance
and this is not on it — it is a descriptive re-analysis of a pre-registered comparison.

| 103 subjects, per-subject means | wins | Pearson *r* | mean gap | median gap |
|---|---|---|---|---|
| GCN + attention vs **CSP+LDA** | **37/103 (36 %)** [27 %, 46 %] | **+0.825** | −0.0202 | −0.011 |
| GCN + both (the H4 arm) vs **CSP+LDA** | **34/103 (33 %)** [24 %, 43 %] | +0.831 | **−0.0263** | −0.033 |
| GCN + attention vs **fixed graph** (H2) | **82/103 (80 %)** [72 %, 87 %] | +0.823 | +0.0600 | +0.056 |

Between-subject SD is **0.1247** against a method gap of ~0.02-0.026, so **subject identity
accounts for on the order of 23-38x the variance that the choice of method does**. And
*r* ≈ 0.83 between two structurally unrelated methods means they largely succeed and fail
on the *same people*: what separates subjects is not which algorithm they get.

**The third row is the most useful thing this analysis produced, and it was not anticipated.**
The two findings differ sharply in evidential quality once you look past the means:

- **H2 (attention) is robust per subject** — it wins on **80 %** of them, and the CI excludes
  a coin flip by a wide margin. This is a consistent effect, not an average of a scattered
  distribution.
- **H4 (the CSP gap) is a thin mean difference over an interleaved distribution** — we win on
  a third of subjects, the median per-subject gap is −0.033, and the per-subject differences
  run from −0.267 to +0.200. The population-level verdict is correct and pre-registered, but
  "CSP is better" is a statement about a mean, not about subjects.

Both statements should appear in the paper. Reporting the H4 mean without the 33 % win rate
overstates it; reporting the 33 % win rate without the pre-registered mean would understate
it. **11 of 103 subjects (11 %) are at or below chance under *both* methods**, consistent
with the 15-30 % BCI-illiteracy rate and with a subject-level ceiling no architecture lifts.

The external benchmarks agree, on the same dataset. [24] finds the single best pipeline is
optimal for **only 35 % of PhysionetMI participants**, that matching pipelines to
individuals is worth **about seven accuracy points** — versus the 0.0263 we are chasing —
and that cov-tgsp and CSP are statistically indistinguishable on the largest cohort
(Nemenyi p = 0.27, Kendall's W = 0.11). [23] finds 93 distinct winning pipelines across 109
subjects.

- **E1. Report the per-subject distribution, not the mean. DONE — see above.** "Our mean is 0.6038, CSP's is
  0.6239" is answering a question the data does not support. "Our model wins on 36 % of
  subjects, the two methods correlate at 0.825, and subject identity dominates method
  choice by ~38x in variance" is both more honest and more interesting. **The data already
  exists; this is an afternoon of analysis.**
- **E2. Screen for BCI illiteracy. DONE — see 12.7.** Resting SMR explains ~30 % of the between-subject variance, equally for the network (+0.547) and CSP (+0.548).
  *Original entry:* 15-30 % of people do not produce classifiable
  sensorimotor rhythms; our 15 of 103 at or below chance sits inside that range, and
  *r* = 0.825 is exactly what a subject-level ceiling would produce. Resting mu-band power
  from the baseline runs — already downloaded, never used — is the standard predictor.
- **E3. Contribute the pipeline to MOABB.** Makes the audit evaluable across 14 MI datasets
  and turns a single-paper rebuttal into reusable infrastructure.

### 17.7 A scope expansion available for free

**Verified in the local checkout during this review, not inferred.** The leakage documented
in 4.4 is not a property of GCNs-Net. It is a property of the **shared data pipeline** that
`SuperBruceJia/EEG-DL` uses for nearly every model it ships:

- **22 of the 25 `main-*.py` entry points** import `Models/DatasetAPI/DataLoader.py`, which
  reads the `training_set.csv` / `test_set.csv` pair produced by `make_dataset.m`. The three
  that do not are `main-Transformer.py`, `main-finetuning_model.py` and
  `main-pretrain_model.py`.
- **All five `make_dataset.m` variants** (For-CNN-, For-DNN-, For-GCN-, For-RNN-,
  For-Siamese-based-Models) contain both the CAR line and the `randperm` shuffle. In the
  GCN variant these are lines 22 and 135; in the other four, lines 27 and 76/77 — so 4.4's
  "line 22" citation is correct *for the GCN pipeline* and should be qualified as such.
- The affected entry points span GCN, ResGCN, DenseGCN, CNN, ResCNN, DenseCNN,
  FullyConvCNN, DNN, LSTM, BiLSTM, GRU, BiGRU, RNN, BiRNN, their attention variants, and
  the Siamese network.

At least two published papers draw their headline numbers from this pipeline: GCNs-Net [1]
at 93.06 %, and **[25] "Attention-based Graph ResNet" (Jia, Hou, Shi, Li, 2020) at 98.08 %
subject-level and 94.28 % for 20 subjects** on the same PhysioNet dataset, with the same
four L/R/B/F classes, splitting by dividing "the total dataset" into ten random pieces.

**This lets the paper make a materially larger and still fully evidenced claim**: the
inflation is architecture-independent, because the split happens upstream of every
architecture. Framed carefully — we verified the code path and reproduced the effect for
one model, so the claim is about *the pipeline*, and for other architectures it is a
prediction our evidence supports rather than a measurement we have made.

### 17.8 If only three things happen

Revised from the first draft: A2 is promoted to joint-first because 17.1 gave it a specific
hypothesis to discriminate, and E1 replaces the A3+B1 pairing because it is nearly free and
the evidence for it strengthened considerably.

| # | Direction | Cost | Why this one |
|---|---|---|---|
| **A1 + A2** | Unconstrained spatial filter; EEGNet on identical folds | Small | Together they discriminate *graph prior* from *sample size* — the two live explanations for the whole result |
| **E1** | Per-subject distribution from the existing log | Near zero | 37/103 wins, *r* = 0.825, ~38x variance ratio. Already computed here; needs writing up, not running |
| **B4 + B5** | Dirichlet energy (needs one arm re-run for checkpoints); window-length sweep | Small | A second, independent mechanism for the failure, plus the one untested knob that interacts with the component that works |

If a fourth fits, make it **D3 (BCI IV-2a)** — it addresses 13's first limitation and is the
objection most likely to be raised in a viva.

**What not to do.**

- **Do not tune to close the 0.0263 gap.** 17.0 now puts a number on why: the whole spread
  between top pipeline families on this dataset is 0.054, the ceiling after 216,714
  configurations is ~0.65, and our design cannot reliably detect anything below 0.020.
- **Do not add architectural components before A1 and A2.** Two of three repairs did
  nothing and there are now *two* competing explanations for why. Adding a fourth before
  discriminating them repeats the mistake at greater cost.
- **Do not quote a 4-class number without the family decomposition.**
- **Do not fine-tune a public EEG foundation model without auditing its pretraining
  manifest.** See D1.
- **Do not add a third dataset.** IV-2a answers the generalisation question; a third adds
  cost without adding an argument.

### 17.9 Provenance and reading depth

Two passes: nine searches on 2026-09-02, then a second deeper pass the same day adding
seven searches, four full-text fetches and **four analyses of our own experiment log and
the local EEG-DL checkout**. The second pass changed conclusions — 17.0 and 17.1 did not
exist before it, and A2 and E1 both moved up.

**All twelve new references were verified against Crossref or the arXiv API before being
written down**, continuing the practice that caught the wrong author list on [1] and that
caught a fabricated DOI in the first draft of the 3.5 note added today.

**Read in full (HTML full text or complete results summary):** [15], [16], [23], [24],
[20], [14], [25].
**Read at abstract level only:** [19], [21], [22], [18] — paywalled or returning
unparseable PDFs. Claims from those are stated directionally, and any that becomes
load-bearing should be read properly first.

**Still open, and it is 8.1's action item.** [4] MHD-GCN uses PhysioNet *and* BCI
Competition data, but **its evaluation protocol remains unknown** — ScienceDirect and SSRN
both refused retrieval. [5] SAMGCN's over-smoothing motivation is confirmed but its
protocol likewise is not. **The novelty claim in 8.1 still rests on abstracts.** Obtaining
these two through institutional access is the single cheapest way to firm up the
positioning, and it needs a library login rather than a search engine.

Two absence-of-evidence results, stated as such rather than as findings: four targeted
searches found **no published critique or replication of GCNs-Net**, and a further search
found **no published treatment of the EEGMMIDB run-family confound**. Neither is a
systematic review.

### 17.10 References added 2026-09-02

All verified against Crossref or the arXiv API on 2026-09-02.

**[14]** Z. Shuqfa, A. Lakas, and A. N. Belkacem, "Increasing accessibility to a large
brain-computer interface dataset: Curation of PhysioNet EEG Motor Movement/Imagery dataset
for decoding and classification," *Data in Brief*, vol. 54, art. 110181, 2024.
DOI: 10.1016/j.dib.2024.110181. — **independently excludes exactly our six subjects; see 3.5.**

**[15]** S. Chevallier, I. Carrara, B. Aristimunha, P. Guetschel, S. Sedlar, B. Lopes,
S. Velut, S. Khazem, and T. Moreau, "The largest EEG-based BCI reproducibility study for
open science: the MOABB benchmark," arXiv:2404.15319, 2024.

**[16]** M. Eder, J. Xu, and M. Grosse-Wentrup, "Benchmarking brain-computer interface
algorithms: Riemannian approaches vs convolutional neural networks," *Journal of Neural
Engineering*, vol. 21, art. 044002, 2024. DOI: 10.1088/1741-2552/ad6793.

**[17]** H. He and D. Wu, "Transfer Learning for Brain-Computer Interfaces: A Euclidean
Space Data Alignment Approach," *IEEE Transactions on Biomedical Engineering*, vol. 67,
no. 2, pp. 399-410, 2020. DOI: 10.1109/TBME.2019.2913914.

**[18]** D. Wu, "Revisiting Euclidean alignment for transfer learning in EEG-based
brain-computer interfaces," *Journal of Neural Engineering*, vol. 22, art. 031005, 2025.
DOI: 10.1088/1741-2552/addd49.

**[19]** C. Ju and C. Guan, "Tensor-CSPNet: A Novel Geometric Deep Learning Framework for
Motor Imagery Classification," *IEEE Transactions on Neural Networks and Learning Systems*,
2023. DOI: 10.1109/TNNLS.2022.3172108.

**[20]** D. Liu, Z. Chen, J. Luo, S. Lian, and D. Wu, "MIRepNet: A Pipeline and Foundation
Model for EEG-Based Motor Imagery Classification," *Knowledge-Based Systems*, vol. 343,
art. 115966, 2026. DOI: 10.1016/j.knosys.2026.115966.

**[21]** D. Klepl, M. Wu, and F. He, "Graph Neural Network-Based EEG Classification: A
Survey," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 32,
pp. 493-503, 2024. DOI: 10.1109/TNSRE.2024.3355750.

**[22]** A. El Badaoui, H. Ben Alla, M. Hilali, S. Ben Alla, and A. Ezzati, "Representation
and Geometric Collapse in Spatiotemporal EEG Classifiers: A Mathematical Diagnostic
Framework," *Signals*, vol. 7, art. 76, 2026. DOI: 10.3390/signals7040076.

**[23]** X. Vasques, P. Barbaste, and O. Oullier, "Subject-Level Heterogeneity in EEG Motor
Imagery Decoding: A Large-Scale Benchmark and Portfolio-Based Reduction of the Search
Space," arXiv:2607.22778, 2026.

**[24]** X. Vasques, P. Barbaste, and O. Oullier, "Average Rankings Mask Per-Subject
Optimality: A Friedman-Nemenyi Benchmark of EEG Motor-Imagery BCI Decoders,"
arXiv:2606.24394, 2026.

**[27]** Y. Xie, K. Wang, J. Meng, J. Yue, L. Meng, W. Yi, T.-P. Jung, M. Xu, and
D. Ming, "Cross-dataset transfer learning for motor imagery signal classification via
multi-task learning and pre-training," *Journal of Neural Engineering*, vol. 20,
art. 056037, 2023. DOI: 10.1088/1741-2552/acfe9c.

**[28]** R. Mane, E. Chew, K. Chua, K. K. Ang, N. Robinson, A. P. Vinod, S.-W. Lee, and
C. Guan, "FBCNet: A Multi-view Convolutional Neural Network for Brain-Computer Interface,"
arXiv:2104.01233, 2021.

**[26]** B. Blankertz, C. Sannelli, S. Halder, E. M. Hammer, A. Kübler, K.-R. Müller,
G. Curio, and T. Dickhaus, "Neurophysiological predictor of SMR-based BCI performance,"
*NeuroImage*, vol. 51, no. 4, pp. 1303-1309, 2010. DOI: 10.1016/j.neuroimage.2010.03.022.

**[25]** S. Jia, Y. Hou, Y. Shi, and Y. Li, "Attention-based Graph ResNet for Motor Intent
Detection from Raw EEG signals," arXiv:2007.13484, 2020. — **same group and same shipped
pipeline as [1]; reports 98.08 % subject-level. See 17.7.**

---

### 17.11 Reaching 0.70 — what the target actually requires

*Written 2026-09-03 in response to a target of >0.70 mean within-family, i.e. surpassing
ShallowConvNet's 0.6725. Eight searches; every reference verified against Crossref or the
arXiv API. The three G arms were mid-run and no G number is used here.*

#### 17.11.1 The target is +0.0275, and the arithmetic says where it must come from

ShallowConvNet over 103 subjects: **mean 0.6725, SD 0.1348, median 0.644**, quartiles
[0.433, 0.567, 0.644, 0.772, 0.989]. **41 of 103 subjects are already at or above 0.70**;
**7 are below chance.**

Reaching a *mean* of 0.700 needs **+0.0275**. Three routes, and only one is possible:

| route | what it requires |
|---|---|
| **A. lift every subject** | +0.0275 each — a broad, uniform improvement |
| **B. lift the bottom quartile only** | those 26 subjects from 0.518 to **0.627** (+0.109 each) |
| **C. lift the 7 sub-chance subjects only** | to **0.871** each — impossible |

**The deficit lives in the lower tail, not at the top.** The median (0.644) sits well below
the mean target, and the distribution is left-skewed by subjects the model barely decodes.
That is the same tail 12.7 showed is partly physiological: the resting SMR predictor
explains ~30 % of between-subject variance, and subjects who fail under *both* CSP and the
network have 1.70 dB less mu prominence. **Some of route B is not model-addressable at all.**

This immediately ranks the candidate levers: the ones that help *weak* subjects are worth
more here than the ones that sharpen already-good ones.

#### 17.11.2 The strongest lever, and it needs no new data

**Cross-subject pre-training followed by per-subject fine-tuning.** Pre-train on the other
102 subjects, then fine-tune on the target subject's *training folds only*. The target's
test trials are never seen, so this is leakage-free by construction and needs no external
dataset.

**This is the direct synthesis of two results we already have and have never combined.**
9.9 measured LOSO at 0.6040 and subject-dependent at 0.6038 — statistically the same
number, reached from opposite directions. LOSO has ~100x the training data and no
subject-specific calibration; subject-dependent has calibration and almost no data.
**Neither uses the other's information.** Pre-train-then-fine-tune uses both, and 9.9's own
conclusion — that the subject-dependent protocol is data-starved — is the argument that it
should work.

[27] reports a maximum accuracy increase of **7.76 %** from pre-training in general, and
**27.34 % where training data is limited**, along with faster convergence and a **75.22 %
reduction in the variance of accuracy across subjects**. That last figure matters more than
the first: variance reduction is precisely route B, lifting the weak tail.

Our fine-tuning stage would have ~36 training trials per fold — squarely in the
"limited training data" regime those numbers describe.

#### 17.11.3 The strongest architectural lever

**A filter bank, not a single learned temporal filter.** FBCNet [28] band-passes into
several fixed frequency bands, learns a spatial filter *per band*, and aggregates time with
a variance layer. It reports 76.20 % 4-class on BCI IV-2a and **up to 8 % higher binary
accuracy** on other datasets, and is explicitly motivated by *"lack of adequate training
samples"* — our exact constraint.

`GraphBandPowerNet` currently has **one** temporal convolution bank feeding one spatial
stage. FBCNet's insight is that band and spatial filter should be chosen *jointly per band*,
which is what makes FBCSP stronger than plain CSP — and our CSP baseline is plain CSP, not
FBCSP (13). Extending `graphbp` to multiple bands is a contained change: the temporal conv
becomes a fixed filter bank, and the spatial stage runs per band.

#### 17.11.4 The rest, ranked and costed honestly

| lever | reported gain | our expectation | cost |
|---|---|---|---|
| **Pre-train + fine-tune** [27] | up to +7.76 %, +27.34 % on limited data, −75 % accuracy variance | **largest single lever; targets the weak tail** | medium — a pre-training stage per fold |
| **Filter bank (FBCNet-style)** [28] | up to +8 % binary | moderate; compounds with the above | small — contained change to `graphbp` |
| **Window length sweep** (B5) | untested anywhere here | unknown; ShallowConvNet is handicapped at 0.40 s so this may raise the *bar* rather than help us | ~6 h measured |
| **Data augmentation** | 8-36 % *relative* in the literature, wide spread | small in absolute terms; the reported figures are relative and mostly from weaker baselines | small |
| **Ensembling existing arms** | — | **ruled out**: the per-subject *oracle* over all five arms is 0.6946, so even perfect selection lands under 0.70 | none |

#### 17.11.5 The honest assessment

**0.70 is reachable but not by a safe margin, and only if pre-training delivers near its
reported range.** The arithmetic is unforgiving: +0.0275 on a mean whose median is 0.644,
concentrated in a tail that 12.7 shows is ~30 % physiological.

Three statements to keep in view:

1. **The design detects ~0.020** (17.2). A result of 0.69 or 0.71 is one draw from a
   distribution, not a rank ordering. Any claim of "we beat ShallowConvNet" needs the
   paired CI and the per-subject win rate, not the mean.
2. **0.6725 already exceeds the best of a 216,714-configuration published search** (17.0).
   We are not chasing a number someone else has hit; we would be setting one.
3. **CLAUDE.md forbids tuning to move a number.** The correct framing is *"does
   pre-training plus a filter bank help, and by how much"* — pre-registered, with the
   direction fixed in advance — not *"reach 0.70"*. If the answer is 0.68, that is the
   result. **Deciding in advance that 0.70 is the deliverable is how a project ends up
   reporting the best of many attempts.**

#### 17.11.6 Recommended order

1. **Finish the G arms.** They are 46 % complete and answer a pre-registered question
   (9.12). Interpreting a new architecture before the current one is scored is exactly the
   stacking CLAUDE.md forbids.
2. **Pre-train + fine-tune on the best existing architecture**, whichever the G arms name.
   Largest expected gain, no new data, leakage-free, and it targets the weak tail.
3. **Add the filter bank** to that architecture. Contained, compounds, well-motivated.
4. **Then** the window sweep, which also removes the A2 caveat (13).

Steps 2 and 3 should be pre-registered together before either is run, with the primary
outcome fixed as mean within-family accuracy and the comparison paired against
ShallowConvNet.

---

## 18. Changelog

### 2026-11-07

- **Montage-robustness Phase 1 result recorded (9.25, EXPLORATORY): a null, in fact a
  measured loss. The fifth challenger to the ~0.67 -> 0.73 picture measured nothing in its
  favour.** The node-agnostic region-pooled graph model (M1, `ROB1-chebrp-*`) trails
  ShallowConvNet (M2, `ROB1-shallow-*`, interpolation) on every cap, 20 development
  subjects, 100 clean + 700 condition rows per tag. **Full guard fires** (M1 - M2 =
  -0.1117 [-0.1356, -0.0867]); both primaries are losses (IV-2a -0.0633, random 25 %
  -0.0940, CIs exclude zero, M1 ahead on 3/20 and 1/20). All five 9.24.6 predictions
  are refuted. The Phase 2 gate (M1 ahead on a primary by >= 0.020) is not met, so
  **Phases 2 and 3 are not run** and the experiment stops here, as 9.24.7 committed.
- **Likely mechanism, exploratory and not proven:** M1's pre-trained weights transfer
  (fine-tuning starts above chance) but it underfits (train accuracy 0.58 -> 0.64 against
  PTG-gbp's 0.65 -> 0.88); node-shared Chebyshev weights plus region-averaged band power
  remove electrode-specific spatial filtering, the same property that lets it run on any cap.
- **POST HOC, labelled as such and not part of the decision:** M1 loses less than M2 from
  its own Full score (difference of differences +0.0483 [+0.0139, +0.0828] on IV-2a,
  +0.0177 [+0.0059, +0.0309] on random 25 %), yet is behind in every condition. This
  does not make M1 the better model on any cap. Added `drop_from_full_table` and a
  `--post-hoc` flag to `scripts/robustness_results.py`, with one unit test.
- **Provenance:** the run crashed twice on host-memory errors, was paused on 2026-11-05,
  and resumed with `--resume` after the behaviour-neutral release-per-group change
  (`8f47b73`); 160 rows are stamped `3562b23`, the rest `8f47b73`. M2's clean scores
  reproduce the logged PTG-shallow rows exactly for all 20 subjects. `CLAUDE.md` Expected
  Results updated with one bullet.

### 2026-11-04

- **All documents moved into `docs/`.** Papers, review deliverables, speech guides, decks,
  PDFs and this file now live in `docs/`; only `README.md` and `CLAUDE.md` stay at the
  root. Image links in the moved Markdown now read `../figures/...` so they still render in
  previews. Every build script reads its source from and writes its output to `docs/`, and
  resolves figures relative to the document. `tests/test_documentation_consistency.py`
  resolves figure links the same way. Verified: 337 tests pass, all seven document builds
  dry-run cleanly up to the Edge print step, no image link is unresolved.
- **Backfill: PW-1 Review-1 deliverables, produced 2026-09-20 → 2026-10 and not logged
  here until now.** The detail is in `tasks/todo.md`; in summary:
  - `REVIEW_PAPER_2.md` / `Review-Paper-2.pdf` (36 references, adds a leakage-audit figure
    and §7.4 with the pre-training results) and the one-page `REVIEW_ANALYSIS_2.md`.
  - `PW-1_Review-1_Presentation.pptx` (12 slides) and `PRESENTATION_SPEECH.md` /
    `PW-1_Presentation_Speech.pdf` (3-speaker script, 54 viva questions).
  - A "no implementation" (NIP) variant of all four, removing every internal result, made
    because the Review-1 circular scopes that review to survey and design.
- **Open issues found while auditing those deliverables — not yet fixed.** They predate this
  entry and matter before the review paper is submitted for Review-2:
  1. **The 0.650 comparison prohibited by 9.15.4 is made repeatedly.** The speech guide
     (Slide 12 script, Q-answers), `build_presentation.py` (Slides 2 and 12) and Figure 4
     in `figures/review_2/` present the pre-trained 73.01 % as "surpassing the 65.0 %
     Vasques ceiling". PTG sees ~92 extra subjects; that comparison is not like-for-like.
     The automated guard missed it because it searches for `0.7206`/`0.650`, not
     `73.01 %`/`65.0 %`.
  2. **GBP-Net is called a "breakthrough architecture"** in both review papers (abstract,
     contributions, §7 heading) and the speech guides. 9.17 measured the pre-training
     gain as architecture-agnostic (ShallowConvNet gains the same), so the 0.73 cannot be
     attributed to the architecture.
  3. **"93 % → 50.9 %, a 42-point collapse"** sets a four-class subject-level figure against
     a binary within-family one. The 2026-09-20 entry already notes the comparable form
     for the 0.650 benchmark; the same care is missing here.
  4. Nothing since commit `f596cca` (2026-09-20) is committed. *(Resolved the same day:
     commits `d920d45` and `8856ab3`.)*
- **Decision: the review paper submitted before Review-2 is the "no implementation"
  version, and every experimental result is kept for a separate research paper** built from
  `PAPER.md`. Publishing the measurements inside a review without a methods section would
  invite rejection and could count as prior publication of the research paper's results.
  `REVIEW_PAPER_2.md` (with results) is superseded and stays as presented.
- **`REVIEW_PAPER_NIP.md` rewritten for journal submission.** A full audit against the record
  and Crossref found, beyond issues 1–3 above:
  - **Two references with invented details.** *musallam2021* had a title ("…fused with
    graph convolution"), journal and DOI that do not exist; the real Musallam et al. 2021 is
    TCNet-Fusion (*BSPC* 69:102826), not a graph model, so the "GCN-TCNet" entry built on it
    was removed along with the reference. *sun2021* had wrong authors, journal, volume and a
    non-resolving DOI; corrected to *IEEE SPL* 28:219–223, doi 10.1109/LSP.2021.3049683.
  - **A PRISMA claim and database searches that contradicted `REVIEW_PAPER.md`**, which
    states the review is not systematic. Replaced with the honest citation-tracing method;
    the PRISMA reference was dropped. References 36 → 34; Figs. 1–2 regenerated.
  - Own measurements (0.624, 0.630, 0.673, 0.898) presented under other papers' citations;
    TSMNet's gain misquoted (record: 2.4–3.9 % from DSBN); BiLSTM-GCN's 94.2 % and code-level
    claims unsupported; "genuine decoders cap around 65 %" (ShallowConvNet's 0.6608 L-vs-R
    already exceeds it); "four formal theoretical guarantees" with no proofs; a temporal
    kernel of 25 where the model uses 16 for a 64-sample window; a plan that omitted the
    within-family metric G2 calls mandatory.
  - The rewrite cites only published figures and dataset/code facts, describes the EEG-DL
    findings as a code reading whose effect was not measured, states that GBP-Net's
    macro-architecture is ShallowConvNet's, adds a planned graph-free pre-training control,
    removes college-specific wording, and adds declarations (three marked "to be confirmed
    by the authors"). Fig. 4 (NIP) text narrowed to what a code reading supports.
  - Rebuilt: `Review-Paper-NIP.pdf` 11 pages, 34 references; one-page analysis updated
    (34 works) and still one page. 337 tests pass.
- **Still open:** the Review-1 decks and speech guides (both variants) carry the issues
  listed by the deck audit — the 65 % comparison, "breakthrough", the 42-point mixed-metric
  claim, a Slide 9 architecture the model does not have, wrong preprocessing/stack/statistics
  details, and deck-order problems. To be corrected when the Review-2 format is known.
- **Montage-robustness experiment pre-registered (9.24), before any `ROB` run.** Tests
  whether a graph model (M1, region-pooled readout, tags `ROB1-chebrp-binary-*`) degrades
  more gracefully than ShallowConvNet (M2, cached PTG-shallow source models, tags
  `ROB1-shallow-binary-*`) when electrodes are missing at test time. Eight conditions, two
  primaries (IV-2a, Random 25 %), development subjects S001-S020, Phase 1 EXPLORATORY.
  9.24 fixes the region table (printed by `scripts/print_region_table.py`), the decision
  rule and full-montage guard, five predictions and the Phase 2 channel-dropout settings
  before any result exists. Facts recorded for reading the results: every condition
  re-references the test trials over the electrodes present, from the no-reference cache
  (`data/cache_bp830_noref`, verified equal to the referenced cache after re-referencing
  for 103/103 subjects); M1 gets graph and standardiser recomputed from source subjects
  re-referenced over the same subset, M2 gets MNE-equivalent spherical-spline interpolation;
  masks are identical for both arms and nested across fractions within a draw; M2's clean
  scores are bit-identical to the logged PTG-shallow rows for S001. The log held 0 `ROB`
  rows at commit time.

### 2026-09-20

- **Review-paper selection re-tested and documented.** The choice of Klepl, Wu & He (*IEEE
  TNSRE* 2024) for the required one-page analysis was challenged and re-run against explicit
  criteria rather than defended as-is. Candidates were enumerated via Crossref and screened
  for open access with Unpaywall and Semantic Scholar:
  - *Altaheri et al.*, "Deep learning techniques for classification of EEG motor imagery
    signals: a review," **NCAA** 35(20):14681–14722, 2021/2023 — closest to our task, but
    closed access with no OA copy, so the analysis table could not be filled from the full
    text. Rejected.
  - *Saibene, Ghaemi & Dagdevir*, "Deep learning in motor imagery EEG signal decoding: A
    systematic review," **Neurocomputing** 610:128577, 2024 — a PRISMA-style systematic
    review of exactly our task, and on relevance the strongest candidate. Its only OA copy
    (a submitted-version preprint in the Milano-Bicocca repository) sits behind a bot
    challenge and the published version is paywalled; no abstract is exposed by Crossref or
    Semantic Scholar. **Not rejected on merit — rejected because we cannot read it.** If
    institute library access to Elsevier becomes available, this is the paper to re-do the
    analysis from.
  - *Wu, Xu & Lu*, "Transfer learning for EEG-based BCIs: A review of progress made since
    2016," **IEEE TCDS** 14(1):4–19, 2022 — full text readable (arXiv:2004.06286), but it is
    a narrative review that states no databases, no search query and no inclusion criteria,
    so the "databases", "keywords" and "number of papers" rows would all have read "not
    stated". Rejected for the analysis; retained as a citation.
  - **Klepl retained.** Decisive facts, all re-verified against the full text this session:
    it reviews GCNs-Net itself (its ref [77]) and Hou's BiLSTM-GCN ([76]), so its conclusions
    apply to our baseline directly; it is gold OA; and it states its databases (Google
    Scholar, ScienceDirect), query string and search date (1 November 2022) for 63 articles.
  - Two rows added to `REVIEW_ANALYSIS.md` — the selection rationale and the survey's own
    limitations — and the sheet re-tuned to stay within the required single page.
  - **Correction:** the analysis previously stated that motor imagery "accounts for only
    about eight papers" in Klepl's Fig. 3. That count was read off a bar chart and is not
    stated in the text; it has been replaced with the paper's own wording (motor imagery is
    among the three most common tasks) plus the verified fact that the survey deliberately
    does *not* break results down by task. Error 33.
  - `REVIEW_PAPER.md` now positions itself against three surveys rather than one — Klepl
    (by architecture), Saibene (by task) and Wu (by adaptation method) — and notes that none
    of them examines evaluation protocol. References: 29 → 31.

- **PW-1 review paper and its one-page analysis prepared** (`REVIEW_PAPER.md`,
  `REVIEW_ANALYSIS.md`; built by `scripts/build_review_pdf.py` and
  `scripts/make_review_figures.py`). The review covers the 27 surveyed studies plus a
  published GNN–EEG survey [Klepl et al., IEEE TNSRE 2024] used for positioning and for the
  required one-page analysis. It is a literature review: no result from this project's
  experiments is reported in it; section 7 describes the proposed model and evaluation design
  and points to the project report for results.
  - Citations are keys numbered at build time; the build refuses to run if any key is cited
    but undefined or defined but uncited. All 29 references verified against Crossref/arXiv
    except the EEG-DL repository, cited as software.
  - The methodology section describes the actual iterative, citation-tracing selection and says
    it is not a PRISMA systematic review, rather than inventing screening counts in imitation
    of the sample paper.
  - The comparability argument is stated more precisely than in the survey: the benchmark's
    0.650 is for a *binary* task, while GCNs-Net's 93.06 % is *four-class*, which makes the
    gap larger, not smaller.
  - An FBCSP claim ("won its competition category") was removed as not supported by the cited
    2008 paper. Error 32 (the Blankertz effect size) was found while checking claims for this
    document.

### 2026-09-07

- **9.23: domain-specific batch normalisation does not help.** PTD = 0.7224 against PTG's
  0.7301: **−0.0077** [−0.0164, +0.0013], inside the 0.020 MDD, so decision rule B of 9.22
  fires and it is reported as indistinguishable. One of four predictions holds, the trivial
  one.
  - **The magnitude and the direction disagree, and both are reported.** PTD wins on only
    39/103 subjects, sign test p = 0.0176 — a consistent small decrement, which is what
    9.22.4's rule C anticipated: ~1,665 windows per subject make per-domain statistics
    noisier than the ~145,000-window pooled estimate.
  - **The ceiling survives its fourth challenger**, after spatial priors (9.19), window
    length (9.21) and the architecture-agnosticism null (9.17). All were chosen because
    published work reported 2–6 point gains from them.
  - **It sharpens 9.17**: whatever pre-training's +0.0557 buys, it is not normalisation —
    the most direct intervention on the between-subject axis does not reproduce any of it.
  - DSBN implemented in `models/dsbn.py`, wired through an optional `Samples.domains` and
    domain-homogeneous batching. The default path is verified inert: the non-DSBN smoke
    reproduces 0.5520 ± 0.0240 bit-identically after the wiring.

### 2026-09-06 (night)

- **9.21: the window is not the constraint either.** A fourfold sweep (0.40 / 0.80 / 1.60 s)
  on the development set spans **0.0250**, all inside the pre-registered 0.0328 MDD, and the
  best window is the inherited 0.40 s. **Zero of three predictions hold** -- both longer
  windows are numerically worse and the trend is monotone.
  - Combined with 9.19, the ~0.67 within-subject ceiling now survives four spatial priors,
    three architectures, a Riemannian classical method and a fourfold change in the sample
    definition.
  - **Retires a limitation.** 7 calls the graph-free networks' advantage conservative because
    they ran outside their 2-4 s design regime. Inside that regime ShallowConvNet is *worse*.
    The caveat is now measured rather than assumed.
  - **A confound is recorded rather than buried** (9.21.3): stride was fixed, so longer
    windows also mean up to 32 % fewer training samples, and this design cannot separate the
    two explanations.
  - **Second refuted hypothesis of the day.** 9.19 refuted the CSP prior, 9.21 the window --
    both were this session's own literature-driven recommendations, and both pre-registrations
    are the only reason they read as results rather than as stories.

### 2026-09-06 (evening)

- **9.19: the spatial prior does not matter, whatever it is.** CSP-initialising the
  learnable spatial operator -- the intervention CSP-Net reports as worth +6.0 points on a
  ShallowConvNet backbone -- measures **-0.0008** [-0.0094, +0.0076] against the
  graph-initialised control, on 49/103 subjects. Decision rule B of 9.18 fires; one of four
  pre-registered predictions holds and it is the trivial one.
  - **Seven methods sharing almost nothing now span 0.0108** (0.6629-0.6737): four priors on
    the same operator, a graph-free CNN, and a Riemannian classical method. That is a
    ceiling statement, and a stronger claim than 9.13's "the graph contributes nothing".
  - **A3 (cov-tgsp) beats CSP+LDA by +0.0454** on 74/103, putting a number on the "CSP was
    untuned" caveat that 14 has carried unquantified since the start, and reproducing [23]'s
    family ranking on PhysionetMI under our own protocol.
  - Errors 26-29 recorded, including the process-management failure that filled the GPU and
    the fact that this session's own headline recommendation was refuted by measurement.

### 2026-09-06

- **9.17: the pre-training gain is architecture-agnostic, and it is now measured.**
  ShallowConvNet gains **+0.0483** from the same pre-training where the reordered graph model
  gains +0.0557; the difference of differences is **−0.0073** [−0.0204, +0.0056] on 47/103
  subjects. Decision rule A of 9.16.5 fires. Four of four pre-registered predictions hold and
  the eight-point leakage audit passed before any figure was read.
  - **PTG-shallow 0.7369 and PTG 0.7301 are tied**, not ranked: +0.0068 [−0.0022, +0.0155],
    below the 0.010 floor 9.16.3 fixed in advance. Shallow wins L-vs-R and loses B-vs-F.
  - **Two pre-run judgements were wrong and are recorded as such.** The power estimate
    assumed the two architectures' gains would correlate substantially; measured ρ = +0.160,
    so the real MDD was nearer 0.019 than the projected 0.013–0.016. And the projection
    predicted outcome C from PTC-shallow's baseline — it got the point estimate close
    (0.743 projected, 0.7369 measured) and the verdict wrong, by treating a nominal lead as
    a win. The pre-registered floor is what caught it.
  - `CLAUDE.md`'s ban on "our architecture reaches 0.73" is now **permanent rather than
    provisional**, and the 9.15.4 prohibition extends to PTG-shallow's 0.7387.

### 2026-09-05

- **9.16 pre-registers the architecture-agnosticism test** (PTG on ShallowConvNet) before
  the run: the difference of differences `Δ_shallow − Δ_gbp` as the primary quantity, a
  validity check on PTC-shallow, an MDD of ~0.013–0.016, four predictions and three decision
  rules — including the rule for the outcome in which our architecture turns out to be the
  *worse* vehicle for the project's largest effect.

- **Costed the pre-training arm properly and cached the expensive half.** The cost is
  measured from timestamp deltas in the log, not estimated — error 24 was an 8x estimate
  miss, so this is now derived from what actually ran.
  - **PTG's 4.86 h wall clock decomposes as 2.6 h of source-model fitting (~70 %) plus
    1.21 h of fine-tuning plus a 1.12 h power-cut gap.** The source fit is 20 models
    (10 groups x 2 problems) at ~8.0 min each; `prepare_fold` is under 60 s per group and
    is not a factor.
  - **The source models are now cached** (`--source-cache`, on by default, keyed by a hash
    of everything that determines the weights). A resume previously re-paid the full 8 min
    for any group with even one fold outstanding — which the three power cuts each
    triggered. A re-run of the same arm now costs the fine-tuning hour alone.
  - **This exposed error 25.** The cache was supposed to change nothing and changed the
    result, because the fine-tuning stage inherited whatever global RNG state the process
    happened to be in. Fixed by seeding each stage from its own identity. **Consequence:
    the logged PTG and PTC numbers were produced under the old, resume-dependent seeding
    and are not exactly reproducible by the current code** — they were never exactly
    reproducible, since they depended on where the power cuts fell. Future runs are.
  - **There is no engineering optimisation left.** Measured on an idle GPU, `graphbp`
    reaches 8,004 samples/s at batch 256 and 8,431 at batch 384 — flat, so the card is
    already saturated at the production batch size. Peak VRAM at batch 256 is **1.23 GB of
    8.5 GB (14 %)**, so the arm is not memory-constrained either. Making the training
    tensor GPU-resident instead of gathering each batch on CPU measured **1.0x**: the time
    is genuinely in the model's forward and backward pass, not in data movement.
  - **`CLAUDE.md`'s "the model is tiny, VRAM is not a constraint" holds for parameters and
    misleads about activations.** `graphbp` has 23 K parameters but reshapes to
    `(B x t_out, M, F)` before the spatial stage, so activations scale with batch x time.
    At batch 1024 that overflows 8 GB and spills to host memory, collapsing throughput to
    488 samples/s — a 16x slowdown. Batch 256 is not a tuning choice to revisit; it is
    near the ceiling this architecture allows.

### 2026-09-04

- **Pre-training complete (9.15). PTG = 0.7301 — the project's best result, and the first
  time it has cleared 0.70.** +0.0557 over its own control on **93/103 subjects**,
  Holm-corrected. 98/103 above chance; 59/103 at or above 0.70.
  - **Three of four 9.14 predictions hold; the fourth is refuted favourably.** PTG > PTC ✓;
    the gain is larger in the weak tail ✓ (bottom quartile **+0.0720** against top
    **+0.0346**, SD 0.1548 → 0.1490); PTG beats ShallowConvNet ✓; **"PTG does not reach
    0.70" ✗ — it reached 0.7301.**
  - **Eight leakage checks were run before any number was written down**, because
    `CLAUDE.md` says an unexpectedly high result means suspecting leakage first. All pass:
    group assignment matches the deterministic rule, source sets are exactly 103 − group
    size, no target appears in its own source set, **all 1,030 folds report
    `selection = final_epoch`** so nothing was selected on the target's test trials, and the
    gain is spread across groups (SD 0.0189) rather than concentrated. The *shape* is also
    wrong for leakage: a leak would help uniformly or help strong subjects, not help the
    weakest quartile twice as much as the strongest.
  - **9.15.4 records the comparison this number must never be given.** PTG's 0.7206 on
    left-versus-right exceeds [23]'s 0.650 best-of-216,714 — but that benchmark is
    within-session and sees only the target's own data, while PTG consumes ~92 other
    subjects. **It is a different protocol with far more data, not a better pipeline.**
    Every other arm is comparable to 0.650; PTG is not.
  - **9.15.5 records what this does not settle: the gain is a *data* effect, not an
    architectural one.** It would very likely transfer to ShallowConvNet or EEGNet
    unchanged, and that has not been tested. The supported claim is "this architecture plus
    architecture-agnostic pre-training reaches 0.73", not "our architecture reaches 0.73".
    Running PTG on ShallowConvNet is the obvious missing control.
- `CLAUDE.md` Expected Results gains the PTC and PTG rows plus three rules: pre-training is
  the largest single intervention; never compare PTG to the 0.650 ceiling; and do not write
  "our architecture reaches 0.73".
- `posthoc_controls.py` and the figure extended to twelve arms, PTG in its own colour.

- **Infrastructure caught up with the G arms.** Everything below is no-compute and was
  stale: nothing outside this document knew the G arms existed.
  - **12.8 added** — the three G arms and the PTC control in the consolidated table, with
    the Holm-corrected contrasts. 12.8 states plainly that **PTC is a control, not a
    result**: landing at 0.6744, indistinguishable from G3's 0.6737, is what makes the
    eventual PTG − PTC contrast interpretable.
  - **15 reproduction recipe** had no `graphbp` commands at all, so the G arms were not
    reproducible from the document. Added, along with the grouped pre-training commands and
    a comment that `--groups 10` is what makes that arm tractable.
  - **The doc-consistency test now guards the four new headlines** (0.6629, 0.6694, 0.6737,
    0.6744). Each is recomputed from the log and must also appear in a write-up, so a number
    cannot drift or become orphaned. 37 checks, 272 tests overall.
  - **`subject_heterogeneity.py` extended** to the graph-free, reordered and pre-training
    arms — five new comparisons. E1 previously stopped at the attention arm, so the most
    interesting per-subject contrasts did not exist.
  - **`smr_predictor.py` extended** likewise, and now drops arms with fewer than 50 subjects
    rather than correlating against a fragment.
- **A guard added because the extended E1 immediately walked into error 10.** With PTG at 11
  subjects the script cheerfully printed `pre-trained vs control +0.0596, wins 9/11` beside
  the complete arms. Every comparison over fewer than 103 subjects is now labelled
  **`[PARTIAL -- DO NOT QUOTE]`**, and incomplete arms are named in the header. The number
  it printed is not recorded here and has not been read.

- **Grouped pre-training implemented** (`--groups`, default 10), 5 new tests, suite 272.
  `assign_groups` partitions by `index % k`, which is deterministic, needs no seed, and
  interleaves subjects so no group is a contiguous block of the recording order. The
  load-bearing test asserts that **no target is ever in its own source set** for any group
  count — the guarantee grouping had to preserve.
- **Cost, stated with its uncertainty rather than guessed again (error 24).** A single
  pre-training exceeds ten minutes, so it could not be timed inside one tool call. The
  basis is the measured 14.9 min per pre-training at 102 source subjects *while sharing the
  GPU*; alone, at 92 subjects, 8-12 min is the plausible range. Twenty pre-trainings gives
  **~3-4 h**, plus roughly 1.5 h of fine-tunes, so **4-5.5 h for both problems**. **The
  first group's elapsed time will confirm or refute this within about fifteen minutes of
  launch** — the run prints it, and that is the check error 24 skipped.

- **The control arm (PTC) is COMPLETE and lands at 0.6744** — L/R 0.6533, B/F 0.6956,
  SD 0.1548, 89/103 above chance. That is statistically the same as G3 (0.6737) and
  ShallowConvNet (0.6725). **This is the control doing exactly its job**: it shows the
  source-fit representation is neutral, so any PT − PTC difference will be attributable to
  pre-training and nothing else. Had PTC come out at 0.63 or 0.71 the comparison would
  already be confounded.
- **Error 24, mine: I under-costed the pre-training arm by roughly 8x.** I projected 6-8 h
  for both problems from the LOSO timing of ~1.9 min per fold. Measured: **14.9 min per
  pre-training**, giving **~51 h**, not 6-8.
  - **The projection used the wrong architecture's cost.** LOSO was measured on `GCNsNet`
    with temporal attention, which pools the time axis away *before* the convolutions.
    `graphbp` deliberately does not — 9.11.2's whole point is that power is taken *after*
    the spatial stage, so the time axis is carried through it. The spatial op therefore runs
    on `N x T'` rows rather than `N`, with `T' = 49` at our window. On ~36 training trials
    that is invisible (36 s/subject); on 102 pooled subjects it is the dominant cost.
    Roughly 4x from the architecture and 2x from sharing the GPU with PTC.
  - **The lesson is the same one as errors 10 and 18**: a number measured under one
    configuration was carried to another without checking that the configuration matched.
    A timing probe on the actual arm would have cost three minutes and did not happen.
- **The fix, for tomorrow: grouped pre-training.** Pre-training once per *target* means 206
  pre-trainings. Pre-training once per *group of targets* — say 10 groups — means 20, a 10x
  reduction to about **5 h**, and it stays leakage-free: each model excludes the target's
  entire group, so a target is never in its own source set. The only cost is that a target
  draws on ~92 source subjects instead of 102, which is immaterial. This needs a small
  change to `run_pretrain_finetune.py` and a note in 9.14 that the protocol was amended for
  tractability **before** any PT result was read.
- Ran to the 01:00 deadline as instructed. **PT reached 50/515 folds on problem 0-1 only**
  and is far too incomplete to read; `posthoc_controls.py` will refuse it.
- `--stop-at` gained an absolute `YYYY-MM-DD HH:MM` form. A bare `HH:MM` means today and a
  past one stops immediately (error 23), which is right inside a shell loop but makes an
  overnight deadline unexpressible — `01:00` at 22:00 reads as past. The ambiguity was in
  the format, not the rule.

### 2026-09-03

- **G arms complete (3,090 folds) and scored: 9.13. All four 9.12 predictions hold.**
  - **A graph model now beats CSP+LDA.** G1 — the *hard* Chebyshev constraint, the same
    formulation the paper proposes — reaches **0.6629** and beats CSP by **+0.0389**
    (p = 0.0011 Holm, 72/103 subjects). Every previous graph arm lost to CSP.
  - **The reordering was the entire story.** The four changes in 9.11.2 lift the graph model
    from 0.6038 to 0.6629: **+0.0591 on 80 % of subjects.** Nothing about the graph changed.
  - **The graph itself contributes nothing.** Hard constraint (0.6629), free operator
    (0.6694) and soft prior (0.6737) span 0.0108 and are mutually indistinguishable. The
    prior wins on **51/103** subjects — a coin flip. The contribution 17.11 hoped for is not
    there.
  - **The observation that revises 9.11 again.** In the *wrong* architecture, freeing the
    spatial operator was significant (A1: +0.0128, p = 0.0046). In the *right* one the same
    relaxation gains +0.0065 and is not distinguishable. **The graph constraint only cost
    anything while the rest of the architecture was broken.** A1's 16 % was measuring slack
    in a badly-ordered network, not a graph problem.
  - **G3 at 0.6737 is the project's best model and is statistically identical to
    ShallowConvNet** (+0.0012, p = 0.99, 46/103). 17.11 said reaching ShallowConvNet with a
    graph-informed model would be a result and exceeding it would be surprising: the first
    happened, the second did not.
  - **Four-for-four on pre-registered predictions**, which is worth stating because 9.7.1 is
    the counter-example. The difference is that these were written down first.
- **`CLAUDE.md` Expected Results corrected again.** "Graph-convolutional deep learning does
  not work on this data" — added yesterday — is now false. Two G rows added, and the
  attention arm's "the best model we have" note retired.
- **On 0.70:** still +0.0263 away, almost exactly the gap CSP once held over us. Three
  architecturally distinct spatial stages landed within 0.011 of each other *and* of
  ShallowConvNet, which reads as a ceiling for this representation rather than a plateau
  between designs. 17.11.2's pre-training route is untouched by these results and remains
  the strongest untested lever.

- **`scripts/run_pretrain_finetune.py` implemented** (17.11.2), with 8 leakage tests;
  suite is 267. Not yet run beyond a smoke test on four subjects.
  - **Protocol.** For each target subject, pre-train on the other 102 — the target
    contributes nothing, not their trials, not the graph, not the normalisation
    statistics — then fine-tune on the target's own training folds, which are the folds
    `run_subject_dependent.py` uses at the same seed. The arms are therefore paired
    subject by subject and fold by fold.
  - **One representation across both phases.** The graph and standardiser are fit on the
    source subjects and used throughout, because a pre-trained model cannot have its input
    rescaled underneath it between phases. That does mean the normalisation differs from
    the subject-dependent arm, which is why **`--no-pretrain` is built in as the control**:
    the identical pipeline with pre-training skipped, isolating the effect of pre-training
    from the effect of the representation. Comparing against the subject-dependent arm
    alone would confound the two, so both arms must be run or neither.
  - **The target's training trials are excluded from pre-training too**, though they need
    not be in principle. Including them would require a separate pre-training per fold —
    1,030 instead of 206 — and excluding the subject outright is both cheaper and stricter.
  - **Fine-tuning learning rate defaults to 1e-4 against 1e-3 for pre-training.** At ~36
    trials the pre-training rate would destroy the transferred features.
  - Cost: one pre-training per (target, problem), reused across that target's five folds.
    206 pre-trainings at roughly a LOSO fold each (1.57 min measured, 12.5), so **6-8 hours
    for both binary problems**. `--resume` and `--stop-at` both supported.
- **Second power cut, ~15:50. Cost nothing** — the G arms had been stopped at 03:02 and
  nothing was in flight. The log is byte-identical to before: 14,017 lines, zero malformed,
  proper trailing newline, no duplicates.

- **17.11 added: what a >0.70 target actually requires.** Eight searches, two new verified
  references, plus an arithmetic analysis of our own ShallowConvNet distribution.
  - **The target is +0.0275 and the deficit is in the lower tail, not the top.** 41 of 103
    subjects are *already* at or above 0.70; the median is 0.644 and 7 subjects are below
    chance. Lifting only the bottom quartile would need +0.109 each; lifting only the
    sub-chance subjects is arithmetically impossible. **This ranks the levers: what helps
    weak subjects is worth more here than what sharpens strong ones** — and 12.7 already
    showed part of that tail is physiological, not model-addressable.
  - **The strongest lever is cross-subject pre-training then per-subject fine-tuning, and
    it needs no new data.** It is the direct synthesis of 9.9's two numbers, which we have
    never combined: LOSO 0.6040 has ~100x the data and no calibration, subject-dependent
    0.6038 has calibration and almost no data, and neither uses the other's information.
    [27] reports up to +7.76 % generally, +27.34 % on limited data, and a **75 % reduction
    in accuracy variance across subjects** — that last figure is exactly the weak tail.
    Leakage-free by construction: pre-train on the other 102, fine-tune on the target's
    training folds only.
  - **The strongest architectural lever is a filter bank** [28]. `GraphBandPowerNet` has
    one temporal bank feeding one spatial stage; FBCNet learns a spatial filter *per
    frequency band*, reports up to +8 % binary, and is explicitly motivated by small
    training sets. It is also the difference between CSP and FBCSP, and our baseline is
    plain CSP.
  - **Ensembling the existing arms is ruled out by our own data**: the per-subject oracle
    over all five is 0.6946, so even perfect selection lands under 0.70.
  - **Stated plainly in 17.11.5: 0.70 is reachable but not by a safe margin.** The design
    detects ~0.020, so 0.69 versus 0.71 is one draw rather than a ranking; 0.6725 already
    exceeds the best of a published 216,714-configuration search; and `CLAUDE.md` forbids
    tuning to move a number. **The correct framing is "does pre-training plus a filter bank
    help, and by how much", pre-registered — not "reach 0.70".** Deciding in advance that
    0.70 is the deliverable is how a project ends up reporting the best of many attempts.
  - Recommended order: finish the G arms first (46 % done, pre-registered in 9.12), then
    pre-train + fine-tune, then the filter bank, then the window sweep.

- **Error 23, mine, found 2 minutes after it mattered.** `--stop-at` — added 50 minutes
  earlier — treated a time already past as *tomorrow*. Each of the three processes honoured
  03:00 exactly and stopped cleanly at a fold boundary, which is the feature working. But
  the runs are driven by a shell loop over the two binary problems, so when the
  `--classes 0 1` invocation exited at 03:00 the `--classes 2 3` invocation started at
  03:00:0x, read its own deadline as past, and scheduled itself for **2026-09-04 03:00**.
  A flag meant to bound a run to fifty minutes had just started three 24-hour runs.
  - Caught because the user asked for status at 03:02 and the process count was still 3,
    with `-2-3` tags gaining rows. Killed with `taskkill //T //F`; the log came through
    intact again (14,017 lines, zero malformed, no duplicates), and the 26 `-2-3` folds
    written are valid and will be reused by `--resume`. Net cost: about two minutes.
  - **Fixed so a past time stops immediately.** That is the only reading that composes
    correctly under repetition, and the rollover was a convenience I added without
    considering the one usage I then chose. The help text now says so explicitly.
  - The general lesson, and it is the same one as errors 11 and 12: a control that behaves
    correctly *per invocation* can still be wrong *in composition*. The unit worked; the
    loop was the bug.

- **A power cut at ~00:30 killed the three G arms mid-run. Nothing was lost or corrupted.**
  Verified rather than assumed, because this is exactly the scenario error 13 came from:
  the log is 13,493 lines with **zero malformed rows** and ends on a proper newline, and
  `find_duplicates` reports it clean. The locked single-syscall append did what it exists
  for — an interrupted write leaves no partial row, so the fold in flight is simply absent
  and gets redone. **This was the first real test of the offset-0 lock fix (error 22).**
  903 folds survived; `completed_keys` reads all three tags correctly. Committed at
  `7a457dc` so the rows were not exposed to a second interruption.
- **`--stop-at HH:MM` added to `run_subject_dependent.py`.** Checked at a **fold boundary**,
  never mid-fold, so the log never gains a partial row and `--resume` continues exactly
  where it stopped. This is the right way to bound a run: `timeout` or a kill can orphan
  the python child (error 11) and can interrupt a write. A time already past is read as
  tomorrow. Verified by running it against a deadline 45 s out — it stopped cleanly before
  a named fold and printed the resume command.
  - Two false alarms while testing it, both mine: `--stop-at 00:01` at 02:03 correctly meant
    *tomorrow* and I had read the non-stop as a bug; and a piped run killed by a tool
    timeout lost its buffered output, which looked like the deadline never firing. Writing
    to a file instead showed it working. Neither was a code fault.
- Resumed all three G arms at 02:10 with `--resume --stop-at 03:00`.

### 2026-09-02

- **New architecture: `models/graph_bandpower.py` (`GraphBandPowerNet`), implementing the
  four changes 9.11.2 identified plus the graph-as-prior variant.** Built as a *new class*
  rather than more flags on `GCNsNet`, so every existing arm stays bit-identical and no
  prior result is invalidated. `--model graphbp` on the same runner. 16 tests, 259 passing.
  Not yet run beyond a smoke test.
  - **Change 1, and the one that matters: power is taken AFTER the spatial stage.**
    `square -> mean over time -> log`, where `GCNsNet` applies softplus to voltage.
    **This is why 9.6's null result does not close the door.** 9.6 supplied band power as
    node features, computed per channel by FFT *before* the network — power-then-spatial.
    CSP, FBCSP and ShallowConvNet are all spatial-then-power, and the order is essential:
    a spatial filter works by *cancellation* between channels, which needs phase-coherent
    voltage. Square first and the sign is gone. 9.6 tested the ordering that cannot work
    in principle, so its result says nothing about this one.
  - **Change 2: a learned temporal filter first**, replacing reliance on the fixed 8-30 Hz
    band-pass and keeping the time axis alive until power is taken. Temporal attention
    pooled time away *before* the first convolution, so the six graph layers saw four
    numbers per node.
  - **Change 3: depth 1, not 6.** **Change 4: no Graclus node pooling** (`--pool-nodes`
    restores it, so dropping it is an ablation rather than an assumption).
  - **The regulariser variant, which is the actual contribution on offer.**
    `--spatial free --graph-smoothness X` leaves the spatial operator unconstrained and
    adds a Dirichlet-energy penalty over the electrode graph. A1 measured the graph as a
    *constraint* recovering 16 % of the gap; this asks whether it earns its place as a
    *prior*, which is plausible at ~45 trials exactly where a constraint is not. Testing
    against `--graph-smoothness 0` isolates the prior from the parameters.
  - **Prediction recorded before running**, because 9.7.1 is a lesson about not doing that:
    this should land near ShallowConvNet (0.6725) if changes 1-4 are what matter, and the
    graph terms should make little difference either way. *Reaching* ShallowConvNet with a
    graph-informed model would be a result; exceeding it would be surprising.
- **The penalty uses the COMBINATORIAL Laplacian, not the normalised one, and a test
  caught the difference.** `v(D-A)v` is exactly `0.5 * sum_ij A_ij (v_i - v_j)^2`, whose
  null space is the constant vector — "an electrode's weights should resemble its
  neighbours'". The normalised Laplacian's null vector is `D^(1/2)1`, so it would penalise
  a *uniform* spatial pattern, which is not the prior intended. Written normalised first;
  the test asserting a constant operator scores zero returned 18.67 and was right.
- **The `--model` guard now runs both ways.** Adding `graphbp` had quietly re-opened the
  hole error #12 came from: GCNsNet-only flags were accepted and ignored. Both directions
  now exit with an error — GCNsNet flags under `graphbp`, and graphbp flags under any other
  model.
- `Trainer._loss` gained an opt-in `graph_smoothness` hook alongside the existing
  `adjacency_l1` one. Purely additive: any arm that does not set it is bit-identical.

- **Error 22: the experiment log's lock was not actually exclusive.** `_exclusive` seeked
  to end-of-file and then locked one byte, but `msvcrt.locking` locks relative to the
  *current position* — so two processes at different EOFs locked **different bytes** and
  could append simultaneously. This is the guard for error 13, and it was subtly broken.
  Found because its own regression test failed once under full-suite load and passed in
  isolation; a flaky guard is worse than no guard, so it was worth chasing rather than
  re-running. The lock byte is now fixed at offset 0 — a rendezvous every process agrees on,
  and harmless because append-mode writes ignore the seek. 10/10 clean afterwards.
  **No existing data is affected**: `find_duplicates` reports the 12,590-row log clean, and
  the two apparent duplicates found during the audit were a learning-rate sweep that the
  real identity function distinguishes correctly.
- **`tests/test_analysis_scripts.py`** — the three scripts producing numbers quoted in the
  paper had no tests. A bug in a training script shows up as a bad accuracy; a bug in an
  *analysis* script produces a plausible number that goes straight into a paper, which is
  exactly what error 17 was. Nine tests. The SMR predictor gets four of them because it was
  implemented from a published method description rather than a reference implementation and
  its output is now in the abstract: injecting a 10 Hz sinusoid must move it, monotonically
  in amplitude, while scaling the whole signal tenfold must **not** move it — that last one
  is the property distinguishing a fitted-background excess from raw band power.
- **1 Status brought up to date.** It still read "Writeup: Not started", "LOSO evaluation:
  Not started" and "Contribution: Not started", all of which were done. Rewritten with the
  A1/A2/E1/E2 rows and an explicit three-way split of what remains: needs compute, needs
  institutional access, needs the authors.
- Suite is 243 tests, from 234.

- **E2 done (12.7): resting SMR explains ~30 % of the between-subject variance.**
  `scripts/smr_predictor.py` computes Blankertz et al.'s predictor [26] from **run R01**,
  the 61-second eyes-open baseline — downloaded from the start, never used by any experiment
  here, and containing no imagery, so it cannot leak. Correlation with per-subject accuracy:
  **+0.547** for the repaired graph network, **+0.548** for CSP+LDA, +0.480 for
  ShallowConvNet. The 11 subjects at or below chance under both methods average 1.70 dB less
  mu prominence. [26] reports r = 0.53 (~28 %) on a different task; our 30 % is consistent.
  - **The two headline coefficients being +0.547 and +0.548 is the finding.** Two methods
    sharing no component are predicted equally well by a property of the participant
    measured before either ran. That is a subject-level ceiling, and it bounds what any
    architecture could recover on this dataset.
  - Written into 12.7, 13, `PAPER.md` 5.10 (with Figure 5) plus a new contribution 8 and a
    sentence in the abstract, and `RESULTS.md` R8b. Reference [26] verified against Crossref.
- **`tests/test_documentation_consistency.py` — the audit turned into a test suite.**
  Errors 20 and 21 were both documentation errors that repeated readings missed and that a
  mechanical check finds instantly. Nine checks: every headline accuracy quoted in the
  write-ups is recomputed from `runs/experiments.jsonl` and must match to four decimals;
  any table row labelled as a mean must actually be the mean of the rows above it; top-level
  section numbering must be contiguous; referenced figures must exist; and every citation in
  `PAPER.md` must resolve both ways.
  - **Verified by reintroducing both errors**, which the suite caught with the exact
    diagnostic — including the 0.6239-versus-0.6217 discrepancy that was error 20. A test
    that has never failed is not evidence of anything.
  - Two false positives were tightened out first: rows whose labels merely *contain* "mean"
    (`fixed mean pool`, `mean edge weight`) are no longer treated as summary rows.
- **7.1 and 7.2 were written as level-2 headings**, so the numbering check read them as
  three consecutive section 7s. Promoted to subsections; this is the kind of drift the new
  test now prevents.
- Suite is 234 tests, from 205.

- **Documentation currency pass — four documents had fallen behind the A1/A2 results.**
  - **`RESULTS.md` was actively contradicting the paper.** Its headline read "a linear
    method from 2000 beats every repair we tried" and R4 was titled "A linear method from
    2000 wins". Both are now scoped to graph models, a new **R8** reports the controls with
    the figure, and the limitations and conclusion are rewritten. It also carried the same
    cohort-mixing error as the paper (error 20), now corrected.
  - **`PAPER.md` had no per-subject reporting at all** — it argued for careful reporting
    and then quoted only means, which is the precise failure 17.6 E1 exists to correct.
    New **5.10** with `figures/subject_heterogeneity.png` as Figure 4: win rates, r = 0.825,
    the 23-38x variance ratio, and the point that H2 (80 % of subjects) and H4 (33 %) differ
    sharply in evidential quality once decomposed.
  - **`README.md`** carried the 20-subject CSP value (0.6217) labelled as "identical
    splits", "16 of 20 subjects", and a stale test count. All corrected; a fourth headline
    finding added for the controls.
  - **12.4 corrected and 12.5 / 12.6 added**; **13** gained four limitations (post-hoc
    status, the CNNs' design-regime caveat, the unidentified 84 %, and that the published
    classical bar is higher than our untuned CSP); **15** gained the A1/A2/E1 commands.
- **Error 21, self-inflicted and caught by audit: the 5.9 edit deleted the `## 6.
  Discussion` heading.** The discussion text survived intact but the paper ran 5 -> 7 for
  three commits. Found by listing headings while checking whether E1 was represented, not by
  reading the prose. Restored.
- **A recurring mechanical failure, recorded because it caused three separate corruptions
  today:** shell heredocs collapse backslash-newline continuations into a literal `
`.
  It mangled the caption regex, the `msedge` line in 15, and the new 15 recipe block. The
  fix that works is writing the patch to a file and running it, never inlining it in a
  heredoc.

- **`PAPER.md` revised for the A1/A2 results** (5,244 -> ~6,900 words) and the PDF and
  artifact rebuilt. Targeted revision, not a rewrite: the central finding, the leakage
  audit, the confound decomposition and the pre-registered results are unchanged.
  - Abstract, contributions (new #7), conclusion and the discussion's inductive-bias
    paragraph all rewritten. The framing "CSP outperforms every configuration we construct"
    is now scoped to "every *graph-based* configuration" in all four places it appeared.
  - New **5.9**, with `figures/posthoc_controls.png` as Figure 3. It states both competing
    explanations, tests both, and reports that one is refuted while the other is
    quantitatively wrong — rather than presenting the outcome as a clean confirmation.
  - The discussion now says explicitly that our own inference from null results was
    "directionally right and quantitatively wrong by a factor of six", and that the
    attribution we tested also motivates the adaptive-graph literature [4, 5].
  - Limitations gained four entries: 5.9 is exploratory; the CNNs run outside their design
    regime at a 0.40 s window (which makes their advantage conservative but leaves a
    matched long-window comparison unrun); 5.9 does not identify the remaining 84 %; and
    the published benchmark implies a tuned classical baseline would be stronger than ours.
- **Error 20, found while revising: `PAPER.md` 5.4 mixed cohorts in one table.** Its
  per-problem rows were 20-subject development values while its mean was the 103-subject
  figure — the rows did not average to the mean (0.6217 vs 0.6239). Caught by noticing the
  table was internally inconsistent, not by re-deriving it. Recomputed on 103 subjects:
  −0.0794, −0.0809, mean **−0.0802** [−0.0985, −0.0621].
- Errors 15-19 added to 14, which had not been updated since the A1/A2 work began: the
  fabricated DOI, the checkpoint claim, the significance-versus-effect-size bug, the failed
  sample-size prediction, and 9.7.1's inference-from-nulls. **Two of the twenty (17 and 20)
  would have changed a reported number or conclusion had they not been caught**, which is
  now stated in the paper's data-availability section.

- **A1 and A2 RUN on all 103 subjects; results and full analysis in 9.11.** 3,090 folds.
  They resolve 17.1 and they change what the project claims.
  - **Sample size refuted.** Graph-free CNNs on the *same ~45 trials and the same folds*
    reach and pass CSP: EEGNet 0.6296 (vs CSP 0.6239, p = 0.49 — indistinguishable),
    **ShallowConvNet 0.6725 (+0.0485 over CSP, p = 0.0006, wins 67/103)**. Deep learning is
    not below its data threshold here.
  - **9.7.1's mechanism is minor.** A1 lifted graph locality, symmetry and non-negativity
    and recovered **16 %** of the gap (+0.0128, p = 0.0046). Real, four times the learnable
    adjacency's effect, but small — ~84 % of the gap is elsewhere. 9.7.1 now carries a
    superseding banner; the original text is kept verbatim because the gap between what was
    inferred there and what was later measured is itself the finding.
  - **ShallowConvNet's 0.6608 on left-vs-right exceeds the best single configuration [23]
    reports from a 216,714-pipeline search on this dataset**, and it puts 94/103 subjects
    above chance against CSP's 81. It is the project's best result.
- **`CLAUDE.md` corrected: "a linear method from 2000 still wins" is FALSE.** It held for
  every *graph* arm and was generalised too far. The Expected Results table now carries the
  A1/A2 rows and the bullet is replaced with the corrected statement plus the note that both
  arms are post-hoc, not confirmatory.
- **A bug in the analysis code that would have produced the wrong conclusion.**
  `posthoc_controls.py`'s first verdict rule tested whether A1 *significantly* beat the
  fixed graph. With 103 subjects the design detects ~0.020, so an arm can clear significance
  while recovering almost none of the gap — exactly what A1 does. It would have printed
  "9.7.1 supported". The rule now tests the **fraction of the gap recovered**, with
  significance as a precondition. Caught by running the tool on real results, not by
  re-reading it. Also fixed: a `KeyError` on `holm_bonferroni`'s return schema.
- Added `figures/posthoc_controls.png` — the arm ladder with per-subject spread against the
  published ceiling. The ordering is the argument: every arm left of CSP is a graph model,
  every arm right of it is not.
- **`PAPER.md` and `RESULTS.md` now contain claims contradicted by 9.11** and need revision
  before submission. The central finding is untouched and arguably strengthened, but the
  "a linear method still wins" framing must go, and the paper gains a constructive result it
  did not previously have. Flagged in 9.11.4; not yet done.

- **A2 implemented** (`models/eegnet.py`: EEGNet-8,2 and ShallowConvNet; `--model
  {gcn,eegnet,shallow}`; 21 tests, 205 passing). Smoke-run end to end on real subjects for
  both baselines, and the GCN path re-checked afterwards since shared code was touched.
  - **It shares the runner rather than duplicating it.** `--model` selects the architecture
    inside `run_subject_dependent.py`, so folds, seeds, cache, standardiser, windowing,
    trial aggregation, resume and log schema are identical *by construction*.
    `prepare_fold` gained `build_graph: bool = True` — additive, default unchanged — and
    `False` skips only adjacency, coarsening and the node permutation. 11's steps 1, 4 and
    5 are untouched, which is the reason to route A2 through the same function instead of a
    parallel one.
  - **Caveat that belongs in the paper, not just the code.** EEGNet assumes ~500 samples at
    128 Hz and ShallowConvNet ~500+ at 250 Hz; our confirmatory window is 64 samples at
    160 Hz = **0.40 s**, so the published kernel and pool sizes do not fit. Every temporal
    hyperparameter is derived from T and recovers the published value once T is long enough
    (EEGNet `kern_length` = 32 at T=64, 80 at T=640). **A2 is being run outside these
    architectures' design regime**, which is a genuine limitation and independent
    motivation for B5.
  - Graph-only flags now **error** rather than being ignored when `--model` is not `gcn`;
    silently logging a row that claims a configuration the run did not have is error #12.
  - EEGNet's max-norm constraints are applied inside `forward` so the shared `Trainer` is
    untouched — modifying it would alter the arms A2 is compared against, which CLAUDE.md
    forbids.

- **Started on the roadmap rather than only writing it.** Two items delivered:
  - **A1 implemented** (`models/free_spatial.py`, `--free-spatial`, 8 tests; 184 passing).
    `FreeSpatialConv` replaces the fixed Chebyshev basis with free dense `(K, M, M)`
    operators **initialised at that basis**, so step 0 is bit-for-bit the fixed baseline —
    the same design guarantee `dynamic_adjacency` uses, for the same reason. It lifts all
    three constraints the learnable adjacency kept (locality, symmetry, non-negativity),
    which is precisely the distinction 9.7.1 draws. +10,920 params (3 %); refuses to
    combine with `--learnable-graph` because they answer different questions and CLAUDE.md
    forbids stacking. **Ready to run; not yet run.**
  - **E1 done** (`scripts/subject_heterogeneity.py`, `figures/subject_heterogeneity.png`).
    No compute — the rows already existed.
- **E1's result, and one thing it turned up that was not anticipated.** Against CSP+LDA our
  attention arm wins on 37/103 subjects (36 %) and the H4 `both` arm on 34/103 (33 %), with
  *r* ≈ 0.83 between per-subject accuracies of the two methods and a between-subject SD of
  0.1247 — subject identity carries 23-38x the variance of the method choice. **The
  unanticipated part: attention beats the fixed graph on 82/103 subjects (80 %).** So H2 is
  a consistent per-subject effect while H4 is a thin mean difference over an interleaved
  distribution (median −0.033, range −0.267 to +0.200). Both belong in the paper; reporting
  either alone misleads. Also: **11/103 subjects are at or below chance under *both*
  methods**, consistent with the 15-30 % BCI-illiteracy rate.
- **Correction to 17 as first written.** B4 and C2 were both described as needing "no
  retraining" because they could run on existing checkpoints. **They cannot** — the
  2026-09-02 cleanup removed `checkpoints/` and made checkpointing opt-in
  (`--checkpoints`), so nothing is on disk. Both now require one arm re-run with that flag,
  and their cost is corrected from "near zero" to "small". Caught by checking the
  filesystem before recommending the work, not by re-reading the recommendation.
- **A1 is post-hoc and must be labelled so.** It is not on the pre-registered list (10.3)
  and it tests a claim (9.7.1) that was itself written after the confirmatory runs.
  Recorded at the top of 17 so it cannot be quietly promoted to confirmatory later.

- **Second research pass on 17, same day, and it changed the conclusions.** Seven more
  searches, four full-text fetches, and four analyses of our own log and the local
  EEG-DL checkout. Three things moved:
  1. **17.0 (new) establishes the achievable ceiling, which we had never done.** A
     216,714-configuration benchmark on PhysionetMI [23] tops out at 0.650 for the
     matched left-vs-right problem, with top families spanning 0.596-0.650. Our
     untuned CSP (0.5711) and best model (0.6038) are in the normal range, and the
     0.0263 gap is small against a 0.054 spread between families. It also rates
     PhysionetMI the hardest and most heterogeneous MI dataset it benchmarked.
  2. **17.1 (new) records a competing explanation for the CSP gap that 9.7.1 does not
     consider.** [15] reports deep learning needs >150 trials per class generally and
     ~50 on an easy binary problem, while classical methods do well below 50. We have
     ~45 per binary problem and ~36 per training fold. The gap may be sample size
     rather than the graph prior. **A2 (EEGNet on identical folds) discriminates the
     two and was promoted from third to joint-first.** Caveat recorded in both
     directions: MOABB's DL arm was untuned and its classical arm tuned — the reverse
     of our asymmetry, so the biases oppose rather than compound.
  3. **Tier E promoted on the strength of our own data.** Computed during the review:
     our network beats CSP on **37/103 subjects (36 %)**, per-subject accuracies
     correlate at **r = 0.825**, and between-subject SD (0.1247) implies subject
     identity accounts for ~38x the variance of the method choice. External work
     agrees on this dataset — [24] finds the best pipeline optimal for only 35 % of
     PhysionetMI participants and per-subject matching worth ~7 points.
- **17.7 (new): the leakage is a property of the shared pipeline, not of GCNs-Net.**
  Verified in the local checkout: **22 of 25** `main-*.py` entry points read the CSVs
  produced by `make_dataset.m`, and **all five** `make_dataset.m` variants carry both
  the CAR line and the `randperm` shuffle. A second published paper [25] (98.08 %)
  draws from the same pipeline. This lets the paper make a larger claim, framed as a
  verified code path plus a supported prediction for architectures we did not run.
  Also: 4.4's "line 22" is correct for the GCN variant only (the other four are at
  line 27) and should be qualified.
- **Power, recomputed for the viva:** paired H2 difference over 103 subjects has SD
  0.0741 and dz = 0.810, giving a **minimum detectable difference of ~0.020 at 80 %
  power**. Adequately powered for the 0.0263 found; not powered for less.
- **B5 added: raw-voltage window length has never been swept.** Every confirmatory raw
  run is `window=64` = 0.40 s; the `window=160` rows in the log are band-power only.
  ERD develops over 1-2 s, and attention currently operates over 37 windows of 0.4 s,
  so the knob interacts with the one component that worked.
- 8.1's action item is **still open**: [4] and [5] remain unread ([4] uses PhysioNet
  and BCI Competition data; protocol unknown). ScienceDirect and SSRN both refused
  retrieval — this needs institutional access, not another search.
- Added **17. Future directions** — a ranked roadmap compiled from nine literature
  searches, published as an artifact alongside it. Ordered by *what each direction
  would establish*, not by expected accuracy gain, because the project's value now
  rests on a negative result and a negative result is strengthened by controls and
  weakened by tuning. The changelog moved from section 17 to section 18.
- The roadmap's central point: **9.7.1's conclusion is an inference from two null
  results, not a measurement.** A1 (swap the Chebyshev stack for an unconstrained
  spatial mixing layer, holding everything else fixed) is the experiment that would
  make it one, or falsify it. It is small, and nothing else on the list should start
  before it runs.
- Twelve references added ([14]-[25]), **all verified against Crossref or the arXiv API**.
  The first draft of the 3.5 note carried a *fabricated* DOI and the wrong author
  list, caught by running the verification rather than by reading it back. Recorded
  here because it is the same failure mode as error #10 and #12: a plausible-looking
  output is not evidence of a correct one.
- **External corroboration of the exclusion list (3.5).** The published curation of
  EEGMMIDB [14] independently excludes exactly S088, S089, S092, S100, S104 and S106
  and reports N = 103 — a peer-reviewed confirmation, by a different route, of the
  S089-not-S090 correction made on 2026-08-31 from the EDF header audit. The
  exclusion list stops being our judgement call and becomes the field's convention.
  Cite it in the paper's dataset section.
- Four targeted searches found **no published critique, failed replication, or
  reproduction study of GCNs-Net**. The audit is genuinely first. Stated in 17.7 as
  an absence-of-evidence result from four searches, not a systematic review.
- Wrote the complete research paper as `PAPER.md` (5,244 words) and published a rendering.
  Adds what `RESULTS.md` lacks for an academic submission: an abstract, an enumerated
  contributions list, a related-work section placing the work against the GCNs-Net lineage,
  adaptive-graph successors [4,5], classical CSP [10-12] and the ML-leakage literature [8],
  formal method definitions (Laplacian, Chebyshev recurrence, the within-family
  decomposition), a discussion engaging with why the finding generalises beyond one paper,
  and thirteen references.
- **Every reference was verified against Crossref or arXiv before citation**, continuing the
  practice that caught the wrong author list on [1]. Crossref returned the wrong record for
  Defferrard (NeurIPS proceedings index poorly there); confirmed via the arXiv API instead.
  A citation-integrity check confirms all 13 are both cited and defined, with no orphans.
- The paper adds one framing not in `RESULTS.md`: a **dataset-level recommendation** that
  four-class accuracies on EEGMMIDB be reported with the within-family decomposition, since
  the run-family confound is structural and affects any result on this dataset, not only
  ours.
- Produced the typeset paper as `GCNs-Net-Reassessed.pdf` (12 pages, 1.3 MB, tracked in
  the repository) and made it reproducible: `scripts/build_paper_pdf.py` renders
  `PAPER.md` to `build/paper_print.html` with A4 geometry, a fixed light palette, and
  `break-inside: avoid` on tables and figures so neither straddles a page boundary; a
  headless Chromium then prints it. Figures are inlined as base64 data URIs, so the
  intermediate HTML is 1 MB and is git-ignored while the PDF is not. `PAPER.md` stays
  the single source of truth — the PDF and the artifact are two renderings of it.
- Two bugs in that build, both silent: the print CSS selected captions with
  `p em:only-child`, which also matched inline emphasis because `:only-child` ignores
  text-node siblings, so single emphasised words inside body paragraphs were restyled
  and broken onto their own lines; and the first repair of it dropped the ``
  backreference to shell escaping and emptied all three captions. Both were caught by
  reading the rendered pages, not by the build, which exited 0 in every case —
  the same lesson recorded in §14.
- `markdown` **is** now pinned in `requirements.txt` (`Markdown==3.10.3`), reversing the
  earlier decision to leave it out: once the PDF became a tracked deliverable with a
  build script in `scripts/`, the renderer stopped being an ad-hoc tool and became part
  of what has to be reproducible. It is commented as build-only and is not imported
  anywhere in the pipeline.

- Wrote the results up as `RESULTS.md` and as a published artifact. Seven numbered findings
  in publication order, because the argument is cumulative and each step only stands if the
  previous one did: the protocol reproduces (R1), collapses under a correct split (R2), the
  residual is a run-family artefact (R3), CSP+LDA beats it (R4), one repair of three works
  (R5), attention weights each trial differently (R6), subject-independent is not worse (R7).
  Three things were deliberate: H4's negative sign is called out in its own box, because a
  two-sided "supported" verdict read without direction says the opposite of the truth; the
  refuted mean-pool hypothesis is reported rather than omitted; and the inductive-bias claim
  is explicitly bounded to this dataset, protocol and formulation.

- Regenerated `confusion_decomposition.png` on the **full-scale** Phase 1 runs (103
  subjects) rather than 20 development subjects, with the family / within-family
  decomposition printed on each panel. The published formulation's panel now shows the
  argument visually: strong block structure across the run-family divide, but near-equal
  cells *within* each block (0.37/0.32, 0.34/0.33) — family identification with no
  decoding. The 8-30 Hz panel shows `both_feet` at 0.35 against 0.20, i.e. real
  within-family signal.
- **Cleanup: 8.2 GB -> 5.6 GB.** Removed the project-root copy of the source zip (1.9 GB;
  an identical copy in `~/Downloads` was verified present first), `checkpoints/` (763 MB),
  a stray nohup log and `__pycache__`. Kept `data/raw`, both caches, `figures/` and
  `runs/experiments.jsonl`.
- **Fixed the cause of the checkpoint waste.** `run_subject_dependent.py` wrote to
  `checkpoints/{subject}_fold{n}.pt` with no tag, so 42 distinct runs overwrote one set of
  515 files — 763 MB that identified nothing, was referenced by no result, and could not
  be attributed to a run. Checkpoints are now scoped under `checkpoints/<tag>/` and are
  **opt-in** via `--checkpoints`, since no reported result ever loaded one.

- Rewrote `CLAUDE.md`'s Expected Results with measured values and CIs, closing open
  decision 5. Recorded that LOSO came out *above* subject-dependent, that 4-class LOSO has
  never been measured, and added the two reporting rules learned the hard way.
- Closed decisions 1 (scheme (a) used throughout; the paper/code discrepancy is reported
  as a finding) and 4 (option (i) executed).
- **Expanded this document substantially at the user's request.** Added section 11
  (complete methods specification — pipeline, architecture, training, splits, statistics,
  with every deviation from the reference marked and justified), section 12 (all results
  consolidated in one place), section 13 (limitations and threats to validity), section 14
  (every error made and how it was caught), section 15 (exact reproduction commands per
  result), and section 16 (glossary). The changelog moved to section 17.
- Section 14 is the one worth reading twice: the recurring failure mode in this project was
  a process exiting successfully while silently losing work, and every instance was caught
  by an independent count disagreeing with an expected target — never by the process
  reporting an error.


### 2026-08-31

- Diagnosed the stalled `wget` mirror (18/109 subjects); superseded by the user-supplied zip.
- Extracted the full dataset to `data/raw/eegmmidb/`; **all 3,052 files SHA256-verified**.
- Verified environment and GPU (sm_120 / CUDA 13.2 confirmed working).
- Audited all 1,526 EDF headers and all 654 imagery-run annotation channels.
- **Found and corrected the S089/S090 error in the documented exclusion list.**
- Confirmed the run-to-class mapping against MNE `eegbci` documentation.
- Analysed `SuperBruceJia/EEG-DL`; identified it as a `cnn_graph` (ChebNet) fork and
  documented three defects not to port (timepoint samples, always-training BatchNorm,
  missing adjacency construction).
- Established the port plan and module layout.
- Created this document; updated `CLAUDE.md`.
- Read `make_dataset.m`: the reference applies **no filtering**, mean-centres over the
  **whole dataset pre-split**, and splits with `randperm` at the **timepoint level** —
  a third leakage path on top of the two in §4.3. Raised as open decision 4.
- Initialised the git repository with a `.gitignore` covering the 3.4 GB payload;
  first commit `1011e03`.
- Installed `mne`, `h5py`, `torch_geometric`, `pyyaml`, `pytest`.
- Built `configs/default.yaml`, `data/{config,labels,edf_loader,cache}.py`,
  `scripts/build_cache.py`, `tests/test_labels.py` (12 tests passing).
- **Built the full trial cache: 103 subjects, 9,270 trials, 0 dropped, 447 MB.**
- Read the GCNs-Net paper (arXiv:2006.08924). Confirmed the four classes are L/R/B/F
  with no rest (**resolves decision 1, scheme (a)**), that a sample is one timepoint,
  that the adjacency is A = |P| - I computed over **all** signals including test
  (**a fourth leakage path**), and that the 88.57 % "group level" figure is **not**
  subject-independent. There is therefore no published LOSO baseline for this model.
- User chose option (i) for the baseline protocol (**resolves decision 4**); the
  two-run plan is recorded in 7.2.
- Documented the run-family confound inherent to scheme (a) and the diagnostics that
  must accompany any 4-class number (7.1).
- Built `data/splits.py` (trial-level k-fold, LOSO with disjoint validation subjects,
  and the leaky reference split with a `count_shared_trials` audit),
  `data/normalize.py`, `graphs/adjacency.py` and `graphs/laplacian.py`. Test suite now
  52 tests, all passing.
- **Observed on real data:** the paper's adjacency is 98.4 % dense with mean edge
  weight 0.675 - a near-complete graph with little topology, which motivates the
  learnable-adjacency contribution; and the true lambda_max is 1.03, so the reference's
  hardcoded `lmax = 2` confines the rescaled spectrum to [-1, 0.03] and wastes half the
  Chebyshev domain. Both are logged as ablations (6.5).
- Ported Graclus coarsening (`graphs/coarsening.py`), fixing an off-by-one in the
  reference's row-length scan that made the matching consider non-neighbours. Test suite
  now 68 tests, all passing.
- **Coarsening validates the graph:** on the real adjacency it gives a perfect
  64/32/16/8/4/2 hierarchy with zero fake nodes, and every level-0 cluster is a genuinely
  adjacent electrode pair (AF7+F7, CP2+CP4, Oz+Iz, ...). The correlation graph recovers
  scalp topography without any electrode-position input (6.6).

**Corrections after reading the local EEG-DL checkout (`D:\projects\EEG-DL`).** Earlier
entries in this changelog were written from web summaries of the repository. Reading the
source directly overturned four of those claims. The affected sections have been corrected
in place; the errors are recorded here.

1. **CAR, not dataset-wide mean centring.** `make_dataset.m` line 22 operates on a
   (64 channels x N timepoints) matrix, so `mean(...,1)` is the across-channel mean per
   timepoint — Common Average Reference. It uses no cross-sample statistics and is **not
   leakage**. The reference's leakage-path count drops from four to **three**. Config
   updated to `rereference: average` and the cache rebuilt (103 subjects, 9,270 trials,
   42.5 s).
2. **The PCC adjacency construction is not absent.** It is in `make_dataset.m`
   (`abs(corrcoef(...)) - eye(64)`), exactly `A = |P| - I`, matching our implementation.
   The earlier "no reference implementation, highest-risk item" framing was wrong — what is
   true is that it is computed over all pooled data including test.
3. **`b2relu`, not `b1relu`.** `main-GCN.py` sets `params['brelu'] = 'b2relu'`, so the bias
   is `[1, M, F]` (per vertex per filter), not `[1, 1, F]`. Both hardcode `training=True`,
   so that finding stands.
4. **The code's four classes are not the paper's.** Both-fists is discarded and the fourth
   class is resting-state windows drawn at random offsets from the eyes-open baseline run.
   The paper's own 99.42 % single-class accuracy corroborates this. Open decision 1 is
   reopened.

Also corrected: the "98.4 % dense, little topology" observation was measured without CAR.
With CAR the mean edge weight falls 0.686 -> 0.353 and the neighbour/distant contrast rises
1.47x -> 4.45x, so the fixed graph is a stronger baseline than first reported (6.5).

Also noted: the paper states Adam at lr 0.01 and weight decay 1e-2, while `main-GCN.py`
sets lr 1e-6 and regularisation 1e-3.

- Verified the five-item reference list against Crossref and the arXiv title page (8).
  **[1] carried the author list of [2]** — the real authors include Ziqian Hao, Yan Shi,
  Yang Li and Rui Zeng, not S. Zhang / T. Chen / F. Wang — and its year is **2024**
  (vol. 35, no. 6, pp. 7312-7323), not 2022. [5] had no authors listed. [2] and [3] were
  correct. Added the required PhysioNet resource citation (Goldberger et al., 2000).
- Flagged that [4] MHD-GCN and [5] SAMGCN are **direct prior art** for adaptive graph
  learning in MI, so the novelty claim needs to rest on LOSO evaluation, the leakage
  audit, or interpretability rather than on learnable adjacency alone (8.1).
- Ported the model to PyTorch (`models/`), **359,428 parameters**, matching the ~350K in
  `CLAUDE.md`. Found that the reference's `dropout = 0.50` is dead code under `M = [4]`.
  Benchmarked at 2.2 ms / 1024 samples and 54 MB VRAM.
- Built `evaluation/` (metrics with trial aggregation and paired bootstrap CIs, seeding and
  git-stamped JSONL experiment log, fold assembly with every leakage-sensitive step in one
  place, trainer with best-by-validation checkpointing) plus four runner scripts. 108 tests.
- **Ran both arms over 20 subjects (9).** The leaky replication reproduces the published
  88.35 % (we get 88.24 %); the correct protocol gives 27.73 % timepoint / 30.00 % trial,
  a paired gap of **+0.6051, CI [+0.5364, +0.6691]**.
- **Decomposed the correct-protocol result (9.2): run-family accuracy 0.6239 but
  within-family accuracy 0.4809 — at chance.** The above-chance 4-class number is entirely
  the run-family artefact predicted in 7.1. Evaluated correctly, the published formulation
  decodes no motor imagery.
- Added window-based samples (`to_window_samples`) and a band-pass config, then ran a 2x2
  representation ablation (9.4). **Only band-pass + 64-tp window (arm D) is significantly
  above chance within-family: 0.5418, CI [0.5050, 0.5817]**; paired against the published
  formulation, +0.0559 CI [+0.0184, +0.0921]. Neither factor is significant alone.
  Arm D's 4-class accuracy is no better than the others, but its composition shifts from
  run-family artefact toward genuine decoding — the headline number hides this.
- **Recorded a failed prediction**: band-passing did not reduce family accuracy as 9.2's
  drift mechanism implied, and the 7.1 diagnostic was over-stated because family accuracy
  includes a real bilateral-vs-unilateral neural difference. 114 tests passing.
- Published the phased research plan as an artifact (Phases 1-6, decision gates, fallback).
- Wrote `scripts/run_csp_baseline.py` and added a `--classes` flag to the runner so the
  network can be measured on the identical binary problems.
- **Gate A resolved (9.5): CSP+LDA beats the network.** Mean within-family 0.6217 vs
  0.5472, paired −0.0744 [−0.1111, −0.0367], CSP ahead on 16/20 subjects. The ceiling is
  the architecture/representation, not the task, so Phase 3 proceeds.
- Started Phase 1 anchor runs (103 subjects x 3 configs x 3 seeds, unattended). Stopped
  and re-launched them as three concurrent per-seed drivers after profiling showed the
  workload is launch-bound rather than GPU-bound (33 % utilisation, 47 W); concurrency
  raised utilisation to 84 % and cut the estimate from ~11 h to ~4 h with no change to any
  result. The original driver had to be killed to avoid duplicate (subject, fold, seed)
  rows, which would have silently double-counted in every summary.
- **Analysed what attention learned (9.10, exploratory).** Proposed that H2's gain was
  averaging plus regularisation rather than attention, and **refuted it** with a fixed
  mean-pool control: mean pooling gains nothing (-0.0106, p=0.885) while attention beats
  it by +0.0572 (p=0.0009). The population-mean attention profile looks uniform (2.1 %
  deviation) only because individual trials peak at *different* moments — per-sample
  deviation is 104.5 % with a ~5x max/min ratio. The figure was corrected to plot
  individual profiles, since the mean alone supports the wrong conclusion. Decodability
  peaks 0.60 s after cue onset, the expected ERD shape, measured rather than assumed.
- **All five confirmatory hypotheses resolved (9.9), 103 subjects, Holm-corrected.**
  H2 (temporal attention) **supported**, +0.0600 [+0.0463, +0.0745]. H1 (learnable
  adjacency) **not supported**, +0.0031, reproducing the exploratory null on 83 unseen
  subjects. H3 **not supported** — combining adds nothing over attention alone. H4
  significant but **negative**: CSP+LDA still ahead by 0.0263. H5 **supported**, LOSO
  within-family 0.6040 with 88/103 subjects above chance — *higher* than the
  subject-dependent figure, because a LOSO fold trains on ~100x more data, which implies
  the subject-dependent protocol is data-starved.
- **Phase 1 complete at full scale (9.8).** 103 subjects x 3 seeds, zero duplicates.
  Published protocol 0.8975 [0.8794, 0.9147]; published formulation under a correct split
  is **at chance within-family** (0.5090, CI [0.4984, 0.5199]); best representation 0.5470
  [0.5291, 0.5667]. Headline gap **+0.6096 [+0.5767, +0.6411]** paired over 103 subjects.
  Every 20-subject estimate landed inside its full-scale CI, which is evidence the
  development set was not overfitted by repeated inspection.
- **Learnable adjacency: no effect (9.7, exploratory).** Mean within-family 0.5439
  against the fixed graph's 0.5472, paired −0.0033 [−0.0200, +0.0161], p=0.62; better on
  only 7/20 subjects. Checkpoint inspection confirms 80-90 % of first-layer edges moved,
  so the mechanism engaged and simply did not help. With 9.6 this rules out both the
  spectral and the spatial-graph explanations for the CSP gap, leaving the
  graph-convolutional inductive bias itself (9.7.1).
- Stopped an orphaned `phase1.sh` that survived TaskStop for hours, deduplicated the 238
  rows it caused, and added `warn_if_duplicated` to the analysis path. Rewrote the drivers
  with `--resume` on every command and explicit `--subjects` on every ablation. Full
  integrity audit passed: raw data 3,052/3,052 verified, both caches clean, log free of
  duplicates and config-heterogeneous tags, no payload tracked in git.
- **Documentation drift, and its correction.** Sections 6.8-6.10 and the entries below
  were written in one catch-up pass after the user asked whether the documentation was
  still being maintained. It was not: two commits (`8372c11`, `2e17ddb`) landed with no
  doc update, and the learnable adjacency, temporal attention, figures, Holm correction,
  README and requirements were all absent from this record. The cause was treating
  detailed commit messages as the record once the work became code-heavy. They are not —
  a commit message is not discoverable by someone reading the project. Recorded here
  rather than quietly fixed, because a living document that silently skips a day is worse
  than one that admits the gap.
- Implemented the **learnable adjacency** and **temporal attention** (6.8), the **LOSO
  runner** (6.9), and the **figures, Holm correction and reproducibility files** (6.10).
  166 tests passing.
- **Pre-registered the remaining ablations (10)** before reading the learnable-graph
  result: five confirmatory hypotheses with directions, within-family accuracy on the
  binary problems as the primary outcome, Holm correction across the family, and the
  20 subjects used so far demoted to a **development set** with all confirmatory tests
  moved to the full 103. Exploratory analyses listed separately and barred from promotion.
- Added `scripts/run_loso.py`; `loso_folds` existed but nothing called it, so Phase 4 was
  blocked on a missing script. It loads subjects once rather than ~10,000 times and refuses
  configurations whose per-fold sample count makes the run infeasible — the reference's
  one-timepoint formulation needs 5.9M samples per fold, which is another way that sample
  definition fails.
- **Phase 3a: band power as node features (9.6).** 4-class rose 0.3300 -> 0.4222, but the
  decomposition shows ~4/5 of that is better *run-family identification* (0.6100 ->
  0.7378), with within-family up only +0.0298 [+0.0051, +0.0532]. On the matched binary
  problems band power is **indistinguishable from raw voltage** and still loses to CSP by
  ~0.076 on both; band-power left-vs-right is **at chance** (0.4944 [0.4644, 0.5222]).
  **The hypothesis that the network was losing for want of spectral features is not
  supported.** The remaining gap is most likely *spatial* — CSP fits a discriminative
  spatial filter, our graph is fixed and label-blind — which is now a measured argument
  for bringing the learnable adjacency forward ahead of 3b/3c.
- Surveyed the research directions across [1], [2], [4], [5] (8.2). Key takeaways:
  **band-pass filtering to mu/beta is the largest missing lever** and should be the first
  ablation; per-band adjacency is nearly free with existing code; over-smoothing should be
  *measured* on the baseline before being designed against; and **[2] already pairs a GCN
  with temporal attention**, so the contribution must be positioned against [2], not only
  [1]. [5] does not use PhysioNet; [4] does, and its protocol is the critical unknown.
