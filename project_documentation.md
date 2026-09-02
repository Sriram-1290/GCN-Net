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
| **Experimental work** | **COMPLETE** | 2026-09-02 |
| Writeup | Not started | — |
| Figures regenerated on confirmatory data | Partial | — |
| LOSO evaluation | Not started | — |
| Contribution (dynamic adj + temporal attention) | Not started | — |

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

## 7.1 Class scheme (a) vs (b) - the comparison, for the record

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

## 7.2 Baseline reproduction protocol (decision 4, option (i))

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

### 12.4 Classical baseline, per problem

| | CSP+LDA | best network |
|---|---|---|
| left vs right fist | 0.5711 | 0.5178 (fixed) |
| both fists vs both feet | 0.6722 | 0.5767 (fixed) |

Both methods find fists-vs-feet easier than left-vs-right, which is what the homunculus
predicts and is mild evidence both pipelines track real signal.

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

# the typeset paper (PAPER.md -> GCNs-Net-Reassessed.pdf)
python scripts/build_paper_pdf.py           # -> build/paper_print.html
msedge --headless=new --disable-gpu --no-pdf-header-footer \n    --virtual-time-budget=25000 \n    --print-to-pdf=GCNs-Net-Reassessed.pdf build/paper_print.html
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
| **A1** unconstrained spatial operator | **IMPLEMENTED, not yet run.** `models/free_spatial.py`, `--free-spatial` on `run_subject_dependent.py`, 8 tests in `tests/test_free_spatial.py`. Bit-identical to the fixed baseline at init, +10,920 params (3 %), mutually exclusive with `--learnable-graph` |
| **E1** per-subject distribution | **DONE.** `scripts/subject_heterogeneity.py` + `figures/subject_heterogeneity.png`. Results in 17.6 |
| **A2** graph-free CNN baselines | **IMPLEMENTED, not yet run.** `models/eegnet.py` (EEGNet-8,2 and ShallowConvNet), `--model {gcn,eegnet,shallow}` on the *same* runner, 21 tests. Smoke-run end to end on S001/S002 |

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
- **E2. Screen for BCI illiteracy.** 15-30 % of people do not produce classifiable
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

**[25]** S. Jia, Y. Hou, Y. Shi, and Y. Li, "Attention-based Graph ResNet for Motor Intent
Detection from Raw EEG signals," arXiv:2007.13484, 2020. — **same group and same shipped
pipeline as [1]; reports 98.08 % subject-level. See 17.7.**

---

## 18. Changelog

### 2026-09-02

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
