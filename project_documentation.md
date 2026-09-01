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
| 1 | **4-class definition.** (a) left/right fist + both fists + both feet, all MI. (b) a rest class substituted for one MI class. | **REOPENED 2026-08-31.** The paper *names* L/R/B/F, but the shipped code discards both-fists and substitutes resting-state windows from the baseline run (4.5). Paper and code disagree. Current position: keep (a) for the main pipeline, use the code's scheme only in the leaky replication arm. Needs a call — see 7.1 |
| 2 | Whether to apply the S089/S090 correction or keep `CLAUDE.md` literal | Correction applied 2026-08-31; revertible |
| 3 | Delete the 4.4 GB of redundant zip + dead mirror | Awaiting user |
| 5 | **`CLAUDE.md`'s Expected Results section** states ~90-93 % subject-dependent as the reproduction target and 60-70 % as normal for LOSO. Both derive from the leaky literature, and 9.1-9.2 show the correct-protocol number is ~30 % with no decoding in it. As written the rule reads as "you failed to reproduce". | **Open — flagged to the user 2026-08-31, not yet decided.** Proposal: separate *published-protocol* targets from *correct-protocol* expectations. Not edited unilaterally: it is a research judgement |
| 4 | **What "reproduce the baseline" means**, given the reference headline rests on four stacked leakage paths. (i) Re-implement the leaky protocol to confirm ~93 %, then show it collapses under a correct split. (ii) Only ever run the correct protocol. | **RESOLVED 2026-08-31 to (i)**, user decision. We reproduce the leaky protocol as a documented artefact, then quantify how much each leakage path contributes. See 7.2 |

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

## 11. Changelog

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
