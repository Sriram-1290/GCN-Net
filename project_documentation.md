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
| Baseline reproduction (subject-dependent) | Not started | — |
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
| `tests/` | 68 tests, all passing (labels, splits, graphs, normalisation, coarsening) |

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

**Not yet built:** the model, training loop, evaluation, visualisation.

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

---

## 7. Open decisions

| # | Question | Status |
|---|---|---|
| 1 | **4-class definition.** (a) left/right fist + both fists + both feet, all MI. (b) a rest class substituted for one MI class. | **REOPENED 2026-08-31.** The paper *names* L/R/B/F, but the shipped code discards both-fists and substitutes resting-state windows from the baseline run (4.5). Paper and code disagree. Current position: keep (a) for the main pipeline, use the code's scheme only in the leaky replication arm. Needs a call — see 7.1 |
| 2 | Whether to apply the S089/S090 correction or keep `CLAUDE.md` literal | Correction applied 2026-08-31; revertible |
| 3 | Delete the 4.4 GB of redundant zip + dead mirror | Awaiting user |
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

## 8. Changelog

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
