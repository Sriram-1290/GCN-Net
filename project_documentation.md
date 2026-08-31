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
| Data loader | Not started | — |
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

**Required, not yet installed** (all confirmed installable on Python 3.14 via
`pip install --dry-run`): `mne 1.12.1`, `h5py 3.16.0`, `torch_geometric 2.8.0`.
PyG 2.8 is pure-Python, so no compiled `torch-scatter` wheel is needed.

```bash
pip install mne h5py torch_geometric
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

2. **BatchNorm is hardcoded to training mode.** Both `b1relu` and `fc` call
   `tf.layers.batch_normalization(..., training=True)` with no train/eval switch, so at
   test time it normalises using *test batch* statistics. With `batch_size=1024` of
   shuffled timepoints that is a second, independent leakage path stacked on the first.
   In PyTorch this is just `model.eval()` — but it means the repo's published numbers
   are not a trustworthy baseline.

3. **The adjacency construction is absent.** `main-GCN.py` only does
   `pd.read_csv('Adjacency_Matrix.csv')`. The `adjacency()` in `graph.py` is
   Defferrard's kNN-Gaussian builder for *grid* graphs, not the Pearson correlation the
   paper describes, and the README documents nothing. **The PCC to adjacency step — the
   paper's core claim and the most leakage-prone operation in the pipeline — has no
   reference implementation.** We implement it from the paper text alone. This is the
   highest-risk item in the reproduction.

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
configs/        yaml — subjects, runs, class map, seeds, hyperparameters
data/           edf_loader, epoching, caching (npy/h5), splits (trial-level, LOSO)
graphs/         pcc_adjacency, laplacian, coarsening (Graclus port)
models/         cheb_conv, gcns_net baseline, dynamic_adj + temporal_attn
evaluation/     train loop, LOSO runner, metrics, experiment log (jsonl)
visualization/  topomaps, confusion matrices, learned-adjacency plots
```

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

## 6. Open decisions

| # | Question | Status |
|---|---|---|
| 1 | **4-class definition.** (a) left fist / right fist / both fists / both feet, T0 discarded, all 6 imagery runs, ~90 trials/subject. (b) rest / left / right / feet, mixing T0 in. `M=[4]` confirms four classes but not which. | **Proceeding with (a)** as the config default; not yet confirmed by user |
| 2 | Whether to apply the S089/S090 correction or keep `CLAUDE.md` literal | Correction applied 2026-08-31; revertible |
| 3 | Delete the 4.4 GB of redundant zip + dead mirror | Awaiting user |

---

## 7. Changelog

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
