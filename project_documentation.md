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

### 4.4 Preprocessing in the reference (`make_dataset.m`)

Read directly from `Preprocess_EEG_Data/For-GCN-based-Models/make_dataset.m`. This
settles what preprocessing the baseline actually used, and it is less than expected:

- **No filtering of any kind** — no bandpass, no notch, no re-referencing.
- **Mean-centring computed over the entire dataset** (`Stack_Dataset - mean(Stack_Dataset, 1)`)
  *before* the split.
- **`randperm()` over all samples, then a 90/10 cut.** Since each row is 64 channels
  at a *single timepoint*, this is precisely the timepoint-level shuffled split that
  `CLAUDE.md` forbids: timepoints from the same trial land in both train and test.
- Output layout: columns 1–64 are channels, the final column is the label 0–3.
  Confirms four classes and confirms `Fin = 1`.

**Consequence — this needs stating plainly in the writeup.** The reference's headline
accuracy is produced by *three stacked leakage paths*: timepoint-level shuffled
splitting, full-dataset mean-centring, and BatchNorm running on test-batch statistics
(§4.3.2). Any of the three alone would inflate the number.

This creates a tension with `CLAUDE.md`'s stated reproduction target of ~90–93 %
subject-dependent accuracy. That figure is very likely **not reachable without the
leakage that produces it**. An honest trial-level subject-dependent result will
probably land lower, and that is a finding rather than a failure. **Open question for
the supervisor/team: does "reproduce the baseline" mean (i) re-implement the leaky
protocol to confirm we can hit ~93 % and then demonstrate it collapses under a correct
split, or (ii) only ever run the correct protocol and report the honest number?**
Option (i) is more work but makes a much stronger argument, since it quantifies the
leakage instead of asserting it. Not yet decided — see §6, decision 4.

### 4.5 The paper itself (arXiv:2006.08924 / IEEE TNNLS 2022)

Read directly, because the repo could not answer these. Four things are now settled:

**1. The four classes are L / R / B / F, with no rest.** "Four MI tasks are termed as L
(image left fist), R (image right fist), B (image both fists), and F (image both
feet)." The rest state (T0) is not a classification target. **This confirms scheme (a)**
and closes open decision 1.

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

**Total leakage paths in the reference: four.** Timepoint-level shuffled split (4.4),
whole-dataset mean-centring (4.4), BatchNorm on test-batch statistics (4.3.2), and
adjacency built from all data including test (here).

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
| `tests/` | 52 tests, all passing (labels, splits, graphs, normalisation) |

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

**Not yet built:** Graclus coarsening, the model, training loop, evaluation,
visualisation.

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

98.4 % of all electrode pairs carry a non-trivial edge. Because the pipeline applies
no filtering and no re-referencing (matching the reference), every channel shares a
large common-mode component, so everything correlates with everything. Spatial
structure *is* present and in the right direction -

```
|PCC|(C3, C1) 0.973   (C3, C5) 0.912   (C3, C4) 0.851   (C3, Oz) 0.718   (C3, Fp1) 0.638
mean neighbours 0.950  vs  mean distant 0.645
```

- but compressed into a narrow band. A near-complete graph gives the spectral
convolution little topology to exploit, which is a real weakness of the baseline and a
direct motivation for the **learnable adjacency** contribution. `threshold_adjacency()`
exists so sparsification can be ablated.

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

---

## 7. Open decisions

| # | Question | Status |
|---|---|---|
| 1 | **4-class definition.** (a) left fist / right fist / both fists / both feet, T0 discarded. (b) rest / left / right / feet, mixing T0 in. | **RESOLVED 2026-08-31 to (a).** The paper states the four tasks are L/R/B/F with no rest (4.5). The cache is already built under (a); no rebuild needed. Full comparison in 7.1 |
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
