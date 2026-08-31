# Project Rules — EEG Motor Imagery Decoding (GCNs-Net Improvement)

## Project Context

Building an improved graph convolutional network for EEG motor imagery decoding, based on **GCNs-Net** (Hou et al., IEEE TNNLS 2022).

**Base architecture:** Spectral GCN — Pearson correlation adjacency → normalized graph Laplacian → Chebyshev polynomial filters (K=2) → Graclus graph coarsening/pooling → softmax. 6 graph-conv layers + 6 pooling layers. 4-class output.

**Our contribution:** dynamic/learnable adjacency matrix + temporal attention, evaluated under Leave-One-Subject-Out (LOSO), plus scalp-topography interpretability.

**Primary dataset:** PhysioNet EEG Motor Movement/Imagery (109 subjects, 64ch, 160 Hz).
**Secondary (optional):** BCI Competition IV-2a (9 subjects, 22ch, 250 Hz).

**Hardware:** RTX 5060 8GB. Model is small (~350K params) — VRAM is NOT a constraint. Do not add mixed precision, gradient checkpointing, or memory optimizations unless explicitly asked.

**Stack:** Python, PyTorch, PyTorch Geometric, MNE-Python, NumPy/SciPy, scikit-learn, Matplotlib.

---

## CRITICAL — Data Leakage Rules

These are the highest-risk failure modes in this project. Violating any of them silently inflates accuracy and invalidates all results.

- **NEVER split at the timepoint level.** GCNs-Net treats each timepoint as a sample, so naive `train_test_split` on shuffled samples puts timepoints from the SAME trial in both train and test. Split at **trial level minimum**, **subject level for LOSO**.
- **NEVER compute the correlation/adjacency matrix using test data.** Build the graph from training split only. This applies to the static PCC baseline AND to any statistics used to initialize a learnable adjacency.
- **NEVER fit normalization statistics (mean/std, scalers) on the full dataset.** Fit on train, apply to test.
- **NEVER let the same subject appear in both train and test under LOSO.** The whole point of the protocol is unseen subjects.
- If accuracy jumps unexpectedly high, **suspect leakage first** — report it, don't celebrate it.

---

## Dataset Rules

- **Exclude corrupted subjects:** S088, S089, S092, S100, S104, S106 — inconsistent sampling rates / trial lengths. Drop them and log that they were dropped.
  - *Corrected 2026-08-31:* this list previously read **S090**, which is verifiably clean (all 14 runs at 160 Hz, 30 annotations in every imagery run, profile identical to S091). The subject that actually carries a defect is **S089** (R03 is 181 s instead of 123 s). Evidence in `project_documentation.md` §3.5. N = 103 either way.
  - Only **S088, S092, S100** are unusable on the imagery runs themselves (128 Hz *and* wrong trial counts). S089 and S106 are defective only in execution runs we never load; S104 loses 4 trials from R08. The 6-subject list is a **comparability convention, not a data-driven necessity** — describe it that way when defending the work.
- **Use imagery runs only.** PhysioNet contains both executed-movement and imagery runs. Imagery runs are 4, 6, 8, 10, 12, 14. Including movement runs inflates accuracy and invalidates comparison with the paper.
- **The class label is `f(run_number, T-code)` — never the T-code alone.** Runs 4/8/12: T1 = left fist, T2 = right fist. Runs 6/10/14: T1 = both fists, T2 = both feet. T0 = rest in all runs. Mapping `T1 -> class` without consulting the run number silently merges *left fist* with *both fists*. Assert on it.
- **Raw data lives at `data/raw/eegmmidb/S001 ... S109/`** — 109 subjects x 14 runs, all 3,052 files SHA256-verified against the bundled manifest. Do not re-download.
- **Cache preprocessed data to disk** (`.npy` / HDF5). Never re-run full 109-subject preprocessing on every experiment.
- Do not download or preprocess OpenBMI or High Gamma unless explicitly asked — not in current scope.

---

## Expected Results — Do Not "Fix" These

- **Subject-dependent accuracy ~90-93%** is the expected reproduction target for the baseline.
- **LOSO / subject-independent 4-class accuracy of 60-70% is NORMAL and expected**, not a bug. The gap between these two numbers is the entire point of the project.
- Do not tune hyperparameters, change splits, or alter the protocol to make LOSO numbers look closer to subject-dependent numbers. Report what is measured.
- A well-analyzed negative result (improvement doesn't beat baseline) is an acceptable outcome. Do not manufacture gains.

---

## DO

- **Build end-to-end first, optimize later.** A rough pipeline (load → train → evaluate → print metric) working badly is more valuable than one perfect component. Get the full loop running before improving any part.
- **Keep `project_documentation.md` current.** It is the living record of what was done, what was decided, and why. Update it as work proceeds — new findings, decisions, results — and add a dated changelog entry. Append; don't rewrite history.
- **Log every experiment** to a persistent CSV/JSONL: timestamp, git commit, config/hyperparameters, split type, seed, accuracy, notes. One row per run.
- **Set and log random seeds** for torch, numpy, and Python random. Results must be reproducible.
- **Checkpoint models** during training. Save best-by-validation, not just final epoch.
- **Compare baseline and improved model under identical conditions** — same splits, same seeds, same preprocessing, same evaluation code. Fair comparison is the core requirement.
- **Write modular code**: separate `data/`, `models/`, `evaluation/`, `visualization/`. Three people are working in parallel on different components.
- **Agree on and document data formats/interfaces** between modules before implementing them.
- **Report statistical significance** (paired tests across subjects/folds) plus confidence intervals — not just raw accuracy means.
- **Include confusion matrices and per-class accuracy** in evaluation output, not just aggregate accuracy.
- **Verify GPU is actually being used** — RTX 5060 is Blackwell (sm_120); confirm PyTorch build supports it (CUDA 12.8+).

## DON'T

- **Don't add architectural complexity without an ablation.** Every new component must be justified by a with/without comparison.
- **Don't stack multiple improvements at once.** Change one thing, measure, then change the next. Simultaneous changes make attribution impossible.
- **Don't silently change preprocessing between experiments.** If preprocessing changes, all prior results are invalid — re-run the baseline too.
- **Don't hardcode paths, hyperparameters, or subject lists** in the middle of scripts. Use a config file.
- **Don't use `localStorage`-style ad-hoc state or global mutable config** shared across modules — pass explicit arguments.
- **Don't report the paper's 93% as our result.** Ours is whatever we reproduce, and it may differ.
- **Don't skip the baseline reproduction** and jump straight to the improved model. Without a verified baseline, no comparison is meaningful.
- **Don't optimize for VRAM/speed.** The model is tiny; complexity here is wasted effort.
- **Don't add dependencies casually.** Stick to the declared stack unless there's a clear need.
- **Don't port the TensorFlow reference (`SuperBruceJia/EEG-DL`) faithfully.** It is a fork of Defferrard's `cnn_graph` with three defects documented in `project_documentation.md` §4.3: samples are single timepoints (no temporal axis at all), BatchNorm is hardcoded to `training=True` so test batches leak their own statistics, and the Pearson-correlation adjacency construction is simply absent from the repo. Its published numbers are not a trustworthy baseline.
- **Don't rewrite working code for style.** Prefer minimal, targeted edits.

---

## Code Quality Expectations

- Prefer clear, readable code over clever code — this is being submitted and defended academically.
- Add docstrings explaining *why*, especially for graph construction, Laplacian normalization, and coarsening logic (these are non-obvious).
- Keep functions small and testable. Data pipeline functions should be unit-testable with tiny synthetic inputs.
- When implementing something from the paper, cite the relevant equation/section in a comment.
- Fail loudly on unexpected data shapes — silent broadcasting bugs are common with EEG tensors. Assert shapes explicitly.

## Communication Preferences

- When something is ambiguous in the paper, **say so and ask** rather than guessing silently.
- When a result looks suspicious, **flag it** rather than moving on.
- Explain non-obvious design decisions briefly — this code needs to be defended in a viva.
