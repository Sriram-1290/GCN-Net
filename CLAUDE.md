# Project Rules — EEG Motor Imagery Decoding (GCNs-Net Improvement)

## Project Context

Building an improved graph convolutional network for EEG motor imagery decoding, based on **GCNs-Net** (Hou et al., IEEE TNNLS 35(6), 2024; online 2022).

**Base architecture:** Spectral GCN — Pearson correlation adjacency → normalized graph Laplacian → Chebyshev polynomial filters (K=2) → Graclus graph coarsening/pooling → softmax. 6 graph-conv layers + 6 pooling layers. 4-class output.

**Our contribution:** dynamic/learnable adjacency matrix + temporal attention, evaluated under Leave-One-Subject-Out (LOSO), plus scalp-topography interpretability.

**Primary dataset:** PhysioNet EEG Motor Movement/Imagery (109 subjects, 64ch, 160 Hz).
**Secondary (optional):** BCI Competition IV-2a (9 subjects, 22ch, 250 Hz).

**Hardware:** RTX 5060 8GB. Model is small (~350K params) — VRAM is NOT a constraint. Do not add mixed precision, gradient checkpointing, or memory optimizations unless explicitly asked.

- *Qualified 2026-09-05:* true of parameters, **not of activations for `graphbp`**. That
  architecture reshapes to `(B x t_out, M, F)` before the spatial stage, so activations
  scale with batch x time: 1.23 GB at batch 256 (14 % of the card) but an overflow to host
  memory at batch 1024, which collapses throughput 16x. **Batch 256 is near this
  architecture's ceiling, not an arbitrary default — do not raise it.** Throughput is flat
  from batch 256 to 384 (8,004 → 8,431 samples/s), so the GPU is already saturated and
  there is no speed left to win in the training loop; making the data GPU-resident measures
  1.0x. The cost that *can* be reduced is algorithmic — see §18 (2026-09-05).

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
  - *Corrected 2026-08-31:* this list previously read **S090**, which is verifiably clean (all 14 runs at 160 Hz, 30 annotations in every imagery run, profile identical to S091). The subject that actually carries a defect is **S089** (R03 is 181 s instead of 123 s). Evidence in `docs/project_documentation.md` §3.5. N = 103 either way.
  - Only **S088, S092, S100** are unusable on the imagery runs themselves (128 Hz *and* wrong trial counts). S089 and S106 are defective only in execution runs we never load; S104 loses 4 trials from R08. The 6-subject list is a **comparability convention, not a data-driven necessity** — describe it that way when defending the work.
- **Use imagery runs only.** PhysioNet contains both executed-movement and imagery runs. Imagery runs are 4, 6, 8, 10, 12, 14. Including movement runs inflates accuracy and invalidates comparison with the paper.
- **The class label is `f(run_number, T-code)` — never the T-code alone.** Runs 4/8/12: T1 = left fist, T2 = right fist. Runs 6/10/14: T1 = both fists, T2 = both feet. T0 = rest in all runs. Mapping `T1 -> class` without consulting the run number silently merges *left fist* with *both fists*. Assert on it.
- **Raw data lives at `data/raw/eegmmidb/S001 ... S109/`** — 109 subjects x 14 runs, all 3,052 files SHA256-verified against the bundled manifest. Do not re-download.
- **Cache preprocessed data to disk** (`.npy` / HDF5). Never re-run full 109-subject preprocessing on every experiment.
- Do not download or preprocess OpenBMI or High Gamma unless explicitly asked — not in current scope.

---

## Expected Results — Do Not "Fix" These

*Rewritten 2026-09-02, replacing predicted targets with measured values. The previous
version predicted ~90-93% subject-dependent and 60-70% LOSO; both came from the published
literature and both are contradicted by measurement. Full detail and CIs in
`docs/project_documentation.md` §9.8-9.10.*

**These are measurements, not targets.** The old numbers were goals to hit, which is
precisely the wrong framing — it invites tuning until the number appears. What follows is
what the pipeline produces. If a run disagrees, the first hypothesis is a bug in the run,
not a result worth reporting.

**Reference points — 103 subjects, 3 seeds, within-family accuracy unless stated:**

| | value | note |
|---|---|---|
| Published protocol (leaky), timepoint accuracy | **0.8975** [0.8794, 0.9147] | reproduces the paper's 0.8835 |
| Correct split, published formulation | **0.5090** [0.4984, 0.5199] | **at chance** — CI contains 0.5 |
| Correct split, 8-30 Hz + 64-tp window | 0.5470 [0.5291, 0.5667] | |
| + temporal attention | **0.6038** [0.5805, 0.6278] | best *published-formulation* arm; superseded by G1-G3 |
| CSP+LDA on identical splits | **0.6239** [0.5977, 0.6521] | ahead of every *graph* arm |
| LOSO, subject-independent | **0.6040** [0.5823, 0.6268] | 88/103 subjects above chance |
| A1: free spatial operator (no graph constraint) | **0.5566** | +0.0128 over fixed; only 16 % of the gap |
| A2: EEGNet, no graph at all | **0.6296** | indistinguishable from CSP (p = 0.49) |
| **A2: ShallowConvNet** | **0.6725** | beats CSP by 0.0485 |
| **G1: reordered architecture, graph constraint** | **0.6629** | **the first graph model to beat CSP (+0.0389)** |
| **G3: reordered + graph as prior** | **0.6737** | identical to ShallowConvNet (p = 0.99) |
| PTC: same pipeline, pre-training skipped | 0.6744 | **the control** for the row below |
| **PTG: + cross-subject pre-training** | **0.7301** | **+0.0557 over its control, 93/103 subjects** |
| PTC-shallow: ShallowConvNet, pre-training skipped | 0.6886 | **the control** for the row below |
| **PTG-shallow: ShallowConvNet + pre-training** | **0.7369** | **tied with PTG, not ahead of it (see below)** |
| PTD: PTG + domain-specific batch norm | 0.7224 | **−0.0077 against PTG — no gain; see below** |

- **The published formulation decodes nothing under a correct split.** 0.5090 against a
  chance level of 0.5. Its 4-class accuracy of 0.3097 looks above chance only because of
  the run-family artefact (§9.2). This is the project's central finding — do not treat it
  as a failure to reproduce and do not tune to move it.
- **LOSO is not worse than subject-dependent here — it is marginally better** (0.6040 vs
  0.6038). That reverses the old expectation. A LOSO fold trains on ~100x more data, which
  outweighs never having seen the test subject, and implies the subject-dependent protocol
  is data-starved. **4-class LOSO has not been measured**; the 0.6040 figure is binary
  within-family. Do not quote a 4-class LOSO number until one exists.
- **CORRECTED 2026-09-02 (A1/A2, §9.11). The previous claim here — "a linear method from
  2000 still wins" — is FALSE and must not be repeated.** It was true of every *graph* arm
  and was generalised too far. ShallowConvNet beats CSP+LDA by **+0.0485** [+0.0318,
  +0.0650] on identical folds, winning on 67/103 subjects, and EEGNet matches it. Deep
  learning works on this data; graph-convolutional deep learning does not.
- **The limitation is architectural, not sample size.** MOABB's ">150 trials per class"
  threshold suggested our ~45 might be the cause. A2 refutes that directly: graph-free CNNs
  on the *same ~45 trials and the same folds* reach and pass CSP.
- **CORRECTED 2026-09-03 (G arms, §9.13). "Graph-convolutional deep learning does not work
  on this data" is now FALSE.** Reordering the architecture — power *after* the spatial stage,
  a learned temporal filter, depth 1, no node pooling — lifts the graph model from 0.6038 to
  0.6629 (+0.0591, 80 % of subjects) and makes it beat CSP by +0.0389. The published graph
  *formulation* was never the problem; the surrounding architecture was.
- **But the graph contributes nothing either.** Hard constraint, free operator and soft prior
  span 0.0108 and are mutually indistinguishable. **In the wrong architecture, freeing the
  operator was significant (A1, +0.0128, p = 0.0046); in the right one it is not (+0.0065,
  p = 0.52).** The graph constraint only cost anything while the rest was broken.
- **CORRECTED 2026-09-04 (pre-training, §9.15). Cross-subject pre-training is the largest
  single intervention in the project**: +0.0557 over its own control on 90 % of subjects,
  and it lifts the weakest quartile twice as much as the strongest (+0.0720 vs +0.0346).
  §9.9's reading that the subject-dependent protocol is data-starved is now confirmed, not
  inferred.
- **NEVER compare PTG's 0.7206 to the published 0.650 ceiling.** That benchmark is
  within-session and sees only the target's own data; PTG additionally consumes ~92 other
  subjects. It is a different protocol with vastly more data, not a better pipeline. Every
  other arm *is* comparable to 0.650; PTG is not.
- **CONFIRMED 2026-09-06 (§9.17). The 0.73 is a data effect, not an architectural one, and
  this is now measured rather than assumed.** ShallowConvNet gains **+0.0483** from the same
  pre-training against graphbp's +0.0557; the difference of differences is **−0.0073**
  [−0.0204, +0.0056] on 47/103 subjects — a coin flip. Pre-training transfers at the same
  magnitude to an architecture sharing no component with ours. **The instruction never to
  write "our architecture reaches 0.73" is therefore permanent, not provisional.**
- **PTG-shallow (0.7369) and PTG (0.7301) are TIED — do not call either one the best.**
  The head-to-head is +0.0068 [−0.0022, +0.0155], below the 0.010 floor §9.16.3 fixed in
  advance and with a CI containing zero. Shallow wins L-vs-R (0.7387 vs 0.7206) and loses
  B-vs-F (0.7351 vs 0.7396). **Do not write "ShallowConvNet is the project's best model"**,
  and **never compare** PTG-shallow's 0.7387 to the 0.650 ceiling either — it is a
  different protocol with ~92 extra subjects, and the §9.15.4 prohibition covers every
  pre-trained arm.
- **CONFIRMED 2026-09-07 (§9.23). Domain-specific batch normalisation does not help.**
  PTD = 0.7224 against PTG's 0.7301, i.e. **−0.0077** [−0.0164, +0.0013], inside the 0.020
  MDD so reported as indistinguishable. But it wins on only **39/103** subjects (sign test
  p = 0.0176), so the direction is consistent even though the magnitude is not resolvable:
  DSBN is very slightly *harmful* here, most likely because ~1,665 windows per subject make
  per-domain statistics noisier than the pooled estimate. TSMNet's 2.4–3.9 % does not
  transfer. **Four challengers to the ~0.67→0.73 picture have now measured nothing**
  (§9.19 spatial priors, §9.21 window length, §9.23 normalisation, plus §9.17's null DiD).
- **MEASURED 2026-11-07 (§9.25, EXPLORATORY, 20 development subjects). Montage robustness is
  not a route to a clear win with this design.** The node-agnostic region-pooled graph model
  (M1) trails ShallowConvNet (M2, interpolation) on **every** cap: **−0.1117** on the full
  64-channel cap (Full guard fires), −0.0633 on IV-2a and −0.0940 on random 25 % (both
  primaries are measured *losses*, CIs exclude zero, M1 ahead on 3/20 and 1/20). All five
  pre-registered predictions are refuted. The Phase 2 gate is not met, so Phases 2-3 were
  not run. M1 loses *less* than M2 when channels vanish (post hoc: +0.0483 on IV-2a), but
  that is a smaller drop from a much lower starting point and **never makes M1 the better
  model on any cap**. Likely mechanism (exploratory, not proven): node-shared Chebyshev
  weights plus region-averaged band power remove electrode-specific spatial filtering, so
  M1 underfits. **This is the fifth challenger measured; do not re-run it with tuning.**
- **Never report a G-arm difference below 0.020 as an effect.** §9.12 fixed that rule before
  the runs; the design's minimum detectable difference is ~0.020 at 80 % power.
- **But the mechanism named in §9.7.1 is minor.** A1 lifted graph locality and node-shared
  weights and recovered only **16 %** of the fixed-graph-to-CSP gap (+0.0128, p = 0.0046).
  Real but small; ~84 % of the gap is elsewhere in the architecture. §9.7.1 was right about
  *where* and wrong about *what*, and has been rewritten.
- **A1 and A2 are post-hoc, not confirmatory.** Neither is on the pre-registered list
  (§10.3), and A1 tests a claim written after the confirmatory runs. Label them exploratory
  wherever they appear.
- **Never quote a 4-class accuracy without the family decomposition.** It has twice risen
  while the rise was entirely better run-family identification (§9.4, §9.6). Report
  within-family accuracy — the confound-free measure — alongside any 4-class figure.
- **Never quote a partial run.** One arm read 0.5774 at 35 of 100 folds and 0.5418
  completed. Check the fold count against its target before reporting anything.
- Do not tune hyperparameters, change splits, or alter the protocol to move any of these
  numbers. Report what is measured.
- A well-analyzed negative result is an acceptable outcome. It is also, here, the actual
  outcome for two of the three components tried — say so plainly rather than burying it.

---

## DO

- **Build end-to-end first, optimize later.** A rough pipeline (load → train → evaluate → print metric) working badly is more valuable than one perfect component. Get the full loop running before improving any part.
- **Keep `docs/project_documentation.md` current.** It is the living record of what was done, what was decided, and why. Update it as work proceeds — new findings, decisions, results — and add a dated changelog entry. Append; don't rewrite history.
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
- **Don't port the TensorFlow reference (`SuperBruceJia/EEG-DL`) faithfully.** Local checkout at `D:\projects\EEG-DL`. It is a fork of Defferrard's `cnn_graph` carrying **three leakage paths**, verified against the source and documented in `docs/project_documentation.md` §4.3–4.4: (1) each sample is a single timepoint and the split is `randperm` over timepoints, so the same trial lands in train and test; (2) BatchNorm is hardcoded to `training=True` in `b1relu`/`b2relu`/`fc`, so test batches normalise on their own statistics; (3) the adjacency `A = |P| - I` is computed over the whole pooled dataset before splitting. Its published numbers are not a trustworthy baseline.
  - It does apply **Common Average Reference** (`make_dataset.m` line 22 — that line is CAR, not dataset-wide mean centring) and **no filtering** (the bandpass/notch calls are commented out). Match both.
  - Use `b2relu` (bias `[1, M, F]`), which is what `main-GCN.py` configures — not `b1relu`.
  - Its four classes are **left fist / right fist / both feet / resting-state**, not the L/R/B/F the paper names: both-fists is discarded and the fourth class is random windows from the eyes-open baseline run.
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
