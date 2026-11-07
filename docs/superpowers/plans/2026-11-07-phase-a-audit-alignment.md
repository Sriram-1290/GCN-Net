# Phase A — Inflation Audit and Euclidean Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the run-family confound, the GCNs-Net leakage paths one at a time, early-cue dependence and 4-class subject-independent decoding, and test Euclidean alignment pre-registered, all on the existing EEGMMIDB cache.

**Architecture:** Small, tested additions:
- an EA module (`data/alignment.py`);
- run-disjoint folds and a window-start offset in the pipeline;
- a rest-period (T0) cache;
- one confound-audit script (A3/A4) and one leak-ablation script (A5);
- zero-shot scoring and EA options in `scripts/run_pretrain_finetune.py`, plus a tangent-space transfer script (A1/A6);
- one analysis script.

Everything is pre-registered before any run, and the long runs are resumable background jobs.

**Tech Stack:** Python 3.14, PyTorch 2.12, NumPy, scikit-learn, MNE 1.12.1, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-07-phase-a-audit-alignment-design.md`

## Global Constraints

- Python `C:/Python314/python.exe`; tests `C:/Python314/python.exe -m pytest`; the 417 existing tests keep passing.
- **No new dependencies, no downloads.** Tangent space, CSP and EA are in-house; `sklearn` is already in use.
- Existing referenced cache: `configs/bandpass_8_30.yaml` (8–30 Hz, common average reference, 4.0 s epochs at 160 Hz). Windows 64 samples, stride 16.
- **Subjects:** the 103 non-excluded subjects. Groups are always `assign_groups(all 103 subjects, 10)`, index mod 10.
- **Leakage:** graphs, standardisers and EA references come from training data only. The sole exception is the explicitly tagged transductive EA variant. Run-disjoint folds come from run numbers only.
- **EA:** per-subject reference R = mean trial covariance; whitening by R<sup>-1/2</sup> via a pseudo-inverse square root, keeping eigenvalues above relative tolerance **1e-8** of the largest; EA is applied after the band-pass.
- **Pseudo-online EA reference:** the subject's earliest run of the task's family (run **4** for the fist family, run **6** for the fists-vs-feet family). That run is excluded from scoring in **both** arms of every A6 subject-independent comparison.
- **Decision rule (A6):** an effect is at least **0.020** with a 95 % paired bootstrap CI excluding zero; Holm correction over the three primary contrasts; per-subject win counts reported.
- **A2 settings** copy the logged `A2-shallow-binary-*` arm: `--model shallow`, `configs/bandpass_8_30.yaml`, window 64, stride 16, 5 folds, 50 epochs, lr 1e-3, batch 256, seed 42. Early-window exclusion threshold: **160 samples (1.0 s)**.
- Default behaviour of every existing script and cache key is unchanged; new options are opt-in.
- Batch 256 for `graphbp` stays. Long runs resume with `--resume`, retry on host-memory failure, and release memory between groups.
- New tags start with `PA-` (Phase A).

## Review Focus

1. **Pseudo-inverse correctness on common-average-referenced data.** A rank-63 covariance must not blow up the null direction, and the aligned data's covariance must be identity within the rank-63 subspace. Pinned in Task 1.
2. **Calibration run leaking into scoring.** In the A6 subject-independent arms, trials from run 4 or 6 must never be scored, in either arm, or the arms aren't paired. Pinned in Task 6.
3. **Run-disjoint folds that aren't run-disjoint.** Every fold's test runs must be absent from its training runs, and each fold must hold both classes. Pinned in Task 2.
4. **Rest cache mislabelled by run.** Each T0 epoch must carry the run it came from, and runs 4/8/12 versus 6/10/14 must map to families 0/1. Pinned in Task 3.
5. **Leak toggles not independent.** With all three toggles on, the existing reference protocol must be reproduced; with all off, the correct protocol. Pinned in Task 5.

---

### Task 1: Euclidean alignment module

**Files:**
- Create: `data/alignment.py`
- Test: `tests/test_alignment.py`

**Interfaces:**
- Produces:
  - `trial_covariances(X: np.ndarray) -> np.ndarray`: `(n, C, T)` → `(n, C, C)` float64, `X Xᵀ / T` per trial.
  - `euclidean_reference(X: np.ndarray) -> np.ndarray`: mean trial covariance, `(C, C)`.
  - `inv_sqrt_psd(R: np.ndarray, rtol: float = 1e-8) -> np.ndarray`: pseudo-inverse square root. Eigen-decompose; keep eigenvalues > `rtol * max`; return `V diag(λ^-1/2) Vᵀ` over the kept ones.
  - `align(X: np.ndarray, W: np.ndarray) -> np.ndarray`: `W @ X` per trial, float32, same shape.
  - `EAReference`, a frozen dataclass with `W: np.ndarray`, `rank: int`, `n_trials: int`, built by `fit_ea(X_train: np.ndarray, rtol=1e-8) -> EAReference`.

- [ ] **Step 1: Write the failing tests**
  - `test_full_rank_whitens_to_identity`: random full-rank `(50, 8, 200)`; after `align(X, fit_ea(X).W)`, `euclidean_reference` ≈ I (atol 1e-5).
  - `test_car_rank_deficient_no_blowup` (Review Focus 1): common-average-referenced data (8 channels, rank 7). `fit_ea(...).rank == 7`; the aligned covariance has eigenvalues ≈ 1 (seven of them) and ≈ 0 (one), with no value above 1 + 1e-4. The aligned data's maximum absolute value stays below 1e3 × the input's.
  - `test_reference_uses_only_given_trials`: changing trials not passed to `fit_ea` leaves `W` unchanged.
  - `test_rejects_bad_shapes`: a 2-D `X` raises `ValueError`; a non-square `R` raises `ValueError`.
- [ ] **Step 2: Run them.** Expected: FAIL, module not found.
- [ ] **Step 3: Implement.** Use `np.linalg.eigh` on the symmetrised matrix. The module docstring explains rank-aware whitening and why not a plain inverse (research report §9.4).
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit:** `Add Euclidean alignment with a rank-aware inverse square root`.

### Task 2: Run-disjoint folds and a window-start offset

**Files:**
- Modify: `data/splits.py`, `evaluation/pipeline.py`, `scripts/run_subject_dependent.py`
- Test: `tests/test_splits.py`, `tests/test_evaluation.py`

**Interfaces:**
- Produces:
  - `run_disjoint_folds(runs: np.ndarray, y: np.ndarray, fold_runs: list[tuple[int, ...]]) -> list[TrialSplit]`. One fold per entry of `fold_runs`; the test set is trials whose run is in that tuple, the training set is all others. It raises `ValueError` if any fold's training or test set lacks a class present in `y`.
  - Standard run-pair folds as a constant, `RUN_PAIR_FOLDS = [(4, 6), (8, 10), (12, 14)]`.
  - `to_window_samples(..., start: int = 0)` and `prepare_fold(..., window_start: int = 0)`: windows begin at `start` instead of 0. The default 0 leaves behaviour unchanged.
  - `run_subject_dependent.py --window-start N` (default 0), passed through to `prepare_fold`.

- [ ] **Step 1: Write the failing tests**
  - `test_run_disjoint_folds_are_disjoint` (Review Focus 3): for synthetic runs/labels, every fold's test runs ∩ train runs = ∅, and the folds cover every trial exactly once.
  - `test_run_disjoint_folds_reject_missing_class`: a fold whose test set has one class raises `ValueError`.
  - `test_window_start_offsets_windows`: `to_window_samples(X, y, 64, 16, start=160)` gives a first window at sample 160 and a window count of `len(range(160, T-64+1, 16))`.
  - `test_window_start_default_unchanged`: `start=0` equals the current output exactly.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them plus the full suite.** Expected: PASS.
- [ ] **Step 5: Commit:** `Add run-disjoint folds and a window-start offset`.

### Task 3: Rest-period (T0) cache

**Files:**
- Modify: `data/edf_loader.py`
- Create: `scripts/build_rest_cache.py`, `configs/rest_bp830.yaml`, `configs/rest_car.yaml`
- Test: `tests/test_rest_cache.py`

**Interfaces:**
- Produces:
  - `epoch_rest(raw, run: int, cfg: Config) -> tuple[np.ndarray, np.ndarray]`: T0 epochs `(n, 64, 640)` in µV, and onsets. It raises `RunValidationError` on bad counts, mirroring `epoch_run`.
  - Cache files: `data/cache_rest_bp830/S###.npz` and `data/cache_rest_car/S###.npz`, holding `X`, `run`, `onset` and `ch_names`; `family = 0` for runs 4/8/12 and `1` for runs 6/10/14.
  - Configs: `rest_bp830.yaml` is `bandpass_8_30.yaml` with `cache_dir: data/cache_rest_bp830`. `rest_car.yaml` has no band-pass, common average reference only, with `cache_dir: data/cache_rest_car`.

- [ ] **Step 1: Write the failing tests**
  - `test_rest_family_mapping` (Review Focus 4): runs 4/8/12 give family 0 and runs 6/10/14 give family 1; any other run raises `ValueError`.
  - `test_epoch_rest_on_synthetic_raw`: an MNE `RawArray` with 15 T0 and 15 T1/T2 annotations gives 15 rest epochs of shape `(15, C, 640)`, each starting at its T0 onset.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** The build script uses the same subject list and exclusions as `build_cache.py`, and writes a manifest.
- [ ] **Step 4: Run the tests, then build both caches.** Expected: 103 subjects each; about 90 rest epochs per subject (15 per run × 6). Print the per-subject count range.
- [ ] **Step 5: Commit** (the caches are git-ignored): `Add a rest-period cache for the label-free run-family test`.

### Task 4: Confound audit script (A3 and A4)

**Files:**
- Create: `scripts/run_confound_audit.py`
- Test: `tests/test_confound_audit.py`

**Interfaces:**
- Consumes: `run_disjoint_folds`, `RUN_PAIR_FOLDS` (Task 2); the rest caches (Task 3); the tangent-space helpers in `scripts/run_tangent_space.py` (move the reused helpers into `data/riemann.py` if importing a script is awkward, keeping the existing script working); `models.eegnet.build_baseline("shallow", ...)`; `evaluation.trainer`.
- Produces: the CLI `run_confound_audit.py --experiment {rest-family,cross-family} --model {tgsp,shallow} --split {trial,run} --config <cfg> --tag <tag> [--resume]`. It logs one row per (subject, fold) with `accuracy`, `roc_auc`, `split`, `experiment`, `task` and the confusion matrix.
  - `rest-family`: labels are the rest epochs' family. Trial splits use 5-fold stratified; run splits use folds holding out one run of each family, i.e. `RUN_PAIR_FOLDS`.
  - `cross-family`: tasks are `rh_vs_feet` (class 1 from runs 4/8/12 vs class 3 from runs 6/10/14), plus the controls `lr` (0 vs 1) and `bf` (2 vs 3). For `lr` and `bf`, run splits hold out one run of that family per fold (3 folds); for `rh_vs_feet`, they use `RUN_PAIR_FOLDS`.
  - ShallowConvNet training uses the A2 settings (Global Constraints); the standardiser is fitted on training trials.

- [ ] **Step 1: Write the failing tests**
  - `test_rh_vs_feet_task_draws_from_both_families`: synthetic trials give class 1 only from runs 4/8/12 and class 3 only from runs 6/10/14.
  - `test_run_split_never_tests_on_training_runs`: a spy on the fold builder confirms test runs ∉ train runs for every fold, for both experiments.
  - `test_auc_and_accuracy_logged`: a synthetic separable problem with the tangent-space model logs accuracy > 0.9 and `roc_auc` > 0.9 for every fold row.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and a 2-subject smoke run per experiment, logging to `runs/check.jsonl`; delete it afterwards.**
- [ ] **Step 5: Commit:** `Add the run-family confound audit (rest decoding and cross-family task)`.

### Task 5: Leak-ablation script (A5)

**Files:**
- Create: `scripts/run_leak_ablation.py`
- Modify (only to expose the parts): `evaluation/pipeline.py`
- Test: `tests/test_leak_ablation.py`

**Interfaces:**
- Produces: `run_leak_ablation.py --leaks {none,split,bn,graph,all} --config configs/default.yaml --tag <tag> [--resume]`.
  - The reference formulation is used: window 1, the reference GCNs-Net, 4-class training, `configs/default.yaml` with no band-pass, seed 42.
  - **`split`:** random timepoint holdout (`leaky_timepoint_holdout`, test fraction 0.1) instead of trial-level 5-fold.
  - **`bn`:** BatchNorm in training mode at test time (`always_train_bn=True`).
  - **`graph`:** adjacency and standardiser fitted on pooled train and test data.
  - Each row logs timepoint-level and trial-level confusion matrices, so the 4-class and within-family decomposition can be computed for every arm.

- [ ] **Step 1: Write the failing tests**
  - `test_toggles_are_independent` (Review Focus 5): on synthetic data, `bn` alone changes only the BatchNorm mode, `graph` alone only the graph and standardiser fit source, and `split` alone only the split.
  - `test_all_matches_reference_path`: `--leaks all` builds the fold with the same settings as `prepare_leaky_fold` plus `always_train_bn`.
  - `test_none_matches_correct_path`: `--leaks none` builds the fold exactly as the correct protocol does.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and a 3-subject smoke run of `all`.** Its timepoint accuracy should be near the logged leaky-replication values for those subjects (within 0.05). Log to `runs/check.jsonl` and delete it afterwards.
- [ ] **Step 5: Commit:** `Add a one-leak-at-a-time ablation of the GCNs-Net reference protocol`.

### Task 6: Zero-shot scoring, EA and tangent-space transfer (A1 and A6)

**Files:**
- Modify: `scripts/run_pretrain_finetune.py`
- Create: `scripts/run_tangent_transfer.py`
- Test: `tests/test_pretrain_finetune.py`, `tests/test_tangent_transfer.py`

**Interfaces:**
- Consumes: `fit_ea`, `align` (Task 1); `assign_groups`, `plan_groups`, `source_cache_key`, `--source-tag` (existing).
- Produces:
  - **`run_pretrain_finetune.py --zero-shot`:** for each group, score that group's targets with the group's source model, without fine-tuning; log one row per target, tag `<tag>`. With `--calibration-run-excluded`, trials from run 4 (fist family) or run 6 (feet family) are never scored (Review Focus 2). With `--classes` omitted, the task is 4-class and the confusion matrix is logged (A1).
  - **`--ea {none,pseudo-online,offline}`** (default `none`):
    - Source subjects: each aligned with `fit_ea` on its own trials.
    - Targets, pseudo-online: reference from the calibration run (requires `--calibration-run-excluded` under `--zero-shot`).
    - Targets, offline: reference from all of the target's trials; tag rows `ea=offline`, transductive.
    - Fine-tuning (no `--zero-shot`): the target reference comes from the training fold only, and the fold's test trials are aligned with it.
  - `source_cache_key` adds `"ea"` only when not `none`, so existing keys are unchanged.
  - **`run_tangent_transfer.py --ea {none,pseudo-online} --classes A B --tag <tag> [--resume]`:** for each group, compute tangent-space features at the Riemannian mean of the pooled source covariances, fit `LogisticRegression` on the sources, and score the group's targets with the calibration run excluded.
- [ ] **Step 1: Write the failing tests**
  - `test_calibration_run_never_scored` (Review Focus 2): a spy on the scoring function sees no trial from run 4 (fist family) or run 6 (feet family), with and without EA.
  - `test_zero_shot_uses_cached_source_without_finetune`: with a cached state dict, no optimiser step happens and the logged row has `finetune_epochs == 0`.
  - `test_ea_reference_sources`: in fine-tune mode, `fit_ea` receives only training-fold trials; in pseudo-online mode, only calibration-run trials.
  - `test_ea_absent_keeps_cache_key`: `test_shallow_cache_key_unchanged` still passes, and an `ea` value changes the key.
  - `test_tangent_transfer_separable`: on synthetic two-subject data, transfer accuracy > 0.9.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests, the full suite, and a zero-shot smoke run on the cached PTG-shallow model for one group, logging to `runs/check.jsonl`; delete it afterwards.**
- [ ] **Step 5: Commit:** `Add zero-shot scoring, Euclidean alignment and tangent-space transfer`.

### Task 7: Pre-registration (before any PA- run)

**Files:**
- Modify: `docs/project_documentation.md` (new §9.26, plus a changelog entry)

- [ ] **Step 1: Write §9.26 "Phase A — pre-registration".** It contains:
  - time and commit;
  - experiments A1–A6 with their exact arms, tags, splits and metrics (from the spec and this plan);
  - the A6 predictions, verbatim from spec §3.6;
  - the decision rule;
  - the statement that the development/confirmation split is not used and why (spec §3).
- [ ] **Step 2: Confirm `grep -c '"tag": "PA-' runs/experiments.jsonl` prints 0, run `tests/test_documentation_consistency.py`, and commit:** `Pre-register Phase A`.

### Task 8: Analysis script

**Files:**
- Create: `scripts/phase_a_results.py`
- Test: `tests/test_analysis_scripts.py` (add tests)

**Interfaces:**
- Consumes: `evaluation/metrics.py` (`paired_bootstrap_ci`, `bootstrap_p_value`, `holm_bonferroni`); `scripts/run_family_analysis.decompose`.
- Produces: `phase_a_results.py {a1,a2,a3,a4,a5,a6}` prints a Markdown table per experiment.
  - Every experiment refuses partial runs: fewer than 5 × 103 rows, or the experiment's own fold count, per tag.
  - **A6:** applies the decision rule with Holm over the three primary contrasts.
  - **A4:** computes the inflation estimate (AUC<sub>trial</sub> − AUC<sub>run</sub> on `rh_vs_feet`, minus the same difference on the within-family tasks), with a paired bootstrap CI.
  - **A1:** reports 4-class, family and within-family accuracy per arm.
- [ ] **Step 1: Write failing tests on synthetic rows:**
  - A6 decides `effect` for a unanimous +0.03 and `no effect` for +0.01;
  - A4's inflation equals the constructed value;
  - A1's decomposition of a known confusion matrix matches `decompose`;
  - a missing row makes each experiment refuse.
- [ ] **Step 2: Implement and pass. Commit:** `Add the Phase A analysis script`.

### Task 9: Runs and write-up

- [ ] **Step 1: Launch, as one sequential, retrying, resumable background job** (the controller runs this; host memory permits one arm at a time):
  1. A5 (five toggles);
  2. A2 (`--window-start 160`, both problems);
  3. A3 (rest-family: two models × two splits × two caches);
  4. A4 (cross-family and the two controls: two models × two splits);
  5. A6 subject-independent ShallowConvNet: zero-shot without EA on cached PTG-shallow, and with EA after new EA source models;
  6. A6 tangent space with and without EA;
  7. A6 fine-tuned ShallowConvNet with EA;
  8. A1 (4-class source models, zero-shot) and tangent-space 4-class transfer.
- [ ] **Step 2: When each experiment is complete, run `phase_a_results.py`, and record §9.27 "Phase A — results"** (EXPLORATORY for A1–A5 as measurements; A6 as pre-registered), with the prediction scorecard. Add a changelog entry and update `CLAUDE.md`'s Expected Results. Commit the log and the docs together.
- [ ] **Step 3: Report to the user.**
