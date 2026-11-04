# Montage Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a graph-convolution model trained on the full 64-electrode cap degrades more gracefully than ShallowConvNet when electrodes are missing or the cap is smaller, under a decision rule fixed before any run.

**Architecture:** One new no-reference data cache, from which every test condition is re-referenced over the electrodes actually present. A small `data/montage.py` defines regions, montages and seeded channel patterns. `graphs/subset.py` derives the graph and standardiser for any electrode subset exactly from per-subject source moments. `data/interpolate.py` gives ShallowConvNet its spherical-spline repair. `GraphBandPowerNet` gains a region-pooled readout whose graph can be swapped at test time. `evaluation/robustness.py` scores every condition after each fine-tuning fold, and `run_pretrain_finetune.py` gains `--targets`, `--source-tag`, `--readout` and `--robustness`.

**Tech Stack:** Python 3.14, PyTorch 2.12, NumPy, MNE 1.12.1 (standard_1005 montage, `mne.channels.interpolation._make_interpolation_matrix`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-04-montage-robustness-design.md`

## Global Constraints

- Python: `C:/Python314/python.exe`. Tests: `C:/Python314/python.exe -m pytest`. The existing 337 tests must keep passing after every task.
- No hyperparameter changes to either model: window 64, stride 16, batch 256, pre-train 30 epochs at 1e-3, fine-tune 30 epochs at 1e-4, 10 groups, 5 folds, seed 42, `lmax` 2.0, Chebyshev order K = 2, `n_temporal` = `n_spatial` = 40, dropout 0.5.
- Batch size stays 256 (`CLAUDE.md`: near the `graphbp` activation ceiling).
- Leakage: graph and standardiser come from **source subjects only**. No target data, training or test, ever enters a graph, a standardiser or a channel pattern.
- Development subjects: S001–S020. Confirmation subjects: the 83 others. Groups are always assigned over all 103 subjects (`index % 10`).
- Regions: 9 = {left, midline, right} × {frontal, central, posterior}. Column by suffix (`z` → midline, odd → left, even → right). Row by prefix: Fp, AF, F, FC, FT → frontal; C, T, CP, TP → central; P, PO, O, I → posterior.
- Conditions: Full; IV-2a (22 channels, Primary 1); Random 25 % (Primary 2); 10-20 (19 channels); Random 10 %; Random 50 %; Clustered 8; Clustered 16. Ten draws per (subject, fold) for random and clustered conditions; any draw that empties a region is rejected and redrawn.
- IV-2a channels: Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz.
- 10-20 channels: Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2.
- Decision rule (spec §6): margin ≥ 0.020, 95 % bootstrap CI excluding zero, Holm-corrected one-sided *p* < 0.05 over the two primaries, M1 ahead on a majority of subjects; full-montage guard: if M1 trails M2 on Full by more than 0.020, report a trade-off, not a win.
- Every new run's tag starts with `ROB`; condition rows use `"<tag>.<condition>"`.

## Review Focus

1. **Shallow cache-key drift.** Adding `--readout` must not change ShallowConvNet's source-cache key, or the five hours of cached pre-training are silently re-run with a different model. Pinned in Task 6.
2. **Node order versus region map.** If a subset's basis and its region vector are built in different channel orders, the model reads one electrode's power as another region's, with no error. Pinned in Task 4.
3. **Graph left swapped after a condition.** If the Full condition is scored after a subset condition without restoring the full graph, the Full number is silently wrong. Pinned in Task 7.
4. **Partial resume.** A fold whose clean row was logged but whose condition rows were not, because of a power cut mid-evaluation (this has happened three times), must be re-run, not skipped. Pinned in Task 7.
5. **Two caches out of step.** The no-reference cache must hold the same trials in the same order, with the same labels and channel names, as the referenced cache, or the wrong trials are scored. Pinned in Task 1.

---

### Task 1: No-reference cache and common average

**Files:**
- Create: `configs/bandpass_8_30_noref.yaml`
- Create: `data/montage.py`
- Create: `scripts/check_noref_cache.py`
- Test: `tests/test_montage.py`

**Interfaces:**
- Produces: `data.montage.common_average(X: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray`, which takes `X` of shape `(trials, channels, time)` and a boolean `keep` of length `channels` (None means all), and returns `(trials, keep.sum(), time)` float32, re-referenced to the mean of the kept channels.
- Produces: the cache `data/cache_bp830_noref/`, with the same subjects, trial order, labels and `ch_names` as `data/cache_bp830/`.

- [ ] **Step 1: Write the failing tests** in `tests/test_montage.py`:
  - `test_common_average_all_channels_has_zero_channel_mean`: on random `(3, 8, 50)`, `common_average(X).mean(axis=1)` is all zero within 1e-6.
  - `test_common_average_subset_uses_only_kept_channels`: with `keep` selecting channels 0–3 of 8, the output equals `X[:, :4] - X[:, :4].mean(axis=1, keepdims=True)`, and changing `X[:, 4:]` does not change it.
  - `test_common_average_rejects_bad_shapes`: a 2-D `X`, or a `keep` of the wrong length, raises `ValueError`.
- [ ] **Step 2: Run** `C:/Python314/python.exe -m pytest tests/test_montage.py -v`. Expected: FAIL, module not found.
- [ ] **Step 3: Implement `common_average`** in a new `data/montage.py`, whose module docstring explains why every condition is re-referenced (spec §4.1).
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Create `configs/bandpass_8_30_noref.yaml`**, a copy of `configs/bandpass_8_30.yaml` with `cache_dir: data/cache_bp830_noref` and `rereference: null`, and a header comment citing spec §4.1.
- [ ] **Step 6: Build the cache:** `C:/Python314/python.exe scripts/build_cache.py --config configs/bandpass_8_30_noref.yaml`. Expected: 103 subjects written and the 6 exclusions logged.
- [ ] **Step 7: Write `scripts/check_noref_cache.py`.** For every subject, load both caches. Assert identical `ch_names`, identical `y`, and an identical trial count. Assert that `common_average(noref.X)` matches the referenced `X` with a max absolute difference no greater than `1e-3 × median |X|`. Print the worst subject. Exit non-zero on any failure.
- [ ] **Step 8: Run it.** `C:/Python314/python.exe scripts/check_noref_cache.py`. Expected: `103/103 subjects match`.
- [ ] **Step 9: Commit** the config, the module, the test and the script (`data/cache_bp830_noref/` is git-ignored by `data/cache*/`). Message: `Add a no-reference cache so every test condition can be re-referenced`.

### Task 2: Regions, montages and channel patterns

**Files:**
- Modify: `data/montage.py`
- Test: `tests/test_montage.py`

**Interfaces:**
- Consumes: `common_average` (Task 1).
- Produces:
  - `REGION_NAMES: tuple[str, ...]`, nine names in the order `left-frontal, midline-frontal, right-frontal, left-central, midline-central, right-central, left-posterior, midline-posterior, right-posterior`.
  - `region_of(name: str) -> int`, raising `ValueError` for an unrecognised name.
  - `region_vector(ch_names: list[str]) -> np.ndarray`, an int64 array of length `len(ch_names)`.
  - `IV2A_CHANNELS`, `TEN_TWENTY_CHANNELS: tuple[str, ...]`, exactly as in Global Constraints.
  - `positions_3d(ch_names) -> np.ndarray`, the `(C, 3)` standard_1005 positions, matched case-insensitively.
  - `Condition`, a frozen dataclass with `name: str`, `kind: Literal["full", "montage", "random", "cluster"]` and `param: float | int | tuple[str, ...] | None`.
  - `CONDITIONS: tuple[Condition, ...]` named `full, iv2a, random25, tentwenty, random10, random50, cluster8, cluster16`, and `PRIMARY = ("iv2a", "random25")`.
  - `available_masks(cond: Condition, ch_names: list[str], subject: str, fold: int, n_draws: int = 10) -> list[np.ndarray]`, a list of boolean masks of length `C`. It returns one mask for `full` and `montage`, and `n_draws` masks otherwise.

- [ ] **Step 1: Write the failing tests:**
  - `test_region_examples`: C3 is left-central, Cz is midline-central, C4 is right-central, FC1 is left-frontal, T10 is right-central, POz is midline-posterior, Iz is midline-posterior, and Fp2 is right-frontal.
  - `test_every_dataset_channel_has_a_region`: `region_vector` succeeds on the cached 64 `ch_names` of S001, and every region index 0–8 occurs.
  - `test_montages_occupy_every_region`: for both IV-2a and 10-20, the regions of the kept channels cover all nine.
  - `test_random25_shape_and_occupancy`: every mask keeps exactly 48 channels and occupies all nine regions.
  - `test_masks_are_deterministic_and_seed_dependent`: two calls with the same `(cond, subject, fold)` are equal; changing `fold` changes at least one mask.
  - `test_cluster8_removes_nearest_neighbours`: each mask removes exactly 8 channels, namely a centre electrode and its 7 nearest by 3-D distance.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Parse each name with a regex: a letter prefix, then `z` or digits. Seed each draw with `np.random.default_rng([seed_hash(subject), fold, draw, attempt])`, where `seed_hash` is a stable integer from SHA-256 of the subject string. Rejection loop: redraw with `attempt + 1` until every region is occupied; raise after 1,000 attempts. A cluster's centre is drawn uniformly, and it is removed together with its k − 1 nearest neighbours.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit.** Message: `Define regions, montages and seeded channel patterns`.

### Task 3: Interpolation repair for ShallowConvNet

**Files:**
- Create: `data/interpolate.py`
- Test: `tests/test_interpolate.py`

**Interfaces:**
- Consumes: `positions_3d` and `common_average` (Tasks 1–2).
- Produces:
  - `interpolation_matrix(ch_names: list[str], keep: np.ndarray) -> np.ndarray`, of shape `(n_missing, n_kept)` float64, built from `mne.channels.interpolation._make_interpolation_matrix(pos_kept, pos_missing)`.
  - `repair_full_cap(X_noref: np.ndarray, keep: np.ndarray, W: np.ndarray | None) -> np.ndarray`, mapping `(trials, 64, time)` to `(trials, 64, time)` float32 in four steps (spec §4.1): common average over the kept channels, then fill the missing channels with `W @ kept`, then common average over all 64, then return.

- [ ] **Step 1: Write the failing tests:**
  - `test_repair_with_nothing_missing_is_car`: with an all-True `keep` and `W=None`, the output equals `common_average(X)`.
  - `test_matrix_matches_mne_interpolate_bads`: on random data for the 64 cached channels with 6 marked bad, `repair_full_cap` matches an `mne.EpochsArray` (standard_1005 montage, average reference) put through `interpolate_bads(reset_bads=True)` and then re-referenced to average, within 1e-5 relative.
  - `test_smooth_field_is_recovered`: for a field that is linear in electrode x-position, with 8 channels removed, the interpolated channels' correlation with the true values exceeds 0.95.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement both functions.** Add a module docstring that names the private MNE function and points to the test that guards it against MNE changes.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit.** Message: `Add spherical-spline repair for the ShallowConvNet baseline`.

### Task 4: Exact subset representation from source moments

**Files:**
- Create: `graphs/subset.py`
- Test: `tests/test_subset.py`

**Interfaces:**
- Consumes: `data.normalize.Standardizer`, `graphs.laplacian.{laplacian, rescale_laplacian, chebyshev_basis}`, `data.montage.region_vector`.
- Produces:
  - `Moments`, a frozen dataclass of `sum: np.ndarray` (C,) float64, `outer: np.ndarray` (C, C) float64 and `n: int` (count of trials × time samples), with `__add__`.
  - `moments(X_noref: np.ndarray) -> Moments`, over trials and time.
  - `SubsetRepresentation`, a frozen dataclass of `keep: np.ndarray`, `standardizer: Standardizer`, `basis: np.ndarray` (K, S, S) float32 and `regions: np.ndarray` (S,) int64.
  - `subset_representation(m: Moments, keep: np.ndarray, ch_names: list[str], K: int = 2, lmax: float = 2.0) -> SubsetRepresentation`. It applies the centring matrix `H = I − 11ᵀ/S` to the kept block, giving mean `Hμ` and covariance `H Σ H`. From that it builds the standardiser (population std), `A = |corr| − I` clipped to [0, 1] with a zero diagonal, then the normalised Laplacian, rescaling and the Chebyshev basis. The basis and the regions are both in the kept channels' montage order.

- [ ] **Step 1: Write the failing tests:**
  - `test_moments_add`: `moments(a) + moments(b)` equals `moments(np.concatenate([a, b]))`.
  - `test_matches_direct_computation`: on random `(20, 12, 100)` data with a 7-channel `keep`, the standardiser mean and std and the adjacency implied by the basis equal `fit_standardizer` and `pcc_adjacency` applied to `common_average(X, keep)`, within 1e-6.
  - `test_full_keep_matches_existing_pipeline`: with all channels kept on CAR'd data, the basis equals `chebyshev_basis(rescale_laplacian(laplacian(pcc_adjacency(X)), 2.0), 2)`.
  - `test_basis_and_regions_share_order` (Review Focus 2): for a `keep` whose channels span several regions, `regions[i] == region_of(ch_names[kept_index[i]])` for every row `i` of the basis.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Raise `ValueError` if any kept channel has a standard deviation below 1e-8 after centring.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit.** Message: `Derive subset graphs and standardisers exactly from source moments`.

### Task 5: Region-pooled readout in GraphBandPowerNet

**Files:**
- Modify: `models/graph_bandpower.py`
- Test: `tests/test_graph_bandpower.py`

**Interfaces:**
- Produces:
  - `GraphBandPowerConfig.readout: str = "flat"` (`"flat"` or `"region"`) and `n_regions: int = 9`. `"region"` requires `spatial="graph"`, `n_layers=1` and `pool_nodes=False`; anything else raises `ValueError`.
  - `GraphBandPowerNet.__init__(..., node_regions: np.ndarray | None = None)`, which is required when `readout="region"`.
  - `GraphBandPowerNet.set_graph(basis: np.ndarray, node_regions: np.ndarray) -> None`, which replaces the Chebyshev basis buffer and the region map. It is allowed only for `readout="region"`.
  - In region mode the classifier is `Linear(n_regions * n_spatial, n_classes)`, and `forward` accepts any node count equal to the current basis size. Region features are the mean, over the region's present nodes, of the log band-power, giving shape `(N, n_regions, n_spatial)`. A region with no nodes contributes zeros.

- [ ] **Step 1: Write the failing tests:**
  - `test_region_readout_shapes`: a 64-node model produces `(N, 2)`; after `set_graph` with a 22-node basis and regions, a `(N, 22, 64)` input produces `(N, 2)`.
  - `test_region_readout_is_order_invariant` (Review Focus 2): permuting the input nodes together with the basis rows and columns and with `node_regions` leaves the logits unchanged, within 1e-5.
  - `test_empty_region_gives_zero_features`: regions that never contain index 8 make that region's pooled features zero.
  - `test_flat_readout_unchanged`: with `readout="flat"`, the `state_dict` keys equal the hard-coded pre-change list (`temporal.weight`, `spatial.0.weight`, `norms.0.*`, `classifier.weight`, `classifier.bias`), and the existing tests in this file still pass, so the flat path is untouched.
  - `test_region_requires_graph_single_layer`: `readout="region"` with `spatial="free"` raises `ValueError`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Store the region map as a buffer `region_onehot` of shape `(n_regions, M)`, normalised by the counts clamped to at least 1, and pool with one matmul. In `set_graph`, replace `self.spatial[0].basis` and `region_onehot` in place, on the model's device.
- [ ] **Step 4: Run the tests, plus the whole file.** Expected: PASS.
- [ ] **Step 5: Commit.** Message: `Add a region-pooled readout whose graph can be swapped at test time`.

### Task 6: Runner options: targets, source tag and readout

**Files:**
- Modify: `scripts/run_pretrain_finetune.py`
- Test: `tests/test_pretrain_finetune.py`

**Interfaces:**
- Consumes: `subset_representation` and `moments` (Task 4), `region_vector` (Task 2), the region readout (Task 5).
- Produces:
  - **`--targets SUBJECT ...`:** subjects to fine-tune and evaluate (default: all). Groups are still assigned over every subject; groups with no target are skipped entirely, including their pre-training.
  - **`--source-tag TAG`:** the tag used in the source-cache **filename** (default: `--tag`).
  - **`--readout {flat,region}`:** default `flat`, for `graphbp` only. With `region`, the fold is prepared with `build_graph=False`, so data stay in montage order. The basis and regions then come from `subset_representation(source moments, all-True keep, ch_names)`, with moments from the **no-reference** cache over the source training subjects. Add `--noref-config` (default `configs/bandpass_8_30_noref.yaml`).
  - `source_cache_key` adds `"readout"` to its payload **only when** `args.readout != "flat"`.

- [ ] **Step 1: Write the failing tests:**
  - `test_shallow_cache_key_unchanged` (Review Focus 1): for the arguments the PTG-shallow run used (shallow, graph, smoothness 0, classes [0, 1], window 64, stride 16, lmax 2.0, seed 42, groups 10, dsbn False, epochs 30, lr 1e-3, batch 256, config `configs/bandpass_8_30.yaml`) and group 1's source and validation lists, the key equals the hash embedded in the existing file `runs/source_models/src-PTG-shallow-binary-0-1-g1-*.pt`. Skip if that file is absent.
  - `test_readout_region_changes_graphbp_key`: the `flat` and `region` keys differ for graphbp.
  - `test_targets_keep_full_grouping`: with targets S001–S020, the groups used equal `assign_groups(all 103 subjects, 10)` restricted to groups containing a target, and every source list excludes its whole group.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement the four options.** Do not change any default behaviour.
- [ ] **Step 4: Run the tests and the whole suite.** Expected: PASS, 337 + new.
- [ ] **Step 5: Reproduction check.** Run `C:/Python314/python.exe scripts/run_pretrain_finetune.py --classes 0 1 --model shallow --targets S001 --source-tag PTG-shallow-binary-0-1 --tag CHK-shallow-binary-0-1 --log runs/check.jsonl`. Expected: `reused cached source model` printed for S001's group, and five fold accuracies equal to the logged `PTG-shallow-binary-0-1` rows for S001. Compare with a one-line Python check over the two logs.
- [ ] **Step 6: Commit.** Message: `Let the runner target a subset, reuse a source tag and use the region readout`.

### Task 7: Robustness evaluation hook

**Files:**
- Create: `evaluation/robustness.py`
- Modify: `scripts/run_pretrain_finetune.py`
- Test: `tests/test_robustness.py`

**Interfaces:**
- Consumes: Tasks 2–6, `Trainer.evaluate_samples`, `to_window_samples`.
- Produces:
  - `evaluate_conditions(trainer, kind: Literal["graph", "shallow"], test_noref: Trials, trial_offset: int, ch_names: list[str], src_moments: Moments, full_standardizer: Standardizer, full_rep: SubsetRepresentation | None, subject: str, fold: int, conditions: tuple[Condition, ...], window: int, stride: int) -> dict[str, dict]`. It returns, for each condition name, `{"accuracy": float, "per_draw": list[float], "n_draws": int}`. Condition by kind:
    - `graph`: `X = common_average(test_noref.X, keep)`, then `rep.standardizer.apply`, then windows, then `trainer.model.set_graph(rep.basis, rep.regions)`.
    - `shallow`: `repair_full_cap`, then `full_standardizer.apply`, then windows.
    
    After the loop the full graph is always restored with `set_graph(full_rep.basis, full_rep.regions)`.
  - Runner `--robustness`: after each fold's clean evaluation, call `evaluate_conditions` for every non-`full` condition and log one row per condition under tag `"<tag>.<condition>"`. Each row has every field of the clean row, plus `condition`, `n_draws` and `per_draw`. The clean row is logged **last**.
  - With `--robustness`, a `(subject, fold)` counts as done only if it appears under the main tag **and** every condition tag.
  - The spec's secondary variants are scheduled here. **M1-interp** (graph model fed `repair_full_cap` output, full graph unchanged) is deferred, and runs only if Phase 1 shows an effect worth attributing. **M2-zero** arrives with Task 10, where it is needed for scoring. Spec §9 leaves their timing to the builder.

- [ ] **Step 1: Write the failing tests:**
  - `test_full_restored_after_conditions` (Review Focus 3): with a small region model, the logits on the full-montage input before and after `evaluate_conditions` over all conditions are identical.
  - `test_shallow_path_never_calls_set_graph`: a stub model without `set_graph` runs every condition without error.
  - `test_paired_masks_across_kinds`: the masks drawn for `graph` and `shallow` with the same `(subject, fold)` are identical (spy on `available_masks`).
  - `test_resume_requires_all_condition_tags` (Review Focus 4): a log in which `(S001, 0)` appears under the main tag but not under `.random25` yields a done-set without `(S001, 0)`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement `evaluate_conditions`, and wire `--robustness` into the runner.** Load the no-reference trials of target subjects only, and compute source moments by summing per-subject `moments` over the source training subjects, so the no-reference cache is never held whole in memory. Assert that each target's no-reference `y` equals its referenced `y` before use.
- [ ] **Step 4: Run the tests and the whole suite.** Expected: PASS.
- [ ] **Step 5: Smoke run.** `--model shallow --targets S001 --source-tag PTG-shallow-binary-0-1 --tag CHK2-shallow-binary-0-1 --robustness --folds 5 --log runs/check.jsonl`. Expected: 5 clean rows identical to Task 6's and 35 condition rows, with Full condition accuracy above IV-2a accuracy for at least 3 of 5 folds (a sanity check, not a result). Delete `runs/check.jsonl` afterwards.
- [ ] **Step 6: Commit.** Message: `Score every montage condition after each fine-tuning fold`.

### Task 8: Pre-registration (before any `ROB` run)

**Files:**
- Create: `scripts/print_region_table.py`
- Modify: `docs/project_documentation.md` (new §9.24, plus a changelog entry)

- [ ] **Step 1: Write `scripts/print_region_table.py`.** It prints a Markdown table of the 64 cached channels with their region, followed by the IV-2a and 10-20 sets.
- [ ] **Step 2: Add §9.24 "Montage robustness — pre-registration".** Include:
  - the time and the current commit;
  - the printed region table;
  - the eight conditions and the two primaries;
  - the arms (M1 = `ROB1-chebrp-binary-*`, M2 = `ROB1-shallow-binary-*`);
  - the decision rule and full-montage guard, verbatim from the spec;
  - the Phase 2 settings, verbatim from spec §7.
- [ ] **Step 3: Fix these predictions in §9.24:**
  1. Full: M1 is within 0.020 of M2.
  2. Random 10 %: no difference greater than 0.020.
  3. IV-2a: M1 beats M2 by at least 0.020.
  4. Random 50 %: M1 beats M2 by at least 0.020.
  5. Random 25 %: weaker confidence; M1 is ahead, but possibly by less than 0.020.
- [ ] **Step 4: Commit** with no `ROB` rows in the log. Verify with `grep -c '"tag": "ROB' runs/experiments.jsonl`, which should print 0. Message: `Pre-register the montage-robustness experiment`.

### Task 9: Phase 1 runs and analysis (development set, exploratory)

**Files:**
- Create: `scripts/robustness_results.py`
- Test: `tests/test_analysis_scripts.py` (add one test)
- Modify: `docs/project_documentation.md` (new §9.25 results, plus a changelog entry)

**Interfaces:**
- Consumes: `paired_bootstrap_ci`, `bootstrap_p_value` (`alternative="greater"`) and `holm_bonferroni` from `evaluation/metrics.py`.
- Produces: `robustness_table(rows, m1_prefix: str, m2_prefix: str) -> dict`. It reports per-condition M1 and M2 means, paired difference, CI, *p*, Holm *p*, subject wins and the decision. The subject score is the mean over folds of both problems (within-family). Aggregation refuses to run if any tag has fewer than 5 × the number of targets rows.

- [ ] **Step 1: Write the failing test** `test_robustness_table_decision`: on synthetic rows where M1 is better by 0.03 on every subject for `iv2a` and equal on `random25`, the decision is `win` on `iv2a` and `no effect` on `random25`; with the Full difference set to −0.03, the decision becomes `trade-off`.
- [ ] **Step 2: Implement `scripts/robustness_results.py`** and pass the test.
- [ ] **Step 3: Launch M1 in the background, one problem after the other:** `C:/Python314/python.exe scripts/run_pretrain_finetune.py --classes 0 1 --model graphbp --spatial graph --readout region --targets S001 ... S020 --robustness --tag ROB1-chebrp-binary-0-1 --resume`, then the same with `--classes 2 3` and tag `ROB1-chebrp-binary-2-3`. **Check the first group's elapsed time against about 15 minutes before leaving it.**
- [ ] **Step 4: Launch M2 in the background:** `--model shallow --source-tag PTG-shallow-binary-0-1 --tag ROB1-shallow-binary-0-1`, with the same targets and `--robustness --resume`, then the 2-3 problem with `--source-tag PTG-shallow-binary-2-3`. Expected: `reused cached source model` for every group.
- [ ] **Step 5: When every tag holds 100 rows, run** `C:/Python314/python.exe scripts/robustness_results.py --m1 ROB1-chebrp --m2 ROB1-shallow`. Record the table in §9.25 as EXPLORATORY, with the prediction scorecard and the decision. Commit the log and the doc together.
- [ ] **Step 6: Gate.** If either primary condition shows M1 ahead by at least 0.020, continue to Task 10. Otherwise stop: write the null into §9.25 and the changelog, update `CLAUDE.md`'s Expected Results with one line, and commit.

### Task 10 (gated by Task 9): Phase 2, the channel-dropout objection

**Files:**
- Modify: `evaluation/trainer.py` (an optional per-batch transform hook), `scripts/run_pretrain_finetune.py` (`--channel-dropout 0.25`)
- Test: `tests/test_robustness.py`

**Interfaces:**
- **`--channel-dropout p`:** applied in pre-training *and* fine-tuning.
  - **Shallow:** each training window zeroes `round(p · 64)` channels, chosen afresh per window.
  - **Graph:** each **batch** drops one random channel set (the same pattern for the whole batch, so one basis serves it). The data are re-referenced over the kept channels, the subset representation is set, the forward pass runs, and the full graph is restored after the step.
- The source-cache key gains `"channel_dropout"` only when non-zero.
- Scoring follows spec §7: M2 is scored with both `repair_full_cap` and zero-fill, and the per-subject maximum of the two counts.

- [ ] **Step 1: Write the tests:**
  - dropout 0 leaves training bit-identical (one seeded batch);
  - each shallow window zeroes exactly 16 channels;
  - the graph model's basis is the full basis after `fit` returns;
  - M2's score is the per-subject maximum of the two repairs.
- [ ] **Step 2: Implement, run the tests and the whole suite, and commit.**
- [ ] **Step 3: Run tags `ROB2-chebrp-*` and `ROB2-shallow-*`** on S001–S020 with `--channel-dropout 0.25 --robustness`. Analyse with `robustness_table`, record §9.26, and commit.
- [ ] **Step 4: Gate.** Continue to Task 11 only if M1 still meets all four criteria on a primary condition. Otherwise record it as not surviving, and stop as in Task 9 Step 6.

### Task 11 (gated by Task 10): Phase 3 confirmation

- [ ] **Step 1: Append to §9.24, before launching,** a dated note that confirmation starts from the Task 10 configuration, unchanged, with the commit hash.
- [ ] **Step 2: Run the Task 10 configuration** with `--targets` set to the 83 non-development subjects, tags `ROB3-*`.
- [ ] **Step 3: Analyse, then record §9.27 (CONFIRMATORY for this question) with the decision, update `CLAUDE.md`'s Expected Results, and commit.**
