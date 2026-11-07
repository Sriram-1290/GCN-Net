# Phase A: inflation audit and Euclidean alignment on EEGMMIDB

*Design, 2026-11-07. Implements Phase A of the PW-2 plan in `docs/RESEARCH_DIRECTIONS.md`
(§7.2 WP1–WP3, as revised by §9). The user opted out of design questions; every decision
below not taken from that document is the author's call and is marked **(call)**.*

## 1. Intent

The planned Q1 paper (working title *"What inflates and what improves motor-imagery decoding
on PhysioNet"*) needs two kinds of evidence from this dataset before any new dependency or
download.

- **Audit evidence:** measurements that show *how much* published-style evaluations inflate
  results. These are the novel contributions confirmed open in §9.1 of that document: the
  EEGMMIDB run-family confound, and the GCNs-Net leakage paths named individually.
- **One method result:** whether Euclidean alignment (EA), the best-supported untried method,
  improves subject-independent and fine-tuned decoding here.

Phase B covers foundation models and extra datasets. It needs new libraries and downloads and
is **out of scope**.

## 2. Goals, non-goals, anti-goals

**Goals**
- Quantify the run-family confound three ways: label-free from rest data, on the
  cross-family task MOABB benchmarks, and in 4-class subject-independent decoding.
- Measure each GCNs-Net leakage path on its own.
- Test EA, pre-registered, in the subject-independent and fine-tuned settings.
- Check whether the project's own models depend on the first second after the cue.

**Non-goals**
- New architectures.
- Foundation models; any dataset other than EEGMMIDB; data augmentation (deferred to a later
  phase); test-time BatchNorm adaptation (deferred, so EA is measured alone — the
  one-change-at-a-time rule).
- Hyperparameter tuning of anything.

**Anti-goals** (failures even if a number comes out)
- Any test-trial statistic entering training, including EA references computed from test
  trials, except in the explicitly labelled transductive variant (§3.6).
- Giving EA to the deep model but not to the classical baselines.
- Quoting a 4-class figure without its run-family decomposition, or quoting any partial run.
- Reporting an audit measurement as a "method win".

## 3. Experiments

All experiments use the existing referenced cache (8–30 Hz, common average reference,
64-sample windows), 103 subjects, the existing seeds, logging to `runs/experiments.jsonl`
with new tags, and paired statistics as in the rest of the project. **(call)** Run everything
on all 103 subjects:
- the audit items (A1–A5) are measurements, not method claims;
- EA (A6) is one literature-fixed method with no tuned settings and is pre-registered.

The development/confirmation split is not used here; this is stated in the pre-registration.

### A1 — 4-class subject-independent decoding with the family decomposition

- **Models:** ShallowConvNet, plus the in-house tangent-space classifier.
- **Protocol (call):** 10-group subject-disjoint cross-validation (`assign_groups`, index mod
  10), not true leave-one-subject-out. That is 10 trainings instead of 103 per arm, and the
  same groups as the pre-training arms, so results pair with them.
  - The existing binary LOSO figure (0.604) came from 103-fold LOSO with a different model,
    so it is **not** compared directly.
  - The binary baseline in this scheme is re-measured as part of A6.
- **Reported:** 4-class accuracy, run-family accuracy and within-family accuracy (the
  decomposition in `scripts/run_family_analysis.py`), with per-subject values.

### A2 — dependence on the first second after the cue

- **Arms:** ShallowConvNet, subject-dependent, trained and tested in two ways:
  - (i) on all windows, which is the existing arm (reused);
  - (ii) excluding every window that starts before 1.0 s after the cue.
- **Contrast:** within-family accuracy (ii) − (i).
- **Reading:** a drop of 0.020 or more says the model leans on early-cue activity, the
  NeuroAtlas concern. A smaller difference says it does not.

### A3 — label-free run-family confound from rest data

- **Data:** the rest periods (T0, about 15 per imagery run) carry no class label, so a new
  cache variant is needed that holds T0 epochs with their run number.
  - **(call)** Use the same preprocessing as the referenced cache.
  - Also build an unfiltered (common average reference only) twin to test whether band-pass
    filtering removes the confound, which is Li et al.'s finding.
- **Task:** classify run family (runs 4/8/12 versus 6/10/14) from rest epochs, per subject.
- **Splits:**
  - (i) trial-disjoint 5-fold, mixing runs. Measures whether run-level signatures exist at all.
  - (ii) run-disjoint: hold out one run of each family; there are three such pairs, so 3
    folds. Measures whether family is decodable on runs never seen.
- **Models:** tangent space and ShallowConvNet.
- **Reading:** above-chance accuracy under (i) means rest data carry run-family information
  with no class signal; the gap (i) − (ii) is how much is run-specific.

### A4 — the cross-family task (MOABB's right hand versus feet)

- **Tasks:** right fist (runs 4/8/12) versus both feet (runs 6/10/14), a cross-family task.
  Controls: left versus right fist, and both fists versus both feet, which are within-family
  tasks.
- **Splits:**
  - (i) trial-disjoint 5-fold mixing runs, as MOABB does within a session;
  - (ii) run-disjoint, with folds holding out one fist run and one feet run.
- **Metric:** ROC-AUC, to match MOABB, and accuracy. Models: tangent space and ShallowConvNet.
- **Inflation estimate:** (AUC<sub>i</sub> − AUC<sub>ii</sub>) on the cross-family task minus
  the same difference on the within-family tasks. That separates the confound from the
  generic cost of run-disjoint splitting.

### A5 — GCNs-Net leakage paths, one at a time

Starting from the correct protocol (0.509), switch on each leakage path alone, then all three:
- timepoint-level random split;
- BatchNorm in training mode at test time;
- graph built on pooled train and test data.

**(call)** One seed, 103 subjects, the reference formulation (published preprocessing, no
band-pass), using the existing leaky-replication code paths. The all-three arm should
reproduce the logged 0.8975 within noise, which checks the toggles. Each row is mapped to its
NERVE-ML item in the write-up.

### A6 — Euclidean alignment (pre-registered method test)

- **EA module:** a per-subject reference R = mean trial covariance; whitening by
  R<sup>-1/2</sup> computed as a **pseudo-inverse square root**, keeping eigenvalues above a
  relative tolerance of 1e-8 (the remedy for the rank-63 covariance after the common average
  reference, §9.4). EA is applied after the band-pass.
- **Where each reference comes from:**
  - source subjects: each from its own training trials;
  - a fine-tuned target: from its own training-fold trials only;
  - a subject-independent target, primary (**call**): pseudo-online. The reference comes from
    the subject's **earliest run of the task's family** (run 4 for the fist family, run 6 for
    the fists-vs-feet family; 15 trials). That run is excluded from scoring in **both** arms of
    the comparison, so the arms stay paired. The literature's "first 24 trials" would leave too
    few test trials in a roughly 45-trial binary task.
  - Secondary and labelled transductive: an offline reference from all of the target's
    unlabelled trials, reported separately.
- **Arms**, each with and without EA, sharing groups, seeds and folds:
  1. **Subject-independent, ShallowConvNet.**
     - Without EA: the cached PTG-shallow source models, evaluated zero-shot on their own
       group's targets. No new training is needed.
     - With EA: new source models trained on EA-aligned source data.
  2. **Subject-independent, tangent space** (classical control; must also get EA).
  3. **Fine-tuned ShallowConvNet.**
     - Without EA: PTG-shallow (logged, 0.7369).
     - With EA: the EA source models, fine-tuned on EA-aligned target training folds.
- **Pre-registered predictions (call):**
  1. EA improves subject-independent ShallowConvNet by at least 0.020.
  2. EA improves subject-independent tangent space by at least 0.020.
  3. EA changes fine-tuned ShallowConvNet by less than 0.020, because pre-training absorbs it,
     as it absorbed domain-specific batch norm.
- **Decision rule:** the project standard. A difference counts as an effect only if it is at
  least 0.020 with a 95 % paired bootstrap CI excluding zero; Holm correction across the three
  primary contrasts; per-subject win counts reported.

## 4. Order and dependencies

1. Pre-register A1–A6 in `docs/project_documentation.md` (§9.26 onwards) before any run.
2. A5 and A2 reuse existing code paths and are the cheapest.
3. Build the T0 rest cache, then run A3.
4. Run-disjoint splits use the cached `run` field, then run A4.
5. EA module, then A6.
6. A1 uses the A6 infrastructure: zero-shot scoring of group source models, but trained 4-class.

## 5. Constraints

- No new dependencies and no downloads. Tangent space, CSP and EA are in-house; `sklearn` is
  already used.
- Batch 256 for `graphbp` stays as it is. Long runs must be resumable and must release memory
  between groups (the hotfix pattern).
- **Leakage rules hold everywhere:**
  - graphs, standardisers and EA references come from training data, except the labelled
    transductive EA variant;
  - run-disjoint folds are generated from run numbers only.
- Every new path gets unit tests on synthetic data. EA in particular needs tests for:
  - whitening identity on full-rank data;
  - rank-deficient (common-average-referenced) input without blow-up;
  - "reference from training trials only".
- Default behaviour of existing scripts stays unchanged. New options are opt-in, and any
  cache key changes only when they are used.

## 6. Left to the builder

- Tag names; how zero-shot scoring and EA are wired into `scripts/run_pretrain_finetune.py`
  or a sibling script.
- Whether A3 and A4 share one evaluation script.
- The exact T0 epoch length: 4.0 s to match imagery trials if T0 segments are long enough,
  otherwise the longest common length, stated in the pre-registration.

## 7. Risks

- **A6's EA source models need new pre-training.** Two problems × 10 groups of ShallowConvNet
  is hours of compute, so they run in the background with retry and resume.
- **4-class source models (A1) are a further pre-training set.** If compute is short, A1 drops
  to tangent space only, and that is reported.
- **A3 may find no run-family signal under band-pass.** That is still a reportable result: it
  would localise the confound to unfiltered pipelines, as Li et al. found.
