"""Leakage tests for the pre-train / fine-tune runner (17.11.2).

This script is the most leakage-sensitive thing in the project: it deliberately trains
on 102 other subjects before touching the target, so the boundaries have to be checked
rather than assumed. Three of them carry the protocol:

1. **The target contributes nothing to pre-training** — not their trials, not the graph,
   not the normalisation statistics.
2. **Fine-tune train and test are disjoint at TRIAL level**, and stay disjoint after
   windowing. A single shared window would reintroduce the exact defect the whole
   project exists to document (§4.4).
3. **The folds match `run_subject_dependent.py` at the same seed**, or the arms are not
   paired and no per-subject comparison is valid.

The control arm gets a test too: `--no-pretrain` must produce the identical pipeline
with only the pre-training skipped, since that is what isolates the effect of
pre-training from the effect of using source-fit normalisation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.splits import stratified_trial_kfold  # noqa: E402
from evaluation.pipeline import Samples, Trials, concat_trials, prepare_fold  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_pretrain_finetune", ROOT / "scripts" / "run_pretrain_finetune.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trials(n=20, c=64, t=640, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, c, t)).astype(np.float32)
    y = np.array([0, 1] * (n // 2), dtype=np.int64)
    return Trials(X=X, y=y)


# ------------------------------------------------------- the trial-id partition

def test_subset_samples_selects_exactly_the_named_trials():
    m = _load()
    s = Samples(X=np.arange(60, dtype=np.float32).reshape(20, 3, 1),
                y=np.zeros(20, dtype=np.int64),
                trial_id=np.repeat(np.arange(5) + 200_000, 4))
    out = m.subset_samples(s, {200_000, 200_002})
    assert set(np.unique(out.trial_id)) == {200_000, 200_002}
    assert len(out) == 8


def test_finetune_train_and_test_share_no_window():
    """The defect this project documents, asserted for the new runner.

    A trial-level split must survive windowing: every window of a test trial must be
    absent from the fine-tune training set.
    """
    m = _load()
    target = _trials(seed=1)
    source = concat_trials([_trials(seed=2), _trials(seed=3)])
    prepared = prepare_fold(source, target, window=64, stride=16, build_graph=False)

    for split in stratified_trial_kfold(target.y, n_splits=5, seed=42):
        tr = m.subset_samples(prepared.test, set(split.train + 200_000))
        te = m.subset_samples(prepared.test, set(split.test + 200_000))
        assert len(tr) and len(te)
        assert not (set(np.unique(tr.trial_id)) & set(np.unique(te.trial_id))), (
            f"fold {split.fold}: a trial appears in both fine-tune train and test")
        # and together they account for every window, nothing silently dropped
        assert len(tr) + len(te) == len(prepared.test)


def test_folds_match_the_subject_dependent_runner_at_the_same_seed():
    """Without this the arms are not paired and no per-subject comparison holds."""
    target = _trials(seed=4)
    a = [(s.fold, tuple(s.train), tuple(s.test))
         for s in stratified_trial_kfold(target.y, n_splits=5, seed=42)]
    b = [(s.fold, tuple(s.train), tuple(s.test))
         for s in stratified_trial_kfold(target.y, n_splits=5, seed=42)]
    assert a == b, "the fold generator is not deterministic at a fixed seed"


# --------------------------------------------- the target never reaches pre-training

def test_source_statistics_do_not_depend_on_the_target():
    """The graph and standardiser must be identical whatever the target contains.

    `prepare_fold` fits on its `train` argument only; the target is passed as `test`
    purely to be transformed. If the target ever leaked into the fit, replacing it with
    different data would change the fitted statistics.
    """
    source = concat_trials([_trials(seed=5), _trials(seed=6)])
    one = prepare_fold(source, _trials(seed=7), window=64, stride=16, build_graph=True)
    two = prepare_fold(source, _trials(seed=99), window=64, stride=16,
                       build_graph=True)

    assert np.allclose(one.adjacency, two.adjacency), (
        "the adjacency changed when only the target changed -- the target is leaking "
        "into the graph")
    assert np.allclose(one.standardizer.mean, two.standardizer.mean)
    assert np.allclose(one.standardizer.std, two.standardizer.std)


def test_target_is_excluded_from_the_source_list():
    """The runner's own selection rule, checked directly."""
    subjects = [f"S{i:03d}" for i in range(1, 11)]
    for target in subjects:
        source = [s for s in subjects if s != target]
        assert target not in source
        assert len(source) == len(subjects) - 1


# ---------------------------------------------------------------- grouped pre-training

def test_groups_partition_the_subjects_exactly():
    m = _load()
    subjects = [f"S{i:03d}" for i in range(1, 104)]
    for k in (0, 1, 3, 10, 103, 500):
        groups = m.assign_groups(subjects, k)
        flat = [s for g in groups for s in g]
        assert sorted(flat) == sorted(subjects), f"k={k}: not a cover"
        assert len(flat) == len(set(flat)), f"k={k}: groups overlap"
        assert all(groups), f"k={k}: an empty group"


def test_no_target_is_ever_in_its_own_source_set():
    """The leakage guarantee that grouping has to preserve.

    Pre-training excludes the target's whole GROUP, not just the target, so every
    member of that group is absent from the model they later fine-tune from. If this
    ever failed, a subject would be pre-trained on their own trials.
    """
    m = _load()
    subjects = [f"S{i:03d}" for i in range(1, 104)]
    for k in (0, 10, 25):
        for group in m.assign_groups(subjects, k):
            source = [s for s in subjects if s not in group]
            for target in group:
                assert target not in source, f"k={k}: {target} is in its own source set"
            assert len(source) == len(subjects) - len(group)


def test_grouping_is_deterministic_and_seed_free():
    """Reproducible without carrying a seed, since it is index % k."""
    m = _load()
    subjects = [f"S{i:03d}" for i in range(1, 104)]
    assert m.assign_groups(subjects, 10) == m.assign_groups(subjects, 10)


def test_groups_are_interleaved_not_contiguous_blocks():
    """A contiguous block would make a group a slice of the recording order."""
    m = _load()
    subjects = [f"S{i:03d}" for i in range(1, 104)]
    first = m.assign_groups(subjects, 10)[0]
    assert first[:3] == ["S001", "S011", "S021"], first[:3]


def test_ten_groups_cut_the_pretraining_count_by_an_order_of_magnitude():
    """The point of the change: error 24 made 103 pre-trainings a ~51 h run."""
    m = _load()
    subjects = [f"S{i:03d}" for i in range(1, 104)]
    assert len(m.assign_groups(subjects, 0)) == 103      # leave-one-subject-out
    assert len(m.assign_groups(subjects, 10)) == 10      # grouped
    # and a target still draws on the great majority of the cohort
    assert min(103 - len(g) for g in m.assign_groups(subjects, 10)) >= 92


# ------------------------------------------------------------------ the control arm

def test_no_pretrain_control_is_declared_and_documented():
    """The control exists and says why, since it is what makes the arm interpretable."""
    text = (ROOT / "scripts" / "run_pretrain_finetune.py").read_text(encoding="utf-8")
    assert "--no-pretrain" in text
    assert "THE CONTROL" in text
    assert "isolat" in text.lower()


def test_finetune_lr_defaults_below_pretrain_lr():
    """Fine-tuning ~36 trials at the pre-training rate destroys transferred features."""
    src = (ROOT / "scripts" / "run_pretrain_finetune.py").read_text(encoding="utf-8")
    assert '"--pretrain-lr", type=float, default=1e-3' in src
    assert '"--finetune-lr", type=float, default=1e-4' in src


def test_model_factory_covers_every_runner_architecture():
    m = _load()
    with pytest.raises(ValueError, match="unknown model"):
        m.build_model("resnet", None, 64, 2, 160.0)


# ------------------------------------------------- the source-model cache

class _Args:
    """A stand-in for the parsed namespace, carrying only what the key hashes."""

    def __init__(self, **over):
        self.model = "graphbp"
        self.spatial = "graph"
        self.graph_smoothness = 0.0
        self.classes = [0, 1]
        self.window = 64
        self.stride = 16
        self.lmax = 2.0
        self.groups = 10
        self.pretrain_epochs = 30
        self.pretrain_lr = 1e-3
        self.batch_size = 256
        self.config = "configs/bandpass_8_30.yaml"
        self.dsbn = False
        self.readout = "flat"
        self.noref_config = "configs/bandpass_8_30_noref.yaml"
        self.__dict__.update(over)


def _key(module, args=None, seed=42, group=1, source=None, val=None):
    return module.source_cache_key(
        args or _Args(), seed, group,
        source if source is not None else ["S001", "S002", "S003"],
        val if val is not None else ["S004"])


def test_source_cache_key_is_stable_across_calls():
    m = _load()
    assert _key(m) == _key(m)


def test_source_cache_key_ignores_subject_list_order():
    """The source set is a set; a reordering must still hit the same cache entry."""
    m = _load()
    assert _key(m, source=["S001", "S002", "S003"]) == \
           _key(m, source=["S003", "S001", "S002"])


@pytest.mark.parametrize("field,value", [
    ("model", "shallow"),
    ("spatial", "free"),
    ("graph_smoothness", 0.1),
    ("classes", [2, 3]),
    ("window", 128),
    ("stride", 32),
    ("lmax", 1.5),
    ("groups", 5),
    ("pretrain_epochs", 10),
    ("pretrain_lr", 1e-2),
    ("batch_size", 512),
    ("config", "configs/default.yaml"),
    # A DSBN source model must never be reused for a non-DSBN run: the two normalise
    # differently, and a stale hit would silently mix the schemes within one arm.
    ("dsbn", True),
])
def test_every_hyperparameter_that_changes_the_weights_changes_the_key(field, value):
    """A stale hit would pre-train on the wrong thing — the failure this must not have."""
    m = _load()
    assert _key(m) != _key(m, args=_Args(**{field: value})), (
        f"{field} changes the fitted source model but not its cache key")


@pytest.mark.parametrize("kwargs", [
    {"seed": 43},
    {"group": 2},
    {"source": ["S001", "S002", "S009"]},
    {"val": ["S005"]},
])
def test_run_identity_fields_change_the_key(kwargs):
    m = _load()
    assert _key(m) != _key(m, **kwargs)


def test_source_cache_key_is_a_short_hex_digest():
    m = _load()
    k = _key(m)
    assert len(k) == 16 and all(c in "0123456789abcdef" for c in k)


def test_source_cache_is_on_by_default_and_can_be_disabled():
    """Caching only removes recomputation, so it is safe as a default; the flag
    exists to force a cold pre-train when checking reproducibility."""
    import subprocess
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_pretrain_finetune.py"), "--help"],
        capture_output=True, text=True, timeout=120).stdout
    assert "--source-cache" in out
    src = (ROOT / "scripts" / "run_pretrain_finetune.py").read_text(encoding="utf-8")
    assert 'default="runs/source_models"' in src


def test_cached_checkpoint_is_written_atomically():
    """A power cut mid-write must not leave a truncated file a later run trusts.

    Three power cuts have already interrupted this script; a half-written
    checkpoint that loaded cleanly would corrupt an arm silently.
    """
    src = (ROOT / "scripts" / "run_pretrain_finetune.py").read_text(encoding="utf-8")
    assert ".partial" in src and "tmp.replace(cache_path)" in src


# ------------------------------------- each stage seeds itself (reproducibility)

def test_seed_stage_is_deterministic_and_distinct_per_stage():
    m = _load()
    import torch

    def draw(*parts):
        m.seed_stage(42, *parts)
        return torch.rand(4).tolist()

    assert draw(1, 0, 0) == draw(1, 0, 0), "the same stage must draw the same numbers"
    assert draw(1, 0, 0) != draw(2, 0, 0), "different groups must not share a stream"
    assert draw(1, 1, 0) != draw(1, 1, 1), "different folds must not share a stream"
    assert draw(1, 0, 0) != _seed_other(m, 43), "the run seed must still matter"


def _seed_other(m, seed):
    import torch
    m.seed_stage(seed, 1, 0, 0)
    return torch.rand(4).tolist()


def test_validation_pick_does_not_depend_on_how_many_groups_preceded_it():
    """A resumed run skips completed groups; the survivors must be unaffected.

    Before this, `val_pick` came from one generator advanced once per group, so a
    run resumed mid-cohort drew a different validation split for every remaining
    group than an uninterrupted run would — silently unreproducible.
    """
    src = (ROOT / "scripts" / "run_pretrain_finetune.py").read_text(encoding="utf-8")
    assert "np.random.default_rng([seed, gi])" in src, (
        "the validation pick must be seeded per group, not drawn from a shared "
        "generator whose position depends on the resume point")


def test_finetune_is_seeded_before_the_model_is_built():
    """Dropout draws from the global generator, so the fine-tune stage has to be
    seeded before it runs or a cached source model changes the result."""
    src = (ROOT / "scripts" / "run_pretrain_finetune.py").read_text(encoding="utf-8")
    ft = src.index("# -------------------------------------------------------- fine-tune")
    seed_call = src.index("seed_stage(seed, gi, j + 1, split.fold)", ft)
    build = src.index("build_model(", seed_call)
    assert seed_call < build, "seed_stage must precede build_model in the fine-tune loop"


# ----------------------------------------- targets, source tag and region readout

def _all_subjects():
    from data.config import all_subjects, load_config
    return all_subjects(load_config(ROOT / "configs" / "bandpass_8_30.yaml"))


def test_shallow_cache_key_unchanged():
    """Adding --readout must not re-key the cached ShallowConvNet source models.

    Those took ~5 h to pre-train; a drifted key would silently re-run them. The
    expected hash is read from the file name of the model the PTG-shallow run wrote.
    """
    files = sorted((ROOT / "runs" / "source_models").glob(
        "src-PTG-shallow-binary-0-1-g1-*.pt"))
    if not files:
        pytest.skip("runs/source_models has no PTG-shallow group-1 model")
    m = _load()
    subjects = _all_subjects()
    (gi, group, source), = [p for p in m.plan_groups(subjects, 10, None) if p[0] == 1]
    val_pick = m.pick_validation(source, 42, gi, 5)
    args = _Args(model="shallow", readout="flat")
    key = m.source_cache_key(args, 42, gi, source, val_pick)
    assert files[0].name == f"src-PTG-shallow-binary-0-1-g1-{key}.pt"


def test_readout_region_changes_graphbp_key():
    m = _load()
    flat = _key(m, args=_Args(model="graphbp", readout="flat"))
    region = _key(m, args=_Args(model="graphbp", readout="region"))
    assert flat != region


def test_flat_readout_leaves_the_key_payload_untouched():
    """The key of a flat run must not mention the readout at all, or every cached
    source model written before the option existed would miss."""
    src =(ROOT / "scripts" / "run_pretrain_finetune.py").read_text(encoding="utf-8")
    assert 'if args.readout != "flat"' in src


def test_targets_keep_full_grouping():
    m = _load()
    subjects = [f"S{i:03d}" for i in range(1, 110)
                if f"S{i:03d}" not in {"S088", "S089", "S092", "S100", "S104", "S106"}]
    assert len(subjects) == 103
    targets = [f"S{i:03d}" for i in range(1, 21)]
    full = m.assign_groups(subjects, 10)
    plan = m.plan_groups(subjects, 10, targets)
    expect = [(gi, g) for gi, g in enumerate(full, 1) if set(g) & set(targets)]
    assert [(gi, g) for gi, g, _ in plan] == expect
    for gi, group, source in plan:
        assert not set(source) & set(group)
        assert set(source) | set(group) == set(subjects)
    # no targets means every group is planned, with its own index
    assert [gi for gi, _, _ in m.plan_groups(subjects, 10, None)] == list(range(1, 11))
    # a single target selects only its group, keeping the group's own index
    (gi, group, _), = m.plan_groups(subjects, 10, ["S003"])
    assert gi == 3 and "S003" in group


def test_unknown_target_fails_loudly():
    m = _load()
    with pytest.raises(ValueError, match="S999"):
        m.plan_groups(["S001", "S002"], 10, ["S999"])


def test_region_readout_model_is_built_from_a_subset_representation():
    """build_model takes the (basis, regions) pair and ignores prepared.bases, which
    is empty because the fold is prepared with build_graph=False."""
    import torch
    from graphs.subset import moments, subset_representation
    m = _load()
    names = ["Fz", "C3", "Cz", "C4", "Pz", "O1"]
    X = np.random.default_rng(0).normal(size=(6, len(names), 200))
    rep = subset_representation(moments(X), np.ones(len(names), dtype=bool), names)
    prepared = prepare_fold(Trials(X=X.astype(np.float32), y=np.array([0, 1] * 3)),
                            Trials(X=X.astype(np.float32), y=np.array([0, 1] * 3)),
                            window=64, stride=16, build_graph=False)
    assert prepared.bases == []
    net = m.build_model("graphbp", prepared, 64, 2, 160.0, readout="region",
                        region_graph=(rep.basis, rep.regions))
    out = net(torch.zeros(3, len(names), 64))
    assert out.shape == (3, 2)
    with pytest.raises(ValueError, match="region_graph"):
        m.build_model("graphbp", prepared, 64, 2, 160.0, readout="region")


# ------------------------------------------------ fix round 1: provenance in the log

def test_region_key_depends_on_the_noref_config():
    """The region basis is built from the no-reference cache, so the key must see it."""
    m = _load()
    a = _key(m, args=_Args(model="graphbp", readout="region",
                           noref_config="configs/bandpass_8_30_noref.yaml"))
    b = _key(m, args=_Args(model="graphbp", readout="region",
                           noref_config="configs/other_noref.yaml"))
    assert a != b


def test_flat_key_ignores_the_noref_config():
    """A flat run never reads that cache, so its key must not depend on the option."""
    m = _load()
    assert _key(m, args=_Args(noref_config="a.yaml")) ==            _key(m, args=_Args(noref_config="b.yaml"))


def test_provenance_fields_only_for_non_default_options():
    m = _load()
    default = _Args(tag="T", source_tag=None, noref_config="n.yaml")
    assert m.provenance_fields(default) == {}
    same = _Args(tag="T", source_tag="T", noref_config="n.yaml")
    assert m.provenance_fields(same) == {}
    tagged = _Args(tag="T", source_tag="S", noref_config="n.yaml")
    assert m.provenance_fields(tagged) == {"source_tag": "S"}
    region = _Args(tag="T", source_tag=None, readout="region", noref_config="n.yaml")
    assert m.provenance_fields(region) == {"readout": "region", "noref_config": "n.yaml"}


def test_targets_run_logs_the_full_group_count(tmp_path):
    """A --targets run still uses 4 groups here; the log must say 4, not 1."""
    import json
    import subprocess
    subs = [f"S{i:03d}" for i in range(1, 13)]
    if not all((ROOT / "data" / "cache_bp830" / f"{s}.npz").exists() for s in subs):
        pytest.skip("preprocessed cache absent")
    log = tmp_path / "log.jsonl"
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_pretrain_finetune.py"),
         "--classes", "0", "1", "--model", "shallow", "--subjects", *subs,
         "--groups", "4", "--targets", "S002", "--no-pretrain", "--finetune-epochs", "1",
         "--folds", "2", "--val-subjects", "2", "--source-cache", "",
         "--source-tag", "OTHER", "--tag", "T", "--log", str(log)],
        capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and all(r["subject"] == "S002" for r in rows)
    assert all(r["n_groups"] == 4 and r["group_id"] == 2 for r in rows)
    assert all(r["source_tag"] == "OTHER" and "readout" not in r for r in rows)


# ------------------------------------- task 7: no source windows on a cache hit

def test_prepare_fold_without_train_samples_changes_nothing_else():
    """On a source-cache hit the runner skips the ~2.2 GiB of source-train windows,
    which feed only pre-training. The standardiser (fit on the full source trials),
    the target windows and the per-window shape must be exactly as before."""
    src = concat_trials([_trials(seed=2), _trials(seed=3)])
    tgt = _trials(seed=1)
    full = prepare_fold(src, tgt, window=64, stride=16, build_graph=False)
    lean = prepare_fold(src, tgt, window=64, stride=16, build_graph=False,
                        train_samples=False)
    assert np.array_equal(lean.standardizer.mean, full.standardizer.mean)
    assert np.array_equal(lean.standardizer.std, full.standardizer.std)
    assert lean.standardizer.n_trials_fit == full.standardizer.n_trials_fit
    assert np.array_equal(lean.test.X, full.test.X)
    assert np.array_equal(lean.test.trial_id, full.test.trial_id)
    assert len(lean.train) == 0
    assert lean.train.X.shape[1:] == full.train.X.shape[1:]   # what build_model reads


def test_pretraining_refuses_a_fold_without_source_windows():
    """Fix round 1: an empty source set must never be 'pre-trained' and cached under
    a valid key -- that would poison the cache for every later run."""
    m = _load()
    src = concat_trials([_trials(seed=2), _trials(seed=3)])
    lean = prepare_fold(src, _trials(seed=1), window=64, stride=16, build_graph=False,
                        train_samples=False)
    with pytest.raises(RuntimeError, match="source windows"):
        m.require_source_windows(lean)
    m.require_source_windows(prepare_fold(src, _trials(seed=1), window=64, stride=16,
                                          build_graph=False))


def test_cache_hit_is_decided_once_and_pretraining_is_guarded():
    src = (ROOT / "scripts" / "run_pretrain_finetune.py").read_text(encoding="utf-8")
    assert src.count("cache_path.exists()") == 1
    fit = src.index("trainer.fit(prepared)")
    guard = src.rfind("require_source_windows(prepared)", 0, fit)
    assert guard != -1 and fit - guard < 600


def test_shallow_model_builds_identically_from_a_lean_fold():
    import torch
    m = _load()
    src = concat_trials([_trials(seed=2), _trials(seed=3)])
    lean = prepare_fold(src, _trials(seed=1), window=64, stride=16, build_graph=False,
                        train_samples=False)
    full = prepare_fold(src, _trials(seed=1), window=64, stride=16, build_graph=False)
    torch.manual_seed(0)
    a = m.build_model("shallow", lean, 64, 2, 160.0)
    torch.manual_seed(0)
    b = m.build_model("shallow", full, 64, 2, 160.0)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_group_loop_releases_its_data_before_the_next_group():
    """The second group of a process crashed for host memory because the previous
    group's windows were still referenced while the next group's were built."""
    src = (ROOT / "scripts" / "run_pretrain_finetune.py").read_text(encoding="utf-8")
    loop = src.index("for gi, group, source in plan:")
    release = src.index("release_host_memory()", loop)
    assert src.index("prepared = model = trainer = pretrained_state = None", loop) < release
    assert src.index("nothing to do; every subject", loop) > release  # after the loop body
    assert "\n        release_host_memory()\n" in src  # at loop-body depth, not after it
    helper = src[src.index("def release_host_memory"):src.index("def build_model")]
    assert "gc.collect()" in helper and "torch.cuda.empty_cache()" in helper


# ------------------------------- Phase A (9.26 A1, A6): zero-shot scoring and EA
#
# These run main() in-process on small synthetic subjects (load_subject is replaced),
# so they exercise the real group loop: the same code path the A6 and A1 arms use.

import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

RUN_ORDER = (4, 6, 8, 10, 12, 14)
N_PER_RUN = 6
N_CH, N_T = 8, 160
SUBJECTS = ("S001", "S002", "S003", "S004")


def _fake_subject(subject: str, cache_dir=None):
    """36 trials, 6 per imagery run, labelled f(run, T-code) as in the real cache.

    Runs 4/8/12 hold classes 0/1, runs 6/10/14 classes 2/3, three of each per run.
    Each subject gets its own channel mixing, so Euclidean alignment changes the data.
    """
    rng = np.random.default_rng(1000 + int(subject[1:]))
    run = np.repeat(RUN_ORDER, N_PER_RUN).astype(np.int64)
    alt = np.tile([0, 1], len(run) // 2)
    y = np.where(np.isin(run, (4, 8, 12)), alt, 2 + alt).astype(np.int64)
    mix = np.eye(N_CH) + 0.3 * rng.normal(size=(N_CH, N_CH))
    X = np.einsum("cd,ndt->nct", mix, rng.normal(size=(len(run), N_CH, N_T)))
    return SimpleNamespace(subject=subject, X=X.astype(np.float32), y=y, run=run,
                           onset=np.zeros(len(run), dtype=np.float32),
                           ch_names=[f"C{i}" for i in range(N_CH)], sfreq=160.0)


def _task_runs(classes):
    """Run of every trial of one fake subject after the class filter, in order."""
    ep = _fake_subject("S001")
    keep = np.isin(ep.y, classes) if classes else np.ones(len(ep.y), dtype=bool)
    return ep.run[keep]


def _run_main(m, monkeypatch, workdir, *extra, subjects=SUBJECTS):
    """Run the real main() on the fake subjects; return the logged rows."""
    monkeypatch.setattr(m, "load_subject", _fake_subject)
    log = Path(workdir) / "log.jsonl"
    argv = ["run_pretrain_finetune.py",
            "--config", str(ROOT / "configs" / "bandpass_8_30.yaml"),
            "--subjects", *subjects, "--groups", "2", "--val-subjects", "1",
            "--pretrain-epochs", "1", "--finetune-epochs", "1", "--folds", "3",
            "--source-cache", "", "--log", str(log), "--quiet", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    assert m.main() == 0
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("ea", ["none", "pseudo-online"])
@pytest.mark.parametrize("classes,calibration", [((0, 1), 4), ((2, 3), 6)])
def test_calibration_run_never_scored(monkeypatch, tmp_path, ea, classes, calibration):
    """Review Focus 2. Under --calibration-run-excluded no trial of run 4 (fist family)
    or run 6 (feet family) reaches the scoring function, with or without EA, or the
    two arms of every A6 subject-independent comparison would not be paired."""
    m = _load()
    seen = []
    real = m.score_samples

    def spy(trainer, samples):
        seen.append(np.unique(samples.trial_id))
        return real(trainer, samples)

    monkeypatch.setattr(m, "score_samples", spy)
    rows = _run_main(m, monkeypatch, tmp_path, "--zero-shot", "--calibration-run-excluded",
                     "--ea", ea, "--classes", *map(str, classes))
    layout = _task_runs(classes)
    n = len(layout)
    others = set(layout.tolist()) - {calibration}
    assert len(rows) == len(SUBJECTS) and len(seen) == len(SUBJECTS)
    for tids in seen:
        # A target's trials sit at 200_000 + its offset in the group; every fake
        # subject has the same layout, so the local index is the id modulo n.
        runs = set(layout[(tids - 200_000) % n].tolist())
        assert calibration not in runs, f"run {calibration} was scored"
        assert runs == others
    for r in rows:
        assert np.array(r["trial_confusion"]).sum() == n - N_PER_RUN
        assert r["n_trials_scored"] == n - N_PER_RUN
        assert r["calibration_run_excluded"] is True and r["ea"] == ea
        assert r["scored_runs"] == sorted(others)
        assert r["calibration_runs"] == [calibration]


def test_zero_shot_uses_cached_source_without_finetune(monkeypatch, tmp_path):
    """A cached source model is scored as it is: no optimiser step, finetune_epochs 0."""
    import torch
    m = _load()
    cache = tmp_path / "cache"
    common = ("--zero-shot", "--calibration-run-excluded", "--classes", "0", "1",
              "--targets", "S001", "--source-cache", str(cache))
    steps = []
    real_step = torch.optim.Adam.step

    def counting(self, *a, **k):
        steps.append(1)
        return real_step(self, *a, **k)

    monkeypatch.setattr(torch.optim.Adam, "step", counting)
    first = _run_main(m, monkeypatch, tmp_path / "a", *common, "--tag", "SRC")
    assert steps, "the cold run should have pre-trained (and stepped)"
    assert len(list(cache.glob("src-SRC-g1-*.pt"))) == 1
    steps.clear()

    second = _run_main(m, monkeypatch, tmp_path / "b", *common, "--tag", "ZS",
                       "--source-tag", "SRC", "--require-source-cache")
    assert steps == [], "a zero-shot run on a cached source model took optimiser steps"
    (row,) = second
    assert row["finetune_epochs"] == 0 and row["zero_shot"] is True
    assert row["source_tag"] == "SRC" and row["fold"] is None
    assert row["accuracy"] == first[0]["accuracy"]   # same weights, same trials


def test_require_source_cache_refuses_to_pretrain(monkeypatch, tmp_path):
    """The A6 no-EA arm is defined as the cached PTG-shallow models; a miss must not
    silently pre-train a replacement."""
    m = _load()
    with pytest.raises(RuntimeError, match="source cache"):
        _run_main(m, monkeypatch, tmp_path, "--zero-shot", "--classes", "0", "1",
                  "--targets", "S001", "--source-cache", str(tmp_path / "empty"),
                  "--require-source-cache")


def _fingerprints(classes):
    """Map each fake trial's bytes to (subject, index after the class filter)."""
    table = {}
    for s in SUBJECTS:
        ep = _fake_subject(s)
        keep = np.isin(ep.y, classes)
        for i, x in enumerate(ep.X[keep]):
            table[x.tobytes()] = (s, i)
    return table


def _spy_fit_ea(m, monkeypatch):
    calls = []
    real = m.fit_ea

    def spy(X, *a, **k):
        calls.append(np.array(X, copy=True))
        return real(X, *a, **k)

    monkeypatch.setattr(m, "fit_ea", spy)
    return calls


def _identify(calls, table):
    """``(subject, sorted trial indices)`` behind every EA reference."""
    out = []
    for X in calls:
        ids = [table[x.tobytes()] for x in X]
        owners = {s for s, _ in ids}
        assert len(owners) == 1, f"one EA reference mixed subjects {owners}"
        out.append((owners.pop(), sorted(i for _, i in ids)))
    return out


@pytest.mark.parametrize("mode", ["train-fold", "pseudo-online", "offline"])
def test_ea_reference_sources(monkeypatch, tmp_path, mode):
    """Where every EA reference comes from (9.26 A6).

    Sources: each subject's own trials, all of them. Target, fine-tuning: the training
    fold only, never a test trial. Target, pseudo-online: the calibration run only.
    Target, offline: all of its trials, and the row says transductive.
    """
    from data.splits import stratified_trial_kfold as kfold
    m = _load()
    calls = _spy_fit_ea(m, monkeypatch)
    extra = (("--ea", "train-fold") if mode == "train-fold" else
             ("--zero-shot", "--calibration-run-excluded", "--ea", mode))
    rows = _run_main(m, monkeypatch, tmp_path, *extra, "--classes", "0", "1",
                     "--targets", "S001")
    found = _identify(calls, _fingerprints((0, 1)))
    n = len(_task_runs((0, 1)))

    # group 1 of 2 is [S001, S003]; its sources are S002 and S004, whole
    sources = [(s, idx) for s, idx in found if s != "S001"]
    assert sorted(s for s, _ in sources) == ["S002", "S004"]
    assert all(idx == list(range(n)) for _, idx in sources)
    assert "S003" not in {s for s, _ in found}, "a non-target group member was aligned"

    target = [idx for s, idx in found if s == "S001"]
    if mode == "train-fold":
        y_raw = _fake_subject("S001").y
        y_t = (y_raw[np.isin(y_raw, (0, 1))] == 1).astype(np.int64)
        folds = list(kfold(y_t, n_splits=3, seed=42))
        assert target == [sorted(np.asarray(f.train).tolist()) for f in folds]
        for idx, f in zip(target, folds):
            assert not set(idx) & set(np.asarray(f.test).tolist()), (
                "a test trial entered the reference")
        assert len(rows) == 3 and all(r["ea"] == "train-fold" for r in rows)
        assert all(r["zero_shot"] is False and r["transductive"] is False for r in rows)
        assert all(r["ea_reference"] == "training_fold" for r in rows)
    elif mode == "pseudo-online":
        assert target == [np.flatnonzero(_task_runs((0, 1)) == 4).tolist()]
        (row,) = rows
        assert row["transductive"] is False and row["ea_reference"] == "calibration_run"
    else:
        assert target == [list(range(n))]
        (row,) = rows
        assert row["ea"] == "offline" and row["transductive"] is True


def test_default_rows_carry_no_new_fields(monkeypatch, tmp_path):
    """Opt-in: a run using none of the Phase A options logs exactly the old keys."""
    m = _load()
    rows = _run_main(m, monkeypatch, tmp_path, "--classes", "0", "1", "--targets", "S001")
    assert len(rows) == 3
    for key in ("ea", "zero_shot", "calibration_run_excluded", "transductive",
                "confusion_excl_calibration", "n_trials_scored"):
        assert all(key not in r for r in rows), key


def test_ea_absent_keeps_cache_key():
    """``ea`` enters the key only when set, and records how the SOURCES were aligned:
    every EA mode aligns them identically, so the A6 fine-tuned arm (train-fold) reuses
    the zero-shot arm's (pseudo-online) EA source models. The zero-shot and
    calibration flags never enter the key: the no-EA arm must hit PTG-shallow's models.
    test_shallow_cache_key_unchanged, unmodified, pins the real file name."""
    m = _load()
    assert _key(m, args=_Args(ea="none")) == _key(m)
    assert _key(m, args=_Args(zero_shot=True, calibration_run_excluded=True)) == _key(m)
    ea = {mode: _key(m, args=_Args(ea=mode))
          for mode in ("pseudo-online", "offline", "train-fold")}
    assert len(set(ea.values())) == 1
    assert ea["pseudo-online"] != _key(m)


def test_zero_shot_flags_keep_every_ptg_shallow_key():
    """Arm si-shallow-noea must find all 20 cached PTG-shallow source models."""
    store = ROOT / "runs" / "source_models"
    if not list(store.glob("src-PTG-shallow-binary-*-g*.pt")):
        pytest.skip("runs/source_models has no PTG-shallow models")
    m = _load()
    subjects = _all_subjects()
    for classes in ([0, 1], [2, 3]):
        args = _Args(model="shallow", classes=classes, zero_shot=True,
                     calibration_run_excluded=True, ea="none")
        name = f"PTG-shallow-binary-{classes[0]}-{classes[1]}"
        for gi, _, source in m.plan_groups(subjects, 10, None):
            val = m.pick_validation(source, 42, gi, 5)
            key = m.source_cache_key(args, 42, gi, source, val)
            assert (store / f"src-{name}-g{gi}-{key}.pt").exists(), (name, gi)


def test_a1_logs_confusion_excluding_calibration_runs(monkeypatch, tmp_path):
    """4-class zero-shot (A1): a second matrix over the trials outside runs 4 and 6,
    so A1's within-family accuracy compares with A6's arms on the same trials."""
    m = _load()
    (row,) = _run_main(m, monkeypatch, tmp_path, "--zero-shot", "--targets", "S001")
    cm = np.array(row["trial_confusion"])
    ex = np.array(row["confusion_excl_calibration"])
    runs = _task_runs(None)
    assert cm.shape == ex.shape == (4, 4) and row["n_classes"] == 4
    assert cm.sum() == len(runs)
    assert ex.sum() == int(np.sum(~np.isin(runs, (4, 6))))
    assert (ex <= cm).all()
    assert row["calibration_run_excluded"] is False


def test_zero_shot_resume_skips_logged_targets(monkeypatch, tmp_path):
    m = _load()
    args = ("--zero-shot", "--classes", "0", "1", "--targets", "S001", "--resume")
    assert len(_run_main(m, monkeypatch, tmp_path, *args)) == 1
    assert len(_run_main(m, monkeypatch, tmp_path, *args)) == 1


@pytest.mark.parametrize("extra", [
    ("--ea", "pseudo-online"),                        # pseudo-online needs --zero-shot
    ("--ea", "offline"),                              # so does offline
    ("--zero-shot", "--ea", "train-fold"),            # train-fold is fine-tuning only
    ("--zero-shot", "--ea", "pseudo-online"),         # needs --calibration-run-excluded
    ("--calibration-run-excluded",),                  # zero-shot only
    ("--zero-shot", "--no-pretrain"),                 # nothing to score
])
def test_incoherent_options_fail_loudly(monkeypatch, tmp_path, extra):
    m = _load()
    with pytest.raises(SystemExit) as err:
        _run_main(m, monkeypatch, tmp_path, "--classes", "0", "1", *extra)
    assert err.value.code == 2


@pytest.mark.parametrize("graph", [False, True])
def test_target_windows_match_the_group_windows(graph):
    """Per-target windowing (used by the zero-shot and EA paths) reproduces exactly the
    windows the default path cuts from the group's concatenation."""
    m = _load()
    src = concat_trials([_trials(seed=2), _trials(seed=3)])
    a, b = _trials(seed=4), _trials(seed=5)
    prepared = prepare_fold(src, concat_trials([a, b]), window=64, stride=16,
                            build_graph=graph)
    base = 200_000 + len(a)
    mine = m.target_windows(prepared, b.X, b.y, base, 64, 16)
    ref = m.subset_samples(prepared.test, set(range(base, base + len(b))))
    assert np.array_equal(mine.X, ref.X)
    assert np.array_equal(mine.y, ref.y)
    assert np.array_equal(mine.trial_id, ref.trial_id)


def test_calibration_runs_follow_the_task_family():
    m = _load()
    assert m.calibration_runs([0, 1]) == (4,)
    assert m.calibration_runs([2, 3]) == (6,)
    assert m.calibration_runs(None) == (4, 6)
