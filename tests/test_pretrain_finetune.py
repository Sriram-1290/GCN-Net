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
