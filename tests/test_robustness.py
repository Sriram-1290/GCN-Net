"""Tests for the test-time montage conditions (montage-robustness spec, sections 4 and 6).

Two failures here would corrupt the result silently, so each is pinned:

* **Graph left swapped after a condition (Review Focus 3).** The region model's
  basis and region map are persistent buffers; if a subset graph is still in
  place when the Full condition (or the next fold, or a saved state) is used, the
  number is wrong with no error. ``evaluate_conditions`` must restore the full
  graph after its loop, also when the loop raises.
* **Partial resume (Review Focus 4).** A fold whose clean row was logged but whose
  condition rows were not must be re-run, not skipped.

Everything is synthetic except the 64 channel names, which are the PhysioNet cap's
and are needed because the conditions are defined by electrode name and position.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluation.robustness as rob  # noqa: E402
from data.montage import CONDITIONS, common_average  # noqa: E402
from evaluation.pipeline import Trials, to_window_samples  # noqa: E402
from evaluation.trainer import TrainConfig, Trainer  # noqa: E402
from graphs.subset import moments, subset_representation  # noqa: E402
from models.graph_bandpower import GraphBandPowerConfig, GraphBandPowerNet  # noqa: E402

# The PhysioNet cap, in the cache's channel order.
CH = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2',
      'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2',
      'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6',
      'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1',
      'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2',
      'Iz']
WINDOW, STRIDE = 64, 16
CPU = TrainConfig(batch_size=256, device="cpu", verbose=False)


def _noref(n=6, t=96, seed=0) -> Trials:
    rng = np.random.default_rng(seed)
    # A shared offset per sample, as an un-referenced recording would carry.
    X = rng.normal(size=(n, 64, t)) + rng.normal(size=(n, 1, t)) * 3.0
    return Trials(X=X.astype(np.float32), y=np.array([0, 1] * (n // 2), dtype=np.int64))


def _source_moments():
    return moments(_noref(n=8, t=200, seed=1).X)


def _graph_trainer():
    m_src = _source_moments()
    full_rep = subset_representation(m_src, np.ones(64, dtype=bool), CH)
    torch.manual_seed(0)
    net = GraphBandPowerNet([full_rep.basis], WINDOW, GraphBandPowerConfig(readout="region"),
                            node_regions=full_rep.regions)
    return Trainer(net, CPU, n_classes=2), m_src, full_rep


def _full_input(test: Trials, full_rep):
    X = full_rep.standardizer.apply(common_average(test.X))
    return to_window_samples(X, test.y, WINDOW, STRIDE, trial_offset=0)


def _run(trainer, kind, test, m_src, full_rep, conditions=CONDITIONS,
         subject="S001", fold=0):
    std = full_rep.standardizer if full_rep is not None else \
        subset_representation(_source_moments(), np.ones(64, dtype=bool), CH).standardizer
    return rob.evaluate_conditions(
        trainer, kind, test, 0, CH, m_src, std, full_rep, subject, fold,
        tuple(conditions), WINDOW, STRIDE)


# ----------------------------------------------------- Review Focus 3: full graph back

def test_full_restored_after_conditions():
    trainer, m_src, full_rep = _graph_trainer()
    test = _noref()
    samples = _full_input(test, full_rep)
    before = trainer.predict_logits(samples)
    scores = _run(trainer, "graph", test, m_src, full_rep)
    after = trainer.predict_logits(samples)
    assert np.array_equal(before, after)
    assert trainer.model.n_nodes == 64
    assert np.array_equal(trainer.model.spatial[0].basis.cpu().numpy(), full_rep.basis)
    assert set(scores) == {c.name for c in CONDITIONS}


def test_full_restored_even_when_a_condition_raises(monkeypatch):
    trainer, m_src, full_rep = _graph_trainer()
    test = _noref()
    samples = _full_input(test, full_rep)
    before = trainer.predict_logits(samples)
    real = trainer.evaluate_samples
    calls = {"n": 0}

    def flaky(s):
        calls["n"] += 1
        if calls["n"] == 3:      # inside a subset condition, after set_graph
            raise RuntimeError("power cut")
        return real(s)

    monkeypatch.setattr(trainer, "evaluate_samples", flaky)
    with pytest.raises(RuntimeError, match="power cut"):
        _run(trainer, "graph", test, m_src, full_rep,
             conditions=[c for c in CONDITIONS if c.name != "full"])
    assert np.array_equal(trainer.predict_logits(samples), before)


def test_graph_refuses_moments_that_did_not_build_the_full_graph():
    """The subset graphs must come from the same source moments as the model's graph."""
    trainer, _, full_rep = _graph_trainer()
    other = moments(_noref(n=8, t=200, seed=7).X)
    with pytest.raises(ValueError, match="moments"):
        _run(trainer, "graph", _noref(), other, full_rep)


def test_graph_subset_input_is_the_subset_car_standardised_by_source():
    """What the graph model sees under a montage: CAR over the kept electrodes,
    then the subset standardiser derived from source moments."""
    trainer, m_src, full_rep = _graph_trainer()
    test = _noref()
    seen = []
    real = trainer.evaluate_samples

    def spy(s):
        seen.append(s.X.copy())
        return real(s)

    trainer.evaluate_samples = spy
    iv2a = next(c for c in CONDITIONS if c.name == "iv2a")
    _run(trainer, "graph", test, m_src, full_rep, conditions=[iv2a])
    keep = rob.available_masks(iv2a, CH, "S001", 0)[0]
    rep = subset_representation(m_src, keep, CH)
    expect = to_window_samples(rep.standardizer.apply(common_average(test.X, keep)),
                               test.y, WINDOW, STRIDE)
    assert len(seen) == 1 and np.array_equal(seen[0], expect.X)


# ------------------------------------------------------------- the shallow path

class _Stub(nn.Module):
    """ShallowConvNet-shaped: fixed 64 inputs, no set_graph."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(64, 2)
        self.shapes = []

    def forward(self, x):
        self.shapes.append(tuple(x.shape[1:]))
        return self.lin(x.mean(dim=2))


def test_shallow_path_never_calls_set_graph():
    torch.manual_seed(0)
    stub = _Stub()
    assert not hasattr(stub, "set_graph")
    trainer = Trainer(stub, CPU, n_classes=2)
    full_std = subset_representation(_source_moments(), np.ones(64, dtype=bool),
                                     CH).standardizer
    scores = rob.evaluate_conditions(trainer, "shallow", _noref(), 0, CH, None, full_std,
                                     None, "S001", 0, CONDITIONS, WINDOW, STRIDE)
    assert set(scores) == {c.name for c in CONDITIONS}
    for c in CONDITIONS:
        s = scores[c.name]
        n = 1 if c.kind in ("full", "montage") else 10
        assert s["n_draws"] == n and len(s["per_draw"]) == n
        assert s["accuracy"] == pytest.approx(np.mean(s["per_draw"]))
        assert 0.0 <= s["accuracy"] <= 1.0
    # every condition reached the network as the full 64-channel cap
    assert set(stub.shapes) == {(64, WINDOW)}


def test_shallow_input_is_the_repaired_cap_with_the_full_standardiser():
    from data.interpolate import interpolation_matrix, repair_full_cap
    torch.manual_seed(0)
    trainer = Trainer(_Stub(), CPU, n_classes=2)
    full_std = subset_representation(_source_moments(), np.ones(64, dtype=bool),
                                     CH).standardizer
    test = _noref()
    seen = []
    real = trainer.evaluate_samples

    def spy(s):
        seen.append(s.X.copy())
        return real(s)

    trainer.evaluate_samples = spy
    r25 = next(c for c in CONDITIONS if c.name == "random25")
    rob.evaluate_conditions(trainer, "shallow", test, 0, CH, None, full_std, None,
                            "S001", 0, (r25,), WINDOW, STRIDE)
    masks = rob.available_masks(r25, CH, "S001", 0)
    assert len(seen) == len(masks) == 10
    for X, keep in zip(seen, masks):
        rep = repair_full_cap(test.X, keep, interpolation_matrix(CH, keep))
        expect = to_window_samples(full_std.apply(rep), test.y, WINDOW, STRIDE)
        assert np.array_equal(X, expect.X)


# ------------------------------------------------------------- pairing across models

def test_paired_masks_across_kinds(monkeypatch):
    real = rob.available_masks
    drawn: dict[str, list] = {"graph": [], "shallow": []}
    current = {"kind": None}

    def spy(cond, ch_names, subject, fold, *a, **k):
        masks = real(cond, ch_names, subject, fold, *a, **k)
        drawn[current["kind"]].append((cond.name, [m.copy() for m in masks]))
        return masks

    monkeypatch.setattr(rob, "available_masks", spy)
    test = _noref()

    current["kind"] = "graph"
    trainer, m_src, full_rep = _graph_trainer()
    _run(trainer, "graph", test, m_src, full_rep, subject="S007", fold=3)

    current["kind"] = "shallow"
    _run(Trainer(_Stub(), CPU, n_classes=2), "shallow", test, None, None,
         subject="S007", fold=3)

    assert [n for n, _ in drawn["graph"]] == [c.name for c in CONDITIONS]
    assert len(drawn["graph"]) == len(drawn["shallow"])
    for (ng, mg), (ns, ms) in zip(drawn["graph"], drawn["shallow"]):
        assert ng == ns and len(mg) == len(ms)
        assert all(np.array_equal(a, b) for a, b in zip(mg, ms))


# ------------------------------------------------------- Review Focus 4: resume

def _write_log(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_resume_requires_all_condition_tags(tmp_path):
    names = [c.name for c in CONDITIONS if c.name != "full"]
    rows = [{"tag": "ROB", "subject": "S001", "fold": f} for f in (0, 1)]
    rows += [{"tag": f"ROB.{n}", "subject": "S001", "fold": 1} for n in names]
    # fold 0: every condition but random25 -- a power cut mid-evaluation
    rows += [{"tag": f"ROB.{n}", "subject": "S001", "fold": 0}
             for n in names if n != "random25"]
    log = tmp_path / "log.jsonl"
    _write_log(log, rows)

    done = rob.completed_with_conditions(log, "ROB", names)
    assert ("S001", 1) in done
    assert ("S001", 0) not in done

    runner = _load_runner()
    assert runner.resume_keys(log, "ROB", robustness=True) == done
    # without --robustness, only the clean rows count, exactly as before
    assert runner.resume_keys(log, "ROB", robustness=False) == {("S001", 0), ("S001", 1)}


def test_condition_rows_never_count_as_clean_rows(tmp_path):
    log = tmp_path / "log.jsonl"
    _write_log(log, [{"tag": "ROB.iv2a", "subject": "S001", "fold": 0}])
    assert rob.completed_with_conditions(log, "ROB", ["iv2a"]) == set()


# ------------------------------------------------------------- the logged rows

def test_condition_row_keeps_every_clean_field_and_replaces_the_metrics():
    clean = {"tag": "ROB-x", "subject": "S001", "fold": 2, "accuracy": 0.9,
             "trial_accuracy": 0.9, "timepoint_accuracy": 0.7,
             "trial_confusion": [[4, 0], [1, 4]], "trial_per_class": [1.0, 0.8],
             "trial_kappa": 0.8, "trial_macro_f1": 0.89, "history": {"a": 1},
             "model": "shallow"}
    score = {"accuracy": 0.6, "per_draw": [0.5, 0.7], "n_draws": 2,
             "timepoint_accuracy": 0.55, "confusion": [[6, 2], [4, 6]],
             "per_class": [0.75, 0.6], "kappa": 0.2, "macro_f1": 0.58}
    row = rob.condition_row(clean, "random25", score)
    assert set(clean) <= set(row)
    assert row["tag"] == "ROB-x.random25" and row["condition"] == "random25"
    assert row["accuracy"] == row["trial_accuracy"] == 0.6
    assert row["per_draw"] == [0.5, 0.7] and row["n_draws"] == 2
    assert row["timepoint_accuracy"] == 0.55
    assert row["trial_confusion"] == [[6, 2], [4, 6]]
    assert row["trial_per_class"] == [0.75, 0.6]
    assert row["trial_kappa"] == 0.2 and row["trial_macro_f1"] == 0.58
    assert row["subject"] == "S001" and row["fold"] == 2 and row["model"] == "shallow"
    assert clean["tag"] == "ROB-x" and clean["accuracy"] == 0.9   # not mutated


# ------------------------------------------------- runner: the no-reference trials

def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_pretrain_finetune", ROOT / "scripts" / "run_pretrain_finetune.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save(cache_dir, subject, X, y, ch_names):
    from data.cache import save_subject
    from data.edf_loader import SubjectEpochs
    n = len(y)
    save_subject(SubjectEpochs(subject=subject, X=X, y=y, run=np.full(n, 4),
                               onset=np.zeros(n, dtype=np.float32),
                               ch_names=list(ch_names), sfreq=160.0), cache_dir)


def test_noref_trials_are_class_filtered_like_the_referenced_ones(tmp_path):
    m = _load_runner()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3, 20)).astype(np.float32)
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    _save(tmp_path, "S001", X, y, ["C3", "Cz", "C4"])
    keep = np.isin(y, [0, 1])
    ref = Trials(X=X[keep], y=(y[keep] == 1).astype(np.int64))
    got = m.load_noref("S001", tmp_path, [0, 1], ref, ["C3", "Cz", "C4"])
    assert np.array_equal(got.X, X[keep]) and np.array_equal(got.y, ref.y)


def test_noref_trials_with_other_labels_fail_loudly(tmp_path):
    m = _load_runner()
    X = np.zeros((4, 3, 20), dtype=np.float32)
    _save(tmp_path, "S001", X, np.array([0, 1, 0, 1]), ["C3", "Cz", "C4"])
    ref = Trials(X=X, y=np.array([1, 0, 0, 1]))
    with pytest.raises(RuntimeError, match="labels"):
        m.load_noref("S001", tmp_path, None, ref, ["C3", "Cz", "C4"])
    with pytest.raises(RuntimeError, match="channel order"):
        m.load_noref("S001", tmp_path, None, Trials(X=X, y=np.array([0, 1, 0, 1])),
                     ["Cz", "C3", "C4"])
