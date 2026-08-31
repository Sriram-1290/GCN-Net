"""Unit tests for the annotation -> class mapping.

This mapping is the single highest-consequence piece of bookkeeping in the
pipeline: getting it wrong merges two classes and produces a plausible but
meaningless accuracy, with no error anywhere. It is worth testing exhaustively.
"""

from __future__ import annotations

import pytest

from data.labels import (
    CLASS_NAMES,
    FEET_RUNS,
    FIST_RUNS,
    IMAGERY_RUNS,
    LabelError,
    class_index,
    class_name,
)


def test_imagery_runs_are_the_documented_six():
    assert IMAGERY_RUNS == (4, 6, 8, 10, 12, 14)
    assert set(FIST_RUNS) | set(FEET_RUNS) == set(IMAGERY_RUNS)
    assert not set(FIST_RUNS) & set(FEET_RUNS)


def test_no_execution_run_is_mappable():
    """Runs 3/5/7/9/11/13 are executed movement and must never be loaded."""
    for run in (3, 5, 7, 9, 11, 13):
        with pytest.raises(LabelError):
            class_name(run, "T1")


@pytest.mark.parametrize("run", FIST_RUNS)
def test_fist_runs_map_to_left_and_right(run):
    assert class_name(run, "T1") == "left_fist"
    assert class_name(run, "T2") == "right_fist"


@pytest.mark.parametrize("run", FEET_RUNS)
def test_feet_runs_map_to_both_fists_and_both_feet(run):
    assert class_name(run, "T1") == "both_fists"
    assert class_name(run, "T2") == "both_feet"


def test_same_code_means_different_classes_across_run_families():
    """The regression this whole module exists to prevent."""
    assert class_name(4, "T1") != class_name(6, "T1")
    assert class_name(4, "T2") != class_name(6, "T2")


def test_rest_returns_none_rather_than_raising():
    for run in IMAGERY_RUNS:
        assert class_name(run, "T0") is None
        assert class_index(run, "T0") is None


def test_unknown_code_raises():
    with pytest.raises(LabelError):
        class_name(4, "T3")


def test_class_indices_cover_exactly_four_classes():
    seen = {class_index(run, code) for run in IMAGERY_RUNS for code in ("T1", "T2")}
    assert seen == {0, 1, 2, 3}
    assert len(CLASS_NAMES) == 4
