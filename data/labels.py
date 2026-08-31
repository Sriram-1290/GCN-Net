"""Mapping from PhysioNet EEGMMIDB annotations to class labels.

The single most important fact in this module: **the class is a function of
(run_number, T-code), never the T-code alone.** T1 means "left fist" in runs
4/8/12 but "both fists" in runs 6/10/14. Code that maps T1 -> class without
consulting the run number silently merges two different classes, which shows up
as a plausible-looking but meaningless accuracy.

Run assignment (confirmed against the MNE ``eegbci`` documentation and the
PhysioNet dataset description):

    runs 1, 2       baseline, eyes open / eyes closed
    runs 3, 7, 11   execution: left vs right fist
    runs 4, 8, 12   imagery:   left vs right fist
    runs 5, 9, 13   execution: both fists vs both feet
    runs 6, 10, 14  imagery:   both fists vs both feet
"""

from __future__ import annotations

# Motor-imagery runs only. The execution runs are deliberately absent.
IMAGERY_RUNS: tuple[int, ...] = (4, 6, 8, 10, 12, 14)

#: Imagery runs in which T1/T2 mean left/right fist.
FIST_RUNS: tuple[int, ...] = (4, 8, 12)
#: Imagery runs in which T1/T2 mean both fists / both feet.
FEET_RUNS: tuple[int, ...] = (6, 10, 14)

#: Class order for scheme (a). Index into this is the integer label.
CLASS_NAMES: tuple[str, ...] = ("left_fist", "right_fist", "both_fists", "both_feet")

REST_CODE = "T0"

# (run, code) -> class name. Built explicitly rather than with a clever expression
# so that the mapping is readable in a viva and greppable in a diff.
_RUN_CODE_TO_CLASS: dict[tuple[int, str], str] = {}
for _run in FIST_RUNS:
    _RUN_CODE_TO_CLASS[(_run, "T1")] = "left_fist"
    _RUN_CODE_TO_CLASS[(_run, "T2")] = "right_fist"
for _run in FEET_RUNS:
    _RUN_CODE_TO_CLASS[(_run, "T1")] = "both_fists"
    _RUN_CODE_TO_CLASS[(_run, "T2")] = "both_feet"

CLASS_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


class LabelError(ValueError):
    """Raised when a (run, code) pair cannot be mapped. Never silently skipped."""


def class_name(run: int, code: str) -> str | None:
    """Return the class name for an annotation, or ``None`` for rest (T0).

    ``None`` is returned only for T0, which callers drop under scheme (a). Any
    other unmappable pair raises rather than returning None, so a typo in a run
    list cannot quietly delete a whole class.
    """
    if code == REST_CODE:
        return None
    try:
        return _RUN_CODE_TO_CLASS[(run, code)]
    except KeyError:
        raise LabelError(
            f"cannot map annotation {code!r} in run {run}: "
            f"expected T0/T1/T2 in an imagery run {IMAGERY_RUNS}"
        ) from None


def class_index(run: int, code: str) -> int | None:
    """Integer label for an annotation, or ``None`` for rest."""
    name = class_name(run, code)
    return None if name is None else CLASS_TO_INDEX[name]
