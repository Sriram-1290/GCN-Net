"""Do the numbers in the write-ups match the experiment log?

**Why this exists.** Two of the twenty errors in `project_documentation.md` 14 were
documentation errors that no amount of re-reading caught:

* **Error 20** — `PAPER.md` 5.4, `RESULTS.md` R4 and 12.4 each carried a table whose
  per-problem rows were 20-subject *development* values while its mean was the 103-subject
  figure. The rows did not average to the mean. It survived several readings and one PDF
  release, and was found only by noticing the arithmetic did not close.
* **Error 21** — an edit deleted the `## 6. Discussion` heading. The prose survived, so the
  document read correctly; the paper simply ran 5 → 7 for three commits.

Both are mechanical, and both are cheap to check. This module turns that check into a test
so the next one fails in CI rather than in a viva.

It deliberately does **not** re-derive the science. It asserts three things:

1. every headline accuracy quoted in the write-ups is reproducible from
   `runs/experiments.jsonl` to within rounding;
2. tables are internally consistent — a row labelled as a mean of other rows is one;
3. the documents are structurally intact — section numbering is contiguous, and every
   figure and reference they point at exists.

The tests skip rather than fail when the log is absent, so a fresh clone without the
experiment record still has a green suite.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "runs" / "experiments.jsonl"
DOCS = ("docs/PAPER.md", "docs/RESULTS.md", "README.md", "CLAUDE.md",
        "docs/project_documentation.md")

# Headline numbers that appear across the write-ups, with how to recompute each from the
# log. Tolerance is 5e-4: the documents quote four decimal places.
#   label -> (tag prefix, accuracy field, expected value)
HEADLINES = {
    "fixed graph":     ("C-fixed-binary", "trial_accuracy", 0.5438),
    "attention":       ("C-attention-binary", "trial_accuracy", 0.6038),
    "both":            ("C-both-binary", "trial_accuracy", 0.5976),
    "learnable graph": ("C-learngraph-binary", "trial_accuracy", 0.5469),
    "CSP+LDA":         ("C-csp", "accuracy", 0.6239),
    "A1 free spatial": ("A1-free-binary", "trial_accuracy", 0.5566),
    "A2 EEGNet":       ("A2-eegnet-binary", "trial_accuracy", 0.6296),
    "A2 ShallowConvNet": ("A2-shallow-binary", "trial_accuracy", 0.6725),
    "G1 graph constraint": ("G1-gbp-graph-binary", "trial_accuracy", 0.6629),
    "G2 free operator":    ("G2-gbp-free-binary", "trial_accuracy", 0.6694),
    "G3 graph as prior":   ("G3-gbp-prior-binary", "trial_accuracy", 0.6737),
    "PTC control":         ("PTC-gbp-binary", "trial_accuracy", 0.6744),
    "PTG pre-trained":     ("PTG-gbp-binary", "trial_accuracy", 0.7301),
    "PTC shallow control": ("PTC-shallow-binary", "trial_accuracy", 0.6886),
    "PTG shallow":         ("PTG-shallow-binary", "trial_accuracy", 0.7369),
    "CI CSP-init":         ("CI-cspinit-binary", "trial_accuracy", 0.6686),
    "A3 cov-tgsp":         ("A3-tgsp", "trial_accuracy", 0.6694),
    "PTD dsbn":            ("PTD-gbp-binary", "trial_accuracy", 0.7224),
}
TOL = 5e-4


def _rows():
    if not LOG.exists():
        pytest.skip(f"no experiment log at {LOG}")
    return [json.loads(l) for l in LOG.open(encoding="utf-8") if l.strip()]


def _per_subject_mean(rows, prefix: str, field: str) -> float:
    acc = defaultdict(list)
    for r in rows:
        if str(r.get("tag", "")).startswith(prefix) and r.get(field) is not None:
            acc[r["subject"]].append(float(r[field]))
    if not acc:
        pytest.skip(f"no rows for tag prefix {prefix!r}")
    return float(np.mean([np.mean(v) for v in acc.values()]))


# ------------------------------------------------- 1. numbers match the log

@pytest.mark.parametrize("label", sorted(HEADLINES))
def test_headline_accuracy_matches_the_log(label):
    """Every quoted headline is reproducible from the log to four decimal places."""
    prefix, field, expected = HEADLINES[label]
    actual = _per_subject_mean(_rows(), prefix, field)
    assert abs(actual - expected) < TOL, (
        f"{label}: documents quote {expected:.4f}, log gives {actual:.4f}. "
        "Either the documents are stale or an arm was re-run."
    )


@pytest.mark.parametrize("label", sorted(HEADLINES))
def test_headline_appears_in_at_least_one_write_up(label):
    """A number we verify should actually be quoted somewhere, or the check is dead."""
    _, _, expected = HEADLINES[label]
    needle = f"{expected:.4f}"
    assert any(needle in (ROOT / d).read_text(encoding="utf-8") for d in DOCS), (
        f"{label} ({needle}) is verified here but quoted in no write-up; "
        "drop it from HEADLINES or add it to the documents."
    )


# ---------------------------------------- 2. tables are internally consistent

@pytest.mark.parametrize("doc", DOCS)
def test_mean_rows_are_the_mean_of_their_table(doc):
    """A row labelled as a mean must equal the mean of the rows above it.

    This is exactly error 20: a table whose two problem rows were development-set
    values and whose mean row was the full-cohort figure. The arithmetic did not
    close, and nothing checked it.
    """
    text = (ROOT / doc).read_text(encoding="utf-8")
    failures = []
    for block in re.findall(r"(?:^\|.*\n)+", text, re.M):
        lines = [l for l in block.strip().splitlines() if l.startswith("|")]
        if len(lines) < 4:
            continue
        rows = []
        for l in lines:
            cells = [c.strip() for c in l.replace(r"\|", "\x00").split("|")[1:-1]]
            rows.append([c.replace("\x00", "|") for c in cells])
        # A summary row must (a) be labelled as a mean at the start or end of its
        # label — "fixed mean pool" is a configuration, not a summary — and (b) be
        # the last row, since a mean sits under the values it summarises.
        def is_summary(row):
            label = row[0].replace("*", "").strip().lower()
            return label.startswith("mean") or label.endswith("mean")

        mean_rows = [rows[-1]] if rows and rows[-1] and is_summary(rows[-1]) else []
        if not mean_rows:
            continue
        data = [r for r in rows[2:] if r not in mean_rows]
        for mrow in mean_rows:
            for col in range(1, min(len(mrow), 4)):
                got = _first_number(mrow[col])
                parts = [_first_number(r[col]) for r in data if col < len(r)]
                parts = [p for p in parts if p is not None]
                if got is None or len(parts) < 2:
                    continue
                if abs(got - float(np.mean(parts))) > 0.002:
                    failures.append(
                        f"{doc}: '{mrow[0]}' column {col} is {got:.4f} but the rows "
                        f"above average to {np.mean(parts):.4f}"
                    )
    assert not failures, "\n".join(failures)


def _first_number(cell: str) -> float | None:
    """First bare decimal in a cell, ignoring CIs in brackets and percentages."""
    cell = re.sub(r"\[.*?\]", "", cell)
    if "%" in cell:
        return None
    m = re.search(r"(?<![\d.])(0\.\d{3,4}|[−-]0\.\d{3,4})(?![\d%])", cell)
    if not m:
        return None
    return float(m.group(1).replace("−", "-"))


# --------------------------------------------- 3. documents are intact

@pytest.mark.parametrize("doc", ("docs/PAPER.md", "docs/project_documentation.md"))
def test_section_numbering_is_contiguous(doc):
    """No gaps in the top-level numbering — error 21 deleted a whole heading."""
    text = (ROOT / doc).read_text(encoding="utf-8")
    nums = [int(m) for m in re.findall(r"^## (\d+)\.", text, re.M)]
    assert nums, f"{doc} has no numbered sections"
    expected = list(range(nums[0], nums[0] + len(nums)))
    assert nums == expected, (
        f"{doc} section numbering is {nums}; expected {expected}. "
        "A heading was probably deleted or duplicated."
    )


@pytest.mark.parametrize("doc", DOCS)
def test_referenced_figures_exist(doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    missing = [m for m in re.findall(r"\(((?:\.\./)?figures/[^)]+)\)", text)
               if not ((ROOT / doc).parent / m).exists()]
    assert not missing, f"{doc} references missing figures: {missing}"


@pytest.mark.parametrize("doc", ("docs/PAPER.md",))
def test_every_citation_resolves(doc):
    """Every [n] in the body is defined, and every definition is cited."""
    text = (ROOT / doc).read_text(encoding="utf-8")
    body, refs = text.split("## References", 1)
    defined = {int(m) for m in re.findall(r"^\[(\d+)\]", refs, re.M)}
    cited = {int(n) for grp in re.findall(r"\[([\d,\s]+)\]", body)
             for n in grp.split(",") if n.strip().isdigit()}
    assert not cited - defined, f"cited but undefined: {sorted(cited - defined)}"
    assert not defined - cited, f"defined but never cited: {sorted(defined - cited)}"


# ------------------------------- 4. the one comparison that must never be made

@pytest.mark.parametrize("doc", DOCS)
def test_ptg_is_never_compared_to_the_within_session_ceiling(doc):
    """PTG sees ~92 other subjects; the 0.650 benchmark arms see only their own.

    9.15.4 forbids presenting PTG against that figure as if it were like-for-like.
    Any paragraph that mentions both must also carry the disclaimer, so the guard is:
    wherever 0.7206 and 0.650 appear near each other, a denial appears too.
    """
    text = (ROOT / doc).read_text(encoding="utf-8")
    for m in re.finditer(r"0\.7206", text):
        window = text[max(0, m.start() - 900):m.start() + 900]
        if "0.650" not in window:
            continue
        assert re.search(
            r"not\s+(a\s+)?like[- ]for[- ]like"
            r"|must not be drawn"
            r"|must (not|never) be (given|presented)"
            r"|never compare"
            r"|different protocol",
            window, re.I), (
            f"{doc} places PTG's 0.7206 beside the 0.650 within-session ceiling "
            "without the disclaimer required by 9.15.4."
        )
