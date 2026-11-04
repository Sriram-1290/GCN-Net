"""Montage robustness: M1 (graph model) against M2 (ShallowConvNet), 9.24.

    python scripts/robustness_results.py --m1 ROB1-chebrp --m2 ROB1-shallow
    python scripts/robustness_results.py --m1 ROB1-chebrp --m2 ROB1-shallow --json out.json

Implements the decision rule pre-registered in project_documentation.md 9.24.5, which
is binding and is not reinterpreted here. For a *primary* condition (iv2a, random25) a
**win** for M1 needs all four of

1. paired mean difference in within-family accuracy >= 0.020;
2. the 95 % bootstrap CI excludes zero (lower bound > 0);
3. Holm-corrected one-sided p < 0.05 across the two primaries (direction M1 > M2,
   fixed in advance);
4. M1 ahead on a strict majority of subjects.

The Full-montage guard: if M1 - M2 on Full is below -0.020, any would-be win is a
**trade-off**, not a win. A non-win with diff <= -0.020 and a CI wholly below zero is a
measured **loss** (9.24.5: "reported as measured, including a loss"); everything else is
**no effect**, with its direction reported.
Secondary conditions get the same numbers, labelled EXPLORATORY, and no decision.

Data layout (written by ``run_pretrain_finetune.py --robustness``): the clean rows of
tag ``<prefix>-binary-0-1`` / ``-2-3`` are the Full condition; condition rows carry the tag
``<that tag>.<condition>`` and a mean-over-draws accuracy. A subject's score is the mean
over the 5 folds of both problems (left/right fist and both fists/both feet), i.e. the
mean of the two per-problem subject means, so each problem counts equally. Pairing is on
subject.

**Completeness guard.** A partial run is never quoted (CLAUDE.md; one arm once read
0.5774 at 35 of 100 folds and 0.5418 when finished). The script refuses, with a non-zero
exit, when M1 and M2 cover different subject sets, when the subjects found are not exactly
the pre-registered development set (``--subjects``, default S001..S020, so 100 rows per
tag), or when any required tag lacks 5 folds for any subject. There is no
partial override: the point is that it cannot be talked round.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.montage import CONDITIONS, PRIMARY  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    bootstrap_p_value,
    holm_bonferroni,
    paired_bootstrap_ci,
)

DEFAULT_LOG = Path("runs/experiments.jsonl")
PROBLEMS = ("binary-0-1", "binary-2-3")
N_FOLDS = 5
#: The pre-registered development set (9.24.4, 9.24.8): S001..S020, 20 x 5 = 100 rows
#: per tag. The guard demands exactly this set, not merely that M1 and M2 agree.
DEV_SUBJECTS = tuple(f"S{i:03d}" for i in range(1, 21))
FIELD = "trial_accuracy"
MARGIN = 0.020          # 9.24.5 (1) and the Full guard
ALPHA = 0.05
_EPS = 1e-9             # float slack: 0.53 - 0.50 must still count as a 0.03 margin

#: Non-Full conditions, in the order the table prints them: primaries first.
CONDITION_NAMES = tuple(
    sorted((c.name for c in CONDITIONS if c.name != "full"),
           key=lambda n: (n not in PRIMARY, [c.name for c in CONDITIONS].index(n))))


def condition_tag(tag: str, condition: str) -> str:
    """``"<tag>.<condition>"``, as ``evaluation.robustness.condition_tag`` writes it.

    Re-stated rather than imported: that module pulls in torch, and this is a
    read-only analysis of a log. A test asserts it equals the original for every
    condition, so the writer and reader formats cannot drift.
    """
    return f"{tag}.{condition}"


class IncompleteRun(ValueError):
    """Raised when the log cannot support a quotable comparison."""


def load(log: Path) -> list[dict]:
    if not log.exists():
        raise SystemExit(f"no experiment log at {log}")
    return [json.loads(l) for l in log.open(encoding="utf-8") if l.strip()]


def _folds(rows: list[dict], tag: str) -> dict[tuple[str, int | None], float]:
    """``(subject, fold) -> accuracy`` for one exact tag; a re-run replaces its twin.

    Exact tag match: ``ROB1-x-binary-0-1`` is a string prefix of its own condition
    tags, so ``startswith`` would mix Full with every condition.
    """
    return {(r["subject"], r.get("fold")): float(r[FIELD]) for r in rows
            if r.get("tag") == tag and r.get(FIELD) is not None}


def _problem_scores(rows: list[dict], tag: str) -> dict[str, float]:
    """Per-subject mean over folds for one exact tag."""
    by_subject: dict[str, list[float]] = defaultdict(list)
    for (subject, _), v in _folds(rows, tag).items():
        by_subject[subject].append(v)
    return {s: float(np.mean(v)) for s, v in by_subject.items()}


def _arm_tags(prefix: str) -> list[str]:
    """Every tag an arm must hold: both problems x (clean + each condition)."""
    tags = []
    for problem in PROBLEMS:
        base = f"{prefix}-{problem}"
        tags.append(base)
        tags.extend(condition_tag(base, c) for c in CONDITION_NAMES)
    return tags


def _check_complete(rows: list[dict], m1: str, m2: str,
                    expected: tuple[str, ...]) -> list[str]:
    """Subjects scored, after refusing on any gap. Raises ``IncompleteRun``.

    Three refusals: M1 and M2 cover different subjects; the subjects found are not
    exactly the pre-registered set (an arm that is short on *both* sides would
    otherwise agree with itself and report a verdict on fewer than 20 subjects); any
    required tag lacks 5 folds for any expected subject.
    """
    def clean_subjects(prefix: str) -> set[str]:
        return {r["subject"] for r in rows
                if r.get("tag") in {f"{prefix}-{p}" for p in PROBLEMS}}

    s1, s2 = clean_subjects(m1), clean_subjects(m2)
    if s1 != s2:
        raise IncompleteRun(
            f"M1 and M2 cover different subjects: only M1 {sorted(s1 - s2)}, "
            f"only M2 {sorted(s2 - s1)}")
    if not s1:
        raise IncompleteRun(f"no rows for {m1!r} or {m2!r}")
    subjects = sorted(expected)
    if s1 != set(subjects):
        raise IncompleteRun(
            f"subjects found are not the expected {len(subjects)} "
            f"({subjects[0]}..{subjects[-1]}): missing {sorted(set(subjects) - s1)}, "
            f"unexpected {sorted(s1 - set(subjects))}")
    target = N_FOLDS * len(subjects)

    problems = []
    for prefix in (m1, m2):
        for tag in _arm_tags(prefix):
            seen = _folds(rows, tag)
            per_subject: dict[str, int] = defaultdict(int)
            for subject, _ in seen:
                per_subject[subject] += 1
            if (len(seen) != target or set(per_subject) != set(subjects)
                    or any(n != N_FOLDS for n in per_subject.values())):
                problems.append(f"{tag}: {len(seen)} of {target} rows")
    if problems:
        raise IncompleteRun(
            f"incomplete ({N_FOLDS} folds x {len(subjects)} subjects required per "
            "tag); refusing to report a partial run:\n  " + "\n  ".join(problems))
    return subjects


def _subject_scores(rows: list[dict], prefix: str, condition: str | None,
                    subjects: list[str]) -> np.ndarray:
    """Within-family score per subject: mean of the two problems' subject means."""
    per_problem = []
    for problem in PROBLEMS:
        tag = f"{prefix}-{problem}"
        if condition is not None:
            tag = condition_tag(tag, condition)
        means = _problem_scores(rows, tag)
        per_problem.append([means[s] for s in subjects])
    return np.mean(per_problem, axis=0)


def _compare(m1: np.ndarray, m2: np.ndarray) -> dict:
    diff, lo, hi = paired_bootstrap_ci(m1, m2)
    d = m1 - m2
    return {
        "m1": float(m1.mean()), "m2": float(m2.mean()), "diff": diff,
        "ci_lo": lo, "ci_hi": hi,
        "p": bootstrap_p_value(m1, m2, alternative="greater"),
        "wins": int((d > _EPS).sum()), "n": len(d),
    }


def robustness_table(rows: list[dict], m1_prefix: str, m2_prefix: str,
                     subjects: tuple[str, ...] = DEV_SUBJECTS) -> dict:
    """Per-condition M1/M2 means, paired difference, CI, p, Holm p, wins, decision.

    ``subjects`` is the exact set the log must hold (default: the pre-registered
    S001..S020). Raises ``IncompleteRun`` rather than reporting from a partial log.
    """
    subjects = _check_complete(rows, m1_prefix, m2_prefix, subjects)

    full = _compare(_subject_scores(rows, m1_prefix, None, subjects),
                    _subject_scores(rows, m2_prefix, None, subjects))
    guard_fires = full["diff"] < -MARGIN - _EPS
    guard = {"fires": bool(guard_fires), "diff": full["diff"], "threshold": -MARGIN}

    conditions = {}
    for name in CONDITION_NAMES:
        c = _compare(_subject_scores(rows, m1_prefix, name, subjects),
                     _subject_scores(rows, m2_prefix, name, subjects))
        c["role"] = "primary" if name in PRIMARY else "exploratory"
        conditions[name] = c

    holm = holm_bonferroni({n: conditions[n]["p"] for n in PRIMARY}, alpha=ALPHA)
    for name, c in conditions.items():
        if name not in PRIMARY:
            c["p_holm"], c["decision"], c["direction"] = None, None, None
            continue
        c["p_holm"] = holm[name]["p_adjusted"]
        criteria = {
            "margin": c["diff"] >= MARGIN - _EPS,
            "ci_excludes_zero": c["ci_lo"] > 0,
            "holm": c["p_holm"] < ALPHA,
            "majority": c["wins"] > c["n"] / 2,
        }
        c["criteria"] = criteria
        c["direction"] = "M1 ahead" if c["diff"] > _EPS else (
            "M1 behind" if c["diff"] < -_EPS else "level")
        if all(criteria.values()):
            c["decision"] = "trade-off" if guard_fires else "win"
        elif c["diff"] <= -MARGIN + _EPS and c["ci_hi"] < 0:
            # 9.24.5: "everything else is reported as measured, including a loss".
            # A detectable loss is a finding, not an absence of one. The decision
            # rule above is unchanged; this only names the non-win.
            c["decision"] = "loss"
        else:
            c["decision"] = "no effect"

    return {"m1": m1_prefix, "m2": m2_prefix, "subjects": subjects,
            "n_subjects": len(subjects), "full": full, "guard": guard,
            "conditions": conditions}


def format_markdown(result: dict) -> str:
    def line(label, c, decision):
        holm = "-" if c.get("p_holm") is None else f"{c['p_holm']:.4f}"
        return (f"| {label} | {c['m1']:.4f} | {c['m2']:.4f} | {c['diff']:+.4f} | "
                f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | {c['p']:.4f} | {holm} | "
                f"{c['wins']}/{c['n']} | {decision} |")

    out = [f"M1 = {result['m1']}   M2 = {result['m2']}   "
           f"{result['n_subjects']} subjects (within-family accuracy, paired on subject; "
           "one-sided p, M1 > M2)", "",
           "| condition | M1 | M2 | diff | 95 % CI | p | Holm p | wins/n | decision |",
           "|---|---|---|---|---|---|---|---|---|",
           line("Full (guard)", result["full"],
                "guard fires" if result["guard"]["fires"] else "guard clear")]
    for name, c in result["conditions"].items():
        if c["role"] == "primary":
            decision = c["decision"] + (
                f" ({c['direction']})" if c["decision"] in ("no effect", "loss") else "")
            out.append(line(f"{name} (primary)", c, decision))
        else:
            out.append(line(f"{name} (EXPLORATORY)", c, "none (exploratory)"))
    g = result["guard"]
    out += ["", f"Full-montage guard: M1 - M2 = {g['diff']:+.4f} against a threshold of "
            f"{g['threshold']:+.3f}: " + ("FIRES -- a would-be win is reported as a "
            "trade-off." if g["fires"] else "does not fire.")]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m1", required=True, help="M1 tag prefix, e.g. ROB1-chebrp")
    ap.add_argument("--m2", required=True, help="M2 tag prefix, e.g. ROB1-shallow")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--subjects", nargs="+", default=list(DEV_SUBJECTS),
                    help="the exact subject set the log must hold (default: the "
                         "pre-registered development set S001..S020)")
    ap.add_argument("--json", type=Path, default=None, help="also write the result here")
    args = ap.parse_args()

    try:
        result = robustness_table(load(args.log), args.m1, args.m2,
                                  tuple(args.subjects))
    except IncompleteRun as e:
        print(f"refusing to report: {e}", file=sys.stderr)
        return 1

    print(format_markdown(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
