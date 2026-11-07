"""The Phase A results tables, A1-A6, exactly as project_documentation.md 9.26 defines them.

    python scripts/phase_a_results.py a6
    python scripts/phase_a_results.py a1 --tgsp-only
    python scripts/phase_a_results.py a3 --allow-partial     # watermarked progress view

Each experiment prints one Markdown report. Every quantity, margin and reading is the one
9.26 fixed before any ``PA-`` row existed, including its dated amendments (A3 drift-opposed
split (iii), A4 drift note, the A5 ``split`` bundling and its ``none`` gate procedure, the A6
Holm and equivalence wording, the A1 ``confusion_excl_calibration`` comparison). Nothing here
is a new analysis choice; where 9.26 left a detail open it is listed in the module notes at
the bottom and in the Task 8 report.

**Completeness guard.** CLAUDE.md: "Never quote a partial run" (one arm read 0.5774 at 35 of
100 folds and 0.5418 completed). The script refuses, with a non-zero exit and a message
naming the tag and its row count, whenever a *primary* tag holds fewer than its full rows:

* 5 folds x 103 subjects = 515 for 5-fold tags;
* 3 folds x 103 = 309 for run-disjoint (``run``, ``runopp``) tags;
* 103 for zero-shot, timepoint-holdout (A5 ``split``/``all``) and A1 tags.

Secondary tags (A3/A4 ``runopp``, A6 ``*-offline``, and the A6 ``noea`` comparator inside A1)
print when complete and read "not run" otherwise; they never block the primary table.

``--allow-partial`` exists only to watch a run in progress. It prints the subjects that are
complete in every primary tag, under a ``PARTIAL — NOT A RESULT`` banner, with every verdict,
reading and prediction withheld. Nothing it prints may be quoted.

Duplicate (tag, subject, fold) rows are last-wins, as ``--resume`` and
``robustness_results.py`` treat them, and are listed in a note.

Statistics are the project's: paired bootstrap over subjects, 10,000 resamples, 95 % CI,
per-subject win counts; an *effect* is a difference of at least 0.020 whose CI excludes zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, namedtuple
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.config import load_config  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    bootstrap_p_value,
    holm_bonferroni,
    paired_bootstrap_ci,
)
from scripts.run_family_analysis import decompose  # noqa: E402

DEFAULT_LOG = ROOT / "runs" / "experiments.jsonl"
CONFIG = ROOT / "configs" / "bandpass_8_30.yaml"
EXPERIMENTS = ("a1", "a2", "a3", "a4", "a5", "a6")

MARGIN = 0.020            # the 9.12 floor, 9.26.2 "Statistics"
ALPHA = 0.05
EPS = 1e-9                # 0.53 - 0.50 must still count as a 0.03 margin
PROBLEMS = ("0-1", "2-3")  # left vs right fist; both fists vs both feet
PARTIAL = "PARTIAL — NOT A RESULT"

# A5 validity gates (9.26.3, revised 2026-11-07 in 4b2e702): logged three-seed 95 % CIs.
GATE_ALL = (0.8794, 0.9147)    # `all` mean timepoint accuracy ~ P1-leaky-103-* 0.8975
GATE_NONE = (0.4984, 0.5199)   # `none` within-family ~ P1-A-103-* 0.5090
A5_ARMS = ("none", "split", "bn", "graph", "all")
A5_HOLDOUT = ("split", "all")  # random timepoint holdout: one row per subject, fold null

Plan = namedtuple("Plan", "primary secondary")     # lists of (tag, folds); folds None = 1 row
Report = namedtuple("Report", "text exit_code")


class IncompleteRun(ValueError):
    """The log cannot support a quotable result."""


# ------------------------------------------------------------------- the log
def default_subjects() -> list[str]:
    """The 103 non-excluded subjects: S001..S109 minus the configured exclusions."""
    cfg = load_config(CONFIG)
    return [f"S{i:03d}" for i in range(1, 110)
            if f"S{i:03d}" not in cfg.dataset.exclude_subjects]


def load_rows(path: Path) -> list[dict]:
    """Read the JSONL log. A final line cut off by a job mid-append is skipped."""
    if not Path(path).exists():
        raise SystemExit(f"no experiment log at {path}")
    lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for k, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if k != len(lines) - 1:
                raise
            print("note: ignored an unfinished last line in the log", file=sys.stderr)
    return rows


class Log:
    """Rows indexed ``tag -> (subject, fold) -> row``; a later duplicate replaces an earlier."""

    def __init__(self, rows: list[dict], subjects):
        self.subjects = list(subjects)
        self._rows: dict[str, dict] = defaultdict(dict)
        self.surplus: dict[str, int] = defaultdict(int)
        for r in rows:
            tag = r.get("tag")
            if not tag:
                continue
            key = (r.get("subject"), r.get("fold"))
            if key in self._rows[tag]:
                self.surplus[tag] += 1
            self._rows[tag][key] = r
        self._by_subject: dict[str, dict] = {}

    def restrict(self, subjects) -> "Log":
        out = object.__new__(Log)
        out.subjects = list(subjects)
        out._rows, out.surplus, out._by_subject = self._rows, self.surplus, {}
        return out

    def rows_for(self, tag: str) -> dict:
        return self._rows.get(tag, {})

    def status(self, tag: str, folds: int | None) -> tuple[int, int, bool]:
        """``(rows held, rows needed, complete)``: complete = exactly subjects x folds."""
        want = {(s, f) for s in self.subjects for f in (range(folds) if folds else [None])}
        have = set(self.rows_for(tag))
        return len(have), len(want), have == want

    def complete_subjects(self, tag: str, folds: int | None) -> set[str]:
        have = set(self.rows_for(tag))
        need = list(range(folds)) if folds else [None]
        return {s for s in self.subjects if all((s, f) in have for f in need)}

    def subject_rows(self, tag: str) -> dict[str, list[dict]]:
        if tag not in self._by_subject:
            idx: dict[str, list[dict]] = defaultdict(list)
            for (s, _), r in self.rows_for(tag).items():
                idx[s].append(r)
            self._by_subject[tag] = idx
        return self._by_subject[tag]


# ------------------------------------------------------------ the tag plans
def a6_tag(arm: str, problem: str) -> str:
    return (f"PTG-shallow-binary-{problem}" if arm == "PTG-shallow"
            else f"PA-A6-{arm}-binary-{problem}")


def plan(experiment: str, tgsp_only: bool = False) -> Plan:
    """Every tag an experiment reads, with its fold count (None = one row per subject)."""
    if experiment == "a1":
        models = ("tgsp",) if tgsp_only else ("shallow", "tgsp")
        return Plan([(f"PA-A1-{m}-4class", None) for m in models],
                    [(a6_tag(f"si-{m}-noea", p), None) for m in models for p in PROBLEMS])
    if experiment == "a2":
        return Plan([(f"PA-A2-shallow-ws160-binary-{p}", 5) for p in PROBLEMS]
                    + [(f"A2-shallow-binary-{p}", 5) for p in PROBLEMS], [])
    if experiment == "a3":
        return Plan([(f"PA-A3-rest-{m}-{s}-{c}", 5 if s == "trial" else 3)
                     for m in ("tgsp", "shallow") for s in ("trial", "run")
                     for c in ("bp830", "car")],
                    [(f"PA-A3-rest-tgsp-runopp-{c}", 3) for c in ("bp830", "car")])
    if experiment == "a4":
        return Plan([(f"PA-A4-{t}-{m}-{s}", 5 if s == "trial" else 3)
                     for t in ("rh_vs_feet", "lr", "bf") for m in ("tgsp", "shallow")
                     for s in ("trial", "run")],
                    [("PA-A4-rh_vs_feet-tgsp-runopp", 3)])
    if experiment == "a5":
        return Plan([(f"PA-A5-leak-{a}", None if a in A5_HOLDOUT else 5) for a in A5_ARMS], [])
    if experiment == "a6":
        primary = [(a6_tag(a, p), None) for a in ("si-shallow-noea", "si-shallow-ea",
                                                    "si-tgsp-noea", "si-tgsp-ea")
                   for p in PROBLEMS]
        primary += [(a6_tag(a, p), 5) for a in ("ft-shallow-ea", "PTG-shallow")
                    for p in PROBLEMS]
        return Plan(primary, [(a6_tag(f"si-{m}-ea-offline", p), None)
                              for m in ("shallow", "tgsp") for p in PROBLEMS])
    raise ValueError(f"unknown experiment {experiment!r}")


# --------------------------------------------------------- per-subject scores
def per_subject(log: Log, tag: str, field: str) -> np.ndarray:
    """Mean of ``field`` over a subject's rows (its folds), in ``log.subjects`` order."""
    idx = log.subject_rows(tag)
    out = []
    for s in log.subjects:
        if s not in idx:
            raise IncompleteRun(f"{tag}: no rows for {s}")
        vals = []
        for r in idx[s]:
            if r.get(field) is None:
                raise IncompleteRun(f"{tag}: a row for {s} has no {field!r}")
            vals.append(float(r[field]))
        out.append(np.mean(vals))
    return np.array(out)


def confusions(log: Log, tag: str, key: str, required: bool = True):
    """Per-subject confusion matrices summed over the subject's folds: ``(n, k, k)``."""
    idx = log.subject_rows(tag)
    out = []
    for s in log.subjects:
        if s not in idx:
            raise IncompleteRun(f"{tag}: no rows for {s}")
        cms = [r.get(key) for r in idx[s]]
        if any(c is None for c in cms):
            if required:
                raise IncompleteRun(f"{tag}: a row for {s} has no {key!r}")
            return None
        out.append(np.sum([np.asarray(c, dtype=int) for c in cms], axis=0))
    return np.array(out)


def decompose_each(cms: np.ndarray) -> dict[str, np.ndarray]:
    """``decompose`` per subject, as arrays keyed like its result."""
    ds = [decompose(c) for c in cms]
    return {k: np.array([d[k] for d in ds])
            for k in ("accuracy_4class", "family_accuracy", "within_family_accuracy")}


def within_family_by_subject(log: Log, tag: str, key: str = "trial_confusion") -> np.ndarray:
    """The A5 ``none`` gate procedure: per-subject pooled confusion -> ``decompose``."""
    return decompose_each(confusions(log, tag, key))["within_family_accuracy"]


def binary_arm(log: Log, template: str, field: str = "trial_accuracy") -> dict[str, np.ndarray]:
    """Per-subject scores for a binary arm: each problem alone, and their mean.

    ``template`` holds ``{p}``. A problem's subject score is the mean over the subject's
    rows (folds); the subject's score is the mean of the two problems. This is how the
    logged A2 0.6725 and PTG-shallow 0.7369 are formed (``confirmatory_results``).
    """
    arm = {p: per_subject(log, template.format(p=p), field) for p in PROBLEMS}
    arm["mean"] = np.mean([arm[p] for p in PROBLEMS], axis=0)
    return arm


# ----------------------------------------------------------------- statistics
def compare(a, b, alternative: str = "two-sided", with_p: bool = True) -> dict:
    """Paired ``a - b`` over subjects: means, diff, 95 % CI, p, win/loss counts."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    diff, lo, hi = paired_bootstrap_ci(a, b)
    d = a - b
    return {"a": float(a.mean()), "b": float(b.mean()), "diff": diff, "ci_lo": lo,
            "ci_hi": hi,
            "p": bootstrap_p_value(a, b, alternative) if with_p else None,
            "wins": int((d > EPS).sum()), "losses": int((d < -EPS).sum()), "n": int(len(d))}


def boot_mean(v) -> dict:
    """Mean of one per-subject vector with a bootstrap 95 % CI."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    m, lo, hi = paired_bootstrap_ci(v, np.zeros_like(v))
    return {"mean": m, "lo": lo, "hi": hi, "n": int(len(v))}


def classify(diff: float, lo: float, hi: float) -> tuple[str, str | None]:
    """The project's effect rule: >= 0.020 with the 95 % CI excluding zero."""
    if diff >= MARGIN - EPS and lo > 0:
        return "effect", "+"
    if diff <= -MARGIN + EPS and hi < 0:
        return "effect", "-"
    return "no effect", None


def chance_reading(v, chance: float = 0.5) -> dict:
    """Above / at / below chance: the mean differs by >= 0.020 and the CI excludes chance."""
    m = boot_mean(np.asarray(v, float) - chance)
    mean, lo, hi = m["mean"] + chance, m["lo"] + chance, m["hi"] + chance
    if m["mean"] >= MARGIN - EPS and m["lo"] > 0:
        reading = "above chance"
    elif m["mean"] <= -MARGIN + EPS and m["hi"] < 0:
        reading = "below chance"
    else:
        reading = "at chance"
    return {"mean": mean, "lo": lo, "hi": hi, "n": m["n"], "reading": reading}


# ----------------------------------------------------------------- formatting
def f4(x) -> str:
    return f"{x:.4f}"


def sg(x) -> str:
    return f"{x:+.4f}"


def ci(lo, hi, signed=True) -> str:
    return f"[{lo:+.4f}, {hi:+.4f}]" if signed else f"[{lo:.4f}, {hi:.4f}]"


def md(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _mask(partial: bool, text: str) -> str:
    return "PARTIAL" if partial else text


def _compare_row(label, c, verdict="") -> list:
    p = "-" if c["p"] is None else f"{c['p']:.4f}"
    return [label, f4(c["a"]), f4(c["b"]), sg(c["diff"]), ci(c["ci_lo"], c["ci_hi"]), p,
            f"{c['wins']}/{c['n']}", verdict]


def _not_run(log: Log, tag: str, folds) -> str:
    have, need, _ = log.status(tag, folds)
    return f"{tag}: not run ({have} of {need} rows; not quoted)"


# ================================================================== A2
def a2_results(log: Log) -> dict:
    """(ii) ws160 minus (i) all windows, within-family, per-subject mean of two problems."""
    arms = {"all": binary_arm(log, "A2-shallow-binary-{p}"),
            "ws160": binary_arm(log, "PA-A2-shallow-ws160-binary-{p}")}
    c = compare(arms["ws160"]["mean"], arms["all"]["mean"])
    per_problem = {p: compare(arms["ws160"][p], arms["all"][p], with_p=False)
                   for p in PROBLEMS}
    if c["diff"] <= -MARGIN + EPS and c["ci_hi"] < 0:
        verdict = ("drop: leans on early-cue activity (an upper bound: arm (ii) trains on "
                   "about 73 % of the windows)")
    elif c["diff"] >= MARGIN - EPS and c["ci_lo"] > 0:
        verdict = "rise: early windows hurt the model (not a success of the exclusion)"
    elif abs(c["diff"]) < MARGIN:
        verdict = "no measurable dependence on the first second"
    else:
        verdict = "indistinguishable: |difference| >= 0.020 but the CI includes zero"
    return {"arms": arms, "contrast": c, "per_problem": per_problem, "verdict": verdict}


def format_a2(res: dict, partial: bool) -> list[str]:
    arms, c = res["arms"], res["contrast"]
    out = ["Within-family accuracy, ShallowConvNet, subject-dependent; subject score = mean "
           "of the two binary problems' fold means.", ""]
    out += md(["arm", "0-1", "2-3", "mean", "95 % CI of mean"],
              [[name, f4(a["0-1"].mean()), f4(a["2-3"].mean()), f4(a["mean"].mean()),
                ci(*[boot_mean(a["mean"])[k] for k in ("lo", "hi")], signed=False)]
               for name, a in (("(i) all windows (A2-shallow-binary, reused)", arms["all"]),
                               ("(ii) windows from 160 samples (ws160)", arms["ws160"]))])
    out += ["", "Contrast (ii) - (i), paired over subjects:", ""]
    rows = [_compare_row("mean of both problems", c, "")]
    rows += [_compare_row(f"problem {p} alone", pc) for p, pc in res["per_problem"].items()]
    out += md(["", "(ii)", "(i)", "diff", "95 % CI", "p (two-sided)", "(ii) ahead", ""], rows)
    out += ["", f"Reading: {_mask(partial, res['verdict'])}"]
    return out


# ================================================================== A3
def a3_results(log: Log) -> dict:
    models: dict = {}
    secondary: dict = {}
    for model in ("tgsp", "shallow"):
        models[model] = {}
        for cache in ("bp830", "car"):
            sc = {s: per_subject(log, f"PA-A3-rest-{model}-{s}-{cache}", "accuracy")
                  for s in ("trial", "run")}
            cell = {s: chance_reading(v) for s, v in sc.items()}
            cell["gap"] = compare(sc["trial"], sc["run"])              # (i) - (ii)
            if model == "tgsp":
                tag = f"PA-A3-rest-tgsp-runopp-{cache}"
                if log.status(tag, 3)[2]:
                    opp = per_subject(log, tag, "accuracy")
                    cell["runopp"] = chance_reading(opp)
                    # (ii) and (iii) share the signal that does not depend on time
                    cell["drift_free"] = chance_reading((sc["run"] + opp) / 2)
                    cell["drift"] = boot_mean((sc["run"] - opp) / 2)
                else:
                    secondary[tag] = _not_run(log, tag, 3)
            models[model][cache] = cell
    return {"models": models, "secondary": secondary}


def a3_scenario(trial: str, run: str, opp: str | None = None) -> str:
    """The pre-registered reading of (i), (ii) and, for tangent space, (iii)."""
    if trial != "above chance":
        return f"(i) {trial}: no detectable run-level signature under this model"
    if run == "below chance":
        return ("(i) above, (ii) below chance: signatures that do not transfer or that "
                "invert on unseen runs; never read as family information")
    if run != "above chance":
        return ("(i) above, (ii) at chance: run-specific signatures that do not generalise "
                "(run identity, not family)")
    if opp is None:
        return ("(i) and (ii) above chance: family decodable on unseen runs; drift not yet "
                "separated (needs (iii))")
    if opp == "above chance":
        return ("(i), (ii) and (iii) above chance: family decodable on unseen runs without "
                "any trial label (supported)")
    return (f"(ii) above but (iii) {opp}: consistent with slow within-session drift; not "
            "family information")


def format_a3(res: dict, partial: bool) -> list[str]:
    out = ["Run family (runs 4/8/12 vs 6/10/14) from REST epochs; accuracy per subject "
           "(mean over folds), chance 0.5. Above/below chance = differs from 0.5 by at "
           "least 0.020 with the 95 % CI excluding 0.5.", "",
           "Caveat (pre-stated): every T0 except a run's first follows an imagery trial, so "
           "family decoding may reflect neural after-effects as well as non-neural "
           "artefacts; A3 shows family is identifiable without labels, not that it is "
           "non-neural.", ""]
    rows = []
    for model, caches in res["models"].items():
        for cache, cell in caches.items():
            for split, label in (("trial", "(i) trial-disjoint"), ("run", "(ii) run-disjoint"),
                                 ("runopp", "(iii) drift-opposed run")):
                if split not in cell:
                    continue
                m = cell[split]
                rows.append([model, cache, label, m["n"], f4(m["mean"]),
                             ci(m["lo"], m["hi"], signed=False),
                             _mask(partial, m["reading"])])
    out += md(["model", "cache", "split", "n", "mean acc", "95 % CI", "vs chance"], rows)
    out += ["", "Run-specific part, (i) - (ii):", ""]
    out += md(["model", "cache", "(i)", "(ii)", "(i) - (ii)", "95 % CI", "p", "(i) ahead"],
              [[m, c, f4(cell["gap"]["a"]), f4(cell["gap"]["b"]), sg(cell["gap"]["diff"]),
                ci(cell["gap"]["ci_lo"], cell["gap"]["ci_hi"]), f"{cell['gap']['p']:.4f}",
                f"{cell['gap']['wins']}/{cell['gap']['n']}"]
               for m, caches in res["models"].items() for c, cell in caches.items()])
    drift = [(c, cell) for c, cell in res["models"]["tgsp"].items() if "drift" in cell]
    out += ["", "Drift decomposition, tangent space ((ii) run-disjoint, (iii) drift-opposed): "
            "drift-free part = mean of (ii) and (iii); drift part = half their difference.", ""]
    if drift:
        out += md(["cache", "drift-free part", "95 % CI", "vs chance", "drift part",
                   "95 % CI"],
                  [[c, f4(cell["drift_free"]["mean"]),
                    ci(cell["drift_free"]["lo"], cell["drift_free"]["hi"], signed=False),
                    _mask(partial, cell["drift_free"]["reading"]), sg(cell["drift"]["mean"]),
                    ci(cell["drift"]["lo"], cell["drift"]["hi"])] for c, cell in drift])
    for msg in res["secondary"].values():
        out.append(f"- secondary (iii): {msg}")
    if not partial:
        out += ["", "Readings (9.26.5, fixed before the run; models reported separately, "
                "never pooled):", ""]
        for model, caches in res["models"].items():
            for cache, cell in caches.items():
                opp = cell["runopp"]["reading"] if "runopp" in cell else None
                out.append(f"- {model} / {cache}: " + a3_scenario(
                    cell["trial"]["reading"], cell["run"]["reading"], opp))
    return out


# ================================================================== A4
TASKS = ("rh_vs_feet", "lr", "bf")


def a4_results(log: Log) -> dict:
    models: dict = {}
    for model in ("tgsp", "shallow"):
        sc = {(t, s, f): per_subject(log, f"PA-A4-{t}-{model}-{s}", f)
              for t in TASKS for s in ("trial", "run")
              for f in ("roc_auc", "accuracy")}
        out: dict = {}
        for name, field in (("auc", "roc_auc"), ("accuracy", "accuracy")):
            d = {t: sc[(t, "trial", field)] - sc[(t, "run", field)] for t in TASKS}
            infl = d["rh_vs_feet"] - (d["lr"] + d["bf"]) / 2
            m = boot_mean(infl)
            if m["mean"] >= MARGIN - EPS and m["lo"] > 0:
                verdict = "inflation"
            elif m["mean"] <= -MARGIN + EPS and m["hi"] < 0:
                verdict = "reversed"
            elif abs(m["mean"]) < MARGIN:
                verdict = "no extra inflation"
            else:
                verdict = "indistinguishable"
            out[name] = {"estimate": m["mean"], "ci_lo": m["lo"], "ci_hi": m["hi"],
                         "n": m["n"], "verdict": verdict,
                         "wins": int((infl > EPS).sum())}
            if name == "auc":
                out["drops"] = {t: float(d[t].mean()) for t in TASKS}
        out["cells"] = {(t, s): {"auc": float(sc[(t, s, "roc_auc")].mean()),
                                 "accuracy": float(sc[(t, s, "accuracy")].mean())}
                        for t in TASKS for s in ("trial", "run")}
        models[model] = out
    tag = "PA-A4-rh_vs_feet-tgsp-runopp"
    secondary = {}
    if log.status(tag, 3)[2]:
        auc_o, acc_o = per_subject(log, tag, "roc_auc"), per_subject(log, tag, "accuracy")
        auc_r = per_subject(log, "PA-A4-rh_vs_feet-tgsp-run", "roc_auc")
        acc_r = per_subject(log, "PA-A4-rh_vs_feet-tgsp-run", "accuracy")
        models["tgsp"]["runopp"] = {
            "auc_runopp": float(auc_o.mean()), "acc_runopp": float(acc_o.mean()),
            "auc_drift_free": boot_mean((auc_r + auc_o) / 2),
            "auc_drift": boot_mean((auc_r - auc_o) / 2),
            "acc_drift_free": boot_mean((acc_r + acc_o) / 2),
            "acc_drift": boot_mean((acc_r - acc_o) / 2)}
    else:
        secondary[tag] = _not_run(log, tag, 3)
    return {"models": models, "secondary": secondary}


_A4_READING = {
    "inflation": "estimate >= 0.020 with the CI excluding zero: the cross-family task loses "
                 "more from run-disjoint splitting than the within-family controls, so part "
                 "of its trial-split score is run-family identity (a lower bound: drift can "
                 "still raise AUC_run)",
    "no extra inflation": "|estimate| < 0.020: no measurable extra inflation; the "
                          "run-disjoint cost is generic",
    "reversed": "estimate <= -0.020: the controls lose more than the cross-family task; "
                "check for a run-pair imbalance before any reading",
    "indistinguishable": "|estimate| >= 0.020 but the CI includes zero: indistinguishable",
}


def format_a4(res: dict, partial: bool) -> list[str]:
    out = ["Cross-family task rh_vs_feet vs within-family controls lr and bf; ROC-AUC "
           "(primary) and accuracy (secondary) per subject, mean over folds.", ""]
    rows = []
    for model, m in res["models"].items():
        for task in TASKS:
            tr, ru = m["cells"][(task, "trial")], m["cells"][(task, "run")]
            rows.append([model, task, f4(tr["auc"]), f4(ru["auc"]),
                         sg(tr["auc"] - ru["auc"]), f4(tr["accuracy"]), f4(ru["accuracy"])])
    out += md(["model", "task", "AUC trial", "AUC run", "drop (trial - run)", "acc trial",
               "acc run"], rows)
    out += ["", "Inflation estimate = (trial - run on rh_vs_feet) - mean of (trial - run) on "
            "lr and bf, per subject; paired bootstrap CI over subjects:", ""]
    rows = []
    for model, m in res["models"].items():
        for name in ("auc", "accuracy"):
            e = m[name]
            rows.append([model, "ROC-AUC (primary)" if name == "auc" else "accuracy (secondary)",
                         sg(e["estimate"]), ci(e["ci_lo"], e["ci_hi"]),
                         f"{e['wins']}/{e['n']}", _mask(partial, e["verdict"])])
    out += md(["model", "quantity", "estimate", "95 % CI", "subjects > 0", "reading"], rows)
    r = res["models"]["tgsp"].get("runopp")
    out += ["", "Secondary, tangent space, drift-opposed run pairing "
            "(tag PA-A4-rh_vs_feet-tgsp-runopp), read as in A3:", ""]
    if r:
        run_auc = res["models"]["tgsp"]["cells"][("rh_vs_feet", "run")]["auc"]
        out += md(["quantity", "AUC", "accuracy"],
                  [["rh_vs_feet, drift-opposed run split (iii)", f4(r["auc_runopp"]),
                    f4(r["acc_runopp"])],
                   ["rh_vs_feet, standard run split (ii)", f4(run_auc),
                    f4(res["models"]["tgsp"]["cells"][("rh_vs_feet", "run")]["accuracy"])],
                   ["drift-free part (mean of (ii), (iii))",
                    f"{f4(r['auc_drift_free']['mean'])} "
                    f"{ci(r['auc_drift_free']['lo'], r['auc_drift_free']['hi'], False)}",
                    f"{f4(r['acc_drift_free']['mean'])} "
                    f"{ci(r['acc_drift_free']['lo'], r['acc_drift_free']['hi'], False)}"],
                   ["drift part (half the difference)",
                    f"{sg(r['auc_drift']['mean'])} "
                    f"{ci(r['auc_drift']['lo'], r['auc_drift']['hi'])}",
                    f"{sg(r['acc_drift']['mean'])} "
                    f"{ci(r['acc_drift']['lo'], r['acc_drift']['hi'])}"]])
    for msg in res["secondary"].values():
        out.append(f"- {msg}")
    if not partial:
        out += ["", "Readings (9.26.6; the two models are reported separately, a "
                "disagreement is stated, not averaged):", ""]
        out += [f"- {model}: {_A4_READING[m['auc']['verdict']]}"
                for model, m in res["models"].items()]
    return out


# ================================================================== A5
def a5_results(log: Log) -> dict:
    arms: dict = {}
    for arm in A5_ARMS:
        tag = f"PA-A5-leak-{arm}"
        dec = decompose_each(confusions(log, tag, "trial_confusion"))
        tcm = confusions(log, tag, "timepoint_confusion", required=False)
        arms[arm] = {"timepoint": per_subject(log, tag, "timepoint_accuracy"),
                     "trial": dec,
                     "tp_within": (decompose_each(tcm)["within_family_accuracy"]
                                   if tcm is not None else None)}
    gates = {}
    v_all = float(arms["all"]["timepoint"].mean())
    v_none = float(np.mean(arms["none"]["trial"]["within_family_accuracy"]))
    for name, v, (lo, hi) in (("all", v_all, GATE_ALL), ("none", v_none, GATE_NONE)):
        gates[name] = {"value": v, "lo": lo, "hi": hi, "pass": bool(lo <= v <= hi)}
    res = {"arms": arms, "gates": gates, "gates_pass": all(g["pass"] for g in gates.values()),
           "contrasts": None}
    if not res["gates_pass"]:
        return res
    none = arms["none"]
    contrasts: dict = {}
    for arm in ("split", "bn", "graph", "all"):
        c = {}
        for name, a, b in (("timepoint", arms[arm]["timepoint"], none["timepoint"]),
                           ("within", arms[arm]["trial"]["within_family_accuracy"],
                            none["trial"]["within_family_accuracy"])):
            cc = compare(a, b)
            cc["verdict"], cc["direction"] = classify(cc["diff"], cc["ci_lo"], cc["ci_hi"])
            c[name] = cc
        contrasts[arm] = c
    gap = contrasts["all"]["timepoint"]["diff"]
    for arm, c in contrasts.items():
        c["timepoint"]["share_of_gap"] = c["timepoint"]["diff"] / gap if gap > EPS else None
    res["contrasts"] = contrasts
    singles = sorted(("split", "bn", "graph"), key=lambda a: -contrasts[a]["timepoint"]["diff"])
    best = singles[0]
    res["ranking"] = singles
    res["all_vs_best"] = {"best": best, **compare(arms["all"]["timepoint"],
                                                  arms[best]["timepoint"])}
    res["close_to"] = [a for a in singles
                       if abs(contrasts["all"]["timepoint"]["diff"]
                              - contrasts[a]["timepoint"]["diff"]) < MARGIN]
    return res


def format_a5(res: dict, partial: bool) -> list[str]:
    out = ["Validity gates (9.26.3), checked before any contrast:", ""]
    out += md(["gate", "quantity", "value", "must lie in", "result"],
              [["all", "mean timepoint accuracy", f4(res["gates"]["all"]["value"]),
                f"[{GATE_ALL[0]:.4f}, {GATE_ALL[1]:.4f}]",
                "pass" if res["gates"]["all"]["pass"] else "GATE FAILED"],
               ["none", "within-family (per-subject pooled trial_confusion, mean over "
                "subjects)", f4(res["gates"]["none"]["value"]),
                f"[{GATE_NONE[0]:.4f}, {GATE_NONE[1]:.4f}]",
                "pass" if res["gates"]["none"]["pass"] else "GATE FAILED"]])
    if not res["gates_pass"]:
        failed = [k for k, g in res["gates"].items() if not g["pass"]]
        out += ["", f"GATE FAILED ({', '.join(failed)}): the toggles are broken. No A5 number "
                "is quoted until that is fixed and the run repeated; no arm table or "
                "contrasts are printed."]
        return out
    arms = res["arms"]
    out += ["", "Arms (mean over subjects of per-subject values; trial-level figures from "
            "per-subject pooled confusion matrices):", ""]
    rows = []
    for arm in A5_ARMS:
        a = arms[arm]
        rows.append([arm, f4(a["timepoint"].mean()), f4(a["trial"]["accuracy_4class"].mean()),
                     f4(a["trial"]["family_accuracy"].mean()),
                     f4(np.nanmean(a["trial"]["within_family_accuracy"])),
                     "n/a" if a["tp_within"] is None else f4(np.nanmean(a["tp_within"]))])
    out += md(["arm", "timepoint acc", "4-class (trial)", "family (trial)",
               "within-family (trial)", "within-family (timepoint)"], rows)
    out += ["", "### Contrasts: each arm minus `none`, paired over subjects (exploratory, no "
            "multiplicity correction)", ""]
    rows = []
    for arm, c in res["contrasts"].items():
        for name, label in (("timepoint", "timepoint accuracy"),
                            ("within", "within-family (trial)")):
            cc = c[name]
            share = cc.get("share_of_gap")
            rows.append([arm, label, f4(cc["a"]), f4(cc["b"]), sg(cc["diff"]),
                         ci(cc["ci_lo"], cc["ci_hi"]), f"{cc['p']:.4f}",
                         f"{cc['wins']}/{cc['n']}",
                         "-" if share is None else f"{100 * share:.0f} %",
                         _mask(partial, cc["verdict"] + (f" ({cc['direction']})"
                                                         if cc["direction"] else ""))])
    out += md(["arm", "metric", "arm", "none", "diff", "95 % CI", "p", "arm ahead",
               "share of none->all gap", "verdict"], rows)
    if not partial:
        t = {a: res["contrasts"][a]["timepoint"] for a in res["contrasts"]}
        out += ["", "Readings (9.26.3, on timepoint accuracy, the metric the published "
                "protocol reports):", ""]
        for a in res["ranking"]:
            out.append(f"- {a} vs none: {sg(t[a]['diff'])} -> " + (
                "that path alone inflates the reference protocol"
                if t[a]["verdict"] == "effect" and t[a]["direction"] == "+"
                else "that path alone does not measurably inflate this formulation "
                     "(says nothing about its effect in combination)"))
        av = res["all_vs_best"]
        out.append(f"- all vs the largest single arm ({av['best']}): {sg(av['diff'])} "
                   f"{ci(av['ci_lo'], av['ci_hi'])} -> " + (
                       "the paths interact; single-arm effects do not add up to the "
                       "published inflation"
                       if av["diff"] >= MARGIN - EPS and av["ci_lo"] > 0
                       else "no interaction detected"))
        if res["close_to"]:
            out.append("- all is within 0.020 of: " + ", ".join(res["close_to"])
                       + " (that path carries the inflation)")
        out.append("- `split` also loses validation-based checkpoint selection (25 epochs, "
                   "final epoch), and `graph` bundles adjacency and standardiser; this "
                   "experiment cannot separate them (9.26.3).")
        out.append("- Three code paths in one formulation with one seed; not a method "
                   "comparison and not a win.")
    return out


# ================================================================== A6
A6_CONTRASTS = (
    ("C1", "si-shallow-ea - si-shallow-noea", "si-shallow-ea", "si-shallow-noea"),
    ("C2", "si-tgsp-ea - si-tgsp-noea", "si-tgsp-ea", "si-tgsp-noea"),
    ("C3", "ft-shallow-ea - PTG-shallow (reused, logged 0.7369)", "ft-shallow-ea",
     "PTG-shallow"),
)
A6_ARM_ORDER = ("si-shallow-noea", "si-shallow-ea", "si-tgsp-noea", "si-tgsp-ea",
                "PTG-shallow", "ft-shallow-ea")


def a6_results(log: Log) -> dict:
    arms = {a: binary_arm(log, a6_tag(a, "{p}")) for a in A6_ARM_ORDER}
    contrasts = []
    for cid, label, a, b in A6_CONTRASTS:
        c = compare(arms[a]["mean"], arms[b]["mean"], "two-sided")
        c.update(id=cid, label=label, arm_a=a, arm_b=b,
                 per_problem={p: compare(arms[a][p], arms[b][p], with_p=False)
                              for p in PROBLEMS})
        contrasts.append(c)
    holm = holm_bonferroni({c["id"]: c["p"] for c in contrasts}, alpha=ALPHA)
    for c in contrasts:
        c["p_holm"] = holm[c["id"]]["p_adjusted"]
        c["holm_reject"] = bool(holm[c["id"]]["reject"])
        raw, direction = classify(c["diff"], c["ci_lo"], c["ci_hi"])
        c["margin_ci_met"] = raw == "effect"
        # An effect needs the margin, the CI and survival of Holm over the three contrasts.
        c["verdict"] = "effect" if raw == "effect" and c["holm_reject"] else "no effect"
        c["direction"] = direction if c["verdict"] == "effect" else None
    c1, c2, c3 = contrasts

    def pred(c, name):
        if c["verdict"] == "effect" and c["direction"] == "+":
            return f"holds: effect of {sg(c['diff'])}"
        if c["verdict"] == "effect":
            return f"does not hold: effect in the opposite direction ({sg(c['diff'])})"
        return f"does not hold: no effect ({sg(c['diff'])})"

    if abs(c3["diff"]) < MARGIN:
        inside = c3["ci_lo"] > -MARGIN and c3["ci_hi"] < MARGIN
        p3 = ("holds: equivalence demonstrated" if inside
              else "holds: not resolved (the unadjusted CI reaches beyond ±0.020)")
    else:
        p3 = f"fails: |difference| = {abs(c3['diff']):.4f} >= 0.020"
    predictions = {"P1": pred(c1, "P1"), "P2": pred(c2, "P2"), "P3": p3}
    h1, h2 = predictions["P1"].startswith("holds"), predictions["P2"].startswith("holds")
    if h1 and h2:
        meaning = ("1 and 2 hold: EA helps subject-independent decoding for a deep and a "
                   "classical model alike, a data-alignment effect, not an architecture "
                   "effect.")
    elif h2:
        meaning = ("2 holds, 1 does not: the network already absorbs the between-subject "
                   "covariance shift, or the new source training lost what the alignment "
                   "gave; the tangent-space gain is reported on its own.")
    elif h1:
        meaning = ("1 holds, 2 does not: unexpected for a method built for covariance-based "
                   "models; check against the rank handling before it is reported.")
    else:
        meaning = ("Neither 1 nor 2 holds: a sixth challenger to the ~0.67 -> 0.73 picture "
                   "measured nothing; reported plainly as a null, no setting is then varied.")
    if predictions["P3"].startswith("fails"):
        meaning += (" 3 fails: EA moves the fine-tuned model by 0.020 or more, so "
                    "pre-training did not absorb the shift and the 9.17 reading needs a "
                    "qualification.")
    offline: dict = {}
    secondary: dict = {}
    for m in ("shallow", "tgsp"):
        tags = [(a6_tag(f"si-{m}-ea-offline", p), None) for p in PROBLEMS]
        if all(log.status(t, f)[2] for t, f in tags):
            arm = binary_arm(log, a6_tag(f"si-{m}-ea-offline", "{p}"))
            offline[m] = {**compare(arm["mean"], arms[f"si-{m}-noea"]["mean"]),
                          "arm": arm}
        else:
            for t, f in tags:
                if not log.status(t, f)[2]:
                    secondary[t] = _not_run(log, t, f)
    return {"arms": arms, "contrasts": contrasts, "predictions": predictions,
            "meaning": meaning, "offline": offline, "secondary": secondary}


def format_a6(res: dict, partial: bool) -> list[str]:
    out = ["Trial-level within-family accuracy; subject score = mean of the two binary "
           "problems (0-1 left/right fist, 2-3 both fists/feet), paired over subjects. "
           "Subject-independent arms exclude the calibration run (4 for 0-1, 6 for 2-3) "
           "from scoring in both arms. Never compared to the 0.650 within-session ceiling "
           "(9.15.4).", ""]
    out += md(["arm", "0-1", "2-3", "mean of both", "95 % CI of mean"],
              [[a + (" (reused)" if a == "PTG-shallow" else ""), f4(res["arms"][a]["0-1"].mean()),
                f4(res["arms"][a]["2-3"].mean()), f4(res["arms"][a]["mean"].mean()),
                ci(boot_mean(res["arms"][a]["mean"])["lo"],
                   boot_mean(res["arms"][a]["mean"])["hi"], signed=False)]
               for a in A6_ARM_ORDER])
    out += ["", "### Primary contrasts (three), Holm-corrected over the three two-sided "
            "p-values; effect = at least 0.020, 95 % paired bootstrap CI excluding zero, "
            "and surviving Holm", ""]
    rows = []
    for c in res["contrasts"]:
        rows.append([c["id"], c["label"], f4(c["a"]), f4(c["b"]), sg(c["diff"]),
                     ci(c["ci_lo"], c["ci_hi"]), f"{c['p']:.4f}", f"{c['p_holm']:.4f}",
                     f"{c['wins']}/{c['n']} (lost {c['losses']})",
                     _mask(partial, c["verdict"] + (f" ({c['direction']})"
                                                    if c["direction"] else ""))])
    out += md(["", "contrast", "with EA", "reference", "diff", "95 % CI", "p", "Holm p",
               "subjects ahead", "verdict"], rows)
    out += ["", "Each problem alone (no decision):", ""]
    out += md(["", "problem", "diff", "95 % CI", "subjects ahead"],
              [[c["id"], p, sg(pc["diff"]), ci(pc["ci_lo"], pc["ci_hi"]),
                f"{pc['wins']}/{pc['n']}"]
               for c in res["contrasts"] for p, pc in c["per_problem"].items()])
    if not partial:
        out += ["", "### Prediction scorecard (9.26.7)", ""]
        out += md(["prediction", "statement", "outcome"],
                  [["P1", "EA improves subject-independent ShallowConvNet by >= 0.020",
                    res["predictions"]["P1"]],
                   ["P2", "EA improves subject-independent tangent space by >= 0.020",
                    res["predictions"]["P2"]],
                   ["P3", "EA changes fine-tuned ShallowConvNet by < 0.020",
                    res["predictions"]["P3"]]])
        out += ["", f"Meaning (fixed in advance): {res['meaning']}"]
    out += ["", "### Secondary, TRANSDUCTIVE: offline EA (reference from all of the target's "
            "unlabelled trials). Never enters a prediction or the decision rule.", ""]
    if res["offline"]:
        out += md(["arm", "0-1", "2-3", "mean", "- noea of the same model", "95 % CI",
                   "subjects ahead"],
                  [[f"si-{m}-ea-offline (TRANSDUCTIVE)", f4(o["arm"]["0-1"].mean()),
                    f4(o["arm"]["2-3"].mean()), f4(o["a"]), sg(o["diff"]),
                    ci(o["ci_lo"], o["ci_hi"]), f"{o['wins']}/{o['n']}"]
                   for m, o in res["offline"].items()])
    for msg in res["secondary"].values():
        out.append(f"- {msg}")
    return out


# ================================================================== A1
def a1_results(log: Log, tgsp_only: bool = False) -> dict:
    models: dict = {}
    for m in (("tgsp",) if tgsp_only else ("shallow", "tgsp")):
        tag = f"PA-A1-{m}-4class"
        cms = confusions(log, tag, "trial_confusion")
        excl_cms = confusions(log, tag, "confusion_excl_calibration")
        dec, excl = decompose_each(cms), decompose_each(excl_cms)
        mean = {k: float(np.nanmean(v)) for k, v in dec.items()}
        chance = {"accuracy_4class": 0.25, "family_accuracy": 0.5,
                  "within_family_accuracy": 0.5}
        readings = {k: chance_reading(dec[k][np.isfinite(dec[k])], chance[k]) for k in dec}
        vs_a6 = None
        noea = [a6_tag(f"si-{m}-noea", p) for p in PROBLEMS]
        if all(log.status(t, None)[2] for t in noea):
            a6 = binary_arm(log, a6_tag(f"si-{m}-noea", "{p}"))["mean"]
            c = compare(excl["within_family_accuracy"], a6)
            if abs(c["diff"]) < MARGIN:
                reading = "within 0.020: 4-class training costs nothing within-family"
            elif c["diff"] <= -MARGIN + EPS and c["ci_hi"] < 0:
                reading = (f"lower by {abs(c['diff']):.4f}: 4-class training costs that "
                           "much within-family")
            elif c["diff"] >= MARGIN - EPS and c["ci_lo"] > 0:
                reading = (f"higher by {c['diff']:.4f} (no reading pre-registered for this "
                           "direction)")
            else:
                reading = "|difference| >= 0.020 but the CI includes zero: indistinguishable"
            vs_a6 = {"a1": c["a"], "a6": c["b"], "diff": c["diff"], "ci_lo": c["ci_lo"],
                     "ci_hi": c["ci_hi"], "wins": c["wins"], "n": c["n"], "p": c["p"],
                     "reading": reading}
        models[m] = {"per_subject": dec, "mean": mean, "readings": readings,
                     "pooled": decompose(cms.sum(axis=0)), "pooled_cm": cms.sum(axis=0),
                     "excl": {k: float(np.nanmean(v)) for k, v in excl.items()},
                     "vs_a6": vs_a6}
    return {"models": models}


CLASS_NAMES = ["left_fist", "right_fist", "both_fists", "both_feet"]


def format_a1(res: dict, partial: bool) -> list[str]:
    out = ["4-class subject-independent decoding, zero-shot, no EA, no calibration run "
           "excluded in the main figures. Chance: 0.25 for 4-class, 0.5 for family and "
           "within-family. A 4-class figure is never quoted without the family and "
           "within-family lines beside it. Source models are pre-trained on ~92 other "
           "subjects: not comparable to the 0.650 ceiling (9.15.4). No model is called "
           "better on A1.", ""]
    rows = []
    for m, r in res["models"].items():
        for key, label in (("accuracy_4class", "4-class"), ("family_accuracy", "family"),
                           ("within_family_accuracy", "within-family")):
            q = r["readings"][key]
            rows.append([m, label, q["n"], f4(r["mean"][key]), ci(q["lo"], q["hi"], False),
                         f4(r["pooled"][key]), _mask(partial, q["reading"])])
    out += md(["model", "measure", "n subjects", "mean of per-subject", "95 % CI",
               "pooled confusion", "vs chance"], rows)
    for m, r in res["models"].items():
        out += ["", f"Pooled confusion matrix, {m} (rows true, columns predicted; "
                f"{r['pooled']['n']} trials):", ""]
        out += md([""] + [n[:9] for n in CLASS_NAMES],
                  [[CLASS_NAMES[i]] + [int(v) for v in row]
                   for i, row in enumerate(r["pooled_cm"])])
    out += ["", "Within-family against the A6 `noea` arm of the same model, on the trials "
            "outside runs 4 and 6 only (`confusion_excl_calibration`), paired over "
            "subjects:", ""]
    rows = []
    for m, r in res["models"].items():
        v = r["vs_a6"]
        if v is None:
            rows.append([m, f4(r["excl"]["within_family_accuracy"]), "not run", "-", "-", "-",
                         "-"])
        else:
            rows.append([m, f4(v["a1"]), f4(v["a6"]), sg(v["diff"]),
                         ci(v["ci_lo"], v["ci_hi"]), f"{v['wins']}/{v['n']}",
                         _mask(partial, v["reading"])])
    out += md(["model", "A1 within-family (excl. 4/6)", "A6 noea binary", "diff", "95 % CI",
               "A1 ahead", "reading"], rows)
    if not partial:
        out += ["", "Readings (9.26.8):", ""]
        for m, r in res["models"].items():
            fam, win = r["readings"]["family_accuracy"], r["readings"]["within_family_accuracy"]
            four = r["readings"]["accuracy_4class"]
            out.append(f"- {m}: family accuracy is {fam['reading']}"
                       + (" -> run family is identifiable across subjects (this alone does "
                          "not separate the run artefact from fist-vs-feet task information; "
                          "A3 is the label-free test)" if fam["reading"] == "above chance"
                          else ""))
            if four["reading"] == "above chance" and win["reading"] != "above chance":
                out.append(f"  - 4-class accuracy is above 0.25 while within-family stays "
                           f"{win['reading']}: the signature of 9.4 and 9.6, a 4-class "
                           "figure that is family identification; reported as a confound, "
                           "not as decoding.")
    return out


# ================================================================ dispatch
TITLES = {
    "a1": "A1 - 4-class subject-independent decoding with the family decomposition "
          "(EXPLORATORY)",
    "a2": "A2 - dependence on the first second after the cue (EXPLORATORY)",
    "a3": "A3 - label-free run-family confound from rest data (EXPLORATORY)",
    "a4": "A4 - the cross-family task (EXPLORATORY)",
    "a5": "A5 - GCNs-Net leakage paths, one at a time (EXPLORATORY)",
    "a6": "A6 - Euclidean alignment (PRE-REGISTERED method test)",
}
RESULTS = {"a1": (a1_results, format_a1), "a2": (a2_results, format_a2),
           "a3": (a3_results, format_a3), "a4": (a4_results, format_a4),
           "a5": (a5_results, format_a5), "a6": (a6_results, format_a6)}


def _check_primary(log: Log, pl: Plan) -> list[tuple[str, int, int]]:
    out = []
    for tag, folds in pl.primary:
        have, need, complete = log.status(tag, folds)
        if not complete:
            out.append((tag, have, need))
    return out


def analyse(experiment: str, rows: list[dict], subjects=None, allow_partial: bool = False,
            tgsp_only: bool = False) -> Report:
    """Render one experiment. Raises ``IncompleteRun`` when a primary tag is short."""
    subjects = list(subjects) if subjects is not None else default_subjects()
    pl = plan(experiment, tgsp_only=tgsp_only)
    log = Log(rows, subjects)
    short = _check_primary(log, pl)
    if short and not allow_partial:
        n = len(subjects)
        raise IncompleteRun(
            f"{experiment.upper()}: incomplete, refusing to report a partial run "
            f"({n} subjects required in every tag):\n  "
            + "\n  ".join(f"{t}: {have} of {need} rows" for t, have, need in short))
    partial = bool(short)
    progress = [(t, *log.status(t, f)[:2]) for t, f in pl.primary + pl.secondary]
    if partial:
        keep = set(log.subjects)
        for tag, folds in pl.primary:
            keep &= log.complete_subjects(tag, folds)
        keep = [s for s in log.subjects if s in keep]
        used = log.restrict(keep)
    else:
        keep, used = log.subjects, log

    out: list[str] = []
    if partial:
        out += [f"# {PARTIAL}", ""]
    out += [f"## {TITLES[experiment]}", ""]
    if partial:
        out += [f"**{PARTIAL}.** A progress view of a run that is not finished; CLAUDE.md "
                "forbids quoting it. All verdicts and readings are withheld.", "",
                "Progress (rows held of rows needed):", ""]
        out += md(["tag", "held", "needed"], [[t, h, nd] for t, h, nd in progress])
        out += ["", f"Subjects complete in every primary tag: {len(keep)} of "
                f"{len(subjects)}.", ""]
        if not keep:
            out += ["Nothing to tabulate yet.", "", f"# {PARTIAL}"]
            return Report("\n".join(out), 0)
    out += [f"{len(keep)} subjects; paired bootstrap, 10,000 resamples, 95 % CI; an effect "
            f"is a difference of at least {MARGIN:.3f} with the CI excluding zero.", ""]

    compute, fmt = RESULTS[experiment]
    res = compute(used, tgsp_only=tgsp_only) if experiment == "a1" else compute(used)
    out += fmt(res, partial)

    dups = {t: used.surplus[t] for t, _ in pl.primary + pl.secondary if used.surplus.get(t)}
    if dups:
        out += ["", "NOTE: duplicate rows replaced last-wins (surplus rows per tag): "
                + ", ".join(f"{t}: {n}" for t, n in dups.items())]
    if experiment == "a1" and tgsp_only:
        out += ["", "NOTE: A1 reported for tangent space only, as 9.26.8 allows when "
                "compute runs short."]
    status = 3 if experiment == "a5" and not res["gates_pass"] else 0
    if partial:
        out += ["", f"# {PARTIAL}"]
    return Report("\n".join(out), status)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=EXPERIMENTS)
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    ap.add_argument("--allow-partial", action="store_true",
                    help="print a progress table watermarked PARTIAL — NOT A RESULT; "
                         "nothing it shows may be quoted")
    ap.add_argument("--tgsp-only", action="store_true",
                    help="A1 only: report tangent space alone, the 9.26.8 fallback when "
                         "compute runs short")
    args = ap.parse_args(argv)
    if args.tgsp_only and args.experiment != "a1":
        ap.error("--tgsp-only applies to a1 only")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    rows = load_rows(Path(args.log))
    try:
        rep = analyse(args.experiment, rows, allow_partial=args.allow_partial,
                      tgsp_only=args.tgsp_only)
    except IncompleteRun as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(rep.text)
    return rep.exit_code


# ------------------------------------------------------------- module notes
# Choices where 9.26 was silent or ambiguous (all listed in the Task 8 report):
#  * Effect in A6 = margin AND CI AND Holm (two-sided p, bootstrap_p_value) over three.
#  * A5 contrasts carry no multiplicity correction (none is pre-registered).
#  * A1 shallow is required unless --tgsp-only is passed; A6 noea comparator is secondary.
#  * Per-subject score over folds is the plain mean of fold values (trial_accuracy,
#    accuracy, roc_auc, timepoint_accuracy), as confirmatory_results does for A2/PTG.

if __name__ == "__main__":
    raise SystemExit(main())
