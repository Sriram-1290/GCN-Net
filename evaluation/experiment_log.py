"""Reproducibility helpers: seeding, git provenance, and the experiment log.

CLAUDE.md requires one row per run in a persistent log carrying timestamp, git
commit, config, split type, seed and accuracy; and that torch/numpy/random seeds
are set and logged. Both live here so no experiment script has to remember.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_LOG = Path("runs/experiments.jsonl")


def set_seed(seed: int, deterministic: bool = True) -> dict[str, int | bool]:
    """Seed Python, NumPy and torch. Returns what was set, for logging."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuDNN picks algorithms by benchmarking otherwise, which makes runs
        # differ slightly even at a fixed seed.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return {"seed": seed, "deterministic": deterministic}


def git_provenance(repo: Path | str = ".") -> dict[str, Any]:
    """Current commit and whether the tree is dirty.

    A dirty tree means the logged commit does not fully describe the code that
    produced the row, so it is recorded explicitly rather than silently ignored.
    """
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=str(repo), capture_output=True,
                                 text=True, timeout=15)
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    return value


def log_experiment(record: dict[str, Any], path: Path | str = DEFAULT_LOG) -> Path:
    """Append one run to the JSONL log, stamped with time and git provenance.

    JSONL rather than CSV because runs carry nested structures (confusion
    matrices, per-class accuracies, config trees) that a flat CSV would mangle.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **git_provenance(),
        **_jsonable(record),
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return path


def read_experiments(path: Path | str = DEFAULT_LOG) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def completed_keys(path: Path | str, tag: str) -> set[tuple[str, int | None]]:
    """``(subject, fold)`` pairs already recorded under ``tag``.

    The experiment log is the checkpoint. A row is appended only *after* a fold has
    trained and been evaluated, so any fold interrupted part-way leaves no row and
    is simply re-run. That makes every runner safely resumable without a separate
    checkpoint format to keep in sync.

    A crash during a write can leave a truncated final line; malformed lines are
    skipped rather than raising, since refusing to resume because of one damaged
    row would defeat the purpose.
    """
    path = Path(path)
    if not path.exists():
        return set()
    done: set[tuple[str, int | None]] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated trailing row from an interrupted write
            if row.get("tag") == tag and "subject" in row:
                done.add((row["subject"], row.get("fold")))
    return done


def describe_resume(done: set, total: int, tag: str) -> str:
    """One line for the runner to print, so a resumed run says what it skipped."""
    if not done:
        return f"resume: no previous rows for tag {tag!r}, starting from the beginning"
    return (f"resume: {len(done)} of {total} folds already logged under {tag!r}, "
            f"skipping those")
