"""Reproducibility helpers: seeding, git provenance, and the experiment log.

CLAUDE.md requires one row per run in a persistent log carrying timestamp, git
commit, config, split type, seed and accuracy; and that torch/numpy/random seeds
are set and logged. Both live here so no experiment script has to remember.
"""

from __future__ import annotations

import json
import os
import random
import time
import subprocess
from contextlib import contextmanager
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


@contextmanager
def _exclusive(fh):
    """Hold an exclusive lock on an open file for the duration of the block.

    Several experiment runners append to one log concurrently. Rows are ~3 KB,
    well past any platform's atomic-write guarantee, so unsynchronised appends
    can interleave and split a record across lines. That happened once for real:
    two processes wrote in the same second and left a 16-byte orphan fragment,
    losing one fold's row.

    stdlib only -- msvcrt on Windows, fcntl elsewhere. If neither is available the
    lock degrades to a no-op rather than crashing, since a missing lock is worse
    than a failed run only in the rare concurrent case.

    **The lock byte is fixed at offset 0, not at end-of-file.** ``msvcrt.locking``
    locks a byte range starting at the *current file position*, so seeking to EOF
    first makes each process lock whichever byte its own EOF happens to be —
    and once two processes see different EOFs they contend on different bytes and
    the mutual exclusion silently lapses. That was the original implementation and
    it is why the regression test for this was intermittently flaky. Offset 0 is a
    fixed rendezvous every process agrees on; the file is opened in append mode, so
    seeking does not affect where the write lands.
    """
    try:
        import msvcrt
        fh.seek(0)
        while True:
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                break
            except OSError:
                time.sleep(0.05)
        try:
            yield
        finally:
            try:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        return
    except ImportError:
        pass
    try:
        import fcntl
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return
    except ImportError:
        pass
    yield


def log_experiment(record: dict[str, Any], path: Path | str = DEFAULT_LOG) -> Path:
    """Append one run to the JSONL log, stamped with time and git provenance.

    JSONL rather than CSV because runs carry nested structures (confusion
    matrices, per-class accuracies, config trees) that a flat CSV would mangle.

    The append is written as **one locked, unbuffered syscall**. Concurrent
    runners are the normal case here, and a torn row is silent data loss.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **git_provenance(),
        **_jsonable(record),
    }
    payload = (json.dumps(row) + "\n").encode("utf-8")
    with open(path, "ab", buffering=0) as fh:
        with _exclusive(fh):
            fh.write(payload)
    return path


def read_experiments(path: Path | str = DEFAULT_LOG) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def completed_keys(path: Path | str, tag: str, window_start: int | None = None
                   ) -> set[tuple[str, int | None]]:
    """``(subject, fold)`` pairs already recorded under ``tag``.

    The experiment log is the checkpoint. A row is appended only *after* a fold has
    trained and been evaluated, so any fold interrupted part-way leaves no row and
    is simply re-run. That makes every runner safely resumable without a separate
    checkpoint format to keep in sync.

    A crash during a write can leave a truncated final line; malformed lines are
    skipped rather than raising, since refusing to resume because of one damaged
    row would defeat the purpose.

    ``window_start`` (default ``None`` = no filter, the original behaviour) keeps
    only rows logged at that window start, treating a row without the field as 0.
    A run with ``--window-start N`` then cannot mistake another start's rows under
    the same tag for its own completed folds.
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
            if window_start is not None and row.get("window_start", 0) != window_start:
                continue
            if row.get("tag") == tag and "subject" in row:
                done.add((row["subject"], row.get("fold")))
    return done


def describe_resume(done: set, total: int, tag: str) -> str:
    """One line for the runner to print, so a resumed run says what it skipped."""
    if not done:
        return f"resume: no previous rows for tag {tag!r}, starting from the beginning"
    return (f"resume: {len(done)} of {total} folds already logged under {tag!r}, "
            f"skipping those")


def _row_identity(row: dict) -> tuple:
    """Fields that make a log row unique.

    Includes more than (tag, subject, fold) because some runs legitimately write
    several rows per subject under one tag: the CSP baseline writes one per binary
    *problem*, and a sweep writes one per learning rate. Keying on those avoids
    reporting correct rows as duplicates.
    """
    return (
        row.get("tag"),
        row.get("subject"),
        row.get("fold"),
        row.get("problem"),
        row.get("train_config", {}).get("learning_rate"),
        tuple(row["classes"]) if row.get("classes") else None,
    )


def find_duplicates(path: Path | str = DEFAULT_LOG) -> dict[str, int]:
    """Duplicated row identities per tag, empty when the log is clean.

    Worth calling at the top of any analysis: a duplicated fold silently
    double-counts in a per-subject mean, which looks like a result rather than an
    error. This happened once for real -- an orphaned driver survived a stop and
    ran the same tags concurrently with its replacement.
    """
    counts: dict[tuple, int] = {}
    for row in read_experiments(path):
        k = _row_identity(row)
        counts[k] = counts.get(k, 0) + 1
    per_tag: dict[str, int] = {}
    for key, n in counts.items():
        if n > 1:
            per_tag[key[0]] = per_tag.get(key[0], 0) + (n - 1)
    return per_tag


def warn_if_duplicated(path: Path | str = DEFAULT_LOG) -> bool:
    """Print a loud warning if the log has duplicates. Returns True if clean."""
    dupes = find_duplicates(path)
    if not dupes:
        return True
    print("!" * 70)
    print("WARNING: duplicated rows in the experiment log. Per-subject means below")
    print("are double-counting. Deduplicate before trusting any number.")
    for tag, n in sorted(dupes.items(), key=lambda kv: -kv[1]):
        print(f"  {tag}: {n} extra rows")
    print("!" * 70)
    return False
