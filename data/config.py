"""Typed configuration loaded from YAML.

Config is loaded once at the entry point and passed explicitly to whatever needs
it. There is no module-level singleton and nothing mutates a shared object — see
CLAUDE.md, "Don't use global mutable config shared across modules".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    cache_dir: Path
    imagery_runs: tuple[int, ...]
    exclude_subjects: frozenset[str]
    expect_sfreq: float
    expect_n_channels: int
    expect_events_per_run: int


@dataclass(frozen=True)
class EpochConfig:
    tmin: float
    length_s: float

    def n_samples(self, sfreq: float) -> int:
        """Samples per epoch. 4.0 s at 160 Hz -> exactly 640."""
        n = self.length_s * sfreq
        if abs(n - round(n)) > 1e-9:
            raise ValueError(
                f"epoch length {self.length_s}s at {sfreq}Hz is not a whole "
                f"number of samples ({n})"
            )
        return int(round(n))


@dataclass(frozen=True)
class PreprocessConfig:
    # None reproduces the TensorFlow reference, which applies no filtering at all.
    bandpass: tuple[float, float] | None
    notch: float | None
    rereference: str | None
    normalize: bool


@dataclass(frozen=True)
class Config:
    dataset: DatasetConfig
    epoch: EpochConfig
    preprocess: PreprocessConfig
    label_scheme: str
    class_names: tuple[str, ...]
    seed: int
    source_path: Path = field(default=Path("configs/default.yaml"))


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Read a YAML config into a frozen Config."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    ds, ep, pp, lb = raw["dataset"], raw["epoch"], raw["preprocess"], raw["labels"]

    bandpass = tuple(pp["bandpass"]) if pp.get("bandpass") else None
    if bandpass is not None and len(bandpass) != 2:
        raise ValueError(f"preprocess.bandpass must be [l_freq, h_freq], got {bandpass}")

    return Config(
        dataset=DatasetConfig(
            root=Path(ds["root"]),
            cache_dir=Path(ds["cache_dir"]),
            imagery_runs=tuple(ds["imagery_runs"]),
            exclude_subjects=frozenset(ds["exclude_subjects"]),
            expect_sfreq=float(ds["expect_sfreq"]),
            expect_n_channels=int(ds["expect_n_channels"]),
            expect_events_per_run=int(ds["expect_events_per_run"]),
        ),
        epoch=EpochConfig(tmin=float(ep["tmin"]), length_s=float(ep["length_s"])),
        preprocess=PreprocessConfig(
            bandpass=bandpass,
            notch=float(pp["notch"]) if pp.get("notch") else None,
            rereference=pp.get("rereference"),
            normalize=bool(pp.get("normalize", False)),
        ),
        label_scheme=lb["scheme"],
        class_names=tuple(lb["names"]),
        seed=int(raw["seed"]),
        source_path=path,
    )


def all_subjects(cfg: Config) -> list[str]:
    """Subject IDs on disk, minus the configured exclusions, sorted."""
    found = sorted(
        p.name for p in cfg.dataset.root.iterdir()
        if p.is_dir() and len(p.name) == 4 and p.name.startswith("S") and p.name[1:].isdigit()
    )
    if not found:
        raise FileNotFoundError(f"no subject directories under {cfg.dataset.root}")
    return [s for s in found if s not in cfg.dataset.exclude_subjects]
