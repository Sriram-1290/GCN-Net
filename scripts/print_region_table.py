"""Print the electrode-to-region table that the montage-robustness study pre-registers.

    python scripts/print_region_table.py
    python scripts/print_region_table.py --cache data/cache_bp830

Output is Markdown: one row per cached channel (in cache order) with the region the
name-based rule assigns it, then the occupancy of the nine regions under the full
cap, the BCI IV-2a set and the 10-20 set. The rule reads only the electrode *name*
(``data.montage.region_of``), never the data, so this table is fixed by the code and
is pasted into the pre-registration (``project_documentation.md`` 9.24) before any
run. Fails loudly if a montage leaves a region empty or names a channel the cache
does not hold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import load_subject  # noqa: E402
from data.montage import (  # noqa: E402
    IV2A_CHANNELS, REGION_NAMES, TEN_TWENTY_CHANNELS, region_of,
)


def channel_table(ch_names: list[str]) -> str:
    lines = ["| # | channel | region |", "|---|---|---|"]
    for i, name in enumerate(ch_names):
        lines.append(f"| {i} | {name} | {REGION_NAMES[region_of(name)]} |")
    return "\n".join(lines)


def occupancy(title: str, names: list[str]) -> str:
    per_region: list[list[str]] = [[] for _ in REGION_NAMES]
    for n in names:
        per_region[region_of(n)].append(n)
    empty = [REGION_NAMES[i] for i, c in enumerate(per_region) if not c]
    if empty:
        raise SystemExit(f"{title} leaves regions empty: {empty}")
    lines = [f"**{title}** ({len(names)} channels)", "",
             "| region | n | channels |", "|---|---|---|"]
    for i, chans in enumerate(per_region):
        lines.append(f"| {REGION_NAMES[i]} | {len(chans)} | {', '.join(chans)} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="data/cache_bp830")
    ap.add_argument("--subject", default="S001")
    args = ap.parse_args()

    ch_names = list(load_subject(args.subject, Path(args.cache)).ch_names)
    assert len(ch_names) == 64, f"expected 64 cached channels, got {len(ch_names)}"
    for title, names in (("IV-2a", IV2A_CHANNELS), ("10-20", TEN_TWENTY_CHANNELS)):
        missing = [c for c in names if c not in ch_names]
        assert not missing, f"{title} channels absent from the cache: {missing}"

    print(channel_table(ch_names))
    print()
    print(occupancy("Full cap", ch_names))
    print()
    print(occupancy("IV-2a montage", list(IV2A_CHANNELS)))
    print()
    print(occupancy("10-20 montage", list(TEN_TWENTY_CHANNELS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
