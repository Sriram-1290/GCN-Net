"""Render docs/RESEARCH_DIRECTIONS.md to docs/Research-Directions.pdf via headless Edge.

The report collects the two fact-checked literature research rounds of 2026-11-07 and
the future-work plan built on them. Same pipeline as the other document builders:
Markdown -> HTML with print CSS -> Microsoft Edge --print-to-pdf.

Usage:
    python scripts/build_research_report.py
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "RESEARCH_DIRECTIONS.md"
OUT_PDF = ROOT / "docs" / "Research-Directions.pdf"
BUILD = ROOT / "build"

CSS = """
@page { size: A4 portrait; margin: 16mm 15mm 16mm 15mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 9.6pt; line-height: 1.42;
       color: #1d1f24; margin: 0; }
.titlepage { border-bottom: 2.5px solid #1f4e79; padding-bottom: 10px; margin-bottom: 14px; }
.titlepage h1 { font-size: 22pt; margin: 0 0 4px 0; color: #1f4e79; border: none; }
.titlepage .subtitle { font-size: 11.5pt; color: #3d4450; margin-bottom: 8px; }
.titlepage .meta { font-size: 9pt; color: #5a616d; }
h2 { font-size: 13.5pt; color: #1f4e79; margin: 18px 0 6px; padding-bottom: 3px;
     border-bottom: 1px solid #c9d6e3; break-after: avoid; }
h3 { font-size: 11pt; color: #2b2f36; margin: 12px 0 4px; break-after: avoid; }
p { margin: 4px 0 7px; text-align: justify; }
ul, ol { margin: 3px 0 8px 18px; padding-left: 6px; }
li { margin: 2px 0; }
strong { color: #111; }
code { font-family: Consolas, monospace; font-size: 8.6pt; background: #f1f3f6; padding: 0 3px;
       border-radius: 2px; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 8.6pt;
        break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { background: #1f4e79; color: #fff; text-align: left; padding: 4px 6px; font-weight: 600; }
td { border-bottom: 1px solid #d7dde5; padding: 3.5px 6px; vertical-align: top; }
tbody tr:nth-child(even) td { background: #f6f8fb; }
a { color: #1f4e79; text-decoration: none; word-break: break-all; }
em { color: #2b2f36; }
h2#summary + p, h2#summary ~ ol { }
.footer-note { font-size: 8pt; color: #5a616d; }
"""


def parse_front_matter(src: str) -> tuple[dict[str, str], str]:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", src, re.S)
    if not m:
        raise ValueError("expected YAML front matter in RESEARCH_DIRECTIONS.md")
    meta = dict(re.findall(r'^(\w+):\s*"(.*)"\s*$', m.group(1), re.M))
    for key in ("title", "subtitle", "project", "date"):
        if key not in meta:
            raise ValueError(f"front matter is missing {key!r}")
    return meta, m.group(2)


def build_html(meta: dict[str, str], body_md: str) -> str:
    body = markdown.markdown(body_md, extensions=["tables", "sane_lists", "toc", "smarty"])
    # Bare URLs in the reference list become links.
    body = re.sub(r'(?<!href=")(https?://[^\s<)]+)', r'<a href="\1">\1</a>', body)
    title = f"""<div class="titlepage">
<h1>{html.escape(meta['title'])}</h1>
<div class="subtitle">{html.escape(meta['subtitle'])}</div>
<div class="meta">{html.escape(meta['project'])} &middot; {html.escape(meta['date'])}</div>
</div>"""
    return (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            f"<title>{html.escape(meta['title'])}</title><style>{CSS}</style></head>"
            f"<body>{title}{body}</body></html>")


def find_edge() -> Path:
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError("Microsoft Edge not found; it is needed to print the PDF")


def main() -> int:
    meta, body_md = parse_front_matter(SRC.read_text(encoding="utf-8"))
    BUILD.mkdir(exist_ok=True)
    out_html = BUILD / "research_directions_print.html"
    out_html.write_text(build_html(meta, body_md), encoding="utf-8")
    print(f"wrote {out_html}")

    temp_pdf = BUILD / "temp_research_directions.pdf"
    cmd = [str(find_edge()), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={temp_pdf.resolve()}", out_html.resolve().as_uri()]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not temp_pdf.exists():
        print(res.stdout, res.stderr, file=sys.stderr)
        return 1
    shutil.copy2(temp_pdf, OUT_PDF)
    print(f"wrote {OUT_PDF} ({OUT_PDF.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
