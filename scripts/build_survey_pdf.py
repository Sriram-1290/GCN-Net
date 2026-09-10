"""Render LITERATURE_SURVEY.md to a print-ready HTML for PDF export.

Separate from `build_paper_pdf.py` because the submission has different constraints:
a centred title page on its own sheet in the institute's layout, and an eight-column
survey table whose cells carry full sentences. That table is the whole reason this is
not just the paper's build with a different input -- at body font size it overflows A4,
so it is set landscape on its own pages while the rest stays portrait.

Usage:
    python scripts/build_survey_pdf.py
    # then, with any Chromium:
    #   msedge --headless=new --disable-gpu --no-pdf-header-footer \
    #          --print-to-pdf=Literature-Survey-Report.pdf build/survey_print.html
"""
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "survey_print.html"
OUT.parent.mkdir(exist_ok=True)

src = (ROOT / "LITERATURE_SURVEY.md").read_text(encoding="utf-8")

# The title block is hand-laid rather than rendered from Markdown: the institute format
# is a specific centred layout, and Markdown gives no control over vertical placement.
title_md, rest_md = src.split("## 1. Literature Survey Table", 1)
rest = markdown.markdown("## 1. Literature Survey Table" + rest_md,
                         extensions=["tables", "attr_list", "sane_lists", "smarty"])

# Angle-bracket placeholders would be swallowed as HTML tags; show them as fill-in blanks.
import re
rest = re.sub(r"&lt;([^&]*?)&gt;", r'<span class="blank">\1</span>', rest)


def field(line: str) -> str:
    return re.sub(r"<([^>]*)>", r'<span class="blank">\1</span>', line)


TITLE = """
<section class="titlepage">
  <p class="t1">LITERATURE SURVEY REPORT</p>
  <p class="t2">ON</p>
  <p class="t3">Dynamic Graph Convolutional Network with Temporal Attention for Subject-Independent EEG Motor Imagery Decoding</p>
  <p class="t4">BACHELOR OF ENGINEERING<br>IN<br><strong>COMPUTER SCIENCE</strong></p>
  <p class="t45">Project Domain: Deep Learning &nbsp;&middot;&nbsp; Batch No.: 22</p>
  <p class="t5">Submitted by</p>
  <table class="students">
    <tr><td>Saligram Sriram Shravan</td><td>245323753054</td></tr>
    <tr><td>Parcha Seetha Ramarao</td><td>245323753045</td></tr>
    <tr><td>M Priyansh</td><td>245323753030</td></tr>
  </table>
  <p class="t5">Under the guidance of</p>
  <p class="t6">Dr. Gagan Kumar<br><span class="blank">DESIGNATION</span></p>
  <p class="t7">Department of Computer Science</p>
  <p class="t8">NEIL GOGTE INSTITUTE OF TECHNOLOGY</p>
  <p class="t9">Kachavanisingaram Village, Hyderabad, Telangana 500058.</p>
</section>
"""

CSS = """
@page { size: A4 portrait; margin: 18mm 16mm; }
@page landscape { size: A4 landscape; margin: 12mm 10mm; }

html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font: 10.5pt/1.45 "Times New Roman", Georgia, serif; color: #000; background: #fff;
       margin: 0; }

/* ---------------------------------------------------------------- title page */
.titlepage { height: 250mm; display: flex; flex-direction: column; align-items: center;
             justify-content: flex-start; text-align: center; page-break-after: always; }
.titlepage p { margin: 0 0 10mm; }
.t1 { font-size: 13pt; font-weight: bold; letter-spacing: .5px; margin-top: 22mm !important; }
.t2 { font-size: 11pt; }
.t3 { font-size: 16pt; font-weight: bold; max-width: 145mm; line-height: 1.35; }
.t4 { font-size: 12pt; line-height: 1.7; }
.t45 { font-size: 10.5pt; margin-bottom: 7mm !important; }
.t5 { font-size: 11pt; font-weight: bold; margin-bottom: 4mm !important; }
.t6 { font-size: 12pt; font-weight: bold; line-height: 1.6; }
.t7 { font-size: 11pt; margin-bottom: 2mm !important; }
.t8 { font-size: 15pt; font-weight: bold; margin-bottom: 2mm !important; }
.t9 { font-size: 10pt; }
.students { border: none; width: auto; margin: 0 auto 10mm; }
.students td { border: none; padding: 1.5mm 8mm; font-size: 11.5pt; font-weight: bold;
               text-align: left; }
.blank { background: #ffef9f; padding: 0 2mm; }

/* ---------------------------------------------------------------- body */
h2 { font-size: 13pt; margin: 9mm 0 3mm; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 6mm 0 2mm; page-break-after: avoid; }
p { margin: 0 0 3mm; text-align: justify; }
hr { border: none; border-top: .4pt solid #999; margin: 7mm 0; }
em { font-style: italic; }

table { border-collapse: collapse; width: 100%; margin: 3mm 0 5mm; }
th, td { border: .5pt solid #000; padding: 1.4mm 1.8mm; vertical-align: top;
         text-align: left; }
th { background: #eee; font-weight: bold; }

/* The 27-row survey table: eight columns of prose will not fit portrait, so this one
   table gets its own landscape pages. Column widths are tuned so the two widest
   columns -- Key Findings and Limitations -- get the room they need. */
.survey { font-size: 6.8pt; line-height: 1.3; }
.survey th, .survey td { padding: .9mm 1.1mm; }
.survey col.c1 { width: 3%; }   .survey col.c2 { width: 12%; }
.survey col.c3 { width: 4%; }   .survey col.c4 { width: 18%; }
.survey col.c5 { width: 14%; }  .survey col.c6 { width: 10%; }
.survey col.c7 { width: 18%; }  .survey col.c8 { width: 21%; }
/* The wrapper -- not just the table -- carries the landscape page, and it starts at
   the section heading rather than at the table. A named page always begins a new sheet,
   so opening it at the table instead would strand the heading and its intro paragraph
   alone on a near-empty portrait page. */
.survey-wrap { page: landscape; page-break-after: always; }
.survey-wrap h2 { margin-top: 0; }

.compare { font-size: 8.6pt; }
.compare col.k1 { width: 20%; } .compare col.k2 { width: 8%; }
.compare col.k3 { width: 34%; } .compare col.k4 { width: 38%; }
"""

# Tag the two data tables so the CSS above can treat them differently.
tables = rest.split("<table>")
if len(tables) >= 3:
    cols = "".join(f'<col class="c{i}">' for i in range(1, 9))
    tables[1] = f'<colgroup>{cols}</colgroup>' + tables[1]
    kols = "".join(f'<col class="k{i}">' for i in range(1, 5))
    tables[2] = f'<colgroup>{kols}</colgroup>' + tables[2]
    rest = ('<div class="survey-wrap">' + tables[0] + '<table class="survey">'
            + tables[1] + '<table class="compare">' + "<table>".join(tables[2:]))
    # Close the landscape wrapper immediately after the survey table. A <hr> and the
    # next <h2> sit between the two tables, so anchor on the first </table> after the
    # survey opening rather than on the following table's tag.
    i = rest.index('<table class="survey">')
    j = rest.index('</table>', i) + len('</table>')
    rest = rest[:j] + '</div>' + rest[j:]

OUT.write_text(
    f"<!doctype html><html><head><meta charset='utf-8'>"
    f"<title>Literature Survey Report</title><style>{CSS}</style></head>"
    f"<body>{TITLE}{rest}</body></html>", encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
