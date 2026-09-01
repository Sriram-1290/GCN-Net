"""Render PAPER.md to a print-optimised HTML for headless PDF export.

Separate from the artifact build: the screen version is theme-aware and sized for a
browser pane, while print needs a fixed light palette, A4 page geometry, page-break
control, and figures that do not straddle a page boundary.

Usage:
    python scripts/build_paper_pdf.py            # writes build/paper_print.html
    # then render with any Chromium (Edge is what the dev machine has):
    #   msedge --headless=new --disable-gpu --no-pdf-header-footer \
    #          --virtual-time-budget=25000 \
    #          --print-to-pdf=GCNs-Net-Reassessed.pdf build/paper_print.html
"""
import base64
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "paper_print.html"
OUT.parent.mkdir(exist_ok=True)

src = (ROOT / "PAPER.md").read_text(encoding="utf-8")
body = markdown.markdown(src, extensions=["tables", "attr_list", "sane_lists", "smarty"])

for m in set(re.findall(r'src="(figures/[^"]+)"', body)):
    data = base64.b64encode((ROOT / m).read_bytes()).decode("ascii")
    body = body.replace(f'src="{m}"', f'src="data:image/png;base64,{data}"')

# Mark caption paragraphs explicitly: the whole paragraph is a single <em>.
body = re.sub(r"<p><em>(.*?)</em></p>",
              lambda m: '<p class="caption">' + m.group(1) + "</p>",
              body, flags=re.S)

body = body.replace("<h1>", '<h1 class="paper-title">', 1)

CSS = """
@page { size: A4; margin: 20mm 18mm 18mm 18mm; }

html, body {
  background: #FFFFFF; color: #16161A;
  font-family: "Source Serif 4", "Source Serif Pro", Georgia, serif;
  font-size: 10.2pt; line-height: 1.5;
  margin: 0; padding: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
* { box-sizing: border-box; }

h1.paper-title {
  font-size: 19pt; font-weight: 600; line-height: 1.2; letter-spacing: -.01em;
  margin: 0 0 10pt; padding-bottom: 9pt; border-bottom: 1pt solid #16161A;
  text-wrap: balance;
}
h1.paper-title + p em {
  font-style: normal; font-family: "IBM Plex Sans", Arial, sans-serif;
  font-size: 8.4pt; color: #7C7C86;
}

h2 {
  font-size: 12.6pt; font-weight: 600; margin: 17pt 0 6pt;
  padding-bottom: 2.5pt; border-bottom: .5pt solid #DEDEE3;
  break-after: avoid; page-break-after: avoid;
}
h3 {
  font-size: 10.8pt; font-weight: 600; margin: 12pt 0 4pt;
  break-after: avoid; page-break-after: avoid;
}
p { margin: 0 0 7pt; orphans: 3; widows: 3; }
ul, ol { padding-left: 14pt; margin: 0 0 7pt; }
li { margin-bottom: 2.5pt; }
strong { font-weight: 600; }

hr { border: none; border-top: .5pt solid #DEDEE3; margin: 14pt 0; }

table {
  width: 100%; border-collapse: collapse; margin: 9pt 0;
  font-family: "IBM Plex Sans", Arial, sans-serif; font-size: 8.1pt;
  font-variant-numeric: tabular-nums;
  break-inside: avoid; page-break-inside: avoid;
}
th, td { padding: 3.2pt 5pt; text-align: left; border-bottom: .4pt solid #EBEBEF; }
th {
  font-size: 6.9pt; letter-spacing: .05em; text-transform: uppercase;
  font-weight: 600; color: #4A4A52; background: #F2F2F4;
  border-bottom: .6pt solid #DEDEE3;
}
tbody tr:last-child td { border-bottom: .6pt solid #DEDEE3; }

pre {
  background: #F5F5F7; border: .5pt solid #DEDEE3; padding: 6pt 8pt;
  margin: 8pt 0; font-family: "IBM Plex Mono", Consolas, monospace;
  font-size: 7.6pt; line-height: 1.45; white-space: pre-wrap;
  break-inside: avoid; page-break-inside: avoid;
}
code {
  font-family: "IBM Plex Mono", Consolas, monospace; font-size: .88em;
  background: #F5F5F7; padding: .5pt 2pt;
}
pre code { background: none; padding: 0; font-size: inherit; }

img {
  width: 100%; height: auto; display: block; margin: 10pt 0 3pt;
  border: .5pt solid #DEDEE3;
  break-inside: avoid; page-break-inside: avoid;
}
p.caption {
  font-family: "IBM Plex Sans", Arial, sans-serif;
  font-size: 7.9pt; line-height: 1.4; color: #4A4A52;
  margin-top: 1pt; break-before: avoid; page-break-before: avoid;
}

blockquote {
  margin: 9pt 0; padding: 7pt 9pt; background: #FAFAFB;
  border: .5pt solid #DEDEE3; border-left: 2pt solid #7A2E2E;
  break-inside: avoid; page-break-inside: avoid;
}
blockquote p { margin: 0; font-size: 9.4pt; }
blockquote p + p { margin-top: 4pt; }

a { color: #7A2E2E; text-decoration: none; }

h2#references ~ p {
  font-size: 8.4pt; line-height: 1.4; color: #2E2E34;
  padding-left: 15pt; text-indent: -15pt; margin-bottom: 4.5pt;
  break-inside: avoid; page-break-inside: avoid;
}
"""

JS = """
for (const h of document.querySelectorAll('h2')) {
  if (h.textContent.trim().toLowerCase() === 'references') h.id = 'references';
}
"""

HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Reassessing GCNs-Net</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400&display=swap">
<style>{CSS}</style>
</head><body>
{body}
<script>{JS}</script>
</body></html>
"""

OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT} ({len(HTML)/1e6:.2f} MB)")
