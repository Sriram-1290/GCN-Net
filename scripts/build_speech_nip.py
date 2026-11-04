"""
Build Script for Presentation Defense Speech PDF - NIP Version (PW-1_Presentation_Speech-NIP.pdf)
Neil Gogte Institute of Technology (NGIT) · Department of Computer Science & Engineering
Major Project Work-1 (PW-1) · Batch 22 · Review-1 Defense (October 2026)

Converts PRESENTATION_SPEECH_NIP.md to a beautifully styled, print-ready PDF using headless Microsoft Edge.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)

SPEECH_CSS = """
@page {
    size: A4;
    margin: 11mm 12.5mm 11mm 12.5mm;
    @bottom-right {
        content: "Page " counter(page);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
        font-size: 8pt;
        color: #64748b;
    }
    @bottom-left {
        content: "NGIT CSE \\00b7  PW-1 Review-1 Defense Speech & Viva Guide (NIP) \\00b7  Batch 22";
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
        font-size: 8pt;
        color: #64748b;
    }
}

*, *:before, *:after {
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1e293b;
    background: #ffffff;
    line-height: 1.36;
    font-size: 8.85pt;
    margin: 0;
    padding: 0;
}

/* Institutional Header */
.doc-header {
    border-bottom: 2.5px solid #1a365d;
    padding-bottom: 10px;
    margin-bottom: 16px;
}
.inst-title {
    font-size: 14pt;
    font-weight: 800;
    color: #1a365d;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0 0 2px 0;
}
.inst-sub {
    font-size: 8.5pt;
    color: #475569;
    font-weight: 500;
    margin: 0 0 4px 0;
}
.inst-dept {
    font-size: 10pt;
    font-weight: 700;
    color: #2b6cb0;
    text-transform: uppercase;
    margin: 0 0 8px 0;
}
.doc-badge {
    display: inline-block;
    background: #1a365d;
    color: #ffffff;
    font-size: 8pt;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}
.doc-title {
    font-size: 12.5pt;
    font-weight: 800;
    color: #0f172a;
    margin: 8px 0 3px 0;
    line-height: 1.25;
}
.doc-subtitle {
    font-size: 9.5pt;
    font-weight: 600;
    color: #0f766e;
    margin: 0 0 10px 0;
}

/* Meta Information Grid */
.meta-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 16px;
    font-size: 8.5pt;
}
.meta-col p {
    margin: 2px 0;
}
.meta-label {
    font-weight: 700;
    color: #334155;
}

/* Typography & Headings */
h1 {
    font-size: 13pt;
    font-weight: 800;
    color: #1a365d;
    border-bottom: 1.5px solid #2b6cb0;
    padding-bottom: 4px;
    margin: 18px 0 8px 0;
    page-break-after: avoid;
}
h2 {
    font-size: 11pt;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 3px;
    margin: 14px 0 6px 0;
    page-break-after: avoid;
}
h3 {
    font-size: 10pt;
    font-weight: 700;
    color: #1e3a8a;
    margin: 10px 0 4px 0;
    page-break-after: avoid;
}
h4 {
    font-size: 9.2pt;
    font-weight: 700;
    color: #0f172a;
    margin: 8px 0 2px 0;
    page-break-after: avoid;
}

/* Slide Card Containers */
.slide-card {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-left: 4px solid #1a365d;
    border-radius: 5px;
    padding: 10px 12px;
    margin-bottom: 12px;
    page-break-inside: avoid;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
.slide-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 5px;
    margin-bottom: 6px;
}
.slide-title {
    font-size: 10.5pt;
    font-weight: 800;
    color: #1a365d;
    margin: 0;
}
.slide-meta {
    font-size: 8.2pt;
    font-weight: 600;
    color: #475569;
}
.speaker-badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 10px;
    font-size: 7.8pt;
    font-weight: 700;
    text-transform: uppercase;
}
.badge-s1 { background: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe; }
.badge-s2 { background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; }
.badge-s3 { background: #ccfbf1; color: #0f766e; border: 1px solid #99f6e4; }

/* Spoken Script Box */
blockquote {
    background: #f8fafc;
    border-left: 3px solid #2b6cb0;
    margin: 5px 0;
    padding: 5px 9px;
    font-style: italic;
    color: #1e293b;
    font-size: 8.85pt;
    line-height: 1.38;
    border-radius: 0 4px 4px 0;
}
blockquote p {
    margin: 2px 0;
}
blockquote strong {
    color: #0f172a;
    font-style: normal;
}

/* Callouts & Anchors */
.cue-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 4px;
    padding: 5px 8px;
    font-size: 8.2pt;
    color: #92400e;
    margin: 5px 0;
}
.cue-title {
    font-weight: 700;
    color: #b45309;
}

/* Slide Section Container */
.slide-section {
    break-inside: avoid;
    page-break-inside: avoid;
    margin-bottom: 8px;
}

/* QA Item Container */
.qa-item {
    break-inside: avoid;
    page-break-inside: avoid;
    margin-bottom: 6px;
}

/* Q&A Item Box */
.qa-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    padding: 6px 9px;
    margin-bottom: 7px;
    page-break-inside: avoid;
}
.qa-question {
    font-size: 9.4pt;
    font-weight: 800;
    color: #1a365d;
    margin: 0 0 3px 0;
}
.qa-trap {
    background: #fef2f2;
    border-left: 3px solid #ef4444;
    padding: 3px 7px;
    font-size: 8pt;
    color: #991b1b;
    margin: 3px 0 5px 0;
    border-radius: 0 3px 3px 0;
}
.qa-trap-label {
    font-weight: 700;
}
.qa-answer {
    background: #f0fdf4;
    border-left: 3px solid #10b981;
    padding: 5px 8px;
    font-size: 8.8pt;
    color: #064e3b;
    margin: 3px 0 0 0;
    border-radius: 0 3px 3px 0;
    line-height: 1.4;
}

/* Math Blocks & Display Equations */
.math-block {
    display: block;
    text-align: center;
    font-family: "Cambria Math", "Times New Roman", Georgia, serif;
    font-size: 10.2pt;
    color: #0f172a;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 6px 12px;
    margin: 6px auto;
    width: fit-content;
    max-width: 95%;
    line-height: 1.5;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0 12px 0;
    font-size: 8.2pt;
}
th {
    background: #1a365d;
    color: #ffffff;
    font-weight: 700;
    padding: 5px 7px;
    text-align: left;
    border: 1px solid #1a365d;
}
td {
    padding: 4px 7px;
    border: 1px solid #cbd5e1;
    vertical-align: top;
}
tr:nth-child(even) td {
    background: #f8fafc;
}

code {
    font-family: Consolas, "Courier New", monospace;
    font-size: 8.2pt;
    background: #f1f5f9;
    padding: 1px 3px;
    border-radius: 3px;
    color: #0f172a;
}

hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 12px 0;
}

ul, ol {
    margin: 3px 0;
    padding-left: 18px;
}
li {
    margin-bottom: 2px;
}
"""

def generate_html_from_markdown(md_text):
    """Parses PRESENTATION_SPEECH_NIP.md and wraps in rich presentation CSS."""
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "smarty"])

    # Wrap each slide section to prevent awkward breaks
    html_body = re.sub(
        r'(<h3>Slide \d+:.*?)(?=(?:<h[23]>|<hr|\Z))',
        r'<div class="slide-section">\1</div>\n',
        html_body,
        flags=re.DOTALL
    )

    # Wrap each QA item to prevent question-answer fragmentation
    html_body = re.sub(
        r'(<h4>Q\d+:.*?)(?=(?:<h[234]>|<hr|\Z))',
        r'<div class="qa-item">\1</div>\n',
        html_body,
        flags=re.DOTALL
    )

    styled_header = """
    <div class="doc-header">
        <div class="inst-title">Neil Gogte Institute of Technology</div>
        <div class="inst-sub">Approved by AICTE &middot; Affiliated to Osmania University &middot; Kachavanisingaram, Hyderabad</div>
        <div class="inst-dept">Department of Computer Science &amp; Engineering</div>
        <div class="doc-badge">Major Project Work-1 (PW-1) &middot; Review-1 Defense (NIP Formulation Phase)</div>
        <div class="doc-title">Dynamic Graph Convolutional Network with Temporal Attention for Subject-Independent EEG Motor Imagery Decoding</div>
        <div class="doc-subtitle">Architectural Realization: GBP-Net (Graph Band-Power Network with Cross-Subject Pre-Training Framework)</div>
    </div>

    <div class="meta-grid">
        <div class="meta-col">
            <p><span class="meta-label">Project Team (Batch 22):</span></p>
            <p>&bull; <strong>245323753054</strong> &ndash; Saligram Sriram Shravan (Speaker 1)</p>
            <p>&bull; <strong>245323753045</strong> &ndash; Parcha Seetha Ramarao (Speaker 2)</p>
            <p>&bull; <strong>245323753030</strong> &ndash; M Priyansh (Speaker 3)</p>
        </div>
        <div class="meta-col">
            <p><span class="meta-label">Academic Supervision:</span></p>
            <p>&bull; <strong>Project Guide:</strong> Dr. Gagan Kumar</p>
            <p>&bull; <strong>Project Coordinator:</strong> Mr. M. Manoher Rao</p>
            <p>&bull; <strong>Head of Department:</strong> Dr. K. Vinuthna</p>
            <p>&bull; <strong>Defense Date &amp; Venue:</strong> 5th October 2026 &middot; Room 301, B-Block</p>
        </div>
    </div>
    """

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PW-1 Review-1 Presentation Defense Speech Guide (NIP) - NGIT Batch 22</title>
<style>{SPEECH_CSS}</style>
</head>
<body>
{styled_header}
{html_body}
</body>
</html>
"""
    return full_html


def build_pdf(edge_bin):
    md_file = ROOT / "docs" / "PRESENTATION_SPEECH_NIP.md"
    if not md_file.exists():
        raise FileNotFoundError(f"{md_file} not found!")

    md_content = md_file.read_text(encoding="utf-8")
    
    # Strip existing top title in MD to prevent duplicate header
    lines = md_content.splitlines()
    filtered_lines = []
    skip = False
    for line in lines:
        if line.startswith("# PW-1 Review-1 Presentation Defense Speech Guide"):
            skip = True
            continue
        if skip and line.startswith("## Team Roster"):
            skip = False
        if not skip:
            filtered_lines.append(line)
    
    processed_md = "\n".join(filtered_lines)

    html_content = generate_html_from_markdown(processed_md)
    out_html = BUILD / "presentation_speech_nip_print.html"
    out_html.write_text(html_content, encoding="utf-8")
    print(f"Wrote print-ready HTML to: {out_html}")

    out_pdf = ROOT / "docs" / "PW-1_Presentation_Speech-NIP.pdf"
    temp_pdf = BUILD / "temp_speech_nip.pdf"
    if temp_pdf.exists():
        temp_pdf.unlink()

    cmd = [
        edge_bin,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={temp_pdf.resolve()}",
        out_html.resolve().as_uri()
    ]
    print("Executing:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and temp_pdf.exists() and temp_pdf.stat().st_size > 0:
        import shutil
        try:
            shutil.copy2(temp_pdf, out_pdf)
            pdf_size = out_pdf.stat().st_size
            print("=" * 70)
            print(f"Successfully generated PDF: {out_pdf}")
            print(f"File Size: {pdf_size:,} bytes")
            print("=" * 70)
            return 0
        except PermissionError:
            print(f"ERROR: Cannot overwrite {out_pdf} because it is currently open in a PDF viewer.")
            print(f"Saved latest render to {temp_pdf} instead.")
            return 1
    else:
        print("Error generating PDF:", res.stderr or "Output file was not created by Edge.")
        return 1


def main():
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    edge_bin = next((p for p in edge_paths if os.path.exists(p)), None)
    if not edge_bin:
        print("Error: Microsoft Edge executable not found!")
        sys.exit(1)

    sys.exit(build_pdf(edge_bin))


if __name__ == "__main__":
    main()
