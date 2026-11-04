"""Render REVIEW_PAPER_NIP.md to a print-ready PDF using headless Microsoft Edge.

Parses citations [@key], builds the numbered reference list in IEEE style,
embeds local figures as base64 data URIs, and invokes Edge to generate Review-Paper-NIP.pdf.

Usage:
    python scripts/build_review_pdf_2.py
"""
import base64
import os
import re
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)

REFS = {
    "pfurtscheller1999": 'G. Pfurtscheller and F. H. Lopes da Silva, "Event-related EEG/MEG '
                         'synchronization and desynchronization: Basic principles," <i>Clin. '
                         'Neurophysiol.</i>, vol. 110, no. 11, pp. 1842–1857, 1999, '
                         'doi: 10.1016/S1388-2457(99)00141-8.',
    "ramoser2000": 'H. Ramoser, J. Müller-Gerking, and G. Pfurtscheller, "Optimal spatial '
                   'filtering of single trial EEG during imagined hand movement," <i>IEEE Trans. '
                   'Rehabil. Eng.</i>, vol. 8, no. 4, pp. 441–446, 2000, '
                   'doi: 10.1109/86.895946.',
    "blankertz2008": 'B. Blankertz, R. Tomioka, S. Lemm, M. Kawanabe, and K.-R. Müller, '
                     '"Optimizing spatial filters for robust EEG single-trial analysis," <i>IEEE '
                     'Signal Process. Mag.</i>, vol. 25, no. 1, pp. 41–56, 2008, '
                     'doi: 10.1109/MSP.2008.4408441.',
    "ang2008": 'K. K. Ang, Z. Y. Chin, H. Zhang, and C. Guan, "Filter bank common spatial '
               'pattern (FBCSP) in brain-computer interface," in <i>Proc. IEEE Int. Joint Conf. '
               'Neural Netw. (IJCNN)</i>, 2008, pp. 2390–2397, '
               'doi: 10.1109/IJCNN.2008.4634130.',
    "lawhern2018": 'V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and '
                   'B. J. Lance, "EEGNet: A compact convolutional neural network for EEG-based '
                   'brain–computer interfaces," <i>J. Neural Eng.</i>, vol. 15, no. 5, Art. no. '
                   '056013, 2018, doi: 10.1088/1741-2552/aace8c.',
    "schirrmeister2017": 'R. T. Schirrmeister <i>et al.</i>, "Deep learning with convolutional '
                         'neural networks for EEG decoding and visualization," <i>Hum. Brain '
                         'Mapp.</i>, vol. 38, no. 11, pp. 5391–5420, 2017, '
                         'doi: 10.1002/hbm.23730.',
    "mane2021": 'R. Mane, E. Chew, K. Chua, and K. K. Ang, "FBCNet: A multi-view convolutional '
                'neural network for brain-computer interface," arXiv:2104.01233, 2021.',
    "defferrard2016": 'M. Defferrard, X. Bresson, and P. Vandergheynst, "Convolutional neural '
                      'networks on graphs with fast localized spectral filtering," in <i>Adv. '
                      'Neural Inf. Process. Syst. (NeurIPS)</i>, vol. 29, 2016.',
    "dhillon2007": 'I. S. Dhillon, Y. Guan, and B. Kulis, "Weighted graph cuts without '
                   'eigenvectors: A multilevel approach," <i>IEEE Trans. Pattern Anal. Mach. '
                   'Intell.</i>, vol. 29, no. 11, pp. 1944–1957, 2007, '
                   'doi: 10.1109/TPAMI.2007.1115.',
    "hou2024": 'Y. Hou <i>et al.</i>, "GCNs-Net: A graph convolutional neural network approach '
               'for decoding time-resolved EEG motor imagery signals," <i>IEEE Trans. Neural '
               'Netw. Learn. Syst.</i>, vol. 35, no. 6, pp. 7312–7323, 2024, '
               'doi: 10.1109/TNNLS.2022.3202569.',
    "hou2022": 'Y. Hou <i>et al.</i>, "Deep feature mining via the attention-based bidirectional '
               'long short term memory graph convolutional neural network for human motor imagery '
               'recognition," <i>Front. Bioeng. Biotechnol.</i>, vol. 9, Art. no. 706229, 2022, '
               'doi: 10.3389/fbioe.2021.706229.',
    "song2020": 'T. Song, W. Zheng, P. Song, and Z. Cui, "EEG emotion recognition using dynamical '
                'graph convolutional neural networks," <i>IEEE Trans. Affect. Comput.</i>, vol. 11, '
                'no. 3, pp. 532–541, 2020, doi: 10.1109/TAFFC.2018.2817622.',
    "zhong2020": 'P. Zhong, D. Wang, and C. Miao, "EEG-based emotion recognition using regularized '
                 'graph neural networks," <i>IEEE Trans. Affect. Comput.</i>, vol. 13, no. 3, '
                 'pp. 1290–1301, 2022, doi: 10.1109/TAFFC.2020.2994159.',
    "musallam2021": 'Y. K. Musallam <i>et al.</i>, "Electroencephalography-based motor imagery '
                    'classification using temporal convolutional network fused with graph '
                    'convolution," <i>J. Neural Eng.</i>, vol. 18, no. 4, Art. no. 046027, 2021, '
                    'doi: 10.1088/1741-2552/ac05eb.',
    "sun2021": 'B. Sun, F. He, J. Zou, and Y. Ke, "Adaptive spatial-temporal graph convolutional '
               'networks for motor imagery classification," <i>IEEE Trans. Neural Syst. Rehabil. '
               'Eng.</i>, vol. 30, pp. 1888–1897, 2022, doi: 10.1109/TNSRE.2022.3188561.',
    "deng2025": 'X. Deng, D. Xu, H. Huo, X. Hong, and B. Liu, "Multi-level hierarchical dynamic '
                'graph convolutional networks for motor imagery EEG analysis," '
                '<i>Neurocomputing</i>, vol. 626, Art. no. 129594, 2025, '
                'doi: 10.1016/j.neucom.2025.129594.',
    "meng2026": 'M. Meng <i>et al.</i>, "SAMGCN: A self-adaptive multilevel graph convolutional '
                'network for motor imagery classification," <i>Biomed. Signal Process. '
                'Control</i>, vol. 116, Art. no. 109506, 2026, doi: 10.1016/j.bspc.2026.109506.',
    "klepl2024": 'D. Klepl, M. Wu, and F. He, "Graph neural network-based EEG classification: A '
                 'survey," <i>IEEE Trans. Neural Syst. Rehabil. Eng.</i>, vol. 32, pp. 493–503, '
                 '2024, doi: 10.1109/TNSRE.2024.3355750.',
    "saibene2024": 'A. Saibene, H. Ghaemi, and E. Dagdevir, "Deep learning in motor imagery EEG '
                   'signal decoding: A systematic review," <i>Neurocomputing</i>, vol. 610, '
                   'art. 128577, 2024, doi: 10.1016/j.neucom.2024.128577.',
    "wu2022": 'D. Wu, Y. Xu, and B.-L. Lu, "Transfer learning for EEG-based brain-computer '
              'interfaces: A review of progress made since 2016," <i>IEEE Trans. Cogn. Devel. '
              'Syst.</i>, vol. 14, no. 1, pp. 4–19, 2022, doi: 10.1109/TCDS.2020.3007453.',
    "kapoor2023": 'S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in '
                  'machine-learning-based science," <i>Patterns</i>, vol. 4, no. 9, Art. no. '
                  '100804, 2023, doi: 10.1016/j.patter.2023.100804.',
    "chevallier2024": 'S. Chevallier <i>et al.</i>, "The largest EEG-based BCI reproducibility '
                      'study for open science: The MOABB benchmark," arXiv:2404.15319, 2024.',
    "vasques2026": 'X. Vasques, P. Barbaste, and O. Oullier, "Subject-level heterogeneity in EEG '
                   'motor imagery decoding: A large-scale benchmark and portfolio-based reduction '
                   'of the search space," arXiv:2607.22778, 2026.',
    "eder2024": 'M. Eder, J. Xu, and M. Grosse-Wentrup, "Benchmarking brain–computer interface '
                'algorithms: Riemannian approaches vs convolutional neural networks," <i>J. '
                'Neural Eng.</i>, vol. 21, no. 4, Art. no. 044002, 2024, '
                'doi: 10.1088/1741-2552/ad6793.',
    "page2021": 'M. J. Page <i>et al.</i>, "The PRISMA 2020 statement: An updated guideline for '
                'reporting systematic reviews," <i>BMJ</i>, vol. 372, Art. no. n71, 2021, '
                'doi: 10.1136/bmj.n71.',
    "goldberger2000": 'A. L. Goldberger <i>et al.</i>, "PhysioBank, PhysioToolkit, and PhysioNet: '
                      'Components of a new research resource for complex physiologic signals," '
                      '<i>Circulation</i>, vol. 101, no. 23, pp. e215–e220, 2000, '
                      'doi: 10.1161/01.CIR.101.23.e215.',
    "blankertz2010": 'B. Blankertz <i>et al.</i>, "Neurophysiological predictor of SMR-based BCI '
                     'performance," <i>NeuroImage</i>, vol. 51, no. 4, pp. 1303–1309, 2010, '
                     'doi: 10.1016/j.neuroimage.2010.03.022.',
    "schalk2004": 'G. Schalk, D. J. McFarland, T. Hinterberger, N. Birbaumer, and J. R. Wolpaw, '
                  '"BCI2000: A general-purpose brain-computer interface (BCI) system," <i>IEEE '
                  'Trans. Biomed. Eng.</i>, vol. 51, no. 6, pp. 1034–1043, 2004, '
                  'doi: 10.1109/TBME.2004.827072.',
    "eegdl": 'S. Jia, "EEG-DL: A deep learning library for EEG tasks (signals) classification," '
             'GitHub repository, 2021. [Online]. Available: https://github.com/SuperBruceJia/EEG-DL',
    "kobler2022": 'R. J. Kobler, J. Hirayama, Q. Zhao, and M. Kawanabe, "SPD domain-specific '
                  'batch normalization to crack interpretable unsupervised domain adaptation in '
                  'EEG," in <i>Adv. Neural Inf. Process. Syst. (NeurIPS)</i>, vol. 35, 2022.',
    "jiang2024": 'X. Jiang, L. Meng, X. Chen, Y. Xu, and D. Wu, "CSP-Net: Common spatial '
                 'pattern empowered neural networks for EEG-based motor imagery classification," '
                 'arXiv:2411.11879, 2024.',
    "xie2023": 'Y. Xie <i>et al.</i>, "Cross-dataset transfer learning for motor imagery signal '
               'classification via multi-task learning and pre-training," <i>J. Neural Eng.</i>, '
               'vol. 20, no. 5, Art. no. 056037, 2023, doi: 10.1088/1741-2552/acfe9c.',
    "jiang2024b": 'W.-B. Jiang, L.-M. Zhao, and B.-L. Lu, "Large brain model for learning generic '
                  'representations with tremendous EEG data in BCI," in <i>Proc. Int. Conf. '
                  'Learn. Represent. (ICLR)</i>, 2024.',
    "he2020": 'H. He and D. Wu, "Transfer learning for brain–computer interfaces: A Euclidean '
              'space data alignment approach," <i>IEEE Trans. Biomed. Eng.</i>, vol. 67, no. 2, '
              'pp. 399–410, 2020, doi: 10.1109/TBME.2019.2913914.',
    "wu2025": 'D. Wu, "Revisiting Euclidean alignment for transfer learning in EEG-based '
              'brain–computer interfaces," <i>J. Neural Eng.</i>, vol. 22, no. 3, Art. no. '
              '031005, 2025, doi: 10.1088/1741-2552/addd49.',
    "rommel2022": 'C. Rommel, J. Paillard, T. Moreau, and A. Gramfort, "Data augmentation for '
                  'learning predictive models on EEG: A systematic comparison," <i>J. Neural '
                  'Eng.</i>, vol. 19, no. 6, Art. no. 066020, 2022, '
                  'doi: 10.1088/1741-2552/aca220.',
}

CITE = re.compile(r"\[(@[\w]+(?:\s*;\s*@[\w]+)*)\]")


def number_citations(text: str) -> tuple[str, list[str]]:
    order: list[str] = []

    def repl(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        nums = []
        for k in keys:
            if k not in REFS:
                raise SystemExit(f"cited but not defined: {k}")
            if k not in order:
                order.append(k)
            nums.append(order.index(k) + 1)
        return "[" + ", ".join(str(n) for n in sorted(nums)) + "]"

    out = CITE.sub(repl, text)
    unused = set(REFS) - set(order)
    if unused:
        raise SystemExit(f"defined but never cited: {sorted(unused)}")
    return out, order


def embed_figures(html: str) -> str:
    """Embed local PNG figures as base64 images."""
    def repl(m):
        alt, src = m.group(1), m.group(2)
        fig_path = ROOT / "docs" / src
        if not fig_path.exists():
            raise FileNotFoundError(f"Figure file not found: {fig_path}")
        data = base64.b64encode(fig_path.read_bytes()).decode("ascii")
        return (f'<figure><img src="data:image/png;base64,{data}" alt="{alt}">'
                f'<figcaption>{alt}</figcaption></figure>')
    return re.sub(r'<p><img alt="([^"]*)" src="([^"]+)" ?/?></p>', repl, html)


CSS = """
@page { size: A4 portrait; margin: 16.5mm 15mm 16.5mm 15mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font: 9.9pt/1.44 "Times New Roman", Georgia, serif; color: #111; background: #fff; margin: 0; }
.head { text-align: center; margin-bottom: 4.5mm; }
.head h1 { font-size: 15pt; line-height: 1.22; margin: 0 0 2.5mm; font-weight: bold; text-wrap: balance; }
.head .au { font-size: 10.5pt; font-weight: bold; margin: 0 0 1.2mm; }
.head .af { font-size: 9pt; font-style: italic; color: #333; margin: 0; line-height: 1.32; }
hr { border: none; border-top: 0.8pt solid #111; margin: 3.5mm 0 2.5mm; }
.abstract { font-size: 9pt; line-height: 1.39; margin-bottom: 3.5mm; text-align: justify; }
.abstract strong { font-size: 9.4pt; }
.keywords { font-size: 8.6pt; margin-bottom: 4.5mm; }
h2 { font-size: 11.2pt; margin: 4mm 0 1.8mm; text-transform: uppercase; letter-spacing: 0.04em; border-bottom: 0.5pt solid #888; padding-bottom: 0.8mm; break-after: avoid; }
h3 { font-size: 10.1pt; margin: 3mm 0 1.2mm; break-after: avoid; }
p { margin: 0 0 2.2mm; text-align: justify; text-justify: inter-word; }
ul, ol { margin: 0 0 2.5mm 3.5mm; padding-left: 3mm; }
li { margin-bottom: 1.2mm; text-align: justify; }
table { width: 100%; border-collapse: collapse; font-size: 8.2pt; margin: 2.5mm 0 3.5mm; break-inside: avoid; }
th, td { border: 0.5pt solid #555; padding: 1.5mm 2mm; text-align: left; vertical-align: top; }
th { background: #f2f2f5; font-weight: bold; font-size: 8.4pt; }
tr:nth-child(even) td { background: #fafafa; }
figure { margin: 3mm 0; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; height: auto; }
figcaption { font-size: 8.1pt; color: #333; margin-top: 1.2mm; line-height: 1.28; text-align: justify; }
ol.refs { font-size: 7.7pt; line-height: 1.25; padding-left: 5mm; column-count: 2; column-gap: 6mm; }
ol.refs li { margin-bottom: 1.0mm; text-align: left; break-inside: avoid; }
.callout { background: #fbf7f7; border-left: 2.5pt solid #8a2d2d; padding: 2mm 3mm; margin: 2mm 0 3mm; font-size: 8.6pt; }
"""


def build_paper(edge_bin):
    src_file = ROOT / "docs" / "REVIEW_PAPER_NIP.md"
    if not src_file.exists():
        raise FileNotFoundError(f"{src_file} does not exist yet")

    src = src_file.read_text(encoding="utf-8")

    # Frontmatter extraction
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", src, flags=re.S)
    if not m:
        raise ValueError("Expected YAML frontmatter in REVIEW_PAPER_NIP.md")
    frontmatter, body_md = m.group(1), m.group(2)

    title_match = re.search(r'title:\s*"([^"]+)"', frontmatter)
    title = title_match.group(1) if title_match else "Review Paper"

    affil_match = re.search(r'affiliation:\s*"([^"]+)"', frontmatter)
    affiliation = affil_match.group(1) if affil_match else ""

    authors = re.findall(r"-\s+(?:name:\s*)?([^\n]+)", re.search(r"authors:(.*?)(?:affiliation:|$)", frontmatter, re.S).group(1))
    authors_str = ", ".join(a.strip().strip('"').strip("'") for a in authors if a.strip())

    # Process citations
    body_md, order = number_citations(body_md)

    # Render Markdown to HTML
    body_html = markdown.markdown(body_md, extensions=["tables", "sane_lists", "smarty", "attr_list"])
    body_html = embed_figures(body_html)

    # Build reference section
    ref_items = "".join(f"<li>{REFS[k]}</li>\n" for k in order)
    refs_html = f'<h2>References</h2>\n<ol class="refs">\n{ref_items}</ol>'
    body_html = body_html.replace("<!-- REFERENCES -->", refs_html)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<div class="head">
  <h1>{title}</h1>
  <div class="au">{authors_str}</div>
  <div class="af">{affiliation}</div>
</div>
<hr>
{body_html}
</body>
</html>
"""
    out_html = BUILD / "review_paper_nip_print.html"
    out_html.write_text(full_html, encoding="utf-8")
    print(f"Wrote HTML to {out_html}")

    if not edge_bin:
        print("Warning: Microsoft Edge executable not found; skipping automated PDF render.")
        return 0

    out_pdf = ROOT / "docs" / "Review-Paper-NIP.pdf"
    temp_pdf = BUILD / "temp_review_paper_nip.pdf"
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
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and temp_pdf.exists() and temp_pdf.stat().st_size > 0:
        import shutil
        try:
            shutil.copy2(temp_pdf, out_pdf)
            print(f"Successfully generated PDF: {out_pdf} ({out_pdf.stat().st_size:,} bytes)")
        except PermissionError:
            print(f"ERROR: Cannot overwrite {out_pdf} because it is currently open in a PDF viewer.")
            print(f"Saved latest render to {temp_pdf} instead.")
            return 1
    else:
        print("PDF generation error:", res.stderr or "Output file was not created by Edge.")
        return 1
    return 0


ANALYSIS_CSS = """
@page { size: A4 portrait; margin: 6.5mm 9.5mm 6.5mm 9.5mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font: 9.4pt/1.28 "Times New Roman", Georgia, serif; color: #111; background: #fff; margin: 0; }
h1 { font-size: 13.5pt; text-align: center; margin: 0 0 1.8mm; font-weight: bold; text-transform: uppercase; letter-spacing: 0.04em; }
p { margin: 0 0 2.5mm; text-align: center; font-size: 8.5pt; line-height: 1.35; color: #222; }
table { width: 100%; border-collapse: collapse; font-size: 8.7pt; line-height: 1.28; margin: 0; }
th, td { border: 0.55pt solid #444; padding: 1.05mm 1.5mm; text-align: justify; vertical-align: top; }
th { background: #f0f0f4; font-weight: bold; font-size: 9.0pt; }
td:first-child, th:first-child { width: 22%; font-weight: bold; text-align: left; }
tr:nth-child(even) td { background: #fafafa; }
code { font-family: Consolas, monospace; font-size: 8.4pt; }
"""


def build_analysis(edge_bin):
    src_file = ROOT / "docs" / "REVIEW_ANALYSIS_NIP.md"
    if not src_file.exists():
        raise FileNotFoundError(f"{src_file} does not exist yet")

    src = src_file.read_text(encoding="utf-8")
    body_html = markdown.markdown(src, extensions=["tables", "smarty"])

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Review-Paper Analysis - GBP-Net</title>
<style>{ANALYSIS_CSS}</style>
</head>
<body>
{body_html}
</body>
</html>
"""
    out_html = BUILD / "review_analysis_nip_print.html"
    out_html.write_text(full_html, encoding="utf-8")
    print(f"Wrote HTML to {out_html}")

    if not edge_bin:
        print("Warning: Microsoft Edge executable not found; skipping automated analysis PDF render.")
        return 0

    out_pdf = ROOT / "docs" / "Review-Paper-Analysis-NIP.pdf"
    temp_pdf = BUILD / "temp_review_analysis_nip.pdf"
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
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and temp_pdf.exists() and temp_pdf.stat().st_size > 0:
        import shutil
        try:
            shutil.copy2(temp_pdf, out_pdf)
            print(f"Successfully generated PDF: {out_pdf} ({out_pdf.stat().st_size:,} bytes)")
        except PermissionError:
            print(f"ERROR: Cannot overwrite {out_pdf} because it is currently open in a PDF viewer.")
            print(f"Saved latest render to {temp_pdf} instead.")
            return 1
    else:
        print("PDF generation error:", res.stderr or "Output file was not created by Edge.")
        return 1
    return 0


def main():
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    edge_bin = next((p for p in edge_paths if os.path.exists(p)), None)

    ret1 = build_paper(edge_bin)
    ret2 = build_analysis(edge_bin)
    return ret1 or ret2


if __name__ == "__main__":
    sys.exit(main())
