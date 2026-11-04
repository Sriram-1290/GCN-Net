"""Render the review paper and its one-page analysis to print-ready HTML.

REVIEW_PAPER.md cites by key -- [@hou2024; @kapoor2023] -- and this script numbers the
keys IEEE-style in order of first appearance and writes the reference list. Numbering
29 references by hand across a long document with tables is exactly where errors
creep in, and a renumbering after any edit would be manual and silent. Here it is
mechanical, and the script refuses to build if a key is cited but undefined or
defined but never cited.

Every reference below was checked against Crossref or arXiv, except the GitHub
repository, which is cited as software.

Usage:
    python scripts/build_review_pdf.py
    # then, with any Chromium:
    #   msedge --headless=new --disable-gpu --no-pdf-header-footer \
    #          --print-to-pdf=docs/Review-Paper.pdf build/review_print.html
    #   msedge ... --print-to-pdf=docs/Review-Paper-Analysis.pdf build/review_analysis_print.html
"""
import base64
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)

# IEEE style. More than six authors -> first author "et al.", per the IEEE guide.
REFS = {
    "hou2024": 'Y. Hou <i>et al.</i>, "GCNs-Net: A graph convolutional neural network approach '
               'for decoding time-resolved EEG motor imagery signals," <i>IEEE Trans. Neural '
               'Netw. Learn. Syst.</i>, vol. 35, no. 6, pp. 7312–7323, 2024, '
               'doi: 10.1109/TNNLS.2022.3202569.',
    "hou2022": 'Y. Hou <i>et al.</i>, "Deep feature mining via the attention-based bidirectional '
               'long short term memory graph convolutional neural network for human motor imagery '
               'recognition," <i>Front. Bioeng. Biotechnol.</i>, vol. 9, Art. no. 706229, 2022, '
               'doi: 10.3389/fbioe.2021.706229.',
    "schalk2004": 'G. Schalk, D. J. McFarland, T. Hinterberger, N. Birbaumer, and J. R. Wolpaw, '
                  '"BCI2000: A general-purpose brain-computer interface (BCI) system," <i>IEEE '
                  'Trans. Biomed. Eng.</i>, vol. 51, no. 6, pp. 1034–1043, 2004, '
                  'doi: 10.1109/TBME.2004.827072.',
    "deng2025": 'X. Deng, D. Xu, H. Huo, X. Hong, and B. Liu, "Multi-level hierarchical dynamic '
                'graph convolutional networks for motor imagery EEG analysis," '
                '<i>Neurocomputing</i>, vol. 626, Art. no. 129594, 2025, '
                'doi: 10.1016/j.neucom.2025.129594.',
    "meng2026": 'M. Meng <i>et al.</i>, "SAMGCN: A self-adaptive multilevel graph convolutional '
                'network for motor imagery classification," <i>Biomed. Signal Process. '
                'Control</i>, vol. 116, Art. no. 109506, 2026, doi: 10.1016/j.bspc.2026.109506.',
    "defferrard2016": 'M. Defferrard, X. Bresson, and P. Vandergheynst, "Convolutional neural '
                      'networks on graphs with fast localized spectral filtering," in <i>Adv. '
                      'Neural Inf. Process. Syst. (NeurIPS)</i>, vol. 29, 2016.',
    "goldberger2000": 'A. L. Goldberger <i>et al.</i>, "PhysioBank, PhysioToolkit, and PhysioNet: '
                      'Components of a new research resource for complex physiologic signals," '
                      '<i>Circulation</i>, vol. 101, no. 23, pp. e215–e220, 2000, '
                      'doi: 10.1161/01.CIR.101.23.e215.',
    "kapoor2023": 'S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in '
                  'machine-learning-based science," <i>Patterns</i>, vol. 4, no. 9, Art. no. '
                  '100804, 2023, doi: 10.1016/j.patter.2023.100804.',
    "dhillon2007": 'I. S. Dhillon, Y. Guan, and B. Kulis, "Weighted graph cuts without '
                   'eigenvectors: A multilevel approach," <i>IEEE Trans. Pattern Anal. Mach. '
                   'Intell.</i>, vol. 29, no. 11, pp. 1944–1957, 2007, '
                   'doi: 10.1109/TPAMI.2007.1115.',
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
    "pfurtscheller1999": 'G. Pfurtscheller and F. H. Lopes da Silva, "Event-related EEG/MEG '
                         'synchronization and desynchronization: Basic principles," <i>Clin. '
                         'Neurophysiol.</i>, vol. 110, no. 11, pp. 1842–1857, 1999, '
                         'doi: 10.1016/S1388-2457(99)00141-8.',
    "lawhern2018": 'V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and '
                   'B. J. Lance, "EEGNet: A compact convolutional neural network for EEG-based '
                   'brain–computer interfaces," <i>J. Neural Eng.</i>, vol. 15, no. 5, Art. no. '
                   '056013, 2018, doi: 10.1088/1741-2552/aace8c.',
    "schirrmeister2017": 'R. T. Schirrmeister <i>et al.</i>, "Deep learning with convolutional '
                         'neural networks for EEG decoding and visualization," <i>Hum. Brain '
                         'Mapp.</i>, vol. 38, no. 11, pp. 5391–5420, 2017, '
                         'doi: 10.1002/hbm.23730.',
    "chevallier2024": 'S. Chevallier <i>et al.</i>, "The largest EEG-based BCI reproducibility '
                      'study for open science: The MOABB benchmark," arXiv:2404.15319, 2024.',
    "vasques2026": 'X. Vasques, P. Barbaste, and O. Oullier, "Subject-level heterogeneity in EEG '
                   'motor imagery decoding: A large-scale benchmark and portfolio-based reduction '
                   'of the search space," arXiv:2607.22778, 2026.',
    "blankertz2010": 'B. Blankertz <i>et al.</i>, "Neurophysiological predictor of SMR-based BCI '
                     'performance," <i>NeuroImage</i>, vol. 51, no. 4, pp. 1303–1309, 2010, '
                     'doi: 10.1016/j.neuroimage.2010.03.022.',
    "xie2023": 'Y. Xie <i>et al.</i>, "Cross-dataset transfer learning for motor imagery signal '
               'classification via multi-task learning and pre-training," <i>J. Neural Eng.</i>, '
               'vol. 20, no. 5, Art. no. 056037, 2023, doi: 10.1088/1741-2552/acfe9c.',
    "jiang2024": 'X. Jiang, L. Meng, X. Chen, Y. Xu, and D. Wu, "CSP-Net: Common spatial '
                 'pattern empowered neural networks for EEG-based motor imagery classification," '
                 'arXiv:2411.11879, 2024.',
    "kobler2022": 'R. J. Kobler, J. Hirayama, Q. Zhao, and M. Kawanabe, "SPD domain-specific '
                  'batch normalization to crack interpretable unsupervised domain adaptation in '
                  'EEG," in <i>Adv. Neural Inf. Process. Syst. (NeurIPS)</i>, vol. 35, 2022.',
    "rommel2022": 'C. Rommel, J. Paillard, T. Moreau, and A. Gramfort, "Data augmentation for '
                  'learning predictive models on EEG: A systematic comparison," <i>J. Neural '
                  'Eng.</i>, vol. 19, no. 6, Art. no. 066020, 2022, '
                  'doi: 10.1088/1741-2552/aca220.',
    "he2020": 'H. He and D. Wu, "Transfer learning for brain–computer interfaces: A Euclidean '
              'space data alignment approach," <i>IEEE Trans. Biomed. Eng.</i>, vol. 67, no. 2, '
              'pp. 399–410, 2020, doi: 10.1109/TBME.2019.2913914.',
    "wu2025": 'D. Wu, "Revisiting Euclidean alignment for transfer learning in EEG-based '
              'brain–computer interfaces," <i>J. Neural Eng.</i>, vol. 22, no. 3, Art. no. '
              '031005, 2025, doi: 10.1088/1741-2552/addd49.',
    "eder2024": 'M. Eder, J. Xu, and M. Grosse-Wentrup, "Benchmarking brain–computer interface '
                'algorithms: Riemannian approaches vs convolutional neural networks," <i>J. '
                'Neural Eng.</i>, vol. 21, no. 4, Art. no. 044002, 2024, '
                'doi: 10.1088/1741-2552/ad6793.',
    "mane2021": 'R. Mane, E. Chew, K. Chua, and K. K. Ang, "FBCNet: A multi-view convolutional '
                'neural network for brain-computer interface," arXiv:2104.01233, 2021.',
    "jiang2024b": 'W.-B. Jiang, L.-M. Zhao, and B.-L. Lu, "Large brain model for learning generic '
                  'representations with tremendous EEG data in BCI," in <i>Proc. Int. Conf. '
                  'Learn. Represent. (ICLR)</i>, 2024.',
    "klepl2024": 'D. Klepl, M. Wu, and F. He, "Graph neural network-based EEG classification: A '
                 'survey," <i>IEEE Trans. Neural Syst. Rehabil. Eng.</i>, vol. 32, pp. 493–503, '
                 '2024, doi: 10.1109/TNSRE.2024.3355750.',
    "saibene2024": 'A. Saibene, H. Ghaemi, and E. Dagdevir, "Deep learning in motor imagery EEG '
                   'signal decoding: A systematic review," <i>Neurocomputing</i>, vol. 610, '
                   'art. 128577, 2024, doi: 10.1016/j.neucom.2024.128577.',
    "wu2022": 'D. Wu, Y. Xu, and B.-L. Lu, "Transfer learning for EEG-based brain-computer '
              'interfaces: A review of progress made since 2016," <i>IEEE Trans. Cogn. Devel. '
              'Syst.</i>, vol. 14, no. 1, pp. 4–19, 2022, doi: 10.1109/TCDS.2020.3007453.',
    "eegdl": 'S. Jia, "EEG-DL: A deep learning library for EEG tasks (signals) classification," '
             'GitHub repository. [Online]. Available: https://github.com/SuperBruceJia/EEG-DL',
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
    """`![caption](src)` -> a captioned figure with the image inlined."""
    def repl(m):
        alt, src = m.group(1), m.group(2)
        data = base64.b64encode((ROOT / "docs" / src).read_bytes()).decode("ascii")
        return (f'<figure><img src="data:image/png;base64,{data}" alt="">'
                f'<figcaption>{alt}</figcaption></figure>')
    html = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)" ?/?></p>', repl, html)
    return html


CSS = """
@page { size: A4 portrait; margin: 20mm 18mm 20mm 18mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font: 10.4pt/1.5 "Times New Roman", Georgia, serif; color: #000; background: #fff; margin: 0; }
.head { text-align: center; margin-bottom: 6mm; }
.head h1 { font-size: 15.5pt; line-height: 1.3; margin: 0 0 4mm; text-wrap: balance; }
.head .au { font-size: 11pt; font-weight: bold; margin: 0 0 1.5mm; }
.head .af { font-size: 9.6pt; font-style: italic; margin: 0; }
.abstract { border-top: .6pt solid #000; border-bottom: .6pt solid #000; padding: 3mm 0 2mm;
            margin: 0 0 5mm; font-size: 9.6pt; }
.abstract h2 { text-align: center; font-size: 11pt; margin: 0 0 2mm; }
h2 { font-size: 12pt; margin: 7mm 0 2.5mm; page-break-after: avoid; }
h3 { font-size: 10.8pt; margin: 5mm 0 2mm; page-break-after: avoid; }
p { margin: 0 0 2.6mm; text-align: justify; hyphens: auto; }
ul, ol { margin: 0 0 3mm; padding-left: 6mm; }
li { margin-bottom: 1.2mm; text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 5mm; font-size: 8.1pt;
        line-height: 1.3; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th, td { border: .5pt solid #000; padding: 1.1mm 1.4mm; vertical-align: top; text-align: left; }
th { background: #ececec; }
figure { margin: 3mm 0 5mm; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 8.8pt; text-align: justify; margin-top: 1.5mm; }
.tcap { break-after: avoid; page-break-after: avoid; margin-top: 3mm; }
.refs { font-size: 8.8pt; }
.refs li { margin-bottom: 1.4mm; text-align: left; }
.refs ol { padding-left: 7mm; }
"""


def build_paper():
    src = (ROOT / "docs" / "REVIEW_PAPER.md").read_text(encoding="utf-8")
    fm, body = re.match(r"^---\n(.*?)\n---\n(.*)$", src, re.S).groups()
    title = re.search(r'title:\s*"(.*)"', fm).group(1)
    authors = re.findall(r"^\s*-\s*(.+)$", fm, re.M)
    affil = re.search(r'affiliation:\s*"(.*)"', fm).group(1)

    body = body.split("## References")[0]
    body, order = number_citations(body)
    abstract_md, rest_md = body.split("## 1. Introduction", 1)
    abstract = markdown.markdown(abstract_md.replace("## Abstract", ""),
                                 extensions=["tables", "smarty"])
    rest = markdown.markdown("## 1. Introduction" + rest_md,
                             extensions=["tables", "sane_lists", "smarty"])
    rest = embed_figures(rest)
    # Keep each "Table N." caption on the same page as its table.
    rest = rest.replace("<p><strong>Table", '<p class="tcap"><strong>Table')
    refs = "".join(f"<li>{REFS[k]}</li>" for k in order)

    head = (f'<div class="head"><h1>{title}</h1>'
            f'<p class="au">{", ".join(authors)}</p><p class="af">{affil}</p></div>')
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
            f"<style>{CSS}</style></head><body>{head}"
            f'<div class="abstract"><h2>Abstract</h2>{abstract}</div>{rest}'
            f'<h2>References</h2><div class="refs"><ol>{refs}</ol></div></body></html>')
    out = BUILD / "review_print.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(order)} references, all cited, all defined)")


def build_analysis():
    src = (ROOT / "docs" / "REVIEW_ANALYSIS.md").read_text(encoding="utf-8")
    body = markdown.markdown(src, extensions=["tables", "smarty"])
    css = CSS + """
/* The guidelines ask for a one-page analysis, so this sheet is tuned to fill exactly
   one A4 page: narrower margins and a smaller table face than the review paper uses. */
@page { size: A4 portrait; margin: 11mm 12mm; }
body { font-size: 8.6pt; line-height: 1.3; }
h1 { font-size: 12.5pt; text-align: center; margin: 0 0 2mm; }
p { margin: 0 0 2mm; }
table { font-size: 7.5pt; line-height: 1.26; margin: 2mm 0 0; }
th, td { padding: .8mm 1.2mm; }
td:first-child, th:first-child { width: 21%; font-weight: bold; }
"""
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>Review-Paper Analysis</title><style>{css}</style></head>"
            f"<body>{body}</body></html>")
    out = BUILD / "review_analysis_print.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    build_paper()
    build_analysis()
