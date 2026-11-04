# -*- coding: utf-8 -*-
"""
Fix broken symbols and LaTeX notation in PRESENTATION_SPEECH.md and recompile PDF.
Replaces raw LaTeX and broken escape sequences with clean, publication-grade HTML/Unicode.
"""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent

def fix_presentation_speech():
    md_file = ROOT / "docs" / "PRESENTATION_SPEECH.md"
    content = md_file.read_text(encoding="utf-8")

    # 1. Clean control characters and literal tab/bell/formfeed artifacts
    content = content.replace("\x07pprox", "&approx;")
    content = content.replace("\x0crac", "")
    content = content.replace("\x07", "")
    content = content.replace("\x0c", "")

    # Direct literal string replacements
    exact_replacements = [
        # Slide presentations
        (r"$K=2$", "<em>K</em> = 2"),
        (r"$K = 2$", "<em>K</em> = 2"),
        (r"$A_{dyn}$", "<em>A</em><sub>dyn</sub>"),
        (r"$A_{learnable}$", "<em>A</em><sub>learnable</sub>"),
        (r"$\log(\frac{1}{T}\sum t^2)$", "log(&Sigma; <em>t</em><sup>2</sup> / <em>T</em>)"),
        (r"$\log(\frac{1}{T}\sum_{t=1}^T x_t^2)$", "log(&Sigma;<sub><em>t</em>=1</sub><sup><em>T</em></sup> <em>x</em><sub><em>t</em></sub><sup>2</sup> / <em>T</em>)"),
        (r"$\mu$", "&mu;"),
        (r"$\beta$", "&beta;"),
        (r"$\kappa$", "&kappa;"),
        (r"$p < 0.05$", "<em>p</em> &lt; 0.05"),

        # Q14
        (r"$G = (V, E, W)$", "<em>G</em> = (<em>V</em>, <em>E</em>, <em>W</em>)"),
        (r"$v_i \in V$", "<em>v</em><sub><em>i</em></sub> &isin; <em>V</em>"),
        (r"$V$", "<em>V</em>"),
        (r"$E$", "<em>E</em>"),
        (r"$W$", "<em>W</em>"),
        (r"$N = 64$", "<em>N</em> = 64"),
        (r"$A \in \mathbb{R}^{N \times N}$", "<em>A</em> &isin; <strong>R</strong><sup><em>N</em> &times; <em>N</em></sup>"),
        (r"$A \in \mathbb{R}^{N	imes N}$", "<em>A</em> &isin; <strong>R</strong><sup><em>N</em> &times; <em>N</em></sup>"),
        (r"$X$", "<em>X</em>"),
        (r"$x_i \in \mathbb{R}^T$", "<em>x</em><sub><em>i</em></sub> &isin; <strong>R</strong><sup><em>T</em></sup>"),
        (r"$v_i$", "<em>v</em><sub><em>i</em></sub>"),

        # Q15
        (r"$N$", "<em>N</em>"),
        (r"$A \in \mathbb{R}^{N \times N}$", "<em>A</em> &isin; <strong>R</strong><sup><em>N</em> &times; <em>N</em></sup>"),
        (r"$A_{ij}$", "<em>A</em><sub><em>ij</em></sub>"),
        (r"$i$", "<em>i</em>"),
        (r"$j$", "<em>j</em>"),
        (r"$D \in \mathbb{R}^{N \times N}$", "<em>D</em> &isin; <strong>R</strong><sup><em>N</em> &times; <em>N</em></sup>"),
        (r"$D \in \mathbb{R}^{N	imes N}$", "<em>D</em> &isin; <strong>R</strong><sup><em>N</em> &times; <em>N</em></sup>"),
        (r"$D_{ii} = \sum_{j} A_{ij}$", "<em>D</em><sub><em>ii</em></sub> = &Sigma;<sub><em>j</em></sub> <em>A</em><sub><em>ij</em></sub>"),
        (r"$L \in \mathbb{R}^{N \times N}$", "<em>L</em> &isin; <strong>R</strong><sup><em>N</em> &times; <em>N</em></sup>"),
        (r"$L \in \mathbb{R}^{N	imes N}$", "<em>L</em> &isin; <strong>R</strong><sup><em>N</em> &times; <em>N</em></sup>"),
        (r"$L = D - A$", "<em>L</em> = <em>D</em> &minus; <em>A</em>"),

        # Q16
        ("$$L_{sym} = D^{-1/2} L D^{-1/2} = I_N - D^{-1/2} A D^{-1/2}$$",
         '<div class="math-block"><em>L</em><sub>sym</sub> = <em>D</em><sup>&minus;1/2</sup> <em>L</em> <em>D</em><sup>&minus;1/2</sup> = <em>I</em><sub><em>N</em></sub> &minus; <em>D</em><sup>&minus;1/2</sup> <em>A</em> <em>D</em><sup>&minus;1/2</sup></div>'),
        (r"$(d_{max})$", "(<em>d</em><sub>max</sub>)"),
        (r"$[0, 2]$", "[0, 2]"),

        # Q17
        (r"$A_{ij} \ge 0$", "<em>A</em><sub><em>ij</em></sub> &ge; 0"),
        (r"$A_{ij} < 0$", "<em>A</em><sub><em>ij</em></sub> &lt; 0"),

        # Q18
        ("$$g_\\theta \\star x = U g_\\theta(\\Lambda) U^T x$$",
         '<div class="math-block"><em>g</em><sub>&theta;</sub> &lowast; <em>x</em> = <em>U</em> <em>g</em><sub>&theta;</sub>(&Lambda;) <em>U</em><sup>T</sup> <em>x</em></div>'),
        (r"$x$", "<em>x</em>"),
        (r"$g_\theta$", "<em>g</em><sub>&theta;</sub>"),
        (r"$U$", "<em>U</em>"),
        (r"$L$", "<em>L</em>"),
        (r"$\Lambda$", "&Lambda;"),
        (r"$U^T x$", "<em>U</em><sup>T</sup> <em>x</em>"),
        (r"$g_\theta(\Lambda)$", "<em>g</em><sub>&theta;</sub>(&Lambda;)"),
        (r"$\mathcal{O}(N^3)$", "<strong>O</strong>(<em>N</em><sup>3</sup>)"),

        # Q19
        (r"$\mathcal{O}(K \cdot |\mathcal{E}|)$", "<strong>O</strong>(<em>K</em> &middot; |<em>E</em>|)"),
        (r"$K$", "<em>K</em>"),

        # Q20
        (r"$T_0(\tilde{L}) x = x$", "<em>T</em><sub>0</sub>(<em>L&#771;</em>) <em>x</em> = <em>x</em>"),
        (r"$\tilde{L} = \frac{2}{\lambda_{\max}} L - I_N$", "<em>L&#771;</em> = (2 / &lambda;<sub>max</sub>) <em>L</em> &minus; <em>I</em><sub><em>N</em></sub>"),
        (r"$\tilde{L}$", "<em>L&#771;</em>"),
        (r"$T_1(\tilde{L}) x = \tilde{L} x$", "<em>T</em><sub>1</sub>(<em>L&#771;</em>) <em>x</em> = <em>L&#771;</em> <em>x</em>"),
        (r"$T_k(\tilde{L}) x = 2 \tilde{L} T_{k-1}(\tilde{L}) x - T_{k-2}(\tilde{L}) x$",
         "<em>T</em><sub><em>k</em></sub>(<em>L&#771;</em>) <em>x</em> = 2 <em>L&#771;</em> <em>T</em><sub><em>k</em>&minus;1</sub>(<em>L&#771;</em>) <em>x</em> &minus; <em>T</em><sub><em>k</em>&minus;2</sub>(<em>L&#771;</em>) <em>x</em>"),
        (r"$k \ge 2$", "<em>k</em> &ge; 2"),
        (r"$T_{k-1}$", "<em>T</em><sub><em>k</em>&minus;1</sub>"),

        # Q21
        (r"$K=1$", "<em>K</em> = 1"),
        (r"$K \ge 4$", "<em>K</em> &ge; 4"),

        # Q26
        (r"$W_a \in \mathbb{R}^{N \times d}$", "<em>W</em><sub><em>a</em></sub> &isin; <strong>R</strong><sup><em>N</em> &times; <em>d</em></sup>"),
        (r"$W_a \in \mathbb{R}^{N	imes d}$", "<em>W</em><sub><em>a</em></sub> &isin; <strong>R</strong><sup><em>N</em> &times; <em>d</em></sup>"),
        (r"$d \ll N$", "<em>d</em> &ll; <em>N</em>"),
        (r"$A = A^T$", "<em>A</em> = <em>A</em><sup>T</sup>"),

        # Q27
        (r"$W_a$", "<em>W</em><sub><em>a</em></sub>"),
        (r"$\mathcal{L}$", "&Lscr;"),

        # Q29
        (r"$X \in \mathbb{R}^{C \times T}$", "<em>X</em> &isin; <strong>R</strong><sup><em>C</em> &times; <em>T</em></sup>"),
        (r"$X \in \mathbb{R}^{C	imes T}$", "<em>X</em> &isin; <strong>R</strong><sup><em>C</em> &times; <em>T</em></sup>"),
        (r"$C$", "<em>C</em>"),
        (r"$T$", "<em>T</em>"),

        # Q30
        (r"$\sum X \approx 0$", "&Sigma; <em>X</em> &approx; 0"),
        (r"$C \times T \times \text{classes}$", "<em>C</em> &times; <em>T</em> &times; classes"),
        (r"$C	imes T	imes \text{classes}$", "<em>C</em> &times; <em>T</em> &times; classes"),

        # Q32
        (r"$f_s = 160\text{ Hz}$", "<em>f</em><sub><em>s</em></sub> = 160 Hz"),

        # Q34
        (r"$X(t)$", "<em>X</em>(<em>t</em>)"),
        (r"$X(t + \Delta t)$", "<em>X</em>(<em>t</em> + &Delta;<em>t</em>)"),
        (r"$\Delta t = 6.25\text{ ms}$", "&Delta;<em>t</em> = 6.25 ms"),
        (r"$r \approx 0.98$", "<em>r</em> &approx; 0.98"),
        (r"$t$", "<em>t</em>"),
        (r"$t+1$", "<em>t</em> + 1"),
        (r"$X(t+1)$", "<em>X</em>(<em>t</em> + 1)"),

        # Q43
        (r"$(\gamma_s, \beta_s)$", "(&gamma;<sub><em>s</em></sub>, &beta;<sub><em>s</em></sub>)"),

        # Q50
        (r"$(\text{ChebConv}, K=2)$", "(ChebConv, <em>K</em> = 2)"),
        (r"$W_{spatial} \in \mathbb{R}^{C \times F}$", "<em>W</em><sub>spatial</sub> &isin; <strong>R</strong><sup><em>C</em> &times; <em>F</em></sup>"),
        (r"$W_{spatial} \in \mathbb{R}^{C	imes F}$", "<em>W</em><sub>spatial</sub> &isin; <strong>R</strong><sup><em>C</em> &times; <em>F</em></sup>"),

        # Q52
        (r"$\kappa = \frac{p_o - p_e}{1 - p_e}$", "&kappa; = (<em>p</em><sub><em>o</em></sub> &minus; <em>p</em><sub><em>e</em></sub>) / (1 &minus; <em>p</em><sub><em>e</em></sub>)"),
        (r"$p_e = 0.25$", "<em>p</em><sub><em>e</em></sub> = 0.25"),
        (r"$\kappa > 0.5$", "&kappa; &gt; 0.5"),

        # Q53
        (r"$p$", "<em>p</em>"),
        (r"$p < 0.05$", "<em>p</em> &lt; 0.05"),
        (r"$p$-value", "<em>p</em>-value"),
    ]

    for old_s, new_s in exact_replacements:
        content = content.replace(old_s, new_s)

    # Replace block equations with clean HTML div blocks
    # 1. ChebConv equation
    cheb_pattern = re.compile(r"\$\$g_\\theta.*?x\$\$", re.DOTALL)
    content = cheb_pattern.sub(
        '<div class="math-block"><em>g</em><sub>&theta;</sub> &lowast; <em>x</em> &approx; &Sigma;<sub><em>k</em>=0</sub><sup><em>K</em></sup> &theta;<sub><em>k</em></sub> <em>T</em><sub><em>k</em></sub>(<em>L&#771;</em>) <em>x</em></div>',
        content
    )

    # 2. Pearson equation
    pearson_pattern = re.compile(r"\$\$A_\{ij\} = .*?\$\$", re.DOTALL)
    content = pearson_pattern.sub(
        '<div class="math-block"><em>A</em><sub><em>ij</em></sub> = Cov(<em>x</em><sub><em>i</em></sub>, <em>x</em><sub><em>j</em></sub>) / (&sigma;<sub><em>i</em></sub> &sigma;<sub><em>j</em></sub>)</div>',
        content
    )

    # 3. Dynamic Adjacency equation
    dyn_pattern = re.compile(r"\$\$A_\{dyn\} = .*?\$\$", re.DOTALL)
    content = dyn_pattern.sub(
        '<div class="math-block"><em>A</em><sub>dyn</sub> = Softmax(ReLU(<em>W</em><sub><em>a</em></sub> <em>W</em><sub><em>a</em></sub><sup>T</sup>)) &nbsp; <em>or</em> &nbsp; <em>A</em><sub>dyn</sub> = tanh(<em>W</em><sub><em>a</em></sub> <em>W</em><sub><em>a</em></sub><sup>T</sup>)</div>',
        content
    )

    # 4. Gradient equation
    grad_pattern = re.compile(r"\$\$\s*\\frac\{\\partial \\mathcal\{L\}\}\{\\partial W_a\}.*?\$\$", re.DOTALL)
    content = grad_pattern.sub(
        '<div class="math-block">&part;&Lscr; / &part;<em>W</em><sub><em>a</em></sub> = (&part;&Lscr; / &part;<em>L&#771;</em>) &middot; (&part;<em>L&#771;</em> / &part;<em>A</em><sub>dyn</sub>) &middot; (&part;<em>A</em><sub>dyn</sub> / &part;<em>W</em><sub><em>a</em></sub>)</div>',
        content
    )

    # 5. Log-variance equation
    logvar_pattern = re.compile(r"\$\$P_c = .*?\$\$", re.DOTALL)
    content = logvar_pattern.sub(
        '<div class="math-block"><em>P</em><sub><em>c</em></sub> = log( (1 / <em>T</em>) &Sigma;<sub><em>t</em>=1</sub><sup><em>T</em></sup> <em>X</em><sub><em>c</em>,<em>t</em></sub><sup>2</sup> )</div>',
        content
    )

    # Clean any stray dollar signs surrounding numbers or single letters
    content = re.sub(r"\$([A-Za-z0-9_]+)\$", r"<em>\1</em>", content)

    md_file.write_text(content, encoding="utf-8")
    print("Fixed symbols and equations in PRESENTATION_SPEECH.md successfully!")

def update_css():
    """Ensure math-block is styled in build_speech_pdf.py."""
    builder_file = ROOT / "scripts" / "build_speech_pdf.py"
    code = builder_file.read_text(encoding="utf-8")

    math_css = """
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
"""
    if ".math-block" not in code:
        code = code.replace("/* Tables */", math_css + "\n/* Tables */")
        builder_file.write_text(code, encoding="utf-8")
        print("Added .math-block CSS to build_speech_pdf.py")

def main():
    fix_presentation_speech()
    update_css()

if __name__ == "__main__":
    main()
