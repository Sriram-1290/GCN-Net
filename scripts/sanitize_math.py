# -*- coding: utf-8 -*-
"""Sanitize all math expressions and symbols in PRESENTATION_SPEECH.md."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent

def sanitize():
    md_file = ROOT / "docs" / "PRESENTATION_SPEECH.md"
    text = md_file.read_text(encoding="utf-8")

    # Fix corrupted theta and tilde
    text = text.replace("g_&times;heta", "<em>g</em><sub>&theta;</sub>")
    text = text.replace("g_\\theta", "<em>g</em><sub>&theta;</sub>")
    text = text.replace("&times;heta", "&theta;")
    text = text.replace("&times;ilde{L}", "<em>L&#771;</em>")
    text = text.replace("\\tilde{L}", "<em>L&#771;</em>")
    text = text.replace("g_	heta", "<em>g</em><sub>&theta;</sub>")
    text = text.replace("(\\Lambda)", "(&Lambda;)")
    text = text.replace("(\\Lambda)", "(&Lambda;)")

    # Formula 1: Spectral graph convolution
    text = text.replace(
        "$g_&times;heta &lowast; x = U g_&times;heta(\\Lambda) U<sup>T</sup> x",
        '<div class="math-block"><em>g</em><sub>&theta;</sub> &lowast; <em>x</em> = <em>U</em> <em>g</em><sub>&theta;</sub>(&Lambda;) <em>U</em><sup>T</sup> <em>x</em></div>'
    )
    text = text.replace(
        "<em>g</em><sub>&theta;</sub> &lowast; x = U <em>g</em><sub>&theta;</sub>(&Lambda;) U<sup>T</sup> x",
        '<div class="math-block"><em>g</em><sub>&theta;</sub> &lowast; <em>x</em> = <em>U</em> <em>g</em><sub>&theta;</sub>(&Lambda;) <em>U</em><sup>T</sup> <em>x</em></div>'
    )

    # Formula 2: ChebConv approximation
    text = text.replace(
        "<em>g</em><sub>&theta;</sub> &lowast; x &approx; &Sigma;_{k=0}^{K} &theta;<sub><em>k</em></sub> T<sub><em>k</em></sub>(<em>L&#771;</em>) x",
        '<div class="math-block"><em>g</em><sub>&theta;</sub> &lowast; <em>x</em> &approx; &Sigma;<sub><em>k</em>=0</sub><sup><em>K</em></sup> &theta;<sub><em>k</em></sub> <em>T</em><sub><em>k</em></sub>(<em>L&#771;</em>) <em>x</em></div>'
    )

    # Formula 3: Normalized Laplacian
    text = text.replace(
        "$$L_{sym} = D^{-1/2} L D^{-1/2} = I_N - D^{-1/2} A D^{-1/2}$$",
        '<div class="math-block"><em>L</em><sub>sym</sub> = <em>D</em><sup>&minus;1/2</sup> <em>L</em> <em>D</em><sup>&minus;1/2</sup> = <em>I</em><sub><em>N</em></sub> &minus; <em>D</em><sup>&minus;1/2</sup> <em>A</em> <em>D</em><sup>&minus;1/2</sup></div>'
    )

    # Formula 4: Scaled Laplacian
    text = text.replace(
        "\\tilde{L} = \\frac{2}{\\lambda_{\\max}} L - I_N",
        "<em>L&#771;</em> = (2 / &lambda;<sub>max</sub>) <em>L</em> &minus; <em>I</em><sub><em>N</em></sub>"
    )

    # Formula 5: Pearson
    text = text.replace(
        "$$A_{ij} = \\frac{\\text{Cov}(x_i, x_j)}{\\sigma_i \\sigma_j}$$",
        '<div class="math-block"><em>A</em><sub><em>ij</em></sub> = Cov(<em>x</em><sub><em>i</em></sub>, <em>x</em><sub><em>j</em></sub>) / (&sigma;<sub><em>i</em></sub> &sigma;<sub><em>j</em></sub>)</div>'
    )

    # Formula 6: Dynamic adjacency
    text = text.replace(
        "$$A_{dyn} = \\text{Softmax}(\\text{ReLU}(W_a W_a^T)) \\quad \\text{or} \\quad A_{dyn} = \\tanh(W_a W_a^T)$$",
        '<div class="math-block"><em>A</em><sub>dyn</sub> = Softmax(ReLU(<em>W</em><sub><em>a</em></sub> <em>W</em><sub><em>a</em></sub><sup>T</sup>)) &nbsp; <em>or</em> &nbsp; <em>A</em><sub>dyn</sub> = tanh(<em>W</em><sub><em>a</em></sub> <em>W</em><sub><em>a</em></sub><sup>T</sup>)</div>'
    )

    # Formula 7: Gradient
    text = text.replace(
        "$$\\frac{\\partial \\mathcal{L}}{\\partial W_a} = \\frac{\\partial \\mathcal{L}}{\\partial \\tilde{L}} \\cdot \\frac{\\partial \\tilde{L}}{\\partial A_{dyn}} \\cdot \\frac{\\partial A_{dyn}}{\\partial W_a}$$",
        '<div class="math-block">&part;&Lscr; / &part;<em>W</em><sub><em>a</em></sub> = (&part;&Lscr; / &part;<em>L&#771;</em>) &middot; (&part;<em>L&#771;</em> / &part;<em>A</em><sub>dyn</sub>) &middot; (&part;<em>A</em><sub>dyn</sub> / &part;<em>W</em><sub><em>a</em></sub>)</div>'
    )

    # Formula 8: Log-variance
    text = text.replace(
        "$$P_c = \\log\\left(\\frac{1}{T} \\sum_{t=1}^T X_{c,t}^2\\right)$$",
        '<div class="math-block"><em>P</em><sub><em>c</em></sub> = log( (1 / <em>T</em>) &Sigma;<sub><em>t</em>=1</sub><sup><em>T</em></sup> <em>X</em><sub><em>c</em>,<em>t</em></sub><sup>2</sup> )</div>'
    )

    # Eliminate any stray single dollar signs
    text = text.replace("$", "")

    md_file.write_text(text, encoding="utf-8")
    print(f"Sanitization complete! File size: {len(text):,} chars. Stray dollar count: {text.count('$')}")

if __name__ == "__main__":
    sanitize()
