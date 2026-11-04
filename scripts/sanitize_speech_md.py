# -*- coding: utf-8 -*-
"""
Sanitizer script for PRESENTATION_SPEECH.md.
Ensures 100% of mathematical symbols, formulas, and Greek characters are clean HTML/Unicode entities.
"""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent

def sanitize():
    md_file = ROOT / "docs" / "PRESENTATION_SPEECH.md"
    text = md_file.read_text(encoding="utf-8")

    # 1. Clean control characters and escape artifacts
    text = text.replace("\x08", "")
    text = text.replace("\x07", "")
    text = text.replace("\x0c", "")

    # 2. Fix specific known broken lines
    text = text.replace("(&mu; and eta)", "(&mu; and &beta;)")
    text = text.replace("(&mu; and  eta)", "(&mu; and &beta;)")
    text = text.replace("(&mu; and \x08eta)", "(&mu; and &beta;)")

    text = text.replace("f_s = 160 ext{ Hz}", "<em>f</em><sub><em>s</em></sub> = 160 Hz")
    text = text.replace(r"\Delta t = 6.25 ext{ ms}", "&Delta;<em>t</em> = 6.25 ms")
    text = text.replace("have an autocorrelation r &approx; 0.98", "have an autocorrelation <em>r</em> &approx; 0.98")

    text = text.replace(r"(\gamma_s, \x08eta_s)", "(&gamma;<sub><em>s</em></sub>, &beta;<sub><em>s</em></sub>)")
    text = text.replace(r"(\gamma_s,  eta_s)", "(&gamma;<sub><em>s</em></sub>, &beta;<sub><em>s</em></sub>)")
    text = text.replace(r"(\gamma_s, \beta_s)", "(&gamma;<sub><em>s</em></sub>, &beta;<sub><em>s</em></sub>)")

    text = text.replace(r"\kappa = {p_o - p_e}{1 - p_e}", "&kappa; = (<em>p</em><sub><em>o</em></sub> &minus; <em>p</em><sub><em>e</em></sub>) / (1 &minus; <em>p</em><sub><em>e</em></sub>)")

    # Fix Q21 arrows
    text = re.sub(r"C3\s*\\?n?ightarrow\s*Cz\s*\\?n?ightarrow\s*C4", "C3 &rarr; Cz &rarr; C4", text)
    text = text.replace("C3 \nightarrow Cz \nightarrow C4", "C3 &rarr; Cz &rarr; C4")

    # Fix train_test_split
    text = text.replace("train_test<sub><em>s</em></sub>plit", "train_test_split")

    # Fix p-value in Q53
    text = text.replace("A <em>p</em>-value<em>p</em> &lt; 0.05", "A <em>p</em>-value &lt; 0.05 (<em>p</em> &lt; 0.05)")

    # Fix Q27 answer start
    text = text.replace(
        '#### Q27: "How is Adyn updated via gradient descent backpropagation end-to-end?"\n'
        '- **The Underlying Trap:** Verifying backpropagation mechanics through graph Laplacians.\n'
        '- **Master Defense Answer:**\n'
        '  > <div class="math-block">(&part;&Lscr; / &part;<em>W</em><sub><em>a</em></sub>) = (&part;&Lscr; / &part;<em>L&#771;</em>) &middot; (&part;<em>L&#771;</em> / &part;<em>A</em><sub>dyn</sub>) &middot; (&part;<em>A</em><sub>dyn</sub> / &part;<em>W</em><sub><em>a</em></sub>)</div>',
        '#### Q27: "How is <em>A</em><sub>dyn</sub> updated via gradient descent backpropagation end-to-end?"\n'
        '- **The Underlying Trap:** Verifying backpropagation mechanics through graph Laplacians.\n'
        '- **Master Defense Answer:**\n'
        '  > *"The loss gradient with respect to the learnable parameter matrix <em>W</em><sub><em>a</em></sub> is derived via the chain rule through the scaled Laplacian and dynamic adjacency matrix:  \n'
        '  > <div class="math-block">(&part;&Lscr; / &part;<em>W</em><sub><em>a</em></sub>) = (&part;&Lscr; / &part;<em>L&#771;</em>) &middot; (&part;<em>L&#771;</em> / &part;<em>A</em><sub>dyn</sub>) &middot; (&part;<em>A</em><sub>dyn</sub> / &part;<em>W</em><sub><em>a</em></sub>)</div>'
    )

    # Slide 9 formula
    text = text.replace("P = log((1 / <em>T</em>)&Sigma; x_t<sup>2</sup>)", "<em>P</em> = log( (1 / <em>T</em>) &Sigma; <em>x</em><sub><em>t</em></sub><sup>2</sup> )")

    # Q15
    text = text.replace("Adjacency Matrix (A &isin; <strong>R</strong>", "Adjacency Matrix (<em>A</em> &isin; <strong>R</strong>")
    text = text.replace("Degree Matrix (D &isin; <strong>R</strong>", "Degree Matrix (<em>D</em> &isin; <strong>R</strong>")
    text = text.replace("Combinatorial Graph Laplacian (L &isin; <strong>R</strong>", "Combinatorial Graph Laplacian (<em>L</em> &isin; <strong>R</strong>")

    # Titles
    text = text.replace('#### Q1: "Why do you have two titles\u2014\'Dynamic GCN', '#### Q1: "Why do you have two titles &mdash; \'Dynamic GCN')
    text = text.replace('#### Q1: "Why do you have two titles\u2013\'Dynamic GCN', '#### Q1: "Why do you have two titles &mdash; \'Dynamic GCN')
    text = text.replace('#### Q1: "Why do you have two titles-\'Dynamic GCN', '#### Q1: "Why do you have two titles &mdash; \'Dynamic GCN')
    text = text.replace('#### Q8: "What are the Mu (8\u201313 Hz) and Beta (14\u201330 Hz)', '#### Q8: "What are the Mu (8&ndash;13 Hz) and Beta (14&ndash;30 Hz)')
    text = text.replace('#### Q15: "What is an Adjacency Matrix (A), a Degree Matrix (D), and a Graph Laplacian (L)?"', '#### Q15: "What is an Adjacency Matrix (<em>A</em>), a Degree Matrix (<em>D</em>), and a Graph Laplacian (<em>L</em>)?\"')
    text = text.replace('#### Q16: "What is the Normalized Graph Laplacian (L~), and why is normalization necessary?"', '#### Q16: "What is the Normalized Graph Laplacian (<em>L&#771;</em>), and why is normalization necessary?\"')
    text = text.replace('#### Q20: "Walk through the Chebyshev polynomial recursive formula (T0, T1, Tk) step-by-step."', '#### Q20: "Walk through the Chebyshev polynomial recursive formula (<em>T</em><sub>0</sub>, <em>T</em><sub>1</sub>, <em>T</em><sub><em>k</em></sub>) step-by-step.\"')
    text = text.replace('#### Q21: "Why choose filter order K=2? Why not K=1 or K=5?"', '#### Q21: "Why choose filter order <em>K</em> = 2? Why not <em>K</em> = 1 or <em>K</em> = 5?\"')
    text = text.replace('#### Q26: "How is the dynamic adjacency matrix Adyn parameterized mathematically in GBP-Net?"', '#### Q26: "How is the dynamic adjacency matrix <em>A</em><sub>dyn</sub> parameterized mathematically in GBP-Net?\"')
    text = text.replace('#### Q52: "What evaluation metrics are used: Accuracy vs. Cohen\'s Kappa (kappa) vs. Macro F1?"', '#### Q52: "What evaluation metrics are used: Accuracy vs. Cohen\'s Kappa (&kappa;) vs. Macro F1?\"')

    # Domain headers
    domain_ranges = [(1,5),(6,11),(12,17),(18,22),(23,27),(28,32),(33,38),(39,43),(44,48),(49,54)]
    for i, (q_a, q_b) in enumerate(domain_ranges, 1):
        text = re.sub(rf"### Domain {i}: (.*?) \(Q{q_a}\s*[\u2013\u2014\-]\s*Q{q_b}\)", rf"### Domain {i}: \1 (Q{q_a} &ndash; Q{q_b})", text)

    # Convert typographic characters to HTML entities
    text = text.replace("\u2013", "&ndash;")
    text = text.replace("\u2014", "&mdash;")
    text = text.replace("\u2022", "&bull;")
    text = text.replace("\u00b7", "&middot;")

    md_file.write_text(text, encoding="utf-8")
    print(f"Sanitization completed successfully! New length: {len(text):,} chars.")

if __name__ == "__main__":
    sanitize()
