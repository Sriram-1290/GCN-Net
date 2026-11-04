"""Renumber questions in PRESENTATION_SPEECH_NIP.md from Q1 to Q55 and update domain range headers."""

import re

def main():
    with open('docs/PRESENTATION_SPEECH_NIP.md', 'r', encoding='utf-8') as f:
        text = f.read()

    # Update top heading
    text = text.replace('(54 Questions)', '(55 Questions)')

    # Renumber questions sequentially
    q_counter = [0]
    def repl_q(match):
        q_counter[0] += 1
        return f'#### Q{q_counter[0]}:'

    text = re.sub(r'#### Q\d+:', repl_q, text)
    print(f'Total questions renumbered: {q_counter[0]}')

    # Update domain range headers
    domain_ranges = {
        r'### Domain 1: Project Scope, Approved Title & Architectural Grounding \(Q1 &ndash; Q5\)':
            '### Domain 1: Project Scope, Approved Title & Architectural Grounding (Q1 &ndash; Q6)',
        r'### Domain 2: Basic EEG Fundamentals & Neurophysiology \(Q6 &ndash; Q11\)':
            '### Domain 2: Basic EEG Fundamentals & Neurophysiology (Q7 &ndash; Q12)',
        r'### Domain 3: Graph Theory & Non-Euclidean Scalp Representation \(Q12 &ndash; Q17\)':
            '### Domain 3: Graph Theory & Non-Euclidean Scalp Representation (Q13 &ndash; Q18)',
        r'### Domain 4: Chebyshev Spectral Graph Convolution Mechanics \(Q18 &ndash; Q22\)':
            '### Domain 4: Chebyshev Spectral Graph Convolution Mechanics (Q19 &ndash; Q23)',
        r'### Domain 5: Dynamic Learnable Adjacency Matrix Mechanics \(Q23 &ndash; Q27\)':
            '### Domain 5: Dynamic Learnable Adjacency Matrix Mechanics (Q24 &ndash; Q28)',
        r'### Domain 6: Temporal Filter Banks & Band-Power Physics \(Q28 &ndash; Q32\)':
            '### Domain 6: Temporal Filter Banks & Band-Power Physics (Q29 &ndash; Q33)',
        r'### Domain 7: The Data Leakage Crisis & Empirical Audit Findings \(Q33 &ndash; Q38\)':
            '### Domain 7: The Data Leakage Crisis & Literature Reality Gap (Q34 &ndash; Q39)',
        r'### Domain 8: Cross-Subject Transfer & Population Pre-Training \(Q39 &ndash; Q43\)':
            '### Domain 8: Cross-Subject Transfer & Population Pre-Training (Q40 &ndash; Q44)',
        r'### Domain 9: Dataset Specifics, Confounds & Signal Preprocessing \(Q44 &ndash; Q48\)':
            '### Domain 9: Dataset Specifics, Confounds & Signal Preprocessing (Q45 &ndash; Q49)',
        r'### Domain 10: Implementation Stack, Controlled Ablation & Evaluation Metrics \(Q49 &ndash; Q54\)':
            '### Domain 10: Implementation Stack, Controlled Ablation & Evaluation Metrics (Q50 &ndash; Q55)',
    }

    for pattern, repl in domain_ranges.items():
        text, n = re.subn(pattern, repl, text)
        if n == 0:
            print(f'Warning: Pattern not matched: {pattern}')

    with open('docs/PRESENTATION_SPEECH_NIP.md', 'w', encoding='utf-8') as f:
        f.write(text)

    print('Renumbered PRESENTATION_SPEECH_NIP.md successfully!')

if __name__ == '__main__':
    main()
