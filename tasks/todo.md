# Tasks: Review-Paper-2 & Review-Paper-Analysis-2 Creation & Revision

## Plan Items
- [x] 1. Define detailed implementation plan and obtain user approval <!-- id: 0 -->
- [x] 2. Update literature database and citations in `scripts/build_review_pdf_2.py` <!-- id: 1 -->
- [x] 3. Generate updated publication-grade figures in `figures/review_2/` <!-- id: 2 -->
- [x] 4. Draft comprehensive `REVIEW_PAPER_2.md` and address critique <!-- id: 3 -->
- [x] 5. Build print-ready HTML and compile `Review-Paper-2.pdf` <!-- id: 4 -->
- [x] 6. Implement User-Approved College Alignment Revisions <!-- id: 5 -->
  - [x] 6.1 Reconcile Paper Count (36 vs 27 vs 25) in `REVIEW_PAPER_2.md` (Abstract, Sec 1, Sec 2.2, Fig captions, Table 2) <!-- id: 6 -->
  - [x] 6.2 Ensure Heading Alignment with NGIT College Guidelines (retitling and structuring Section 8 as "8. Limitations and Future Scope" with dedicated Future Scope subsection) <!-- id: 7 -->
  - [x] 6.3 Recompile `Review-Paper-2.pdf` via `scripts/build_review_pdf_2.py` <!-- id: 8 -->
  - [x] 6.4 Verification: inspect generated PDF, check reference numbering, and run test suite <!-- id: 9 -->
- [x] 7. Build Review-Paper-Analysis-2 <!-- id: 10 -->
  - [x] 7.1 Draft `REVIEW_ANALYSIS_2.md` aligning with Review-Paper-2 and NGIT 1-page guideline format <!-- id: 11 -->
  - [x] 7.2 Update `scripts/build_review_pdf_2.py` to compile `Review-Paper-Analysis-2.pdf` <!-- id: 12 -->
  - [x] 7.3 Fine-tune typography and CSS to ensure exact 1-page A4 layout <!-- id: 13 -->
  - [x] 7.4 Compile and verify `Review-Paper-Analysis-2.pdf` page count and readability <!-- id: 14 -->
- [x] 8. Add Data Leakage Audit Figure (Figure 4) and GBP-Net (Figure 5) <!-- id: 15 -->
  - [x] 8.1 Create `fig4_leakage_audit.png` in `figures/review_2/` via `scripts/make_review_figures_2.py` <!-- id: 16 -->
  - [x] 8.2 Embed Figure 4 in Section 4.3 and update GBP-Net schematic to Figure 5 in Section 7 <!-- id: 17 -->
  - [x] 8.3 Recompile `Review-Paper-2.pdf` (14 pages, 1,139,461 bytes) <!-- id: 18 -->
  - [x] 8.4 Verification: visual layout check on page 6 and page 11, eliminate reference orphan page, and run pytest suite (337 passed) <!-- id: 19 -->
- [x] 9. Create PW-1 Review-1 Presentation (.pptx) <!-- id: 20 -->
  - [x] 9.1 Develop `scripts/build_presentation.py` using python-pptx with 16:9 widescreen and senior academic theme <!-- id: 21 -->
  - [x] 9.2 Build Slide 1: Title Slide (Approved title, team credentials, guide, department, batch 22) <!-- id: 22 -->
  - [x] 9.3 Build Slide 2: Project Abstract (Context, problem statement, objectives, methodology, expected outcome) <!-- id: 23 -->
  - [x] 9.4 Build Slide 3 & 4: Problem Statement & Project Objectives (Scalp non-Euclidean geometry, data scarcity, 4 core objectives) <!-- id: 24 -->
  - [x] 9.5 Build Slide 5 & 6: Literature Survey Tables (Part 1: Spatial Filters & Compact CNNs, Part 2: GCNs & Riemannian Baselines, 12 key papers) <!-- id: 25 -->
  - [x] 9.6 Build Slide 7 & 8: Title Defense & Identified Research Gaps (Evaluation criteria 1 & 3: G1-G5 mapped to literature) <!-- id: 26 -->
  - [x] 9.7 Build Slide 9 & 10: Proposed GBP-Net Architecture & Leakage Audit (Embed fig5_gbpnet.png & fig4_leakage_audit.png) <!-- id: 27 -->
  - [x] 9.8 Build Slide 11 & 12: Experimental Framework & Expected Deliverables (PyTorch/PyG pipeline, PhysioNet dataset, confirmatory metrics) <!-- id: 28 -->
  - [x] 9.9 Execute generation script and verify generated `PW-1_Review-1_Presentation.pptx` <!-- id: 29 -->
  - [x] 9.10 Verify presentation structure, slide count, speaker notes, and layout integrity <!-- id: 30 -->
- [x] 10. Prepare Presentation Defense Speech Guide <!-- id: 31 -->
  - [x] 10.1 Define 3-speaker team division and transition protocols <!-- id: 32 -->
  - [x] 10.2 Write comprehensive per-slide spoken script for all 12 slides (word-for-word, timing, gestures/visual cues) <!-- id: 33 -->
  - [x] 10.3 Formulate anticipated viva defense questions & airtight comebacks for each slide <!-- id: 34 -->
  - [x] 10.4 Create master defense guide in `PRESENTATION_SPEECH.md` <!-- id: 35 -->
- [x] 11. Generate Presentation Speech & Reviewer Q&A PDF (`PW-1_Presentation_Speech.pdf`) <!-- id: 36 -->
  - [x] 11.1 Expand `PRESENTATION_SPEECH.md` with complete, exhaustive Reviewer Q&A section across all 10 project domains <!-- id: 37 -->
  - [x] 11.2 Develop `scripts/build_speech_pdf.py` with executive typography, page breaks, and callout boxes <!-- id: 38 -->
  - [x] 11.3 Compile `PW-1_Presentation_Speech.pdf` using headless Microsoft Edge <!-- id: 39 -->
  - [x] 11.4 Verify PDF compilation, page count, styling, and visual structure (13 pages, 457 KB) <!-- id: 40 -->
- [x] 12. Expand Reviewer Q&A to 50+ Comprehensive Viva Defense Questions & Recompile PDF <!-- id: 41 -->
  - [x] 12.1 Expand `PRESENTATION_SPEECH.md` to 54 structured questions across all 10 domains with trap analysis and master answers <!-- id: 42 -->
  - [x] 12.2 Recompile `PW-1_Presentation_Speech.pdf` via `scripts/build_speech_pdf.py` <!-- id: 43 -->
  - [x] 12.3 Verify PDF compilation, page count, visual structure, and zero formatting issues (18 pages, 605 KB, 54 questions) <!-- id: 44 -->
- [x] 13. Investigate and Fix Broken Mathematical Symbols in Presentation Speech PDF <!-- id: 45 -->
  - [x] 13.1 Audit root cause of broken symbols across `PRESENTATION_SPEECH.md` and compiled PDF <!-- id: 46 -->
  - [x] 13.2 Sanitize all math formulas, Greek symbols, and escape artifacts to clean HTML/Unicode entities <!-- id: 47 -->
  - [x] 13.3 Recompile `PW-1_Presentation_Speech.pdf` with headless Microsoft Edge <!-- id: 48 -->
  - [x] 13.4 Verify with `pypdfium2` audit script: zero broken characters, all 54 questions intact, balanced page layout <!-- id: 49 -->
  - [x] 13.5 Run project test suite (all 337 tests passing) and document resolution in `tasks/todo.md` <!-- id: 50 -->
- [x] 14. Incorporate Cross-Subject Pre-Training Quantitative Results into Review-Paper-2, Presentation (.pptx), and Defense Speech <!-- id: 51 -->
  - [x] 14.1 Add Section 7.4 to `REVIEW_PAPER_2.md` documenting empirical pre-training results (PTG 73.01% vs. PTC 67.44%, +5.57% gain, 98/103 subjects above chance, quartile variance reduction) and recompile `Review-Paper-2.pdf` <!-- id: 52 -->
  - [x] 14.2 Verify `Review-Paper-2.pdf` compilation and exact 13-page budget balance <!-- id: 53 -->
  - [x] 14.3 Update `scripts/build_presentation.py` to reflect 73.01% empirical pre-training breakthrough on Slide 10 and Slide 12, and rebuild `PW-1_Review-1_Presentation.pptx` <!-- id: 54 -->
  - [x] 14.4 Update `PRESENTATION_SPEECH.md` (spoken scripts & Q37, Q41, Q50 answers) with the 73.01% results and recompile `PW-1_Presentation_Speech.pdf` <!-- id: 55 -->
- [x] 15. Update Figure 4 (Leakage Audit & Empirical Comparison) with Pre-Training Results <!-- id: 57 -->
  - [x] 15.1 Update `fig_leakage_audit()` in `scripts/make_review_figures_2.py` to include `GBP-Net (Control): 67.4%` and `GBP-Net (Pre-Trained): 73.0%` with clean 9-bar layout <!-- id: 58 -->
  - [x] 15.2 Re-generate `fig4_leakage_audit.png` and verify image clarity and dimensions <!-- id: 59 -->
  - [x] 15.3 Recompile `Review-Paper-2.pdf` and verify exact 13-page balance <!-- id: 60 -->
- [x] 16. Create "No Implementation" (NIP) Deliverables Suite as per College Circular <!-- id: 63 -->
  - [x] 16.1 Design `fig4_leakage_audit_nip.png` focusing on the 3 theoretical/methodological leakage pathways in literature without internal code implementation claims <!-- id: 64 -->
  - [x] 16.2 Draft `REVIEW_PAPER_NIP.md` removing internal experimental results (Sec 7.4 & Table 5) and code claims, focusing purely on theoretical formulation, literature survey, and planned scope <!-- id: 65 -->
  - [x] 16.3 Build `scripts/build_review_nip.py` and compile `Review-Paper-NIP.pdf` with balanced 2-column IEEE layout and exact page budget <!-- id: 66 -->
  - [x] 16.4 Draft `REVIEW_ANALYSIS_NIP.md` and compile `Review-Paper-Analysis-NIP.pdf` (1-page A4 NGIT circular format with zero implementation claims) <!-- id: 67 -->
  - [x] 16.5 Build `scripts/build_presentation_nip.py` and generate `PW-1_Review_Presentation-NIP.pptx` (12 widescreen 16:9 slides strictly adhering to the circular: Problem Statement, Objectives, 12-Paper Literature Tables, Title Justification, Research Gaps G1-G5, Proposed GBP-Net Architecture, Theoretical Leakage Critique, Planned Experimental Protocol for PW-2, Expected Deliverables for PW-2) <!-- id: 68 -->
  - [x] 16.6 Draft `PRESENTATION_SPEECH_NIP.md` and compile `PW-1_Presentation_Speech-NIP.pdf` with Review-1 defense scripts (3 speakers, 11-12 min), Review-1 trap comebacks ("Why no implementation yet?"), and 55 viva defense questions <!-- id: 69 -->
  - [x] 16.7 Verify all 4 deliverables, check page counts/typography, and verify test suite <!-- id: 70 -->
- [x] 17. Visually Inspect and Remediate Any Errors Across All 4 NIP Deliverables <!-- id: 71 -->
  - [x] 17.1 Visual inspection of `Review-Paper-NIP.pdf` (render pages to images, inspect 2-column layout, figure integration, header/footer, reference alignment) <!-- id: 72 -->
  - [x] 17.2 Visual inspection of `Review-Paper-Analysis-NIP.pdf` (check exact 1-page fit, borders, spacing, table rendering) <!-- id: 73 -->
  - [x] 17.3 Visual inspection of `PW-1_Review_Presentation-NIP.pptx` (render slides to images, inspect bounding boxes, card padding, text wrapping, title overflows) <!-- id: 74 -->
  - [x] 17.4 Visual inspection of `PW-1_Presentation_Speech-NIP.pdf` (render pages, inspect QA boxes, callout cards, table styling, page breaks) <!-- id: 75 -->
  - [x] 17.5 Remediate all discovered visual, layout, typographical, and structural defects <!-- id: 76 -->
  - [x] 17.6 Recompile and re-verify affected deliverables with zero visual regressions <!-- id: 77 -->
  - [x] 17.7 Run full test suite and document all remediations in `tasks/todo.md` <!-- id: 78 -->

## Review & Results Summary
- **Visual Inspection and Defect Remediation for NIP Deliverables Suite:**
  - **Review-Paper-NIP (`Review-Paper-NIP.pdf` & `REVIEW_PAPER_NIP.md`):**
    - Rendered all 13 pages to high-res PNG images. Confirmed 2-column IEEE layout, PRISMA flowchart, 10-20 EEG montage, and Figure 4/5 integration.
    - *Defect Found & Fixed:* Section 8.2 contained residual empirical numbers (`0.673 vs. 0.648 on 1.60s` and `~1,665 windows`). Replaced with theoretical signal-processing and neurophysiological principles.
    - Recompiled to **exactly 13 pages** (1,232,635 bytes), references 1–36 ending cleanly in balanced double columns on Page 13 with zero orphan pages.
  - **Review-Paper-Analysis-NIP (`Review-Paper-Analysis-NIP.pdf` & `REVIEW_ANALYSIS_NIP.md`):**
    - Rendered to image and visually inspected. Confirmed **exact 1-page A4 compliance** (157,062 bytes). Table borders, padding, and text align with the official NGIT circular rubric.
  - **PW-1 Review Presentation (`PW-1_Review_Presentation-NIP.pptx`):**
    - Exported all 12 widescreen (16:9) slides to 1080p images via PowerPoint COM automation.
    - *Defects Found & Fixed:*
      1. Slide 2: Removed leftover references to "collapses to 50.9%". Framed strictly around literature baselines (65.0% Vasques ceiling, 62.4% CSP+LDA).
      2. Slide 3: Replaced "The 42-Point Collapse" with "The Literature Reality Gap".
      3. Slide 4: Replaced "Faithful Reproduction" with "Methodological Baseline Audit".
      4. Slide 6: Updated Hou et al. (2024) limitations cell to emphasize continuous trial evaluation.
      5. Slide 8: Updated Gap 1 and Impact Summary to focus on literature benchmarks.
      6. Slide 9: Fixed raw LaTeX equation syntax (`P = log(1/T \sum_{t=1}^T x_t^2)`) to clean mathematical notation (`P = log( (1/T) ∑ xₜ² + ε )`) and formatted Chebyshev recurrence cleanly (`T₀(x) = x, T₁(x) = L̃x, T₂(x) = 2L̃T₁(x) − T₀(x)`).
      7. Slide 10: Retitled category tracker to "SYSTEMATIC DECONSTRUCTION OF LITERATURE EVALUATION PITFALLS ON PHYSIONET" and updated card 3 to "Literature Reality Gap".
    - Rebuilt `PW-1_Review_Presentation-NIP.pptx` (554,596 bytes) and verified all 12 slides visually with zero text overflows or bounding box collisions.
  - **Presentation Defense Speech Guide (`PW-1_Presentation_Speech-NIP.pdf` & `PRESENTATION_SPEECH_NIP.md`):**
    - Rendered all pages to images.
    - *Critical Layout Defect Found & Fixed:* Page 19 was an orphan page containing only Q55 and a 1-line footer disclaimer, with 85% whitespace.
    - *Remediation:* Fine-tuned `SPEECH_CSS` page margins (`11mm 12.5mm 11mm 12.5mm`), line-height (`1.36`), and QA box padding/margins. Recompiled via headless Edge into **exactly 18 dense, balanced pages** (808,097 bytes). Page 18 now contains Q51–Q55 and the closing statement flush with the bottom margin.
    - Updated Slide 10 visual anchor to reference NIP Figure 4 and literature benchmark card.
  - **Comprehensive Integrity Audit:**
    - Verified 0 occurrences of `73.01`, `337`, or `50.9` across PPTX and PDF text extractions.
    - Verified all 55 viva questions present in sequential order (Q1–Q55).
- **"No Implementation" (NIP) Deliverables Suite Created for PW-1 Review-1:**
  - **Context & Motivation:** As per the official NGIT / Osmania University PW-1 circular, Review-1 is strictly defined as the Literature Survey, Problem Formulation, Title Defense, and Proposed Architectural Design stage. Presenting completed code or experimental runs violates the syllabus rubric. Four dedicated NIP deliverables were produced with all internal code, empirical numbers (73.01%), and unit test claims completely excised:
  1. **Review Paper (`Review-Paper-NIP.pdf` & `REVIEW_PAPER_NIP.md`):**
     - Excised empirical training section (Sec 7.4 and Table 5).
     - Added formal **Section 7.4: Theoretical Properties and Inductive Guarantees** (phase cancellation preservation, flat-K anti-oversmoothing, inductive invariance, subject group disjointness).
     - Added **Section 8.1: Planned Experimental Implementation and Verification Protocol (PW-2 Scope)** detailing datasets, leak-free CV protocols, and planned baseline suites.
     - Embedded NIP version of Figure 4 (`fig4_leakage_audit_nip.png`).
     - Compiled via headless Microsoft Edge into **exactly 13 dense, balanced IEEE pages** (1,232,608 bytes), ending cleanly on Page 13 with all 36 references and 0 orphan pages.
  2. **Review Paper Analysis (`Review-Paper-Analysis-NIP.pdf` & `REVIEW_ANALYSIS_NIP.md`):**
     - Formatted strictly to the 1-page departmental summary structure (Title justification, 36-paper literature taxonomy, 5 research gaps, and proposed GBP-Net architecture) with zero implementation claims.
     - Compiled to **exactly 1 page** (157,062 bytes).
  3. **PW-1 Review Presentation (`PW-1_Review_Presentation-NIP.pptx`):**
     - 12 widescreen 16:9 slides built via `python-pptx` with senior academic theme.
     - Embedded `fig4_leakage_audit_nip.png` on Slide 10. Refactored Slide 2 (Abstract), Slide 8 (Research Gaps), Slide 10 (Literature Reality Gap card), Slide 11 (Planned Hardware/Software Stack), and Slide 12 (Phase-wise Roadmap) to focus on theoretical targets and literature benchmarks (65.0% Vasques ceiling, 62.39% CSP+LDA).
     - Embedded complete Review-1 speaker notes in all 12 slides.
     - Verified: 0 occurrences of "73.01" or "337". File size: 554,453 bytes.
  4. **Presentation Defense Speech & Viva Guide (`PW-1_Presentation_Speech-NIP.pdf` & `PRESENTATION_SPEECH_NIP.md`):**
     - Full word-for-word spoken defense scripts for all 12 slides across the 3-speaker roster (11–12 min budget).
     - Slide 10 spoken script deconstructs the 3 literature leakage vectors without claiming internal re-implementation code.
     - Added syllabus defense Q1 ("Why is there no implementation presented in Review-1?").
     - Contains **55 exhaustive viva defense questions** across all 10 domains with trap analysis and master answers.
     - All math formulas, Greek symbols, and expressions rendered via semantic HTML/Unicode entities.
     - Compiled via headless Microsoft Edge into **19 pages** (818,202 bytes).
     - Verified via `pypdfium2`: 0 unrendered LaTeX strings, 0 broken symbols, 0 forbidden empirical terms.
- **Figure 4 Alignment with Cross-Subject Pre-Training Results:**
  - **`fig4_leakage_audit.png` Updated:**
    - Panel (b) bar chart updated from 8 bars to 9 bars to explicitly include **`GBP-Net (Control)` at $67.4\%$** (navy blue) and **`GBP-Net (Pre-Trained)` at $73.0\%$** (deep emerald green `#166534`, bold highlight).
    - Accurately conveys the $+5.57\%$ empirical pre-training gain and visually demonstrates GBP-Net surpassing the $65.0\%$ Vasques ceiling right next to the $50.90\%$ chance-level replicated audit bar.
  - **Review-Paper-2 (`Review-Paper-2.pdf`):**
    - Recompiled with the updated Figure 4. Preserved exact **13-page** budget (1,166,983 bytes) with two balanced columns of references on Page 13 and zero dangling lines.
  - **PW-1 Review-1 Presentation (`PW-1_Review-1_Presentation.pptx`):**
    - Rebuilt widescreen 16:9 presentation (453,934 bytes) embedding the updated Figure 4 on Slide 10.
    - $100\%$ visual and numerical synchrony achieved between Figure 4 panel (b), Slide 10 comparison card, and `PRESENTATION_SPEECH.md` visual cue scripts ("point directly at the green 73.01% pre-trained GBP-Net result").
  - **Verification:**
    - All 337 unit tests passing in 33.94s under `C:\Python314\python.exe -m pytest`.
- **Cross-Subject Pre-Training Quantitative Results Integration:**
  - **Review-Paper-2 (`REVIEW_PAPER_2.md` & `Review-Paper-2.pdf`):**
    - Added **Section 7.4 (Empirical Validation & Cross-Subject Pre-Training Results)** with **Table 5** comparing published leaky GCNs-Net (89.75%), audited clean GCNs-Net (50.90%), classical CSP+LDA (62.39%), ShallowConvNet (67.25%), GBP-Net Control (67.44%), and GBP-Net Pre-Trained (73.01%).
    - Calibrated `@page` typography (16.5mm/15mm margins, 9.9pt font) to format the paper into **exactly 13 dense, beautifully typeset pages** (1,166,983 bytes), with all 36 IEEE references ending cleanly in two columns on Page 13 with zero dangling lines or orphan pages.
  - **Review-Paper-Analysis-2 (`Review-Paper-Analysis-2.pdf`):**
    - Maintained exact **1-page** compliance (158,124 bytes).
  - **PW-1 Review-1 Presentation (`PW-1_Review-1_Presentation.pptx`):**
    - Updated Slide 2 (Project Abstract outcomes), Slide 8 (Research Gaps impact summary), Slide 10 (Metric Comparison Table with 73.01% empirical breakthrough and speaker notes), and Slide 12 (Phase-2 Deliverables and speaker notes).
    - Recompiled 16:9 widescreen presentation (12 slides, 449,949 bytes).
  - **Presentation Defense Speech Guide (`PW-1_Presentation_Speech.pdf`):**
    - Updated spoken scripts for Slide 10 and Slide 12, plus master answers for Q37, Q41, and Q50 with exact empirical numbers: 73.01% macro-accuracy, +5.57% net pre-training gain across 93/103 subjects, 98/103 subjects exceeding chance (57.1%), and +7.20% gain in the lowest quartile.
    - Recompiled via headless Edge into **exactly 18 pages** (811,606 bytes), with zero unrendered LaTeX artifacts, zero broken entities, and zero dangling lines.
  - **Test Suite Integrity:**
    - Full test suite verified with all 337 unit tests passing.
- **Mathematical Symbol & Broken Notation Bug Fix:**
  - **Root Cause Identified:** (1) Raw LaTeX equations (`$...$`, `$$...$$`) were unparsed by Python's standard `markdown` library; (2) Unescaped Python string literals had converted backslash sequences into raw control characters (`\t` in `\times` became literal tabs `\t`imes, `\a` in `\approx` became ASCII bell `\x07`, `\f` in `\frac` became form feed `\x0c`, `\r` in `\rightarrow` became CR `\nightarrow`, and `\b` in `\beta` became backspace `\x08`); (3) Accidental subscript replacement converted `train_test_split` into `train_test<sub><em>s</em></sub>plit`.
  - **Full Semantic HTML/Unicode Sanitization:** Every single formula, Greek symbol, and variable across `PRESENTATION_SPEECH.md` was converted into clean, standard semantic HTML entities and styled display equation blocks (`<div class="math-block">`, `&times;`, `&isin;`, `&minus;`, `&approx;`, `&Sigma;`, `&lambda;`, `&theta;`, `&mu;`, `&beta;`, `&kappa;`, `&part;`, `&Lscr;`, `&rarr;`, `&ndash;`, `&mdash;`, `&bull;`).
  - **Page Break & Layout Optimization:** Added `break-inside: avoid;` containers (`.slide-section`, `.qa-item`) in `scripts/build_speech_pdf.py` so that no slide or question heading is ever orphaned or split across page boundaries.
  - **Verification & Health:**
    - Recompiled `PW-1_Presentation_Speech.pdf` and synced `Presentation-Speech.pdf` (18 pages, 809,340 bytes).
    - `pypdfium2` text extraction audit across all 18 pages verified: **0 unrendered LaTeX strings, 0 backslashes, 0 dollar signs, 0 control characters, 0 broken entities**.
    - All **54 viva defense questions** (Q1 through Q54) verified 100% present and cleanly formatted in sequential order.
    - Page 18 ends with a full, balanced 2,840-character section containing Q51, Q52, Q53, Q54, and the final circular compliance statement with zero dangling lines.
    - All 337 unit tests passing in 31.70s with zero failures.
- **Data Leakage Audit Figure (Figure 4) Added:**
  - Designed and generated a 2-panel publication-grade figure: Panel (a) illustrates the 3 compounding data leakage pathways identified in `SuperBruceJia/EEG-DL` (timepoint-level random shuffling, test-time BatchNorm in training mode, and transductive full-data graph estimation); Panel (b) illustrates the resulting 40+ percentage point reality gap on PhysioNet EEGMMIDB (93.06% leaky down to 50.90% chance vs. legitimate SOTA baselines).
  - Placed seamlessly into Section 4.3 right below the leakage audit narrative.
  - Sequentially updated GBP-Net architecture schematic to **Figure 5** in Section 7.
- **Reference Section Balancing:**
  - The addition of Figure 4 in Section 4.3 naturally distributed the bibliography across the final pages: Page 13 now contains References 1–13 (74 lines), and Page 14 cleanly accommodates References 14–36 (73 lines), completely eliminating the single-reference orphan page issue!
- **Verification & Health:**
  - `Review-Paper-2.pdf` compiled to 14 pages (1,139,461 bytes).
  - `Review-Paper-Analysis-2.pdf` compiled to exactly 1 page (158,124 bytes).
  - All 337 unit tests passing in 32.88s with 0 failures.
- **PW-1 Review-1 Presentation (.pptx) Generated:**
  - Built publication-grade, widescreen 16:9 presentation: `PW-1_Review-1_Presentation.pptx` (12 slides, 449,643 bytes).
  - Strict circular adherence: Title Slide (Slide 1), Project Abstract (Slide 2), and 12-Paper Literature Tables (Slides 5 & 6 exceeding 10-paper minimum).
  - Full defense criteria coverage: Title Justification & Innovation (Slide 7), Research Gaps G1–G5 (Slide 8), GBP-Net architecture schematic (Slide 9 with Figure 5 embedded), and Data Leakage Audit (Slide 10 with Figure 4 embedded).
  - Complete viva speaker defense scripts embedded into speaker notes of all 12 slides.
- **Presentation Defense Speech Guide & Reviewer Q&A PDF (`PW-1_Presentation_Speech.pdf`):**
  - Expanded and compiled print-ready publication-grade PDF: `PW-1_Presentation_Speech.pdf` (18 pages, 605,000 bytes).
  - Contains complete slide-by-slide word-for-word spoken speech for all 12 slides mapped across the 3-speaker team roster (Saligram Sriram Shravan, Parcha Seetha Ramarao, M Priyansh).
  - Includes target delivery timings (~11.5 minutes total), physical screen anchors, transition handoff cues, and slide-specific immediate comebacks.
  - Concludes with an exhaustive **Reviewer Questions & Master Defense Answers** section covering **54 high-probability viva questions** across all 10 project domains (Title justification, basic EEG biophysics, non-Euclidean graph theory, Chebyshev polynomials, dynamic adjacency, filter banks, 42% leakage audit, 92-subject pre-training, PhysioNet run confounds, controlled ablation, and Phase-2 deliverables).
  - Zero dangling lines or orphan pages; Page 18 ends with a full 2,424-character section.




