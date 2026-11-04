# Project Lessons & Operational Rules

## 1. Environment & Python Execution
- **Issue:** On this system, `python` or `pytest` in PowerShell can resolve to Anaconda Python 3.13 (`C:\Users\ADMIN\anaconda3`), which does not have domain-specific EEG libraries like `mne` installed.
- **Rule:** Always execute scripts and test suites using `C:\Python314\python.exe` (e.g., `C:\Python314\python.exe -m pytest`). This is the verified environment with PyTorch 2.12.1+cu132, MNE, python-pptx, and pypdfium2.

## 2. Math & Symbol Formatting in PDF Generation
- **Issue:** Compiling raw LaTeX math (`$...$`, `$$...$$`) via Python-Markdown to HTML and headless Edge can result in unparsed raw LaTeX, backslash escape corruption (e.g., `\t` becoming tabs, `\a` becoming ASCII bell, `\b` becoming backspace), or broken symbol boxes in PDFs.
- **Rule:** Sanitize formulas, Greek symbols, and arithmetic notation to valid HTML/Unicode entities (`&times;`, `&minus;`, `&mu;`, `&sigma;`, `&beta;`, `&kappa;`, `&ge;`, `&le;`, `&rarr;`, `<div class="math-block">`, `<sub>`, `<sup>`). Always run a `pypdfium2` audit script over generated PDFs to verify zero unparsed backslashes or corrupted glyphs.

## 3. Page Budget Calibration & Reference Sections
- **Issue:** Adding new empirical sections or figures can cause reference lists or conclusion sections to spill over into half-empty or single-reference orphan pages.
- **Rule:** When modifying content, dynamically adjust `@page` margins and typography (`margin: 16.5mm 15mm; font-size: 9.9pt; line-height: 1.34;`) so that documents land on exact, dense, balanced final pages with zero dangling headers and zero orphan lines.

## 4. Headless Edge PDF Rendering on Windows
- **Issue:** When invoking Microsoft Edge (`--headless=new --print-to-pdf=...`), passing a raw Windows file path instead of a valid file URI (`file:///D:/...`) can cause Edge to silently ignore the input. Additionally, if the target PDF file is open in a viewer, Edge can return code 0 without overwriting the file, causing scripts to falsely claim the PDF was generated.
- **Rule:** Always format input paths as proper file URIs (`path.resolve().as_uri()`), render first to a staging/temporary file in `build/`, verify that the temp file was created with size > 0, and then copy to the target location with explicit `PermissionError` trapping.
