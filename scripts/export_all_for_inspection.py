"""
Export all pages of all 4 NIP deliverables to PNG images for thorough visual inspection.
"""

import os
import subprocess
from pathlib import Path
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "inspect_nip"
OUT.mkdir(parents=True, exist_ok=True)

def export_pdf_pages(pdf_name, prefix, scale=2.0):
    pdf_path = ROOT / pdf_name
    print(f"Exporting {pdf_path.name}...")
    doc = pdfium.PdfDocument(pdf_path)
    paths = []
    for i, page in enumerate(doc):
        img = page.render(scale=scale).to_pil()
        out_path = OUT / f"{prefix}_p{i+1}.png"
        img.save(out_path)
        paths.append(out_path)
    print(f"Exported {len(paths)} pages for {pdf_name}.")
    return paths

def export_pptx_slides():
    pptx_path = (ROOT / "docs" / "PW-1_Review_Presentation-NIP.pptx").resolve()
    slides_dir = OUT / "slides"
    slides_dir.mkdir(exist_ok=True)
    print(f"Exporting slides for {pptx_path.name} via PowerShell PowerPoint COM...")
    
    ps_script = f"""
    $ppt = New-Object -ComObject PowerPoint.Application
    $pres = $ppt.Presentations.Open('{pptx_path}', [Microsoft.Office.Core.MsoTriState]::msoTrue, [Microsoft.Office.Core.MsoTriState]::msoFalse, [Microsoft.Office.Core.MsoTriState]::msoFalse)
    $i = 1
    foreach ($slide in $pres.Slides) {{
        $outPath = Join-Path '{slides_dir}' ('slide_' + $i + '.png')
        $slide.Export($outPath, 'PNG', 1920, 1080)
        $i++
    }}
    $pres.Close()
    $ppt.Quit()
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
    """
    
    res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], capture_output=True, text=True)
    if res.returncode != 0:
        print("PowerShell Error:", res.stderr)
    else:
        exported = list(slides_dir.glob("slide_*.png"))
        print(f"Exported {len(exported)} slide images to {slides_dir}.")

def main():
    export_pdf_pages("docs/Review-Paper-NIP.pdf", "paper", scale=2.0)
    export_pdf_pages("docs/Review-Paper-Analysis-NIP.pdf", "analysis", scale=2.0)
    export_pdf_pages("docs/PW-1_Presentation_Speech-NIP.pdf", "speech", scale=1.5)
    export_pptx_slides()
    print("All deliverables exported to image successfully!")

if __name__ == "__main__":
    main()
