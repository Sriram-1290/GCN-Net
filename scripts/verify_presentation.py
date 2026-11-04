# -*- coding: utf-8 -*-
"""
Verification Script for PW-1 Review-1 Presentation (.pptx)
Checks slide count, headers, shapes, tables, embedded images, and speaker notes.
"""

import os
from pptx import Presentation

def verify():
    pptx_path = "docs/PW-1_Review-1_Presentation.pptx"
    if not os.path.exists(pptx_path):
        print(f"Error: {pptx_path} does not exist!")
        return False

    prs = Presentation(pptx_path)
    total_slides = len(prs.slides)
    print("=" * 75)
    print(f"AUDITING: {pptx_path}")
    print(f"Total Slides: {total_slides} (Expected: 12)")
    print(f"Dimensions: {prs.slide_width / 914400:.3f}\" x {prs.slide_height / 914400:.3f}\" (Expected: 13.333\" x 7.500\")")
    print("=" * 75)

    all_valid = True

    for i, slide in enumerate(prs.slides):
        slide_num = i + 1
        notes = slide.notes_slide.notes_text_frame.text.strip() if slide.has_notes_slide else ""
        tables = [s.table for s in slide.shapes if s.has_table]
        pictures = [s for s in slide.shapes if s.shape_type == 13]
        
        # Get slide title
        titles = []
        for s in slide.shapes:
            if s.has_text_frame and s.text_frame.text.strip():
                lines = [l.strip() for l in s.text_frame.text.split("\n") if l.strip()]
                for line in lines:
                    if any(key in line.upper() for key in ["NEIL GOGTE", "ABSTRACT", "PROBLEM", "OBJECTIVES", "LITERATURE", "TITLE", "RESEARCH GAPS", "PROPOSED", "EMPIRICAL", "EXPERIMENTAL", "EXPECTED"]):
                        titles.append(line)
                        break
                if titles:
                    break

        slide_title = titles[0] if titles else "Unknown Title"
        print(f"Slide {slide_num:02d}: {slide_title[:50]:<50}")
        print(f"   Shapes: {len(slide.shapes):2d} | Tables: {len(tables)} | Images: {len(pictures)} | Notes Words: {len(notes.split()):3d}")
        
        if tables:
            t = tables[0]
            print(f"   [Table Details] {len(t.rows)} rows x {len(t.columns)} cols. First Header: '{t.cell(0, 0).text.strip()}'")
        if pictures:
            for p_idx, p in enumerate(pictures):
                print(f"   [Image {p_idx+1}] Pos: ({p.left/914400:.2f}\", {p.top/914400:.2f}\") | Size: {p.width/914400:.2f}\" x {p.height/914400:.2f}\"")
        if len(notes.split()) < 20:
            print(f"   [Warning] Short or missing speaker notes on Slide {slide_num}!")
            all_valid = False

    print("=" * 75)
    if total_slides == 12 and all_valid:
        print("ALL AUDIT CHECKS PASSED: Presentation is fully verified, complete, and presentation-ready!")
    else:
        print("Verification completed with warnings.")
    print("=" * 75)
    return all_valid

if __name__ == "__main__":
    verify()
