"""
Build Script for PW-1 Review-1 Presentation (.pptx)
Neil Gogte Institute of Technology (NGIT) - Department of Computer Science & Engineering
Major Project Work-1 (PW-1) · Batch 22 · Review-1 Defense (October 2026)

Title: Dynamic Graph Convolutional Network with Temporal Attention for Subject-Independent EEG Motor Imagery Decoding
Architectural Realization: GBP-Net: A Graph Band-Power Network with Cross-Subject Pre-Training
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# -----------------------------------------------------------------------------
# Color Palette (Senior Academic Executive Theme)
# -----------------------------------------------------------------------------
NAVY = RGBColor(26, 54, 93)         # #1A365D - Institutional Deep Navy
SLATE = RGBColor(43, 108, 176)      # #2B6CB0 - Accent Blue
CHARCOAL = RGBColor(30, 41, 59)     # #1E293B - High-contrast readable body text
MUTED = RGBColor(100, 116, 139)     # #64748B - Secondary labels / trackers
CARD_BG = RGBColor(255, 255, 255)   # #FFFFFF - Clean card background
SLIDE_BG = RGBColor(248, 250, 252)  # #F8FAFC - Soft background
BORDER = RGBColor(226, 232, 240)    # #E2E8F0 - Subtle card border
CRIMSON = RGBColor(197, 48, 48)     # #C53030 - Leakage / Gap alerts
TEAL = RGBColor(15, 118, 110)       # #0F766E - Breakthrough / Solutions
GOLD = RGBColor(217, 119, 6)        # #D97706 - Title badges & highlights

# Card tinted backgrounds
CRIMSON_BG = RGBColor(254, 242, 242)
TEAL_BG = RGBColor(240, 253, 250)
BLUE_BG = RGBColor(239, 246, 255)
AMBER_BG = RGBColor(254, 243, 199)

FONT_HEADING = "Arial"
FONT_BODY = "Arial"


def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    return prs, blank_layout


def add_header(slide, title_text, category_text, slide_num, total_slides=12):
    """Adds standard executive header, divider, and footer to a content slide."""
    # Top Category Tracker
    cat_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12.133), Inches(0.3))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.name = FONT_HEADING
    p_cat.font.size = Pt(9.5)
    p_cat.font.bold = True
    p_cat.font.color.rgb = SLATE

    # Slide Title
    title_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.65), Inches(12.133), Inches(0.55))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.name = FONT_HEADING
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = NAVY

    # Thin Divider Line under header
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.22), Inches(12.133), Inches(0.02))
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER
    line.line.color.rgb = BORDER

    # Bottom Divider Line
    line_bot = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(7.0), Inches(12.133), Inches(0.015))
    line_bot.fill.solid()
    line_bot.fill.fore_color.rgb = BORDER
    line_bot.line.color.rgb = BORDER

    # Footer Left (Project & Batch)
    fl_box = slide.shapes.add_textbox(Inches(0.6), Inches(7.08), Inches(5.0), Inches(0.3))
    tf_fl = fl_box.text_frame
    tf_fl.margin_left = tf_fl.margin_top = tf_fl.margin_right = tf_fl.margin_bottom = 0
    p_fl = tf_fl.paragraphs[0]
    p_fl.text = "Dynamic GCN with Temporal Attention (GBP-Net) · PW-1 Review-1 · Batch 22"
    p_fl.font.name = FONT_BODY
    p_fl.font.size = Pt(8.5)
    p_fl.font.color.rgb = MUTED

    # Footer Center (Institution)
    fc_box = slide.shapes.add_textbox(Inches(5.0), Inches(7.08), Inches(4.5), Inches(0.3))
    tf_fc = fc_box.text_frame
    tf_fc.margin_left = tf_fc.margin_top = tf_fc.margin_right = tf_fc.margin_bottom = 0
    p_fc = tf_fc.paragraphs[0]
    p_fc.text = "Neil Gogte Institute of Technology (Affiliated to Osmania University)"
    p_fc.alignment = PP_ALIGN.CENTER
    p_fc.font.name = FONT_BODY
    p_fc.font.size = Pt(8.5)
    p_fc.font.color.rgb = MUTED

    # Footer Right (Slide Number)
    fr_box = slide.shapes.add_textbox(Inches(10.5), Inches(7.08), Inches(2.233), Inches(0.3))
    tf_fr = fr_box.text_frame
    tf_fr.margin_left = tf_fr.margin_top = tf_fr.margin_right = tf_fr.margin_bottom = 0
    p_fr = tf_fr.paragraphs[0]
    p_fr.text = f"Slide {slide_num} of {total_slides}"
    p_fr.alignment = PP_ALIGN.RIGHT
    p_fr.font.name = FONT_HEADING
    p_fr.font.size = Pt(9)
    p_fr.font.bold = True
    p_fr.font.color.rgb = SLATE


def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=BORDER, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE):
    """Creates a container card shape with solid fill and clean border."""
    card = slide.shapes.add_shape(shape_type, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    if border_color:
        card.line.color.rgb = border_color
        card.line.width = Pt(1)
    else:
        card.line.fill.background()
    return card


def set_notes(slide, notes_text):
    """Sets speaker defense notes for a slide."""
    slide.notes_slide.notes_text_frame.text = notes_text.strip()


# =============================================================================
# Slide 1: Title Slide (Cover Slide)
# =============================================================================
def build_slide_1(prs, blank_layout):
    slide = prs.slides.add_slide(blank_layout)

    # Full Dark Navy Background
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()

    # Top Accent Gold Strip
    strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.12))
    strip.fill.solid()
    strip.fill.fore_color.rgb = GOLD
    strip.line.fill.background()

    # Institutional Header
    inst_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.45), Inches(11.733), Inches(1.1))
    tf_inst = inst_box.text_frame
    tf_inst.word_wrap = True
    
    p0 = tf_inst.paragraphs[0]
    p0.text = "NEIL GOGTE INSTITUTE OF TECHNOLOGY"
    p0.font.name = FONT_HEADING
    p0.font.size = Pt(20)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(255, 255, 255)
    p0.alignment = PP_ALIGN.CENTER

    p1 = tf_inst.add_paragraph()
    p1.text = "Affiliated to Osmania University · Approved by AICTE · Kachavanisingaram, Hyderabad"
    p1.font.name = FONT_BODY
    p1.font.size = Pt(11)
    p1.font.color.rgb = RGBColor(203, 213, 225)
    p1.alignment = PP_ALIGN.CENTER

    p2 = tf_inst.add_paragraph()
    p2.text = "DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING"
    p2.font.name = FONT_HEADING
    p2.font.size = Pt(13)
    p2.font.bold = True
    p2.font.color.rgb = RGBColor(251, 191, 36) # Amber gold
    p2.alignment = PP_ALIGN.CENTER

    # Project Review Badge
    badge_bg = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(3.66), Inches(1.75), Inches(6.0), Inches(0.42))
    badge_bg.fill.solid()
    badge_bg.fill.fore_color.rgb = RGBColor(30, 64, 115)
    badge_bg.line.color.rgb = RGBColor(59, 130, 246)
    
    tf_badge = badge_bg.text_frame
    p_b = tf_badge.paragraphs[0]
    p_b.text = "MAJOR PROJECT WORK-1 (PW-1) · REVIEW-1 DEFENSE"
    p_b.font.name = FONT_HEADING
    p_b.font.size = Pt(11)
    p_b.font.bold = True
    p_b.font.color.rgb = RGBColor(255, 255, 255)
    p_b.alignment = PP_ALIGN.CENTER

    # Main Project Title Box
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(2.35), Inches(11.733), Inches(1.6))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True

    p_t = tf_title.paragraphs[0]
    p_t.text = "Dynamic Graph Convolutional Network with Temporal Attention for Subject-Independent EEG Motor Imagery Decoding"
    p_t.font.name = FONT_HEADING
    p_t.font.size = Pt(25)
    p_t.font.bold = True
    p_t.font.color.rgb = RGBColor(255, 255, 255)
    p_t.alignment = PP_ALIGN.CENTER

    p_sub = tf_title.add_paragraph()
    p_sub.text = "Architectural Breakthrough: GBP-Net (Graph Band-Power Network with Cross-Subject Pre-Training)"
    p_sub.font.name = FONT_HEADING
    p_sub.font.size = Pt(14)
    p_sub.font.bold = True
    p_sub.font.color.rgb = RGBColor(56, 189, 248) # Sky blue
    p_sub.alignment = PP_ALIGN.CENTER

    # Two Lower Metadata Cards (Presenters vs Supervision)
    card_w = Inches(5.6)
    card_h = Inches(2.4)
    card_y = Inches(4.25)

    # Left Card: Presenters
    c_left = add_card(slide, Inches(0.8), card_y, card_w, card_h, bg_color=RGBColor(15, 34, 64), border_color=RGBColor(43, 108, 176))
    tf_l = c_left.text_frame
    tf_l.word_wrap = True
    tf_l.margin_left = tf_l.margin_top = tf_l.margin_right = tf_l.margin_bottom = Inches(0.18)

    pl_head = tf_l.paragraphs[0]
    pl_head.text = "PROJECT TEAM (BATCH 22)"
    pl_head.font.name = FONT_HEADING
    pl_head.font.size = Pt(12)
    pl_head.font.bold = True
    pl_head.font.color.rgb = RGBColor(251, 191, 36)

    members = [
        ("245323753054", "Saligram Sriram Shravan"),
        ("245323753045", "Parcha Seetha Ramarao"),
        ("245323753030", "M Priyansh")
    ]
    for roll, name in members:
        p_m = tf_l.add_paragraph()
        r_roll = p_m.add_run()
        r_roll.text = f"•  {roll}  —  "
        r_roll.font.bold = True
        r_roll.font.size = Pt(11)
        r_roll.font.color.rgb = RGBColor(147, 197, 253)
        r_name = p_m.add_run()
        r_name.text = name
        r_name.font.size = Pt(11)
        r_name.font.color.rgb = RGBColor(255, 255, 255)

    p_deg = tf_l.add_paragraph()
    p_deg.text = "B.Tech IV Year · VII Semester · CSE"
    p_deg.font.size = Pt(10)
    p_deg.font.color.rgb = RGBColor(148, 163, 184)

    # Right Card: Supervision
    c_right = add_card(slide, Inches(6.933), card_y, card_w, card_h, bg_color=RGBColor(15, 34, 64), border_color=RGBColor(43, 108, 176))
    tf_r = c_right.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_top = tf_r.margin_right = tf_r.margin_bottom = Inches(0.18)

    pr_head = tf_r.paragraphs[0]
    pr_head.text = "ACADEMIC SUPERVISION & DOMAIN"
    pr_head.font.name = FONT_HEADING
    pr_head.font.size = Pt(12)
    pr_head.font.bold = True
    pr_head.font.color.rgb = RGBColor(251, 191, 36)

    supervisors = [
        ("Project Guide:", "Dr. Gagan Kumar"),
        ("Project Coordinator:", "Mr. M. Manoher Rao"),
        ("Head of Department:", "Dr. K. Vinuthna"),
        ("Project Domain:", "Deep Learning · Brain-Computer Interfaces")
    ]
    for role, name in supervisors:
        p_s = tf_r.add_paragraph()
        r_role = p_s.add_run()
        r_role.text = f"•  {role} "
        r_role.font.bold = True
        r_role.font.size = Pt(10.5)
        r_role.font.color.rgb = RGBColor(147, 197, 253)
        r_name = p_s.add_run()
        r_name.text = name
        r_name.font.size = Pt(10.5)
        r_name.font.color.rgb = RGBColor(255, 255, 255)

    p_date = tf_r.add_paragraph()
    p_date.text = "Presentation Date: 5th October 2026 · Room: 301, B-Block, NGIT"
    p_date.font.size = Pt(10)
    p_date.font.color.rgb = RGBColor(148, 163, 184)

    # Footer note
    set_notes(slide, """
Respected evaluators, Head of Department Dr. Vinuthna, Project Coordinator Mr. Manoher Rao, and our guide Dr. Gagan Kumar:
Good morning. We are Batch 22, representing Computer Science & Engineering.
Today, we present our Major Project Work-1 Review-1 defense on:
"Dynamic Graph Convolutional Network with Temporal Attention for Subject-Independent EEG Motor Imagery Decoding," introducing our breakthrough architecture, GBP-Net.
This presentation strictly adheres to NGIT circular guidelines and addresses all four evaluation defense criteria: title justification, abstract clarity, research gap depth, and technical submission quality.
""")


# =============================================================================
# Slide 2: Project Abstract (Mandatory Circular Slide 2)
# =============================================================================
def build_slide_2(prs, blank_layout):
    slide = prs.slides.add_slide(blank_layout)
    add_header(slide, "Project Abstract", "Mandatory Circular Compliance · Executive Overview", 2)

    card_w = Inches(3.8)
    card_h = Inches(5.45)
    card_y = Inches(1.4)

    # Card 1: Problem Statement
    c1 = add_card(slide, Inches(0.6), card_y, card_w, card_h, bg_color=CARD_BG, border_color=CRIMSON)
    # Accent top border
    s1 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), card_y, card_w, Inches(0.38))
    s1.fill.solid()
    s1.fill.fore_color.rgb = CRIMSON
    s1.line.fill.background()
    s1.text_frame.paragraphs[0].text = "1. PROBLEM STATEMENT"
    s1.text_frame.paragraphs[0].font.name = FONT_HEADING
    s1.text_frame.paragraphs[0].font.size = Pt(11)
    s1.text_frame.paragraphs[0].font.bold = True
    s1.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
    s1.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    tf1 = slide.shapes.add_textbox(Inches(0.75), Inches(1.85), Inches(3.5), Inches(4.85)).text_frame
    tf1.word_wrap = True
    tf1.margin_left = tf1.margin_top = tf1.margin_right = tf1.margin_bottom = 0

    points_p1 = [
        ("Motor Imagery BCIs: ", "Translate imagined hand/feet movements into device control, but suffer from low SNR, volume conduction, and high cross-subject variability."),
        ("Euclidean Grid Flaw: ", "Standard CNNs treat scalp electrodes as flat 2D pixels, ignoring 3D spherical scalp neuroanatomy."),
        ("GCNs-Net Baseline Collapse: ", "Prior SOTA GCNs-Net (IEEE TNNLS 2024) reports 93% accuracy using static Pearson graphs on single 6.25ms timepoints."),
        ("Data Leakage Reality: ", "Our audit reveals 3 compounding leakage paths (timepoint shuffling, test-mode BatchNorm). Under genuine leak-free trials, published models fail to generalize."),
        ("Clinical Barrier: ", "Severe drop in cross-subject generalization prevents zero-calibration clinical BCI deployment.")
    ]
    for i, (title, desc) in enumerate(points_p1):
        p = tf1.paragraphs[0] if i == 0 else tf1.add_paragraph()
        p.space_after = Pt(6)
        r1 = p.add_run()
        r1.text = "• " + title
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(10)
        r2.font.color.rgb = CHARCOAL

    # Card 2: Objectives & Methodology
    c2 = add_card(slide, Inches(4.75), card_y, card_w, card_h, bg_color=CARD_BG, border_color=SLATE)
    s2 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.75), card_y, card_w, Inches(0.38))
    s2.fill.solid()
    s2.fill.fore_color.rgb = SLATE
    s2.line.fill.background()
    s2.text_frame.paragraphs[0].text = "2. OBJECTIVES & METHODOLOGY"
    s2.text_frame.paragraphs[0].font.name = FONT_HEADING
    s2.text_frame.paragraphs[0].font.size = Pt(11)
    s2.text_frame.paragraphs[0].font.bold = True
    s2.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
    s2.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    tf2 = slide.shapes.add_textbox(Inches(4.9), Inches(1.85), Inches(3.5), Inches(4.85)).text_frame
    tf2.word_wrap = True
    tf2.margin_left = tf2.margin_top = tf2.margin_right = tf2.margin_bottom = 0

    points_p2 = [
        ("Dynamic Adjacency Learning: ", "Formulate a learnable graph adjacency matrix that dynamically adapts to active motor execution mental states."),
        ("Temporal Band-Power Processing: ", "Decompose signals into μ (8–13 Hz) and β (14–30 Hz) rhythm bands, applying log-variance pooling to capture true multi-second ERD/ERS."),
        ("Signed Chebyshev Filters: ", "Implement stable signed ChebConv (K=2) preventing over-smoothing while capturing negative inter-hemispheric inhibition."),
        ("Cross-Subject Pre-Training: ", "Pre-train spatial filters across 92 background subjects to defeat single-subject data scarcity (~42 trials/class)."),
        ("Strict Leak-Free Protocols: ", "Benchmark under 5-Fold Trial CV and Leave-One-Subject-Out (LOSO) cross-subject evaluation on PhysioNet (109 subjects).")
    ]
    for i, (title, desc) in enumerate(points_p2):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.space_after = Pt(6)
        r1 = p.add_run()
        r1.text = "• " + title
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = SLATE
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(10)
        r2.font.color.rgb = CHARCOAL

    # Card 3: Expected Deliverables & Impact
    c3 = add_card(slide, Inches(8.9), card_y, card_w, card_h, bg_color=CARD_BG, border_color=TEAL)
    s3 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.9), card_y, card_w, Inches(0.38))
    s3.fill.solid()
    s3.fill.fore_color.rgb = TEAL
    s3.line.fill.background()
    s3.text_frame.paragraphs[0].text = "3. EXPECTED OUTCOMES & IMPACT"
    s3.text_frame.paragraphs[0].font.name = FONT_HEADING
    s3.text_frame.paragraphs[0].font.size = Pt(11)
    s3.text_frame.paragraphs[0].font.bold = True
    s3.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
    s3.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    tf3 = slide.shapes.add_textbox(Inches(9.05), Inches(1.85), Inches(3.5), Inches(4.85)).text_frame
    tf3.word_wrap = True
    tf3.margin_left = tf3.margin_top = tf3.margin_right = tf3.margin_bottom = 0

    points_p3 = [
        ("Target Decoding Performance: ", "Formulated to surpass classical CSP and compact CNN baselines under leak-free protocols, targeting performance beyond the 65.0% Vasques ceiling."),
        ("Resolving the Reality Gap: ", "Bridge the gap between published leaky claims (>93%) and legitimate literature ceilings (65.0% Vasques benchmark; 62.4% CSP)."),
        ("Physiological Interpretability: ", "Deliver 2D scalp topographic heatmaps demonstrating learned attention aligns with sensorimotor cortex rhythms (C3/Cz/C4)."),
        ("Definitive Graph Ablation: ", "Empirically resolve whether graph convolutions provide genuine inductive bias over spatial filters when leakage is eliminated."),
        ("Open-Source Benchmark: ", "Deliver modular PyTorch/PyG repository supporting reproducible, leak-free BCI research.")
    ]
    for i, (title, desc) in enumerate(points_p3):
        p = tf3.paragraphs[0] if i == 0 else tf3.add_paragraph()
        p.space_after = Pt(6)
        r1 = p.add_run()
        r1.text = "• " + title
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = TEAL
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(10)
        r2.font.color.rgb = CHARCOAL

    set_notes(slide, """
Respected panel members, Slide 2 strictly satisfies the mandatory circular requirement for the Project Abstract.
We have decomposed the abstract into three crisp pillars:
1. Problem Statement: GCNs-Net claims 93% accuracy on PhysioNet, but processes 6.25ms timepoints in isolation. When audited, timepoint shuffling leaked information across train and test splits. Under genuine continuous trials, published models fail to generalize.
2. Objectives & Methodology: We replace static correlation graphs with dynamic learnable adjacency, introduce temporal filter banks for mu and beta bands, and pre-train spatial filters across 92 subjects.
3. Expected Outcomes: A theoretically sound, leak-free graph model designed to surpass classical baselines, physiologically interpretable topographic scalp maps, and a reproducible evaluation suite for PW-2.
""")


# =============================================================================
# Slide 3: Problem Statement & Domain Challenges
# =============================================================================
def build_slide_3(prs, blank_layout):
    slide = prs.slides.add_slide(blank_layout)
    add_header(slide, "Problem Statement & BCI Domain Challenges", "Motivation & Neurophysiological Bottlenecks", 3)

    card_w = Inches(3.8)
    card_h = Inches(5.45)
    card_y = Inches(1.4)

    # Challenge 1: Non-Euclidean Scalp Topography
    c1 = add_card(slide, Inches(0.6), card_y, card_w, card_h, bg_color=CARD_BG, border_color=BORDER)
    s1 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), card_y, card_w, Inches(0.08))
    s1.fill.solid()
    s1.fill.fore_color.rgb = NAVY
    s1.line.fill.background()

    tf1 = slide.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(3.4), Inches(5.0)).text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "1. Non-Euclidean Topography"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.space_after = Pt(10)

    points1 = [
        ("3D Curved Manifold: ", "EEG electrodes reside on a 3D spherical scalp surface. Distances are geodesic, not planar."),
        ("2D Grid Distortion: ", "Standard 2D CNNs treat channels as rectangular image pixels, distorting true anatomical distances and tearing apart bilateral motor cortex symmetries (C3 and C4)."),
        ("Static Adjacency Pitfall: ", "Existing GCNs compute a single Pearson correlation matrix across the entire dataset. This static graph cannot adapt to state-dependent mental dynamics."),
        ("Inductive Bias Need: ", "Graph networks model non-Euclidean channels naturally, but must be dynamic and constrained to avoid over-smoothing.")
    ]
    for title, desc in points1:
        p = tf1.add_paragraph()
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + title
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(10)
        r2.font.color.rgb = CHARCOAL

    # Challenge 2: Data Scarcity & Cross-Subject Shift
    c2 = add_card(slide, Inches(4.75), card_y, card_w, card_h, bg_color=CARD_BG, border_color=BORDER)
    s2 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(4.75), card_y, card_w, Inches(0.08))
    s2.fill.solid()
    s2.fill.fore_color.rgb = SLATE
    s2.line.fill.background()

    tf2 = slide.shapes.add_textbox(Inches(4.95), Inches(1.6), Inches(3.4), Inches(5.0)).text_frame
    tf2.word_wrap = True

    p = tf2.paragraphs[0]
    p.text = "2. Data Scarcity & Inter-Subject Shift"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = SLATE
    p.space_after = Pt(10)

    points2 = [
        ("Extreme Sample Starvation: ", "In PhysioNet EEGMMIDB, each subject performs only ~42 trials per class (~168 trials total)."),
        ("Overfitting in Deep Models: ", "High-capacity deep architectures (100k+ parameters) rapidly overfit and memorize training trials when trained from scratch."),
        ("Neuroanatomical Variability: ", "Differences in skull bone thickness, cortical gyri folding, and user BCI literacy create severe domain shift between subjects."),
        ("The Calibration Bottleneck: ", "Subject-dependent models demand 30–60 minutes of tedious calibration per patient, making real-world clinical neurorehabilitation infeasible.")
    ]
    for title, desc in points2:
        p = tf2.add_paragraph()
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + title
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = SLATE
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(10)
        r2.font.color.rgb = CHARCOAL

    # Challenge 3: The Published Data Leakage Crisis
    c3 = add_card(slide, Inches(8.9), card_y, card_w, card_h, bg_color=CRIMSON_BG, border_color=CRIMSON)
    s3 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8.9), card_y, card_w, Inches(0.08))
    s3.fill.solid()
    s3.fill.fore_color.rgb = CRIMSON
    s3.line.fill.background()

    tf3 = slide.shapes.add_textbox(Inches(9.1), Inches(1.6), Inches(3.4), Inches(5.0)).text_frame
    tf3.word_wrap = True

    p = tf3.paragraphs[0]
    p.text = "3. The Data Leakage Crisis"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CRIMSON
    p.space_after = Pt(10)

    points3 = [
        ("The 93% SOTA Claim: ", "GCNs-Net (IEEE TNNLS 2024) and BiLSTM-GCN report 93.06% accuracy on PhysioNet 4-class motor imagery."),
        ("Timepoint Shuffling Leak: ", "6.25ms timepoints were randomly shuffled, placing adjacent points of the identical trial into both train and test folds."),
        ("Test-Time BatchNorm Leak: ", "Inference ran with BatchNorm in training mode, recalculating mini-batch statistics over test data and leaking target labels."),
        ("The Literature Reality Gap: ", "When evaluated on continuous trials with zero timepoint leakage, published models fail, while extensive benchmarks confirm the true clean ceiling is 65.0% (Vasques et al., 2026)."),
        ("Literature Silence: ", "No prior paper in literature has audited or corrected this benchmark flaw.")
    ]
    for title, desc in points3:
        p = tf3.add_paragraph()
        p.space_after = Pt(7)
        r1 = p.add_run()
        r1.text = "• " + title
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = CRIMSON
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(10)
        r2.font.color.rgb = CHARCOAL

    set_notes(slide, """
On Slide 3, we detail the core neurophysiological and computational challenges:
First: Non-Euclidean Topography. Standard CNNs flatten electrodes into 2D grids, destroying bilateral symmetry between left and right motor cortices. GCNs solve this, but static correlation graphs cannot adapt to dynamic mental states.
Second: Data Scarcity. 42 trials per class is insufficient to train a 100k-parameter neural network from scratch.
Third: The Data Leakage Crisis. Our thorough audit of the published codebase (SuperBruceJia/EEG-DL) revealed that published 93% accuracies rely on shuffling 6.25ms timepoints. When evaluated on clean continuous trials, published claims prove unreplicable, falling short of classical literature baselines.
""")


# =============================================================================
# Slide 4: Project Objectives & Scope of Work
# =============================================================================
def build_slide_4(prs, blank_layout):
    slide = prs.slides.add_slide(blank_layout)
    add_header(slide, "Project Objectives & Scope of Work", "Four Systematic Pillars of Investigation", 4)

    card_w = Inches(5.89)
    card_h = Inches(2.55)
    
    # 2x2 Grid of Cards
    coords = [
        (Inches(0.6), Inches(1.45)),   # Obj 1 (Top Left)
        (Inches(6.84), Inches(1.45)),  # Obj 2 (Top Right)
        (Inches(0.6), Inches(4.25)),   # Obj 3 (Bottom Left)
        (Inches(6.84), Inches(4.25))   # Obj 4 (Bottom Right)
    ]

    objectives = [
        (
            "Objective 1: Baseline Reproduction & Rigorous Audit",
            NAVY,
            [
                ("Methodological Baseline Audit: ", "Replicate and deconstruct the published GCNs-Net pipeline on PhysioNet EEGMMIDB to isolate literature leakage pathways."),
                ("Leakage Isolation: ", "Empirically dissect the 3 leakage vectors (timepoint shuffling, test-mode BatchNorm, transductive graphs)."),
                ("Open Benchmark Standard: ", "Establish a verified, leak-free trial-level splitting protocol for fair BCI model evaluation.")
            ]
        ),
        (
            "Objective 2: Dynamic Spatial-Temporal Architecture (GBP-Net)",
            SLATE,
            [
                ("Learnable Dynamic Adjacency: ", "Formulate a dynamic adjacency matrix that optimizes inter-electrode graph edge weights end-to-end."),
                ("Temporal Filter Banks: ", "Integrate multi-band FIR filtering decomposing EEG into μ (8–13 Hz) and β (14–30 Hz) rhythm bands."),
                ("Band-Power Aggregation: ", "Incorporate log-variance pooling over multi-second windows (1–2.5s) to capture physiological ERD/ERS power.")
            ]
        ),
        (
            "Objective 3: Controlled Inductive Bias Isolation",
            TEAL,
            [
                ("Direct Spatial Ablation: ", "Compare Chebyshev spectral graph convolutions against unconstrained spatial linear projections on identical folds."),
                ("Hyperparameter Neutrality: ", "Hold all temporal filtering, pooling, and optimizer settings identical to isolate graph inductive bias."),
                ("Definitive Empirical Answer: ", "Resolve whether graph convolutions provide genuine spatial benefits or merely act as regularizers.")
            ]
        ),
        (
            "Objective 4: Cross-Subject Transfer & Interpretability",
            GOLD,
            [
                ("Two-Stage Transfer Pipeline: ", "Pre-train spatial filters across 92 subjects (15,000+ trials), then fine-tune or evaluate zero-shot on target subjects."),
                ("Overcoming Data Scarcity: ", "Address single-subject sample starvation via population-level representation learning."),
                ("Scalp Topographic Validation: ", "Project learned graph weights onto 2D scalp maps to confirm alignment with contralateral C3/Cz/C4 activations.")
            ]
        )
    ]

    for (left, top), (title, color, bullets) in zip(coords, objectives):
        c = add_card(slide, left, top, card_w, card_h, bg_color=CARD_BG, border_color=BORDER)
        # Accent top bar
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, card_w, Inches(0.06))
        bar.fill.solid()
        bar.fill.fore_color.rgb = color
        bar.line.fill.background()

        tf = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), card_w - Inches(0.4), card_h - Inches(0.3)).text_frame
        tf.word_wrap = True

        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.name = FONT_HEADING
        p0.font.size = Pt(13)
        p0.font.bold = True
        p0.font.color.rgb = color
        p0.space_after = Pt(6)

        for b_title, b_desc in bullets:
            p = tf.add_paragraph()
            p.space_after = Pt(4)
            r1 = p.add_run()
            r1.text = "• " + b_title
            r1.font.bold = True
            r1.font.size = Pt(10)
            r1.font.color.rgb = CHARCOAL
            r2 = p.add_run()
            r2.text = b_desc
            r2.font.size = Pt(9.5)
            r2.font.color.rgb = CHARCOAL

    set_notes(slide, """
Slide 4 details our four systematic objectives:
1. Baseline Reproduction & Audit: Replicate GCNs-Net, verify the pipeline, and document the 3 data leakage pathways.
2. Dynamic Spatial-Temporal Architecture: Design GBP-Net with dynamic adjacency learning, filter banks for mu and beta bands, and band-power pooling.
3. Controlled Graph Isolation: Rigorously test ChebConv against unconstrained spatial projections on identical folds to prove whether graph convolutions truly help.
4. Cross-Subject Transfer & Interpretability: Pre-train across 92 subjects to conquer data scarcity, and generate 2D scalp heatmaps conforming to C3/Cz/C4 sensorimotor physiology.
""")


# =============================================================================
# Slide 5: Literature Survey Table (Part 1: Spatial Filters & Compact CNNs)
# =============================================================================
def build_slide_5(prs, blank_layout):
    slide = prs.slides.add_slide(blank_layout)
    add_header(slide, "Literature Survey: Classical Spatial Filters & Compact CNNs", "Mandatory Circular Compliance · Table Part 1 (6 Key References)", 5)

    rows, cols = 7, 6
    left, top, width, height = Inches(0.6), Inches(1.4), Inches(12.133), Inches(5.4)
    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    col_widths = [Inches(1.5), Inches(2.2), Inches(2.3), Inches(1.4), Inches(2.2), Inches(2.533)]
    for idx, w in enumerate(col_widths):
        table.columns[idx].width = w

    headers = ["Author(s) & Year", "Paper Title & Venue", "Methodology / Algorithm", "Dataset Used", "Key Findings", "Limitations / Research Gaps"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = FONT_HEADING
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)
        p.alignment = PP_ALIGN.CENTER

    data = [
        (
            "Ramoser et al.\n(2000)",
            "Optimal Spatial Filtering of Single-Trial EEG\n(IEEE TRE)",
            "Common Spatial Patterns (CSP): discriminative spatial projection via covariance ratio maximization",
            "Motor Imagery\nEEG (2-class)",
            "Foundational spatial filter; establishes variance maximization as optimal for MI.",
            "Binary by construction; highly sensitive to noise and covariance estimation with few trials."
        ),
        (
            "Blankertz et al.\n(2008)",
            "Optimizing Spatial Filters for Robust EEG Single-Trial Analysis\n(IEEE SPM)",
            "Regularized CSP with covariance shrinkage and robust covariance estimation",
            "Motor Imagery\nEEG",
            "Shrinkage regularisation prevents overfitting when training trials are scarce (<50 trials).",
            "Linear spatial filter only; does not address deep non-linear dynamics or cross-subject transfer."
        ),
        (
            "Ang et al.\n(2008)",
            "Filter Bank Common Spatial Pattern (FBCSP)\n(IJCNN)",
            "Decomposes EEG into multiple frequency sub-bands + CSP per band + Mutual Information selection",
            "BCI Competition IV\nDataset 2a",
            "Multi-band decomposition captures distinct frequency dynamics, winning BCI Competition IV.",
            "Fixed hand-crafted frequency bands; no learnable end-to-end neural representation."
        ),
        (
            "Lawhern et al.\n(2018)",
            "EEGNet: A Compact CNN for EEG-Based BCIs\n(J. Neural Eng.)",
            "2D Depthwise and separable convolutions over temporal then spatial dimensions (~2.6k params)",
            "4 BCI Paradigms\n(MI, P300, ERN, RSV)",
            "Compact, parameter-efficient baseline generalising across multiple EEG paradigms.",
            "Treats electrodes as a flat 2D grid; designed for 2–4s continuous windows; ignores graph topology."
        ),
        (
            "Schirrmeister et al.\n(2017)",
            "Deep Learning with CNNs for EEG Decoding & Visualization\n(Human Brain Mapp.)",
            "ShallowConvNet: Temporal conv → Spatial conv → Square → Mean Pool → Log activation",
            "High-Gamma,\nBCI IV-2a",
            "Matches/exceeds FBCSP; proves that spatial filtering followed by power pooling matches EEG physics.",
            "Euclidean channel grid assumption; no inter-channel graph structure; high within-subject variance."
        ),
        (
            "Mane et al.\n(2021)",
            "FBCNet: A Multi-View CNN for Brain-Computer Interface\n(IEEE TNSRE)",
            "Multi-band filter bank + spatial convolution per band + temporal variance aggregation layer",
            "BCI Competition IV-2a\n& IV-2b",
            "Embeds FBCSP's multi-view spectral decomposition and variance pooling inside an end-to-end CNN.",
            "Uses fixed hand-crafted filter banks; lacks dynamic graph connectivity across non-Euclidean channels."
        )
    ]

    for i, row in enumerate(data):
        row_idx = i + 1
        bg = RGBColor(248, 250, 252) if i % 2 == 1 else RGBColor(255, 255, 255)
        for j, val in enumerate(row):
            cell = table.cell(row_idx, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_BODY
            p.font.size = Pt(8.5)
            p.font.color.rgb = CHARCOAL
            if j == 0:
                p.font.bold = True
                p.alignment = PP_ALIGN.CENTER
            elif j == 3:
                p.alignment = PP_ALIGN.CENTER

    set_notes(slide, """
Respected evaluators, Slides 5 and 6 present our comprehensive Literature Survey Tables, satisfying the circular requirement of at least 10 key references.
Part 1 covers the classical spatial filtering and compact CNN lineage:
- Ramoser (2000) and Blankertz (2008) established Common Spatial Patterns and covariance shrinkage.
- Ang et al. (2008) introduced Filter Bank CSP, proving that multi-band frequency decomposition is critical.
- Lawhern (2018), Schirrmeister (2017), and Mane (2021) demonstrated that CNNs succeed when they mirror CSP physics: temporal filtering, followed by spatial filtering, followed by power/variance pooling. This exact neurophysiological ordering forms the core foundation of our GBP-Net architecture.
""")


# =============================================================================
# Slide 6: Literature Survey Table (Part 2: GCNs & Evaluation Baselines)
# =============================================================================
def build_slide_6(prs, blank_layout):
    slide = prs.slides.add_slide(blank_layout)
    add_header(slide, "Literature Survey: Graph Neural Networks & Evaluation Baselines", "Mandatory Circular Compliance · Table Part 2 (6 Key References)", 6)

    rows, cols = 7, 6
    left, top, width, height = Inches(0.6), Inches(1.4), Inches(12.133), Inches(5.4)
    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    col_widths = [Inches(1.5), Inches(2.2), Inches(2.3), Inches(1.4), Inches(2.2), Inches(2.533)]
    for idx, w in enumerate(col_widths):
        table.columns[idx].width = w

    headers = ["Author(s) & Year", "Paper Title & Venue", "Methodology / Algorithm", "Dataset Used", "Key Findings", "Limitations / Research Gaps"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = FONT_HEADING
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)
        p.alignment = PP_ALIGN.CENTER

    data = [
        (
            "Hou et al.\n(2024)",
            "GCNs-Net: Spectral GCN for Time-Resolved EEG\n(IEEE TNNLS 35(6))",
            "Spectral GCN: Pearson correlation adjacency → Normalized Laplacian → ChebConv (K=2) → Graclus pooling",
            "PhysioNet EEGMMIDB\n(4-class)",
            "Reports 93.06% subject-level and 88.57% group-level accuracy. DIRECT BASELINE OF OUR PROJECT.",
            "Samples are single 6.25ms timepoints; split shuffles timepoints across train/test, causing severe data leakage. Fails under continuous trial splits with zero sample overlap."
        ),
        (
            "Hou et al.\n(2022)",
            "Attention BiLSTM-GCN for Motor Imagery Recognition\n(Front. Bioeng. Biotech.)",
            "Combines spectral graph convolution with temporal Attention + Bidirectional LSTM layers",
            "PhysioNet EEGMMIDB\n(4-class)",
            "Adds temporal sequence modeling to the GCN lineage to capture timepoint dependencies.",
            "Inherits the identical flawed timepoint shuffling split from GCNs-Net; leakage remains unaddressed."
        ),
        (
            "Deng et al.\n(2025)",
            "MHD-GCN: Multi-Level Hierarchical Dynamic GCN\n(Neurocomputing)",
            "Spatiotemporal attention + hierarchical global/local dynamic graph topologies via community clustering",
            "BCI Competition IV;\nPhysioNet",
            "Learns multi-level dynamic inter-channel graph structure; addresses channel connectivity shifts.",
            "High computational complexity; does not isolate graph contribution against non-graph spatial controls."
        ),
        (
            "Meng et al.\n(2026)",
            "SAMGCN: Self-Adaptive Multilevel GCN for MI\n(Biomed. Signal Process.)",
            "Channel position encoding + self-attention graph embedding; multi-level GCN to mitigate over-smoothing",
            "BCI IV-2a,\nOpenBMI",
            "Explicitly identifies GCN over-smoothing as a major failure mode; improves feature distinctiveness.",
            "Asserts over-smoothing but does not test against simple spatial linear filters on identical folds."
        ),
        (
            "Vasques et al.\n(2026)",
            "Subject Heterogeneity in EEG Motor Imagery\n(bioRxiv Benchmark)",
            "Evaluates 216,714 pipeline configurations across 109 subjects; portfolio-based model selection",
            "PhysioNet EEGMMIDB\n(109 subjects)",
            "Best single leak-free pipeline on PhysioNet reaches 65.0%; reveals 93 distinct subject winners.",
            "Establishes realistic empirical ceiling on PhysioNet; proves that 90%+ claims are impossible without leakage."
        ),
        (
            "Kapoor &\nNarayanan (2023)",
            "Leakage & Reproducibility Crisis in ML Science\n(Patterns 4(9))",
            "Systematic survey and taxonomy of data leakage across 294 papers in 17 scientific fields",
            "294 Papers across\n17 Fields",
            "Data leakage is pervasive in applied ML, producing inflated, irreproducible accuracy claims.",
            "Covers computer vision, medical imaging, and genomics, but does not examine EEG or BCI models specifically."
        )
    ]

    for i, row in enumerate(data):
        row_idx = i + 1
        bg = RGBColor(248, 250, 252) if i % 2 == 1 else RGBColor(255, 255, 255)
        for j, val in enumerate(row):
            cell = table.cell(row_idx, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_BODY
            p.font.size = Pt(8.5)
            p.font.color.rgb = CHARCOAL
            if j == 0:
                p.font.bold = True
                p.alignment = PP_ALIGN.CENTER
            elif j == 3:
                p.alignment = PP_ALIGN.CENTER

    set_notes(slide, """
Slide 6 presents Part 2 of our literature table, focusing on Graph Neural Networks and Evaluation Baselines:
- Across Slides 5 and 6, we have summarized 12 key research papers, well exceeding the 10-paper minimum.
- Hou et al. (2024, IEEE TNNLS) is our direct baseline: GCNs-Net. While it reports 93.06%, we discovered its timepoint shuffling split causes severe data leakage.
- Vasques et al. (2026) benchmarked 216,714 pipeline configurations on PhysioNet and showed the true leak-free ceiling is 65.0%.
- Kapoor & Narayanan (2023) documented the reproducibility crisis in Patterns; our project directly fills this gap for EEG and Brain-Computer Interfaces.
""")


# =============================================================================
# Slide 7: Title Defense & Innovative Aspects
# =============================================================================
def build_slide_7(prs, blank_layout):
    slide = prs.slides.add_slide(blank_layout)
    add_header(slide, "Title Justification & Innovative Aspects", "Evaluation Criterion 1 Compliance · Rigorous Defense", 7)

    card_w = Inches(3.8)
    card_h = Inches(5.45)
    card_y = Inches(1.4)

    # Pillar 1: Dynamic Graph Convolutions
    c1 = add_card(slide, Inches(0.6), card_y, card_w, card_h, bg_color=CARD_BG, border_color=NAVY)
    s1 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), card_y, card_w, Inches(0.42))
    s1.fill.solid()
    s1.fill.fore_color.rgb = NAVY
    s1.line.fill.background()
    s1.text_frame.paragraphs[0].text = "DYNAMIC GRAPH CONVOLUTIONS"
    s1.text_frame.paragraphs[0].font.name = FONT_HEADING
    s1.text_frame.paragraphs[0].font.size = Pt(11)
    s1.text_frame.paragraphs[0].font.bold = True
    s1.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
    s1.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    tf1 = slide.shapes.add_textbox(Inches(0.75), Inches(1.9), Inches(3.5), Inches(4.8)).text_frame
    tf1.word_wrap = True

    pts1 = [
        ("Why Graph? ", "Scalp EEG electrodes lie on a curved 3D manifold. Euclidean 2D grids distort geodesic distances and split bilateral motor cortex channels."),
        ("Why Dynamic? ", "Prior GCNs rely on static Pearson correlation matrices. Static graphs assume fixed brain connectivity, ignoring state-dependent transitions during motor imagery."),
        ("Learnable Adjacency: ", "Our dynamic adjacency matrix adapts weights end-to-end to reflect active sensorimotor synchronizations."),
        ("Signed Chebyshev Filters: ", "Formulates signed ChebConv (K=2) that captures negative inter-hemispheric inhibition without numerical divergence or over-smoothing.")
    ]
    for i, (title, desc) in enumerate(pts1):
        p = tf1.paragraphs[0] if i == 0 else tf1.add_paragraph()
        p.space_after = Pt(7)
        r1 = p.add_run()
        r1.text = "• " + title
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(10)
        r2.font.color.rgb = CHARCOAL

    # Pillar 2: Temporal Attention & Band-Power
    c2 = add_card(slide, Inches(4.75), card_y, card_w, card_h, bg_color=CARD_BG, border_color=SLATE)
    s2 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.75), card_y, card_w, Inches(0.42))
    s2.fill.solid()
    s2.fill.fore_color.rgb = SLATE
    s2.line.fill.background()
    s2.text_frame.paragraphs[0].text = "TEMPORAL ATTENTION & BAND-POWER"
    s2.text_frame.paragraphs[0].font.name = FONT_HEADING
    s2.text_frame.paragraphs[0].font.size = Pt(11)
    s2.text_frame.paragraphs[0].font.bold = True
    s2.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
    s2.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    tf2 = slide.shapes.add_textbox(Inches(4.9), Inches(1.9), Inches(3.5), Inches(4.8)).text_frame
    tf2.word_wrap = True

    pts2 = [
        ("Neurophysiological Basis: ", "Motor Imagery is governed by Event-Related Desynchronization (ERD) in μ (8–13 Hz) and β (14–30 Hz) rhythm bands evolving over 1.0–2.5 seconds."),
        ("The Single-Sample Flaw: ", "GCNs-Net processes 6.25ms isolated timepoints, completely incapable of extracting oscillatory band power."),
        ("Multi-Band Filter Banks: ", "Parallel temporal filters isolate mu and beta rhythms, providing multi-view spectral representations."),
        ("Log-Variance Power Pooling: ", "Computes temporal variance: log(1/T ∑ x²), embedding the proven physics of CSP inside a differentiable graph architecture.")
    ]
    for i, (title, desc) in enumerate(pts2):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.space_after = Pt(7)
        r1 = p.add_run()
        r1.text = "• " + title
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = SLATE
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(10)
        r2.font.color.rgb = CHARCOAL

    # Pillar 3: Subject-Independent Decoding
    c3 = add_card(slide, Inches(8.9), card_y, card_w, card_h, bg_color=CARD_BG, border_color=TEAL)
    s3 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.9), card_y, card_w, Inches(0.42))
    s3.fill.solid()
    s3.fill.fore_color.rgb = TEAL
    s3.line.fill.background()
    s3.text_frame.paragraphs[0].text = "SUBJECT-INDEPENDENT DECODING"
    s3.text_frame.paragraphs[0].font.name = FONT_HEADING
    s3.text_frame.paragraphs[0].font.size = Pt(11)
    s3.text_frame.paragraphs[0].font.bold = True
    s3.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
    s3.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    tf3 = slide.shapes.add_textbox(Inches(9.05), Inches(1.9), Inches(3.5), Inches(4.8)).text_frame
    tf3.word_wrap = True

    pts3 = [
        ("The Clinical Reality: ", "Subject-dependent models require 30–60 mins of calibration per user. Practical clinical BCIs must be plug-and-play (subject-independent)."),
        ("Solving Sample Starvation: ", "Single-subject training (~42 trials/class) leads to severe overfitting. Cross-subject pre-training leverages 15,000+ trials across 92 subjects."),
        ("Generalized Spatial Filters: ", "Stage 1 pre-trains a universal spatial representation capturing invariant sensorimotor manifold geometry."),
        ("LOSO Cross-Validation: ", "Evaluated strictly via Leave-One-Subject-Out protocols, guaranteeing zero test subject data leakage and genuine generalization.")
    ]
    for i, (title, desc) in enumerate(pts3):
        p = tf3.paragraphs[0] if i == 0 else tf3.add_paragraph()
        p.space_after = Pt(7)
        r1 = p.add_run()
        r1.text = "• " + title
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = TEAL
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(10)
        r2.font.color.rgb = CHARCOAL

    set_notes(slide, """
Slide 7 directly addresses Assessment Criterion 1: Title Justification & Innovative Aspect.
We systematically justify every component of our registered title:
1. Dynamic Graph Convolutional Network: Justified because EEG channels are on a 3D non-Euclidean sphere, and brain connectivity changes dynamically during motor execution.
2. Temporal Attention & Band-Power: Justified because motor imagery is an oscillatory band-power phenomenon requiring multi-second temporal context (mu and beta bands), which single-timepoint GCNs completely miss.
3. Subject-Independent Decoding: Justified because practical clinical BCIs cannot require tedious per-user calibration. We introduce cross-subject pre-training on 92 subjects to solve single-subject data scarcity.
""")


# =============================================================================
# Slide 8: Research Gaps Identified from Literature
# =============================================================================
def build_slide_8(prs, blank_layout):
    slide = prs.slides.add_slide(blank_layout)
    add_header(slide, "Research Gaps Identified from Literature", "Evaluation Criterion 3 Compliance · Systematic Evidence (G1 – G5)", 8)

    card_w = Inches(5.89)
    card_h = Inches(1.68)
    gap_y = [Inches(1.42), Inches(3.22), Inches(5.02)]

    gaps = [
        # Col 1
        (Inches(0.6), gap_y[0], "Gap 1 (G1): Pervasive Data Leakage in Published GCNs", CRIMSON,
         "Literature Flaw: ", "GCNs-Net (IEEE TNNLS 2024) and BiLSTM-GCN randomly shuffle 6.25ms timepoints across splits.",
         "Our Resolution: ", "Establish strict trial-level partitioning with zero sample overlap, eliminating timepoint leakage and creating an uncorrupted benchmark standard."),

        (Inches(0.6), gap_y[1], "Gap 2 (G2): Violation of Physical Temporal Scales", NAVY,
         "Literature Flaw: ", "Single-timepoint models discard temporal context; neurophysiology proves ERD/ERS evolves over 1.0–2.5s.",
         "Our Resolution: ", "Process continuous 2.0s multi-second windows with multi-band filter banks (μ: 8–13 Hz, β: 14–30 Hz) and band-power pooling."),

        (Inches(0.6), gap_y[2], "Gap 3 (G3): Lack of Graph Inductive Bias Isolation", SLATE,
         "Literature Flaw: ", "Prior GCN papers never compare ChebConv against unconstrained spatial linear projections on identical folds.",
         "Our Resolution: ", "Conduct controlled ablation holding all temporal and pooling parameters constant to empirically verify the value of graph structure."),

        # Col 2
        (Inches(6.84), gap_y[0], "Gap 4 (G4): Structural Dataset Confound in PhysioNet", GOLD,
         "Literature Flaw: ", "Left/right fist trials are in runs 4/8/12; both-fists/feet are in runs 6/10/14, introducing run-level baseline drift.",
         "Our Resolution: ", "Apply strict 4–40 Hz bandpass filtering and baseline detrending to eliminate run-level low-frequency drift confounds."),

        (Inches(6.84), gap_y[1], "Gap 5 (G5): Extreme Cross-Subject Sample Starvation", TEAL,
         "Literature Flaw: ", "Single subjects provide only ~42 trials per class, causing deep networks to overfit when trained from scratch.",
         "Our Resolution: ", "Introduce 2-stage transfer learning: pre-train spatial representations across 92 subjects (15,000+ trials), then fine-tune."),

        (Inches(6.84), gap_y[2], "Impact Summary: Overcoming Literature Reality Gap", CRIMSON,
         "Scientific Impact: ", "Published 93.06% SOTA relies on timepoint leakage; true literature benchmark ceiling on PhysioNet is 65.00% across 216k pipelines (Vasques et al., 2026).",
         "Project Goal: ", "Formulate GBP-Net with cross-subject pre-training to conquer sample scarcity and surpass the 65.0% Vasques ceiling under strict trial-level CV.")
    ]

    for left, top, title, color, tag1, desc1, tag2, desc2 in gaps:
        c = add_card(slide, left, top, card_w, card_h, bg_color=CARD_BG, border_color=BORDER)
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, card_w, Inches(0.04))
        bar.fill.solid()
        bar.fill.fore_color.rgb = color
        bar.line.fill.background()

        tf = slide.shapes.add_textbox(left + Inches(0.18), top + Inches(0.1), card_w - Inches(0.36), card_h - Inches(0.18)).text_frame
        tf.word_wrap = True

        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.name = FONT_HEADING
        p0.font.size = Pt(11)
        p0.font.bold = True
        p0.font.color.rgb = color
        p0.space_after = Pt(3)

        p1 = tf.add_paragraph()
        p1.space_after = Pt(2)
        r1a = p1.add_run()
        r1a.text = "• " + tag1
        r1a.font.bold = True
        r1a.font.size = Pt(9.5)
        r1a.font.color.rgb = CHARCOAL
        r1b = p1.add_run()
        r1b.text = desc1
        r1b.font.size = Pt(9)
        r1b.font.color.rgb = CHARCOAL

        p2 = tf.add_paragraph()
        r2a = p2.add_run()
        r2a.text = "• " + tag2
        r2a.font.bold = True
        r2a.font.size = Pt(9.5)
        r2a.font.color.rgb = TEAL if color != CRIMSON else CRIMSON
        r2b = p2.add_run()
        r2b.text = desc2
        r2b.font.size = Pt(9)
        r2b.font.color.rgb = CHARCOAL

    set_notes(slide, """
Slide 8 directly addresses Assessment Criterion 3: Depth and accuracy in identifying research gaps.
We present five concrete, grounded research gaps:
G1: Pervasive Data Leakage across timepoints in published GCN literature.
G2: Violation of Physical Temporal Scales by processing 6.25ms points instead of 2.0s windows.
G3: Lack of Graph Isolation Controls, where no prior paper compared ChebConv against unconstrained spatial linear projections on identical folds.
G4: The Structural Run Confound in EEGMMIDB, which we eliminate through strict bandpass filtering.
G5: Extreme Cross-Subject Sample Starvation, which we solve through 92-subject pre-training.
""")


# =============================================================================
# Slide 9: Proposed Architecture: GBP-Net
# =============================================================================
def build_slide_9(prs, blank_layout):
    slide = prs.slides.add_slide(blank_layout)
    add_header(slide, "Proposed Architecture: GBP-Net", "Graph Band-Power Network with Cross-Subject Pre-Training", 9)

    # Left Container (Figure 5)
    c_left = add_card(slide, Inches(0.6), Inches(1.4), Inches(7.5), Inches(5.45), bg_color=CARD_BG, border_color=BORDER)
    
    # Header inside Left Container
    tf_lh = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(7.1), Inches(0.35)).text_frame
    p_lh = tf_lh.paragraphs[0]
    p_lh.text = "GBP-Net End-to-End Pipeline & Two-Stage Training Schematic"
    p_lh.font.name = FONT_HEADING
    p_lh.font.size = Pt(12)
    p_lh.font.bold = True
    p_lh.font.color.rgb = NAVY

    # Embed Figure 5 (GBP-Net)
    fig_path = os.path.join("figures", "review_2", "fig5_gbpnet.png")
    if os.path.exists(fig_path):
        slide.shapes.add_picture(fig_path, Inches(0.8), Inches(1.9), width=Inches(7.1))

    # Caption / Stage Summary below figure
    tf_c = slide.shapes.add_textbox(Inches(0.8), Inches(5.05), Inches(7.1), Inches(1.7)).text_frame
    tf_c.word_wrap = True
    p_c1 = tf_c.paragraphs[0]
    p_c1.text = "• Stage 1 (Pre-Training): 92 background subjects (15,000+ trials) train population spatial filters."
    p_c1.font.size = Pt(9.5)
    p_c1.font.bold = True
    p_c1.font.color.rgb = SLATE

    p_c2 = tf_c.add_paragraph()
    p_c2.text = "• Stage 2 (Fine-Tuning / Testing): Target subject data passes through signed ChebConv and log-variance pooling to extract true neurophysiological band power."
    p_c2.font.size = Pt(9)
    p_c2.font.color.rgb = CHARCOAL

    # Right Container (Detailed Mathematical & Architectural Walkthrough)
    c_right = add_card(slide, Inches(8.35), Inches(1.4), Inches(4.383), Inches(5.45), bg_color=CARD_BG, border_color=BORDER)
    bar_r = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8.35), Inches(1.4), Inches(4.383), Inches(0.06))
    bar_r.fill.solid()
    bar_r.fill.fore_color.rgb = TEAL
    bar_r.line.fill.background()

    tf_r = slide.shapes.add_textbox(Inches(8.55), Inches(1.55), Inches(4.0), Inches(5.15)).text_frame
    tf_r.word_wrap = True

    pr_head = tf_r.paragraphs[0]
    pr_head.text = "Architectural Formulation"
    pr_head.font.name = FONT_HEADING
    pr_head.font.size = Pt(13)
    pr_head.font.bold = True
    pr_head.font.color.rgb = TEAL
    pr_head.space_after = Pt(8)

    arch_points = [
        ("1. Multi-Band Filter Bank: ", "Parallel FIR bandpass filters decompose raw EEG into μ (8–13 Hz) and β (14–30 Hz) rhythm sub-bands, isolating motor imagery frequencies."),
        ("2. Dynamic Adjacency (A_dyn): ", "Learnable symmetric adjacency matrix optimized via gradient descent: A = σ(W_a · W_a^T), adapting to subject-specific electrode interactions."),
        ("3. Signed ChebConv (K=2): ", "Chebyshev recurrence: T₀(x) = x, T₁(x) = L̃x, T₂(x) = 2L̃T₁(x) − T₀(x). Captures negative inter-hemispheric inhibition without numerical instability."),
        ("4. Band-Power Log-Variance: ", "Temporal aggregation: P = log( (1/T) ∑ xₜ² + ε ). Converts continuous dynamics into stable power features, mirroring CSP physics."),
        ("5. Population Pre-Training: ", "Pre-trains spatial weights on 92 subjects, overcoming single-subject sample starvation (~42 trials/class).")
    ]
    for title, desc in arch_points:
        p = tf_r.add_paragraph()
        p.space_after = Pt(6)
        r1 = p.add_run()
        r1.text = "• " + title
        r1.font.bold = True
        r1.font.size = Pt(10)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = CHARCOAL

    set_notes(slide, """
Slide 9 showcases our proposed architecture: GBP-Net (Graph Band-Power Network).
As illustrated in Figure 5 on the left:
1. In Stage 1, we solve the extreme data scarcity problem by pre-training spatial filters across 92 background subjects, creating robust, population-level spatial representations.
2. In Stage 2, incoming 2.0-second multi-channel EEG signals are split into mu (8–13 Hz) and beta (14–30 Hz) rhythm bands.
3. Signals are processed by signed Chebyshev graph convolutions over a learnable dynamic adjacency matrix.
4. Next, log-variance pooling computes the temporal band power, embedding the proven physics of Common Spatial Patterns inside deep learning.
5. A lightweight linear layer classifies the 4 motor imagery classes.
""")


# =============================================================================
# Slide 10: Empirical Reality Gap & Leakage Audit
# =============================================================================
def build_slide_10(prs, blank_layout):
    slide = prs.slides.add_slide(blank_layout)
    add_header(slide, "Methodological Audit: Published Data Leakage Crisis", "Systematic Deconstruction of Literature Evaluation Pitfalls on PhysioNet", 10)

    # Left Container (Figure 4)
    c_left = add_card(slide, Inches(0.6), Inches(1.4), Inches(7.5), Inches(5.45), bg_color=CARD_BG, border_color=BORDER)
    
    tf_lh = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(7.1), Inches(0.35)).text_frame
    p_lh = tf_lh.paragraphs[0]
    p_lh.text = "Data Leakage Audit & Performance Discrepancy (PhysioNet EEGMMIDB)"
    p_lh.font.name = FONT_HEADING
    p_lh.font.size = Pt(12)
    p_lh.font.bold = True
    p_lh.font.color.rgb = CRIMSON

    # Embed Figure 4 (Leakage Audit)
    fig_path = os.path.join("figures", "review_2", "fig4_leakage_audit_nip.png")
    if os.path.exists(fig_path):
        slide.shapes.add_picture(fig_path, Inches(0.8), Inches(1.9), width=Inches(7.1))

    # Metric Comparison Box below figure
    metric_box = add_card(slide, Inches(0.8), Inches(5.12), Inches(7.1), Inches(1.62), bg_color=CRIMSON_BG, border_color=CRIMSON)
    tf_mb = metric_box.text_frame
    tf_mb.word_wrap = True
    tf_mb.margin_left = tf_mb.margin_right = Inches(0.15)
    tf_mb.margin_top = Inches(0.05)

    pm_head = tf_mb.paragraphs[0]
    pm_head.text = "CRITICAL TAKEAWAYS FROM LITERATURE AUDIT & UNBIASED BENCHMARKS:"
    pm_head.font.name = FONT_HEADING
    pm_head.font.size = Pt(10)
    pm_head.font.bold = True
    pm_head.font.color.rgb = CRIMSON

    pm_row = tf_mb.add_paragraph()
    pm_row.space_after = Pt(2)
    metrics_str = [
        ("Published Leaky Claims in Literature: ", "93.06% (GCNs-Net) / 94.20% (BiLSTM-GCN)"),
        ("Underlying Methodological Cause: ", "Timepoint shuffling + eval BatchNorm + transductive graphs"),
        ("True Empirical Literature Ceiling: ", "65.00% across 216,714 pipelines (Vasques et al., 2026)"),
        ("Classical Spatial Filtering Baseline: ", "62.39% (CSP+LDA) on continuous PhysioNet trials"),
        ("Proposed GBP-Net Design Target: ", "Conquer data scarcity via cross-subject pre-training without leakage")
    ]
    for label, val in metrics_str:
        p_m = tf_mb.add_paragraph()
        r1 = p_m.add_run()
        r1.text = "• " + label
        r1.font.bold = True
        r1.font.size = Pt(8.8)
        r1.font.color.rgb = NAVY
        r2 = p_m.add_run()
        r2.text = val
        r2.font.bold = True
        r2.font.size = Pt(9.2)
        r2.font.color.rgb = CRIMSON if "50.90" in val else (TEAL if "73.01" in val else CHARCOAL)

    # Right Container (Detailed Leakage Breakdown)
    c_right = add_card(slide, Inches(8.35), Inches(1.4), Inches(4.383), Inches(5.45), bg_color=CARD_BG, border_color=BORDER)
    bar_r = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8.35), Inches(1.4), Inches(4.383), Inches(0.06))
    bar_r.fill.solid()
    bar_r.fill.fore_color.rgb = CRIMSON
    bar_r.line.fill.background()

    tf_r = slide.shapes.add_textbox(Inches(8.55), Inches(1.55), Inches(4.0), Inches(5.15)).text_frame
    tf_r.word_wrap = True

    pr_head = tf_r.paragraphs[0]
    pr_head.text = "Three Compounding Leakage Vectors"
    pr_head.font.name = FONT_HEADING
    pr_head.font.size = Pt(13)
    pr_head.font.bold = True
    pr_head.font.color.rgb = CRIMSON
    pr_head.space_after = Pt(8)

    leak_points = [
        ("1. Timepoint-Level Shuffling: ", "The official SuperBruceJia/EEG-DL code shuffles single 6.25ms timepoints. Adjacent points from the same trial fall into both train and test splits, allowing trivial 1D spline interpolation."),
        ("2. Test-Mode BatchNorm in Train Mode: ", "The test evaluation script calls model.train(), calculating mini-batch mean and variance directly over test data. This leaks test label distributions directly into output logits."),
        ("3. Transductive Pearson Adjacency: ", "Adjacency matrix is computed using Pearson correlation across all trials simultaneously before partitioning, leaking future trial relationships into the graph Laplacian."),
        ("Literature Reality Gap: ", "When evaluated on continuous, leak-free trials, published models cannot generalize, whereas extensive benchmarks prove the true clean ceiling across 216k pipelines is 65.00% (Vasques et al., 2026).")
    ]
    for title, desc in leak_points:
        p = tf_r.add_paragraph()
        p.space_after = Pt(6)
        r1 = p.add_run()
        r1.text = "• " + title
        r1.font.bold = True
        r1.font.size = Pt(10)
        r1.font.color.rgb = CRIMSON
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = CHARCOAL

    set_notes(slide, """
Slide 10 highlights one of the most vital contributions of our literature review: Our Methodological Data Leakage Audit.
In Figure 4 on the left, we deconstruct the three compounding leakage pathways identified in the reference codebase of GCNs-Net:
1. Inter-Trial Timepoint Shuffling: Randomly permuting 6.25ms timepoints destroys epoch boundaries, allowing models to interpolate test labels via continuous voltage autocorrelation.
2. Test-Time BatchNorm in Train Mode: Batch normalization recalibrates mean and variance over test mini-batches, leaking test-set class distributions.
3. Transductive Graph Estimation: Computing the Pearson correlation graph on pooled train and test sets pre-adapts spectral filters to test geometry.
As summarized in the lower card, published claims of >90% are artifacts of this leakage. Independent benchmarks (Vasques et al., 2026 across 216,714 pipelines) prove that the true literature ceiling is 65.0%. Our proposed GBP-Net is formulated specifically to eliminate these three leakage vectors while using cross-subject pre-training to break the 65% ceiling legitimately!
""")


# =============================================================================
# Slide 11: Experimental Framework & Implementation Setup
# =============================================================================
def build_slide_11(prs, blank_layout):
    slide = prs.slides.add_slide(blank_layout)
    add_header(slide, "Experimental Framework & Implementation Setup", "Software Stack, Benchmark Datasets, and Rigorous Evaluation Protocols", 11)

    card_w = Inches(3.8)
    card_h = Inches(5.45)
    card_y = Inches(1.4)

    # Card 1: Software & Hardware Stack
    c1 = add_card(slide, Inches(0.6), card_y, card_w, card_h, bg_color=CARD_BG, border_color=BORDER)
    s1 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), card_y, card_w, Inches(0.08))
    s1.fill.solid()
    s1.fill.fore_color.rgb = NAVY
    s1.line.fill.background()

    tf1 = slide.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(3.4), Inches(5.0)).text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "1. Software & Hardware Stack"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.space_after = Pt(10)

    stack = [
        ("Deep Learning: ", "PyTorch 2.4 + PyTorch Geometric (PyG 2.5) for GCN ChebConv and graph batching."),
        ("EEG Processing: ", "MNE-Python (v1.7) for EDF reading, montage channel coordinates, and notch filtering."),
        ("Signal Processing: ", "SciPy Signal (Butterworth/FIR bandpass filters) and NumPy array operations."),
        ("Compute Platform: ", "NVIDIA GeForce RTX 5060 GPU (8 GB VRAM), CUDA 12.4, Python 3.14 on Windows 11."),
        ("Planned Reproducibility: ", "Deterministic seed control, modular pipeline architecture, and Git version control for PW-2.")
    ]
    for title, desc in stack:
        p = tf1.add_paragraph()
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + title
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(10)
        r2.font.color.rgb = CHARCOAL

    # Card 2: Datasets & Preprocessing
    c2 = add_card(slide, Inches(4.75), card_y, card_w, card_h, bg_color=CARD_BG, border_color=BORDER)
    s2 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(4.75), card_y, card_w, Inches(0.08))
    s2.fill.solid()
    s2.fill.fore_color.rgb = SLATE
    s2.line.fill.background()

    tf2 = slide.shapes.add_textbox(Inches(4.95), Inches(1.6), Inches(3.4), Inches(5.0)).text_frame
    tf2.word_wrap = True

    p = tf2.paragraphs[0]
    p.text = "2. Datasets & Preprocessing"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = SLATE
    p.space_after = Pt(10)

    data_pts = [
        ("Primary: PhysioNet EEGMMIDB: ", "109 subjects, 64 EEG channels, 160 Hz sampling rate, 14 experimental runs."),
        ("4 Motor Imagery Classes: ", "Left Fist, Right Fist, Both Fists, Both Feet (Tasks 2 & 4, ~42 trials/class)."),
        ("Secondary: BCI IV-2a: ", "9 subjects, 22 EEG channels, 250 Hz sampling rate; cross-dataset generalizability benchmark."),
        ("Signal Preprocessing: ", "50 Hz Notch filter (powerline noise), 4–40 Hz bandpass filtering, 0.5s pre-stimulus baseline correction, trial windowing (0.5s to 2.5s post-cue).")
    ]
    for title, desc in data_pts:
        p = tf2.add_paragraph()
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + title
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = SLATE
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(10)
        r2.font.color.rgb = CHARCOAL

    # Card 3: Evaluation Protocol & Metrics
    c3 = add_card(slide, Inches(8.9), card_y, card_w, card_h, bg_color=CARD_BG, border_color=BORDER)
    s3 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8.9), card_y, card_w, Inches(0.08))
    s3.fill.solid()
    s3.fill.fore_color.rgb = TEAL
    s3.line.fill.background()

    tf3 = slide.shapes.add_textbox(Inches(9.1), Inches(1.6), Inches(3.4), Inches(5.0)).text_frame
    tf3.word_wrap = True

    p = tf3.paragraphs[0]
    p.text = "3. Evaluation Protocols & Metrics"
    p.font.name = FONT_HEADING
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TEAL
    p.space_after = Pt(10)

    eval_pts = [
        ("Strict Trial Partitioning: ", "Continuous trial-level splits guaranteeing zero overlapping timepoints between train and test sets."),
        ("Within-Subject Evaluation: ", "5-Fold Stratified Trial-Level Cross-Validation with test-time BatchNorm in eval() mode."),
        ("Cross-Subject LOSO Protocol: ", "Leave-One-Subject-Out (LOSO) cross-validation (pre-train on 92 subjects, test on held-out subject)."),
        ("Quantitative Metrics: ", "Classification Accuracy (%), Cohen's Kappa coefficient (κ), and Macro-averaged F1-Score."),
        ("Statistical Significance: ", "Wilcoxon signed-rank tests (p < 0.05) to validate graph ablation improvements.")
    ]
    for title, desc in eval_pts:
        p = tf3.add_paragraph()
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + title
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = TEAL
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(10)
        r2.font.color.rgb = CHARCOAL

    set_notes(slide, """
Slide 11 describes our experimental setup:
- Tools: PyTorch, PyTorch Geometric, and MNE-Python on an NVIDIA RTX 5060 GPU with 8GB VRAM.
- Datasets: The full 109-subject PhysioNet EEGMMIDB with 64 channels, alongside BCI Competition IV-2a for cross-dataset verification.
- Protocols: Strict continuous trial-level splitting with zero timepoint leakage, evaluated using 5-Fold Stratified CV, Leave-One-Subject-Out CV, Cohen's Kappa, and Wilcoxon signed-rank tests for statistical significance.
""")


# =============================================================================
# Slide 12: Expected Outcomes & Viva Deliverables
# =============================================================================
def build_slide_12(prs, blank_layout):
    slide = prs.slides.add_slide(blank_layout)
    add_header(slide, "Expected Outcomes & Project Deliverables", "Evaluation Criteria Fulfillment · Scientific & Technical Deliverables", 12)

    card_w = Inches(5.89)
    card_h = Inches(2.55)

    coords = [
        (Inches(0.6), Inches(1.45)),   # Card 1 (Top Left)
        (Inches(6.84), Inches(1.45)),  # Card 2 (Top Right)
        (Inches(0.6), Inches(4.25)),   # Card 3 (Bottom Left)
        (Inches(6.84), Inches(4.25))   # Card 4 (Bottom Right)
    ]

    deliverables = [
        (
            "1. Anticipated Decoding Breakthrough",
            NAVY,
            [
                ("Resolving the Classical Deficit: ", "Proposed spatial-then-power sequence is hypothesized to surpass classical CSP and compact CNN baselines without data leakage."),
                ("Breaking the Literature Ceiling: ", "Targeting statistically significant gains over the 65.0% Vasques benchmark via 92-subject cross-subject pre-training."),
                ("Relieving BCI-Illiteracy: ", "Hypothesized to disproportionately uplift low-performing subjects and reduce inter-subject decoding variance.")
            ]
        ),
        (
            "2. Scientific Resolution of the Graph Hypothesis",
            SLATE,
            [
                ("Definitive Empirical Answer: ", "Deliver the first controlled ablation comparing spectral ChebConv against unconstrained spatial linear projections on identical folds."),
                ("Inductive Bias Isolation: ", "Conclusively prove whether graph topology provides genuine inductive bias or acts as a spatial regularizer when leakage is absent."),
                ("Resolving Literature Ambiguity: ", "End the unverified claims prevalent in recent GCN motor imagery literature.")
            ]
        ),
        (
            "3. Neurophysiological Interpretability",
            TEAL,
            [
                ("Topographical Scalp Heatmaps: ", "Generate 2D scalp topographic maps showing learned dynamic adjacency and attention weights across 64 electrodes."),
                ("Sensorimotor Cortex Conformity: ", "Prove that attention weights peak over C3/C4 for hand imagery and Cz for foot imagery, matching established ERD/ERS neurophysiology."),
                ("Clinical Trustworthiness: ", "Provide clinicians with transparent, interpretable feature maps rather than black-box decisions.")
            ]
        ),
        (
            "4. Project Work Phase-Wise Roadmap",
            GOLD,
            [
                ("Phase-1 (PW-1 Milestone): ", "Comprehensive 36-paper literature review, PRISMA synthesis, problem formulation, title defense, and GBP-Net architectural formulation."),
                ("Phase-2 (PW-2 Execution): ", "Full modular PyTorch/PyG implementation, cross-subject pre-training on 92 subjects, controlled ablation benchmarking, and confirmatory statistical tests."),
                ("Final Deliverables: ", "Comprehensive PW-2 Major Project dissertation, open-source benchmark repository, and peer-reviewed conference/journal submission.")
            ]
        )
    ]

    for (left, top), (title, color, bullets) in zip(coords, deliverables):
        c = add_card(slide, left, top, card_w, card_h, bg_color=CARD_BG, border_color=BORDER)
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, card_w, Inches(0.06))
        bar.fill.solid()
        bar.fill.fore_color.rgb = color
        bar.line.fill.background()

        tf = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), card_w - Inches(0.4), card_h - Inches(0.3)).text_frame
        tf.word_wrap = True

        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.name = FONT_HEADING
        p0.font.size = Pt(13)
        p0.font.bold = True
        p0.font.color.rgb = color
        p0.space_after = Pt(6)

        for b_title, b_desc in bullets:
            p = tf.add_paragraph()
            p.space_after = Pt(4)
            r1 = p.add_run()
            r1.text = "• " + b_title
            r1.font.bold = True
            r1.font.size = Pt(10)
            r1.font.color.rgb = CHARCOAL
            r2 = p.add_run()
            r2.text = b_desc
            r2.font.size = Pt(9.5)
            r2.font.color.rgb = CHARCOAL

    set_notes(slide, """
In conclusion, Slide 12 outlines our expected outcomes and project work execution roadmap:
1. Anticipated Decoding Breakthrough: Formulated to resolve the classical deficit without data leakage, break past the 65.0% literature ceiling, and alleviate BCI illiteracy.
2. Controlled Ablation Suite: Planned for PW-2 to definitively isolate whether spectral graph convolutions provide genuine inductive bias compared to unconstrained spatial filters.
3. Neurophysiological Interpretability: Topographical scalp maps confirming activation over C3, Cz, and C4 sensorimotor regions.
4. Clear Phase-Wise Roadmap: Phase-1 (PW-1) successfully establishes the problem statement, literature taxonomy, research gaps, title defense, and architectural design; Phase-2 (PW-2) will execute the full implementation, training, and empirical validation.
Thank you, respected panel members. We are now open for your questions and suggestions.
""")


def main():
    print("=" * 70)
    print("Building PW-1 Review-1 Presentation - NIP (.pptx)")
    print("Neil Gogte Institute of Technology · CSE Batch 22")
    print("=" * 70)

    prs, blank_layout = create_presentation()

    print("Building Slide 1: Title Slide...")
    build_slide_1(prs, blank_layout)

    print("Building Slide 2: Project Abstract (Mandatory Circular Slide 2)...")
    build_slide_2(prs, blank_layout)

    print("Building Slide 3: Problem Statement & BCI Domain Challenges...")
    build_slide_3(prs, blank_layout)

    print("Building Slide 4: Project Objectives & Scope of Work...")
    build_slide_4(prs, blank_layout)

    print("Building Slide 5: Literature Survey Table (Part 1: Spatial Filters & Compact CNNs)...")
    build_slide_5(prs, blank_layout)

    print("Building Slide 6: Literature Survey Table (Part 2: GCNs & Evaluation Baselines)...")
    build_slide_6(prs, blank_layout)

    print("Building Slide 7: Title Justification & Innovative Aspects (Evaluation Criterion 1)...")
    build_slide_7(prs, blank_layout)

    print("Building Slide 8: Research Gaps Identified from Literature (Evaluation Criterion 3)...")
    build_slide_8(prs, blank_layout)

    print("Building Slide 9: Proposed Architecture: GBP-Net (Embedding Figure 5)...")
    build_slide_9(prs, blank_layout)

    print("Building Slide 10: Empirical Reality Gap & Leakage Audit (Embedding Figure 4)...")
    build_slide_10(prs, blank_layout)

    print("Building Slide 11: Experimental Framework & Implementation Setup...")
    build_slide_11(prs, blank_layout)

    print("Building Slide 12: Expected Outcomes & Project Deliverables...")
    build_slide_12(prs, blank_layout)

    output_path = "docs/PW-1_Review_Presentation-NIP.pptx"
    prs.save(output_path)
    file_size = os.path.getsize(output_path)
    print("=" * 70)
    print(f"Successfully generated: {output_path}")
    print(f"Total Slides: {len(prs.slides)} | File Size: {file_size:,} bytes")
    print("=" * 70)


if __name__ == "__main__":
    main()
