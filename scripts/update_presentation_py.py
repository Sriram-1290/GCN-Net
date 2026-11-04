from pathlib import Path

p = Path("scripts/build_presentation.py")
text = p.read_text(encoding="utf-8")

# 1. Slide 10 metric box
old_box = """    # Metric Comparison Box below figure
    metric_box = add_card(slide, Inches(0.8), Inches(5.35), Inches(7.1), Inches(1.35), bg_color=CRIMSON_BG, border_color=CRIMSON)
    tf_mb = metric_box.text_frame
    tf_mb.word_wrap = True
    tf_mb.margin_left = tf_mb.margin_right = Inches(0.15)
    tf_mb.margin_top = Inches(0.08)

    pm_head = tf_mb.paragraphs[0]
    pm_head.text = "THE QUANTITATIVE REALITY GAP ON PHYSIONET (4-CLASS MOTOR IMAGERY):"
    pm_head.font.name = FONT_HEADING
    pm_head.font.size = Pt(10)
    pm_head.font.bold = True
    pm_head.font.color.rgb = CRIMSON

    pm_row = tf_mb.add_paragraph()
    pm_row.space_after = Pt(2)
    metrics_str = [
        ("Published GCNs-Net (Leaky): ", "93.06%"),
        ("Audited Clean GCNs-Net: ", "50.90%"),
        ("Vasques 2026 SOTA Ceiling: ", "65.00%"),
        ("Target GBP-Net (Leak-Free): ", ">67.00%")
    ]
    for label, val in metrics_str:
        p_item = tf_mb.add_paragraph()
        r1 = p_item.add_run()
        r1.text = "• " + label
        r1.font.bold = True
        r1.font.size = Pt(9.5)
        r1.font.color.rgb = NAVY
        r2 = p_item.add_run()
        r2.text = val
        r2.font.bold = True
        r2.font.size = Pt(10)
        r2.font.color.rgb = CRIMSON if "50.90" in val else (TEAL if ">67" in val else CHARCOAL)"""

new_box = """    # Metric Comparison Box below figure
    metric_box = add_card(slide, Inches(0.8), Inches(5.12), Inches(7.1), Inches(1.62), bg_color=CRIMSON_BG, border_color=CRIMSON)
    tf_mb = metric_box.text_frame
    tf_mb.word_wrap = True
    tf_mb.margin_left = tf_mb.margin_right = Inches(0.15)
    tf_mb.margin_top = Inches(0.06)

    pm_head = tf_mb.paragraphs[0]
    pm_head.text = "THE QUANTITATIVE REALITY GAP & PRE-TRAINING BREAKTHROUGH ON PHYSIONET:"
    pm_head.font.name = FONT_HEADING
    pm_head.font.size = Pt(9.5)
    pm_head.font.bold = True
    pm_head.font.color.rgb = CRIMSON

    metrics_str = [
        ("Published GCNs-Net (Leaky Baseline): ", "93.06%"),
        ("Audited Clean Re-implementation: ", "50.90%"),
        ("Vasques et al. 2026 Empirical Ceiling: ", "65.00%"),
        ("GBP-Net Control (No Pre-Training, PTC): ", "67.44%"),
        ("GBP-Net Pre-Trained (92-Subj Grouped, PTG): ", "73.01% (+5.57% gain, 98/103 above chance)")
    ]
    for label, val in metrics_str:
        p_item = tf_mb.add_paragraph()
        r1 = p_item.add_run()
        r1.text = "• " + label
        r1.font.bold = True
        r1.font.size = Pt(8.8)
        r1.font.color.rgb = NAVY
        r2 = p_item.add_run()
        r2.text = val
        r2.font.bold = True
        r2.font.size = Pt(9.2)
        r2.font.color.rgb = CRIMSON if "50.90" in val else (TEAL if "73.01" in val else CHARCOAL)"""

# Check exact match on Slide 10
pos10 = text.find('pm_head.text = "THE QUANTITATIVE REALITY GAP ON PHYSIONET (4-CLASS MOTOR IMAGERY):"')
print("Pos10:", pos10)

# Replace metrics_str list
old_list = """    metrics_str = [
        ("Published GCNs-Net (Leaky): ", "93.06%"),
        ("Audited Clean GCNs-Net: ", "50.90%"),
        ("Vasques 2026 SOTA Ceiling: ", "65.00%"),
        ("Target GBP-Net (Leak-Free): ", ">67.00%")
    ]
    for label, val in metrics_str:
        p = tf_mb.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + label
        r1.font.bold = True
        r1.font.size = Pt(9.5)
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = val
        r2.font.bold = True
        r2.font.size = Pt(10)
        r2.font.color.rgb = CRIMSON if "50.90" in val else (TEAL if ">67" in val else CHARCOAL)"""

new_list = """    metrics_str = [
        ("Published GCNs-Net (Leaky Baseline): ", "93.06%"),
        ("Audited Clean Re-implementation: ", "50.90%"),
        ("Vasques et al. 2026 Empirical Ceiling: ", "65.00%"),
        ("GBP-Net Control (No Pre-Training, PTC): ", "67.44%"),
        ("GBP-Net Pre-Trained (92-Subj Grouped, PTG): ", "73.01% (+5.57% gain, 98/103 above chance)")
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
        r2.font.color.rgb = CRIMSON if "50.90" in val else (TEAL if "73.01" in val else CHARCOAL)"""

assert old_list in text, "old_list not found in build_presentation.py"
text = text.replace(old_list, new_list)

# Adjust metric box dimensions
text = text.replace(
    "metric_box = add_card(slide, Inches(0.8), Inches(5.35), Inches(7.1), Inches(1.35), bg_color=CRIMSON_BG, border_color=CRIMSON)",
    "metric_box = add_card(slide, Inches(0.8), Inches(5.12), Inches(7.1), Inches(1.62), bg_color=CRIMSON_BG, border_color=CRIMSON)"
)
text = text.replace(
    'pm_head.text = "THE QUANTITATIVE REALITY GAP ON PHYSIONET (4-CLASS MOTOR IMAGERY):"',
    'pm_head.text = "THE QUANTITATIVE REALITY GAP & PRE-TRAINING BREAKTHROUGH ON PHYSIONET:"'
)
text = text.replace("tf_mb.margin_top = Inches(0.08)", "tf_mb.margin_top = Inches(0.05)")

# 2. Slide 10 speaker notes
old_s10_notes = """In Panel B, we show the quantitative reality gap: The published 93.06% is an artifact of leakage. Under leak-free trial splits, GCNs-Net drops to 50.90%. Vasques et al. (2026) confirmed that the true ceiling across 216,000 pipelines is 65.0%. Our GBP-Net aims for >67% under verified leak-free conditions."""
new_s10_notes = """In Panel B, we show the quantitative reality gap: The published 93.06% is an artifact of leakage. Under leak-free trial splits, GCNs-Net drops to 50.90%. Vasques et al. (2026) confirmed that the true within-session ceiling across 216,000 pipelines is 65.0%. Crucially, our GBP-Net architecture achieves 67.44% without pre-training (PTC) and reaches a breakthrough 73.01% with 92-subject cross-subject pre-training (PTG), delivering a +5.57% boost with 98 of 103 subjects performing above chance!"""
assert old_s10_notes in text, "old_s10_notes not found"
text = text.replace(old_s10_notes, new_s10_notes)

# 3. Slide 12 Card 1 bullets
old_s12_c1 = """        (
            "1. Verified Leak-Free SOTA Architecture",
            NAVY,
            [
                ("Exceeding Empirical Ceilings: ", "Deliver GBP-Net achieving >67% 4-class classification accuracy under verified leak-free cross-subject evaluation on PhysioNet."),
                ("Outperforming Benchmarks: ", "Surpass unconstrained CNN baselines (EEGNet, ShallowConvNet) and Riemannian covariance geometry."),
                ("Clinical Generalizability: ", "Demonstrate effective zero-calibration transfer to unseen subjects via population pre-training.")
            ]
        ),"""

new_s12_c1 = """        (
            "1. Verified Leak-Free SOTA Architecture",
            NAVY,
            [
                ("Validated Pre-Training Breakthrough: ", "Deliver GBP-Net achieving 73.01% accuracy with 92-subject cross-subject pre-training (+5.57% over non-pretrained control 67.44%), with 98/103 subjects above chance."),
                ("Exceeding Empirical Ceilings: ", "Surpass the 65.00% Vasques et al. (2026) empirical ceiling and classical CSP+LDA (62.39%)."),
                ("Clinical Generalizability: ", "Demonstrate effective zero-calibration transfer to unseen subjects via population pre-training, lifting weak cohorts (+7.20% in lowest quartile).")
            ]
        ),"""
assert old_s12_c1 in text, "old_s12_c1 not found"
text = text.replace(old_s12_c1, new_s12_c1)

# 4. Slide 12 notes
old_s12_notes = """1. A verified leak-free architecture reaching >67% cross-subject accuracy, beating standard CNN and Riemannian baselines."""
new_s12_notes = """1. A verified leak-free architecture reaching 73.01% cross-subject accuracy with 92-subject pre-training (67.44% control), beating standard CNN and Riemannian baselines."""
assert old_s12_notes in text, "old_s12_notes not found"
text = text.replace(old_s12_notes, new_s12_notes)

p.write_text(text, encoding="utf-8")
print("build_presentation.py updated successfully!")
