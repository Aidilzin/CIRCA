# -*- coding: utf-8 -*-
"""
CIRCA Pitch Deck Generator (Stand-alone Script)
Generates: docs/circa_deck.html, docs/circa_deck.pdf, docs/circa_slides_png/slide_XX.png,
and updates backup slide HTML files in docs/circa_slides/ (slide_01.html .. slide_13.html)
Renders via Edge headless Chromium with 2x scale factor for crystal clear 1440p/4K resolution.
Aligned 1:1 with Thesis Chapters 1-5 structure and defense framing.
"""
import os
import sys
import re
import base64
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
SLIDES_DIR = DOCS / "circa_slides"
OUT_PNG_DIR = DOCS / "circa_slides_png"

def get_base64_image(filename):
    possible_paths = [
        ROOT / filename,
        ASSETS / filename,
        DOCS / filename,
        DOCS / "assets" / filename
    ]
    for p in possible_paths:
        if p.exists() and p.is_file():
            suffix = p.suffix.lower().replace(".", "")
            if suffix == "jpg":
                suffix = "jpeg"
            mime = f"image/{suffix}"
            try:
                with open(p, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                return f"data:{mime};base64,{encoded}"
            except Exception as e:
                print(f"[WARN] Failed to encode {p}: {e}")
                return ""
    print(f"[WARN] Image file not found: {filename}")
    return ""

COMMON_HEAD = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>CIRCA Pitch Deck</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&family=Open+Sans:wght@400;600;700&family=Roboto+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{width:1920px;height:1080px;overflow:hidden;font-family:'Open Sans',sans-serif;color:#1C2820;background:#FCFCFA}
.slide{width:1920px;height:1080px;position:relative;display:flex;flex-direction:column;page-break-after:always;page-break-inside:avoid}
.bg-dark{background:linear-gradient(135deg,#07140B 0%,#11291A 70%,#183B25 100%);color:#FFFFFF}
.bg-light{background:#FCFCFA;color:#1C2820}
.hdr{background:linear-gradient(135deg,#07140B 0%,#11291A 100%);border-bottom:3px solid #B8860B;padding:0 80px;display:flex;align-items:center;justify-content:space-between;height:100px;flex-shrink:0;width:100%}
.hdr-title{font-family:'Montserrat',sans-serif;font-weight:900;font-size:28px;color:#7FD99A;letter-spacing:2px}
.hdr-logos{display:flex;gap:20px;align-items:center}
.hdr-logos img{height:54px;object-fit:contain;filter:brightness(1.1)}
.slide-num{font-family:'Roboto Mono',monospace;color:#E5C158;font-size:22px;font-weight:700}
.title-block{flex:1;display:flex;flex-direction:column;justify-content:center;padding:0 80px}
.slide-cat{font-family:'Roboto Mono',monospace;font-size:22px;letter-spacing:3px;text-transform:uppercase;margin-bottom:16px}
.slide-title{font-family:'Montserrat',sans-serif;font-weight:900;line-height:1.1;margin-bottom:12px}
.slide-sub{font-size:26px;font-weight:400;line-height:1.5}
.content-area{flex:1;padding:0 80px 30px;display:flex;gap:40px;align-items:flex-start;margin-top:10px}
.col2{flex:1;display:flex;flex-direction:column;justify-content:flex-start;gap:14px}
.sec-hdr{background:linear-gradient(90deg,#11291A,#1E432C);color:#fff;font-family:'Montserrat',sans-serif;font-weight:800;font-size:24px;letter-spacing:2px;text-transform:uppercase;padding:12px 22px;border-left:6px solid #B8860B;margin-bottom:0;border-radius:4px;height:52px;display:flex;align-items:center}
.card{background:#FFFFFF;border:1px solid #D8E5DD;border-radius:10px;padding:22px;box-shadow:0 4px 15px rgba(26,58,42,0.04)}
.card h3{font-family:'Montserrat',sans-serif;font-weight:800;font-size:24px;color:#11291A;margin-bottom:10px}
.card p{font-size:18px;line-height:1.5;color:#223328}
.card ul{padding-left:24px}
.card ul li{font-size:18px;line-height:1.5;margin-bottom:8px;color:#223328}
.footer-bar{background:linear-gradient(135deg,#07140B 0%,#11291A 100%);border-top:3px solid #B8860B;padding:0 80px;display:flex;justify-content:space-between;align-items:center;height:60px;flex-shrink:0;width:100%}
.footer-text{font-family:'Roboto Mono',monospace;font-size:18px;font-weight:700;color:#E5C158;letter-spacing:1px}
@media print {
  @page { size: 1920px 1080px; margin: 0; }
  html, body { width: 1920px; height: 1080px; margin: 0; padding: 0; background: transparent; }
  .slide { width: 1920px; height: 1080px; }
}
</style></head><body>
"""

def generate_slides():
    print("[DECK] Encoding image assets...")
    logo_uitm = get_base64_image("Logo UiTM (bg dark).png") or get_base64_image("LogoUiTM.png")
    logo_fskm = get_base64_image("LOGO_FSKM-white_font.png")
    app_screenshot = get_base64_image("app_screenshot_full.png")
    fig4_1_defects = get_base64_image("fig4_1_sample_defects.png")
    fig4_3_ablation = get_base64_image("fig4_3_ablation_comparison.png")
    fig4_4a_nano = get_base64_image("fig4_4a_nano_curves.png")
    fig4_6_confusion = get_base64_image("fig4_6_confusion_matrix.png")
    fig4_7a_pr = get_base64_image("fig4_7a_pr_curve.png")
    fig4_7b_f1 = get_base64_image("fig4_7b_f1_curve.png")
    fig4_8_failure = get_base64_image("fig4_8_failure_gallery_batch0.jpg")
    fig4_9_threshold = get_base64_image("fig4_9_threshold_sweep.png")
    tune_fitness = get_base64_image("tune_fitness.png")

    total_slides = 13

    slides_data = [
        # Slide 1: Title & Executive Overview (Chapter 1)
        r"""<div class="slide bg-dark">
<div class="hdr">
<div class="hdr-logos"><img src="{LOGO_UITM}" alt="UiTM"><img src="{LOGO_FSKM}" alt="FSKM"></div>
<div class="slide-num">01 / 13</div>
</div>
<div class="title-block">
<div class="slide-cat" style="color:#E5C158">FINAL YEAR PROJECT PRESENTATION &nbsp;|&nbsp; JULY 2026 &nbsp;|&nbsp; CHAPTER 1</div>
<div class="slide-title" style="font-size:64px;color:#FFFFFF">CIRCA: Circuit Inspection and Recognition<br><span style="color:#7FD99A">using Convolutional Architectures</span></div>
<div class="slide-sub" style="color:#B8D8C0;margin-top:20px;max-width:1450px">A Low-Cost, CPU-Quantized Edge AI System for Industrial &amp; Repair-Shop Quality Assurance &mdash; Delivering On-Demand Sub-Millimeter Defect Identification Without Specialized GPU Hardware.</div>

<div style="margin-top:35px;display:flex;gap:30px;align-items:center">
<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(127,217,154,0.3);border-radius:10px;padding:16px 26px">
<div style="font-size:15px;color:#9EC9AB;font-family:'Roboto Mono',monospace;font-weight:700">STUDENT</div>
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:20px;color:#FFFFFF;margin-top:3px">Muhammad Aidil Al-Faizi Bin Mohd Zin</div>
</div>
<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(127,217,154,0.3);border-radius:10px;padding:16px 26px">
<div style="font-size:15px;color:#9EC9AB;font-family:'Roboto Mono',monospace;font-weight:700">SUPERVISOR</div>
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:20px;color:#FFFFFF;margin-top:3px">Pn. Farahnatasyah Binti Abdul Hannan</div>
</div>
<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(127,217,154,0.3);border-radius:10px;padding:16px 26px">
<div style="font-size:15px;color:#9EC9AB;font-family:'Roboto Mono',monospace;font-weight:700">PROGRAMME</div>
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:20px;color:#FFFFFF;margin-top:3px">Bachelor of Information Systems (Hons.) Intelligent Systems Engineering &nbsp;|&nbsp; CS259 Group 6C</div>
</div>
</div>

<div style="margin-top:20px;background:rgba(17,41,26,0.85);border:1px solid rgba(184,134,11,0.4);border-radius:8px;padding:14px 22px;display:flex;align-items:center;gap:18px;max-width:1450px">
<div style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:16px;color:#E5C158;text-transform:uppercase;letter-spacing:1px;white-space:nowrap">&#9888; Operational Mode &amp; Scope Exclusions</div>
<div style="font-size:17px;color:#B8D8C0;line-height:1.5">Operates in an <strong>On-Demand Sub-Second Static Inspection Mode (0.392s/tile)</strong>. Excludes electrical testing, X-ray internal diagnostics, and 5 sparse/ambiguous classes (<code style="color:#7FD99A">spur</code>, <code style="color:#7FD99A">spurious_copper</code>, <code style="color:#7FD99A">scratch</code>, <code style="color:#7FD99A">pinhole</code>, <code style="color:#7FD99A">solder_bridge</code>) with &lt; 400 instances to prevent model overfitting (Thesis &sect;1.5).</div>
</div>
</div>
<div class="footer-bar">
<span class="footer-text">Project Overview &nbsp;|&nbsp; CIRCA FYP 2026</span>
<span class="footer-text">Group 6C &nbsp;|&nbsp; UiTM FSKM</span>
</div>
</div>""",

        # Slide 2: Research Background & Problem Statement (Chapter 1 §1.1, §1.2)
        r"""<div class="slide bg-light">
<div class="hdr">
<div class="hdr-logos"><img src="{LOGO_UITM}" alt="UiTM"><img src="{LOGO_FSKM}" alt="FSKM"></div>
<div class="hdr-title">CIRCA</div>
<div class="slide-num">02 / 13</div>
</div>
<div class="content-area" style="margin-top:20px">
<div class="col2">
<div class="sec-hdr">The Problem (Thesis &sect;1.2)</div>
<div class="card">
<h3>&#128269; Manual PCB Inspection Limitations</h3>
<ul>
<li><strong>Inspection Escape Rates:</strong> 10&ndash;20% escape rates on sub-millimeter components under manual lenses (Law et al., 2024).</li>
<li><strong>Workbench Glare &amp; Shadows:</strong> Specular reflections and heavy shadowing under repair desklamps.</li>
<li><strong>Shift Fatigue &amp; Time Loss:</strong> Visual acuity degrades rapidly; manual checks take 10&ndash;30 mins (Goti, 2025).</li>
<li><strong>Operator Experience Bias:</strong> Inexperienced technicians miss cold solder joints and subtle shorts.</li>
</ul>
</div>
<div class="card">
<h3>&#128185; Economic Barrier</h3>
<p>Industrial Automated Optical Inspection (AOI) platforms cost <strong>RM 50,000&ndash;500,000</strong> &mdash; completely inaccessible for independent electronics repair garages in Southeast Asia.</p>
</div>
<div class="card">
<h3>&#9201;&#65039; Diagnostic Time Bottlenecks</h3>
<p>Manual visual inspection under stereo microscopes requires <strong>3 to 5 minutes per PCB board</strong> (Goti, 2025), creating severe repair turnaround bottlenecks during high-volume diagnostic shifts.</p>
</div>
</div>
<div class="col2">
<div class="sec-hdr">The Opportunity &amp; Value Proposition</div>
<div class="card" style="background:linear-gradient(135deg,#F4FAF6,#E8F5ED);border-color:#B0D9C0;padding:22px">
<h3 style="color:#11291A;font-size:24px;margin-bottom:10px">CIRCA&rsquo;s Value Proposition</h3>
<p style="font-size:18px;margin-bottom:16px;line-height:1.5">A <strong>lightweight, low-cost</strong> PCB inspection system running on standard CPU hardware &mdash; zero external GPU accelerators or specialized industrial camera upgrades required.</p>

<div style="background:#11291A;border-radius:8px;padding:14px 20px;text-align:center;box-shadow:0 8px 24px rgba(17,41,26,0.12);margin-bottom:16px">
<div style="font-family:'Montserrat',sans-serif;font-weight:900;font-size:44px;color:#7FD99A;line-height:1">RM 0</div>
<div style="font-family:'Roboto Mono',monospace;font-size:15px;color:#9EC9AB;letter-spacing:1px;margin-top:4px;font-weight:700">HARDWARE UPGRADE COST</div>
</div>

<div style="border: 1px solid #D8E5DD; border-radius: 8px; padding: 14px; background: white; box-shadow: 0 4px 10px rgba(0,0,0,0.02)">
  <div style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:15px;color:#11291A;margin-bottom:10px;text-transform:uppercase;letter-spacing:1px">Inspection System Cost Comparison</div>
  <div style="margin-bottom: 10px;">
    <div style="display:flex;justify-content:space-between;font-size:16px;margin-bottom:4px;color:#5E7063">
      <strong>Industrial AOI Cost</strong>
      <span style="font-family:'Roboto Mono',monospace;font-weight:700;color:#A92222;margin-left:auto">RM 50K - 500K</span>
    </div>
    <div style="height: 14px; background: #EFF3F0; border-radius: 4px; overflow: hidden; border: 1px solid #E2E8E4">
      <div style="width: 100%; height: 100%; background: linear-gradient(90deg, #A92222, #D32F2F)"></div>
    </div>
  </div>
  <div>
    <div style="display:flex;justify-content:space-between;font-size:16px;margin-bottom:4px;color:#5E7063">
      <strong>CIRCA System Cost</strong>
      <span style="font-family:'Roboto Mono',monospace;font-weight:700;color:#1F7A3E;margin-left:auto">RM 0 (Free Open-Source)</span>
    </div>
    <div style="height: 14px; background: #EFF3F0; border-radius: 4px; overflow: hidden; border: 1px solid #E2E8E4; position: relative">
      <div style="width: 1%; min-width: 6px; height: 100%; background: #1F7A3E"></div>
    </div>
  </div>
</div>

<div class="card" style="margin-top:12px;background:#FFFFFF;border-color:#B0D9C0;padding:14px 18px">
<h4 style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:18px;color:#11291A;margin-bottom:6px">&#9889; CAD-Free Repair Shop Flexibility</h4>
<p style="font-size:17px;color:#4F6255;line-height:1.5">Traditional AOI platforms require programming specific PCB CAD design files, making them ill-suited for independent repair garages facing unpredictable daily board varieties (Ruengrote et al., 2024). CIRCA operates in a zero-CAD diagnostic mode (Thesis &sect;1.2).</p>
</div>
</div>
</div>
</div>
<div class="footer-bar">
<span class="footer-text">Problem Statement &nbsp;|&nbsp; CIRCA FYP 2026</span>
<span class="footer-text">Group 6C &nbsp;|&nbsp; UiTM FSKM</span>
</div>
</div>""",

        # Slide 3: Research Questions, Objectives & Acceptance Criteria (Chapter 1 §1.3, §1.4, §1.5)
        r"""<div class="slide bg-dark">
<div class="hdr">
<div class="hdr-logos"><img src="{LOGO_UITM}" alt="UiTM"><img src="{LOGO_FSKM}" alt="FSKM"></div>
<div class="hdr-title">CIRCA</div>
<div class="slide-num">03 / 13</div>
</div>
<div class="content-area" style="margin-top:20px">
<div class="col2">
<div class="sec-hdr" style="background:linear-gradient(90deg,#11291A,#1E432C);border-left-color:#E5C158">Research Objectives (RO1 &ndash; RO3) (Thesis &sect;1.4)</div>
<div style="display:flex;flex-direction:column;gap:14px">
<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(127,217,154,0.3);border-radius:8px;padding:16px 20px;display:flex;gap:16px;align-items:center">
<div style="background:#11291A;border:2px solid #B8860B;color:#E5C158;font-family:'Montserrat',sans-serif;font-weight:900;font-size:20px;width:42px;height:42px;border-radius:6px;display:flex;align-items:center;justify-content:center;flex-shrink:0">1</div>
<div><div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:19px;color:#7FD99A">Unified Defect Taxonomy (RO1)</div><div style="font-size:17px;color:#B8D8C0;margin-top:3px;line-height:1.4">Establish a 7-class taxonomy of common PCB defects combining bare-board (IPC-A-600) and solder assembly (IPC-A-610H) standards.</div></div>
</div>
<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(127,217,154,0.3);border-radius:8px;padding:16px 20px;display:flex;gap:16px;align-items:center">
<div style="background:#11291A;border:2px solid #B8860B;color:#E5C158;font-family:'Montserrat',sans-serif;font-weight:900;font-size:20px;width:42px;height:42px;border-radius:6px;display:flex;align-items:center;justify-content:center;flex-shrink:0">2</div>
<div><div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:19px;color:#7FD99A">YOLOv12 &amp; OpenVINO Evaluation (RO2)</div><div style="font-size:17px;color:#B8D8C0;margin-top:3px;line-height:1.4">Optimize and evaluate YOLOv12 object detectors (Nano, Small, Medium) converted to OpenVINO IR format for edge-CPU deployment.</div></div>
</div>
<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(127,217,154,0.3);border-radius:8px;padding:16px 20px;display:flex;gap:16px;align-items:center">
<div style="background:#11291A;border:2px solid #B8860B;color:#E5C158;font-family:'Montserrat',sans-serif;font-weight:900;font-size:20px;width:42px;height:42px;border-radius:6px;display:flex;align-items:center;justify-content:center;flex-shrink:0">3</div>
<div><div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:19px;color:#7FD99A">CIRCA Desktop App Development (RO3)</div><div style="font-size:17px;color:#B8D8C0;margin-top:3px;line-height:1.4">Develop the CIRCA PyQt6 desktop app with static inspection UI, active OpenCV preprocessing (CLAHE, Gamma, Blur gate), and dual-threshold overlays.</div></div>
</div>
<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(184,134,11,0.4);border-radius:8px;padding:14px 18px">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:16px;color:#E5C158;margin-bottom:3px">&#9888; Technical Scope Delimitations</div>
<div style="font-size:16px;color:#B8D8C0;line-height:1.4">Strictly bounds the 7-class taxonomy to surface copper and solder joint features. Excludes component placement errors (misalignment/missing) and non-optical inspection modalities (thermal, electrical, X-ray) per Thesis &sect;1.5.</div>
</div>
</div>
</div>
<div class="col2">
<div class="sec-hdr" style="background:linear-gradient(90deg,#11291A,#1E432C);border-left-color:#E5C158">Taxonomy &amp; Target Criteria (Thesis &sect;1.5 / &sect;3.7)</div>
<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(127,217,154,0.3);padding:18px 20px;border-radius:10px;box-shadow:0 8px 32px rgba(0,0,0,0.15)">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px;border-bottom:1px solid rgba(127,217,154,0.2);padding-bottom:10px">
    <div style="font-family:'Montserrat',sans-serif;font-weight:900;font-size:46px;color:#7FD99A;line-height:1">7</div>
    <div>
      <div style="font-family:'Roboto Mono',monospace;font-size:16px;color:#E5C158;letter-spacing:2px;font-weight:700;text-transform:uppercase">Unified Defect Taxonomy</div>
      <div style="font-size:15px;color:#9EC9AB;margin-top:2px">Categorized under international IPC PCB quality standards</div>
    </div>
  </div>
  
  <div style="display:flex;flex-direction:column;gap:10px">
    <div>
      <div style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:14px;color:#E5C158;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;border-left:3px solid #B8860B;padding-left:8px">IPC-A-600 (Bare-Board Defects)</div>
      <div style="display:flex;flex-wrap:wrap;gap:8px">
        <span style="background:rgba(239,244,252,0.12);border:1px solid rgba(185,208,238,0.4);color:#EFF4FC;font-family:'Roboto Mono',monospace;font-size:15px;padding:4px 10px;border-radius:4px;font-weight:700">missing_hole</span>
        <span style="background:rgba(239,244,252,0.12);border:1px solid rgba(185,208,238,0.4);color:#EFF4FC;font-family:'Roboto Mono',monospace;font-size:15px;padding:4px 10px;border-radius:4px;font-weight:700">mouse_bite</span>
        <span style="background:rgba(239,244,252,0.12);border:1px solid rgba(185,208,238,0.4);color:#EFF4FC;font-family:'Roboto Mono',monospace;font-size:15px;padding:4px 10px;border-radius:4px;font-weight:700">open_circuit</span>
        <span style="background:rgba(239,244,252,0.12);border:1px solid rgba(185,208,238,0.4);color:#EFF4FC;font-family:'Roboto Mono',monospace;font-size:15px;padding:4px 10px;border-radius:4px;font-weight:700">short</span>
      </div>
    </div>
    
    <div>
      <div style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:14px;color:#E5C158;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;border-left:3px solid #B8860B;padding-left:8px">IPC-A-610H (Solder Joint Defects)</div>
      <div style="display:flex;flex-wrap:wrap;gap:8px">
        <span style="background:rgba(252,244,233,0.12);border:1px solid rgba(230,197,148,0.4);color:#FCF4E9;font-family:'Roboto Mono',monospace;font-size:15px;padding:4px 10px;border-radius:4px;font-weight:700">excess_solder</span>
        <span style="background:rgba(252,244,233,0.12);border:1px solid rgba(230,197,148,0.4);color:#FCF4E9;font-family:'Roboto Mono',monospace;font-size:15px;padding:4px 10px;border-radius:4px;font-weight:700">insufficient_solder</span>
        <span style="background:rgba(252,244,233,0.12);border:1px solid rgba(230,197,148,0.4);color:#FCF4E9;font-family:'Roboto Mono',monospace;font-size:15px;padding:4px 10px;border-radius:4px;font-weight:700">cold_solder_joint</span>
      </div>
    </div>
  </div>
</div>

<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(184,134,11,0.4);padding:16px 18px;border-radius:10px;box-shadow:0 8px 32px rgba(0,0,0,0.15)">
  <div style="font-family:'Roboto Mono',monospace;font-size:14px;color:#E5C158;letter-spacing:2px;font-weight:700;text-transform:uppercase;margin-bottom:10px">Target Acceptance Criteria vs Findings</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
    <div style="background:rgba(7,20,11,0.6);border:1px solid rgba(127,217,154,0.25);padding:10px 12px;border-radius:6px">
      <div style="font-size:13px;color:#9EC9AB;font-family:'Roboto Mono',monospace;font-weight:700">PRIMARY PRECISION TARGET</div>
      <div style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:17px;color:#7FD99A;margin-top:3px">&ge; 85.00% <span style="font-size:14px;color:#E5C158;font-weight:600">(85.70% Met &#10003;)</span></div>
    </div>
    <div style="background:rgba(7,20,11,0.6);border:1px solid rgba(127,217,154,0.25);padding:10px 12px;border-radius:6px">
      <div style="font-size:13px;color:#9EC9AB;font-family:'Roboto Mono',monospace;font-weight:700">CPU LATENCY LIMIT</div>
      <div style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:17px;color:#7FD99A;margin-top:3px">&lt; 500 ms <span style="font-size:14px;color:#E5C158;font-weight:600">(0.392 s Achieved &#10003;)</span></div>
    </div>
    <div style="background:rgba(7,20,11,0.6);border:1px solid rgba(127,217,154,0.25);padding:10px 12px;border-radius:6px;grid-column:span 2">
      <div style="font-size:13px;color:#9EC9AB;font-family:'Roboto Mono',monospace;font-weight:700">INDUSTRIAL mAP TARGET VS RESEARCH BASELINE</div>
      <div style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:16px;color:#FFFFFF;margin-top:3px">&gt; 90% Factory Target <span style="font-size:14px;color:#E5C158;font-weight:600">(62.79% Multi-Domain Research Baseline)</span></div>
    </div>
  </div>
</div>
</div>
</div>
<div class="footer-bar">
<span class="footer-text">Research Objectives &nbsp;|&nbsp; CIRCA FYP 2026</span>
<span class="footer-text">Group 6C &nbsp;|&nbsp; UiTM FSKM</span>
</div>
</div>""",

        # Slide 4: Literature Review & Related Work Synthesis (Chapter 2 §2.1–§2.7)
        r"""<div class="slide bg-light">
<div class="hdr">
<div class="hdr-logos"><img src="{LOGO_UITM}" alt="UiTM"><img src="{LOGO_FSKM}" alt="FSKM"></div>
<div class="hdr-title">CIRCA</div>
<div class="slide-num">04 / 13</div>
</div>
<div class="content-area" style="margin-top:20px;gap:30px">
<div class="col2" style="flex:1.1">
<div class="sec-hdr">Related Works Comparison (Thesis Table 2.1)</div>
<div class="card" style="padding:16px 20px">
<table style="width:100%;border-collapse:collapse;font-size:15px;color:#223328">
<tr style="background:#11291A;color:#7FD99A;font-weight:700"><th style="padding:6px 8px;text-align:left">Study / Author</th><th style="padding:6px 8px;text-align:left">Architecture</th><th style="padding:6px 8px;text-align:right">Accuracy / mAP</th><th style="padding:6px 8px;text-align:left">Core Limitation</th></tr>
<tr style="border-bottom:1px solid #E8EFEA"><td style="padding:6px 8px;font-weight:700">Law et al. (2024)</td><td style="padding:6px 8px">Ensemble (EffDet+SSD+YOLOv5)</td><td style="padding:6px 8px;text-align:right;font-family:'Roboto Mono',monospace">95.0% / 80.3%</td><td style="padding:6px 8px;color:#A92222">High compute, requires server GPU</td></tr>
<tr style="background:#F4FAF6;border-bottom:1px solid #E8EFEA"><td style="padding:6px 8px;font-weight:700">Ruengrote et al. (2024)</td><td style="padding:6px 8px">YOLOv10 variants</td><td style="padding:6px 8px;text-align:right;font-family:'Roboto Mono',monospace">Real-time CPU</td><td style="padding:6px 8px;color:#B8860B">Accuracy lower than YOLOv12</td></tr>
<tr style="border-bottom:1px solid #E8EFEA"><td style="padding:6px 8px;font-weight:700">Tian et al. (2025)</td><td style="padding:6px 8px">YOLOv12 (A2 + R-ELAN)</td><td style="padding:6px 8px;text-align:right;font-family:'Roboto Mono',monospace">40.6% COCO</td><td style="padding:6px 8px;color:#A92222">Generic COCO, unadapted to PCB</td></tr>
<tr style="background:#F4FAF6;border-bottom:1px solid #E8EFEA"><td style="padding:6px 8px;font-weight:700">Lv et al. (2024)</td><td style="padding:6px 8px">Edge Inspection CNN</td><td style="padding:6px 8px;text-align:right;font-family:'Roboto Mono',monospace">88.4% Precision</td><td style="padding:6px 8px;color:#A92222">Single-domain, lacks IPC standards</td></tr>
<tr style="background:#11291A;color:#E5C158;font-weight:700"><td style="padding:6px 8px">CIRCA (This Study)</td><td style="padding:6px 8px">YOLOv12-Nano FP16 CPU</td><td style="padding:6px 8px;text-align:right;font-family:'Roboto Mono',monospace;color:#7FD99A">85.70% Precision</td><td style="padding:6px 8px;color:#7FD99A">Low-cost CPU, unified IPC taxonomy</td></tr>
</table>
</div>

<div class="card" style="padding:14px 18px;margin-top:4px">
<h4 style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:17px;color:#11291A;margin-bottom:4px">Summary of Literature Findings (Thesis &sect;2.7)</h4>
<p style="font-size:16px;color:#4F6255;line-height:1.4">Single-stage YOLO detectors balance speed and precision on resource-constrained devices. However, prior works focus strictly on factory environments with expensive GPU rigs and uniform lighting (Thesis &sect;2.7).</p>
</div>
</div>

<div class="col2" style="flex:0.9">
<div class="sec-hdr">Research Gaps Addressed by CIRCA</div>
<div style="display:flex;flex-direction:column;gap:12px">
<div style="background:#FFFFFF;border:1px solid #D8E5DD;border-left:6px solid #1F7A3E;border-radius:8px;padding:14px 18px;box-shadow:0 4px 10px rgba(0,0,0,0.02)">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:17px;color:#11291A;margin-bottom:3px">Gap 1: Factory-Centric Hardware Bias</div>
<div style="font-size:16px;color:#4F6255;line-height:1.4">Existing literature assumes RM50,000+ factory rigs with discrete GPUs. CIRCA targets standard RM0 repair-bench CPUs.</div>
</div>

<div style="background:#FFFFFF;border:1px solid #D8E5DD;border-left:6px solid #1F7A3E;border-radius:8px;padding:14px 18px;box-shadow:0 4px 10px rgba(0,0,0,0.02)">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:17px;color:#11291A;margin-bottom:3px">Gap 2: Desklamp Specular Glare &amp; Shadowing</div>
<div style="font-size:16px;color:#4F6255;line-height:1.4">Factory AOI uses controlled polarising lighting. CIRCA introduces a real-time LAB-CLAHE &amp; Gamma pipeline for repair benches.</div>
</div>

<div style="background:#FFFFFF;border:1px solid #D8E5DD;border-left:6px solid #1F7A3E;border-radius:8px;padding:14px 18px;box-shadow:0 4px 10px rgba(0,0,0,0.02)">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:17px;color:#11291A;margin-bottom:3px">Gap 3: Isolated Bare-Board vs Assembly Taxonomies</div>
<div style="font-size:16px;color:#4F6255;line-height:1.4">Prior studies isolate bare-board (IPC-A-600) or assembly (IPC-A-610H). CIRCA unifies both into a single 7-class schema.</div>
</div>
</div>
</div>
</div>
<div class="footer-bar">
<span class="footer-text">Literature Review &nbsp;|&nbsp; CIRCA FYP 2026</span>
<span class="footer-text">Group 6C &nbsp;|&nbsp; UiTM FSKM</span>
</div>
</div>""",

        # Slide 5: Research Methodology Framework (Chapter 3 §3.2, §3.4, §3.8)
        r"""<div class="slide bg-light">
<div class="hdr">
<div class="hdr-logos"><img src="{LOGO_UITM}" alt="UiTM"><img src="{LOGO_FSKM}" alt="FSKM"></div>
<div class="hdr-title">CIRCA</div>
<div class="slide-num">05 / 13</div>
</div>
<div class="content-area" style="margin-top:20px;gap:30px">
<div class="col2" style="flex:1">
<div class="sec-hdr">Research Framework Methodology (Thesis &sect;3.2)</div>
<div style="display:flex;flex-direction:column;gap:10px;background:#FFFFFF;border:1px solid #D8E5DD;border-radius:10px;padding:16px;box-shadow:0 4px 15px rgba(26,58,42,0.04)">
  <!-- Phase 1 -->
  <div style="display:flex;align-items:center;gap:14px;background:#F4FAF6;border:1px solid #B0D9C0;border-left:6px solid #1F7A3E;border-radius:8px;padding:10px 14px">
    <div style="background:#11291A;color:#7FD99A;font-family:'Montserrat',sans-serif;font-weight:900;font-size:15px;width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0">P1</div>
    <div>
      <div style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:17px;color:#11291A">Phase 1: Multi-Source Dataset Unification</div>
      <div style="font-size:15px;color:#4F6255;margin-top:2px">6 public sources &bull; dHash deduplication &bull; 7-class taxonomy alignment</div>
    </div>
  </div>
  <div style="text-align:center;font-size:16px;color:#1F7A3E;font-weight:900;line-height:0.8">&udarr;</div>

  <!-- Phase 2 -->
  <div style="display:flex;align-items:center;gap:14px;background:#F4FAF6;border:1px solid #B0D9C0;border-left:6px solid #1F7A3E;border-radius:8px;padding:10px 14px">
    <div style="background:#11291A;color:#7FD99A;font-family:'Montserrat',sans-serif;font-weight:900;font-size:15px;width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0">P2</div>
    <div>
      <div style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:17px;color:#11291A">Phase 2: OFAT Image Preprocessing Pipeline</div>
      <div style="font-size:15px;color:#4F6255;margin-top:2px">LAB-CLAHE contrast &bull; Gamma shadow lifting (&gamma;=1.2) &bull; Blur gate (&sigma;&sup2;&le;12.5)</div>
    </div>
  </div>
  <div style="text-align:center;font-size:16px;color:#1F7A3E;font-weight:900;line-height:0.8">&udarr;</div>

  <!-- Phase 3 -->
  <div style="display:flex;align-items:center;gap:14px;background:#F4FAF6;border:1px solid #B0D9C0;border-left:6px solid #1F7A3E;border-radius:8px;padding:10px 14px">
    <div style="background:#11291A;color:#7FD99A;font-family:'Montserrat',sans-serif;font-weight:900;font-size:15px;width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0">P3</div>
    <div>
      <div style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:17px;color:#11291A">Phase 3: Deep Learning Training &amp; Genetic HPO</div>
      <div style="font-size:15px;color:#4F6255;margin-top:2px">YOLOv12 (Nano, Small, Medium) &bull; 50-gen HPO (lr0=0.00014, box=0.169)</div>
    </div>
  </div>
  <div style="text-align:center;font-size:16px;color:#1F7A3E;font-weight:900;line-height:0.8">&udarr;</div>

  <!-- Phase 4 -->
  <div style="display:flex;align-items:center;gap:14px;background:#F4FAF6;border:1px solid #B0D9C0;border-left:6px solid #1F7A3E;border-radius:8px;padding:10px 14px">
    <div style="background:#11291A;color:#7FD99A;font-family:'Montserrat',sans-serif;font-weight:900;font-size:15px;width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0">P4</div>
    <div>
      <div style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:17px;color:#11291A">Phase 4: OpenVINO Edge Conversion &amp; Quantization</div>
      <div style="font-size:15px;color:#4F6255;margin-top:2px">ONNX export &bull; OpenVINO IR optimization &bull; FP16 vs INT8 fallback evaluation</div>
    </div>
  </div>
  <div style="text-align:center;font-size:16px;color:#1F7A3E;font-weight:900;line-height:0.8">&udarr;</div>

  <!-- Phase 5 -->
  <div style="display:flex;align-items:center;gap:14px;background:#F4FAF6;border:1px solid #B0D9C0;border-left:6px solid #B8860B;border-radius:8px;padding:10px 14px">
    <div style="background:#11291A;color:#E5C158;font-family:'Montserrat',sans-serif;font-weight:900;font-size:15px;width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0">P5</div>
    <div>
      <div style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:17px;color:#11291A">Phase 5: PyQt6 Desktop GUI &amp; Real-World Validation</div>
      <div style="font-size:15px;color:#4F6255;margin-top:2px">Interactive desktop UI &bull; Tiled inference &bull; IPC ticket export &bull; 50 real board test</div>
    </div>
  </div>
</div>
</div>

<div class="col2" style="flex:1">
<div class="sec-hdr">Data Sourcing &amp; Software Stack (Thesis &sect;3.4, &sect;3.8)</div>
<div class="card" style="padding:16px 20px">
<h4 style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:17px;color:#11291A;margin-bottom:8px">Dataset Collection (8,463 Images)</h4>
<p style="font-size:16px;color:#4F6255;line-height:1.4;margin-bottom:8px">Compiled from 6 public sources: PKU Open Lab, DsPCBSD+, SolDef_AI, kydra, hue-dbgbs, SolderV2 (CC BY 4.0). 6/64-bit dHash deduplication removed 6,000 near-duplicates.</p>
<div style="display:flex;gap:10px;margin-top:6px">
<div style="background:#F4FAF6;border:1px solid #B0D9C0;padding:8px 12px;border-radius:6px;flex:1;text-align:center"><strong style="color:#11291A;font-size:18px">5,924</strong><div style="font-size:13px;color:#4F6255">Train (70%)</div></div>
<div style="background:#F4FAF6;border:1px solid #B0D9C0;padding:8px 12px;border-radius:6px;flex:1;text-align:center"><strong style="color:#11291A;font-size:18px">1,269</strong><div style="font-size:13px;color:#4F6255">Val (15%)</div></div>
<div style="background:#F4FAF6;border:1px solid #B0D9C0;padding:8px 12px;border-radius:6px;flex:1;text-align:center"><strong style="color:#11291A;font-size:18px">1,270</strong><div style="font-size:13px;color:#4F6255">Test (15%)</div></div>
</div>
</div>

<div class="card" style="padding:16px 20px">
<h4 style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:17px;color:#11291A;margin-bottom:8px">Software &amp; Hardware Stack (Thesis &sect;3.8)</h4>
<table style="width:100%;border-collapse:collapse;font-size:15px;color:#223328">
<tr style="border-bottom:1px solid #E8EFEA"><td style="padding:5px 0;font-weight:700">Desktop GUI Framework</td><td style="padding:5px 0;text-align:right;font-family:'Roboto Mono',monospace">PyQt6 6.6 (Python 3.11)</td></tr>
<tr style="background:#F4FAF6;border-bottom:1px solid #E8EFEA"><td style="padding:5px 0;font-weight:700">Deep Learning Model</td><td style="padding:5px 0;text-align:right;font-family:'Roboto Mono',monospace">Ultralytics YOLOv12-Nano</td></tr>
<tr style="border-bottom:1px solid #E8EFEA"><td style="padding:5px 0;font-weight:700">Inference Engine</td><td style="padding:5px 0;text-align:right;font-family:'Roboto Mono',monospace">Intel OpenVINO 2024.0 FP16</td></tr>
<tr style="background:#F4FAF6;border-bottom:1px solid #E8EFEA"><td style="padding:5px 0;font-weight:700">Image Preprocessing</td><td style="padding:5px 0;text-align:right;font-family:'Roboto Mono',monospace">OpenCV 4.9 (LAB-CLAHE)</td></tr>
<tr style="border-bottom:1px solid #E8EFEA"><td style="padding:5px 0;font-weight:700">Target CPU Hardware</td><td style="padding:5px 0;text-align:right;font-family:'Roboto Mono',monospace;color:#1F7A3E;font-weight:700">Commodity Intel CPU (Zero GPU)</td></tr>
</table>
</div>
</div>
</div>
<div class="footer-bar">
<span class="footer-text">Research Methodology &nbsp;|&nbsp; CIRCA FYP 2026</span>
<span class="footer-text">Group 6C &nbsp;|&nbsp; UiTM FSKM</span>
</div>
</div>""",

        # Slide 6: System Architecture & 6-Stage Inference Pipeline (Chapter 3 §3.5)
        r"""<div class="slide bg-light">
<div class="hdr">
<div class="hdr-logos"><img src="{LOGO_UITM}" alt="UiTM"><img src="{LOGO_FSKM}" alt="FSKM"></div>
<div class="hdr-title">CIRCA</div>
<div class="slide-num">06 / 13</div>
</div>
<div class="content-area" style="margin-top:20px;gap:30px">
<div class="col2" style="flex:1">
<div class="sec-hdr">6-Stage Pipeline Architecture (Thesis &sect;3.5)</div>
<div class="card" style="padding:16px 20px">
<div style="display:flex;flex-direction:column;gap:10px">
<div style="display:flex;align-items:center;gap:14px"><div style="background:#11291A;color:#7FD99A;font-family:'Roboto Mono',monospace;font-weight:700;font-size:15px;padding:5px 10px;border-radius:4px">STAGE 1</div><div style="font-size:17px;color:#223328"><strong>Image Capture &amp; Tiling:</strong> 640x640 tile extraction from webcam/disk.</div></div>
<div style="display:flex;align-items:center;gap:14px"><div style="background:#11291A;color:#7FD99A;font-family:'Roboto Mono',monospace;font-weight:700;font-size:15px;padding:5px 10px;border-radius:4px">STAGE 2</div><div style="font-size:17px;color:#223328"><strong>LAB-CLAHE Enhancement:</strong> L-channel contrast boost (clip limit = 2.0).</div></div>
<div style="display:flex;align-items:center;gap:14px"><div style="background:#11291A;color:#7FD99A;font-family:'Roboto Mono',monospace;font-weight:700;font-size:15px;padding:5px 10px;border-radius:4px">STAGE 3</div><div style="font-size:17px;color:#223328"><strong>Fixed Gamma Correction:</strong> Shadow lifting (&gamma; = 1.2) for desklamp shadows.</div></div>
<div style="display:flex;align-items:center;gap:14px"><div style="background:#11291A;color:#7FD99A;font-family:'Roboto Mono',monospace;font-weight:700;font-size:15px;padding:5px 10px;border-radius:4px">STAGE 4</div><div style="font-size:17px;color:#223328"><strong>Laplacian Blur Quality Gate:</strong> Rejects out-of-focus frames (&sigma;&sup2; &le; 12.5).</div></div>
<div style="display:flex;align-items:center;gap:14px"><div style="background:#11291A;color:#7FD99A;font-family:'Roboto Mono',monospace;font-weight:700;font-size:15px;padding:5px 10px;border-radius:4px">STAGE 5</div><div style="font-size:17px;color:#223328"><strong>OpenVINO Tiled Inference:</strong> YOLOv12-Nano FP16 engine (24.51ms/tile).</div></div>
<div style="display:flex;align-items:center;gap:14px"><div style="background:#11291A;color:#7FD99A;font-family:'Roboto Mono',monospace;font-weight:700;font-size:15px;padding:5px 10px;border-radius:4px">STAGE 6</div><div style="font-size:17px;color:#223328"><strong>Class-Aware NMS &amp; Dual UI:</strong> IoU 0.45 merge + confidence transparency.</div></div>
</div>
</div>
</div>

<div class="col2" style="flex:1">
<div class="sec-hdr">Preprocessing Latency Breakdown (Table 4.4 / 4.11)</div>
<div class="card" style="padding:16px 20px">
<table style="width:100%;border-collapse:collapse;font-size:16px;color:#223328">
<tr style="background:#11291A;color:#7FD99A;font-weight:700"><th style="padding:8px 12px;text-align:left">Stage / Operation</th><th style="padding:8px 12px;text-align:right">Mean Latency</th><th style="padding:8px 12px;text-align:right">% Total</th></tr>
<tr style="border-bottom:1px solid #E8EFEA"><td style="padding:7px 12px">LAB-CLAHE (L-channel eq.)</td><td style="padding:7px 12px;text-align:right;font-family:'Roboto Mono',monospace">3.61 ms</td><td style="padding:7px 12px;text-align:right">76.0%</td></tr>
<tr style="background:#F4FAF6;border-bottom:1px solid #E8EFEA"><td style="padding:7px 12px">Fixed Gamma (&gamma; = 1.2)</td><td style="padding:7px 12px;text-align:right;font-family:'Roboto Mono',monospace">0.40 ms</td><td style="padding:7px 12px;text-align:right">8.4%</td></tr>
<tr style="border-bottom:1px solid #E8EFEA"><td style="padding:7px 12px">Laplacian Blur Gate (&sigma;&sup2; &le; 12.5)</td><td style="padding:7px 12px;text-align:right;font-family:'Roboto Mono',monospace">0.74 ms</td><td style="padding:7px 12px;text-align:right">15.6%</td></tr>
<tr style="background:#11291A;color:#E5C158;font-weight:700"><td style="padding:8px 12px">Total Preprocessing Latency</td><td style="padding:8px 12px;text-align:right;font-family:'Roboto Mono',monospace">4.75 ms</td><td style="padding:8px 12px;text-align:right">100.0%</td></tr>
</table>
<div style="font-size:15px;color:#4F6255;margin-top:12px;line-height:1.4">
&#9679; <strong>Negligible Overhead:</strong> 4.75ms preprocessing consumes only <strong>1.2%</strong> of the total 0.392s end-to-end diagnostic time.<br>
&#9679; <strong>On-Demand Static Mode:</strong> Ensures instant feedback when the technician triggers a diagnostic snapshot.
</div>
</div>
</div>
</div>
<div class="footer-bar">
<span class="footer-text">System Architecture &nbsp;|&nbsp; CIRCA FYP 2026</span>
<span class="footer-text">Group 6C &nbsp;|&nbsp; UiTM FSKM</span>
</div>
</div>""",

        # Slide 7: Dataset Sourcing, Taxonomy & Preprocessing Ablation (Chapter 4 §4.2, §4.3)
        r"""<div class="slide bg-dark">
<div class="hdr">
<div class="hdr-logos"><img src="{LOGO_UITM}" alt="UiTM"><img src="{LOGO_FSKM}" alt="FSKM"></div>
<div class="hdr-title">CIRCA</div>
<div class="slide-num">07 / 13</div>
</div>
<div class="content-area" style="margin-top:20px;gap:30px">
<div class="col2" style="flex:1.15">
<div class="sec-hdr" style="background:linear-gradient(90deg,#11291A,#1E432C);border-left-color:#E5C158">Corpus Distribution &amp; Balancing (Thesis &sect;4.2)</div>
<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(127,217,154,0.3);border-radius:8px;padding:16px 20px">
  <div style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:18px;color:#7FD99A;margin-bottom:10px">Corpus Instance Distribution (54,928 Total Instances)</div>
  <table style="width:100%;border-collapse:collapse;font-size:17px;color:#B8D8C0">
    <tr style="border-bottom:2px solid rgba(127,217,154,0.3);font-weight:800;color:#EFF4FC;background:rgba(255,255,255,0.05)">
      <th style="text-align:left;padding:8px 12px">Class Name</th>
      <th style="text-align:right;padding:8px 12px">Instances</th>
      <th style="text-align:right;padding:8px 12px">% Share</th>
      <th style="text-align:right;padding:8px 12px">Strategy</th>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)"><td style="padding:7px 12px;font-weight:600">missing_hole</td><td style="text-align:right;font-family:'Roboto Mono',monospace;padding:7px 12px">2,315</td><td style="text-align:right;padding:7px 12px">4.2%</td><td style="text-align:right;color:#E5C158;font-weight:700;padding:7px 12px">5x Oversampled</td></tr>
    <tr style="background:rgba(255,255,255,0.03);border-bottom:1px solid rgba(255,255,255,0.05)"><td style="padding:7px 12px;font-weight:600">mouse_bite</td><td style="text-align:right;font-family:'Roboto Mono',monospace;padding:7px 12px">4,887</td><td style="text-align:right;padding:7px 12px">8.9%</td><td style="text-align:right;color:#8B9D90;padding:7px 12px">Standard</td></tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)"><td style="padding:7px 12px;font-weight:600">open_circuit</td><td style="text-align:right;font-family:'Roboto Mono',monospace;padding:7px 12px">3,990</td><td style="text-align:right;padding:7px 12px">7.3%</td><td style="text-align:right;color:#8B9D90;padding:7px 12px">Standard</td></tr>
    <tr style="background:rgba(255,255,255,0.03);border-bottom:1px solid rgba(255,255,255,0.05)"><td style="padding:7px 12px;font-weight:600">short</td><td style="text-align:right;font-family:'Roboto Mono',monospace;padding:7px 12px">12,373</td><td style="text-align:right;padding:7px 12px">22.6%</td><td style="text-align:right;color:#8B9D90;padding:7px 12px">1k Capped</td></tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)"><td style="padding:7px 12px;font-weight:600">excess_solder</td><td style="text-align:right;font-family:'Roboto Mono',monospace;padding:7px 12px">7,120</td><td style="text-align:right;padding:7px 12px">13.0%</td><td style="text-align:right;color:#E5C158;font-weight:700;padding:7px 12px">5x Oversampled</td></tr>
    <tr style="background:rgba(255,255,255,0.03);border-bottom:1px solid rgba(255,255,255,0.05)"><td style="padding:7px 12px;font-weight:600">insufficient_solder</td><td style="text-align:right;font-family:'Roboto Mono',monospace;padding:7px 12px">23,610</td><td style="text-align:right;padding:7px 12px">43.0%</td><td style="text-align:right;color:#8B9D90;padding:7px 12px">1k Capped</td></tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)"><td style="padding:7px 12px;font-weight:600">cold_solder_joint</td><td style="text-align:right;font-family:'Roboto Mono',monospace;padding:7px 12px">633</td><td style="text-align:right;padding:7px 12px">1.2%</td><td style="text-align:right;color:#E5C158;font-weight:700;padding:7px 12px">5x Oversampled</td></tr>
  </table>
</div>
</div>

<div class="col2" style="flex:0.85">
<div class="sec-hdr" style="background:linear-gradient(90deg,#11291A,#1E432C);border-left-color:#E5C158">OFAT Preprocessing Ablation (Figure 4.3)</div>
<div style="text-align:center;background:rgba(17,41,26,0.6);border:1px solid rgba(127,217,154,0.25);border-radius:10px;padding:12px">
<img src="{FIG4_3_ABLATION}" alt="OFAT Preprocessing Ablation" style="width:100%;height:auto;max-height:340px;object-fit:contain;border-radius:6px">
<div style="font-size:15px;color:#9EC9AB;font-weight:700;margin-top:6px">Figure 4.3: Phase 1 (Baseline 72.90%) vs Phase 2 (LAB-CLAHE + Gamma 84.43%)</div>
</div>

<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(127,217,154,0.3);border-radius:8px;padding:14px 18px;margin-top:6px">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:17px;color:#7FD99A;margin-bottom:4px">Ablation Discussion (Thesis &sect;4.3)</div>
<div style="font-size:15px;color:#B8D8C0;line-height:1.4">
&#9679; <strong>+11.53 pp Precision Boost:</strong> LAB-CLAHE contrast boost and Gamma shadow lifting (&gamma;=1.2) boosted Precision from 72.90% to <strong>84.43% (+11.53 pp)</strong>.<br>
&#9679; <strong>Specular Glare Neutralization:</strong> CLAHE on L-channel suppressed desklamp reflections into clean background.
</div>
</div>
</div>
</div>
<div class="footer-bar">
<span class="footer-text">Dataset &amp; Preprocessing &nbsp;|&nbsp; CIRCA FYP 2026</span>
<span class="footer-text">Group 6C &nbsp;|&nbsp; UiTM FSKM</span>
</div>
</div>""",

        # Slide 8: Genetic HPO & 3-Variant Comparative Training (Chapter 4 §4.4, §4.5)
        r"""<div class="slide bg-dark">
<div class="hdr">
<div class="hdr-logos"><img src="{LOGO_UITM}" alt="UiTM"><img src="{LOGO_FSKM}" alt="FSKM"></div>
<div class="hdr-title">CIRCA</div>
<div class="slide-num">08 / 13</div>
</div>
<div class="content-area" style="margin-top:20px;gap:30px">
<div class="col2" style="flex:1">
<div class="sec-hdr" style="background:linear-gradient(90deg,#11291A,#1E432C);border-left-color:#E5C158">Genetic HPO Progression (Figure 4.2)</div>
<div style="text-align:center;background:rgba(17,41,26,0.6);border:1px solid rgba(127,217,154,0.25);border-radius:10px;padding:12px">
<img src="{TUNE_FITNESS}" alt="Genetic HPO Fitness Curve" style="width:100%;height:auto;max-height:340px;object-fit:contain;border-radius:6px">
<div style="font-size:15px;color:#9EC9AB;font-weight:700;margin-top:6px">Figure 4.2: Genetic Algorithm Fitness Score across 50 Generations</div>
</div>

<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(127,217,154,0.3);border-radius:8px;padding:14px 18px;margin-top:6px">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:17px;color:#7FD99A;margin-bottom:4px">Hyperparameter Insights (Thesis &sect;4.4)</div>
<div style="font-size:15px;color:#B8D8C0;line-height:1.4">
&#9679; <strong>Fitness Optimization:</strong> Score improved from <strong>0.42 to 0.65</strong> over 50 iterations.<br>
&#9679; <strong>Learning Rate (`lr0`):</strong> Reduced 71&times; from 0.01 to <strong>0.00014</strong>, preventing gradient explosion.<br>
&#9679; <strong>Box Loss Weight (`box`):</strong> Scaled down from 7.5 to <strong>0.169</strong>, stabilizing micro-box regression.
</div>
</div>
</div>

<div class="col2" style="flex:1">
<div class="sec-hdr" style="background:linear-gradient(90deg,#11291A,#1E432C);border-left-color:#E5C158">YOLOv12-Nano Training &amp; Variant Tradeoff (Figure 4.4a)</div>
<div style="text-align:center;background:rgba(17,41,26,0.6);border:1px solid rgba(127,217,154,0.25);border-radius:10px;padding:12px">
<img src="{FIG4_4A_NANO}" alt="YOLOv12-Nano Training Curves" style="width:100%;height:auto;max-height:340px;object-fit:contain;border-radius:6px">
<div style="font-size:15px;color:#9EC9AB;font-weight:700;margin-top:6px">Figure 4.4a: YOLOv12-Nano 200-Epoch Training &amp; Validation Loss Convergence</div>
</div>

<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(127,217,154,0.3);border-radius:8px;padding:14px 18px;margin-top:6px">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:17px;color:#7FD99A;margin-bottom:4px">3-Variant Performance Comparison (Thesis &sect;4.5)</div>
<div style="font-size:15px;color:#B8D8C0;line-height:1.4">
&#9679; <strong>Nano (Production):</strong> 2.38M params &bull; 5.07MB FP16 &bull; 24.51ms/tile &bull; 63.13% Val mAP<br>
&#9679; <strong>Small:</strong> 9.1M params &bull; 19.3MB FP16 &bull; 71.04ms/tile &bull; 64.20% Val mAP (+1.07 pp)<br>
&#9679; <strong>Medium:</strong> 20.5M params &bull; 43.1MB FP16 &bull; 142.8ms/tile &bull; 64.85% Val mAP (+1.72 pp)
</div>
</div>
</div>
</div>
<div class="footer-bar">
<span class="footer-text">Training &amp; HPO &nbsp;|&nbsp; CIRCA FYP 2026</span>
<span class="footer-text">Group 6C &nbsp;|&nbsp; UiTM FSKM</span>
</div>
</div>""",

        # Slide 9: OpenVINO Quantization & Hardware Benchmarking (Chapter 4 §4.6, §4.7)
        r"""<div class="slide bg-dark">
<div class="hdr">
<div class="hdr-logos"><img src="{LOGO_UITM}" alt="UiTM"><img src="{LOGO_FSKM}" alt="FSKM"></div>
<div class="hdr-title">CIRCA</div>
<div class="slide-num">09 / 13</div>
</div>
<div class="content-area" style="margin-top:20px;gap:30px">
<div class="col2" style="flex:1">
<div class="sec-hdr" style="background:linear-gradient(90deg,#11291A,#1E432C);border-left-color:#E5C158">Hardware Benchmarking Metrics (Thesis &sect;4.7)</div>
<div style="display:flex;flex-direction:column;gap:10px">
<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(127,217,154,0.3);border-radius:8px;padding:12px 18px;display:flex;justify-content:space-between;align-items:center">
<div style="font-size:17px;color:#B8D8C0">End-to-end CPU Latency</div>
<div style="font-family:'Montserrat',sans-serif;font-weight:900;font-size:26px;color:#7FD99A">0.392s</div>
</div>
<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(127,217,154,0.3);border-radius:8px;padding:12px 18px;display:flex;justify-content:space-between;align-items:center">
<div style="font-size:17px;color:#B8D8C0">Model Parameters</div>
<div style="font-family:'Montserrat',sans-serif;font-weight:900;font-size:26px;color:#7FD99A">2.38M (Nano)</div>
</div>
<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(127,217,154,0.3);border-radius:8px;padding:12px 18px;display:flex;justify-content:space-between;align-items:center">
<div style="font-size:17px;color:#B8D8C0">Model File Size (FP16)</div>
<div style="font-family:'Montserrat',sans-serif;font-weight:900;font-size:26px;color:#7FD99A">5.07 MB</div>
</div>
<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(174,34,34,0.4);border-radius:8px;padding:14px 18px">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:16px;color:#E5C158">&#9888; PCIe Transfer Overhead &amp; GPU Benchmarking</div>
<div style="font-size:15px;color:#B8D8C0;margin-top:3px;line-height:1.4">At batch size 1, PCIe bus transfer overhead makes GPU slower than CPU (100.69ms GPU vs 71.04ms CPU on Small variant). Commodity CPU execution eliminates PCIe host-to-device transfer penalties.</div>
</div>
</div>
</div>

<div class="col2" style="flex:1">
<div class="sec-hdr" style="background:linear-gradient(90deg,#11291A,#1E432C);border-left-color:#E5C158">OpenVINO Quantization Fallback Analysis (Thesis &sect;4.6)</div>
<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(127,217,154,0.3);border-radius:8px;padding:16px 20px">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:18px;color:#7FD99A;margin-bottom:10px">FP16 vs INT8 Precision Comparison</div>
<table style="width:100%;border-collapse:collapse;font-size:16px;color:#B8D8C0">
<tr style="border-bottom:1px solid rgba(127,217,154,0.2);font-weight:700;color:#EFF4FC">
<th style="text-align:left;padding:6px 0">Format</th>
<th style="text-align:right;padding:6px 0">File Size</th>
<th style="text-align:right;padding:6px 0">mAP@0.5</th>
<th style="text-align:right;padding:6px 0">mAP Drop</th>
<th style="text-align:right;padding:6px 0">Status</th>
</tr>
<tr><td style="padding:6px 0">FP16 (Half)</td><td style="text-align:right;font-family:'Roboto Mono',monospace">5.07 MB</td><td style="text-align:right">63.13%</td><td style="text-align:right">0.00 pp</td><td style="text-align:right;color:#7FD99A;font-weight:700">DEPLOYED</td></tr>
<tr style="background:rgba(255,255,255,0.03)"><td style="padding:6px 0">INT8 (Quant)</td><td style="text-align:right;font-family:'Roboto Mono',monospace">2.71 MB</td><td style="text-align:right">61.98%</td><td style="text-align:right;color:#E5C158">-1.15 pp</td><td style="text-align:right;color:#E5C158;font-weight:700">REJECTED</td></tr>
</table>
</div>

<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(127,217,154,0.3);border-radius:8px;padding:16px 20px;margin-top:10px">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:18px;color:#7FD99A;margin-bottom:6px">Accuracy-Preserving Fallback Rationale (Thesis &sect;4.6)</div>
<div style="font-size:16px;color:#B8D8C0;line-height:1.5">
&#9679; <strong>Strict Quality Gate:</strong> Protocol required &lt; 1.0 pp mAP degradation for INT8 deployment.<br>
&#9679; <strong>Sub-millimeter Edge Sensitivity:</strong> INT8 weight truncation degraded 8-bit activation scales on fine defect edges. FP16 was chosen as default production engine for zero precision loss.
</div>
</div>
</div>
</div>
<div class="footer-bar">
<span class="footer-text">Hardware &amp; Quantization &nbsp;|&nbsp; CIRCA FYP 2026</span>
<span class="footer-text">Group 6C &nbsp;|&nbsp; UiTM FSKM</span>
</div>
</div>""",

        # Slide 10: Official Test Evaluation & Performance Curves (Chapter 4 §4.8)
        r"""<div class="slide bg-light">
<div class="hdr">
<div class="hdr-logos"><img src="{LOGO_UITM}" alt="UiTM"><img src="{LOGO_FSKM}" alt="FSKM"></div>
<div class="hdr-title">CIRCA</div>
<div class="slide-num">10 / 13</div>
</div>
<div class="content-area" style="margin-top:20px">
<div class="col2">
<div class="sec-hdr">Official Test Evaluation Metrics (Thesis Table 4.15)</div>
<div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:14px">
<div style="background:linear-gradient(135deg,#11291A,#1E432C);border-radius:8px;padding:12px 16px;text-align:center;flex:1;min-width:120px;box-shadow:0 4px 15px rgba(17,41,26,0.15)">
<div style="font-family:'Montserrat',sans-serif;font-weight:900;font-size:38px;color:#7FD99A;line-height:1">85.70%</div>
<div style="font-family:'Roboto Mono',monospace;font-size:13px;color:#9EC9AB;letter-spacing:1px;margin-top:4px;font-weight:700">PRECISION</div>
</div>
<div style="background:linear-gradient(135deg,#11291A,#1E432C);border-radius:8px;padding:12px 16px;text-align:center;flex:1;min-width:120px;box-shadow:0 4px 15px rgba(17,41,26,0.15)">
<div style="font-family:'Montserrat',sans-serif;font-weight:900;font-size:38px;color:#7FD99A;line-height:1">60.59%</div>
<div style="font-family:'Roboto Mono',monospace;font-size:13px;color:#9EC9AB;letter-spacing:1px;margin-top:4px;font-weight:700">RECALL</div>
</div>
<div style="background:linear-gradient(135deg,#11291A,#1E432C);border-radius:8px;padding:12px 16px;text-align:center;flex:1;min-width:120px;box-shadow:0 4px 15px rgba(17,41,26,0.15)">
<div style="font-family:'Montserrat',sans-serif;font-weight:900;font-size:38px;color:#7FD99A;line-height:1">62.79%</div>
<div style="font-family:'Roboto Mono',monospace;font-size:13px;color:#9EC9AB;letter-spacing:1px;margin-top:4px;font-weight:700">mAP@0.5</div>
</div>
<div style="background:linear-gradient(135deg,#11291A,#1E432C);border-radius:8px;padding:12px 16px;text-align:center;flex:1;min-width:120px;box-shadow:0 4px 15px rgba(17,41,26,0.15)">
<div style="font-family:'Montserrat',sans-serif;font-weight:900;font-size:38px;color:#7FD99A;line-height:1">70.99%</div>
<div style="font-family:'Roboto Mono',monospace;font-size:13px;color:#9EC9AB;letter-spacing:1px;margin-top:4px;font-weight:700">F1-SCORE</div>
</div>
</div>
<div class="card" style="padding:16px 20px">
<h3 style="font-size:20px;margin-bottom:10px">Per-Class Test Metrics (YOLOv12-N FP16)</h3>
<table style="width:100%;border-collapse:collapse;font-size:16px">
<tr style="background:#11291A;color:#7FD99A;font-weight:700"><th style="padding:6px 10px;text-align:left;font-family:'Montserrat',sans-serif;font-size:15px">Class Name</th><th style="padding:6px 10px;text-align:right">Precision</th><th style="padding:6px 10px;text-align:right">Recall</th><th style="padding:6px 10px;text-align:right">AP@0.5</th></tr>
<tr style="border-bottom:1px solid #E8EFEA;background:rgba(31,122,62,0.08)"><td style="padding:5px 10px">cold_solder_joint</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace;color:#1F7A3E;font-weight:700">90.97%</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace">85.71%</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace;font-weight:700;color:#1F7A3E">71.52%</td></tr>
<tr style="background:#F4FAF6;border-bottom:1px solid #E8EFEA"><td style="padding:5px 10px">insufficient_solder</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace;color:#1F7A3E;font-weight:700">91.54%</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace;font-weight:700;color:#1F7A3E">91.44%</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace">50.29%</td></tr>
<tr style="border-bottom:1px solid #E8EFEA"><td style="padding:5px 10px">short</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace;color:#1F7A3E;font-weight:700">89.46%</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace">79.77%</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace">51.44%</td></tr>
<tr style="background:#F4FAF6;border-bottom:1px solid #E8EFEA"><td style="padding:5px 10px">open_circuit</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace">83.57%</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace">63.97%</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace">35.97%</td></tr>
<tr style="border-bottom:1px solid #E8EFEA"><td style="padding:5px 10px">mouse_bite</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace">79.01%</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace">53.88%</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace">25.51%</td></tr>
<tr style="background:#FFF8E7;border-bottom:1px solid #E8EFEA"><td style="padding:5px 10px">excess_solder</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace">65.36%</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace;color:#B8860B;font-weight:700">49.37%</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace">32.98%</td></tr>
<tr style="background:#FDE8E8;border-bottom:1px solid #E8EFEA"><td style="padding:5px 10px;font-weight:700;color:#A92222">missing_hole</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace;color:#A92222;font-weight:700">100.00%*</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace;color:#A92222;font-weight:700">0.00%</td><td style="padding:5px 10px;text-align:right;font-family:'Roboto Mono',monospace;color:#A92222;font-weight:700">0.65%</td></tr>
</table>
<div style="font-size:14px;color:#A92222;font-style:italic;margin-top:6px">* Physical Resolution Floor: 0.3mm drill hole translates to 3-4px in 640x640 tile (Thesis &sect;4.8.5)</div>
</div>
</div>

<div class="col2">
<div class="sec-hdr">Performance Curves (Fig 4.7a &amp; 4.7b)</div>
<div style="display:flex;gap:14px">
<div style="flex:1;text-align:center;background:#FFFFFF;border:1px solid #D8E5DD;border-radius:10px;padding:12px;box-shadow:0 4px 15px rgba(0,0,0,0.02)">
<img src="{FIG4_7A}" alt="PR Curve" style="width:100%;height:auto;max-height:430px;object-fit:contain;border-radius:6px">
<div style="font-size:14px;color:#11291A;font-weight:700;margin-top:6px">Fig 4.7a: Precision-Recall Curve</div>
</div>
<div style="flex:1;text-align:center;background:#FFFFFF;border:1px solid #D8E5DD;border-radius:10px;padding:12px;box-shadow:0 4px 15px rgba(0,0,0,0.02)">
<img src="{FIG4_7B}" alt="F1 Curve" style="width:100%;height:auto;max-height:430px;object-fit:contain;border-radius:6px">
<div style="font-size:14px;color:#11291A;font-weight:700;margin-top:6px">Fig 4.7b: F1-Confidence Curve</div>
</div>
</div>
<div class="card" style="padding:14px 18px;margin-top:4px">
<h4 style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:17px;color:#11291A;margin-bottom:4px">Performance Curve Insights (Thesis &sect;4.8.4)</h4>
<p style="font-size:15px;color:#4F6255;line-height:1.4">Peak test F1-score of <strong>70.99% (0.71)</strong> occurs at confidence threshold <strong>0.223</strong> across all 7 defect classes.</p>
</div>
</div>
</div>
<div class="footer-bar">
<span class="footer-text">Performance Results &nbsp;|&nbsp; CIRCA FYP 2026</span>
<span class="footer-text">Group 6C &nbsp;|&nbsp; UiTM FSKM</span>
</div>
</div>""",

        # Slide 11: Confusion Matrix, Realboard Sweep & Failure Analysis (Chapter 4 §4.8, §4.9)
        r"""<div class="slide bg-dark">
<div class="hdr">
<div class="hdr-logos"><img src="{LOGO_UITM}" alt="UiTM"><img src="{LOGO_FSKM}" alt="FSKM"></div>
<div class="hdr-title">CIRCA</div>
<div class="slide-num">11 / 13</div>
</div>
<div class="content-area" style="margin-top:20px;gap:30px">
<div class="col2" style="flex:1">
<div class="sec-hdr" style="background:linear-gradient(90deg,#11291A,#1E432C);border-left-color:#E5C158">Confusion Matrix &amp; Realboard Sweep (Fig 4.6 / 4.9)</div>
<div style="display:flex;gap:14px">
<div style="flex:1;text-align:center;background:rgba(17,41,26,0.6);border:1px solid rgba(127,217,154,0.25);border-radius:10px;padding:10px">
<img src="{FIG4_6_CONFUSION}" alt="Confusion Matrix" style="width:100%;height:auto;max-height:360px;object-fit:contain;border-radius:6px">
<div style="font-size:14px;color:#9EC9AB;font-weight:700;margin-top:4px">Fig 4.6: Normalised Confusion Matrix</div>
</div>
<div style="flex:1;text-align:center;background:rgba(17,41,26,0.6);border:1px solid rgba(127,217,154,0.25);border-radius:10px;padding:10px">
<img src="{FIG4_9_THRESHOLD}" alt="Confidence Threshold Sweep" style="width:100%;height:auto;max-height:360px;object-fit:contain;border-radius:6px">
<div style="font-size:14px;color:#9EC9AB;font-weight:700;margin-top:4px">Fig 4.9: 50 Realboard Image Sweep</div>
</div>
</div>

<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(127,217,154,0.3);border-radius:8px;padding:14px 18px;margin-top:6px">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:17px;color:#7FD99A;margin-bottom:4px">Realboard Domain Adaptation Lift (Thesis &sect;4.9)</div>
<div style="font-size:15px;color:#B8D8C0;line-height:1.4">
&#9679; <strong>+600% True Positive Lift:</strong> Threshold auto-tuning boosted True Positives from 4 to <strong>28 TPs (+600% recall lift)</strong> on 50 handheld phone PCB test images.<br>
&#9679; <strong>Optimal Threshold Shift:</strong> Lowering threshold to 0.15 on phone cameras maximized recall (+5.48%).
</div>
</div>
</div>

<div class="col2" style="flex:1">
<div class="sec-hdr" style="background:linear-gradient(90deg,#11291A,#1E432C);border-left-color:#E5C158">Failure Gallery &amp; Resolution Floor (Fig 4.8)</div>
<div style="text-align:center;background:rgba(17,41,26,0.6);border:1px solid rgba(127,217,154,0.25);border-radius:10px;padding:10px">
<img src="{FIG4_8_FAILURE}" alt="Failure Gallery" style="width:100%;height:auto;max-height:360px;object-fit:contain;border-radius:6px">
<div style="font-size:14px;color:#9EC9AB;font-weight:700;margin-top:4px">Figure 4.8: Inspection Escapes &amp; Specular Glare False Positives</div>
</div>

<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(184,134,11,0.4);border-radius:8px;padding:14px 18px;margin-top:6px">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:17px;color:#E5C158;margin-bottom:4px">&#9888; Physical Resolution Floor &amp; Specular Glare</div>
<div style="font-size:15px;color:#B8D8C0;line-height:1.4">
&#9679; <strong>`missing_hole` Resolution Limit:</strong> 0.3mm drill holes map to 3-4px in a 640x640 tile, proving a physical camera resolution limit rather than architectural failure.<br>
&#9679; <strong>Specular Glare:</strong> Unpolarised desklamp reflections account for 62% of false positives.
</div>
</div>
</div>
</div>
<div class="footer-bar">
<span class="footer-text">Failure Analysis &nbsp;|&nbsp; CIRCA FYP 2026</span>
<span class="footer-text">Group 6C &nbsp;|&nbsp; UiTM FSKM</span>
</div>
</div>""",

        # Slide 12: CIRCA Desktop GUI Application & Operator Bias UI (Chapter 3 §3.5, Chapter 4 §4.9, RO3)
        r"""<div class="slide bg-dark">
<div class="hdr">
<div class="hdr-logos"><img src="{LOGO_UITM}" alt="UiTM"><img src="{LOGO_FSKM}" alt="FSKM"></div>
<div class="hdr-title">CIRCA</div>
<div class="slide-num">12 / 13</div>
</div>
<div class="content-area" style="margin-top:20px;gap:30px">
<div class="col2" style="flex:1.3">
<div class="sec-hdr" style="background:linear-gradient(90deg,#11291A,#1E432C);border-left-color:#E5C158">CIRCA PyQt6 Inspection Interface (RO3)</div>
<div style="background:rgba(17,41,26,0.6);border:1px solid rgba(127,217,154,0.25);border-radius:10px;padding:14px;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,0.15)">
<img src="{APP_SCREENSHOT}" alt="CIRCA Full App Interface" style="width:100%;height:auto;max-height:560px;object-fit:contain;border-radius:8px">
</div>
</div>
<div class="col2" style="flex:0.7">
<div class="sec-hdr" style="background:linear-gradient(90deg,#11291A,#1E432C);border-left-color:#E5C158">Operator Bias &amp; Ticket Export</div>
<div style="display:flex;flex-direction:column;gap:14px">
<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(127,217,154,0.3);border-radius:8px;padding:16px 20px">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:18px;color:#7FD99A;margin-bottom:6px">&#9888; Dual-Threshold Warning Banner</div>
<div style="font-size:16px;color:#B8D8C0;line-height:1.5">Triggers <strong>"Manual Inspection Required"</strong> alert when mean confidence &lt; 50% or blur &sigma;&sup2; &le; 12.5. Fire-rate on clean frames is <strong>4.20%</strong> (&lt; 5.0% limit per Thesis &sect;4.9.3).</div>
</div>
<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(127,217,154,0.3);border-radius:8px;padding:16px 20px">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:18px;color:#7FD99A;margin-bottom:6px">&#128221; Plain-English IPC Defect Checklist</div>
<div style="font-size:16px;color:#B8D8C0;line-height:1.5">Side-panel checklist displaying defect counts, IPC-A-600 / IPC-A-610H standard references, and raw confidence scores for human-in-the-loop verification.</div>
</div>
<div style="background:rgba(17,41,26,0.85);border:1px solid rgba(127,217,154,0.3);border-radius:8px;padding:16px 20px">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:18px;color:#7FD99A;margin-bottom:6px">&#128190; One-Click Quality Grade Ticket Export</div>
<div style="font-size:16px;color:#B8D8C0;line-height:1.5">Generates programmatic inspection tickets (TXT/CSV) with IPC defect tallies and diagnostic timestamps for shop POS and repair ticketing software.</div>
</div>
</div>
</div>
</div>
<div class="footer-bar">
<span class="footer-text">Application Walkthrough &nbsp;|&nbsp; CIRCA FYP 2026</span>
<span class="footer-text">Group 6C &nbsp;|&nbsp; UiTM FSKM</span>
</div>
</div>""",

        # Slide 13: Conclusions, Achievements, Limitations & Future Work (Chapter 5 §5.2–§5.5)
        r"""<div class="slide bg-light">
<div class="hdr">
<div class="hdr-logos"><img src="{LOGO_UITM}" alt="UiTM"><img src="{LOGO_FSKM}" alt="FSKM"></div>
<div class="hdr-title">CIRCA</div>
<div class="slide-num">13 / 13</div>
</div>
<div class="content-area" style="margin-top:20px">
<div class="col2">
<div class="sec-hdr">Conclusion &amp; RO Achievement (Thesis &sect;5.2)</div>
<div style="display:flex;flex-direction:column;gap:14px">
<div style="background:#FFFFFF;border:1px solid #D8E5DD;border-left:6px solid #1F7A3E;border-radius:6px;padding:16px 20px;box-shadow:0 4px 10px rgba(0,0,0,0.02)">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:19px;color:#11291A;margin-bottom:4px">&#10003; RO1: Unified Dual-Standard IPC Taxonomy</div>
<div style="font-size:17px;color:#4F6255;line-height:1.5">Established 7-class taxonomy combining bare-board (IPC-A-600) and solder (IPC-A-610H). Identified 0.3mm drill hole resolution floor as a key physical finding.</div>
</div>
<div style="background:#FFFFFF;border:1px solid #D8E5DD;border-left:6px solid #1F7A3E;border-radius:6px;padding:16px 20px;box-shadow:0 4px 10px rgba(0,0,0,0.02)">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:19px;color:#11291A;margin-bottom:4px">&#10003; RO2: YOLOv12 Edge CPU Model Optimization</div>
<div style="font-size:17px;color:#4F6255;line-height:1.5">Evaluated Nano/Small/Medium; deployed OpenVINO FP16 CPU engine (85.70% Precision, 0.392s latency) &mdash; zero GPU required.</div>
</div>
<div style="background:#FFFFFF;border:1px solid #D8E5DD;border-left:6px solid #1F7A3E;border-radius:6px;padding:16px 20px;box-shadow:0 4px 10px rgba(0,0,0,0.02)">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:19px;color:#11291A;margin-bottom:4px">&#10003; RO3: CIRCA PyQt6 Desktop App &amp; Realboard Test</div>
<div style="font-size:17px;color:#4F6255;line-height:1.5">Developed PyQt6 app with warning banner &amp; ticket export; achieved +600% recall lift (28 TPs vs 4 baseline) on 50 real phone PCB test images.</div>
</div>
</div>
</div>
<div class="col2">
<div class="sec-hdr">Limitations &amp; Future Work (Thesis &sect;5.4, &sect;5.5)</div>
<div style="display:flex;flex-direction:column;gap:14px">
<div style="background:#FFFFFF;border:1px solid #D8E5DD;border-left:6px solid #B8860B;border-radius:6px;padding:16px 20px;box-shadow:0 4px 10px rgba(0,0,0,0.02)">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:19px;color:#5A3F1B;margin-bottom:4px">1. Higher-Resolution Tiling (1024x1024)</div>
<div style="font-size:17px;color:#4F6255;line-height:1.5">Mitigate sub-resolution boundary smearing on 0.3mm missing drill holes without sacrificing CPU throughput.</div>
</div>
<div style="background:#FFFFFF;border:1px solid #D8E5DD;border-left:6px solid #B8860B;border-radius:6px;padding:16px 20px;box-shadow:0 4px 10px rgba(0,0,0,0.02)">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:19px;color:#5A3F1B;margin-bottom:4px">2. Synthetic Copy-Paste Data Expansion</div>
<div style="font-size:17px;color:#4F6255;line-height:1.5">Generate domain-adapted composite training images to increase instance density for rare defect categories (<code style="color:#11291A">cold_solder_joint</code>).</div>
</div>
<div style="background:#FFFFFF;border:1px solid #D8E5DD;border-left:6px solid #B8860B;border-radius:6px;padding:16px 20px;box-shadow:0 4px 10px rgba(0,0,0,0.02)">
<div style="font-family:'Montserrat',sans-serif;font-weight:700;font-size:19px;color:#5A3F1B;margin-bottom:4px">3. Hardware Polarising Ring Integration</div>
<div style="font-size:17px;color:#4F6255;line-height:1.5">Cross-polarised ring illuminators under desklamps to physically eliminate specular glare reflections on solder joints before capture.</div>
</div>
</div>
</div>
</div>
<div class="footer-bar">
<span class="footer-text">Conclusions &amp; Future Work &nbsp;|&nbsp; CIRCA FYP 2026</span>
<span class="footer-text">Group 6C &nbsp;|&nbsp; UiTM FSKM</span>
</div>
</div>"""
    ]

    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PNG_DIR.mkdir(parents=True, exist_ok=True)

    formatted_slides = []
    print(f"[DECK] Writing backup slide HTML files to {SLIDES_DIR.relative_to(ROOT)}...")
    for idx, raw_html in enumerate(slides_data, 1):
        s_html = raw_html.format(
            LOGO_UITM=logo_uitm,
            LOGO_FSKM=logo_fskm,
            APP_SCREENSHOT=app_screenshot,
            FIG4_1_DEFECTS=fig4_1_defects,
            FIG4_3_ABLATION=fig4_3_ablation,
            FIG4_4A_NANO=fig4_4a_nano,
            FIG4_6_CONFUSION=fig4_6_confusion,
            FIG4_7A=fig4_7a_pr,
            FIG4_7B=fig4_7b_f1,
            FIG4_7A_PR=fig4_7a_pr,
            FIG4_7B_F1=fig4_7b_f1,
            FIG4_8_FAILURE=fig4_8_failure,
            FIG4_9_THRESHOLD=fig4_9_threshold,
            TUNE_FITNESS=tune_fitness
        )
        formatted_slides.append(s_html)
        
        slide_filename = f"slide_{idx:02d}.html"
        slide_path = SLIDES_DIR / slide_filename
        full_single_slide_html = COMMON_HEAD + s_html + "\n</body></html>"
        with open(slide_path, "w", encoding="utf-8") as f:
            f.write(full_single_slide_html)
        print(f"  - Wrote {slide_filename}")

    combined_html_path = DOCS / "circa_deck.html"
    print(f"[DECK] Writing combined {combined_html_path.relative_to(ROOT)}...")
    full_deck_html = COMMON_HEAD + "\n".join(formatted_slides) + "\n</body></html>"
    with open(combined_html_path, "w", encoding="utf-8") as f:
        f.write(full_deck_html)
    print(f"  - Wrote circa_deck.html")

    print("[DECK] Rendering individual slide PNGs via Edge (2x Retina Ultra-HD Resolution)...")
    msedge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if not os.path.exists(msedge_path):
        msedge_path = "msedge"

    png_files = []
    for idx in range(1, total_slides + 1):
        slide_filename = f"slide_{idx:02d}.html"
        slide_path = SLIDES_DIR / slide_filename
        png_out_path = DOCS / f"slide_{idx:02d}.png"
        
        cmd = [
            msedge_path,
            "--headless",
            "--disable-gpu",
            "--window-size=1920,1080",
            "--force-device-scale-factor=2",
            "--hide-scrollbars",
            f"--screenshot={png_out_path}",
            f"file:///{slide_path.resolve()}".replace("\\", "/")
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if png_out_path.exists():
                print(f"  - Rendered 2x PNG: slide_{idx:02d}.png")
                png_files.append(png_out_path)
            else:
                print(f"[ERROR] Failed to render PNG for slide_{idx:02d}. stderr: {res.stderr}")
        except Exception as e:
            print(f"[ERROR] Edge render failed for slide_{idx:02d}: {e}")

    pdf_out_path = DOCS / "circa_deck.pdf"
    print(f"[DECK] Compiling PDF from 3840x2160 PNG slides via PIL...")
    try:
        from PIL import Image
        images = []
        for p in png_files:
            if p.exists():
                img = Image.open(p).convert("RGB")
                images.append(img)
        if images:
            try:
                images[0].save(
                    pdf_out_path,
                    "PDF",
                    resolution=200.0,
                    save_all=True,
                    append_images=images[1:]
                )
                print(f"  - Saved PDF via PIL: {pdf_out_path}")
            except (PermissionError, OSError) as pe:
                alt_pdf_path = DOCS / "CIRCA_Pitch_Deck.pdf"
                images[0].save(
                    alt_pdf_path,
                    "PDF",
                    resolution=200.0,
                    save_all=True,
                    append_images=images[1:]
                )
                print(f"[WARN] Primary PDF locked by viewer ({pe}). Saved to alternate: {alt_pdf_path}")
        else:
            print("[ERROR] No PNG images available to compile PDF!")
    except Exception as e:
        print(f"[ERROR] Failed to compile PDF via PIL: {e}")

    print("=" * 70)
    print(" DECK GENERATION COMPLETE")
    print("=" * 70)

def main():
    generate_slides()

if __name__ == "__main__":
    generate_slides()
