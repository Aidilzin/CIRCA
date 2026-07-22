# -*- coding: utf-8 -*-
"""
CIRCA Poster Generator (Stand-alone Script)
Generates: docs/circa_poster.html, docs/circa_poster_A3.png, docs/circa_poster.png, docs/circa_poster.pdf
"""
import os
import sys
import base64
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"

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

def build_poster_html():
    print("[POSTER] Encoding image assets...")
    logo_uitm = get_base64_image("Logo UiTM (bg dark).png") or get_base64_image("LogoUiTM.png")
    logo_fskm = get_base64_image("LOGO_FSKM-white_font.png")
    app_screenshot = get_base64_image("app_screenshot_full.png")
    fig4_1_defects = get_base64_image("fig4_1_sample_defects.png")
    fig4_6_confusion = get_base64_image("fig4_6_confusion_matrix.png")

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>CIRCA Poster</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&family=Open+Sans:wght@400;600;700&family=Roboto+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{width:794px;height:1123px;overflow:hidden;background:#FCFCFA;font-family:'Open Sans',sans-serif;color:#1C2820;font-size:8.5px;line-height:1.4;display:flex;flex-direction:column}}
.header{{background:linear-gradient(135deg,#07140B 0%,#11291A 60%,#183B25 100%);padding:8px 20px;display:flex;align-items:center;gap:16px;border-bottom:4px solid #B8860B;position:relative}}
.header::after{{content:'';position:absolute;bottom:0;left:0;right:0;height:1px;background:rgba(255,255,255,0.1)}}
.logo-block{{flex-shrink:0;display:flex;align-items:center;justify-content:center}}
.logo-left{{width:140px;flex-shrink:0;display:flex;flex-direction:column;align-items:flex-start;justify-content:center;gap:1px}}
.logo-right{{width:140px;flex-shrink:0;display:flex;align-items:center;justify-content:flex-end}}
.logo-left img, .logo-right img, .logo-block img{{height:38px;object-fit:contain}}
.hdr-ctr{{flex:1;text-align:center}}
.hdr-tag{{font-family:'Roboto Mono',monospace;color:#D4AF37;font-size:7px;font-weight:700;letter-spacing:3px;text-transform:uppercase;margin-bottom:4px}}
.hdr-title{{font-family:'Montserrat',sans-serif;font-weight:900;font-size:16.5px;color:#FFFFFF;line-height:1.15;letter-spacing:-0.2px;margin-bottom:4px}}
.hdr-title span{{color:#7FD99A}}
.hdr-student{{font-family:'Montserrat',sans-serif;font-weight:700;font-size:9px;color:#E8F5EC;letter-spacing:0.8px;margin-top:2px}}
.hdr-meta{{font-size:6.8px;color:#9EC9AB;margin-top:2px;font-weight:600}}
.hdr-badges{{display:flex;justify-content:center;gap:8px;margin-top:6px}}
.badge{{background:rgba(184,134,11,0.15);border:1px solid rgba(184,134,11,0.4);color:#E5C158;font-family:'Roboto Mono',monospace;font-size:6.2px;padding:2px 8px;border-radius:12px;font-weight:700}}

.abst{{background:#F4FAF6;border-left:4px solid #B8860B;border-right:4px solid #B8860B;padding:8px 16px;border-bottom:1px solid #D0E5D7}}
.sec-lbl{{font-family:'Montserrat',sans-serif;font-weight:900;font-size:8px;color:#B8860B;letter-spacing:2px;text-transform:uppercase;margin-bottom:3px}}
.abst p{{font-size:7.8px;color:#223328;line-height:1.5;text-align:justify;font-weight:400}}
.abst p strong{{color:#11291A}}

.stat-bar{{background:#11291A;display:flex;border-bottom:3px solid #B8860B;padding:4px 0}}
.si{{flex:1;text-align:center;padding:4px 2px;border-right:1px solid rgba(184,134,11,0.25)}}
.si:last-child{{border-right:none}}
.sv{{font-family:'Montserrat',sans-serif;font-weight:900;font-size:13.5px;color:#7FD99A;line-height:1}}
.sl{{font-size:5.8px;color:#9EC9AB;font-family:'Roboto Mono',monospace;letter-spacing:0.5px;margin-top:2px;font-weight:700}}

.body-cols{{display:flex;gap:10px;padding:10px 10px 5px 10px;flex:1;overflow:hidden}}
.col{{flex:1;display:flex;flex-direction:column;gap:9px}}
.card{{background:#FFFFFF;border:1px solid #D8E5DD;border-radius:6px;overflow:hidden;box-shadow:0 3px 10px rgba(26,58,42,0.04);display:flex;flex-direction:column}}
.sh{{background:linear-gradient(90deg,#11291A,#1A3B28);color:#FFFFFF;font-family:'Montserrat',sans-serif;font-weight:800;font-size:8px;letter-spacing:1.5px;text-transform:uppercase;padding:5px 10px;border-left:4px solid #B8860B;display:flex;align-items:center;gap:6px}}
.sh::before{{content:'';display:inline-block;width:4px;height:4px;border-radius:50%;background:#7FD99A;flex-shrink:0}}
.cb{{padding:8px 10px;flex:1;display:flex;flex-direction:column;justify-content:space-between}}

p{{margin-bottom:5px;text-align:justify}}
ul{{padding-left:12px;margin-bottom:5px}}
ul li{{margin-bottom:3px;line-height:1.4}}
.oi{{display:flex;gap:7px;margin-bottom:4px;align-items:flex-start}}
.on{{background:#11291A;color:#7FD99A;font-family:'Montserrat',sans-serif;font-weight:900;font-size:7.5px;width:15px;height:15px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;border:1px solid #B8860B}}
.ot{{font-size:7.8px;line-height:1.4}}
.ot strong{{font-family:'Montserrat',sans-serif;font-weight:700;font-size:7.8px;color:#11291A;display:block;line-height:1.2;margin-bottom:1px}}

.pl{{display:flex;flex-direction:column;gap:4px}}
.ps{{display:flex;align-items:center;gap:6px}}
.pn{{background:#B8860B;color:white;font-family:'Montserrat',sans-serif;font-weight:900;font-size:7px;width:15px;height:15px;border-radius:3px;display:flex;align-items:center;justify-content:center;flex-shrink:0}}
.pl-lbl strong{{font-family:'Montserrat',sans-serif;font-weight:700;font-size:7.8px;color:#11291A;display:block;line-height:1.2}}
.pl-lbl span{{font-size:6.8px;color:#4F6255;font-weight:600}}

.mt{{width:100%;border-collapse:collapse;margin-bottom:5px;font-size:7.8px}}
.mt th{{background:#11291A;color:#7FD99A;font-family:'Montserrat',sans-serif;font-weight:700;font-size:7.2px;padding:4px 6px;text-align:left}}
.mt td{{padding:4px 6px;border-bottom:1px solid #E8EFEA;color:#223328}}
.mt tr:nth-child(even) td{{background:#F4FAF6}}
.vh{{font-family:'Roboto Mono',monospace;font-weight:700;color:#1F7A3E}}
.vl{{font-family:'Roboto Mono',monospace;font-weight:700;color:#A92222}}
.vn{{font-family:'Roboto Mono',monospace;color:#1A291E}}
.mb{{height:4.5px;border-radius:2px;background:linear-gradient(90deg,#11291A,#7FD99A)}}

.fw{{text-align:center;margin:4px 0;background:#FAFDFC;border:1px solid #E4EDE8;border-radius:4px;padding:4px}}
.fw img{{max-width:100%;max-height:85px;object-fit:contain;border:1px solid #D8E5DD;border-radius:3px}}
.fc{{font-size:6.5px;color:#5E7063;font-style:italic;margin-top:3px;text-align:center;font-weight:600}}

.hr{{display:flex;gap:6px;margin-bottom:5px}}
.hb{{flex:1;background:linear-gradient(135deg,#EFF8F2,#F5F9F6);border:1px solid #B0D9C0;border-radius:4px;padding:6px;text-align:center}}
.hv{{font-family:'Montserrat',sans-serif;font-weight:900;font-size:16px;color:#1F7A3E;line-height:1}}
.hl{{font-size:6.2px;color:#4F6255;margin-top:2px;font-weight:700;line-height:1.2}}

.ci{{display:flex;gap:6px;margin-bottom:4px;align-items:flex-start;font-size:7.8px}}
.ci span:first-child{{font-size:9.5px;font-weight:bold;color:#B8860B;flex-shrink:0;line-height:1}}
.ci span:last-child{{color:#223328}}
.itag{{display:inline-block;font-family:'Roboto Mono',monospace;font-size:6.2px;padding:1px 4px;border-radius:2px;font-weight:700;vertical-align:middle}}
.t600{{background:#EFF4FC;color:#1B355A;border:1px solid #B9D0EE}}
.t610{{background:#FCF4E9;color:#5A3F1B;border:1px solid #E6C594}}

.footer{{background:linear-gradient(135deg,#07140B,#11291A);border-top:3px solid #B8860B;padding:8px 20px;display:flex;align-items:center;justify-content:space-between;margin-top:auto}}
.fl{{font-family:'Montserrat',sans-serif;font-weight:700;font-size:7.2px;color:#9EC9AB;line-height:1.5}}
.fl span{{color:#FFFFFF;font-weight:800}}
.fc2{{text-align:center}}
.acr{{font-family:'Montserrat',sans-serif;font-weight:900;font-size:16px;color:#7FD99A;letter-spacing:2px;line-height:1}}
.tgl{{font-size:6.5px;color:#9EC9AB;letter-spacing:1px;margin-top:3px;font-family:'Roboto Mono',monospace}}
.fr{{text-align:right;font-size:6.8px;color:#9EC9AB;font-family:'Roboto Mono',monospace;line-height:1.5}}

.methodology-flow {{
  margin-top: 6px;
  background: #F4FAF6;
  border: 1px solid #D0E5D7;
  border-radius: 4px;
  padding: 8px 10px;
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}}
.flow-grid {{
  display: flex;
  flex-direction: column;
  flex: 1;
  justify-content: space-between;
  position: relative;
}}
.flow-grid::before {{
  content: '';
  position: absolute;
  left: 6.5px;
  top: 7px;
  bottom: 7px;
  width: 1px;
  background: #B8860B;
  z-index: 1;
}}
.flow-step {{
  display: flex;
  align-items: flex-start;
  gap: 6px;
  position: relative;
}}
.step-num {{
  background: #11291A;
  color: #7FD99A;
  font-family: 'Montserrat', sans-serif;
  font-weight: 900;
  font-size: 7px;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  border: 1px solid #B8860B;
  z-index: 2;
  position: relative;
}}
.step-body {{
  font-size: 7.5px;
  color: #223328;
  line-height: 1.35;
}}
.step-body strong {{
  color: #11291A;
  font-family: 'Montserrat', sans-serif;
  font-weight: 700;
  display: block;
  font-size: 7.8px;
}}

@media print {{
  @page {{
    size: 794px 1123px;
    margin: 0px;
  }}
  body {{
    width: 794px;
    height: 1123px;
    margin: 0px;
    padding: 0px;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}
}}
</style></head><body>
<div class="header">
<div class="logo-left">
<div style="background:#B8860B;color:#FFFFFF;font-family:'Roboto Mono',monospace;font-weight:900;font-size:7.5px;padding:2px 7px;border-radius:3px;letter-spacing:1px;white-space:nowrap;margin-bottom:2px">BOOTH NO: MIIS-9</div>
<img src="{logo_uitm}" alt="UiTM">
</div>
<div class="hdr-ctr">
<div class="hdr-tag">&#9658; Final Year Project Exhibition 2026 &#9668;</div>
<div class="hdr-title">CIRCA: <span>Circuit Inspection</span> and Recognition using Convolutional Architectures</div>
<div class="hdr-student">MUHAMMAD AIDIL AL-FAIZI BIN MOHD ZIN</div>
<div class="hdr-meta">Student ID: 2023276732 &nbsp;|&nbsp; Bachelor of Information Systems (Hons.) Intelligent Systems Engineering &nbsp;|&nbsp; Faculty of Computer Science and Mathematics, UiTM Shah Alam</div>
<div class="hdr-badges">
<span class="badge">GROUP 6C</span>
<span class="badge">SUPERVISOR: PN. FARAHNATASYAH BINTI ABDUL HANNAN</span>
<span class="badge">EXAMINER: DR. MOHD ZAKI ZAKARIA</span>
</div></div>
<div class="logo-right"><img src="{logo_fskm}" alt="FSKM"></div>
</div>
<div class="abst">
<div class="sec-lbl">Introduction</div>
<p>Printed Circuit Boards (PCBs) form the physical and electrical foundation of modern consumer electronics. Manual visual inspection is slow, subjective, and prone to fatigue-induced escape rates of 10&ndash;20% on sub-millimeter components. Industrial AOI systems cost RM 50,000&ndash;500,000, making them inaccessible for independent repair shops. <strong>CIRCA</strong> is a lightweight AI-driven visual inspection system unifying bare-board (<span class="itag t600">IPC-A-600</span>) and solder assembly (<span class="itag t610">IPC-A-610H</span>) defect taxonomies into a seven-class schema, trained on 8,463 images with a six-stage pipeline (LAB-CLAHE, Gamma Correction, Laplacian blur gating). The production YOLOv12-Nano FP16 model achieves <strong>86.61% precision</strong> and <strong>0.392 s CPU latency</strong>, demonstrating that factory-grade on-demand PCB inspection is viable on commodity edge hardware&mdash;directly lowering capital barriers for independent electronics repair and supporting the circular economy.</p>
</div>
<div class="stat-bar">
<div class="si"><div class="sv">86.61%</div><div class="sl">PRECISION</div></div>
<div class="si"><div class="sv">65.47%</div><div class="sl">mAP@0.5</div></div>
<div class="si"><div class="sv">73.69%</div><div class="sl">F1-SCORE</div></div>
<div class="si"><div class="sv">64.12%</div><div class="sl">RECALL</div></div>
<div class="si"><div class="sv">0.392s</div><div class="sl">CPU LATENCY</div></div>
<div class="si"><div class="sv">7</div><div class="sl">DEFECT CLASSES</div></div>
<div class="si"><div class="sv">8,463</div><div class="sl">TRAINING IMAGES</div></div>
</div>
<div class="body-cols">
<div class="col">
<div class="card"><div class="sh">Problem Statement</div><div class="cb">
<p>PCBs are ubiquitous in modern electronics. In repair environments, manual visual inspection under magnifying lenses suffers from serious bottlenecks:</p>
<ul style="margin-left:2px">
<li><strong>Slow &amp; Fatigue-Dependent</strong> &mdash; takes 10&ndash;25 mins per board; accuracy degrades over extended shifts.</li>
<li><strong>High Escape Rate</strong> &mdash; 10&ndash;20% of sub-millimeter defects are missed under manual examination.</li>
<li><strong>Prohibitive Cost</strong> &mdash; industrial AOI machines cost RM 50K&ndash;500K, out of reach for SMEs &amp; repair labs.</li>
</ul>
<p>CIRCA bridges this gap by delivering a lightweight, on-demand visual inspection system optimized for cheap commodity edge CPUs.</p>
<div class="fw"><img src="{fig4_1_defects}" alt="Defects"><div class="fc">Fig 1. Sample PCB defect instances across all 7 defect classes from the training corpus</div></div>
</div></div>
<div class="card" style="flex:1"><div class="sh">Research Objectives &amp; Scope</div><div class="cb">
<div style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:7px;color:#B8860B;text-transform:uppercase;margin-bottom:3px;letter-spacing:0.5px">Research Objectives:</div>
<div class="oi"><div class="on">1</div><div class="ot"><strong>Unified Taxonomy</strong>Unify IPC-A-600 (bare-board) and IPC-A-610H (assembly) into a single 7-class schema.</div></div>
<div class="oi"><div class="on">2</div><div class="ot"><strong>On-Demand Pipeline</strong>Build low-latency pipeline with LAB-CLAHE, Gamma prep, &amp; Laplacian blur gate.</div></div>
<div class="oi"><div class="on">3</div><div class="ot"><strong>Edge Optimization</strong>Achieve sub-500ms inference latency on standard Intel CPUs via FP16 OpenVINO.</div></div>
<div class="oi"><div class="on">4</div><div class="ot"><strong>Human Factors</strong>Mitigate automation bias via dual-threshold display &amp; global warning banner.</div></div>
<div style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:7px;color:#B8860B;text-transform:uppercase;margin-top:6px;margin-bottom:2px;letter-spacing:0.5px">Scope of Study:</div>
<p style="font-size:7.5px;line-height:1.3">Focuses on 7 defect classes (<span class="itag t600">IPC-A-600</span> bare-board &amp; <span class="itag t610">IPC-A-610H</span> assembly) evaluated on 8,463 annotated PCB images across commodity Intel edge CPU targets.</p>
<div class="hr" style="margin-top:14px">
  <div class="hb">
    <div class="hv">12,000+</div>
    <div class="hl">Repair Shops<br>Target addressable market across MY &amp; SEA excluded by RM 50K&ndash;500K industrial AOI costs.</div>
  </div>
  <div class="hb">
    <div class="hv" style="color:#B8860B">RM 0</div>
    <div class="hl">Hardware Upgrade Cost<br>CIRCA runs on any standard Intel CPU &mdash; no dedicated GPU or specialist hardware required.</div>
  </div>
</div>
</div></div>
</div>
<div class="col">
<div class="card"><div class="sh">Research Significance &amp; System</div><div class="cb">
<div style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:7px;color:#B8860B;text-transform:uppercase;margin-bottom:3px;letter-spacing:0.5px">Research Significance:</div>
<p style="font-size:7.5px;line-height:1.3;margin-bottom:4px">Lowers capital barriers for independent repair technicians and local SMEs, boosting electronic repair success rates to reduce e-waste and support the circular economy.</p>
<div class="fw" style="margin-bottom:4px"><img src="{app_screenshot}" alt="CIRCA App"><div class="fc">Fig 2. CIRCA desktop application &mdash; live PCB inspection interface</div></div>
<div class="pl">
<div class="ps"><div class="pn">1</div><div class="pl-lbl"><strong>Image Input</strong><span>Webcam or file import (JPEG/PNG)</span></div></div>
<div class="ps"><div class="pn">2</div><div class="pl-lbl"><strong>Pre-processing</strong><span>LAB-CLAHE &rarr; Gamma &rarr; Laplacian blur gate</span></div></div>
<div class="ps"><div class="pn">3</div><div class="pl-lbl"><strong>YOLOv12-N FP16</strong><span>Intel OpenVINO CPU inference backend</span></div></div>
<div class="ps"><div class="pn">4</div><div class="pl-lbl"><strong>Dual-Threshold Filter</strong><span>Lo-conf warning / hi-conf bounding box display</span></div></div>
<div class="ps"><div class="pn">5</div><div class="pl-lbl"><strong>IPC Grade Report</strong><span>Pass / Warn / Fail per IPC standard classification</span></div></div>
</div></div></div>
<div class="card" style="flex:1"><div class="sh">Research Methodology</div><div class="cb">
<div class="methodology-flow" style="margin-top:0;flex:1">
  <div style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:7px;color:#B8860B;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px">7-Phase Research Methodology Pipeline:</div>
  <div class="flow-grid">
    <div class="flow-step"><div class="step-num">1</div><div class="step-body"><strong>Data Curation &amp; Split</strong>8,463 images (70% train, 20% val, 10% test)</div></div>
    <div class="flow-step"><div class="step-num">2</div><div class="step-body"><strong>Image Pre-processing</strong>Contrast-limited LAB-CLAHE &amp; Gamma correction</div></div>
    <div class="flow-step"><div class="step-num">3</div><div class="step-body"><strong>OFAT Ablation Gate</strong>Laplacian variance check for motion blur detection</div></div>
    <div class="flow-step"><div class="step-num">4</div><div class="step-body"><strong>Genetic HPO Optimization</strong>GA auto-tuned learning rate (71x default reduction)</div></div>
    <div class="flow-step"><div class="step-num">5</div><div class="step-body"><strong>YOLOv12 Deep Learning</strong>Trained Nano, Small, &amp; Medium variants for 200 epochs</div></div>
    <div class="flow-step"><div class="step-num">6</div><div class="step-body"><strong>OpenVINO Quantization</strong>Model converted from FP32 to optimized FP16/INT8</div></div>
    <div class="flow-step"><div class="step-num">7</div><div class="step-body"><strong>Dual-Threshold Deploy</strong>Calibrated live display &amp; IPC Quality Grade report</div></div>
  </div>
</div>
<div style="margin-top:6px;background:#FAFDFC;border:1px solid #D8E5DD;border-radius:4px;padding:6px">
  <div style="font-family:'Montserrat',sans-serif;font-weight:800;font-size:6.8px;color:#11291A;text-transform:uppercase;margin-bottom:4px;letter-spacing:0.5px;text-align:center">Methodology Execution &amp; Verification Flow:</div>
  <div style="display:flex;flex-direction:column;gap:3px;font-size:6.5px">
    <div style="display:flex;gap:4px">
      <div style="flex:1;background:#EFF8F2;border:1px solid #B0D9C0;border-radius:3px;padding:3px;text-align:center">
        <strong style="color:#11291A;display:block;font-size:6.8px">Phase 2: OFAT Ablation</strong>
        <span style="color:#4F6255;font-size:6px">LAB-CLAHE + Blur Gate</span>
      </div>
      <div style="flex:1;background:#EFF8F2;border:1px solid #B0D9C0;border-radius:3px;padding:3px;text-align:center">
        <strong style="color:#11291A;display:block;font-size:6.8px">Phase 4: Genetic HPO</strong>
        <span style="color:#4F6255;font-size:6px">71&times; LR, 44&times; Box Weight</span>
      </div>
    </div>
    <div style="background:#11291A;color:#7FD99A;border-radius:3px;padding:4px;text-align:center;margin-top:1px">
      <strong style="color:#FFFFFF;font-family:'Montserrat',sans-serif;font-size:6.8px;display:block;margin-bottom:2px">4 Benchmark Acceptance Criteria Verification:</strong>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:2px;font-family:'Roboto Mono',monospace;font-size:6px;color:#D4AF37">
        <div>&check; Precision &ge; 85% (86.61%)</div>
        <div>&check; CPU Latency &lt; 500ms (0.392s)</div>
        <div>&check; 7 IPC Defect Classes</div>
        <div>&check; Zero GPU Requirement</div>
      </div>
    </div>
  </div>
</div>
</div></div>
</div>
<div class="col">
<div class="card"><div class="sh">Results and findings</div><div class="cb">
<table class="mt"><tr><th>Metric</th><th>Score</th><th style="width:58px">Bar</th></tr>
<tr><td>Precision</td><td class="vh">86.61%</td><td><div class="mb" style="width:86%"></div></td></tr>
<tr><td>Recall</td><td class="vn">64.12%</td><td><div class="mb" style="width:64%"></div></td></tr>
<tr><td>F1-Score</td><td class="vn">73.69%</td><td><div class="mb" style="width:74%"></div></td></tr>
<tr><td>mAP@0.5</td><td class="vn">65.47%</td><td><div class="mb" style="width:65%"></div></td></tr>
<tr><td>mAP@0.5:0.95</td><td class="vn">41.52%</td><td><div class="mb" style="width:42%"></div></td></tr>
</table>
<div style="font-size:7.2px;font-weight:700;color:#11291A;margin-bottom:3px">Per-Class AP@0.5 Breakdown:</div>
<table class="mt" style="font-size:7px"><tr><th>Class</th><th>Prec.</th><th>AP@0.5</th></tr>
<tr><td>cold_solder</td><td class="vh">96.82%</td><td>74.06%</td></tr>
<tr><td>insuff._solder</td><td class="vh">90.73%</td><td>52.63%</td></tr>
<tr><td>short</td><td class="vh">90.81%</td><td>56.70%</td></tr>
<tr><td>open_circuit</td><td class="vn">81.58%</td><td>39.08%</td></tr>
<tr><td>mouse_bite</td><td class="vn">82.87%</td><td>29.30%</td></tr>
<tr><td>excess_solder</td><td class="vn">63.45%</td><td>38.20%</td></tr>
<tr><td>missing_hole</td><td class="vl">100%*</td><td class="vl">0.70%</td></tr>
</table>
<div style="font-size:6.2px;color:#8B4513;font-style:italic;margin-top:2px;margin-bottom:3px">* Trivial precision; zero true-positive detections due to dataset imbalance.</div>
<div class="fw" style="margin:2px 0"><img src="{fig4_6_confusion}" alt="Confusion Matrix"><div class="fc">Fig 3. Normalised confusion matrix &mdash; YOLOv12-N FP16</div></div>
</div></div>
<div class="card"><div class="sh">Conclusion</div><div class="cb">
<div class="ci"><span>&#10003;</span><span><strong>86.61% precision</strong> at 0.392s CPU latency &mdash; viable for real repair bench deployment, no GPU required</span></div>
<div class="ci"><span>&#10003;</span><span>First system unifying IPC-A-600 + IPC-A-610H in a single lightweight end-to-end inference pipeline</span></div>
<div class="ci"><span>&#10003;</span><span>Dual-threshold display + warning banner effectively mitigates operator automation bias</span></div>
</div></div>
<div class="card" style="flex:1"><div class="sh">References</div><div class="cb" style="font-size:6.5px;line-height:1.2">
<div style="margin-bottom:2px">1. <strong>Adibhatla, V.A., et al. (2020)</strong>. Defect Detection in PCBs Using YOLO CNNs. <em>Electronics, 9(9)</em>.</div>
<div style="margin-bottom:2px">2. <strong>Tian, Y., et al. (2025)</strong>. YOLOv12: Attention-Centric Real-Time Object Detectors. <em>arXiv:2502.12524</em>.</div>
<div style="margin-bottom:2px">3. <strong>Lv, S., et al. (2024)</strong>. A dataset for DL-based detection of PCB surface defects. <em>Scientific Data, 11(1)</em>.</div>
<div style="margin-bottom:2px">4. <strong>Fontana, G., et al. (2024)</strong>. SolDef_AI: Open Source PCB Dataset for Mask R-CNN. <em>J. Manuf. &amp; Mater. Processing, 8(3)</em>.</div>
<div style="margin-bottom:2px">5. <strong>Bhattacharya, A., &amp; Cloutier, S.G. (2022)</strong>. End-to-end DL for PCB manufacturing defect classification. <em>Scientific Reports, 12(1)</em>.</div>
<div style="margin-bottom:2px">6. <strong>Kupfer, C., et al. (2023)</strong>. Check the box! Automation bias in AI-based selection. <em>Frontiers in Psychology, 14</em>.</div>
<div style="margin-bottom:2px">7. <strong>IPC Association (2020)</strong>. <em>IPC-A-610H: Acceptability of Electronic Assemblies</em>. IPC.</div>
<div>8. <strong>IPC Association (2020)</strong>. <em>IPC-A-600K: Acceptability of Printed Boards</em>. IPC.</div>
</div></div>
</div>
</div>
<div class="footer">
<div class="fl"><span>Faculty of Computer Science and Mathematics</span><br>Universiti Teknologi MARA (UiTM) Shah Alam<br>Supervisor: Pn. Farahnatasyah Binti Abdul Hannan</div>
<div class="fc2"><div class="acr">CIRCA</div><div class="tgl">On-Demand PCB Defect Detection System</div></div>
<div class="fr">YOLOv12-N &middot; FP16 &middot; OpenVINO &middot; IPC-A-600/610H<br>Group 6C &middot; FYP Exhibition 2026<br><span style="color:#B8860B">July 2026</span></div>
</div>
</body></html>"""

    poster_path = DOCS / "circa_poster.html"
    poster_path.write_text(html, encoding="utf-8")
    print(f"[POSTER] Saved {poster_path}")

def verify_column_heights(tolerance_px=3):
    """Measure rendered column bottom edges using Playwright and assert they are within tolerance."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[POSTER] [WARN] Playwright not available — skipping column height verification.")
        return

    poster_html = (DOCS / "circa_poster.html").resolve()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 794, "height": 1123})
        page.goto(poster_html.as_uri(), wait_until="networkidle")
        result = page.evaluate("""() => {
            const cols = document.querySelectorAll('.body-cols .col');
            const measurements = [];
            cols.forEach((col, i) => {
                const cards = col.querySelectorAll('.card');
                const lastCard = cards[cards.length - 1];
                const lastCardRect = lastCard ? lastCard.getBoundingClientRect() : null;
                measurements.push({
                    col: i + 1,
                    lastCardBottom: lastCardRect ? Math.round(lastCardRect.bottom) : null,
                    lastCardHeight: lastCardRect ? Math.round(lastCardRect.height) : null
                });
            });
            return measurements;
        }""")
        browser.close()

    print("[POSTER] === COLUMN HEIGHT VERIFICATION ===")
    bottoms = [c["lastCardBottom"] for c in result]
    ref = max(bottoms)
    all_pass = True
    for col in result:
        diff = ref - col["lastCardBottom"]
        status = "OK" if diff <= tolerance_px else f"SHORT by {diff}px"
        print(f"  Col {col['col']}: lastCard bottom={col['lastCardBottom']}px  height={col['lastCardHeight']}px  [{status}]")
        if diff > tolerance_px:
            all_pass = False
    spread = max(bottoms) - min(bottoms)
    if all_pass:
        print(f"  PASS: All columns within {tolerance_px}px tolerance (spread={spread}px)")
    else:
        raise AssertionError(
            f"[POSTER] FAIL: Column heights not level (spread={spread}px). "
            "Fix the short column(s) listed above before rendering."
        )

def render_poster():
    EDGE_PATHS = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
    ]
    edge_exe = next((p for p in EDGE_PATHS if Path(p).exists()), None)
    if not edge_exe:
        print("[ERROR] Edge browser executable not found.")
        return

    poster_html = DOCS / "circa_poster.html"
    out_png_a3 = DOCS / "circa_poster_A3.png"
    out_png = DOCS / "circa_poster.png"
    pdf_out = DOCS / "circa_poster.pdf"

    print("[POSTER] Rendering PNGs and PDF via Edge...")
    subprocess.run([edge_exe, "--headless", "--disable-gpu", "--window-size=1240,1754", f"--screenshot={out_png_a3}", poster_html.as_uri()], check=True)
    import shutil
    shutil.copyfile(out_png_a3, out_png)
    subprocess.run([edge_exe, "--headless", "--disable-gpu", f"--print-to-pdf={pdf_out}", poster_html.as_uri()], check=True)
    print("[POSTER] Completed rendering poster deliverables successfully.")

def main():
    build_poster_html()
    verify_column_heights()   # ← mandatory height check before every render
    render_poster()

if __name__ == "__main__":
    main()

