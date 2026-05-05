# CIRCA — Chapter 3 & Chapter 4 Outlines

> Aligned with **CSP650 Topic 3 (Research Methodology)** and **CSP650 Topic 4 (Results and Findings)** guidelines.
> Section numbering follows Sample C of the CSP650 Topic 3 guideline, adapted for an applied deep-learning system project.

---

## CHAPTER 3 — RESEARCH METHODOLOGY (Table of Contents)

### 3.1 Introduction
Brief statement of the chapter's purpose: describe how the CIRCA project was carried out — framework, techniques, materials, and experimental design.

### 3.2 Research Framework
- 3.2.1 Overview of Research Phases (one figure: see `CIRCA_DIAGRAMS.md` Figure 3.1)
- 3.2.2 Mapping of Objectives → Activities → Deliverables (one table)

### 3.3 Theoretical Study
- 3.3.1 Preliminary Study (research gap from Chapter 2 — repair-context PCB inspection)
- 3.3.2 Knowledge Acquisition (IPC-A-610 standard, YOLOv12 architecture, OpenVINO toolkit)

### 3.4 Empirical Study
- 3.4.1 Data Collection
  - 3.4.1.1 Public bare-board datasets (PKU-Market-PCB-ver1/JR, RAHUL, Bare PCB Defects) — IPC-A-600 territory, retained for completeness
  - 3.4.1.2 Public assembly-stage datasets (SolDef_AI — Fontana et al. 2024, PCB Solder Joint — Work 2025) — IPC-A-610 solder-defect coverage
  - 3.4.1.3 Repair-context capture protocol (webcam, lighting variation)
  - 3.4.1.4 Unified 12-class IPC taxonomy (6 bare-board IPC-A-600 + 4 solder IPC-A-610 + 2 additional IPC-A-600 surface defects)
  - 3.4.1.5 Scope and Limitations — component-level IPC-A-610 defects (missing component, misalignment, tombstoning, lifted lead, solder ball, component damage) excluded due to absence of suitable public datasets; rationale: IP/privacy, low base-rate, proprietary AOI ownership, no standard taxonomy. Future-work recommendation: custom data-collection campaign.
- 3.4.2 Data Pre-processing
  - 3.4.2.1 CLAHE on L-channel of LAB
  - 3.4.2.2 Gamma Correction (γ = 1.2)
  - 3.4.2.3 Laplacian Variance frame quality gate (inference-time)
  - 3.4.2.4 Polygon→bbox conversion for SolDef_AI annotations
  - 3.4.2.5 Class remap to 15-class taxonomy (drop background `good`/`GOOD` images; archive in negatives reserve)
  - 3.4.2.6 Stratified 70 / 15 / 15 split with `seed=42`
- 3.4.3 Data Analysis
  - 3.4.3.1 Class distribution audit (12 classes; ~5:1 bare-board↔solder imbalance; mitigation: `cls_pw=1.0` inverse-frequency power)
  - 3.4.3.2 Duplicate / leakage detection (perceptual hash dedup, dHash distance ≤ 6, cross-source check)

### 3.5 System Design
- 3.5.1 CIRCA System Architecture (one figure: see `CIRCA_DIAGRAMS.md` Figure 3.2)
- 3.5.2 Inference Pipeline (Webcam → Preproc → OpenVINO → Overlay UI)
- 3.5.3 Confidence Threshold and "Manual Inspection Required" Logic
- 3.5.4 Interface Design (bounding-box overlay, confidence display, warning banner)

### 3.6 System Development
- 3.6.1 Training Engine (`train_engine.py`)
- 3.6.2 Hyperparameter Optimisation Algorithm (Ultralytics genetic tuner)
  - 3.6.2.1 Search space
  - 3.6.2.2 Stopping criteria and trial budget
- 3.6.3 Model Training Procedure (per-variant, with HPO config)
- 3.6.4 OpenVINO Export and INT8 Quantisation
- 3.6.5 Confidence Threshold Calibration Procedure

### 3.7 Experimental Design
- 3.7.1 Phase 1 — Vanilla Baseline (ablation control)
- 3.7.2 Phase 2 — CIRCA-Aligned Baseline
- 3.7.3 Phase 3 — Hyperparameter Optimisation
- 3.7.4 Phase 4 — Three-Variant Final Training (N / S / M)
- 3.7.5 Phase 5 — Quantisation Validation (FP32 / FP16 / INT8)
- 3.7.6 Phase 6 — Hardware Benchmarking and Variant Selection
- 3.7.7 Phase 7 — Final Test Evaluation and Threshold Calibration
- 3.7.8 Acceptance Criteria
- 3.7.9 Evaluation Metrics (Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95, latency, FPS)

### 3.8 Hardware and Software Specification
- 3.8.1 Training Environment (GPU, RAM, OS, framework versions)
- 3.8.2 Deployment Target (Intel Core i5 8th-gen-equivalent CPU + iGPU)
- 3.8.3 Software Stack (Ultralytics, PyTorch, OpenVINO, NNCF, OpenCV, Python version)

### 3.9 Research Plan / Project Timeline
Gantt-style table of phase durations.

### 3.10 Summary
One paragraph closing the chapter and bridging to Chapter 4.

---

## CHAPTER 4 — RESULTS AND FINDINGS (Table of Contents)

### 4.1 Introduction
States the purpose of the chapter and the organisation pattern adopted (Topic 4 guideline allows either *all-results-then-discussion* or *interleaved*; CIRCA uses **interleaved by objective**).

### 4.2 Dataset and Defect Taxonomy Results (answers Objective 1)
- 4.2.1 Final Class Distribution — 12 IPC classes (6 bare-board IPC-A-600, 4 solder IPC-A-610, 2 additional IPC-A-600 surface defects) (Table 4.1)
- 4.2.2 Dataset Statistics across Splits — train / val / test for the unified `unified_pcb_v2` corpus (Table 4.2)
- 4.2.3 Sample Defect Images per IPC Class (Figure 4.1)
- 4.2.4 Discussion: representativeness of repair-context images, bare-board↔solder imbalance, scope limitation on component-level defects

### 4.3 Preprocessing Pipeline Evaluation
- 4.3.1 Vanilla vs CIRCA-Preprocessed Baseline Comparison (Table 4.3)
- 4.3.2 Preprocessing Latency Measurement (Table 4.4)
- 4.3.3 Discussion: ablation lift attributable to CLAHE + Gamma

### 4.4 Hyperparameter Optimisation Results
- 4.4.1 HPO Search Trajectory (Figure 4.2 — fitness curve)
- 4.4.2 Top-10 Trials (Table 4.5)
- 4.4.3 Parameter Importance (Figure 4.3 — scatter / parallel-coordinate plot)
- 4.4.4 Selected Hyperparameter Configuration (Table 4.6)
- 4.4.5 Discussion: which parameters dominated and why

### 4.5 Three-Variant Comparative Training Results (answers Objective 2)
- 4.5.1 Training Curves per Variant (Figure 4.4)
- 4.5.2 Validation Metrics per Variant (Table 4.7)
- 4.5.3 Per-Class Performance Breakdown (Table 4.8)
- 4.5.4 Discussion: accuracy/size trade-off

### 4.6 OpenVINO Quantisation Results
- 4.6.1 FP32 vs FP16 vs INT8 mAP (Table 4.9)
- 4.6.2 INT8 → FP16 Fallback Decision (narrative + Table 4.10)
- 4.6.3 Discussion: quantisation impact per variant

### 4.7 Hardware Benchmarking Results
- 4.7.1 Preprocessing Latency on Target CPU (Table 4.11)
- 4.7.2 Inference Latency: CPU vs iGPU (Table 4.12)
- 4.7.3 End-to-End Live FPS (Figure 4.5 — moving-average plot)
- 4.7.4 Static Image Inference Time (Table 4.13)
- 4.7.5 Variant Selection Matrix and Acceptance-Criteria Verdict (Table 4.14)
- 4.7.6 Discussion: which variant was selected and why

### 4.8 Final Test-Set Evaluation (answers Objective 3)
- 4.8.1 Overall Metrics on Test Set (Table 4.15)
- 4.8.2 Per-Class Precision, Recall, F1 (Table 4.16)
- 4.8.3 Confusion Matrix (Figure 4.6)
- 4.8.4 PR and F1 Curves (Figure 4.7)
- 4.8.5 Failure-Case Gallery (Figure 4.8)
- 4.8.6 Discussion: where the model fails and why

### 4.9 Confidence Threshold Calibration Results
- 4.9.1 Threshold Sweep on Validation (Figure 4.9)
- 4.9.2 Per-Class Display and Warning Thresholds (Table 4.17)
- 4.9.3 Global "Manual Inspection Required" Trigger Calibration
- 4.9.4 Discussion: automation-bias mitigation in the deployed system

### 4.10 Comparison with Related Work
- 4.10.1 Benchmarking against Published PCB Detectors (Table 4.18)
- 4.10.2 Discussion: where CIRCA stands vs Hu & Wang (2020), Liao et al. (2021), Yang & Yu (2024), Tian et al. (2025)

### 4.11 Summary of Findings
Paragraph closing the chapter, restating which research objectives were met and pointing to Chapter 5 (Conclusion).

---

## Notes on Writing Style (per CSP650 guidelines)

- **Chapter 3:** past tense, chronological, replicable detail, no results.
- **Chapter 4:** every figure and table needs surrounding **explanatory text** that points the reader to the significant trend (e.g. *"YOLOv12-S in INT8 retained 91.4% mAP — close to its FP32 ceiling of 92.1% — while halving model size"*), uses round numbers when emphasising trends ("nearly 92%"), and ends each subsection with a brief comparison or implication.
- Cross-reference Chapter 4 results back to Chapter 2 literature wherever a comparison strengthens the discussion.
