# Technical Report: Dataset Unification and Preprocessing
**Project:** CIRCA — PCB Defect Detection System
**Date (v1):** Wednesday, April 29, 2026
**Date (v2 expansion):** Thursday, April 30, 2026
**Subject:** Integration of Multi-Source PCB Defect Datasets for YOLOv12 Training

---

## 1. Executive Summary

Eight source datasets are unified into a single master dataset (`unified_pcb_v2`):
- **Bare-board sources (IPC-A-600):** 3 datasets (PKU/JR, RAHUL, Bare PCB Defects), 8 classes (IDs 0–5, 10–11).
- **Assembly-stage sources (IPC-A-610):** 5 datasets (SolDef_AI, kydra, Hue, f8m5i, v2-s89jo), 4 classes (IDs 6–9).
- **Total: 12 classes.** Final image counts confirmed after global deduplication (dHash ≤ 6).

---

## 2. Source Datasets

### 2.1 Bare-board (IPC-A-600)

1. **PKU-Market-PCB-ver1** (~2,910 images) — laboratory-controlled, 6 fundamental defects.
2. **PCB Defect Dataset (Roboflow)** (~~9,666 images (Bare PCB defects, Roboflow)) — varied lighting and camera angles.

### 2.2 Assembly-stage (IPC-A-610)

3. **SolDef_AI** (Fontana et al., 2024) — `defectdetection-e5sqy/soldef_ai-for-defect-detection`. ~1,150 images. CC BY 4.0. Polygon annotations converted to YOLO bbox.
4. **PCB Solder Joint** (PCB Defect Detection, 2025a) — `work-6qkmv/pcb-solder-joint`. ~1,626 images. CC BY 4.0.

> **Out-of-scope:** Component-level IPC-A-610 defects (missing component, misalignment, tombstoning, lifted lead, solder ball, component damage) — no suitable public datasets. Recorded in thesis Chapter 1 §1.5 and Chapter 3 §3.1.

---

## 3. Preprocessing & Methodology

### 3.1 12-Class Taxonomy (Categorical Standardisation)

| ID | Class | IPC | Source |
|:---:|:---|:---|:---|
| 0 | `missing_hole` | A-600 §3.4 | PKU/JR, Bare PCB |
| 1 | `mouse_bite` | A-600 §3.3 | PKU/JR, Bare PCB |
| 2 | `open_circuit` | A-600 §3.2 | PKU/JR, Bare PCB |
| 3 | `short` | A-600 §3.2 | PKU/JR, Bare PCB, Hue |
| 4 | `spur` | A-600 §3.3 | PKU/JR, Bare PCB |
| 5 | `spurious_copper` | A-600 §3.3 | PKU/JR, Bare PCB |
| **6** | **`excess_solder`** | **A-610 §5** | SolDef_AI, kydra, f8m5i, v2-s89jo |
| **7** | **`insufficient_solder`** | **A-610 §5** | SolDef_AI, kydra, Hue, v2-s89jo |
| **8** | **`solder_spike`** | **A-610 §5** | SolDef_AI |
| **9** | **`cold_solder_joint`** | **A-610 §5** | kydra, v2-s89jo |
| **10** | **`scratch`** | **A-600 §3** | Bare PCB defects |
| **11** | **`pinhole`** | **A-600 §3** | Bare PCB defects |

Remap rules: SolDef_AI drop `good`/`no_good`; remap `exc_solder`→6, `poor_solder`→7, `spike`→8. v2-s89jo remap `Cold_solder`→9, `Excessive_solder`→6, `Insufficient_solder`→7. Dropped images archived in `negatives_reserve/`.

### 3.2 Stratified Split — 70/15/15 (`seed=42`)

### 3.3 Training Stability
- `lr0=0.001`, `warmup_epochs=5.0` — prevents gradient explosion in YOLOv12 Area Attention modules.

### 3.4 Image Enhancement Pipeline
- **CLAHE:** L-channel LAB (Clip Limit 2.0, Tile 8×8).
- **Gamma Correction:** γ = 1.2.
- Outputs: `unified_pcb_v2_preproc/` (one-time offline pass).

### 3.5 Integrity Guardrails
- Missing label detection with explicit `WARNING` logging.
- `seed=42` enforced globally.

---

## 4. Dataset Composition

| Split | Count | Note |
|:---|:---|:---|
| Training | ~70% of deduplicated total | Stratified, `seed=42` |
| Validation | ~15% | HPO / early stopping |
| Test | ~15% | Frozen — final evaluation only |

Class-imbalance mitigation: (1) `cls_pw` inverse-frequency weighting (primary, always active); (2) `mosaic=1.0` + `copy_paste=0.3` augmentation; (3) conditional ≤3× oversampling of solder classes (IDs 6–9) only if any solder class mAP@0.5 < 70% after Phase 2. 10× bare-board oversampling removed — causes overfitting on repeated pixel patterns (Cao et al., 2024).

---

## 5. Experimental Roadmap

| Phase | ID | Description | Goal |
|:---|:---|:---|:---|
| 0 | — | Unify four sources, sanitise, split | 12-class reproducible corpus |
| 1 | `001` | Vanilla YOLOv12s, raw images, 50 epochs | Ablation control |
| 2 | `002` | CLAHE+Gamma, 100 epochs | Preprocessing lift |
| 3 | `003` | Genetic HPO, 50 iter × 30 epochs | Optimal config |
| 4 | `004–006` | Final N/S/M, 200 epochs | Three final variants |
| 5 | — | FP32/FP16/INT8 OpenVINO export | Quantisation delta |
| 6 | — | Latency/FPS benchmarking | Hardware validation |
| 7 | — | Frozen test evaluation | Final thesis metrics |
