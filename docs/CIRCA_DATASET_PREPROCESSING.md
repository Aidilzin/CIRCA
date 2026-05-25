# Technical Report: Dataset Unification and Preprocessing
**Project:** CIRCA — PCB Defect Detection System
**Date (v1):** 2026-04-29 | **Date (v2):** 2026-04-30 | **Date (v3 rebuild):** 2026-05-22
**Subject:** Integration of Multi-Source PCB Defect Datasets into `unified_pcb_v3` (7-class)

---

## 1. Executive Summary

Six source datasets are unified into a single master dataset (`unified_pcb_v3`) under a principled **7-class IPC taxonomy**. The scope was reduced from 12 to 7 classes following a data-availability audit that identified five defect categories with fewer than 100–400 training instances — insufficient for reliable YOLO detection. The resulting corpus covers four IPC-A-600 bare-board defect classes and three IPC-A-610H assembly-stage solder defect classes.

---

## 2. Source Datasets

### 2.1 Bare-Board (IPC-A-600) — Classes 0–3

| # | Dataset | Source | Images | Classes Retained |
|:---|:---|:---|:---|:---|
| 1 | **PKU-Market-PCB via JR** | Roboflow: `jr-mqqnk/pcb-defects-detection-anddl` | ~1,500 | `missing_hole`, `mouse_bite`, `open_circuit`, `short` |
| 2 | **DsPCBSD+** (Lv et al., 2024) | Figshare: DOI 10.6084/m9.figshare.24970329 | 10,259 | Mapped equivalents (audit class names before use) |

### 2.2 Assembly-Stage (IPC-A-610) — Classes 4–6

| # | Dataset | Source | Images | Classes Retained |
|:---|:---|:---|:---|:---|
| 3 | **SolDef_AI** (Fontana et al., 2024) | Kaggle: `mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection` | ~1,150 | `excess_solder`, `insufficient_solder` (drop `spike`, `good`, `no_good`) |
| 4 | **PCB Solder Defect V2 (v2-s89jo)** | Roboflow: `emmts/pcb-solder-defect-detection-v2-s89jo` | 6,116 | `excess_solder`, `insufficient_solder`, `cold_solder_joint` |
| 5 | **Excessive Solder / kydra** | Roboflow: `emmts/excessive-solder-kydra` | 1,162 | `excess_solder`, `insufficient_solder`, `cold_solder_joint` |
| 6 | **Hue** | Roboflow: `emmts/hue-dbgbs-reqtv` | 3,232 | `insufficient_solder`, `short` (as "Shorted") |

> **Component-level IPC-A-610 defects** (missing component, misalignment, tombstoning, lifted lead, solder ball, component damage) remain out of scope — no suitable public datasets with sufficient instance counts exist.

---

## 3. Preprocessing & Methodology

### 3.1 7-Class Taxonomy

| ID | Class | IPC | Source |
|:---:|:---|:---|:---|
| 0 | `missing_hole` | A-600 §3.4 | PKU/JR, DsPCBSD+ |
| 1 | `mouse_bite` | A-600 §3.3 | PKU/JR, DsPCBSD+ |
| 2 | `open_circuit` | A-600 §3.2 | PKU/JR, DsPCBSD+ |
| 3 | `short` | A-600 §3.2 | PKU/JR, DsPCBSD+, Hue |
| 4 | `excess_solder` | A-610H §5 | SolDef_AI, kydra, v2-s89jo |
| 5 | `insufficient_solder` | A-610H §5 | SolDef_AI, kydra, Hue, v2-s89jo |
| 6 | `cold_solder_joint` | A-610H §5 | kydra, v2-s89jo |

**Rationale for scope reduction from 12 to 7 classes:** Five classes from the previous taxonomy were excluded after a systematic data audit confirmed instance counts below the minimum threshold for reliable YOLO learning (~200 instances): `spur` (379, but visually ambiguous with `short`), `spurious_copper` (414, same issue), `solder_spike` (91), `scratch` (52), `pinhole` (56). These are documented in Chapter 1 §1.5 and Chapter 3 §3.4.1.5 as scope limitations.

### 3.2 Stratified Split — 70/15/15 (`seed=42`)

Target counts after Phase 0 build (to be confirmed):

| Split | Target Images | Notes |
|:---|:---:|:---|
| Train | ~6,300 | Stratified on dominant class per image |
| Val | ~1,350 | HPO / early stopping |
| Test | ~1,350 | **Frozen before any training** |
| **Total** | **~9,000** | After deduplication |

### 3.3 Polygon → Bounding Box Conversion (SolDef_AI)

SolDef_AI ships with polygon (segmentation) annotations. When exported from Roboflow using the **"YOLOv8 Object Detection"** format (not Instance Segmentation), Roboflow automatically converts each polygon to its axis-aligned bounding rectangle in normalised YOLO format. No custom conversion script is required. This ensures bounding boxes are mathematically consistent with any future researcher using the same Roboflow export.

### 3.4 Image Enhancement Pipeline
- **CLAHE:** L-channel LAB, Clip Limit 2.0, Tile 8×8.
- **Gamma Correction:** γ = 1.2.
- **Output:** Cached to `unified_pcb_v3_preproc/` (generated during Phase 2 training).

### 3.5 Integrity Guardrails
- pHash deduplication: dHash ≤ 6 Hamming distance, applied globally across all sources.
- Missing-label detection with explicit `WARNING` logging.
- `seed=42` enforced globally across all phases.
- Test split frozen before Phase 1 begins.

---

## 4. Experimental Roadmap

| Phase | ID | Description | Status |
|:---|:---|:---|:---:|
| 0 | — | Download sources, remap, dedup, split → `unified_pcb_v3/` | ⏳ Pending |
| 1 | `001` | Vanilla YOLOv12-S, raw images, **100 epochs** | 🔲 |
| 2 | `002` | CLAHE+Gamma, **100 epochs** (OFAT ablation) | 🔲 |
| 3 | `003` | Genetic HPO, 50 iter × 50 epochs, fraction=0.5 | 🔲 |
| 4 | `004–006` | Final N/S/M, 200 epochs, HPO config | 🔲 |
| 5 | — | FP32/FP16/INT8 OpenVINO export | 🔲 |
| 6 | — | Latency/FPS benchmarking on i5 8th-gen | 🔲 |
| 7 | — | Frozen test evaluation | 🔲 |
