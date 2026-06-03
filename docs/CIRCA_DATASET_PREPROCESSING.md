# Technical Report: Dataset Unification and Preprocessing
**Project:** CIRCA — PCB Defect Detection System
**Date (v1):** 2026-04-29 | **Date (v2):** 2026-04-30 | **Date (v3 rebuild):** 2026-05-22 | **Date (v4 tiers + capping):** 2026-05-27
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

Actual post-rebuild counts (v4, 2026-05-27):

| Split | Images | Notes |
|:---|:---:|:---|
| Train (originals) | 5,924 | Stratified on dominant class per image |
| Train (after cap + oversample) | ~7,800–8,000 | After Step 1.5 cap and Step 2 oversample (exact count from pipeline) |
| Val | 1,269 | HPO / early stopping |
| Test | 1,270 | **Frozen before any training** |

> **Val split note:** `unified_pcb_v3` uses `valid/images`; `unified_pcb_v3_preproc` normalises to `val/images`. Both point to the same 1,269 image set.

### 3.3 Dominant-Class Capping (Step 1.5) — Added 2026-05-27

After the Phase 1 & 2 baseline runs revealed a 9.9:1 annotation imbalance (dominant `insufficient_solder` vs minority `cold_solder_joint`), a capping step was added to the pipeline:

- **Eligible images:** those whose annotations consist **exclusively** of dominant-class labels (`short`=3, `insufficient_solder`=5). These 3,468 images contain zero minority-class signal.
- **Cap:** 1,000 dominant-only images kept (randomly sampled, `seed=42`); remaining 2,468 removed.
- **Effect:** annotation ratio reduced from **9.9:1 → ~5.7:1** (well within the ≤ 10:1 recommended threshold).
- **Script:** `scripts/cap_dominant_classes.py --cap 1000 --seed 42`
- **No minority-class signal lost** — all 4,364 mixed images (containing at least one minority annotation) are preserved.

### 3.4 Oversampling Tiers (v4) — Updated 2026-05-27

Minority-class oversampling is applied **after** capping, based on confusion matrix evidence from Phase 1 & 2 runs:

| Class | Recall (Phase 1) | Old Tier | New Tier (v4) | Justification |
|:---|:---:|:---:|:---:|:---|
| `missing_hole` (0) | **0%** | 3× moderate | **5× critical** | 0% recall — model was blind to this class |
| `excess_solder` (4) | **51%** | None (not oversampled) | **5× critical** | 304 originals, 51% recall — worst-performing non-background class |
| `cold_solder_joint` (6) | 96% | 5× critical | **5× critical** | Maintained — working well |
| `short` (3) | 95% | EXCLUDED | EXCLUDED | Dominant |
| `insufficient_solder` (5) | 95% | EXCLUDED | EXCLUDED | Dominant |

### 3.3 Polygon → Bounding Box Conversion (SolDef_AI)

SolDef_AI ships with polygon (segmentation) annotations. When exported from Roboflow using the **"YOLOv8 Object Detection"** format (not Instance Segmentation), Roboflow automatically converts each polygon to its axis-aligned bounding rectangle in normalised YOLO format. No custom conversion script is required. This ensures bounding boxes are mathematically consistent with any future researcher using the same Roboflow export.

### 3.5 Image Enhancement Pipeline (CLAHE + Gamma)
- **CLAHE:** L-channel LAB, Clip Limit 2.0, Tile 8×8.
- **Gamma Correction:** γ = 1.2.
- **Output:** Cached offline to `unified_pcb_v3_preproc/` via `scripts/prepare_all_datasets.py` (Step 3). No longer generated on-the-fly during Phase 2 training — both datasets are fully prepared before uploading to RunPod.

### 3.6 Integrity Guardrails
- pHash deduplication: dHash ≤ 6 Hamming distance, applied globally across all sources.
- Missing-label detection with explicit `WARNING` logging.
- `seed=42` enforced globally across all phases and in the capping random sampler.
- Test split frozen before Phase 1 begins.
- Oversampling idempotency: `_os*` files are excluded as source material to prevent chain duplication (`_os1_os1` pattern) — fixed 2026-05-27.
- **Dataset is now frozen after v4 rebuild.** No further structural changes before Phase 4 final training.

---

## 4. Full Dataset Preparation Pipeline

Run via `python scripts/prepare_all_datasets.py` from project root. All steps are automated.

| Step | Script | Description | Status |
|:---|:---|:---|:---:|
| **1** | `build_unified_pcb_v3.py` | Merge, remap, dedup (dHash≤6), 70/15/15 split → `unified_pcb_v3/` | ✅ Done |
| **1.5** | `cap_dominant_classes.py --cap 1000` | Remove 2,468 dominant-only images; ratio 9.9:1 → 5.7:1 | ✅ Done |
| **2** | `oversample_minority_classes.py` | 5× for classes 0, 4, 6 (v4 tiers) | ✅ Done |
| **3** | `train_engine.preprocess_dataset()` | CLAHE+Gamma on all train/val/test → `unified_pcb_v3_preproc/` | ✅ Done |
| **4** | `oversample_minority_classes.py` | Oversample preproc dataset (inherits from step 2 — 0 copies written) | ✅ Done |

## 5. Experimental Roadmap

| Phase | ID | Description | Status |
|:---|:---|:---|:---:|
| 0 | — | Full pipeline (steps 1–4 above) | ✅ Done |
| 1 | `001` | Vanilla YOLOv12-S, raw images, **100 epochs** | ⏳ Re-run needed (dataset changed) |
| 2 | `002` | CLAHE+Gamma, **100 epochs** (OFAT ablation) | ⏳ Re-run needed (dataset changed) |
| 3 | `003` | Genetic HPO, 50 iter × 50 epochs, fraction=0.5 | 🔲 Pending |
| 4 | `004–006` | Final N/S/M, 200 epochs, HPO config | 🔲 Pending |
| 5 | — | FP32/FP16/INT8 OpenVINO export | 🔲 Pending |
| 6 | — | Latency/FPS benchmarking on i5 8th-gen | 🔲 Pending |
| 7 | — | Frozen test evaluation | 🔲 Pending |
