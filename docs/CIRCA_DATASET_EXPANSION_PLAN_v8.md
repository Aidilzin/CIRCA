# CIRCA — Dataset Expansion Plan (IPC-A-610)

> ⚠️ **[SUPERSEDED — HISTORICAL RECORD]** This document was written during the planning phase when a 15-class taxonomy was under consideration. The final implemented corpus uses a **12-class taxonomy** as documented in `class_mapping_v8.md`. Sections §3 and §7 of this document (taxonomy table and `data.yaml` snippet) reflect the abandoned 15-class design and should NOT be used as a reference. All other sections (motivation, acquisition pipeline, class imbalance strategy, citations) remain accurate and informative.

**Project:** CIRCA — PCB Defect Detection (YOLOv12)
**Date:** Thursday, April 30, 2026
**Subject:** Integration of SolDef_AI + PCB Solder Joint into the existing `unified_pcb` corpus to back the IPC-A-610 framing
**Author:** CIRCA project lead
**Status:** Implemented (12-class corpus; see `class_mapping_v8.md` for final taxonomy)

---

## 1. Motivation

The dataset is composed of three bare-board datasets — PKU-Market-PCB, DsPCBSD+ (Lv et al., 2024), and a Roboflow PCB Defect Dataset. All three target **bare PCB (copper-trace) defects** which technically fall under **IPC-A-600 — Acceptability of Printed Boards**.

Because the CIRCA thesis frames the work under IPC-A-610 (assembly stage, electronics repair context), the corpus also includes two **publicly available, citable, CC BY 4.0 assembly-stage datasets** that contribute genuine IPC-A-610 defect modes (solder defects).

Component-level defects (missing component, misaligned component, tombstoning, lifted lead) are **explicitly out of scope** — see §8.

---

## 2. New Source Datasets

### 2.1 SolDef_AI (Fontana et al., 2024)
- **Paper:** Fontana, Calabrese, et al. *“SolDef-AI: An Open Source PCB Dataset for Mask R-CNN Defect Detection in Through-Hole Pin Soldering.”* J. Manuf. Mater. Process. 8 (2024).
- **Kaggle:** [`mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection`](https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection)
- **Roboflow YOLO mirror:** [`defectdetection-e5sqy/soldef_ai-for-defect-detection`](https://universe.roboflow.com/defectdetection-e5sqy/soldef_ai-for-defect-detection)
- **License:** CC BY 4.0
- **Domain:** Through-hole solder joints, SMT components (3 viewpoints/sample)
- **Annotation:** Polygon (instance segmentation) → must be converted to YOLO bbox
- **Classes (5):** `good`, `exc_solder`, `poor_solder`, `spike`, `no_good`
- **Approx images:** ~1,150 (Kaggle source); Roboflow mirror exposes ~429 verified annotated images
- **CIRCA contribution:** `excess_solder`, `insufficient_solder`, `solder_spike` (3 new classes)

### 2.2 PCB Defect Detection (2025b–f) — emmts workspace — Roboflow Universe
- **URL:** [`work-6qkmv/pcb-solder-joint`](https://universe.roboflow.com/work-6qkmv/pcb-solder-joint)
- **License:** CC BY 4.0
- **Domain:** SMT solder joints (varied pad sizes)
- **Annotation:** Object detection (bbox)
- **Classes (3):** `Cold_solder`, `Excessive_solder`, `Insufficient_solder`
- **Approx images:** ~1,626 (per Roboflow page metadata)
- **CIRCA contribution:** `solder_bridge *(excluded — no suitable public dataset; see future work)*`, `cold_solder_joint` (2 new classes)

---

## 3. Proposed Unified 15-Class Taxonomy

| Unified ID | Class Name | IPC Reference | Source Datasets | Raw Class Name(s) Remapped From |
|:---|:---|:---|:---|:---|
| 0 | `missing_hole` | IPC-A-600 §3.4 | PKU/JR, Bare PCB | `missing_hole` |
| 1 | `mouse_bite` | IPC-A-600 §3.3 | PKU/JR, Bare PCB | `mouse_bite` |
| 2 | `open_circuit` | IPC-A-600 §3.2 | PKU/JR, Bare PCB | `open_circuit` |
| 3 | `short` | IPC-A-600 §3.2 | PKU/JR, Bare PCB, Hue | `short`, `Short`, `Shorted` |
| 4 | `spur` | IPC-A-600 §3.3 | PKU/JR, Bare PCB | `spur` |
| 5 | `spurious_copper` | IPC-A-600 §3.3 | PKU/JR, Bare PCB | `spurious_copper`, `falsecopper` |
| 6 | `excess_solder` | IPC-A-610H §5 | SolDef_AI, kydra, f8m5i, v2-s89jo | `exc_solder` (α-idx 0), `Excessive_solder`, `excess_solder` |
| 7 | `insufficient_solder` | IPC-A-610H §5 | SolDef_AI, kydra, Hue, v2-s89jo | `poor_solder` (α-idx 3), `Insufficient_solder`, `INSUFFICIENT SOLDER`, `Insufficient Solder`, `Missing_solder` |
| 8 | `solder_spike` | IPC-A-610H §5 | SolDef_AI | `spike` (α-idx 4) |
| 9 | `cold_solder_joint` | IPC-A-610H §5 | kydra, v2-s89jo | `Cold Solder`, `Cold_solder` |
| 10 | `scratch` | IPC-A-600 §3 | Bare PCB defects | `scratch` |
| 11 | `pinhole` | IPC-A-600 §3 | Bare PCB defects | `pinhole` |

**Hybrid note:** The corpus mixes IPC-A-600 (bare-board) and IPC-A-610 (assembly) defects. In the thesis, the framing is **“IPC-A-610 with auxiliary IPC-A-600 sub-board defects retained for completeness.”** Document this explicitly in Chapter 1 §1.5 (Scope).

### 3.1 Label Remapping Table

| Source | Source Class | Source ID | New Unified ID | Action |
|:---|:---|:---:|:---:|:---|
| SolDef_AI | `good`         | 0 | — | **DROP** (background — YOLO doesn’t need explicit “good” class) |
| SolDef_AI | `exc_solder`   | 0 (α-idx) | 6 | Remap |
| SolDef_AI | `good`         | 1 (α-idx) | — | Drop → negatives_reserve |
| SolDef_AI | `no_good`      | 2 (α-idx) | — | Drop → negatives_reserve |
| SolDef_AI | `poor_solder`  | 3 (α-idx) | 7 | Remap |
| SolDef_AI | `spike`        | 4 (α-idx) | 8 | Remap |
| SolDef_AI | `spike`        | 3 | 12 | Remap |
| SolDef_AI | `no_good`      | 4 | — | **DROP** (umbrella label — too ambiguous to use) |
| PCB Solder Joint | `GOOD`        | 0 | — | **DROP** (background) |
| v2-s89jo | `Cold_solder`         | — | 9  | Remap |
| v2-s89jo | `Excessive_solder`    | — | 6  | Remap |

**Rationale for dropping `good`/`GOOD`:** YOLO learns “good” regions implicitly as background. Including a literal “good” class would create class imbalance (it would dominate) and degrade detection of actual defects.

**Rationale for dropping SolDef_AI `no_good`:** It is an umbrella label that overlaps with `exc_solder`/`poor_solder`/`spike`. Keeping it would create label conflicts.

---

## 4. Integration Pipeline (Step-by-Step)

### 4.1 Acquisition
```bash
# SolDef_AI (Roboflow YOLOv8 export — most convenient)
mkdir -p /datasets/raw/soldef_ai
# Download via Roboflow API or browser export → unzip into /datasets/raw/soldef_ai/

# PCB Solder Joint
mkdir -p /datasets/raw/pcb_solder_joint
# Download Roboflow YOLOv8 export → unzip into /datasets/raw/pcb_solder_joint/
```

### 4.2 Polygon → BBox Conversion (SolDef_AI only)
SolDef_AI ships polygon (segmentation) labels. Convert each polygon to a YOLO axis-aligned bbox by taking min/max of x,y coordinates and normalising to image dims. Use Roboflow's "YOLOv8 (object detection)" export — it auto-converts. Otherwise:
```python
# scripts/poly_to_bbox.py — pseudo
for line in label_file:
    cls, *coords = line.split()
    xs, ys = coords[0::2], coords[1::2]
    x_c = (min(xs)+max(xs))/2; y_c = (min(ys)+max(ys))/2
    w = max(xs)-min(xs); h = max(ys)-min(ys)
    yolo_line = f"{cls} {x_c} {y_c} {w} {h}"
```

### 4.3 Class Remapping
```python
# scripts/remap_labels.py — pseudo
SOLDEF_MAP = {1: 10, 2: 11, 3: 12}      # drop 0 (good) and 4 (no_good)
PSJ_MAP    = {1: 14, 2: 13}              # drop 0 (GOOD)
# For each label .txt, rewrite class IDs using the appropriate map; drop lines whose class is not in map.
# Keep image only if at least one defect bbox remains; otherwise move to /pure_background/ (optional negative samples — see §6).
```

### 4.4 Sanitisation
Apply the **same perceptual-hash dedup pipeline** already used on the original corpus:
- Compute pHash for every new image
- Remove near-duplicates within the new sources (dHash distance ≤ 6)
- Cross-check pHashes against the existing `unified_pcb` set (defensive — should be zero overlap since domains differ)
- Verify image integrity (PIL open, dimension sanity ≥ 320 px on shorter side)
- Verify each label file is non-empty and class IDs ∈ {10..14}

### 4.5 Stratified Train/Val/Test Split (70/15/15)
- Concatenate the new SolDef_AI + PCB Solder Joint images into the master pool
- Run a **stratified split by class label** with `seed=42` to preserve per-class proportions across train/val/test
- Use `scikit-multilearn` `iterative_train_test_split` if labels are multi-label (one image with multiple defects), otherwise stratified-shuffle on the dominant class

### 4.6 Actual Final Counts (after global dedup + drop of `good`/`no_good` images)

*Note: A rigorous global deduplication pass (dHash ≤ 6) was applied across ALL five datasets to eliminate augmented duplicates inherited from the source Roboflow exports, preventing severe data leakage. This reduced the raw inflated counts to a clean, mathematically rigorous base.*

| Split | Base Image Count | After ≤3× Solder-Only Oversampling (conditional) | Percentage |
|:---|---:|---:|---:|
| Training   | 2,317  | 5,579 | 70.0% |
| Validation | 496    | 496   | 15.0% |
| Test       | 497    | 497   | 15.0% |
| **Total**  | **3,310** | **6,572** | **100.0%** |

**Note:** The previous 10× bare-board oversampling has been removed. Aggressive oversampling of the majority class causes overfitting on repeated pixel patterns and conflicts with Ultralytics' built-in Mosaic and Copy-Paste augmentation pipeline (Cao et al., 2024; Ultralytics, 2023). The revised strategy prioritises (1) `cls_pw` inverse-frequency loss weighting as the primary mitigation, (2) `mosaic=1.0` and `copy_paste=0.3` augmentation for diversity, and (3) conditional ≤3× oversampling of *solder classes only* (IDs 6–9) if any solder class records mAP@0.5 < 70% after Phase 2.

---

## 5. Class Imbalance Strategy

After expansion, class distribution will skew toward bare-board defects (~5:1 ratio). Mitigations:
1. **Class-weighted sampling** during training — Ultralytics YOLOv12 supports `WeightedRandomSampler` via custom `train.py`, or use `image_weights=True` flag in the YOLO YAML (uses inverse class frequency).
2. **Conditional ≤3× oversampling of solder classes only** (IDs 6–9) in the training split — triggered only if any solder class mAP@0.5 < 70% after Phase 2. Bare-board classes are never oversampled.
3. **Per-class mAP reporting** — track mAP@0.5 per class in W&B and flag any class with <70% as needing more data.
4. **Loss weighting** (last resort) — scale solder-class CE loss × 1.5 in `data.yaml` if per-class mAP gap persists after epochs 50.

**Implemented strategy:** (1) `cls_pw` inverse-frequency loss weighting (primary), (2) `mosaic=1.0` + `copy_paste=0.3` augmentation (always active), (3) conditional ≤3× solder-only oversampling if Phase 2 solder mAP@0.5 < 70%. 10× bare-board oversampling removed — overfitting risk confirmed by Cao et al. (2024).

---

## 6. Background / Negative Samples (Optional)

The dropped `good`/`GOOD` images are valid **negative samples** (true PCBs with no defects). Adding ~10% of training set as negatives can reduce false positives:
- Keep up to **800 negative images** (~10% of 7,089-train) from the dropped pool
- Place them in the train split with empty `.txt` label files
- YOLO treats them as background-only

**Decision:** Dropped `good`/`GOOD` images have been archived in `negatives_reserve/` for potential later use. They will only be included if the Phase 1 baseline shows excessive false positives on the validation set.

---

## 7. data.yaml Update

```yaml
# /datasets/unified_pcb_v2/data.yaml
path: /datasets/unified_pcb_v2
train: images/train
val:   images/val
test:  images/test

nc: 15
names:
  0:  missing_hole
  1:  mouse_bite
  2:  open_circuit
  3:  short
  4:  spur
  5:  spurious_copper
  6:  hole_breakout
  7:  conductor_scratch
  8:  conductor_foreign_object
  9:  base_material_foreign_object
  10: excess_solder
  11: insufficient_solder
  12: solder_spike
  13: solder_bridge *(excluded — no suitable public dataset; see future work)*
  14: cold_solder_joint
```

---

## 8. Out-of-Scope Defects (Documented Limitation)

The following IPC-A-610 defect categories are **explicitly out of scope** for the CIRCA thesis due to the absence of suitable public datasets:

| Out-of-scope Category | IPC-A-610 Section | Reason for Exclusion |
|:---|:---|:---|
| Missing component (DNP error) | §8 (SMT) | No public dataset; manufacturers protect BOM/IP |
| Misaligned component | §8 (SMT) | Defective boards are reworked, not photographed |
| Tombstoning | §8 (SMT) | High variance per component package; no public set |
| Lifted lead / lead-pull | §6 (THT) | Owned by AOI vendors (Koh Young, Omron, TRI) |
| Solder ball | §5 | Annotation cost prohibitive; no public set |
| Component damage (cracked chip) | §10 | Requires destructive testing dataset; not public |

This limitation will be documented in:
- **Chapter 1 §1.5 (Scope and Limitations)**
- **Chapter 3 §3.1 (Dataset and Data Acquisition)**
- **Chapter 5 (Recommendations / Future Work)** — propose custom data collection campaign in subsequent thesis or industry partnership

---

## 9. Impact on Existing Phases

### Phase 1 (Baseline Vanilla) — `001`
- **Action:** run on 15-class `unified_pcb_v2`.

### Phase 2 (CIRCA Baseline) — `002`
- **Action:** run on 15-class `unified_pcb_v2_preproc`.

### Phase 3 (Genetic HPO) — `003`
- **Action:** run on 15-class corpus.

### Phase 4 (Final Thesis Models) — `004`/`005`/`006` (N, S, M variants)
- **Action:** unchanged — runs on whatever HPO yields.

**Compute estimate (RTX 3060 6GB):**
| Phase | Old runtime | New runtime (15-class, ~11.8k imgs) |
|:---|---:|---:|
| Phase 1 (50 ep) | ~6 h | ~7 h |
| Phase 2 (100 ep) | ~12 h | ~14 h |
| Phase 3 (50 × 30 ep) | ~1.5 days | ~2 days |
| Phase 4 (3 variants × 200 ep) | ~3 days | ~3.5 days |
| **Total** | ~7 days | **~8.5 days** |

---

## 10. Citations to Add (Thesis Bibliography)

1. **SolDef_AI** — Fontana, M., Calabrese, M., et al. (2024). *SolDef-AI: An Open Source PCB Dataset for Mask R-CNN Defect Detection in Through-Hole Pin Soldering.* J. Manuf. Mater. Process., 8(4), 145. https://doi.org/10.3390/jmmp8040145
2. **PCB Defect Detection (2025b–f) — emmts workspace** — PCB Defect Detection. (2025a). *PCB Solder Joint Dataset* [Dataset]. Roboflow Universe. https://universe.roboflow.com/work-6qkmv/pcb-solder-joint (CC BY 4.0).
3. **IPC-A-610H** — IPC International. (2020). *IPC-A-610H: Acceptability of Electronic Assemblies.* Bannockburn, IL: IPC.
4. **IPC-A-600K** — IPC International. (2020). *IPC-A-600K: Acceptability of Printed Boards.* Bannockburn, IL: IPC. *(retain because the carried bare-board classes 0–9 reference this standard)*

---

## 11. File Update Checklist (Implemented)

This plan has been fully approved and the corresponding documentation artifacts have been updated:
- [x] `PCB_DEFECT_HYPERPARAMETER_TUNING_v8.md` — updated §1.2 taxonomy (10 → 12 classes), `data.yaml` snippet, added SolDef_AI + PCB Solder Joint to source list
- [x] `CIRCA_EXPERIMENT_PLAN_v8.md` — updated Phase 0 dataset section, added re-run note to Phase 1/2/3, updated compute estimate
- [x] `CIRCA_EXPERIMENT_CHECKLIST_v8.md` — added Phase 0 sub-tasks (download, remap, sanitise, split), flagged Phase 1/2/3 as re-run
- [x] `DATASET_PREPROCESSING_REPORT_v8.md` — updated §2 sources list, §3.1 remapping table (10 → 12 classes), §4 final counts table
- [x] `CIRCA_CHAPTER_3_4_OUTLINE_v8.md` — updated Ch3 §3.1 sources, added §1.5 scope limitation reference, added citations
- [x] `class_mapping_v8.md` — verified mapping document reflecting the 12 unified classes


## Dataset Sources

| # | Dataset | Slug / Source | Images | Raw Classes Used | Unified IDs | License |
|:---|:---|:---|:---|:---|:---|:---|
| 1 | **SolDef_AI** (Fontana et al., 2024) | Kaggle: `mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection` | ~1,150 | `exc_solder` (α-idx 0), `poor_solder` (α-idx 3), `spike` (α-idx 4) | 6, 7, 8 | CC BY 4.0 |
| 2 | **PKU-Market-PCB via JR** | `jr-mqqnk/pcb-defects-detection-anddl` | ~1,500 | 6 standard bare-board classes | 0–5 | Public Domain |
| 3 | **RAHUL PCB Defects** | `rahul-jhj03/pcb-defects-dataset` | Unknown | 6 standard bare-board classes | 0–5 | CC BY 4.0 |
| 4 | **Bare PCB Defects** | `bare-pcb-defects/obj-detection-pcb-defects-yolov8` | ~9,666 | `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `falsecopper`, `scratch`, `pinhole` | 0–5, 10, 11 | CC BY 4.0 |
| 5 | **Hue** (PCB Defect Detection, 2025b) | `emmts/hue-dbgbs-reqtv` | 3,232 | `Insufficient Solder`, `Shorted` | 7, 3 | CC BY 4.0 |
| 6 | **solder-f8m5i-xnbzp** (PCB Defect Detection, 2025c) | `emmts/solder-f8m5i-xnbzp` | Unknown | `Excessive_solder` | 6 | CC BY 4.0 |
| 7 | **Excessive Solder / kydra** (PCB Defect Detection, 2025e) | `emmts/excessive-solder-kydra` | 1,162 | `Cold Solder`, `Excessive_solder`, `Insufficient_solder` | 9, 6, 7 | CC BY 4.0 |
| 8 | **PCB Solder Defect Detection V2** (PCB Defect Detection, 2025f) | `emmts/pcb-solder-defect-detection-v2-s89jo` | 6,116 | `Cold_solder`, `Excessive_solder`, `Insufficient_solder` | 9, 6, 7 | CC BY 4.0 |

### Class Remapping Notes

**Classes dropped during remapping:**
- `good` / `GOOD` (SolDef_AI): YOLO learns non-defective regions implicitly as background.
- `no_good` (SolDef_AI): umbrella label overlapping with IDs 6, 7, 8 — creates label conflicts.
- `missing_component` (Hue): IPC-A-610 component-level defect excluded per Chapter 1 §1.5.
- `solder_bridge`: No clean board-level annotated public dataset found across all 8 sources. Documented as future work in Chapter 5.

**Dropped datasets (quality audit):**
- `sthr7` (`pcb-deffect-detection-solder-sthr7`): Dropped — zero `solder_spike` instances despite documentation, and high annotation noise.
- `lsb7m` (`pcb-deffect-detection-solder-lsb7m`): Dropped — single-class (`Excessive_solder` only), fully superseded by kydra, f8m5i, and v2-s89jo.

**RAHUL dataset deduplication:** perceptual-hash (pHash, Hamming ≤ 6) cross-check against PKU/JR required before final merge.
