# Technical Report: Dataset Unification and Preprocessing
**Project:** CIRCA — PCB Defect Detection System  
**Date (v1):** Wednesday, April 29, 2026  
**Date (v2 expansion):** Thursday, April 30, 2026  
**Subject:** Integration of Multi-Source PCB Defect Datasets for YOLOv12 Training

---

## 1. Executive Summary
To enhance the model's robustness and categorical depth for the CIRCA project, **five** independent datasets are unified into a single master dataset (`unified_pcb_v2`).

- **v1 (completed):** Three bare-board datasets unified, expanding defect coverage from 6 to **10** classes (IPC-A-600 territory). Final corpus: 10,128 sanitised images.
- **v2 (in progress):** Two assembly-stage solder-defect datasets added to back the IPC-A-610 framing of the thesis. Taxonomy expands from 10 to **15** classes (adding excess solder, insufficient solder, solder spike, solder bridge, cold solder joint). Estimated final corpus: ~11,778 images.

This report documents the technical decisions, stability optimisations, and dataset balancing strategies implemented to ensure academic rigor and model performance.

## 2. Source Datasets

### 2.1 Bare-board sources (v1, IPC-A-600 territory — retained for completeness)
1.  **PKU-Market-PCB-ver1:** (2,910 images) High-quality laboratory samples focused on 6 fundamental defects.
2.  **DsPCBSD+ (Lv et al., 2024):** (7,367 images) Industrial-grade data adding 4 critical structural defect categories.
3.  **PCB Defect Dataset (Roboflow):** (8,001 images) High variance in lighting conditions and camera angles.

### 2.2 Assembly-stage sources (v2, IPC-A-610 — newly added)
4.  **SolDef_AI** (Fontana et al., 2024) — Roboflow Universe `defectdetection-e5sqy/soldef_ai-for-defect-detection`. ~1,150 SMT/THT images, polygon annotations (converted to YOLO bbox via Roboflow object-detection export). License: CC BY 4.0. Citable paper: *J. Manuf. Mater. Process.* 8(4), 145.
5.  **PCB Solder Joint** (Work, 2025) — Roboflow Universe `work-6qkmv/pcb-solder-joint`. ~1,626 SMT solder-joint images with bbox annotations. License: CC BY 4.0.

> **Out-of-scope (documented limitation):** Component-level IPC-A-610 defects (missing component, misalignment, tombstoning, lifted lead, solder ball, component damage) are excluded due to absence of suitable public datasets. Recorded in thesis Chapter 1 §1.5 and Chapter 3 §3.1, with a future-work note recommending a custom data-collection campaign.

---

## 3. Applied Preprocessing & Methodology (Thesis Chapter 3 Support)

### 3.1 Categorical Standardization (Class Re-mapping)
The primary challenge in unification was "Class ID Conflict." A master **15-class** ontology was established to normalise every label file (`.txt`) to a unified standard. Classes 0–9 derive from bare-board sources (IPC-A-600 alignment), and classes 10–14 from assembly-stage sources (IPC-A-610 alignment).

| Unified ID | Class Name | IPC Reference | Source Dataset(s) |
| :---: | :--- | :--- | :--- |
| 0  | `missing_hole`                 | IPC-A-600 | All bare-board sources |
| 1  | `mouse_bite`                   | IPC-A-600 | All bare-board sources |
| 2  | `open_circuit`                 | IPC-A-600 | All bare-board sources |
| 3  | `short`                        | IPC-A-600 | All bare-board sources |
| 4  | `spur`                         | IPC-A-600 | All bare-board sources |
| 5  | `spurious_copper`              | IPC-A-600 | All bare-board sources |
| 6  | `hole_breakout`                | IPC-A-600 | DsPCBSD+ |
| 7  | `conductor_scratch`            | IPC-A-600 | DsPCBSD+ |
| 8  | `conductor_foreign_object`     | IPC-A-600 | DsPCBSD+ |
| 9  | `base_material_foreign_object` | IPC-A-600 | DsPCBSD+ |
| **10** | **`excess_solder`**        | **IPC-A-610 §5** | SolDef_AI (`exc_solder`) |
| **11** | **`insufficient_solder`**  | **IPC-A-610 §5** | SolDef_AI (`poor_solder`) |
| **12** | **`solder_spike`**         | **IPC-A-610 §5** | SolDef_AI (`spike`) |
| **13** | **`solder_bridge`**        | **IPC-A-610 §5** | PCB Solder Joint (`STICKSOLDER`) |
| **14** | **`cold_solder_joint`**    | **IPC-A-610 §5** | PCB Solder Joint (`COLDSOLDER`) |

**v2 remap rules:**
- SolDef_AI: drop `good`(0) and `no_good`(4) (umbrella label); remap `exc_solder`(1)→10, `poor_solder`(2)→11, `spike`(3)→12.
- PCB Solder Joint: drop `GOOD`(0); remap `STICKSOLDER`(2)→13, `COLDSOLDER`(1)→14.
- Dropped `good`/`GOOD` images archived in `negatives_reserve/` for potential use as background samples (deferred per the integration plan).
- SolDef_AI polygon annotations are converted to YOLO axis-aligned bboxes via the Roboflow YOLOv8 object-detection export.

### 3.2 Dataset Rebalancing (70/15/15 Split)
To align with academic standards and ensure reliable validation/testing, the dataset was rebalanced from its raw distribution to a strict **70/15/15** ratio.
*   **Method:** Stratified randomized shuffle using a fixed seed (`seed=42`) to ensure reproducibility.
*   **Rationale:** Increasing the Test set from 9% to 15% (2,805 images) provides a more statistically significant evaluation of the model's generalization capability on unseen data.

### 3.3 Training Stability Patches (Numerical Robustness)
During initial baseline runs, numerical instability (EMA NaN/Inf warnings) was observed. The following methodology changes were implemented:
*   **Learning Rate Reduction:** `lr0` reduced from `0.01` to `0.001`.
*   **Extended Warmup:** `warmup_epochs` increased to `5.0`.
*   **Scientific Rationale:** These patches prevent gradient explosion as the model exits the warmup phase, particularly important for the YOLOv12 architecture's Area Attention modules when trained on mixed-source datasets.

### 3.4 Automated Image Enhancement Pipeline (Thesis-Aligned)
To address uncontrolled lighting conditions in repair environments, a standardized OpenCV-based enhancement pipeline was integrated directly into the training engine:
*   **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Applied to the L-channel of the LAB color space (Clip Limit: 2.0, Tile: 8x8) to normalize contrast across various PCB substrates.
*   **Gamma Correction:** Applied with a constant factor of **1.2** to lift mid-tone shadows without washing out highlight details.
*   **Multi-Format Support:** The engine was expanded to support `.jpg`, `.jpeg`, `.png`, and `.bmp` formats, ensuring full coverage of industrial dataset sources.
*   **Offline Processing Strategy:** Preprocessing is performed once as a standalone pass, generating a `unified_pcb_preproc` dataset. This eliminates the per-epoch CPU overhead and ensures deterministic input for both training and inference.

### 3.5 Data Integrity Guardrails
To maintain academic rigor and prevent "silent data loss," two critical guardrails were implemented:
*   **Missing Label Detection:** The preprocessing engine logs an explicit `WARNING` for any image missing a corresponding YOLO label file. This prevents the model from incorrectly learning defect-rich regions as "background" due to annotation omissions.
*   **Deterministic Seeding:** `seed=42` is enforced globally across training and tuning to ensure that performance gains are attributable to architectural or hyperparameter changes rather than stochastic variance.

---

## 4. Final Dataset Composition (Thesis Chapter 4 Support)

### 4.1 v1 corpus (`unified_pcb`, 10 classes — historical reference)

| Split | Image Count | Percentage | Primary Research Purpose |
| :--- | ---: | ---: | :--- |
| Training   | 7,089  | 70.0% | Weight convergence and feature learning |
| Validation | 1,519  | 15.0% | Hyperparameter Tuning (HPO) and early stopping |
| Test       | 1,520  | 15.0% | Final academic performance evaluation |
| **Total**  | **10,128** | **100.0%** | |

### 4.2 v2 corpus (`unified_pcb_v2`, 15 classes — estimated; update with actuals after sanitisation)

| Source | Pre-merge images | Post-sanitise (est.) |
| :--- | ---: | ---: |
| v1 carried (PKU + DsPCBSD+ + Roboflow) | 10,128 | 10,128 |
| SolDef_AI (defect-bearing only)        | ~750   | ~700 |
| PCB Solder Joint (defect-bearing only) | ~1,000 | ~950 |
| **v2 unified total**                   |        | **~11,778** |

### 4.3 Class-imbalance strategy (v2)
- Default: `cls_pw=1.0` during training to weight inverse-frequency by class.
- If any solder class (10–14) shows mAP@0.5 < 70% after Phase 2, oversample those classes 3× in the train split (only) or apply per-class loss weighting via `cls_pw`.
- Per-class mAP tracked in W&B for the entire training programme.

---

## 5. Experimental Design (Thesis Roadmap)
To satisfy the objectives of the CIRCA project, a four-stage experimental sequence is implemented. Each stage is designed to isolate the impact of specific technical interventions.

| Stage | ID | Description | Primary Goal |
| :--- | :--- | :--- | :--- |
| **1. Vanilla Control** | `001` | Default YOLOv12s on raw data. | Establish the absolute baseline performance. |
| **2. CIRCA Baseline** | `002` | Preprocessed data (CLAHE + Gamma). | Measure mAP gain from the OpenCV pipeline. |
| **3. Genetic HPO** | `003` | 150-iteration Hyperparameter Tuning. | Discover optimal weights/augmentations for PCBs. |
| **4. Final Thesis Model** | `004` | 200-epoch training with HPO config. | Produce the final model for OpenVINO deployment. |

## 6. Execution & Hardware Optimization
To maximize throughput on the NVIDIA RTX 3060 (6GB VRAM) laptop hardware while maintaining system stability:
*   **Batch Size:** Set to `12` (Pushing VRAM utilization to ~75-80% for speed without OOM risk).
*   **Data Loading:** `workers=4` with a multiprocessing pool for preprocessing.
*   **Stability Patch:** `lr0=0.001` and `warmup=5` (applied to non-HPO runs).
*   **Augmentation Guardrail:** `close_mosaic=10` to disable mosaic augmentation in final epochs for precise bounding box refinement.

## 7. Conclusion
The **`unified_pcb`** dataset and the accompanying **Stable Training Engine** provide a robust, scientifically valid foundation for the CIRCA project. These decisions ensure that the final model results are not only performant but also academically reproducible and statistically sound.
