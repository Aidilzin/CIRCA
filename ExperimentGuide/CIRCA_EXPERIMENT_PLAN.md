# CIRCA — Experiment Plan Outline

> **Project:** CIRCA — Circuit Inspection and Recognition using Convolutional Architectures
> **Model:** YOLOv12 (Ultralytics)
> **Deployment:** Intel OpenVINO INT8 IR on Intel CPU + iGPU
> **Aligned with:** Thesis Chapter 1 §1.4 Objectives, Chapter 1 §1.5 Acceptance Criteria, Playbook v2

---

## 1. Objectives

This experiment programme is designed to satisfy the three research objectives stated in Chapter 1 §1.4:

1. Identify and document IPC-A-610-aligned PCB defect types for automated detection.
2. Design, train, and comparatively evaluate **YOLOv12-N / S / M** under repair-context conditions, with **Intel OpenVINO INT8** export.
3. Evaluate the CIRCA preprocessing pipeline (CLAHE + Gamma + Laplacian Variance) and the resulting desktop application using precision, recall, F1, mAP, and inference latency.

---

## 2. Acceptance Criteria

A configuration is **only declared "final"** when it passes all four targets simultaneously:

| Metric | Target | Measured On |
|---|---|---|
| mAP@0.5 (test set) | **> 90%** | Curated PCB test set |
| Preprocessing latency | **≤ 5 ms / frame** | Intel Core i5 8th-gen CPU |
| Live inference rate | **≥ 15 FPS** | Webcam end-to-end |
| Static image inference | **≤ 10 s** | Single image, full pipeline |

---

## 3. Experimental Phases

The programme is organised into **six phases**. Each phase has explicit entry conditions, deliverables, and exit criteria. Do not advance to the next phase until the current one is signed off.

### Phase 0 — Environment & Dataset Setup
**Goal:** Establish a reproducible training environment and an IPC-A-610-aligned dataset.
- Install Ultralytics, OpenVINO, NNCF, OpenCV, W&B.
- **Acquire five source datasets:**
  - Bare-board (carried from v1): PKU-Market-PCB-ver1, DsPCBSD+ (Lv et al., 2024), Roboflow PCB Defect Dataset.
  - Assembly-stage (v2 expansion): SolDef_AI (Fontana et al., 2024) and PCB Solder Joint (Work, 2025) — both CC BY 4.0.
- **Convert SolDef_AI polygons to YOLO bbox** (use Roboflow YOLOv8 object-detection export to auto-handle).
- **Remap labels** to the unified 15-class IPC taxonomy:
  - SolDef_AI: drop `good`/`no_good`; remap `exc_solder`→10, `poor_solder`→11, `spike`→12.
  - PCB Solder Joint: drop `GOOD`; remap `STICKSOLDER`→13, `COLDSOLDER`→14.
- **Sanitise** with the same perceptual-hash dedup pipeline (dHash distance ≤ 6) used on v1; cross-check against v1 hashes.
- **Stratified 70/15/15 split** with `seed=42` preserving per-class proportions; freeze the test set.
- **Hold dropped `good`/`GOOD` images in reserve** as potential negative samples (deferred per the integration plan).
- **Estimated final corpus size:** ~11,778 images (10,128 carried + ~1,650 new defect-bearing).
- See `CIRCA_DATASET_EXPANSION_PLAN.md` for the full step-by-step integration plan and `class_mapping.md` for the canonical remap table.

### Phase 1 — Vanilla Baseline (Ablation Control)
**Goal:** Quantify the lift from CIRCA preprocessing.
- Train **YOLOv12-S** on raw images, no preprocessing.
- **50 epochs** (kept short — this run is only used as the ablation control in Chapter 4). 
- **Class Weighting:** Uses `cls_pw=1.0` to compensate for the bare-board/solder imbalance and the addition of 1,623 background images via inverse-frequency power weighting.

> **Important:** the v1 baseline was trained on a 10-class corpus and is no longer comparable to the v2 (15-class) CIRCA model. **Phase 1 must be re-run** on `unified_pcb_v2` once Phase 0 completes.
>
> **Negative-sample decision:** initial Phase 1 run uses the standard training set only. If validation reveals excessive false positives on background regions, the held-back `good`/`GOOD` images will be re-introduced as ~10% background samples and the phase will be repeated. Documented as a contingent decision in the integration plan.

### Phase 2 — CIRCA-Aligned Baseline
**Goal:** Establish the primary baseline on the preprocessed dataset.
- Train YOLOv12-S on CLAHE+Gamma preprocessed images of the **15-class** `unified_pcb_v2` corpus.
- **100 epochs** — confirms the preprocessing pipeline is wired correctly end-to-end and gives HPO a realistic target to beat.
- **Class-imbalance:** train with `cls_pw=1.0` to compensate for the bare-board (≈10×) dominance over solder classes 10–14.
- **Re-run required** on the v2 corpus.

### Phase 3 — Hyperparameter Optimisation (HPO)
**Goal:** Find optimal hyperparameters using Ultralytics' genetic tuner.
- Run on YOLOv12-S over the preprocessed **15-class `unified_pcb_v2`** dataset.
- **150 iterations × 30 epochs each** = 150 independent mini-trainings, each with a different sampled hyperparameter combination. The tuner uses a genetic algorithm to mutate the top performers across iterations, then returns the single best combination.
- Output: `best_hyperparameters.yaml` (fed into Phase 4 via `--cfg`).
- **Re-run required** on the v2 corpus — HPO must be tuned against the final taxonomy, not a 10-class precursor.

> **Note on HPO methodology:** Phase 3 is not a single training run. Ultralytics' `model.tune()` executes the full multi-test sweep internally — sampling, training, evaluating, and mutating the search space across all 150 iterations. The alternative approaches (grid search, random search) would require a manual loop; the genetic tuner handles this automatically and is more sample-efficient than random search. See `PCB_DEFECT_HYPERPARAMETER_TUNING.md` §2–3 for the full search space.

### Phase 4 — Three-Variant Final Training
**Goal:** Produce the final candidate models for the comparative study (Objective 2).
- Train **YOLOv12-N**, **YOLOv12-S**, **YOLOv12-M** with the HPO-tuned config.
- 200 epochs with cosine LR, AMP, EMA, `close_mosaic=10`.
- Output: three `best.pt` checkpoints + three OpenVINO INT8 IR exports.

### Phase 5 — OpenVINO Quantisation Validation
**Goal:** Verify INT8 vs FP32 vs FP16 mAP and apply the fallback decision rule.
- Evaluate every variant in FP32, FP16, and INT8 on the validation set.
- If INT8 mAP@0.5 < 90%, fall back to FP16.
- Output: `quantization_report.md`.

### Phase 6 — Hardware Benchmarking & Variant Selection
**Goal:** Select the production variant against all four acceptance criteria.
- Measure preprocessing latency, inference latency, live FPS, static seconds on Intel Core i5 8th-gen.
- Fill the Variant Selection Matrix.
- Output: `benchmark_report.md`.

### Phase 7 — Final Test Evaluation & Confidence Calibration
**Goal:** One-time test-set evaluation + automation-bias mitigation thresholds.
- Run `model.val(split="test")` once on the chosen variant + precision.
- Sweep confidence thresholds on val to set per-class display + warning cutoffs.
- Output: `test_evaluation.md` + `circa_thresholds.yaml`.

---

## 4. Run Commands (Reference)

```bash
# Phase 1 — Vanilla control (one-off, ablation only)
python train_engine.py --mode train --variant s --id 001 --desc Baseline_Vanilla --epochs 50 --clear

# Phase 2 — CIRCA-aligned baseline
python train_engine.py --mode train --variant s --id 002 --desc Baseline_CIRCA --epochs 100 --preproc

# Phase 3 — HPO (150 iterations × 30 epochs each = 150 mini-trainings)
python train_engine.py --mode tune --variant s --id 003 --desc HPO --epochs 30 --iterations 150 --preproc

# Phase 4 — Final training, three variants, HPO config
python train_engine.py --mode train --variant n --id 004 --desc Final_HPO --epochs 200 --preproc \
    --cfg runs/detect/CIRCA_V12S_003_TUNE_HPO/best_hyperparameters.yaml

python train_engine.py --mode train --variant s --id 005 --desc Final_HPO --epochs 200 --preproc \
    --cfg runs/detect/CIRCA_V12S_003_TUNE_HPO/best_hyperparameters.yaml

python train_engine.py --mode train --variant m --id 006 --desc Final_HPO --epochs 200 --preproc \
    --cfg runs/detect/CIRCA_V12S_003_TUNE_HPO/best_hyperparameters.yaml
```

Phases 5–7 use separate evaluation/benchmark scripts (`evaluate_quantization.py`, `benchmark.py`, `calibrate_thresholds.py`) — to be authored in their respective phases.

---

## 5. Deliverables Map

| Phase | Deliverable | File |
|---|---|---|
| 0 | Class mapping document | `class_mapping.md` |
| 1 | Vanilla baseline weights | `runs/detect/CIRCA_V12S_001_TRAIN_Baseline_Vanilla/weights/best.pt` |
| 2 | CIRCA baseline weights | `runs/detect/CIRCA_V12S_002_TRAIN_Baseline_CIRCA/weights/best.pt` |
| 3 | Tuned hyperparameters | `runs/detect/CIRCA_V12S_003_TUNE_HPO/best_hyperparameters.yaml` |
| 4 | Final variant weights × 3 | `runs/detect/CIRCA_V12{N,S,M}_00{4,5,6}_*/weights/best.pt` |
| 5 | Quantisation comparison | `quantization_report.md` |
| 6 | Variant selection matrix | `benchmark_report.md` |
| 7 | Test metrics + thresholds | `test_evaluation.md`, `circa_thresholds.yaml` |

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Preprocessed dataset cache stale after label fixes | Use `--force-preproc` after any source-data change |
| HPO converges to overfit hyperparameters | Validate Phase-3 winner on Phase-2 val split before committing to Phase 4 |
| INT8 mAP collapse | FP16 fallback rule encoded in Phase 5 |
| Train ↔ inference distribution mismatch | Same CLAHE+Gamma applied to training images and runtime frames |
| Solder-class under-detection (10:1 imbalance vs bare-board classes) | `cls_pw=1.0` by default; if any solder class < 70% mAP after Phase 2, oversample 3× in train split or apply per-class loss weighting |
| Excessive false positives on background regions | Held-back `good`/`GOOD` images re-introduced as ~10% negative samples in a contingent Phase 1 re-run |
| Class imbalance in IPC taxonomy | Use `cls_pw` in HPO + oversample minority classes |
| GPU OOM (6 GB VRAM) | Default `batch=12`, `imgsz=640`, AMP on; downgrade to YOLOv12-N if needed |
| Reproducibility loss | `seed=42` + version logging + W&B run tags |
| Domain gap between bare-board and assembly defects in single model | Single-stage YOLO trained on both; if per-class mAP shows hard split, consider two-head or ensemble in future work (out of scope) |

---

## 7. Chapter 4 Figure & Table Inventory

This is the pre-mapped list of every figure and table Chapter 4 will need. Each entry links a deliverable from Phases 1–7 to a numbered figure or table, so once the runs complete every chapter slot has its source.

### Figures

| # | Title | Source |
|---|---|---|
| 4.1 | Sample defect images per IPC class | Dataset (curated grid of 15 classes) |
| 4.2 | HPO fitness curve over iterations | `runs/detect/CIRCA_V12S_003_TUNE_HPO/tune_fitness.png` |
| 4.3 | HPO parameter scatter / parallel coordinates | `runs/detect/CIRCA_V12S_003_TUNE_HPO/tune_scatter_plots.png` |
| 4.4 | Training curves (loss + mAP) per variant | `runs/detect/CIRCA_V12{N,S,M}_*/results.png` |
| 4.5 | End-to-end live FPS moving-average plot | `benchmark.py` output |
| 4.6 | Confusion matrix on test set | `runs/detect/<final>/confusion_matrix.png` |
| 4.7 | PR and F1 curves on test set | `runs/detect/<final>/PR_curve.png`, `F1_curve.png` |
| 4.8 | Failure-case gallery | Hand-picked from `runs/detect/<final>/val_batch*_pred.jpg` |
| 4.9 | Confidence threshold sweep on validation | `calibrate_thresholds.py` output |

### Tables

| # | Title | Source |
|---|---|---|
| 4.1 | Final class distribution (15 IPC classes — 10 bare-board + 5 solder) | Dataset stats script |
| 4.2 | Dataset statistics across train/val/test splits | Dataset stats script |
| 4.3 | Vanilla vs CIRCA-preprocessed baseline mAP | Phase 1 vs Phase 2 results |
| 4.4 | Preprocessing latency on target CPU | `benchmark.py` output |
| 4.5 | Top-10 HPO trials | `tune_results.csv` |
| 4.6 | Selected hyperparameter configuration | `best_hyperparameters.yaml` |
| 4.7 | Validation metrics per variant (mAP, P, R, F1) | Phase 4 results |
| 4.8 | Per-class performance breakdown | Phase 4 + 7 results |
| 4.9 | FP32 vs FP16 vs INT8 mAP per variant | `quantization_report.md` |
| 4.10 | INT8 → FP16 fallback decision per variant | `quantization_report.md` |
| 4.11 | Preprocessing latency on target CPU | `benchmark_report.md` |
| 4.12 | Inference latency CPU vs iGPU | `benchmark_report.md` |
| 4.13 | Static image inference time | `benchmark_report.md` |
| 4.14 | Variant Selection Matrix with Pass/Fail | `benchmark_report.md` |
| 4.15 | Overall metrics on test set | `test_evaluation.md` |
| 4.16 | Per-class precision, recall, F1 | `test_evaluation.md` |
| 4.17 | Per-class display + warning thresholds | `circa_thresholds.yaml` |
| 4.18 | Comparison with related published PCB detectors | Combine §7 plus Chapter 2 lit |

---

## 8. Comparison-with-Literature Template (Table 4.18)

Pre-filled rows from the Chapter 2 literature; CIRCA columns to be filled after Phase 7 completes.

| Author / Year | Model | Dataset | mAP | FPS | Hardware |
|---|---|---|---|---|---|
| Hu and Wang (2020) | Faster R-CNN + ResNet50-FPN + GARPN | PKU Open Lab | 94.2% | ~12 | GPU |
| Liao et al. (2021) | YOLOv4 + MobileNetV3 | Custom PCB | 98.64% | 56.98 | GPU |
| Bhattacharya and Cloutier (2022) | YOLOv5 + C3TR | Custom PCB | 98.1% | — | GPU |
| Yang and Yu (2024) | YOLOv8 + C2f + SPPF | PKU Open Lab | 92.3% | 157.2 | GPU |
| Tian et al. (2025) | YOLOv12-N | MS COCO | 40.6% | — | GPU |
| Hendriko and Hermanto (2025) | YOLOv12 | MOT17 | 88.0% mAP@50 | — | GPU |
| **CIRCA (this work)** | **YOLOv12-? INT8 / OpenVINO** | **CIRCA repair set** | **?** | **?** | **Intel i5 8th-gen CPU + iGPU** |

> **Note for the discussion:** CIRCA's distinguishing dimensions are (a) commodity CPU/iGPU deployment, (b) repair-context lighting, (c) IPC-A-610-aligned classes, and (d) confidence-transparent UI. Make these explicit when comparing — raw mAP alone is not the only axis.

---

## 9. Discussion Prompts (per Chapter 4 Section)

Use these as starter sentences when drafting Chapter 4. Each prompt is tied to a research objective.

### §4.2 Defect Taxonomy (RO1)
- Which IPC classes were most/least represented, and what is the implication for downstream model bias?
- How does the curated dataset compare in diversity to HRIPCB / DeepPCB (per Lv et al. 2024)?

### §4.3 Preprocessing Impact (RO2 / RO3)
- What absolute mAP lift did CIRCA preprocessing give over the vanilla baseline?
- Is the lift uniform across classes or concentrated on those affected by glare/shadow (e.g. fine conductor defects like `conductor_scratch`, `open_circuit`, `short`, `spur`)?
- Did preprocessing latency stay within the ≤ 5 ms budget?

### §4.4 HPO (RO2)
- Which hyperparameters had the highest importance in the genetic search?
- Did HPO-tuned values differ meaningfully from Ultralytics defaults? Why might that be?

### §4.5 Variant Comparison (RO2)
- How does mAP scale with variant size (N → S → M)?
- Where is the inflection point at which marginal mAP gain no longer justifies latency cost?

### §4.6 Quantisation (RO2)
- For each variant, what was the INT8 mAP penalty relative to FP32?
- Did any variant trigger the FP16 fallback rule? Why?

### §4.7 Hardware Benchmarking (RO3)
- Which configurations passed all four acceptance criteria simultaneously?
- How does CPU-only inference compare with iGPU inference on the same model?

### §4.8 Test Evaluation (RO3)
- Which classes showed the lowest recall, and what visual conditions caused failures?
- How does the final test mAP compare with the validation mAP — is there a generalisation gap?

### §4.9 Threshold Calibration (RO3)
- How often does the global "Manual Inspection Required" trigger fire under realistic webcam conditions?
- Does the chosen threshold strike the intended balance between automation bias and false-alarm fatigue (Goddard 2011, Kupfer 2023)?

### §4.10 Comparison with Literature
- Where does CIRCA win, and where does it concede vs prior work?
- What is the headline contribution: is it accuracy, deployability, repair-context robustness, or human-factors design?
