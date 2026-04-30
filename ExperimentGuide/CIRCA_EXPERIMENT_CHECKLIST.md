# CIRCA — Experiment Execution Checklist

> Tick boxes as you progress. Do not advance to the next phase until all items in the current phase are checked. Companion to `CIRCA_EXPERIMENT_PLAN.md`.

---

## Phase 0 — Environment & Dataset Setup

### Environment
- [x] `pip install ultralytics opencv-python numpy pyyaml`
- [x] `pip install openvino openvino-dev nncf`
- [x] `pip install wandb` and `wandb login`
- [x] `python -c "import torch; print(torch.cuda.is_available())"` returns `True`
- [x] `python -c "import ultralytics, openvino, nncf, cv2; print('ok')"` runs without error
- [x] `train_engine.py` placed in project root
- [x] `training.log` writable in CWD

### Dataset

#### v1 corpus (10-class, completed — retained as historical reference)
- [x] Raw bare-board datasets acquired (PKU-Market-PCB-ver1, DsPCBSD+, Roboflow PCB Defect)
- [x] Repair-context images present (desklamp glare, shadows, off-axis, motion blur)
- [x] v1 labels mapped to the 10-class taxonomy (classes 0–9)
- [x] v1 splits created: 70% train / 15% val / 15% test (10,128 sanitised)
- [x] No duplicates across splits (perceptual hash verified; 8,566 images purged)

#### v2 expansion (15-class, IPC-A-610 — IN PROGRESS)
- [ ] **Acquire SolDef_AI** (Fontana et al., 2024) — Roboflow YOLOv8 export from `defectdetection-e5sqy/soldef_ai-for-defect-detection` (CC BY 4.0)
- [ ] **Acquire PCB Solder Joint** (Work, 2025) — Roboflow YOLOv8 export from `work-6qkmv/pcb-solder-joint` (CC BY 4.0)
- [ ] **Polygon→bbox conversion verified** for SolDef_AI (Roboflow object-detection export auto-handles)
- [ ] **Label remap completed:**
  - [ ] SolDef_AI: drop `good`(0) and `no_good`(4); remap `exc_solder`(1)→10, `poor_solder`(2)→11, `spike`(3)→12
  - [ ] PCB Solder Joint: drop `GOOD`(0); remap `STICKSOLDER`(2)→13, `COLDSOLDER`(1)→14
- [ ] **Dropped `good`/`GOOD` images archived** in `negatives_reserve/` for potential later use as background samples
- [ ] Unified 15-class `class_mapping.md` updated
- [ ] **pHash dedup re-run** across new sources + cross-checked against v1 hashes
- [ ] **Stratified 70/15/15 split** with `seed=42` on the merged corpus
- [ ] Test set frozen (`chmod -w` or read-only mount)
- [ ] Per-class instance counts logged for all 15 classes
- [ ] **Estimated final corpus:** ~11,778 images (10,128 carried + ~1,650 new); record actual after dedup
- [ ] `data.yaml` validated: paths resolve, `nc=15`, `names` match the 15-class taxonomy:
  - [ ] `0: missing_hole`
  - [ ] `1: mouse_bite`
  - [ ] `2: open_circuit`
  - [ ] `3: short`
  - [ ] `4: spur`
  - [ ] `5: spurious_copper`
  - [ ] `6: hole_breakout`
  - [ ] `7: conductor_scratch`
  - [ ] `8: conductor_foreign_object`
  - [ ] `9: base_material_foreign_object`
  - [ ] `10: excess_solder`
  - [ ] `11: insufficient_solder`
  - [ ] `12: solder_spike`
  - [ ] `13: solder_bridge`
  - [ ] `14: cold_solder_joint`

> See `CIRCA_DATASET_EXPANSION_PLAN.md` for the full integration procedure (acquisition, conversion, remap, sanitise, split, class-imbalance strategy).

---

## Phase 1 — Vanilla Baseline (Ablation Control)

> **Re-run required against `unified_pcb_v2` (15 classes).** The v1 baseline was on the 10-class corpus and is no longer comparable. Epoch count is held at 50 for parameter parity — only class count and corpus size change.

- [ ] Command run (against `unified_pcb_v2`):
      ```
      python train_engine.py --mode train --variant s --id 001 --desc Baseline_Vanilla --epochs 50 --clear
      ```
- [ ] Training completed without crashes / NaN / Inf warnings
- [ ] `runs/detect/CIRCA_V12S_001_TRAIN_Baseline_Vanilla/weights/best.pt` exists
- [ ] OpenVINO INT8 export folder created
- [ ] Final mAP@0.5 (val) recorded: ____________
- [ ] Final mAP@0.5:0.95 (val) recorded: ____________
- [ ] **False-positive review on val:** background regions visually inspected; if FPs are excessive, plan a contingent re-run with negative samples per the integration plan.

---

## Phase 2 — CIRCA-Aligned Baseline

> **Re-run required against `unified_pcb_v2` (15 classes).** Train with `cls_pw=1.0` to compensate for the bare-board (≈10×) dominance via inverse-frequency weighting.

- [ ] Command run (against `unified_pcb_v2`):
      ```
      python train_engine.py --mode train --variant s --id 002 --desc Baseline_CIRCA --epochs 100 --preproc
      ```
- [ ] Preprocessing log shows non-zero images and labels for **all three** splits (train, val, test)
- [ ] No "Missing label" warnings (or expected/documented count)
- [ ] `<dataset>_preproc/` folder created with mirrored structure
- [ ] Training completed without crashes / NaN / Inf warnings
- [ ] `runs/detect/CIRCA_V12S_002_TRAIN_Baseline_CIRCA/weights/best.pt` exists
- [ ] OpenVINO INT8 export folder created
- [ ] Final mAP@0.5 (val) recorded: ____________
- [ ] Final mAP@0.5:0.95 (val) recorded: ____________
- [ ] **CIRCA baseline mAP ≥ Vanilla baseline mAP** (ablation lift confirmed)

---

## Phase 3 — Hyperparameter Optimisation (HPO)

> **Re-run required against `unified_pcb_v2` (15 classes).** HPO must be tuned against the final taxonomy.

- [ ] Command run (against `unified_pcb_v2`):
      ```
      python train_engine.py --mode tune --variant s --id 003 --desc HPO --epochs 30 --iterations 150 --preproc
      ```
- [ ] All 150 iterations completed (or early-stopped with rationale logged)
- [ ] `runs/detect/CIRCA_V12S_003_TUNE_HPO/best_hyperparameters.yaml` exists
- [ ] `tune_results.csv` saved
- [ ] `tune_scatter_plots.png` and `tune_fitness.png` saved
- [ ] Best-trial mAP ≥ Phase 2 baseline mAP
- [ ] Top-3 trial mAP values recorded: ____, ____, ____
- [ ] HPO trial diversity check: best trials are not all clustered at search-space edges

---

## Phase 4 — Three-Variant Final Training

### YOLOv12-N
- [ ] Command run:
      ```
      python train_engine.py --mode train --variant n --id 004 --desc Final_HPO --epochs 200 --preproc \
          --cfg runs/detect/CIRCA_V12S_003_TUNE_HPO/best_hyperparameters.yaml
      ```
- [ ] `weights/best.pt` exists
- [ ] OpenVINO INT8 IR exported
- [ ] mAP@0.5 (val): ____________

### YOLOv12-S
- [ ] Command run:
      ```
      python train_engine.py --mode train --variant s --id 005 --desc Final_HPO --epochs 200 --preproc \
          --cfg runs/detect/CIRCA_V12S_003_TUNE_HPO/best_hyperparameters.yaml
      ```
- [ ] `weights/best.pt` exists
- [ ] OpenVINO INT8 IR exported
- [ ] mAP@0.5 (val): ____________

### YOLOv12-M
- [ ] Command run:
      ```
      python train_engine.py --mode train --variant m --id 006 --desc Final_HPO --epochs 200 --preproc \
          --cfg runs/detect/CIRCA_V12S_003_TUNE_HPO/best_hyperparameters.yaml
      ```
- [ ] `weights/best.pt` exists
- [ ] OpenVINO INT8 IR exported
- [ ] mAP@0.5 (val): ____________

---

## Phase 5 — OpenVINO Quantisation Validation

For each of the three variants:
- [ ] FP32 OpenVINO IR exported and val mAP measured
- [ ] FP16 OpenVINO IR exported and val mAP measured
- [ ] INT8 OpenVINO IR exported and val mAP measured
- [ ] INT8 vs FP32 mAP delta computed
- [ ] Fallback decision applied per rule:
  - INT8 ≥ FP32 − 1% **and** ≥ 90% → use INT8
  - 90% ≤ INT8 < FP32 − 1% → INT8 unless latency budget allows FP16
  - INT8 < 90% → fall back to FP16
- [ ] `quantization_report.md` written with full per-variant table

---

## Phase 6 — Hardware Benchmarking & Variant Selection

Run on the deployment target (Intel Core i5 8th-gen-equivalent CPU + iGPU).

For each variant × precision combination:
- [ ] Preprocessing latency measured (target ≤ 5 ms)
- [ ] Inference latency on CPU measured
- [ ] Inference latency on iGPU measured
- [ ] Live FPS via webcam loop measured (target ≥ 15 FPS)
- [ ] Static image inference time measured (target ≤ 10 s)
- [ ] Model size on disk recorded

Selection:
- [ ] Variant Selection Matrix populated with all numbers
- [ ] Each row marked Pass / Fail against the four acceptance criteria
- [ ] Highest mAP **passing** configuration chosen as the production variant
- [ ] `benchmark_report.md` written; winner clearly highlighted

---

## Phase 7 — Final Test Evaluation & Confidence Calibration

### Test Evaluation (run **once**)
- [ ] Selected variant + precision evaluated with `model.val(split="test")`
- [ ] Per-class precision, recall, F1 reported
- [ ] Overall mAP@0.5 and mAP@0.5:0.95 reported
- [ ] Confusion matrix saved
- [ ] PR / F1 curves saved
- [ ] Failure-case gallery compiled (small defects, glare, shadow, motion blur, off-angle)
- [ ] All four acceptance criteria re-verified on test set

### Confidence Threshold Calibration (val set only)
- [ ] Sweep `conf` from 0.10 → 0.90 in 0.05 steps on val
- [ ] Per-class **display threshold** chosen (precision ≥ 0.90)
- [ ] Per-class **warning threshold** chosen (recall ≥ 0.95)
- [ ] Global "Manual Inspection Required" trigger rule documented
- [ ] `circa_thresholds.yaml` written

### Sign-off
- [ ] `test_evaluation.md` written and reviewed
- [ ] `class_mapping.md`, `quantization_report.md`, `benchmark_report.md`, `test_evaluation.md`, `circa_thresholds.yaml` all present
- [ ] All `runs/` artifacts synced to W&B
- [ ] Versions logged for the final run (torch, ultralytics, openvino, nncf, opencv)
- [ ] Thesis Chapter 4 ready to draft from these artifacts

---

## Reproducibility Audit (final gate)

- [ ] `seed=42` present in every recorded run
- [ ] `data.yaml` and IPC class mapping unchanged across Phases 2–7
- [ ] CLAHE/Gamma parameters unchanged across Phases 2–7
- [ ] OpenVINO + NNCF versions unchanged between Phase 4 export and Phase 5 evaluation
- [ ] HPO config used in Phase 4 matches the Phase 3 output byte-for-byte
- [ ] No test-set images appear in any train or val split (perceptual-hash audit re-run)
