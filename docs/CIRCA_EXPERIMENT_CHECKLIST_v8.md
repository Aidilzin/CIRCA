# CIRCA — Experiment Execution Checklist

> Tick boxes as you progress. Do not advance until all items in the current phase are checked.

---

## Phase 0 — Environment & Dataset Setup

### Environment
- [X] Ultralytics, OpenCV, W&B, OpenVINO, NNCF installed
- [X] `torch.cuda.is_available()` = True
- [X] `train_engine.py` in project root

### Dataset (12-class)
- [X] PKU-Market-PCB-ver1 and Roboflow PCB Defect acquired
- [X] SolDef_AI (CC BY 4.0) acquired and polygon→bbox converted
- [X] PCB Solder Joint (CC BY 4.0) acquired
- [X] Label remap: SolDef_AI alphabetical export: raw ID 0 (`exc_solder`)→6, raw ID 3 (`poor_solder`)→7, raw ID 4 (`spike`)→8; IDs 1 (`good`) and 2 (`no_good`) → negatives_reserve
- [X] `good`/`GOOD` images archived in `negatives_reserve/`
- [X] pHash dedup run (dHash ≤ 6)
- [X] Stratified 70/15/15 split, `seed=42`
- [X] Test set frozen
- [X] Per-class instance counts logged for all 12 classes
- [X] `unified_pcb_v2/` folder created
- [X] `data.yaml` validated: `nc=12`, all 12 names present

---

## Phase 1 — Vanilla Baseline (Ablation Control)

- [ ] Command:
  ```
  python train_engine.py --mode train --variant s --id 001 --desc Baseline_Vanilla --epochs 50 --clear --data datasets/unified_pcb_v2/data.yaml
  ```
- [ ] Training complete, no NaN/Inf
- [ ] `best.pt` exists; OpenVINO INT8 exported
- [ ] mAP@0.5 recorded: ____________
- [ ] mAP@0.5:0.95 recorded: ____________

---

## Phase 2 — CIRCA-Aligned Baseline

- [ ] Command:
  ```
  python train_engine.py --mode train --variant s --id 002 --desc Baseline_CIRCA --epochs 100 --preproc --data datasets/unified_pcb_v2/data.yaml
  ```
- [ ] `unified_pcb_v2_preproc/` created; no missing label warnings
- [ ] Training complete; `best.pt` exists; OpenVINO INT8 exported
- [ ] mAP@0.5 ≥ Phase 1 mAP@0.5
- [ ] mAP@0.5 recorded: ____________

---

## Phase 3 — HPO

- [ ] Command:
  ```
  python train_engine.py --mode tune --variant s --id 003 --desc HPO --epochs 30 --iterations 50 --fraction 0.3 --preproc --data datasets/unified_pcb_v2/data.yaml
  ```
- [ ] 50 iterations complete
- [ ] `best_hyperparameters.yaml` exists
- [ ] `tune_results.csv`, `tune_fitness.png` saved
- [ ] Top-3 mAP: ____, ____, ____

---

## Phase 4 — Three-Variant Final Training

- [ ] YOLOv12-N (id 004): `best.pt` + INT8 IR | mAP@0.5: ____
- [ ] YOLOv12-S (id 005): `best.pt` + INT8 IR | mAP@0.5: ____
- [ ] YOLOv12-M (id 006): `best.pt` + INT8 IR | mAP@0.5: ____

---

## Phase 5 — OpenVINO Quantisation

- [ ] FP32/FP16/INT8 val mAP measured per variant
- [ ] Fallback decision applied (INT8 < 90% → FP16)
- [ ] `quantization_report.md` written

---

## Phase 6 — Hardware Benchmarking

- [ ] Preprocessing latency ≤ 5 ms
- [ ] Live FPS ≥ 15
- [ ] Static inference ≤ 10 s
- [ ] Variant Selection Matrix populated
- [ ] `benchmark_report.md` written

---

## Phase 7 — Final Test Evaluation

- [ ] `model.val(split="test")` run once on chosen variant
- [ ] Per-class P/R/F1 for 12 classes logged
- [ ] Confusion matrix and PR/F1 curves saved
- [ ] Confidence threshold sweep complete; `circa_thresholds.yaml` written
- [ ] `test_evaluation.md` written; W&B synced

---

## Reproducibility Audit

- [ ] `seed=42` in every run
- [ ] `nc=12` consistent Phases 1–7
- [ ] CLAHE/Gamma parameters unchanged Phases 2–7
- [ ] No test-set images in train/val (pHash audit)
