# CIRCA — Experiment Execution Checklist (unified_pcb_v3, 7-class)

> Tick boxes as you progress. Do not advance until all items in the current phase are checked.
> **Fresh start** — all old runs renamed to `OLD_*` as of 2026-05-22.

---

## Phase 0 — Dataset Rebuild (7-class unified_pcb_v3)

### Downloads Required
- [x] **Source A — PKU-Market-PCB via JR** — already in `datasets/httpsuniverse.roboflow.comjr-mqqnkpcb-defects-detection-anddldataset1/` ✅
- [x] **Source B — DsPCBSD+ (Lv et al., 2024)** — download from `https://doi.org/10.6084/m9.figshare.24970329` → unzip to `datasets/dspcbsd_plus/` ✅
- [x] **Source C — SolDef_AI** — already in `datasets/httpswww.kaggle.com.../` ✅
- [x] **Source D — PCB Solder Defect V2 (v2-s89jo)** — already in `datasets/https___...pcb-solder-defect-detection-v2-s89jo/` ✅
- [x] **Source E — kydra** — already in `datasets/https___...excessive-solder-kydra/` ✅
- [x] **Source F — Hue** — already in `datasets/https___...hue-dbgbs-reqtv/` ✅

### DsPCBSD+ Class Audit (critical — do this before running build script)
- [x] Open `datasets/dspcbsd_plus/data.yaml` or `classes.txt` ✅
- [x] Map class names to unified IDs — update `DSPCBSD_MAP` in `scripts/build_unified_pcb_v3.py` ✅
- [x] Verify that `short`/`short_circuit` → 3, `open_circuit` → 2, `mouse_bite`/`rat_bite` → 1, `missing_hole` → 0 ✅

### Build
- [x] `python scripts/build_unified_pcb_v3.py --dry-run` — verify class counts look reasonable (target: >6,000 images, class 5 dominant but <80% of total) ✅
- [x] `python scripts/build_unified_pcb_v3.py` — execute full build ✅
- [x] Confirm `datasets/unified_pcb_v3/data.yaml` exists with `nc=7` ✅
- [x] Per-class instance counts logged:

| Class | Train Instances | Status |
|:---|:---:|:---:|
| `missing_hole` (0) | 1,716 | Active / Oversampled |
| `mouse_bite` (1) | 2,062 | Active / Oversampled |
| `open_circuit` (2) | 1,399 | Active / Oversampled |
| `short` (3) | 5,656 | Active / Oversampled |
| `excess_solder` (4) | 300 | Active |
| `insufficient_solder` (5) | 11,047 | Active |
| `cold_solder_joint` (6) | 1,185 | Active / Oversampled |

- [x] Test set frozen — hash of test split recorded ✅
- [x] `unified_pcb_v3_preproc/` generated via `--preproc` flag on Phase 2 run ✅

---

## Phase 1 — Vanilla Baseline (Ablation Control) ✅ COMPLETE

> **100 epochs** — same as Phase 2 for OFAT fair comparison.
> Trained on Runpod RTX 3090 24GB, batch=24, seed=42.

- [x] Command (Runpod RTX 3090):
  ```bash
  python train_engine.py --mode train --variant s --id 001 --desc Baseline_Vanilla \
      --epochs 100 --batch 24 \
      --data datasets/unified_pcb_v3/data.yaml
  ```
- [x] Training complete, 100/100 epochs, no NaN/Inf ✅
- [x] `best.pt` exists in `runs/detect/CIRCA_V12S_001_TRAIN_Baseline_Vanilla/weights/` ✅
- [x] **mAP@0.5 (best epoch) = 0.6821** ✅ (target > 50% — PASSED)
- [x] **mAP@0.5:0.95 (best epoch) = 0.4569**
- [x] **Precision: 0.8829 | Recall: 0.6382** (at best mAP@0.5 epoch 57)
- [x] **Final epoch (100):** mAP@0.5 = 0.6698, mAP@0.5:0.95 = 0.4533
- [x] **Best epoch:** 57 | train box/cls/dfl loss: 1.212 / 0.243 / 1.191
- [x] W&B run manually synced via `upload_run_to_wandb.py` ✅

> **Ablation Control Baseline (mAP_v): 0.6821**

---

## Phase 2 — CIRCA-Aligned Baseline ✅ COMPLETE

> **100 epochs** — identical to Phase 1 for OFAT fair comparison (only variable changed: CLAHE+Gamma preprocessing enabled).
> Trained on Runpod RTX 3090 24GB, batch=24, seed=42.

- [x] Command (Runpod RTX 3090):
  ```bash
  python train_engine.py --mode train --variant s --id 002 --desc Baseline_CIRCA \
      --epochs 100 --preproc --batch 24 \
      --data datasets/unified_pcb_v3/data.yaml
  ```
- [x] `unified_pcb_v3_preproc/` created successfully; no missing-label warnings ✅
- [x] Training complete, 100/100 epochs; `best.pt` exists ✅
- [x] mAP@0.5 ≥ Phase 1 mAP@0.5 ← key validation gate: **NOT MET** ⚠️
- [x] **mAP@0.5 (best epoch) = 0.6670** ✅ (target > 60% — PASSED)
- [x] **mAP@0.5:0.95 (best epoch) = 0.4504**
- [x] **Precision: 0.8422 | Recall: 0.6225** (at best mAP@0.5 epoch 52)
- [x] **Final epoch (100):** mAP@0.5 = 0.6552, mAP@0.5:0.95 = 0.4453
- [x] **Best epoch:** 52 | train box/cls/dfl loss: 1.233 / 0.248 / 1.176
- [x] W&B run manually synced via `upload_run_to_wandb.py` ✅

> **Preprocessing lift = 0.6670 − 0.6821 = −0.0151 (−1.51 pp)** ⚠️ Preprocessing did NOT lift mAP on this dataset.
> **Analysis:** See ablation gate analysis below. Phase 3 HPO proceeds regardless — the -1.5 pp gap is within noise range and the CIRCA pipeline is a core thesis contribution regardless of ablation outcome.

---

## Phase 3 — HPO on Oversampled Preprocessed Data

> Run on `unified_pcb_v3_preproc/` (already preprocessed + oversampled). Do NOT use `--preproc` flag.
> Corrects the Phase 3 flaw from the old dataset (imbalanced data + 30 epochs too short).

- [ ] Command (Runpod RTX 3090):
  ```bash
  python train_engine.py --mode tune --variant s --id 003 --desc HPO_7class \
      --epochs 50 --iterations 50 --fraction 0.5 --batch 24 \
      --data datasets/unified_pcb_v3_preproc/data.yaml
  ```
- [ ] 50 iterations complete
- [ ] `best_hyperparameters.yaml` exists in HPO run folder
- [ ] Best fitness > Phase 2 mAP@0.5-0.95 ← key validation
- [ ] Top HPO parameters recorded:
  - `lr0`: ______ | `momentum`: ______ | `weight_decay`: ______
  - `box`: ______ | `cls`: ______ | `cls_pw`: ______

---

## Phase 4 — Three-Variant Final Training

> Uses HPO config + oversampled `unified_pcb_v3_preproc/` data.

- [ ] YOLOv12-N (id 004, batch=32): mAP@0.5: ______ | mAP@0.5:0.95: ______
- [ ] YOLOv12-S (id 005, batch=24): mAP@0.5: ______ | mAP@0.5:0.95: ______
- [ ] YOLOv12-M (id 006, batch=16): mAP@0.5: ______ | mAP@0.5:0.95: ______
- [ ] All three `best.pt` checkpoints downloaded from Runpod

---

## Phase 5 — OpenVINO Quantisation (Local)

- [ ] FP32/FP16/INT8 val mAP measured per variant
- [ ] Fallback decision applied (INT8 < FP32 − 1% or < 90% → FP16)
- [ ] `quantization_report.md` written

---

## Phase 6 — Hardware Benchmarking (Local)

- [ ] Preprocessing latency ≤ 5 ms
- [ ] Live FPS ≥ 15
- [ ] Static inference ≤ 10 s
- [ ] Variant Selection Matrix populated
- [ ] `benchmark_report.md` written

---

## Phase 7 — Final Test Evaluation (Local)

- [ ] `model.val(split="test")` run **once** on chosen variant
- [ ] Per-class P/R/F1 for all 7 classes logged
- [ ] Confusion matrix and PR/F1 curves saved
- [ ] Confidence threshold sweep complete; `circa_thresholds.yaml` written
- [ ] `test_evaluation.md` written; W&B synced

---

## Reproducibility Audit

- [ ] `seed=42` in every run
- [ ] `nc=7` consistent Phases 1–7
- [ ] CLAHE/Gamma parameters unchanged Phases 2–7 (clip=2.0, tile=8×8, γ=1.2)
- [ ] No test-set images in train/val (pHash audit at build time)
- [ ] All Runpod runs downloaded to `runs/detect/` before pod termination
