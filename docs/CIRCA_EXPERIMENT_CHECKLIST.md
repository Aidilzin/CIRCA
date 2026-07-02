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
- [x] `python scripts/build_unified_pcb_v3.py --dry-run` — verify class counts look reasonable ✅
- [x] `python scripts/build_unified_pcb_v3.py` — execute full build ✅
- [x] Confirm `datasets/unified_pcb_v3/data.yaml` exists with `nc=7` ✅
- [x] **Step 1.5 — Dominant-class capping** (`python scripts/cap_dominant_classes.py --cap 1000 --seed 42`) ✅
  - Removed 2,468 dominant-only images (only `short`/`insufficient_solder` annotations)
  - Annotation ratio: **9.9:1 → 5.7:1** (well within ≤10:1 target)
- [x] **Oversampling tiers v4** (`python scripts/oversample_minority_classes.py`) ✅
  - `missing_hole` (0): **5×** (promoted from 3× — had 0% recall in baseline)
  - `excess_solder` (4): **5×** (added — had 51% recall in baseline, 304 originals)
  - `cold_solder_joint` (6): **5×** (maintained)
  - Excluded: `short` (3), `insufficient_solder` (5) (dominant)
- [x] **CLAHE+Gamma preproc** (`unified_pcb_v3_preproc/` built offline) ✅
- [x] **Full pipeline run** via `python scripts/prepare_all_datasets.py` ✅
- [x] Per-class instance counts logged (v4, after capping + oversampling):

| Class | Train Annotations | Status |
|:---|:---:|:---:|
| `missing_hole` (0) | 4,290 | 5× oversampled |
| `mouse_bite` (1) | 2,062 | No oversampling needed |
| `open_circuit` (2) | 1,399 | No oversampling needed |
| `short` (3) | 5,656 | Excluded (dominant, capped) |
| `excess_solder` (4) | 1,400 | 5× oversampled |
| `insufficient_solder` (5) | 11,739 | Excluded (dominant, capped) |
| `cold_solder_joint` (6) | 1,185 | 5× oversampled |

- [x] Test set frozen — hash of test split recorded ✅
- [x] `unified_pcb_v3_preproc/` generated ✅
- [x] Chain duplicate bug fixed in `oversample_minority_classes.py` — `_os*` files excluded as source material ✅
- [x] **Dataset FROZEN** — no further structural changes before Phase 4 final training ✅

---

## Phase 1 — Vanilla Baseline (Ablation Control) ✅ COMPLETE

> **Dataset changed (v4 tiers + capping).** Old results from 2026-05-25 are diagnostic only — not valid for thesis. Re-run on RunPod after uploading v4 dataset.
> Trained on Runpod RTX 3090 24GB, batch=48, seed=42.

- [x] Command (Runpod RTX 3090):
  ```bash
  python train_engine.py --mode train --variant s --id 001 --desc Baseline_Vanilla \
      --epochs 100 --batch 48 --cache \
      --data datasets/unified_pcb_v3/data.yaml
  ```
- [x] Training complete, 100/100 epochs, no NaN/Inf
- [x] `best.pt` exists in `runs/detect/CIRCA_V12S_001_TRAIN_Baseline_Vanilla/weights/`
- [x] mAP@0.5 (best epoch) = 0.6649 (Epoch 45)
- [x] mAP@0.5:0.95 (best epoch) = 0.4237
- [x] Precision: 0.7290 | Recall: 0.6433 (at best epoch)
- [x] W&B run manually synced via `upload_run_to_wandb.py`

> **Old result (diagnostic only, old dataset):** mAP@0.5=0.6821, P=0.883, R=0.638 (ep.57 of 100)
> **Ablation Control Baseline (mAP_v): 0.6649 (v4 dataset baseline)**

---

## Phase 2 — CIRCA-Aligned Baseline ✅ COMPLETE

> **v4 dataset (capped + oversampled).** Early stop triggered at epoch 80 (patience=30). Best checkpoint at epoch 50.
> Trained on Runpod RTX 3090 24GB, batch=48, seed=42.

- [x] Command (Runpod RTX 3090):
  ```bash
  python train_engine.py --mode train --variant s --id 002 --desc Baseline_CIRCA \
      --epochs 100 --batch 48 --cache \
      --data datasets/unified_pcb_v3_preproc/data.yaml
  ```
  > **Note:** Phase 2 uses `unified_pcb_v3_preproc/data.yaml` directly (preproc is built offline). Do NOT use `--preproc` flag.
- [x] Early stop at epoch 80/100 (patience=30) — best checkpoint at epoch 50 ✅
- [x] `best.pt` exists in `runs/detect/CIRCA_V12S_002_TRAIN_Baseline_CIRCA/weights/`
- [x] mAP@0.5 (best epoch) = 0.6600 (Epoch 50)
- [x] mAP@0.5:0.95 (best epoch) = 0.4284 (**+0.47 pp vs Phase 1 ✅**)
- [x] Precision: **0.8443** (+11.5 pp vs Phase 1 ✅) | Recall: 0.6341
- [x] W&B run manually synced via `upload_run_to_wandb.py`

> **Old result (diagnostic only, old dataset):** mAP@0.5=0.6670, P=0.842, R=0.623 (ep.52 of 100)
> **Preprocessing lift (v4): mAP@0.5 −0.49 pp (within noise) | mAP@0.5:0.95 +0.47 pp | Precision +11.5 pp | Val cls_loss lower from ep.40+**
> **Verdict: Selective improvement — CLAHE lifts precision and classification confidence without broad mAP regression.**

---

## Phase 3 — HPO on Oversampled Preprocessed Data ✅ COMPLETE

> Run on `unified_pcb_v3_preproc/` (already preprocessed + oversampled). Do NOT use `--preproc` flag.
> Corrects the Phase 3 flaw from the old dataset (imbalanced data + 30 epochs too short).

- [x] Command (Runpod RTX 3090):
  ```bash
  python train_engine.py --mode tune --variant s --id 003 --desc HPO_7class \
      --epochs 50 --iterations 50 --fraction 0.5 --batch 48 --cache \
      --data datasets/unified_pcb_v3_preproc/data.yaml
  ```
- [x] 50 iterations complete ✅
- [x] `best_hyperparameters.yaml` exists in HPO run folder ✅
- [x] Best fitness > Phase 2 HPO trial baseline (0.26305 vs 0.19978) ✅
- [x] Top HPO parameters recorded:
  - `lr0`: 0.00014 | `momentum`: 0.785 | `weight_decay`: 0.0009
  - `box`: 0.169 | `cls`: 0.266 | `cls_pw`: 0.100

---

## Phase 4 — Three-Variant Final Training ✅ COMPLETE
 
> Uses HPO config + oversampled `unified_pcb_v3_preproc/` data.
 
- [x] YOLOv12-N (id 004, batch=64): mAP@0.5: **63.13%** | mAP@0.5:0.95: **39.52%**
- [x] YOLOv12-S (id 005, batch=48): mAP@0.5: **66.20%** | mAP@0.5:0.95: **42.97%**
- [x] YOLOv12-M (id 006, batch=32): mAP@0.5: **67.42%** | mAP@0.5:0.95: **43.89%**
- [x] All three `best.pt` checkpoints downloaded from Runpod
 
---
 
## Phase 5 — OpenVINO Quantisation (Local) ✅ COMPLETE
 
- [x] FP32/FP16/INT8 val mAP measured per variant
- [x] Fallback decision applied (INT8 < FP32 − 1% or < 90% → FP16)
- [x] `quantization_report.md` written
 
---
 
## Phase 6 — Hardware Benchmarking (Local) ✅ COMPLETE
 
- [x] Preprocessing latency ≤ 5 ms
- [x] Live FPS ≥ 15
- [x] Static inference ≤ 10 s
- [x] Variant Selection Matrix populated
- [x] `benchmark_report.md` written
 
---
 
## Phase 7 — Final Test Evaluation (Local) ✅ COMPLETE
 
- [x] `model.val(split="test")` run **once** on chosen variant
- [x] Per-class P/R/F1 for all 7 classes logged
- [x] Confusion matrix and PR/F1 curves saved
- [x] Confidence threshold sweep complete; `circa_thresholds.yaml` written
- [x] `test_evaluation.md` written; W&B synced
 
---
 
## Reproducibility Audit ✅ PASSED
 
- [x] `seed=42` in every run
- [x] `nc=7` consistent Phases 1–7
- [x] CLAHE/Gamma parameters unchanged Phases 2–7 (clip=2.0, tile=8×8, γ=1.2)
- [x] No test-set images in train/val (pHash audit at build time)
- [x] All Runpod runs downloaded to `runs/detect/` before pod termination

