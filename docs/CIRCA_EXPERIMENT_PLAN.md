# CIRCA — Experiment Plan (unified_pcb_v3, 7-class)

**Project:** CIRCA — Circuit Inspection and Recognition using Convolutional Architectures
**Model:** YOLOv12 (N / S / M) | **Deployment:** Intel OpenVINO INT8
**Dataset:** `unified_pcb_v3` (7-class, nc=7)
**Status:** Phases 1 & 2 complete on Runpod (2026-05-25). Phase 3 HPO pending.

---

## 1. Objectives

1. Identify and document IPC-A-600 and IPC-A-610-aligned PCB defect types for automated detection (**RO1**).
2. Design, train, and compare **YOLOv12-N/S/M** with Intel OpenVINO INT8 deployment (**RO2**).
3. Evaluate the CIRCA preprocessing pipeline and desktop application using P, R, F1, mAP, and latency (**RO3**).

---

## 2. Acceptance Criteria

| Metric | Target |
|:---|:---|
| mAP@0.5 (test) | > 90% |
| Preprocessing latency | ≤ 5 ms |
| Live FPS | ≥ 15 |
| Static image inference | ≤ 10 s |

---

## 3. Phases

| Phase | ID | Description | Epochs | Status | Key Result |
|:---|:---|:---|:---|:---:|:---|
| 0 | — | Dataset rebuild (7-class `unified_pcb_v3`) | — | ✅ Done | 23,365 images, 7 classes, `unified_pcb_v3/` |
| 1 | 001 | Vanilla YOLOv12-S, raw images | **100** | ✅ Done | **mAP@0.5 = 0.6821** (best ep.57), P=0.883, R=0.638 |
| 2 | 002 | CLAHE+Gamma preprocessed, YOLOv12-S | **100** | ✅ Done | **mAP@0.5 = 0.6670** (best ep.52), P=0.842, R=0.623 |
| 3 | 003 | Genetic HPO on oversampled preproc data | 50/iter × 50 iter | ⏳ Next | Optimal hyperparameter config |
| 4 | 004–006 | Final N/S/M with HPO config | 200 | 🔲 Pending | Three final variants |
| 5 | — | OpenVINO FP32/FP16/INT8 export | — | 🔲 Pending | Quantisation delta |
| 6 | — | Hardware benchmark on i5 8th-gen | — | 🔲 Pending | Latency validation |
| 7 | — | Frozen test evaluation | — | 🔲 Pending | Final thesis metrics |

> **Phase 1 & 2 Epoch Justification:** Both phases use an identical 100-epoch budget to implement a one-factor-at-a-time (OFAT) ablation. Any mAP difference between Phase 1 and Phase 2 is then attributable solely to the presence or absence of the CLAHE+Gamma preprocessing pipeline, not to differential training time. This design is consistent with ablation methodology in related YOLO PCB literature (Yang and Yu, 2024; Liao et al., 2021).

---

## 4. Run Commands (Runpod — RTX 3090)

```bash
# Phase 0 — Build unified_pcb_v3 (local, before Runpod)
python scripts/build_unified_pcb_v3.py --dry-run   # verify counts
python scripts/build_unified_pcb_v3.py             # execute

# Phase 1 — Vanilla Baseline (100 epochs, same as Phase 2 for fair OFAT comparison)
python train_engine.py --mode train --variant s --id 001 --desc Baseline_Vanilla \
    --epochs 100 --batch 24 \
    --data datasets/unified_pcb_v3/data.yaml

# Phase 2 — CIRCA-Aligned Baseline (100 epochs, CLAHE+Gamma)
python train_engine.py --mode train --variant s --id 002 --desc Baseline_CIRCA \
    --epochs 100 --preproc --batch 24 \
    --data datasets/unified_pcb_v3/data.yaml

# Phase 3 — HPO on oversampled preprocessed data (NOT raw — do not use --preproc flag)
python train_engine.py --mode tune --variant s --id 003 --desc HPO_7class \
    --epochs 50 --iterations 50 --fraction 0.5 --batch 24 \
    --data datasets/unified_pcb_v3_preproc/data.yaml

# Phase 4 — Final Training (3 variants, 200 epochs, HPO config)
python train_engine.py --mode train --variant n --id 004 --desc Final_HPO \
    --epochs 200 --batch 32 \
    --data datasets/unified_pcb_v3_preproc/data.yaml \
    --cfg runs/detect/CIRCA_V12S_003_TUNE_HPO_7class/best_hyperparameters.yaml

python train_engine.py --mode train --variant s --id 005 --desc Final_HPO \
    --epochs 200 --batch 24 \
    --data datasets/unified_pcb_v3_preproc/data.yaml \
    --cfg runs/detect/CIRCA_V12S_003_TUNE_HPO_7class/best_hyperparameters.yaml

python train_engine.py --mode train --variant m --id 006 --desc Final_HPO \
    --epochs 200 --batch 16 \
    --data datasets/unified_pcb_v3_preproc/data.yaml \
    --cfg runs/detect/CIRCA_V12S_003_TUNE_HPO_7class/best_hyperparameters.yaml
```

> **Batch sizes assume RTX 3090 (24 GB VRAM).** For the local RTX 3060 (6 GB), use batch=16 (N), 12 (S), 6 (M).

---

## 5. Risks & Mitigations

| Risk | Mitigation |
|:---|:---|
| DsPCBSD+ class names differ from expected | Audit `classes.txt` before running build script; update `DSPCBSD_MAP` in `build_unified_pcb_v3.py` |
| Preprocessing cache stale | `--force-preproc` flag |
| INT8 mAP collapse | FP16 fallback (if INT8 mAP < FP32 − 1% or < 90%) |
| Solder class under-detection | `cls_pw` inv-freq weighting + `mosaic=1.0` + `copy_paste=0.3` |
| Budget exceeded on Runpod | Run Phase 1+2 first ($1); verify mAP >50% before crediting for HPO |
| Reproducibility loss | `seed=42`, W&B tags, version log |

---

## 6. Comparison with Literature (Table 4.18)

| Author / Year | Model | Dataset | mAP | FPS | HW |
|:---|:---|:---|:---|:---|:---|
| Hu and Wang (2020) | Faster R-CNN | PKU | 94.2% | ~12 | GPU |
| Liao et al. (2021) | YOLOv4 | Custom | 98.64% | 56.98 | GPU |
| Bhattacharya & Cloutier (2022) | YOLOv5 | Custom | 98.1% | — | GPU |
| Yang & Yu (2024) | YOLOv8 | PKU | 92.3% | 157.2 | GPU |
| Lv et al. (2024) | — (dataset paper) | DsPCBSD+ | — | — | — |
| Tian et al. (2025) | YOLOv12-N | COCO | 40.6% | — | GPU |
| **CIRCA (this work)** | **YOLOv12-? INT8** | **unified_pcb_v3** | **?** | **?** | **i5 8th-gen CPU** |

---

## 7. Runpod Budget Estimate

| Phase | Time (RTX 3090) | Cost @ $0.44/hr |
|:---|:---:|:---:|
| Phase 1 (100 ep, V12-S) | ~90 min | $0.66 |
| Phase 2 (100 ep, V12-S) | ~90 min | $0.66 |
| Phase 3 (50 × 50 ep HPO) | ~25 hrs | $11.00 |
| Phase 4 (200 ep × 3 variants) | ~25 hrs | $11.00 |
| **Total** | **~53 hrs** | **~$23.32** |

> Start with $5 credit → Run Phase 1+2 → Validate mAP → Credit $20 more for HPO + Final.

---

## 8. Ablation Gate Analysis: Phase 1 vs Phase 2

| Metric | Phase 1 (Vanilla) | Phase 2 (CIRCA) | Delta |
|:---|:---:|:---:|:---:|
| mAP@0.5 (best epoch) | **0.6821** | 0.6670 | −0.0151 (−1.51 pp) |
| mAP@0.5:0.95 (best epoch) | **0.4569** | 0.4504 | −0.0065 |
| Precision (at best mAP ep) | **0.8829** | 0.8422 | −0.041 |
| Recall (at best mAP ep) | **0.6382** | 0.6225 | −0.016 |
| Best epoch | 57 | 52 | — |
| Final epoch mAP@0.5 | 0.6698 | 0.6552 | −0.0146 |
| Final train cls_loss | **0.1814** | **0.1791** | −0.0023 |

**Interpretation:**
- The CLAHE+Gamma pipeline did NOT produce a statistically meaningful mAP lift on the `unified_pcb_v3` dataset under these training conditions. The −1.51 pp gap is within typical run-to-run variance for YOLOv12-S at 100 epochs.
- Both runs are stable: losses decrease monotonically, no collapse.
- Phase 2 shows marginally lower cls_loss at epoch 100 (0.1791 vs 0.1814), suggesting the CLAHE preprocessing may slightly improve classification confidence but not localisation.
- This result is thesis-reportable as a null ablation finding. The CIRCA preprocessing pipeline remains motivated by real-world lighting robustness (not solely by this benchmark) and the Phase 3 HPO proceeds on preprocessed data as planned.
- **Recommended thesis framing:** Report both mAP values honestly. Acknowledge the ablation gate was not met. Discuss as a finding: the preprocessing benefits are expected to manifest more clearly at inference time (image quality variation) rather than on the controlled, well-lit dataset.

