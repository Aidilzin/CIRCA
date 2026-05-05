# CIRCA — Experiment Plan

**Project:** CIRCA — Circuit Inspection and Recognition using Convolutional Architectures
**Model:** YOLOv12 | **Deployment:** Intel OpenVINO INT8

---

## 1. Objectives

1. Identify and document IPC-A-610-aligned PCB defect types for automated detection.
2. Design, train, and compare **YOLOv12-N/S/M** with Intel OpenVINO INT8 deployment.
3. Evaluate the CIRCA preprocessing pipeline and desktop application using P, R, F1, mAP, and latency.

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

| Phase | ID | Description | Epochs | Goal |
|:---|:---|:---|:---|:---|
| 0 | — | Dataset setup (12-class `unified_pcb_v2`) | — | Reproducible corpus |
| 1 | 001 | Vanilla YOLOv12s, raw images | 50 | Ablation control |
| 2 | 002 | CLAHE+Gamma preprocessed | 100 | Preprocessing lift |
| 3 | 003 | Genetic HPO (50 iter) | 30/iter | Optimal config |
| 4 | 004–006 | Final N/S/M with HPO config | 200 | Three final variants |
| 5 | — | OpenVINO FP32/FP16/INT8 export | — | Quantisation delta |
| 6 | — | Hardware benchmark on i5 8th-gen | — | Latency validation |
| 7 | — | Frozen test evaluation | — | Final thesis metrics |

---

## 4. Run Commands

```bash
# Phase 0 — Build unified_pcb_v2 (dedup, remap, split)

# Phase 1
python train_engine.py --mode train --variant s --id 001 --desc Baseline_Vanilla --epochs 50 --clear --data datasets/unified_pcb_v2/data.yaml

# Phase 2
python train_engine.py --mode train --variant s --id 002 --desc Baseline_CIRCA --epochs 100 --preproc --data datasets/unified_pcb_v2/data.yaml

# Phase 3
python train_engine.py --mode tune --variant s --id 003 --desc HPO --epochs 30 --iterations 50 --fraction 0.3 --preproc --data datasets/unified_pcb_v2/data.yaml

# Phase 4
python train_engine.py --mode train --variant n --id 004 --desc Final_HPO --epochs 200 --preproc --data datasets/unified_pcb_v2/data.yaml --cfg runs/detect/CIRCA_V12S_003_TUNE_HPO/best_hyperparameters.yaml
python train_engine.py --mode train --variant s --id 005 --desc Final_HPO --epochs 200 --preproc --data datasets/unified_pcb_v2/data.yaml --cfg runs/detect/CIRCA_V12S_003_TUNE_HPO/best_hyperparameters.yaml
python train_engine.py --mode train --variant m --id 006 --desc Final_HPO --epochs 200 --preproc --data datasets/unified_pcb_v2/data.yaml --cfg runs/detect/CIRCA_V12S_003_TUNE_HPO/best_hyperparameters.yaml
```

---

## 5. Risks & Mitigations

| Risk | Mitigation |
|:---|:---|
| Preprocessing cache stale | `--force-preproc` |
| INT8 mAP collapse | FP16 fallback |
| Solder class under-detection | `cls_pw` inv-freq weighting + `mosaic=1.0` + `copy_paste=0.3`; conditional ≤3× solder-only oversample if Phase 2 mAP@0.5 < 70% |
| GPU OOM | `batch=12`; drop to YOLOv12-N |
| Reproducibility loss | `seed=42`, W&B tags, version log |

---

## 6. Comparison with Literature (Table 4.18)

| Author / Year | Model | Dataset | mAP | FPS | HW |
|:---|:---|:---|:---|:---|:---|
| Hu and Wang (2020) | Faster R-CNN | PKU | 94.2% | ~12 | GPU |
| Liao et al. (2021) | YOLOv4 | Custom | 98.64% | 56.98 | GPU |
| Bhattacharya & Cloutier (2022) | YOLOv5 | Custom | 98.1% | — | GPU |
| Yang & Yu (2024) | YOLOv8 | PKU | 92.3% | 157.2 | GPU |
| Tian et al. (2025) | YOLOv12-N | COCO | 40.6% | — | GPU |
| **CIRCA (this work)** | **YOLOv12-? INT8** | **CIRCA corpus** | **?** | **?** | **i5 8th-gen CPU** |
