# CIRCA — YOLOv12 Hyperparameter Tuning, Training & Deployment Playbook

> **Project:** CIRCA — Circuit Inspection and Recognition using Convolutional Architectures
> **Dataset:** `unified_pcb_v3` (7-class, nc=7)

---

## 0. Project Context

- **Task:** PCB defect detection in electronics repair environments.
- **Models:** `yolov12n`, `yolov12s`, `yolov12m`.
- **Deployment:** Intel OpenVINO INT8 IR (primary); FP16 fallback if INT8 mAP drops > 1% below FP32 or falls below 90%.
- **Target HW (inference):** Intel Core i5 8th-gen CPU + iGPU (no dedicated GPU).
- **Training HW:** Runpod RTX 3090 (24 GB VRAM) for cloud training.
- **Preprocessing:** CLAHE → Gamma Correction → Laplacian Variance frame gate.

### Acceptance Criteria

| Metric | Target |
|:---|:---|
| mAP@0.5 (test) | > 90% |
| Preprocessing latency | ≤ 5 ms |
| Live FPS | ≥ 15 |
| Static inference | ≤ 10 s |

---

## 1. Prerequisites

### 1.1 Environment

```bash
pip install ultralytics opencv-python numpy pyyaml openvino openvino-dev nncf imagehash Pillow scikit-learn
```

### 1.2 7-Class Taxonomy

```yaml
path: D:/FYP/CIRCA/datasets/unified_pcb_v3
train: train/images
val:   valid/images
test:  test/images
nc: 7
names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: excess_solder
  5: insufficient_solder
  6: cold_solder_joint
```

- Classes 0–3: IPC-A-600 bare-board (PKU/JR, DsPCBSD+)
- Classes 4–6: IPC-A-610H assembly-stage solder (SolDef_AI, kydra, v2-s89jo, Hue)
- Out-of-scope (documented in §1.5 and §3.4.1.5): `spur`, `spurious_copper`, `solder_spike`, `scratch`, `pinhole` (data-availability-driven exclusion)

### 1.3 Dataset Spec

- YOLO format, 70/15/15 stratified split, `seed=42`, test frozen.
- Built via `scripts/build_unified_pcb_v3.py`.

### 1.4 Preprocessing Pipeline

- **CLAHE:** L-channel LAB, clip limit 2.0, tile 8×8.
- **Gamma:** γ = 1.2.
- **Cached** to `unified_pcb_v3_preproc/` (one-time offline pass via `--preproc` on Phase 2 run).
- **SolDef_AI polygon note:** Exported via Roboflow "YOLOv8 Object Detection" format — polygons are auto-converted to axis-aligned bounding boxes. No custom conversion required.

---

## 2. Hyperparameter Search Space

| Parameter | Range | Notes |
|:---|:---|:---|
| `lr0` | 1e-5 → 1e-3 (log) | Initial LR |
| `lrf` | 0.01 → 1.0 | Final LR factor |
| `momentum` | 0.7 → 0.98 | |
| `weight_decay` | 0.0 → 1e-3 | |
| `warmup_epochs` | 0.0 → 5.0 | |
| `box` | 1.0 → 10.0 | |
| `cls` | 0.2 → 4.0 | |
| `cls_pw` | 0.1 → 1.0 | Solder class weighting |
| `dfl` | 0.4 → 12.0 | |
| `hsv_v` | 0.0 → 0.9 | Glare/shadow — kept; `hsv_h`/`hsv_s` excluded (copper hue is diagnostic) |
| `degrees` | 0.0 → 10.0 | Boards are roughly top-down aligned |
| `translate` | 0.0 → 0.3 | |
| `scale` | 0.0 → 0.9 | |
| `fliplr` | 0.0 → 0.5 | |
| `mosaic` | 0.0 → 1.0 | |
| `mixup` | 0.0 → 0.2 | |
| `copy_paste` | 0.0 → 0.3 | Helps minority solder classes |

---

## 3. Training Protocol

### Batch Sizes

| Variant | RTX 3090 (24 GB) | RTX 3060 Laptop (6 GB) |
|:---|:---:|:---:|
| `yolov12n` | 32 | 16 |
| `yolov12s` | 24 | 12 |
| `yolov12m` | 16 | 6 |

### Ablation Training Config (Phase 1 & 2)

- `epochs=100`, `imgsz=640`, `optimizer=AdamW`, `cos_lr=True`, `amp=True`
- `seed=42`, `close_mosaic=10`, `patience=30`
- **Phase 1:** no preprocessing flag | **Phase 2:** `--preproc` flag
- **Equal epochs justification:** OFAT ablation — only the preprocessing variable changes between Phase 1 and Phase 2.

### HPO Config (Phase 3)

- `epochs=50` per trial, `iterations=50`, `fraction=0.5`
- Run on `unified_pcb_v3_preproc/` (already preprocessed — do NOT use `--preproc` flag)

### Final Training Config (Phase 4)

- `epochs=200`, `imgsz=640`, `optimizer=AdamW`, `cos_lr=True`, `amp=True`
- `patience=30`, `close_mosaic=10`, `seed=42`
- Uses `best_hyperparameters.yaml` from Phase 3

---

## 4. OpenVINO Export & INT8 Quantisation

```python
model.export(format="openvino", imgsz=640, half=False)           # FP32
model.export(format="openvino", imgsz=640, half=True)            # FP16
model.export(format="openvino", imgsz=640, int8=True,
             data="datasets/unified_pcb_v3/data.yaml")           # INT8
```

### Fallback Rule

| INT8 mAP@0.5 | Decision |
|:---|:---|
| ≥ FP32 − 1% **and** ≥ 90% | Use INT8 |
| < 90% or drop > 1% | Fall back to FP16 |

---

## 5–9. Benchmarking, Calibration, Evaluation, Deliverables, Guardrails

*(See `CIRCA_EXPERIMENT_PLAN.md` and `CIRCA_EXPERIMENT_CHECKLIST.md` for full phase commands and criteria.)*
