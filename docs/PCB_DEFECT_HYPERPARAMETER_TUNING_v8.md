# CIRCA — YOLOv12 Hyperparameter Tuning, Training & Deployment Playbook

> **Project:** CIRCA — Circuit Inspection and Recognition using Convolutional Architectures

---

## 0. Project Context

- **Task:** PCB defect detection in electronics repair environments.
- **Models:** `yolov12n`, `yolov12s`, `yolov12m`.
- **Deployment:** Intel OpenVINO INT8 IR (primary); FP16 fallback if INT8 mAP < 90%.
- **Target HW:** Intel Core i5 8th-gen CPU + iGPU (no dedicated GPU at inference).
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
pip install ultralytics opencv-python numpy pyyaml openvino openvino-dev nncf
```

### 1.2 12-Class Taxonomy

```yaml
path: D:/FYP/CIRCA/datasets/unified_pcb_v2
train: train/images
val:   valid/images
test:  test/images
nc: 12
names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: spur
  5: spurious_copper
  6: excess_solder
  7: insufficient_solder
  8: solder_spike
  9: cold_solder_joint
  10: scratch
  11: pinhole
```

Classes 0–5: PKU/JR, Bare PCB (IPC-A-600). Classes 6–9: SolDef_AI, kydra, f8m5i, Hue, v2-s89jo (IPC-A-610H). [5 assembly sources]
Out-of-scope: component-level IPC-A-610 defects (missing component, misalignment, etc.) — see Chapter 1 §1.5.

### 1.3 Dataset Spec

- YOLO format, 70/15/15 stratified split, `seed=42`, test frozen.
- `cls_pw=1.0` for solder class weighting.

### 1.4 Preprocessing Pipeline

- CLAHE: L-channel LAB, clip 2.0, tile 8×8.
- Gamma: γ = 1.2.
- Cached to `unified_pcb_v2_preproc/`.

### 1.5 Repair-Context Augmentation

| Effect | Range |
|:---|:---|
| Desklamp glare | 5–20% area, +60–120 intensity |
| Shadow | 10–30% area, −40–100 intensity |
| Motion blur | 3–9 px kernel |
| Off-axis | perspective ≤ 0.001 |

---

## 2. Hyperparameter Search Space

| Parameter | Range | Notes |
|:---|:---|:---|
| `lr0` | 1e-5 → 1e-2 (log) | Initial LR |
| `lrf` | 0.01 → 1.0 | Final LR factor |
| `momentum` | 0.7 → 0.98 | |
| `weight_decay` | 0.0 → 1e-3 | |
| `warmup_epochs` | 0.0 → 5.0 | |
| `box` | 0.02 → 0.2 | |
| `cls` | 0.2 → 4.0 | |
| `cls_pw` | 0.1 → 1.0 | |
| `dfl` | 0.4 → 12.0 | |
| `hsv_v` | 0.0 → 0.9 | Glare/shadow |
| `degrees` | 0.0 → 10.0 | |
| `translate` | 0.0 → 0.3 | |
| `scale` | 0.0 → 0.9 | |
| `fliplr` | 0.0 → 0.5 | |
| `mosaic` | 0.0 → 1.0 | |
| `mixup` | 0.0 → 0.2 | |
| `copy_paste` | 0.0 → 0.3 | |

---

## 3. Training Protocol

### Batch Sizes (RTX 3060 6 GB)

| Variant | `batch` |
|:---|:---|
| `yolov12n` | 16 |
| `yolov12s` | 12 |
| `yolov12m` | 6 |

### Final Training Config

- `epochs=200`, `imgsz=640`, `optimizer=AdamW`, `cos_lr=True`, `amp=True`
- `patience=30`, `close_mosaic=10`, `seed=42`

---

## 4. OpenVINO Export & INT8 Quantisation

```python
model.export(format="openvino", imgsz=640, half=False)           # FP32
model.export(format="openvino", imgsz=640, half=True)            # FP16
model.export(format="openvino", imgsz=640, int8=True,
             data="datasets/unified_pcb_v2/data.yaml")           # INT8
```

### Fallback Rule

| INT8 mAP@0.5 | Decision |
|:---|:---|
| ≥ FP32 − 1% and ≥ 90% | Use INT8 |
| < 90% | Fall back to FP16 |

---

## 5–9. Benchmarking, Calibration, Evaluation, Deliverables, Guardrails

*(See full playbook — sections unchanged from original.)*
