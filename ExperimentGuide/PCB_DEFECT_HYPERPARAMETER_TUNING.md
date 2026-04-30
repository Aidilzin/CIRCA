# CIRCA — YOLOv12 Hyperparameter Tuning, Training & Deployment Playbook (v2)

> **Project:** CIRCA — Circuit Inspection and Recognition using Convolutional Architectures
> **Purpose:** Single source of truth for the AI assistant working on this project. Aligned with the CIRCA Chapter 1 / Chapter 2 thesis scope: YOLOv12 + Intel OpenVINO INT8 deployment + CLAHE/Gamma/Laplacian preprocessing + IPC-A-610-aligned defect taxonomy + repair-context evaluation.

---

## 0. Project Context

- **Task:** PCB defect **object detection** in **electronics repair** environments (uncontrolled lighting, webcam input).
- **Model family:** **YOLOv12** (Ultralytics).
- **Variants under comparative study:** `yolov12n`, `yolov12s`, `yolov12m` — the assistant must train and benchmark **all three** to identify the optimal accuracy/latency/size trade-off (per Objective 2 of the thesis).
- **Deployment backend:** Intel **OpenVINO INT8 IR** (primary), **OpenVINO FP16 IR** (fallback if INT8 mAP drops below 90%).
- **Target hardware:** Intel Core i5 8th-gen-equivalent CPU + integrated GPU on Windows 10/11. **No dedicated GPU at inference time.**
- **Input source:** Standard USB webcam or smartphone camera tether (no industrial imaging gear).
- **Preprocessing pipeline (inference and training):** **CLAHE → Gamma Correction → Laplacian Variance frame gate** (OpenCV).

### Acceptance Criteria (from CIRCA Chapter 1 §1.5)

| Metric | Target | Measured On |
|---|---|---|
| mAP@0.5 (test set) | **> 90%** | Curated PCB test set |
| Preprocessing latency | **≤ 5 ms / frame** | Intel Core i5 8th-gen CPU |
| Live inference rate | **≥ 15 FPS** | Webcam @ 640×480 or 1280×720 |
| Static image inference | **≤ 10 s** | Single image, full pipeline |
| Defect classes | IPC-A-610 aligned | See §1.2 |

> **Assistant rule:** Every experiment, model export, and benchmark run must report against these four numbers. Do not declare a configuration "final" until all four pass.

---

## 1. Prerequisites Checklist

### 1.1 Environment

```bash
pip install ultralytics opencv-python numpy pyyaml
pip install openvino openvino-dev nncf       # OpenVINO + Neural Network Compression Framework (for INT8)
# pip install wandb                             # optional experiment tracking
# Optional on Ampere+ NVIDIA cards used for training:
pip install flash-attn --no-build-isolation
```

- [x] `torch.cuda.is_available()` returns `True` on the **training** machine.
- [x] `python -c "import openvino as ov; print(ov.__version__)"` ≥ 2024.x.
- [ ] Intel CPU with AVX2 / VNNI (8th-gen i5 or later) for INT8 acceleration on the **deployment** machine.
- [x] Reproducibility: pass `seed=42` to every `model.train()` and `model.tune()` call.

### 1.2 IPC-A-610-Aligned Defect Taxonomy

The `data.yaml` must encode classes that map directly to IPC-A-610 acceptability criteria and the CIRCA scope. The corpus combines bare-board defects (IPC-A-600 classes 0–9, retained for completeness) and assembly-stage solder defects (IPC-A-610 classes 10–14):

```yaml
# data.yaml
path: D:/FYP/CIRCA/datasets/unified_pcb_v2
train: train/images
val:   valid/images
test:  test/images

nc: 15
names:
  0:  missing_hole
  1:  mouse_bite
  2:  open_circuit
  3:  short
  4:  spur
  5:  spurious_copper
  6:  hole_breakout
  7:  conductor_scratch
  8:  conductor_foreign_object
  9:  base_material_foreign_object
  10: excess_solder
  11: insufficient_solder
  12: solder_spike
  13: solder_bridge
  14: cold_solder_joint
```

> The dataset uses a unified 15-class taxonomy established in `class_mapping.md`. Classes 0–9 derive from the original PKU + DsPCBSD+ + Roboflow PCB Defect bare-board sources; classes 10–14 derive from the SolDef_AI (Fontana et al., 2024) and PCB Solder Joint (Work, 2025) assembly-stage sources added in the v2 expansion. All experiments must adhere to this ontology. See `CIRCA_DATASET_EXPANSION_PLAN.md` for source-by-source remap details.
>
> **Out-of-scope (documented limitation):** component-level IPC-A-610 defects (missing component, misalignment, tombstoning, lifted lead, solder ball, component damage) are excluded due to absence of suitable public datasets. See thesis Chapter 1 §1.5 and Chapter 3 §3.1.

### 1.3 Dataset

- [x] **YOLO format** (one `.txt` per image, normalized `class x_center y_center w h`).
- [x] Folder structure (standardized via `train_engine.py --preproc`):
  ```
  unified_pcb_preproc/
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt
  data.yaml
  ```
- [x] Splits: **70 / 15 / 15** (strictly rebalanced) — test set frozen.
- [x] Per-class instance counts logged; minority classes oversampled or up-weighted via `cls_pw`. Solder classes (10–14) start with `cls_pw=1.0` to compensate for the bare-board dominance (~10:1 imbalance) using inverse-frequency power.
- [x] Duplicate / near-duplicate detection across splits (perceptual hash verified; 8,566 images purged in v1; v2 expansion will re-run pHash dedup across the new sources).
- [x] **Repair-context images present** — a non-trivial portion of the dataset must be captured under uncontrolled lighting (desklamp glare, shadows, off-angle webcam) per Lv et al. (2024).
- [ ] **Negative (background) samples deferred** — the dropped `good`/`GOOD` images from SolDef_AI + PCB Solder Joint are held in reserve. Add to training only if Phase 1 baseline shows excessive false positives on the validation set.

### 1.4 Preprocessing Pipeline (Automated via Train Engine)

To prevent train/inference distribution mismatch, the same OpenCV pipeline used at runtime is applied to training images via the `train_engine.py --preproc` flag.

**Methodology:**
- **CLAHE:** Applied to L-channel of LAB (Clip Limit: 2.0, Tile: 8x8).
- **Gamma Correction:** Factor of **1.2**.
- **Execution:** Performed once as a standalone pass to generate the `_preproc` dataset variant.

> **Assistant rule:** Never tune or evaluate on raw images while deploying on preprocessed ones. Always use the `--preproc` flag for CIRCA-aligned results.

### 1.5 Repair-Context Augmentation Profile

In addition to Ultralytics defaults, simulate the conditions described in Hu & Wang (2020) and Liao et al. (2021):

| Effect | Implementation | Range |
|---|---|---|
| Desklamp glare | random bright Gaussian blob overlay | 5–20% area, intensity +60–120 |
| Shadow casting | random dark gradient overlay | 10–30% area, intensity −40–100 |
| Motion blur | OpenCV motion blur kernel | length 3–9 px |
| Off-axis capture | small perspective transform | ≤ 0.001 (matches `perspective` param) |

Apply via Ultralytics' built-in params where possible (`hsv_v`, `perspective`) and add an **Albumentations** integration for glare/shadow/motion-blur if needed.

### 1.6 Tooling

- [x] Weights & Biases project: `circa-yolov12`.
- [x] Tag every run with the variant (`n` / `s` / `m`) and the precision (`fp32` / `fp16` / `int8`).
- [x] `runs/` directory committed to git-ignored storage; logs synced to W&B.


---

## 2. Hyperparameter Search Space (YOLOv12 / Ultralytics)

Tune the high-impact group first; expand only if budget permits.

| Parameter | Type | Range | Notes |
|---|---|---|---|
| `lr0` | float | 1e-5 → 1e-2 (log) | Initial LR |
| `lrf` | float | 0.01 → 1.0 | Final LR factor |
| `momentum` | float | 0.7 → 0.98 | Or Adam β1 |
| `weight_decay` | float | 0.0 → 1e-3 | L2 |
| `warmup_epochs` | float | 0.0 → 5.0 | |
| `warmup_momentum` | float | 0.0 → 0.95 | |
| `box` | float | 0.02 → 0.2 | Box loss gain |
| `cls` | float | 0.2 → 4.0 | Class loss gain |
| `cls_pw` | float | 0.0 → 1.0 | **Push higher to upweight minority IPC classes** |
| `dfl` | float | 0.4 → 12.0 | Distribution focal loss |
| `hsv_h` | float | 0.0 → 0.05 | Keep low — PCB color matters |
| `hsv_s` | float | 0.0 → 0.7 | |
| `hsv_v` | float | 0.0 → 0.9 | **Wide range simulates glare/shadow** |
| `degrees` | float | 0.0 → 10.0 | Small — boards are roughly aligned |
| `translate` | float | 0.0 → 0.2 | |
| `scale` | float | 0.0 → 0.5 | |
| `shear` | float | 0.0 → 2.0 | Small |
| `perspective` | float | 0.0 → 0.001 | Webcam off-axis |
| `flipud` | float | 0.0 → 0.5 | |
| `fliplr` | float | 0.0 → 0.5 | |
| `mosaic` | float | 0.5 → 1.0 | |
| `mixup` | float | 0.0 → 0.15 | |
| `copy_paste` | float | 0.0 → 0.3 | Useful for rare defects |
| `close_mosaic` | int | 5 → 15 | |

**Fixed during tuning:** `optimizer=AdamW`, `imgsz=640`, `batch` per-variant (see §3).

---

## 3. Three-Variant Comparative Tuning Protocol

The thesis requires comparing **YOLOv12-N / S / M**. Tune each independently, then compare on the same test set.

### 3.1 Per-Variant Batch Sizes (starting points)

CIRCA training hardware is an **NVIDIA RTX 3060 6 GB** (laptop/desktop class). Batch sizes are tuned for that VRAM ceiling at `imgsz=640` with AMP enabled. The 12 GB / 24 GB columns are kept for reference if a larger GPU becomes available.

| Variant | Params | `batch` (RTX 3060 6 GB — CIRCA target) | `batch` (12 GB GPU) | `batch` (24 GB+) |
|---|---|---|---|---|
| `yolov12n` | ~2.6M | 16 | 32 | 64 |
| `yolov12s` | ~9M | **12** (CIRCA default) | 16 | 32 |
| `yolov12m` | ~20M | 6 | 8 | 16 |

If any variant hits CUDA OOM, drop the batch size by 2–4 and retry — `train_engine.py` already passes `nbs=64` so the effective gradient accumulation target stays consistent.

### 3.2 Tuning Code (run per variant)

```python
from ultralytics import YOLO

VARIANT = "yolov12s"   # also run for yolov12n and yolov12m

model = YOLO(f"{VARIANT}.pt")

search_space = {
    "lr0":           (1e-5, 1e-2),
    "lrf":           (0.01, 1.0),
    "momentum":      (0.7, 0.98),
    "weight_decay":  (0.0, 1e-3),
    "warmup_epochs": (0.0, 5.0),
    "box":           (0.02, 0.2),
    "cls":           (0.2, 4.0),
    "cls_pw":        (0.0, 1.0),
    "dfl":           (0.4, 12.0),
    "hsv_v":         (0.0, 0.9),
    "degrees":       (0.0, 10.0),
    "translate":     (0.0, 0.2),
    "scale":         (0.0, 0.5),
    "fliplr":        (0.0, 0.5),
    "mosaic":        (0.5, 1.0),
    "mixup":         (0.0, 0.15),
    "copy_paste":    (0.0, 0.3),
}

model.tune(
    data="data.yaml",
    epochs=30,
    iterations=150,         # 100–200 is typical
    optimizer="AdamW",
    space=search_space,
    plots=True,
    save=False,
    val=True,
    seed=42,
    name=f"tune_{VARIANT}",
)
```

Outputs land in `runs/detect/tune_<variant>/`:
- `best_hyperparameters.yaml`
- `tune_results.csv`
- `tune_scatter_plots.png`, `tune_fitness.png`

### 3.3 Final Training (per variant)

```python
from ultralytics import YOLO

VARIANT = "yolov12s"
model = YOLO(f"{VARIANT}.pt")
model.train(
    data="data.yaml",
    cfg=f"runs/detect/tune_{VARIANT}/best_hyperparameters.yaml",
    epochs=200,
    imgsz=640,
    batch=12,            # RTX 3060 6 GB safe default; raise to 16/32 on larger GPUs
    optimizer="AdamW",
    patience=30,
    close_mosaic=10,
    amp=True,
    cos_lr=True,
    seed=42,
    name=f"final_{VARIANT}",
)
```

Best checkpoint: `runs/detect/final_<variant>/weights/best.pt`.

---

## 4. OpenVINO Export & INT8 Quantization

This is the **deployment backbone** of CIRCA. Run for each of the three trained variants.

### 4.1 FP32 → OpenVINO IR (sanity check)

```python
from ultralytics import YOLO
model = YOLO("runs/detect/final_yolov12s/weights/best.pt")
model.export(format="openvino", imgsz=640, half=False)   # FP32 IR
# produces best_openvino_model/  (.xml + .bin)
```

### 4.2 FP16 IR (fallback target)

```python
model.export(format="openvino", imgsz=640, half=True)    # FP16 IR
```

### 4.3 INT8 IR via NNCF Post-Training Quantization (primary)

Ultralytics supports INT8 export directly, which uses NNCF under the hood and requires a **calibration dataset**:

```python
model.export(
    format="openvino",
    imgsz=640,
    int8=True,
    data="data.yaml",   # used for calibration; ~300 images sampled from train split
)
# produces best_int8_openvino_model/
```

Calibration tips:
- 200–500 representative images from the **train** split (never val/test).
- Include images covering all defect classes and lighting conditions.
- If NNCF complains about layer coverage, fall back to FP16.

### 4.4 Post-Quantization Validation (REQUIRED)

Validate every exported model on the **val** set before touching the test set.

```python
from ultralytics import YOLO

for path in [
    "runs/detect/final_yolov12s/weights/best.pt",                          # PT FP32
    "runs/detect/final_yolov12s/weights/best_openvino_model",              # OV FP32
    "runs/detect/final_yolov12s/weights/best_openvino_model_fp16",         # OV FP16
    "runs/detect/final_yolov12s/weights/best_int8_openvino_model",         # OV INT8
]:
    m = YOLO(path)
    metrics = m.val(data="data.yaml", split="val", imgsz=640)
    print(path, metrics.box.map50, metrics.box.map)
```

### 4.5 INT8 → FP16 Fallback Decision Rule

| INT8 mAP@0.5 (val) | Decision |
|---|---|
| ≥ baseline FP32 mAP − 1% **and** ≥ 90% absolute | Use **INT8** in production |
| < FP32 − 1% **but** ≥ 90% | Acceptable; use INT8 unless latency budget allows FP16 |
| < 90% | **Fall back to FP16 IR** |

Document the decision in `quantization_report.md` with the exact numbers.

---

## 5. Hardware Benchmark Protocol

Run on the **deployment** machine — Intel Core i5 8th-gen-equivalent (or the user's actual target).

### 5.1 Preprocessing Latency

```python
import time, cv2, numpy as np
from circa_preproc import circa_preprocess, is_sharp

img = cv2.imread("sample.jpg")
N = 1000
t0 = time.perf_counter()
for _ in range(N):
    if is_sharp(img):
        _ = circa_preprocess(img)
elapsed_ms = (time.perf_counter() - t0) * 1000 / N
print(f"Preproc per frame: {elapsed_ms:.2f} ms  (target ≤ 5 ms)")
```

### 5.2 Inference Latency (OpenVINO)

```python
from ultralytics import YOLO
import time

m = YOLO("best_int8_openvino_model")
img = "sample.jpg"

# warmup
for _ in range(10):
    m.predict(img, imgsz=640, device="cpu", verbose=False)

N = 100
t0 = time.perf_counter()
for _ in range(N):
    m.predict(img, imgsz=640, device="cpu", verbose=False)
inf_ms = (time.perf_counter() - t0) * 1000 / N
print(f"Inference per frame: {inf_ms:.2f} ms ({1000/inf_ms:.1f} FPS)")
```

For **iGPU**, set `device="intel:gpu"` (OpenVINO plugin).

### 5.3 End-to-End Live FPS (webcam)

Run a 60-second webcam loop with full preprocess + inference and log the moving-average FPS. Acceptance: **≥ 15 FPS**.

### 5.4 Static Image Pipeline

Single high-resolution image through full pipeline (preproc + inference + draw overlays). Acceptance: **≤ 10 s**.

### 5.5 Variant Selection Matrix

After all three variants are trained, exported, and benchmarked, fill in:

| Variant | Precision | mAP@0.5 (test) | mAP@0.5:0.95 | Preproc (ms) | Inference (ms, CPU) | Live FPS | Static (s) | Model size (MB) | Pass? |
|---|---|---|---|---|---|---|---|---|---|
| yolov12n | INT8 |  |  |  |  |  |  |  | |
| yolov12n | FP16 |  |  |  |  |  |  |  | |
| yolov12s | INT8 |  |  |  |  |  |  |  | |
| yolov12s | FP16 |  |  |  |  |  |  |  | |
| yolov12m | INT8 |  |  |  |  |  |  |  | |
| yolov12m | FP16 |  |  |  |  |  |  |  | |

**Selection rule:** highest mAP among configurations that pass all four acceptance criteria.

---

## 6. Confidence Threshold Calibration ("Manual Inspection Required")

Per Goddard et al. (2011) and Kupfer et al. (2023), CIRCA must transparently flag low-confidence detections to mitigate automation bias.

### 6.1 Per-Class Threshold Sweep

Sweep `conf` from 0.10 → 0.90 in steps of 0.05 on the **val** set. For each class, plot precision–recall and pick:
- **Per-class display threshold** = the conf where precision ≥ 0.90 (do not show boxes below this).
- **Per-class warning threshold** = the conf where recall ≥ 0.95 (boxes between warning and display thresholds are shown faded with a "low confidence" tag).

### 6.2 Global "Manual Inspection Required" Trigger

The system should display a screen-level warning when:
- Mean detection confidence across all visible boxes < **0.50**, **OR**
- Laplacian variance of the current frame < `BLUR_THRESHOLD`, **OR**
- No boxes detected for ≥ 1 s while a board is clearly in frame (heuristic; optional).

Save the calibrated thresholds to `circa_thresholds.yaml` and load them at runtime.

---

## 7. Final Test-Set Evaluation (run **once**)

```python
from ultralytics import YOLO
m = YOLO("best_int8_openvino_model")     # winning variant + precision
metrics = m.val(
    data="data.yaml",
    split="test",
    imgsz=640,
    save_json=True,
    plots=True,
    conf=0.001,                           # report full PR curves; thresholds applied at display time
)
print(metrics.box.map, metrics.box.map50, metrics.box.maps)
```

Report:
- Per-class precision, recall, F1.
- Overall mAP@0.5, mAP@0.5:0.95.
- Confusion matrix.
- PR / F1 curves.
- Failure-case gallery: small defects, glare, shadow, motion blur, off-angle.
- Latency (CPU and iGPU), live FPS, static seconds, preprocessing milliseconds.

---

## 8. Deliverables

The assistant must produce:

1. `runs/detect/tune_yolov12{n,s,m}/best_hyperparameters.yaml` — three tuning outputs.
2. `runs/detect/final_yolov12{n,s,m}/weights/best.pt` — three trained models.
3. OpenVINO IR exports (FP32, FP16, INT8) for each variant.
4. `quantization_report.md` — per-variant FP32 vs FP16 vs INT8 mAP table + INT8 ↔ FP16 fallback decision.
5. `benchmark_report.md` — variant selection matrix from §5.5 with the chosen winner highlighted.
6. `circa_thresholds.yaml` — per-class display + warning thresholds and global trigger rule.
7. `tuning_report.md` — top-N trials, scatter plots, parameter importance commentary per variant.
8. `test_evaluation.md` — final test-set metrics, per-class breakdown, confusion matrix, qualitative failure gallery.
9. `class_mapping.md` — mapping from raw dataset labels to the IPC-A-610 taxonomy.
10. `runs/` directory with all experiment logs (W&B-synced).

---

## 9. Guardrails for the Assistant

- **Never** look at the test set during tuning or threshold calibration. Calibrate on val only.
- **Never** report a metric without naming the split, the variant, and the precision (FP32/FP16/INT8).
- **Always** log seeds, dataset version, Ultralytics version, OpenVINO version, NNCF version with each run.
- **Always** apply the same CLAHE → Gamma preprocessing to the data the model was trained on. Train ↔ inference distribution mismatch is the most common failure mode.
- **Always** run the post-quantization validation step (§4.4) before declaring an INT8 model usable.
- **Always** cite the trial ID / iteration when claiming "best result".
- If preprocessing latency exceeds 5 ms, profile with `cProfile` before tuning hyperparameters — it is almost always a bad colorspace conversion or LUT path.
- If INT8 mAP collapses, check (a) calibration set diversity, (b) layer coverage warnings from NNCF, (c) whether `model.fuse()` was called, before retraining.
- If a step is ambiguous (variant scope, dataset path, hardware target), **stop and ask** before consuming GPU time.
- Tuning trials must mirror final training settings (same dataset, same preprocessing, comparable augmentations) — short coco8-style tuning runs do **not** transfer reliably.
- The CIRCA system is a **decision-support tool**, not an authority. Confidence transparency and the "Manual Inspection Required" trigger are **non-negotiable** features.
