# CIRCA — PCB Defect Detection System

> **C**omputer vision **I**nspection system for **R**eal-time **C**ircuit board **A**nalysis  
> FYP — Universiti Teknologi MARA | YOLOv12 · OpenVINO INT8 · PyQt6 · IPC-A-600/610H

---

## Overview

CIRCA is a real-time PCB defect detection system that runs a quantised YOLOv12 INT8 model via Intel OpenVINO on a standard repair-bench PC. It detects **7 defect classes** spanning both bare-board (IPC-A-600) and assembly-stage solder (IPC-A-610H) defects from a live UVC camera feed.

| Class | Standard | Description |
|:---|:---|:---|
| `missing_hole` | IPC-A-600 §3.4 | Drill breakthrough failure |
| `mouse_bite` | IPC-A-600 §3.3 | Board edge notching |
| `open_circuit` | IPC-A-600 §3.2 | Broken conductor trace |
| `short` | IPC-A-600 §3.2 | Unintended conductor bridge |
| `excess_solder` | IPC-A-610H §5 | Solder volume above spec |
| `insufficient_solder` | IPC-A-610H §5 | Solder volume below spec |
| `cold_solder_joint` | IPC-A-610H §5 | Dull/granular joint |

---

## Architecture

```
Camera (UVC)
    │ cv2.VideoCapture (DirectShow)
    ▼
CameraWorker (Camera Thread)
    ├─ CLAHE (LAB L-channel)
    ├─ Gamma correction (γ=1.2)
    └─ Laplacian variance blur gate
         │ sharp frames only
         ▼
InferenceWorker (Inference Thread)
    └─ InferenceEngine (OpenVINO INT8)
         ├─ Letterbox resize → 640×640
         ├─ YOLOv12 INT8 inference (CPU)
         └─ NMS → BoundingBox list
              │
              ▼
         MainWindow (GUI Thread)
              ├─ VideoWidget (bounding box overlay)
              ├─ WarningBanner (low-confidence alert)
              └─ StatusFooter (camera/model/detection status)
```

---

## Requirements

- Python 3.10+
- Windows 10/11 (for DirectShow UVC camera support)
- Intel CPU with iGPU (OpenVINO INT8 optimised for Intel CPU/iGPU)
- 16 GB RAM recommended for `--cache` during training

**Training additionally requires:**
- NVIDIA GPU (RTX 3090 recommended; 6 GB minimum)
- CUDA 12.x + cuDNN

---

## Installation

```powershell
# Clone the repository
git clone <repo-url>
cd CIRCA

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## Running the GUI

> **Prerequisite:** A trained and exported OpenVINO model must exist at `models/yolov12_int8.xml`.  
> See [Training](#training) below to produce this file.

```powershell
python main.py
```

The GUI will start, enumerate available cameras, and begin live defect detection automatically.

### GUI Controls (Side Panel)

| Control | Function |
|:---|:---|
| Camera selector | Switch between available UVC cameras |
| Contrast slider | CLAHE clip limit (1.0–8.0) |
| Brightness slider | Gamma correction (0.5–2.5) |
| Sharpness Gate | Laplacian variance blur threshold |
| Sensitivity slider | Model confidence threshold (10–95%) |
| Theme toggle | Dark / Light / System |

---

## Training

Training is managed by `train_engine.py`. All experiments follow the 7-phase programme documented in [`docs/CIRCA_EXPERIMENT_PLAN.md`](docs/CIRCA_EXPERIMENT_PLAN.md).

### Phase 1 — Vanilla Baseline

```bash
python train_engine.py \
    --mode train --variant s --id 001 --desc Baseline_Vanilla \
    --epochs 100 --batch 48 --cache \
    --data datasets/unified_pcb_v3/data.yaml
```

### Phase 2 — CIRCA Baseline (CLAHE + Gamma)

```bash
python train_engine.py \
    --mode train --variant s --id 002 --desc Baseline_CIRCA \
    --epochs 100 --batch 48 --cache \
    --data datasets/unified_pcb_v3_preproc/data.yaml
```

### Phase 3 — Hyperparameter Optimisation

```bash
python train_engine.py \
    --mode tune --variant s --id 003 --desc HPO_7class \
    --epochs 50 --iterations 50 --fraction 0.5 --batch 48 --cache \
    --data datasets/unified_pcb_v3_preproc/data.yaml
```

### Full CLI Reference

```
python train_engine.py --help
```

### SAHI Sliced Inference (high-resolution boards)

```bash
python sahi_infer.py \
    --model runs/detect/<EXPERIMENT>/weights/best.pt \
    --source path/to/image.jpg
```

---

## Dataset

The unified dataset (`unified_pcb_v3`) merges 6 public PCB defect sources with class remapping, pHash deduplication, and stratified 70/15/15 splitting.

```bash
# Preview dataset counts (no files written)
python scripts/build_unified_pcb_v3.py --dry-run

# Build full dataset
python scripts/build_unified_pcb_v3.py
```

See [`docs/CIRCA_CLASS_MAPPING.md`](docs/CIRCA_CLASS_MAPPING.md) for the full class taxonomy and source dataset provenance.

---

## Testing

```powershell
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=core --cov=workers --cov=ui

# Run a specific test file
pytest tests/test_inference_engine.py -v
```

---

## Project Structure

```
CIRCA/
├── main.py                    # GUI entry point
├── train_engine.py            # Training & HPO engine
├── sahi_infer.py              # SAHI sliced inference
├── requirements.txt           # Local (GUI + training) dependencies
├── requirements_runpod.txt    # RunPod (headless training) dependencies
│
├── core/
│   ├── inference_engine.py    # OpenVINO INT8 inference + letterbox
│   ├── preprocessor.py        # CLAHE (LAB) + Gamma preprocessing
│   ├── utils.py               # Camera enumeration + BGR→QImage
│   ├── models.py              # BoundingBox, DetectionResult, Params
│   └── debug.py               # trace_execution decorator
│
├── workers/
│   ├── camera_worker.py       # UVC capture + preprocessing thread
│   └── inference_worker.py    # Event-driven inference thread
│
├── ui/
│   ├── main_window.py         # Application shell
│   ├── sidebar.py             # Activity bar + side panel (VS Code style)
│   ├── image_inspect_widget.py # Static PCB inspection panel + bounding box overlay
│   ├── warning_banner.py      # Low-confidence advisory bar
│   ├── status_footer.py       # Camera/model/detection status bar
│   ├── theme.py               # Design tokens + QSS builder
│   ├── analytics_dashboard.py # Live performance analytics and charts
│   └── help_dialog.py         # Onboarding tutorial and user instructions
│
├── scripts/                   # Dataset build & RunPod utilities
├── docs/                      # Experiment plans, checklists, thesis diagrams
└── tests/                     # pytest suite (unit + integration + UI)
```

---

## Experiment Results

| Phase | Dataset | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|:---|:---|:---:|:---:|:---:|:---:|
| Phase 1 — Vanilla Baseline | unified_pcb_v3 | 0.6649 | 0.4237 | 0.7290 | 0.6433 |
| Phase 2 — CIRCA (CLAHE+γ) | unified_pcb_v3_preproc | 0.6600 | 0.4284 | **0.8443** | 0.6341 |
| Phase 3 — HPO (YOLOv12-S) | unified_pcb_v3_preproc | 0.4350 (fitness 0.263) | 0.2631 | 0.7161 | 0.4196 |
| Phase 4 — Final N (FP16) | unified_pcb_v3_preproc | 0.6313 | 0.3952 | 0.8316 | 0.6023 |
| Phase 4 — Final S (FP16) | unified_pcb_v3_preproc | 0.6620 | 0.4297 | 0.7306 | 0.6700 |
| Phase 4 — Final M (FP16) | unified_pcb_v3_preproc | 0.6742 | 0.4389 | 0.7478 | 0.6707 |
| **Phase 7 — Test (N FP16)** | unified_pcb_v3 test split | **0.6279** | **0.3834** | **0.8570** | **0.6059** |

Full experiment log: [`docs/CIRCA_EXPERIMENT_CHECKLIST.md`](docs/CIRCA_EXPERIMENT_CHECKLIST.md)  
Training flow diagrams: [`CIRCA_TRAINING_FLOW.md`](CIRCA_TRAINING_FLOW.md)

---

## License

Academic use only — FYP submission, Universiti Teknologi MARA.
