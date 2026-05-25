# CIRCA Training Engine User Guide (`train_engine.py`)

---

## 1. Overview

`train_engine.py` is a high-performance wrapper around the Ultralytics YOLOv12 framework:
- **Reproducibility:** Seeded runs (`seed=42`) and full version logging.
- **CIRCA Preprocessing:** Automated CLAHE + Gamma correction pipeline.
- **Hyperparameter Optimisation:** Integrated genetic tuner with thesis-aligned search space.
- **Deployment-Ready Exports:** Automatic OpenVINO INT8 IR export after each training run.
- **Multi-GPU Support:** Optimised for Runpod RTX 3090 (24 GB VRAM) for training; local RTX 3060 Laptop (6 GB VRAM) for development.

---

## 2. Modes of Operation

### 2.1 Training Mode (`--mode train`)
Resumes automatically if `last.pt` is found. Outputs `best.pt`, `last.pt`, training curves, and OpenVINO INT8 export.

### 2.2 Tuning Mode (`--mode tune`)
Genetic hyperparameter sweep. Outputs `best_hyperparameters.yaml`.

---

## 3. Command Reference

```powershell
python train_engine.py --mode [train|tune] --variant [n|s|m] --id [ID] --desc [Description] [OPTIONS]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mode` | string | `train` | `train` or `tune` |
| `--variant` | string | `s` | `n`, `s`, or `m` |
| `--id` | string | **Required** | Experiment ID (e.g., `001`) |
| `--desc` | string | **Required** | Short description |
| `--epochs` | int | `100` | Training epochs |
| `--iterations` | int | `50` | HPO trials (tune mode only) |
| `--imgsz` | int | `640` | Resolution |
| `--batch` | int | `24` | Batch size (RTX 3090 default for S variant) |
| `--data` | string | `datasets/unified_pcb_v3/data.yaml` | Dataset config path |
| `--preproc` | flag | `False` | Enable CLAHE + Gamma preprocessing |
| `--force-preproc` | flag | `False` | Regenerate preprocessed dataset cache |
| `--cfg` | string | `None` | Path to `best_hyperparameters.yaml` |
| `--cls-pw` | float | `1.0` | Inverse-frequency class weighting |
| `--clear` | flag | `False` | Delete existing experiment folder |
| `--device` | string | `0` | GPU device ID or `cpu` |

---

## 4. Key Capabilities

### 4.1 CIRCA Preprocessing Pipeline (`--preproc`)
1. **CLAHE:** L-channel of LAB, clip limit 2.0, tile 8×8.
2. **Gamma Correction:** Fixed at γ = 1.2.
3. **Disk Caching:** Saved to `unified_pcb_v3_preproc/`.

> **SolDef_AI polygon note:** Polygon annotations in SolDef_AI are auto-converted to axis-aligned bounding boxes by Roboflow during export. No manual conversion needed.

### 4.2 Class Imbalance Mitigation
`--cls-pw 1.0`: Inverse-frequency weighting for solder classes (4–6) which are underrepresented relative to bare-board classes (0–3).

### 4.3 W&B Integration
Project: `circa-yolov12`. Auto-tags: variant, mode, preprocessing status.

### 4.4 Automated Export
Locates `best.pt` → exports to **OpenVINO INT8 IR**.

---

## 5. Phase Command Examples (Runpod RTX 3090)

### Phase 1 — Vanilla Baseline (100 epochs — same as Phase 2 for OFAT comparison)
```powershell
python train_engine.py --mode train --variant s --id 001 --desc Baseline_Vanilla `
    --epochs 100 --batch 24 `
    --data datasets/unified_pcb_v3/data.yaml
```

### Phase 2 — CIRCA Baseline (100 epochs — OFAT, only preprocessing changes)
```powershell
python train_engine.py --mode train --variant s --id 002 --desc Baseline_CIRCA `
    --epochs 100 --preproc --batch 24 `
    --data datasets/unified_pcb_v3/data.yaml
```

### Phase 3 — HPO (run on already-preprocessed corpus — NO --preproc flag)
```powershell
python train_engine.py --mode tune --variant s --id 003 --desc HPO_7class `
    --epochs 50 --iterations 50 --fraction 0.5 --batch 24 `
    --data datasets/unified_pcb_v3_preproc/data.yaml
```

### Phase 4 — Final Training (N/S/M variants)
```powershell
# YOLOv12-N
python train_engine.py --mode train --variant n --id 004 --desc Final_HPO `
    --epochs 200 --batch 32 `
    --data datasets/unified_pcb_v3_preproc/data.yaml `
    --cfg runs/detect/CIRCA_V12S_003_TUNE_HPO_7class/best_hyperparameters.yaml

# YOLOv12-S
python train_engine.py --mode train --variant s --id 005 --desc Final_HPO `
    --epochs 200 --batch 24 `
    --data datasets/unified_pcb_v3_preproc/data.yaml `
    --cfg runs/detect/CIRCA_V12S_003_TUNE_HPO_7class/best_hyperparameters.yaml

# YOLOv12-M
python train_engine.py --mode train --variant m --id 006 --desc Final_HPO `
    --epochs 200 --batch 16 `
    --data datasets/unified_pcb_v3_preproc/data.yaml `
    --cfg runs/detect/CIRCA_V12S_003_TUNE_HPO_7class/best_hyperparameters.yaml
```

> **Batch sizes for local RTX 3060 (6 GB VRAM):** N=16, S=12, M=6. Use these for development/debugging only.

---

## 6. Troubleshooting

| Issue | Fix |
|---|---|
| GPU OOM (RTX 3060) | Reduce `--batch` (try 8 or 4) |
| Stale preprocessing cache | Run `--force-preproc` |
| W&B hang | Run `wandb login` manually |
| Missing OpenVINO | `pip install openvino openvino-dev nncf` |
| `AttributeError: VideoWidget has no clear_feed` | Ensure `VideoWidget.clear_feed()` is defined in `widgets.py` |
