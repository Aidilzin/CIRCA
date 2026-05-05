# Thesis Documentation: Baseline Experiment (Exp 001)

## Chapter 3: Research Methodology (Draft)

### 3.7.1 Experimental Setup

**YOLOv12s** selected for its balance between efficiency and feature extraction (Area Attention module).
Trained on `unified_pcb_v2` (12-class, 70/15/15 stratified split, `seed=42`).

- **Hardware:** NVIDIA RTX 3060 Laptop GPU (6 GB VRAM)
- **Batch size:** 12 | **Workers:** 4 | **AMP:** enabled
- **Learning Rate:** `lr0=0.001` | **Warmup:** `warmup_epochs=5.0`
- **Optimizer:** AdamW with cosine LR schedule
- **Augmentation:** Mosaic for first 40 epochs, disabled for final 10 (`close_mosaic=10`)

---

## Chapter 4: Results (Draft)

### 4.3.1 Baseline Performance Analysis

| Metric | Value |
|:---|:---|
| **Precision** | ____________ |
| **Recall** | ____________ |
| **mAP@0.5** | ____________ |
| **mAP@0.5:0.95** | ____________ |

| Loss | Value |
|:---|:---|
| Box Loss | ____________ |
| Classification Loss | ____________ |
| DFL Loss | ____________ |

This baseline provides the ablation control for comparing against the CIRCA preprocessing pipeline
(Phase 2) and genetic HPO (Phase 3). Class imbalance between bare-board (0–5) and solder classes
(6–9) is the primary challenge expected at this stage.
