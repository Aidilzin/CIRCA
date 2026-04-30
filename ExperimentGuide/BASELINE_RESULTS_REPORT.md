# Thesis Documentation: Baseline Experiment (Exp 001)

> **IMPORTANT:** This report contains results from an initial run on the non-sanitized dataset. These metrics are **invalidated** due to the discovery of 318 cases of data leakage. A new "Clean" baseline is required for Chapter 4.

## Chapter 3: Research Methodology (Draft Content)

### 3.X Experimental Setup and Model Training
For the initial baseline evaluation, the **YOLOv12s** architecture was selected due to its balance between computational efficiency and feature extraction capability via the Area Attention (A2) module. The model was trained on the `unified_pcb` dataset (18,694 images) utilizing a 70/15/15 stratified split.

**Hardware and Optimization:**
Training was conducted on an **NVIDIA RTX 3060 Laptop GPU (6GB VRAM)**. To maximize throughput without exceeding memory limits, a batch size of **12** and **4 DataLoader workers** were utilized. Mixed Precision (AMP) was enabled to accelerate training.

**Hyperparameter Strategy:**
To address initial numerical instability (EMA NaN warnings), a **Stable Training Protocol** was implemented:
- **Learning Rate:** Initial rate (`lr0`) was set to `0.001` (reduced from the default `0.01`).
- **Warmup Phase:** Extended to **5.0 epochs** to allow weight stabilization.
- **Optimizer:** AdamW with a cosine learning rate schedule.
- **Augmentation:** Mosaic augmentation was utilized for the first 40 epochs and disabled for the final 10 epochs (`close_mosaic=10`) to refine bounding box accuracy.

---

## Chapter 4: Results and Findings (Draft Content)

### 4.X Baseline Performance Analysis
The baseline experiment (`CIRCA_V12S_001_TRAIN_Baseline_Optimized`) reached convergence within 50 epochs. The final metrics on the validation set demonstrate a robust foundation for subsequent optimization.

**Final Validation Metrics (Epoch 50):**
| Metric | Value |
| :--- | :--- |
| **Precision** | 84.27% |
| **Recall** | 64.35% |
| **mAP@0.5** | 69.43% |
| **mAP@0.5:0.95** | 40.57% |

**Performance Commentary:**
The model achieved high precision (84.27%), suggesting a low false-positive rate—critical for industrial PCB inspection. The moderate recall (64.35%) indicates that while the model is accurate in its detections, it currently misses a subset of defects, likely the smallest categories or those with high occlusion. The `mAP@0.5` of 69.43% provides a strong baseline for the hyperparameter tuning phase, where we will focus on improving recall and overall detection sensitivity.

**Loss Convergence:**
Final training losses settled at:
- **Box Loss:** 1.514
- **Classification Loss:** 0.895
- **DFL Loss:** 1.185

These results confirm that the `unified_pcb` dataset is learnable and the 10-class ontology is statistically valid for YOLOv12 training.
