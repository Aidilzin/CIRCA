# Unified PCB Defect Dataset v2

## Overview
The `unified_pcb_v2` dataset is a comprehensive collection of printed circuit board (PCB) defect images. It integrates multiple open-source datasets to provide a robust training set for object detection models, specifically targeting a 12-class taxonomy aligned with IPC-A-600 and IPC-A-610 standards.

## Data Sources
The dataset is composed of 8 active source datasets spanning bare-board and assembly stages:
1. **SolDef_AI** (Fontana et al., 2024)
2. **PKU-Market-PCB via JR**
3. **RAHUL PCB Defects**
4. **Bare PCB Defects**
5. **Hue**
6. **solder-f8m5i-xnbzp**
7. **Excessive Solder / kydra**
8. **PCB Solder Defect Detection V2 (v2-s89jo)**

*Note: Two candidate datasets — `pcb-deffect-detection-solder-lsb7m` and `pcb-deffect-detection-solder-sthr7` — were evaluated but dropped after a quality audit (lsb7m was single-class and superseded by kydra/f8m5i/v2-s89jo; sthr7 had zero solder_spike instances and high annotation noise). The `solder_bridge` defect mode (IPC-A-610) is documented as future work due to no suitable public dataset.*

## Taxonomy
The dataset utilizes a 12-class unified taxonomy:
- `0`: missing_hole
- `1`: mouse_bite
- `2`: open_circuit
- `3`: short
- `4`: spur
- `5`: spurious_copper
- `6`: excess_solder
- `7`: insufficient_solder
- `8`: solder_spike
- `9`: cold_solder_joint
- `10`: scratch
- `11`: pinhole

## Data Preparation Pipeline

### 1. Data Cleaning & Integration
- **Label Remapping:** Labels from disparate datasets were standardized to the 12-class taxonomy. Invalid or out-of-scope labels (e.g., "good" class in SolDef_AI, or component-level defects) were dropped.
- **Filtering:** Images with dimensions smaller than 320x320 or missing labels were excluded and moved to a `negatives_reserve` pool.
- **Deduplication:** Perceptual hashing (pHash with a threshold of 6) was used to identify and remove duplicate or near-duplicate images across the combined dataset.

### 2. Offline Preprocessing (Data Transformation)
A preprocessed variant of the dataset (`unified_pcb_v2_preproc`) is generated with the following transformations to enhance feature visibility:
- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Applied in the LAB color space (L-channel only) with a clip limit of 2.0 and an 8x8 grid to improve local contrast without over-amplifying noise.
- **Gamma Correction:** A gamma value of 1.2 is applied to adjust overall image brightness and improve details in shadow regions.

### 3. Data Augmentation Strategy
**Important Note:** Previous versions of this dataset utilized offline oversampling to combat class imbalance. This strategy was dropped due to issues with overfitting (memorizing repeated pixel patterns), co-occurrence distortion (accidentally over-representing majority classes), and conflicts with modern data augmentation pipelines.

Instead, the dataset relies strictly on **online augmentations** built into the YOLO framework during the training phase. Specifically:
- **Mosaic:** Combines 4 training images into one in certain ratios, greatly expanding the context and reducing the need for large batch sizes.
- **MixUp:** Blends two images and their labels, smoothing the decision boundaries and improving generalization, especially for underrepresented classes.

This approach has been empirically shown to outperform offline oversampling for foreground-foreground class imbalance in single-stage detectors.

## Usage
To train with this dataset, point your YOLO configuration to the `data.yaml` file located in the root of the dataset directory (`datasets/unified_pcb_v2/data.yaml` or `datasets/unified_pcb_v2_preproc/data.yaml`).