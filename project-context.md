# CIRCA Project Context & Local Memory

> [!NOTE]
> This file acts as the local memory layer for the CIRCA project. 
> All future AI assistants (Antigravity, Claude, Gemini, Cline) must read this file at startup to align with the current architecture, recent decisions, and performance constraints.

---

## 🤖 AI Agent Self-Updating Memory Rule

To prevent context drift and preserve localized repository knowledge across sessions, all AI agents MUST adhere to the following memory maintenance protocol:

1. **When to Update:** You MUST update this file (`project-context.md`) immediately after any of the following events:
   * **Dataset Structural Changes:** Changes in class mappings, capping ratios, oversampling multipliers, or folder paths.
   * **Phase Transitions:** Transitioning to a new experiment phase (e.g., commencing Phase 3 HPO or completing Phase 4 training).
   * **Critical Findings & Verdicts:** Populating ablation gate verdicts, significant metric swings, or new deployment benchmarks.
   * **Architectural/System Constraint Updates:** New hardware profiles, code consolidations (e.g., deleting obsolete scripts), or IDE safety guardrails.
2. **Writing Rules:**
   * Keep edits focused, clean, and highly structured.
   * Use alerts (`> [!NOTE]`, `> [!IMPORTANT]`, etc.) for critical changes.
   * Avoid removing historical context of the dataset evolution unless it's completely deprecated.
3. **Traceability:** Always log the date/timestamp of the last update and specify the agent's name who made the change.

> **Last Updated:** 2026-07-08 by **Antigravity** — Refactored core architecture from live video streaming to static image inspection with tiled inference and PCB scene guard. ImageInspectWidget fully replaces VideoWidget, and all 483 tests pass.

---

## 📋 Project Overview

*   **Name:** CIRCA (PCB Defect Detection System)
*   **Domain:** Computer vision for industrial quality control, specifically detecting printed circuit board (PCB) defects.
*   **Core Stack:** YOLOv12 (Object Detection), SAHI (Slicing Aided Hyper Inference), OpenCV, PySide/Qt (Diagnostics GUI), and WandB (Experiment Tracking).
*   **Workspace Root:** `d:\FYP\CIRCA`

---

## ⚙️ Critical System Constraints (Crucial)

> [!IMPORTANT]
> **CPU & Memory Safety Rule (IDE Crash Mitigation):**
> High-throughput image preprocessing using `multiprocessing` can completely saturate CPU and memory bandwidth, causing Electron-based IDEs to crash. 
>
> *   **Rule:** When spawning parallel workers in `train_engine.py` or any preprocessing script, **NEVER** use unlimited `cpu_count() - 1` on local runs (always cap at 8).
> *   **Implementation:** Always cap workers at a maximum of `8` for local runs using:
>     `n_workers = min(8, max(1, cpu_count() - 1))`
> *   Do not change this without explicit user confirmation.

---

## 📊 Dataset & Balance Strategy (Frozen v4)

*   **Unified Dataset Location:** `datasets/unified_pcb_v3` (rebuilt from source folders PKU, DsPCBSD+, SolDef_AI, etc. via `scripts/prepare_all_datasets.py`).
*   **Dominant-Class Capping:** Capped dominant-only images (only `short` and `insufficient_solder` annotations) at 1,000 maximum per tier. This successfully reduced the annotation imbalance ratio from **9.9:1 to 5.7:1**.
*   **Class Imbalance Handling (Oversampling Tiers):**
    *   **Dominant Classes (Excluded from oversampling):** `insufficient_solder` (class 5) and `short` (class 3). 
    *   **Minority Classes (Strategic 5× Oversampling):** 
        *   `missing_hole` (class 0) -> Promoted to **5×** oversampling (was getting 0% recall in preliminary baseline).
        *   `excess_solder` (class 4) -> Promoted to **5×** oversampling (stands at 1,400 annotations).
        *   `cold_solder_joint` (class 6) -> Maintained at **5×** oversampling (1,185 annotations).

---

## 📈 Completed Experiment Phases & Metrics

### Phase 1 — Vanilla Baseline (Control) ✅ COMPLETE
*   **Command:** 
    ```bash
    python train_engine.py --mode train --variant s --id 001 --desc Baseline_Vanilla --epochs 100 --batch 48 --cache --data datasets/unified_pcb_v3/data.yaml
    ```
*   **Best Metrics (Epoch 45):** 
    *   `mAP@0.5`: **0.6649**
    *   `mAP@0.5:0.95`: **0.4237**
    *   `Precision`: **0.7290** | `Recall`: **0.6433**
*   **WandB Link:** [CIRCA_V12S_001_TRAIN_Baseline_Vanilla_2](https://wandb.ai/aidilzin-universiti-teknologi-mara/circa-yolov12/runs/19mp6dku)

### Phase 2 — CIRCA Baseline (CLAHE + Gamma) ✅ COMPLETE
*   **Command:** 
    ```bash
    python train_engine.py --mode train --variant s --id 002 --desc Baseline_CIRCA --epochs 100 --batch 48 --cache --data datasets/unified_pcb_v3_preproc/data.yaml
    ```
*   **Best Metrics (Epoch 50, Early stopped at Epoch 80):** 
    *   `mAP@0.5`: **0.6600** (within run-to-run noise of -0.49 pp)
    *   `mAP@0.5:0.95`: **0.4284** (**+0.47 pp** lift ✅)
    *   `Precision`: **0.8443** (**+11.5 pp** lift ✅) | `Recall`: **0.6341**
*   **WandB Link:** [CIRCA_V12S_002_TRAIN_Baseline_CIRCA_2](https://wandb.ai/aidilzin-universiti-teknologi-mara/circa-yolov12/runs/9e3zawyw)

### ⚖️ Ablation Gate Verdict (v4 Dataset)
> **Nuanced Null with Classification Signal:** 
> CLAHE + Gamma preprocessing does not broadly lift general mAP@0.5 on a highly controlled benchmark, but consistently improves **Precision (+11.5 pp)**, classification confidence, and lowers validation classification loss (`val cls_loss`) from epoch 40 onward. The cleaner input features also produce faster convergence, triggering early stopping at epoch 80.

---

## 🔮 Active & Future Phases (Next Steps)

### Phase 3 — Genetic HPO (RTX 3090) ✅ COMPLETE
*   **Duration:** ~23.4 hours (84,246 seconds), 50/50 iterations
*   **Best fitness:** `0.26305` (mAP@0.5:0.95) found at **iteration 42**
*   **Best trial metrics:** Precision: 0.716 | Recall: 0.420 | mAP@50: 0.435 | mAP@50-95: 0.263
*   **Output:** `runs/detect/CIRCA_V12S_003_TUNE_HPO_7class/best_hyperparameters.yaml`
*   **Key tuned values (significant changes from default):**
    *   `lr0`: 0.01 → **0.00014** (very low LR for subtle PCB features)
    *   `momentum`: 0.937 → **0.785** (more responsive updates)
    *   `box`: 7.5 → **0.169** (less localization dominance)
    *   `cls`: 0.5 → **0.266** (lower classification loss weight)
    *   `mosaic`: 1.0 → **0.722** (reduced — unrealistic for PCB domain)
    *   `scale`: 0.5 → **0.650** (more aggressive scale augmentation)

### Phase 4 — Three-Variant Final Training ✅ COMPLETE
*   **Goal:** Train final YOLOv12-N, YOLOv12-S, and YOLOv12-M variants for 200 epochs each using the hyperparameter profile tuned in Phase 3 on RunPod RTX 3090.
*   **Tuned Parameters:** Commits HPO yaml to the runs.
*   **Validation Metrics:**
    *   **YOLOv12-Nano (FP16):** mAP@0.5 = **63.13%**, mAP@0.5:0.95 = **39.52%**, Precision = **83.16%**, Recall = **60.23%**
    *   **YOLOv12-Small (FP16):** mAP@0.5 = **66.20%**, mAP@0.5:0.95 = **42.97%**, Precision = **73.06%**, Recall = **67.00%**
    *   **YOLOv12-Medium (FP16):** mAP@0.5 = **67.42%**, mAP@0.5:0.95 = **43.89%**, Precision = **74.78%**, Recall = **67.07%**

### Phase 5 — OpenVINO Quantisation ✅ COMPLETE
*   **Goal:** Export checkpoints to OpenVINO FP32/FP16/INT8 formats and evaluate on validation split.
*   **Verdict:** All variants failed the absolute validation target (mAP@0.5 ≥ 90.0%), triggering a global fallback from INT8 to FP16 to preserve detection accuracy and avoid quantization degradation.
*   **Quantised Validation (mAP@0.5 FP32 vs. INT8):**
    *   Nano: FP32 = 63.13% → INT8 = 61.97% (failed absolute and delta target of -1.15 pp)
    *   Small: FP32 = 66.20% → INT8 = 66.30% (failed absolute target)
    *   Medium: FP32 = 67.42% → INT8 = 67.09% (failed absolute target)

### Phase 6 — Hardware Benchmarking ✅ COMPLETE
*   **Goal:** Benchmark latency, static-image processing time, and tiled inference latency CPU vs. integrated GPU.
*   **Deployment Target:** Intel CPU / AMD iGPU virtual machine representing the Core i5 8th-gen target environment.
*   **Verdict:** YOLOv12-Nano FP16 is selected as the production model because it satisfies all speed criteria (preprocessing latency ≤ 5 ms, static image analysis time ≤ 10 s) while maintaining low inference latency when tiled:
    *   **YOLOv12-Nano (CPU):** Preproc = 4.68 ms, Inference = 28.34 ms, Tiled (1080p, 9 tiles) = ~255 ms, Static Time = 0.101 s
    *   **YOLOv12-Nano (GPU):** Preproc = 4.77 ms, Inference = 19.84 ms, Tiled (1080p, 9 tiles) = ~178 ms, Static Time = 0.089 s

### Phase 7 — Test Evaluation & Threshold Calibration ✅ COMPLETE
*   **Goal:** Evaluate the chosen production variant (YOLOv12-Nano FP16) on the frozen test split, and run confidence sweeps on the validation split.
*   **Test Metrics (Single evaluation pass, 1,270 images):**
    *   mAP@0.5: **62.79%** | mAP@0.5:0.95: **38.34%**
    *   Precision: **85.70%** | Recall: **60.59%** | F1-Score: **70.99%**
    *   Solder classes achieved high recall (insufficient solder = 91.44%, cold solder joint = 85.71%), while the minority class missing hole experienced absolute recall suppression (0.00%) due to sub-resolution boundary smearing under class scarcity.
*   **Thresholds Calibrated (`circa_thresholds.yaml`):**
    *   Display: `missing_hole` = 0.10, `mouse_bite` = 0.60, `open_circuit` = 0.50, `short` = 0.40, `excess_solder` = 0.60, `insufficient_solder` = 0.50, `cold_solder_joint` = 0.10.
    *   Warning: All set to 0.10.


---

## 📂 Restored Chat Trajectories
For detailed transcripts and deep technical dialogues from previous sessions, refer directly to the consolidated log:
*   📄 **[restored_conversations_history.md](file:///d:/FYP/CIRCA/restored_conversations_history.md)**

---

## 🐛 Key Bugs Found & Fixed (2026-06-04 Review Session)

> [!IMPORTANT]
> The following **production bugs** (not just test issues) were discovered and fixed during the full project review. Future AI agents must be aware of these patterns.

### 1. `ui/image_inspect_widget.py` — `QPen` Constructor Crash (Critical)
*   **Bug:** `_draw_single_box` passed `Qt.PenJoinStyle.MiterJoin` as the 3rd argument to `QPen(color, width, style)`. PyQt6 requires `Qt.PenStyle` here. This caused a **Windows process crash (0xC0000409)** any time bounding boxes were rendered.
*   **Fix:** Create pen with `Qt.PenStyle.SolidLine`, then call `pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)` separately.

### 2. `core/utils.py` — Denylist Blocked Index 1 (Silent Skip)
*   **Bug:** Denylist entry `"Camera 1"` matched the auto-generated fallback label `"Camera 1 (USB)"` for index 1, silently skipping it on every scan without Qt names.
*   **Fix:** Only apply the denylist when `has_real_name is True` (real OS name from QtMultimedia).

### 3. `core/inference_engine.py` — `_preprocess_frame` API
*   **API:** Returns `(tensor, pad_x, pad_y, scale)` tuple. Callers must unpack.

### 4. Test Suite Status (2026-06-04)
*   **457 / 457 passing** (`pytest tests/ -k "not test_train_engine and not integration_smoke"`)
*   CI workflow: `.github/workflows/ci.yml` runs full suite on push/PR.

### 5. Repository Cleanup & Weights Consolidation (2026-06-22)
*   **Obsolete scripts deleted:** Cleaned up one-time patch/download scripts (`embed_graphs.py`, `patch_slides.py`, `download_references.py`).
*   **Scattered files organized:** Moved pre-trained YOLO weights (`yolo12n.pt`, `yolo12s.pt`) to the `models/` directory, and vertical architecture PNG diagrams to `docs/assets/`.
*   **Weights Fallback:** Patched `train_engine.py` to search in `models/` as a fallback when looking for YOLO base weights.

### 6. Agent Customizations & Rule Consolidation (2026-06-30)
*   **Skills Setup:** Curated, compared, and deployed 18 optimal developer & domain-specific skills (including custom `caveman-talk`, `use-markitdown`, and `academic-thesis-writing` skills) under `.antigravitycli/`, `.claude/`, `.gemini/`, and `.github/`. Removed outdated `bmad` configs.
*   **Conventions Sync:** Consolidated workspace-level custom system rules in [AGENTS.md](file:///d:/FYP/CIRCA/.agents/AGENTS.md) to keep Antigravity aligned, and synchronized `.cursorrules` and `.clinerules`.
*   **RunPod Optimization:** Deleted obsolete S3 bucket rules; established persistent pod volume disk `/workspace/` rules. Cap workers at 8 only for local runs to prevent IDE crashes, while leveraging full AMD EPYC performance on RunPod runs.

### 7. Phase 5–7 Scripts Created — Block A Complete (2026-07-01)
*   **`scripts/evaluate_quantization.py`** — Phase 5: exports best.pt → FP32/FP16/INT8 IR, applies Ch3 §3.6.4 fallback rule, writes `docs/quantization_report.md`.
*   **`scripts/benchmark.py`** — Phase 6: preproc latency, inference latency (CPU/GPU), tiled throughput, static time. Writes `docs/benchmark_report.md` + throughput chart.
*   **`scripts/calibrate_thresholds.py`** — Phase 7: conf sweep on val split → per-class thresholds → `circa_thresholds.yaml` + sweep plot.
*   **`scripts/test_evaluate.py`** — Phase 7: one-shot test split eval → `docs/test_evaluation.md` + figures.
*   All 4 scripts syntax-validated (ALL OK).

### 8. Cleanup and Dependencies Verification (2026-07-01)
*   **Further Cleanup:** Deleted obsolete S3 upload script (`upload_to_runpod.py`), progress presentation scripts (`generate_slides.py`), presentation slides/talking points (`CIRCA_Progress_Report_Presentation.pptx`, `Talking_Points.md/.pdf`, `circa_project_review.md`), and temporary logs.
*   **YOLOv12m weights:** Verified that `models/yolo12m.pt` is downloaded and resolved locally (ready for Phase 4-M).
*   **RunPod Permissions Fix:** Patched `scripts/runpod_setup.sh` to automatically apply recursive `chmod -R 755` on the `datasets/` folder during setup, resolving directory traversal failures and PermissionError issues stemming from Windows-zipped folders.

### 9. Production Model Layout & GUI Fixes (2026-07-01 Antigravity)
*   **OpenVINO 2026.0.0 Compatibility:** Added try-except fallback import block to support both `openvino.runtime` (2024.x) and `openvino` top-level namespace (2026.x).
*   **End-to-End Layout Support:** Resolved model coordinate-to-class bounding box corruptions (e.g. `CLASS 235 63101%`) by implementing runtime tensor shape layout detection (raw anchor `[1, 11, 8400]` vs. end-to-end `[1, 300, 6]`).
*   **Dynamic Model Selector:** Added a GUI combobox selector to Optimisation Sidebar allowing the user to select and load any available YOLOv12 variant (Nano, Small, Medium) in FP16 or INT8 formats dynamically.
*   **HD Webcam Support & Denoising:** Configured the `cv2.VideoCapture` thread to request 1280x720 (720p HD) resolution from webcam hardware. Integrated a fast `cv2.GaussianBlur` noise-reduction gate to smooth sensor graininess without losing edge details.
*   **Dynamic Auto-Optimisation:** Added histogram-based auto-optimisation of CLAHE and Gamma parameters per image based on brightness/contrast standard deviation, with an optional manual override checkbox in the sidebar.
*   **Model Hardware Suitability Diagnostics:** Built a non-blocking 5-pass startup/manual benchmark routine in `InferenceWorker` to auto-detect execution latency on the host CPU/GPU and select the best matching variant (Nano vs. Small vs. Medium).
*   **Thesis & Reference Corrections:** Addressed all 9 examiner corrections including PKU dataset referencing, validation AP discussion updates, standardising Table 4.8 IPC references, and correcting test metrics precision in summary chapters.
*   **Denoising Pipeline Optimisation:** Reordered the camera processing thread to apply bilateral denoising *before* the Laplacian variance blur gate, ensuring sharpness checks run on clean, noise-suppressed frames to prevent camera sensor graininess from false-triggering the gate.
*   **Noise Amplification Mitigation:** Capped the dynamic auto-optimisation CLAHE clip limit to a maximum of `2.2`, avoiding severe webcam graininess blow-up in low-contrast environments.
*   **Calibrated Blur Default:** Changed default `blur_threshold` from `100.0` to the calibrated `12.5` (precision 1) in `PreprocessParams` and `SidePanel` UI, permitting successful detection on re-photographed boards (e.g. phone screen).
*   **On-by-Default Controls:** Enabled dynamic auto-optimisation of CLAHE/Gamma and automatically trigger hardware suitability benchmarking on application startup.

### 10. Core Refactoring to Static Image Inspection (2026-07-08 Antigravity)
*   **Static Image Inspection:** Replaced live video streaming feed with static image inspection workflow. Drag-and-drop / file-picker zone is shown by default.
*   **ImageInspectWidget:** Replaced the legacy `VideoWidget` with a modern `ImageInspectWidget` supporting EMPTY, LOADING, and RESULT states. The legacy widget was removed.
*   **PCB Scene Guard:** Integrated a heuristic-based scene validation guard to reject non-PCB images before running model inference.
*   **Tiled Inference Engine:** Integrated an adaptive tiled inference engine to run YOLOv12 models on overlapping tiles, solving scale-mismatch issues on high-resolution full-board images.
*   **Cleaned Up Codebase:** Renamed and removed all video streaming references and continuous capture/inference connections across the app, test suite, and thesis.

### Current Phase Status (2026-07-08)
| Phase | Status | Notes |
|:--|:--:|:--|
| 0 — Dataset rebuild | ✅ Done | unified_pcb_v3, 7-class |
| 1 — Vanilla baseline | ✅ Done | mAP@0.5=0.6649 |
| 2 — CIRCA baseline | ✅ Done | mAP@0.5=0.6600, P=0.8443 |
| 3 — HPO | ✅ Done | fitness=0.26305, iter 42/50 |
| 4 — Final training (N/S/M, 200ep) | ✅ Done | RunPod RTX 3090, 200 epochs completed |
| 5 — OpenVINO quantisation | ✅ Done | FP32/FP16/INT8 evaluate_quantization.py |
| 6 — Hardware benchmarking | ✅ Done | CPU/GPU latencies and FPS benchmark.py |
| 7 — Test eval + thresholds | ✅ Done | YOLOv12-Nano FP16 calibrated test_evaluate.py |



