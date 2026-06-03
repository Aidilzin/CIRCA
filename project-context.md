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

> **Last Updated:** 2026-06-04 by **Antigravity (Gemini 2.5 Pro)**

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
> *   **Rule:** When spawning parallel workers in `train_engine.py` or any preprocessing script, **NEVER** use unlimited `cpu_count() - 1`. 
> *   **Implementation:** Always cap workers at a maximum of `8` using:
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

### Phase 3 — Genetic HPO (RTX 3090, ~12–14 Hours) ⏳ PENDING
*   **Goal:** Run 50 iterations of genetic tuning to find optimal hyperparameters on the preprocessed v4 dataset.
*   **Run command (to be run inside a `tmux` session):**
    ```bash
    tmux new -s circa_training
    python train_engine.py --mode tune --variant s --id 003 --desc HPO_7class --epochs 50 --iterations 50 --fraction 0.5 --batch 48 --cache --data datasets/unified_pcb_v3_preproc/data.yaml
    ```

### Phase 4 — Three-Variant Final Training ⏳ PENDING
*   **Goal:** Train final YOLOv12-N, YOLOv12-S, and YOLOv12-M variants for 200 epochs each using the hyperparameter profile tuned in Phase 3.

---

## 📂 Restored Chat Trajectories
For detailed transcripts and deep technical dialogues from previous sessions, refer directly to the consolidated log:
*   📄 **[restored_conversations_history.md](file:///d:/FYP/CIRCA/restored_conversations_history.md)**

---

## 🐛 Key Bugs Found & Fixed (2026-06-04 Review Session)

> [!IMPORTANT]
> The following **production bugs** (not just test issues) were discovered and fixed during the full project review. Future AI agents must be aware of these patterns.

### 1. `ui/video_widget.py` — `QPen` Constructor Crash (Critical)
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
