# CIRCA Workspace Custom Rules (Antigravity)

These rules are loaded by Antigravity for all tasks in the CIRCA workspace.

## 1. Local Memory Preservation & Updates (CRITICAL)
*   **Startup:** On startup, read BOTH:
    1. [project-context.md](file:///d:/FYP/CIRCA/project-context.md) — overall project state, datasets, constraints.
    2. [CIRCA_PROGRESS.md](file:///d:/FYP/CIRCA/CIRCA_PROGRESS.md) — live task checklist, current block status, metrics tracker.
*   **Auto-Update Rule:** Proactively update BOTH files after any of the following events:
    *   **Task Completion:** Mark completed tasks `[x]` in [CIRCA_PROGRESS.md](file:///d:/FYP/CIRCA/CIRCA_PROGRESS.md) and fill in results.
    *   **Dataset Changes:** Structural updates, capping ratios, oversampling shifts, or folder paths.
    *   **Phase Transitions:** Starting or finishing any experiment phase (Phase 1–7).
    *   **Key Findings:** Metric changes, ablation gate verdicts, or benchmarking results.
    *   **System Constraints:** Modifications to hardware configs or script patterns.
*   **Format:** Keep [project-context.md](file:///d:/FYP/CIRCA/project-context.md) neat, structured, and log your name and date at the top under the "Last Updated" row of the AI Agent Self-Updating Memory Rule section.


## 2. CPU & Memory Safety Guardrail (CRITICAL)
*   High-throughput image processing/training runs using `multiprocessing` can saturate CPU and memory bandwidth, crashing Electron-based IDEs on local systems.
*   **Rule:** When spawning parallel workers in `train_engine.py` or any preprocessing/utility script, determine the execution environment:
    *   **Local Runs:** Cap workers at a maximum of `8` (e.g., `n_workers = min(8, max(1, cpu_count() - 1))`) to safeguard local IDE stability.
    *   **RunPod Cloud Runs:** Do not cap at `8`. Leverage the full performance of the AMD EPYC CPU (e.g., `n_workers = max(1, cpu_count() - 1)`) since there is no local IDE crash concern and abundant RAM (116.46 GiB) is available.

## 3. Dataset Integrity
*   The dataset is frozen at version 4 (`unified_pcb_v3` / `unified_pcb_v3_preproc`). Do not make structural changes to splits or balance rules without explicit user instructions.
*   Any path configuration inside `data.yaml` files must use relative paths (`datasets/unified_pcb_v3...`) rather than Windows absolute paths (`D:/...`) to prevent FileNotFoundError on Linux RunPod containers.

## 4. RunPod Disk Volume Management
*   **Cost Control:** Upload the packaged dataset (`CIRCA_runpod.zip`) and configurations directly to the RunPod container's persistent volume disk `/workspace/` rather than using S3 buckets or network storage mounts to prevent storage cost creep.
*   **Staging & Packaging:** Use `python scripts/package_runpod.py` to compile dependencies and structure files correctly with forward slashes for Linux compatibility before uploading.

## 5. Automated Skill Integration
*   **Automatic Skill Lookup:** You have access to specialized skills in the `skills/` directories. You MUST check these directories to align on workflows for planning, requirements, code quality, TDD, Git commits, PyTorch model coding, and thesis writing.
*   **Token Optimization Rules:**
    *   **Document Processing:** Before parsing any large non-markdown files (`.docx`, `.pptx`, `.xlsx`, `.pdf`, `.html`), you MUST run the `use-markitdown` skill to convert them to clean Markdown to minimize input token usage.
    *   **Response Style:** If the user requests brief responses, or to minimize output tokens during long-running loops, automatically activate the `caveman-talk` skill to communicate in high-density, telegraphic format.

