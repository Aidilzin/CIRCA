# Graph Report - .  (2026-07-22)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 653 nodes · 1309 edges · 51 communities (44 shown, 7 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 106 edges (avg confidence: 0.55)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f40917d2`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- ThemeManager
- AnalyticsDashboard
- ImageInspectWidget
- PreprocessParams
- InferenceParams
- MainWindow
- oversample-dataset-minority-classes.py
- OnboardingController
- convert_markdown_to_docx
- evaluate_auto_optimisation.py
- generate_deck.py
- evaluate-model-quantization.py
- StatusFooter
- generate_copypaste_data.py
- train_engine.py
- InferenceEngine
- build-unified-dataset.py
- prepare-dataset-splits.py
- InferenceWorker
- HardwareScannerThread
- main.py
- HelpDialog
- CameraWorker
- test_ui_wiring.py
- NavButton
- .run_tiled
- calibrate-confidence-thresholds.py
- MockCore
- WarningBanner
- evaluate-final-model-metrics.py
- setup-runpod-environment.py
- monitor_and_cleanup.py
- ._on_mark_false_positive
- get_dataset_class_distribution
- get_ssh_connection_details
- test_onboarding.py
- ._on_image_loaded
- _format_arg
- package-runpod-deployment.py
- .average_confidence
- ._on_next_board_requested

## God Nodes (most connected - your core abstractions)
1. `MainWindow` - 57 edges
2. `ImageInspectWidget` - 47 edges
3. `InferenceParams` - 39 edges
4. `ThemeManager` - 37 edges
5. `SidePanel` - 34 edges
6. `AnalyticsDashboard` - 33 edges
7. `DetectionResult` - 30 edges
8. `PreprocessParams` - 30 edges
9. `BoundingBox` - 29 edges
10. `TiledInferenceEngine` - 25 edges

## Surprising Connections (you probably didn't know these)
- `InferenceWorker` --uses--> `InferenceEngine`  [INFERRED]
  workers/inference_worker.py → core/inference_engine.py
- `ImageInspectWidget` --uses--> `BoundingBox`  [INFERRED]
  ui/image_inspect_widget.py → core/models.py
- `_State` --uses--> `BoundingBox`  [INFERRED]
  ui/image_inspect_widget.py → core/models.py
- `AnalyticsDashboard` --uses--> `DetectionResult`  [INFERRED]
  ui/analytics_dashboard.py → core/models.py
- `ChecklistItem` --uses--> `DetectionResult`  [INFERRED]
  ui/analytics_dashboard.py → core/models.py

## Import Cycles
- None detected.

## Communities (51 total, 7 thin omitted)

### Community 0 - "ThemeManager"
Cohesion: 0.06
Nodes (23): is_likely_pcb(), ndarray, core/pcb_guard.py ----------------- Lightweight PCB subject-matter heuristic gua, Heuristically determine whether a BGR image plausibly contains a PCB.      Retur, Enum, ui/help_dialog.py ----------------- HelpDialog — Interactive onboarding guide an, ui/image_inspect_widget.py -------------------------- ImageInspectWidget — the p, ui/main_window.py ----------------- MainWindow — refactored for 2026 AI Studio A (+15 more)

### Community 1 - "AnalyticsDashboard"
Cohesion: 0.06
Nodes (24): bgr_frame_to_qimage(), enumerate_cameras(), ndarray, QImage, core/utils.py ------------- Shared utility functions for the CIRCA camera pipeli, Probe UVC camera indices 0 through max_index and return all that respond.      U, Convert an OpenCV BGR frame to a Qt-displayable QImage (RGB888 format).      Thi, test_analytics_dashboard_advisory_fallback() (+16 more)

### Community 2 - "ImageInspectWidget"
Cohesion: 0.07
Nodes (25): QColor, QDragEnterEvent, QDropEvent, QPainter, QWheelEvent, ImageInspectWidget, ndarray, QImage (+17 more)

### Community 3 - "PreprocessParams"
Cohesion: 0.09
Nodes (35): core/debug.py ------------- Execution tracing utilities for the CIRCA pipeline, Decorator that logs the entry and exit of a function with millisecond timestamps, trace_execution(), PreprocessParams, Holds the live, thread-safe parameters for the OpenCV preprocessing pipeline., apply_clahe(), apply_denoise(), apply_gamma() (+27 more)

### Community 4 - "InferenceParams"
Cohesion: 0.13
Nodes (30): core/inference_engine.py ------------------------ Wraps Intel OpenVINO 2024.x, # NOTE: cv2.dnn.NMSBoxes applies a strict `score > score_threshold`, BoundingBox, DetectionResult, InferenceParams, Represents the complete inference result for a single frame., Holds the live parameters for the OpenVINO inference engine., Represents a single defect detected by the OpenVINO model. (+22 more)

### Community 5 - "MainWindow"
Cohesion: 0.10
Nodes (5): QMainWindow, MainWindow, Triggered when a camera is plugged/unplugged., Open file picker and load selected image into inspect widget., Toggle live camera viewfinder preview. Snaps photo and stops camera on second cl

### Community 6 - "oversample-dataset-minority-classes.py"
Cohesion: 0.14
Nodes (24): Random, count_annotations(), get_classes_in_label(), main(), print_distribution(), Counter, Path, cap_dominant_classes.py ======================= CIRCA -- Phase 0 Data Preparatio (+16 more)

### Community 7 - "OnboardingController"
Cohesion: 0.15
Nodes (7): OnboardingController, OnboardingOverlay, OnboardingPopover, QFrame, QObject, QRect, QWidget

### Community 8 - "convert_markdown_to_docx"
Cohesion: 0.14
Nodes (18): _Cell, Paragraph, Run, Table, add_page_number(), convert_markdown_to_docx(), parse_inline_formatting(), Any (+10 more)

### Community 9 - "evaluate_auto_optimisation.py"
Cohesion: 0.17
Nodes (17): Request, calculate_iou(), download_wikimedia_pcb_images(), evaluate_auto_optimisation_benefit(), load_ground_truth(), main(), match_boxes(), Any (+9 more)

### Community 10 - "generate_deck.py"
Cohesion: 0.17
Nodes (13): build_deck_files(), generate_slides(), get_base64_image(), main(), render_deck(), build_poster_html(), get_base64_image(), main() (+5 more)

### Community 11 - "evaluate-model-quantization.py"
Cohesion: 0.22
Nodes (15): apply_fallback_rule(), export_model(), _f(), get_size_mb(), parse_args(), Namespace, Path, scripts/evaluate_quantization.py --------------------------------- Phase 5 — Ope (+7 more)

### Community 12 - "StatusFooter"
Cohesion: 0.27
Nodes (3): QWidget, StatusFooter, _StatusIndicator

### Community 13 - "generate_copypaste_data.py"
Cohesion: 0.24
Nodes (13): augment_crop(), compute_iou(), extract_defect_crops(), feather_mask(), generate_composites(), main(), parse_args(), ndarray (+5 more)

### Community 14 - "train_engine.py"
Cohesion: 0.23
Nodes (13): apply_circa_preproc(), ensure_wandb_login(), _list_images(), _preproc_one(), preprocess_dataset(), Path, CIRCA High-Performance Training Engine ====================================== Th, Apply CLAHE on L-channel of LAB, then Gamma=1.2. Returns True on success. (+5 more)

### Community 15 - "InferenceEngine"
Cohesion: 0.23
Nodes (7): InferenceEngine, ndarray, Load and compile the OpenVINO IR model.          Args:             model_xml_, Run a single synchronous inference pass on a preprocessed BGR image., Letterbox-resize and normalise a BGR frame to a model-ready NCHW tensor., Parse the raw YOLOv12 output tensor into a filtered, NMS-deduped         list o, Wraps an OpenVINO CompiledModel for synchronous YOLOv12 INT8 inference.      L

### Community 16 - "build-unified-dataset.py"
Cohesion: 0.30
Nodes (11): ImageHash, build_dataset(), image_dhash(), load_label_file_id(), load_label_file_name(), Path, CIRCA Dataset Builder — unified_pcb_v3 (7-class) ===============================, Parse a Roboflow data.yaml and return {class_name: id} dict. (+3 more)

### Community 17 - "prepare-dataset-splits.py"
Cohesion: 0.33
Nodes (11): get_python_exe(), main(), prepare_all_datasets.py ======================== CIRCA -- Full local dataset pre, Run a subprocess command, raise on failure., Return the venv Python if it exists, otherwise fall back to the current interpre, run(), step_build_preproc(), step_cap_dominant() (+3 more)

### Community 18 - "InferenceWorker"
Cohesion: 0.18
Nodes (7): InferenceWorker, QObject, Compile and cache the OpenVINO IR model.          Called AFTER the Inference T, Run a quick 5-iteration speed benchmark on the specified model         using a, Thread-safe update of inference parameters from the GUI thread.          Conne, Return a thread-safe immutable copy of the current InferenceParams.          T, Event-driven worker that runs OpenVINO inference on frames received from     Ca

### Community 19 - "HardwareScannerThread"
Cohesion: 0.22
Nodes (5): QThread, HardwareScannerThread, QFrame, QWidget, Short-lived background thread for USB camera enumeration.      Architecture note

### Community 20 - "main.py"
Cohesion: 0.28
Nodes (8): _configure_logging(), exception_hook(), _get_model_path(), main(), Resolve the OpenVINO model path., Set up module-level logging for the application., Global exception handler to ensure unhandled Python exceptions     are logged t, Application entry point.

### Community 21 - "HelpDialog"
Cohesion: 0.36
Nodes (3): QDialog, HelpDialog, QWidget

### Community 22 - "CameraWorker"
Cohesion: 0.25
Nodes (6): CameraWorker, QObject, Args:             device_index: The DirectShow UVC device index to open., Thread-safe parameter update from the GUI thread.          Connected to MainWi, Request graceful shutdown of the run loop.          Called by MainWindow.close, Captures frames from a UVC camera, applies the preprocessing pipeline,     and

### Community 23 - "test_ui_wiring.py"
Cohesion: 0.25
Nodes (4): main_window(), Verify that loading a valid PCB image triggers cross-thread inference dispatch., test_main_window_false_positive_logging(), test_main_window_image_loaded()

### Community 25 - ".run_tiled"
Cohesion: 0.29
Nodes (4): ndarray, Compute (x, y, w, h) crop regions for all tiles covering a frame of         size, Apply Non-Maximum Suppression per class across all detections pooled from every, Run inference on a frame of any size using adaptive sliding-window tiling.

### Community 26 - "calibrate-confidence-thresholds.py"
Cohesion: 0.38
Nodes (6): calibrate(), parse_args(), Namespace, scripts/calibrate_thresholds.py --------------------------------- Phase 7 — Conf, Run model.val() at a single confidence threshold.     Returns per-class precisio, run_val_at_conf()

### Community 27 - "MockCore"
Cohesion: 0.29
Nodes (3): MockCore, qapp(), Fixture to initialize the QApplication once per session.

### Community 29 - "evaluate-final-model-metrics.py"
Cohesion: 0.47
Nodes (5): _f(), parse_args(), Namespace, scripts/test_evaluate.py ------------------------- Phase 7 — Final Test-Set Eval, run_test_evaluation()

### Community 30 - "setup-runpod-environment.py"
Cohesion: 0.60
Nodes (5): check_dependencies(), check_system(), print_banner(), print_launch_playbook(), verify_dataset()

### Community 31 - "monitor_and_cleanup.py"
Cohesion: 0.60
Nodes (5): get_active_log_file(), is_training_active(), main(), sftp_download_dir(), terminate_pod()

### Community 32 - "._on_mark_false_positive"
Cohesion: 0.33
Nodes (3): ndarray, Handle user marking a detection bounding box as a false positive.         Exclud, Save raw frame and corrected labels (excluding the false positive)         to da

### Community 33 - "get_dataset_class_distribution"
Cohesion: 0.40
Nodes (4): get_dataset_class_distribution(), Any, utils/check_dataset_class_balance.py ====================================== Reus, Computes split distribution and class annotation frequency for a YOLO format dat

### Community 34 - "get_ssh_connection_details"
Cohesion: 0.67
Nodes (3): get_ssh_connection_details(), main(), Parse RunPod pod dict to extract IP and external SSH Port.

### Community 37 - "_format_arg"
Cohesion: 0.67
Nodes (3): _format_arg(), Any, Format an argument for logging, truncated if necessary, and avoiding bulky data.

## Knowledge Gaps
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MainWindow` connect `MainWindow` to `ThemeManager`, `AnalyticsDashboard`, `ImageInspectWidget`, `PreprocessParams`, `InferenceParams`, `OnboardingController`, `StatusFooter`, `InferenceWorker`, `HardwareScannerThread`, `main.py`, `HelpDialog`, `CameraWorker`, `test_ui_wiring.py`, `NavButton`, `WarningBanner`, `._on_mark_false_positive`, `test_onboarding.py`, `._on_image_loaded`, `._on_next_board_requested`?**
  _High betweenness centrality (0.132) - this node is a cross-community bridge._
- **Why does `ImageInspectWidget` connect `ImageInspectWidget` to `ThemeManager`, `AnalyticsDashboard`, `InferenceParams`, `MainWindow`, `HardwareScannerThread`, `NavButton`?**
  _High betweenness centrality (0.106) - this node is a cross-community bridge._
- **Why does `InferenceParams` connect `InferenceParams` to `ThemeManager`, `PreprocessParams`, `MainWindow`, `evaluate_auto_optimisation.py`, `InferenceEngine`, `InferenceWorker`, `HardwareScannerThread`, `NavButton`, `.run_tiled`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `MainWindow` (e.g. with `InferenceParams` and `PreprocessParams`) actually correct?**
  _`MainWindow` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `ImageInspectWidget` (e.g. with `BoundingBox` and `DetectionResult`) actually correct?**
  _`ImageInspectWidget` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `InferenceParams` (e.g. with `InferenceEngine` and `TiledInferenceEngine`) actually correct?**
  _`InferenceParams` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `ThemeManager` (e.g. with `AnalyticsDashboard` and `ChecklistItem`) actually correct?**
  _`ThemeManager` has 14 INFERRED edges - model-reasoned connections that need verification._