---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
inputDocuments:
  - "_bmad-output/planning-artifacts/prd.md"
workflowType: 'architecture'
lastStep: 8
status: 'complete'
completedAt: '2026-04-01'
project_name: 'CIRCA'
user_name: 'Aidil'
date: '2026-04-01'
---

# Architecture Decision Document - CIRCA

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Context Analysis

### Requirements Overview

**Functional Requirements (19 total):**
- **Video Acquisition & Preprocessing (FR1–FR5):** Drive the need for a dedicated capture + preprocessing worker. UVC camera enumeration (FR2) requires a runtime device discovery mechanism compatible with Windows `DirectShow`/`Media Foundation` APIs, surfaced through OpenCV's `cv2.VideoCapture`.
- **Inference & Defect Detection (FR6–FR11):** Drive the need for an isolated inference worker consuming the preprocessed frame queue and emitting structured result objects (bounding box coordinates, class labels, confidence scores) back to the UI layer.
- **Live Diagnostic Interface (FR12–FR15):** Drive the need for the Main GUI Thread to render annotated frames at ≥15 FPS (NFR3) without any blocking calls. The "Manual Inspection Required" warning (FR15) is a UI-reactive state change triggered by a signal from the inference worker.
- **System Configuration (FR16–FR19):** Drive the need for a thread-safe parameter update mechanism allowing real-time mutations to CLAHE clip limit, Gamma value, and blur variance threshold without stopping the processing pipeline.

**Non-Functional Requirements (Architecture-Critical):**
- **NFR3 (15 FPS UI):** Mandates strict GUI/worker separation. Any blocking call on the main thread directly violates this.
- **NFR4 (150ms Queue Depth):** Mandates a bounded frame queue with automatic backpressure and frame-dropping logic.
- **NFR5 (Thread Decoupling):** The primary architectural constraint. Mandates PyQt6 `QThread` + `Signal/Slot` mechanism as the inter-thread communication protocol.
- **NFR1 (<5ms preprocessing):** Mandates that CLAHE + Gamma Correction + Laplacian Variance all execute within a single, tightly profiled processing loop.

**Scale & Complexity:**
- Primary domain: Desktop AI / Real-time Edge ML
- Complexity level: **High**
- Estimated architectural components: 5 (Main Window, Camera Worker, Inference Worker, Settings Controller, Result Overlay Renderer)

### Technical Constraints & Dependencies
- **Python + PyQt6/PySide6** for GUI and threading primitives (`QThread`, `pyqtSignal`).
- **OpenCV (`cv2`)** for camera capture (`VideoCapture`), CLAHE, Gamma Correction, and Laplacian Variance computation.
- **Intel OpenVINO Runtime** for loading `.xml`/`.bin` INT8 IR model and executing `CompiledModel.infer_new_request()` on the CPU/iGPU.
- **PyInstaller** for final single-`.exe` bundling; all OpenVINO dynamic libraries must be correctly included in the bundle spec.
- **Windows 10/11 only:** UVC camera enumeration leverages Windows-native camera indices via `cv2.VideoCapture(index, cv2.CAP_DSHOW)`.

### Cross-Cutting Concerns Identified
- **Thread Safety:** All mutable state shared between threads (preprocessing parameters, frame queue, result queue) must be protected via thread-safe primitives (`queue.Queue`, atomic `QMutex`-guarded values, or signal/slot-only communication).
- **Latency Budgeting:** The 5ms + inference + UI rendering budget must be profiled end-to-end; thermal throttling (NFR6) is a secondary latency adversary.
- **Error Propagation:** Camera disconnects, OpenVINO model load failures, and inference exceptions must surface to the UI thread via dedicated error signals rather than crashing the worker silently.
- **PyInstaller Bundling:** OpenVINO's plugin discovery mechanism (uses a `plugins.xml` manifest) requires explicit `--add-data` flags; this must be documented in the architecture to prevent a common deployment failure.

## Starter Template & Project Foundation

### Primary Technology Domain
**Desktop AI Application** — Python-native GUI with embedded hardware-accelerated ML inference. No web-based starter template applies. The project initialization is a **custom Python module structure** tailored to the QThread-decoupled architecture.

### Technology Decisions

| Concern | Decision | Rationale |
|---|---|---|
| **UI Framework** | **PyQt6** (or PySide6 — identical API) | Native Qt6 widgets, `QThread`, `pyqtSignal` for thread-safe signals; excellent Windows integration |
| **ML Runtime** | **Intel OpenVINO Runtime 2024.x** | INT8 IR model execution on CPU/iGPU; mandated by PRD |
| **CV Pipeline** | **OpenCV `cv2` (4.x)** | CLAHE, Gamma, Laplacian Variance; `VideoCapture` with `CAP_DSHOW` for Windows UVC |
| **Packaging** | **PyInstaller 6.x** | Single `.exe` bundle; requires explicit OpenVINO plugin manifest inclusion |

### Proposed Project Structure

```
circa/
├── main.py                    # QApplication entry point. Instantiates MainWindow.
│
├── ui/
│   ├── main_window.py         # MainWindow(QMainWindow). Owns all widgets & workers.
│   ├── video_widget.py        # QLabel subclass. Receives annotated QImage via signal.
│   └── settings_panel.py      # QSliders for CLAHE, Gamma, Blur. Emits param signals.
│
├── workers/
│   ├── camera_worker.py       # QObject (run in QThread). Captures frames, runs preprocessing, drops blurry frames, emits preprocessed frames via signal.
│   └── inference_worker.py    # QObject (run in QThread). Receives frames, runs OpenVINO inference, emits DetectionResult via signal.
│
├── core/
│   ├── preprocessor.py        # Pure functions: apply_clahe(), apply_gamma(), compute_variance(). Stateless; called by CameraWorker.
│   ├── inference_engine.py    # Wraps OpenVINO CompiledModel. Loads .xml/.bin, runs infer_new_request(). Called by InferenceWorker.
│   └── models.py              # Dataclasses: DetectionResult, BoundingBox, PreprocessParams.
│
├── models/
│   └── yolov12_int8.xml       # OpenVINO IR model (+ .bin). Bundled into .exe via PyInstaller.
│
└── circa.spec                 # PyInstaller spec file with --add-data for OpenVINO plugins.xml
```

### Initialization Command (Development)
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install PyQt6 opencv-python openvino pyinstaller

# Run application
python main.py
```

### Build Command (Distribution `.exe`)
```bash
pyinstaller circa.spec
```

## Core Architectural Decisions

### Decision Priority Analysis

**Critical Decisions (Block Implementation):**
1. **Threading Model** — QObject-in-QThread pattern (not QThread subclassing)
2. **Inter-Thread Communication Contract** — Signals/Slots only; no shared mutable state
3. **Frame Pipeline & Bounded Queue Strategy** — `queue.Queue(maxsize=1)` for zero-latency frame-dropping
4. **Data Pipeline Flow** — UVC Camera → CameraWorker → InferenceWorker → MainWindow

**Important Decisions (Shape Architecture):**
5. **OpenVINO Inference Mode** — Synchronous `infer_new_request()` in a dedicated thread
6. **Settings Mutation Strategy** — Thread-safe `pyqtSignal(PreprocessParams)` push from GUI to worker
7. **Error Propagation Pattern** — Typed error signals per worker, never silent failures

**Deferred Decisions (Post-MVP):**
- PDF report generation pipeline (Phase 2)
- Defect logging/history storage (Phase 2)

### Threading Architecture (NFR5 — Critical)

**Decision:** Use the **QObject-in-QThread** pattern, NOT `QThread` subclassing.

**Rationale:** Subclassing `QThread` and overriding `run()` is a widely documented Qt anti-pattern that breaks the Qt event loop and makes signals from within the thread unreliable. The correct pattern:
1. Create a plain `QObject` worker (`CameraWorker`, `InferenceWorker`)
2. Create a `QThread` instance
3. `moveToThread()` the worker into the QThread
4. Connect the thread's `started` signal to the worker's run slot

```python
# Correct Pattern (enforced across all workers)
self.camera_thread = QThread()
self.camera_worker = CameraWorker()
self.camera_worker.moveToThread(self.camera_thread)
self.camera_thread.started.connect(self.camera_worker.run)
self.camera_thread.start()
```

**Thread Count:** 3 threads total.

| Thread | Owner | Responsibility |
|---|---|---|
| **Main GUI Thread** | Qt event loop | Widget rendering, signal dispatch, slider events |
| **Camera Thread** | `CameraWorker` in `QThread` | UVC capture, CLAHE, Gamma, Laplacian blur-drop, frame emit |
| **Inference Thread** | `InferenceWorker` in `QThread` | OpenVINO INT8 infer, bounding box parsing, result emit |

### Inter-Thread Communication Contract (Signal/Slot Only)

**Decision:** All cross-thread data transfer uses **PyQt6 `pyqtSignal` only**. No direct method calls across threads, no shared global state.

**Signal Catalogue:**

```python
# CameraWorker → MainWindow
preview_frame_ready = pyqtSignal(QImage)       # Live display (always emitted)
frame_dropped = pyqtSignal()                    # Laplacian blur-drop event

# CameraWorker → InferenceWorker (via bounded queue relay)
frame_ready = pyqtSignal(np.ndarray)            # Preprocessed BGR frame

# InferenceWorker → MainWindow
detection_ready = pyqtSignal(list)              # List[DetectionResult]
low_confidence_warning = pyqtSignal(float)      # Avg confidence below threshold
inference_error = pyqtSignal(str)               # Error message string

# MainWindow → CameraWorker (settings updates)
params_updated = pyqtSignal(object)             # PreprocessParams dataclass

# MainWindow → Workers (lifecycle)
stop_signal = pyqtSignal()                      # Graceful shutdown
```

### Frame Queue & Backpressure Strategy (NFR4)

**Decision:** Use **`queue.Queue(maxsize=1)`** between CameraWorker and InferenceWorker with `put_nowait()` and exception-caught drop logic.

**Rationale:** A `maxsize=1` queue is the most effective frame-dropping strategy for real-time pipelines. If inference is slow due to a spike, the camera worker's next frame immediately evicts the stale queued frame rather than building a backlog that causes visible lag.

```python
# In CameraWorker.run() — non-blocking, always-fresh frame delivery
try:
    self.frame_queue.put_nowait(preprocessed_frame)
except queue.Full:
    pass  # Frame dropped — NFR4 satisfied, CPU cycles saved
```

### Data Flow Pipeline (End-to-End)

**Decision:** Linear pipeline with no branching. Each stage isolated, communicating only via its defined signal or queue.

```
[UVC Camera]
    │ cv2.VideoCapture(index, CAP_DSHOW).read()
    ▼
[CameraWorker — Camera Thread]
    │ 1. compute_variance(frame) → if < threshold: DROP, emit frame_dropped
    │ 2. apply_clahe(frame, params.clip_limit)
    │ 3. apply_gamma(frame, params.gamma)
    │ 4. frame_queue.put_nowait(preprocessed_frame)   ← NFR4 bounded queue
    │ 5. emit preview_frame_ready(QImage)              ← live display, no inference lag
    ▼
[InferenceWorker — Inference Thread]
    │ 1. frame = frame_queue.get(block=True)
    │ 2. Preprocess to OpenVINO input tensor (resize, normalize, NCHW layout)
    │ 3. compiled_model.infer_new_request({input_layer: tensor})
    │ 4. Parse output tensor → List[BoundingBox(x,y,w,h,class,confidence)]
    │ 5. if avg_confidence < LOW_CONF_THRESHOLD: emit low_confidence_warning
    │ 6. emit detection_ready(List[DetectionResult])
    ▼
[MainWindow — GUI Thread]
    │ 1. On preview_frame_ready: display raw QImage in VideoWidget
    │ 2. On detection_ready: draw bounding boxes + confidence scores over latest frame
    │ 3. On low_confidence_warning: show 'Manual Inspection Required' banner (FR15)
    └ Composite annotated frame displayed at ≥15 FPS (NFR3)
```

### OpenVINO Inference Mode

**Decision:** **Synchronous `infer_new_request()`** in the dedicated Inference Thread (NOT async OpenVINO API).

**Rationale:** For a single-camera, single-model, single-device pipeline, synchronous inference in a background thread is simpler, more debuggable, and avoids OpenVINO async callback complexity. Thread decoupling (NFR5) already provides the non-blocking UI guarantee. Async OpenVINO API is reserved for multi-model or multi-stream scenarios beyond MVP scope.

**Model Loading:** Done **once at application startup** in `InferenceEngine.__init__()`. The `CompiledModel` instance is cached for the entire application lifetime.

## Implementation Patterns & Consistency Rules

### Naming Patterns

**Python Code Naming (PEP 8 — enforced across all modules):**

| Element | Convention | Example |
|---|---|---|
| Classes | `PascalCase` | `CameraWorker`, `InferenceEngine` |
| Methods & functions | `snake_case` | `apply_clahe()`, `compute_variance()` |
| Signals | `snake_case` | `frame_ready`, `detection_ready` |
| Dataclass fields | `snake_case` | `clip_limit`, `gamma_value` |
| Constants | `UPPER_SNAKE_CASE` | `LOW_CONF_THRESHOLD`, `BLUR_MIN_VARIANCE` |
| Module files | `snake_case.py` | `camera_worker.py`, `inference_engine.py` |

**PyQt6 Widget Naming:** All `QWidget` instances created in `MainWindow.__init__()` must be assigned `self.` attribute names using `snake_case`: `self.video_widget`, `self.clahe_slider`, `self.gamma_slider`.

### Signal/Slot Connection Patterns

**Rule: Always use typed signal declarations. Never use generic `pyqtSignal(object)` for data types that can be strongly typed.**

```python
# ✅ CORRECT — typed signals
frame_ready = pyqtSignal(np.ndarray)
detection_ready = pyqtSignal(list)          # List[DetectionResult]
low_confidence_warning = pyqtSignal(float)

# ❌ ANTI-PATTERN — untyped, breaks static analysis
frame_ready = pyqtSignal(object)
```

**Rule: All signal connections made in `MainWindow.__init__()`, never inside workers.**

```python
# ✅ CORRECT — MainWindow owns all wiring
self.camera_worker.preview_frame_ready.connect(self.video_widget.update_frame)
self.inference_worker.detection_ready.connect(self.video_widget.draw_detections)

# ❌ ANTI-PATTERN — worker connecting to UI directly
self.some_ui_widget.connect(...)   # NEVER inside a worker file
```

### Worker Lifecycle Pattern

**All workers must follow this identical start/stop sequence. No exceptions.**

```python
# START (in MainWindow.__init__)
self.worker_thread = QThread()
self.worker = WorkerClass(dependencies)
self.worker.moveToThread(self.worker_thread)
self.worker_thread.started.connect(self.worker.run)
self.worker_thread.start()

# STOP (in MainWindow.closeEvent)
self.worker.stop()               # Sets internal _running = False flag
self.worker_thread.quit()
self.worker_thread.wait()        # Block until thread exits cleanly
```

**All workers must implement:**

```python
def stop(self):
    self._running = False

def run(self):
    self._running = True
    while self._running:
        ...  # processing loop
```

### Frame Conversion Pattern (BGR → QImage)

**One canonical conversion function. All display code must use only this.**

```python
# core/utils.py — single source of truth
def bgr_frame_to_qimage(frame: np.ndarray) -> QImage:
    """Convert OpenCV BGR frame to Qt-displayable QImage (RGB888)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)

# ✅ CORRECT — always use the util
self.preview_frame_ready.emit(bgr_frame_to_qimage(processed_frame))

# ❌ ANTI-PATTERN — inline conversion creates inconsistency
QImage(frame.data, w, h, ...)   # Don't duplicate this anywhere
```

### Error Handling Pattern

**Rule: Workers must NEVER use bare `except`. All recoverable errors emit a typed error signal. All unrecoverable errors log and set `_running = False`.**

```python
# ✅ CORRECT — typed signal propagation
try:
    result = self.compiled_model.infer_new_request(inputs)
except Exception as e:
    self.inference_error.emit(f"OpenVINO inference failed: {e}")
    continue   # Keep running; UI displays the error

# ❌ ANTI-PATTERN — silent failure
try:
    result = self.compiled_model.infer_new_request(inputs)
except:
    pass   # NEVER. Violates error propagation contract.
```

### Enforcement Guidelines

**All AI Agents implementing CIRCA MUST:**
- Place all OpenCV processing code exclusively in `core/preprocessor.py` (stateless functions only)
- Place all OpenVINO calls exclusively in `core/inference_engine.py`
- Never import PyQt6 into any `core/` module — `core/` must remain UI-framework-agnostic
- Never call `cv2.VideoCapture.read()` outside of `workers/camera_worker.py`
- Never call `compiled_model.infer_new_request()` outside of `workers/inference_worker.py`
- Never call a worker method directly from another thread — always use signals

## Project Structure & Boundaries

### Complete Project Directory Structure

```
circa/
├── main.py                        # Entry point: QApplication, MainWindow bootstrap
├── requirements.txt               # PyQt6, opencv-python, openvino, pyinstaller
├── circa.spec                     # PyInstaller spec: bundles OpenVINO plugins.xml
├── README.md
│
├── ui/
│   ├── __init__.py
│   ├── main_window.py             # MainWindow(QMainWindow): owns all workers + wiring
│   ├── video_widget.py            # VideoWidget(QLabel): renders QImage + bounding boxes
│   └── settings_panel.py          # SettingsPanel(QWidget): sliders → params_updated signal
│
├── workers/
│   ├── __init__.py
│   ├── camera_worker.py           # CameraWorker(QObject): capture → preprocess → emit
│   └── inference_worker.py        # InferenceWorker(QObject): frame queue → infer → emit
│
├── core/
│   ├── __init__.py
│   ├── preprocessor.py            # apply_clahe(), apply_gamma(), compute_variance()
│   ├── inference_engine.py        # InferenceEngine: load IR model, run infer_new_request()
│   ├── utils.py                   # bgr_frame_to_qimage(), enumerate_cameras()
│   └── models.py                  # @dataclass: DetectionResult, BoundingBox, PreprocessParams
│
├── models/
│   ├── yolov12_int8.xml           # OpenVINO IR model graph
│   └── yolov12_int8.bin           # OpenVINO IR model weights
│
└── tests/
    ├── test_preprocessor.py       # Unit tests: CLAHE, Gamma, Variance with synthetic frames
    ├── test_inference_engine.py   # Unit tests: model loading, single-frame inference
    └── test_models.py             # Unit tests: DetectionResult dataclass validation
```

### Requirements to Structure Mapping

| FR | Responsible File(s) |
|---|---|
| FR1 (UVC ingest) | `workers/camera_worker.py` → `cv2.VideoCapture(idx, CAP_DSHOW)` |
| FR2 (Camera selection) | `core/utils.py` → `enumerate_cameras()` · `ui/settings_panel.py` → device `QComboBox` |
| FR3 (CLAHE) | `core/preprocessor.py` → `apply_clahe()` · called by `workers/camera_worker.py` |
| FR4 (Gamma) | `core/preprocessor.py` → `apply_gamma()` · called by `workers/camera_worker.py` |
| FR5 (Blur drop) | `core/preprocessor.py` → `compute_variance()` · gating logic in `workers/camera_worker.py` |
| FR6 (INT8 infer) | `core/inference_engine.py` → `InferenceEngine.run()` · called by `workers/inference_worker.py` |
| FR7–FR10 (Defect classes) | `core/inference_engine.py` → class label map · `core/models.py` → `BoundingBox.class_name` |
| FR11 (Confidence scores) | `core/inference_engine.py` → parse output tensor · `core/models.py` → `BoundingBox.confidence` |
| FR12 (Live feed) | `workers/camera_worker.py` → `preview_frame_ready` signal · `ui/video_widget.py` → `update_frame()` |
| FR13 (Bounding boxes) | `ui/video_widget.py` → `draw_detections()` · receives `detection_ready` from `InferenceWorker` |
| FR14 (Confidence overlay) | `ui/video_widget.py` → `draw_detections()` renders score text above each box |
| FR15 (Low-conf warning) | `workers/inference_worker.py` → `low_confidence_warning` signal · `ui/main_window.py` → shows banner |
| FR16–FR18 (Sliders) | `ui/settings_panel.py` → `QSlider` per param → emits `params_updated(PreprocessParams)` |
| FR19 (Live preview) | `workers/camera_worker.py` → applies params before `preview_frame_ready` emit |

### Architectural Boundaries

**Thread Boundary (inviolable):**
- `ui/` and `workers/` communicate **exclusively via PyQt6 signals** — no shared state, no direct method calls across threads.
- `core/` modules are **import-only** — called synchronously within their owning worker thread.

**Module Import Boundary:**
- `core/` → May import: `numpy`, `cv2`, `openvino`. **Must NOT import**: `PyQt6`, `PySide6`.
- `workers/` → May import: `core/*`, `PyQt6.QtCore`. **Must NOT import**: `PyQt6.QtWidgets`, `ui/*`.
- `ui/` → May import: `core/models.py` (type hints only), `PyQt6.*`. **Must NOT import**: `openvino`, `cv2`.

**Data Ownership Boundary:**
```
Raw BGR frame:        camera_worker.py only
Preprocessed BGR:     camera_worker.py → frame_queue → inference_worker.py
QImage (display):     camera_worker.py → preview_frame_ready → video_widget.py
DetectionResult[]:    inference_worker.py → detection_ready → main_window.py → video_widget.py
PreprocessParams:     settings_panel.py → params_updated → camera_worker.py
```

### Integration Points

**Camera Reconfiguration (FR2):** When the user selects a different UVC device from the `QComboBox`, `MainWindow` calls `camera_worker.stop()`, waits for the camera thread to exit via `camera_thread.wait()`, re-instantiates `CameraWorker` with the new device index, and restarts the thread. This is the **only** permitted pattern for camera switching.

**PyInstaller Deployment Boundary:** The `circa.spec` file must include:
```python
datas=[
    ('models/', 'models/'),
    ('<openvino_install>/runtime/lib/intel64/*.dll', 'openvino/libs/'),
    ('<openvino_install>/runtime/bin/intel64/Release/plugins.xml', 'openvino/'),
]
```

## Architecture Validation Results

### Coherence Validation ✅

**Decision Compatibility:** PyQt6 + OpenCV + OpenVINO + PyInstaller are fully compatible on Windows 10/11. All three libraries coexist in the same Python 3.10+ virtualenv without dependency conflicts. The `CAP_DSHOW` flag is Windows-exclusive, aligning perfectly with the strict Windows-only constraint.

**Pattern Consistency:** The QObject-in-QThread pattern is correctly paired with the `pyqtSignal` communication contract. The `queue.Queue(maxsize=1)` backpressure strategy is framework-agnostic and integrates cleanly with the Qt event loop — no cross-framework conflicts.

**Structure Alignment:** The `core/` → `workers/` → `ui/` import boundary enforces the thread-safety model at the file system level. No circular imports are possible given the defined module import rules.

### Requirements Coverage Validation ✅

**Functional Requirements (19/19 covered):** All 19 FRs are mapped to specific files in the Requirements-to-Structure table. No orphaned requirements exist.

**Non-Functional Requirements:**

| NFR | Architectural Coverage |
|---|---|
| NFR1 (<5ms preprocessing) | `core/preprocessor.py` stateless functions profiled independently |
| NFR2 (<10s inference) | `core/inference_engine.py` synchronous infer in dedicated Inference Thread |
| NFR3 (≥15 FPS UI) | Strict GUI/worker separation; `preview_frame_ready` always emitted regardless of inference state |
| NFR4 (150ms queue depth) | `queue.Queue(maxsize=1)` + `put_nowait()` drop logic in `camera_worker.py` |
| NFR5 (Thread decoupling) | QObject-in-QThread + signal-only communication enforced by module import boundary |
| NFR6 (8hr thermal) | No busy-wait loops; workers block on OS-managed `queue.get()` and `VideoCapture.read()` |
| NFR7 (>90% mAP ±30% light) | CLAHE + Gamma preprocessing applied before inference; user-tunable via sliders (FR16–FR18) |
| NFR8 (≤2GB .exe) | Only 3 core libraries bundled; OpenVINO IR weights ~50–150MB typical |

### Gap Analysis Results

**Critical Gaps: None.** All FRs and NFRs have explicit architectural coverage.

**Important Gap — Model Output Post-Processing:** The architecture specifies `parse output tensor → List[BoundingBox]` but does not document the exact YOLOv12 output tensor format or NMS thresholds. This must be specified during implementation of `inference_engine.py` once the actual model export shape is confirmed.
> **Implementation Note:** YOLOv12 ONNX→OpenVINO export typically produces a `[1, 84, 8400]` tensor (COCO layout). Post-processing must apply confidence thresholding and IoU-based NMS before constructing `BoundingBox` objects.

**Minor Gap — Defect Class Label Map:** The 4 defect classes (`solder_bridge`, `missing_component`, `misaligned_component`, `burnt_area`) must be defined as a constant in `core/inference_engine.py` once confirmed from the YOLOv12 training configuration.

### Architecture Completeness Checklist

- [x] Project context thoroughly analyzed; cross-cutting concerns mapped
- [x] Critical architectural decisions documented (threading, signals, queue strategy)
- [x] Technology stack fully specified with install commands
- [x] All 19 FRs mapped to specific source files
- [x] All 8 NFRs addressed architecturally
- [x] Naming conventions (PEP 8 + Qt widget) established
- [x] Module import boundaries defined and structurally enforced
- [x] Worker lifecycle pattern documented with code examples
- [x] Canonical frame conversion function defined (`bgr_frame_to_qimage`)
- [x] Error handling anti-patterns explicitly documented
- [x] PyInstaller deployment boundary specified (including `plugins.xml`)
- [x] Camera reconfiguration pattern documented (the only safe approach)
- [x] Unit test structure defined under `tests/`

### Architecture Readiness Assessment

**Overall Status: ✅ READY FOR IMPLEMENTATION**
**Confidence Level: High**

**Key Strengths:**
- Thread-safety is structurally enforced by module boundaries, not just documented
- Every FR has an exact file-level home — zero ambiguity for implementation agents
- `queue.Queue(maxsize=1)` elegantly satisfies both NFR3 (FPS) and NFR4 (queue depth) simultaneously
- `core/` being PyQt6-free makes preprocessing fully unit-testable without a Qt event loop

**Implementation Sequence:**
```
1. core/models.py          → Define all dataclasses first (no dependencies)
2. core/preprocessor.py    → Pure functions, testable immediately
3. core/inference_engine.py → OpenVINO model loading + infer wrapper
4. core/utils.py           → bgr_frame_to_qimage(), enumerate_cameras()
5. workers/camera_worker.py → Capture + preprocessing loop
6. workers/inference_worker.py → Queue consumer + infer dispatch
7. ui/video_widget.py      → QImage renderer + bounding box overlay
8. ui/settings_panel.py    → Sliders + params_updated signal
9. ui/main_window.py       → Wire everything together
10. main.py                 → QApplication bootstrap
```
