"""
workers/inference_worker.py
---------------------------
InferenceWorker — event-driven OpenVINO inference processor.

Architecture: QObject-in-QThread pattern (NOT QThread subclassing).
  Unlike CameraWorker which has a polling run() loop, InferenceWorker is
  purely slot-driven: it processes frames only when the process_frame()
  slot fires. The Qt event loop in the Inference Thread delivers slots
  between inference calls, so no blocking run() loop is needed.

  Correct wiring (in MainWindow.__init__):
      self.inference_thread = QThread()
      self.inference_worker = InferenceWorker()
      self.inference_worker.moveToThread(self.inference_thread)
      self.inference_thread.start()
      # Load model after thread starts:
      self.inference_worker.load_model("models/yolov12_int8.xml")

Backpressure (NFR4 — Critical):
  InferenceEngine.run() is synchronous and may take 40–200ms per frame.
  If CameraWorker emits frames faster than inference processes them, Qt's
  queued connection mechanism would accumulate a backlog of stale frames.

  Solution: threading.Lock with acquire(blocking=False) as an _is_processing
  gate. When process_frame() is delivered by the event loop:
    - Lock NOT held → acquire it, run inference, release it when done.
    - Lock HELD (previous call still inside InferenceEngine.run()) →
      acquire fails → frame silently dropped.

  This is NOT redundant: even though Qt's event loop is single-threaded
  per QObject, the lock correctly handles cases where:
    1. process_frame() is called directly from another thread (e.g. tests).
    2. The slot is connected with Qt.ConnectionType.DirectConnection.
    3. Future refactors that change the connection type.

  The net effect: the Inference Thread always processes the freshest available
  frame — never a stale backlogged one.

Signals emitted:
  new_detections(object: DetectionResult) — Inference result for each processed frame.
  inference_error(str)                    — Any exception from model load or inference.
  model_loaded()                          — Emitted when InferenceEngine is ready.

Slots received:
  load_model(model_path: str)   — Compile OpenVINO IR model after thread start.
  process_frame(frame: object)  — Run inference on a preprocessed BGR np.ndarray.
  update_params(params: object) — Thread-safe update of InferenceParams from GUI.

Functional requirements covered:
  FR6  — Synchronous INT8 OpenVINO inference via InferenceEngine.run()
  FR7–FR10 — Defect class label mapping (delegated to InferenceEngine)
  FR11 — Confidence scores in BoundingBox.confidence (delegated to InferenceEngine)
  FR15 — low_confidence trigger: average_confidence < threshold (delegated to MainWindow)

Module import boundary (architecture §Boundaries):
  ALLOWED:   core/*, PyQt6.QtCore, threading, logging
  FORBIDDEN: PyQt6.QtWidgets, PyQt6.QtGui, ui/*, cv2 (cv2 belongs in core/ only)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from PyQt6.QtCore import QMutex, QMutexLocker, QObject, pyqtSignal, pyqtSlot

from core.debug import trace_execution
from core.inference_engine import InferenceEngine
from core.tiled_inference import TiledInferenceEngine
from core.models import DetectionResult, InferenceParams

logger = logging.getLogger(__name__)


class InferenceWorker(QObject):
    """
    Event-driven worker that runs OpenVINO inference on frames received from
    CameraWorker and emits structured DetectionResult objects to MainWindow.

    Lifecycle:
        START → inference_thread.start() → (model loads asynchronously) →
                load_model() slot fires → engine ready → model_loaded emitted
        WORK  → process_frame() slot fires per frame → inference → new_detections
        STOP  → MainWindow.closeEvent → inference_thread.quit() → thread exits

    Unlike CameraWorker, InferenceWorker has NO run() loop. It is purely
    reactive: work is done only in response to slot calls delivered by the
    Qt event loop in the Inference Thread.
    """

    # ------------------------------------------------------------------
    # Signal declarations
    # ------------------------------------------------------------------

    # Emitted after each successful inference pass.
    # Carries a DetectionResult (zero or more BoundingBox objects).
    # Declared as object because DetectionResult is a plain Python dataclass
    # (not a registered Qt metatype). MainWindow connects this to ImageInspectWidget.
    new_detections: pyqtSignal = pyqtSignal(object)

    # Emitted on any exception during model loading or inference.
    # MainWindow connects this to the StatusFooter red status dot.
    inference_error: pyqtSignal = pyqtSignal(str)

    # Emitted once after load_model() successfully compiles the IR model.
    # MainWindow uses this to flip the "● Model Ready" status dot.
    model_loaded: pyqtSignal = pyqtSignal()

    # Emitted when a dynamic model suitability benchmark completes.
    # Carries: (model_path: str, avg_latency_ms: float)
    benchmark_completed: pyqtSignal = pyqtSignal(str, float)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @trace_execution
    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)

        # InferenceEngine and TiledInferenceEngine — None until load_model() succeeds.
        self._engine: Optional[InferenceEngine] = None
        self._tiled_engine: Optional[TiledInferenceEngine] = None

        # Live inference parameters from GUI sliders.
        # Protected by QMutex for thread-safe updates from GUI thread.
        self._params: InferenceParams = InferenceParams()
        self._params_mutex: QMutex = QMutex()

        # Backpressure gate (NFR4).
        # threading.Lock.acquire(blocking=False) returns:
        #   True  — lock acquired → we own this frame → run inference
        #   False — lock already held → inference running → drop this frame
        # threading.Lock is chosen over a simple bool because it is atomically
        # safe for concurrent access from multiple threads without a GIL
        # assumption (correct under all Python execution models including PyPy).
        self._inference_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public slots
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    @trace_execution
    def load_model(self, model_path: str) -> None:
        """
        Compile and cache the OpenVINO IR model.

        Called AFTER the Inference Thread has started (connected to
        inference_thread.started or called explicitly by MainWindow after
        inference_thread.start()). Loading here instead of __init__ ensures
        the potentially slow OpenVINO compilation does not block the GUI thread
        during application startup.

        On success: emits model_loaded() → MainWindow updates status dot.
        On failure: emits inference_error(str) → MainWindow shows red dot.

        Args:
            model_path: Path to the .xml OpenVINO IR model file.
                        The companion .bin must be in the same directory.
        """
        try:
            logger.info("InferenceWorker: loading model from '%s'…", model_path)
            self._engine = InferenceEngine(model_path)
            self._tiled_engine = TiledInferenceEngine(self._engine)
            logger.info("InferenceWorker: model loaded successfully.")
            self.model_loaded.emit()
        except Exception as exc:
            msg = f"Model load failed — {type(exc).__name__}: {exc}"
            logger.error(msg, exc_info=True)
            self.inference_error.emit(msg)

    @pyqtSlot(str)
    @trace_execution
    def run_benchmark(self, model_path: str) -> None:
        """
        Run a quick 5-iteration speed benchmark on the specified model
        using a dummy input frame, and emit the average latency.
        """
        try:
            import numpy as np
            logger.info("InferenceWorker: starting benchmark on '%s'…", model_path)
            # Create a separate temporary engine to avoid mutating currently running inference
            engine = InferenceEngine(model_path)
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            params = InferenceParams(confidence_threshold=0.50)

            # Warmup pass
            _ = engine.run(dummy_frame, params)

            # 5 measurement passes
            start = time.perf_counter()
            for _ in range(5):
                _ = engine.run(dummy_frame, params)
            avg_latency = ((time.perf_counter() - start) / 5.0) * 1000.0

            logger.info("InferenceWorker: benchmark done. Avg latency = %.2f ms", avg_latency)
            self.benchmark_completed.emit(model_path, avg_latency)
        except Exception as exc:
            msg = f"Benchmark failed — {type(exc).__name__}: {exc}"
            logger.error(msg, exc_info=True)
            self.inference_error.emit(msg)

    @pyqtSlot(object)
    def update_params(self, params: InferenceParams) -> None:
        """
        Thread-safe update of inference parameters from the GUI thread.

        Connected to MainWindow's params_updated signal (or a dedicated
        inference_params_updated signal). The QMutex ensures the params
        snapshot in _snapshot_params() always reads a consistent object.

        Args:
            params: Updated InferenceParams from the confidence threshold slider.
        """
        with QMutexLocker(self._params_mutex):
            self._params = params

    @pyqtSlot(object)
    def process_frame(self, frame: object) -> None:
        """
        Run a single synchronous inference pass on a preprocessed BGR frame.

        This is the core processing slot — connected to CameraWorker's
        frame_ready_for_inference signal via a Qt queued connection.

        Backpressure gate:
          If _inference_lock is already held (previous frame is still being
          processed), this frame is silently dropped. This guarantees the
          Inference Thread always works on the most recent available frame
          rather than draining a stale queue (NFR4).

        Frame drop conditions (silent, no error emitted):
          1. _inference_lock is held (inference busy) — backpressure drop.
          2. _engine is None (model not loaded yet) — startup drop.

        Error conditions (inference_error emitted):
          3. _engine.run() raises any exception (corrupt frame, OpenVINO failure).

        Args:
            frame: Preprocessed BGR np.ndarray from CameraWorker.
                   Typing as object matches pyqtSignal(object) declaration.
        """
        # BACKPRESSURE: Non-blocking lock acquire. Returns immediately.
        if not self._inference_lock.acquire(blocking=False):
            logger.debug("InferenceWorker: frame dropped — inference busy.")
            return

        try:
            if self._engine is None or self._tiled_engine is None:
                # Model not loaded yet. Drop silently — this is normal during
                # the startup window between thread.start() and load_model().
                logger.debug("InferenceWorker: frame dropped — model not loaded.")
                return

            params = self._snapshot_params()

            start_inference = time.perf_counter()
            result: DetectionResult = self._tiled_engine.run_tiled(frame, params)
            inference_duration_ms = (time.perf_counter() - start_inference) * 1000.0
            result.inference_time_ms = inference_duration_ms
            # Count tiles used (for HUD display)
            h, w = frame.shape[:2]
            from core.tiled_inference import TILE_SIZE, TILE_STRIDE
            tiles_x = max(1, (w - TILE_SIZE + TILE_STRIDE - 1) // TILE_STRIDE + 1) if w > TILE_SIZE else 1
            tiles_y = max(1, (h - TILE_SIZE + TILE_STRIDE - 1) // TILE_STRIDE + 1) if h > TILE_SIZE else 1
            result.tile_count = tiles_x * tiles_y

            logger.debug(
                "InferenceWorker: %d detection(s), avg_conf=%.2f, took %.2fms",
                len(result.boxes),
                result.average_confidence,
                inference_duration_ms,
            )
            try:
                self.new_detections.emit(result)
            except RuntimeError:
                pass

        except Exception as exc:
            msg = f"Inference failed — {type(exc).__name__}: {exc}"
            logger.error(msg, exc_info=True)
            try:
                self.inference_error.emit(msg)
            except RuntimeError:
                pass
        finally:
            # Always release — even on exception — so the worker recovers
            # automatically on the next frame rather than deadlocking.
            self._inference_lock.release()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _snapshot_params(self) -> InferenceParams:
        """
        Return a thread-safe immutable copy of the current InferenceParams.

        The lock is held only for the duration of the field read (microseconds),
        not across the entire inference call.
        """
        with QMutexLocker(self._params_mutex):
            return InferenceParams(
                confidence_threshold=self._params.confidence_threshold,
            )