"""
workers/camera_worker.py
------------------------
CameraWorker — UVC frame capture, OpenCV preprocessing, and frame routing.

Architecture: QObject-in-QThread pattern (NOT QThread subclassing).
  This worker is instantiated as a plain QObject and moved into a QThread
  by MainWindow. Subclassing QThread is explicitly forbidden by the architecture
  because it breaks the Qt event loop and makes cross-thread signal delivery
  unreliable (architecture.md §Threading Architecture).

  Correct wiring (in MainWindow.__init__):
      self.camera_thread = QThread()
      self.camera_worker = CameraWorker(device_index=0)
      self.camera_worker.moveToThread(self.camera_thread)
      self.camera_thread.started.connect(self.camera_worker.run)
      self.camera_thread.start()

Signals emitted:
  new_frame(QImage)                — Live display frame, always emitted.
                                     Blurry: raw frame. Sharp: preprocessed.
  frame_ready_for_inference(object) — Preprocessed BGR np.ndarray for inference.
                                     Emitted ONLY for sharp frames (variance >= threshold).
  camera_error(str)                — Human-readable error message on disconnect/failure.
                                     Triggers red status dot in StatusFooter (UX).

Signal received (slot):
  update_params(object: PreprocessParams) — Thread-safe param update from GUI sliders.

Functional requirements covered:
  FR1  — UVC capture via cv2.VideoCapture with CAP_DSHOW
  FR3  — apply_clahe() on sharp frames
  FR4  — apply_gamma() on sharp frames
  FR5  — compute_variance() motion gate; blurry frames skipped for inference
  FR12 — new_frame emitted every frame for live display (never drops display)
  FR19 — Preprocessing applied to preview (live feedback loop for Sarah, UJ-02)

UX Journey coverage (ux-design-specification.md):
  UJ-01 — Automated initiation; zero-click detection pipeline
  UJ-03 — Mid-shift camera disconnect → non-blocking retry → auto recovery

Module import boundary (architecture §Boundaries):
  ALLOWED:   core/*, PyQt6.QtCore, PyQt6.QtGui, cv2, numpy, time, queue, logging
  FORBIDDEN: PyQt6.QtWidgets, ui/*
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QMutex, QMutexLocker, QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage

from core.debug import trace_execution
from core.models import PreprocessParams
from core.preprocessor import apply_clahe, apply_gamma, compute_variance
from core.utils import bgr_frame_to_qimage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (UX UJ-03 / architecture §Error Recovery)
# ---------------------------------------------------------------------------

# Total wait time between reconnect attempts (UX spec: "10-second retry cycle")
CAMERA_RETRY_INTERVAL_S: float = 10.0

# Granularity of _running checks during retry — keeps stop() responsive.
# Worker stops within this many ms of stop() being called during a retry.
CAMERA_RETRY_POLL_S: float = 0.1


# ---------------------------------------------------------------------------
# CameraWorker
# ---------------------------------------------------------------------------


class CameraWorker(QObject):
    """
    Captures frames from a UVC camera, applies the preprocessing pipeline,
    and routes frames to the GUI display and inference pipeline via signals.

    Lifecycle (managed entirely by MainWindow):
        START → camera_thread.started → run()
        STOP  → stop() → _running = False → run() exits → camera_thread.quit()

    Thread safety:
        PreprocessParams are passed via update_params() slot, protected by
        QMutex. The run() loop snapshots params at each frame to avoid
        holding the lock across slow I/O. All other state is owned exclusively
        by the Camera Thread.
    """

    # ------------------------------------------------------------------
    # Signal declarations
    # ------------------------------------------------------------------

    # Always emitted — raw frame when blurry, preprocessed frame when sharp.
    # VideoWidget connects to this for live display (FR12).
    new_frame: pyqtSignal = pyqtSignal(QImage)

    # Emitted ONLY for frames where Laplacian variance >= blur_threshold.
    # InferenceWorker connects to this (via bounded queue or direct signal).
    # Type: np.ndarray — declared as object because PyQt6 does not support
    # numpy array types as registered Qt metatypes.
    frame_ready_for_inference: pyqtSignal = pyqtSignal(object)

    # Emitted when cv2.VideoCapture fails to open or read() returns False.
    # MainWindow connects this to the StatusFooter red status dot (UJ-03).
    camera_error: pyqtSignal = pyqtSignal(str)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @trace_execution
    def __init__(self, device_index: int = 0, parent: Optional[QObject] = None) -> None:
        """
        Args:
            device_index: The DirectShow UVC device index to open.
                          Populated from enumerate_cameras() in core/utils.py.
            parent:       Qt parent object (usually None; MainWindow owns the thread).
        """
        super().__init__(parent)
        self._device_index: int = device_index
        self._running: bool = False

        # Mutable params — protected by _params_mutex for thread safety.
        self._params: PreprocessParams = PreprocessParams()
        self._params_mutex: QMutex = QMutex()

    # ------------------------------------------------------------------
    # Public slots (called from the Main GUI Thread via signals)
    # ------------------------------------------------------------------

    @pyqtSlot(object)
    def update_params(self, params: PreprocessParams) -> None:
        """
        Thread-safe parameter update from the GUI thread.

        Connected to MainWindow's params_updated signal. PyQt6 signal/slot
        mechanism handles thread boundaries — this slot executes in the
        Camera Thread's event queue (when moveToThread() is used correctly).

        Args:
            params: New PreprocessParams from GUI sliders.
        """
        with QMutexLocker(self._params_mutex):
            self._params = params

    @pyqtSlot()
    @trace_execution
    def stop(self) -> None:
        """
        Request graceful shutdown of the run loop.

        Called by MainWindow.closeEvent() before camera_thread.quit().
        The run loop exits at its next iteration (within CAMERA_RETRY_POLL_S
        if currently in a retry sleep, within one frame otherwise).
        """
        self._running = False
        logger.debug("CameraWorker.stop() called — requesting shutdown.")

    # ------------------------------------------------------------------
    # Main run loop (slot — connected to QThread.started)
    # ------------------------------------------------------------------

    @pyqtSlot()
    @trace_execution
    def run(self) -> None:
        """
        The Camera Thread's primary execution loop.

        Pipeline per frame:
          1. Open camera if not already open (with CAP_DSHOW, FR1)
          2. cap.read() → if failed: emit camera_error, break loop (sudden disconnect)
          3. compute_variance() — FR5 motion gate
          4. Blurry frame path:
               emit new_frame(raw_qimage)     — always show live feed (FR12)
               continue                        — skip inference for blurry frame
          5. Sharp frame path:
               apply_clahe(frame, params)     — FR3
               apply_gamma(processed, params) — FR4
               emit new_frame(processed_qimage)      — FR12, FR19
               emit frame_ready_for_inference(array)  — to InferenceWorker

        Error recovery (UJ-03):
          On sudden disconnect (cap.read() failure), the worker emits camera_error
          and stops running. Re-connection is handled by MainWindow via re-launching
          the worker after re-enumeration.
        """
        self._running = True
        cap: Optional[cv2.VideoCapture] = None

        logger.info("CameraWorker started — device_index=%d", self._device_index)

        while self._running:
            # ----------------------------------------------------------
            # Stage 1: Ensure the camera is open
            # ----------------------------------------------------------
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(self._device_index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    msg = (
                        f"Cannot open camera at index {self._device_index}. "
                        f"Check USB connection."
                    )
                    logger.warning(msg)
                    self.camera_error.emit(msg)
                    # Exit gracefully on open failure to allow re-enumeration
                    self._running = False
                    break
                logger.info(
                    "Camera opened at index %d (DirectShow).", self._device_index
                )

            # ----------------------------------------------------------
            # Stage 2: Read a frame
            # ----------------------------------------------------------
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                msg = "Camera disconnected"
                logger.warning(
                    "Camera %d read() failed — device lost. (UJ-03)", self._device_index
                )
                self.camera_error.emit(msg)
                # Graceful Worker Disconnect.
                # Immediately break and exit to prevent segfault or blocking on lost hardware.
                self._running = False
                break

            # ── Dead Noise Check (Support Lenovo physical switches) ──
            # CALIBRATION:
            # Ghost noise (switch off): std ~4.4, mean ~0.14
            # Real dark sensor: std ~2.4, mean ~0.03
            frame_std = np.std(frame)
            frame_mean = np.mean(frame)
            
            # During live streaming, we are more lenient than during enumeration
            # because the user has already explicitly selected this camera.
            # We ONLY break if the sensor goes completely dead (std < 0.1)
            # OR if we see digital garbage (std > 3.0 + low mean).
            if frame_std < 0.1:
                msg = "Camera disconnected (Dead Sensor)"
                logger.warning("Camera %d: sensor went dead.", self._device_index)
                self.camera_error.emit(msg)
                self._running = False
                break
            elif frame_std > 3.0 and frame_mean < 0.5:
                msg = "Camera disconnected (Physical Switch Off)"
                logger.warning("Camera %d: digital garbage detected (std=%.2f).", self._device_index, frame_std)
                self.camera_error.emit(msg)
                self._running = False
                break

            # ----------------------------------------------------------
            # Stage 3: Thread-safe params snapshot
            # ----------------------------------------------------------
            params = self._snapshot_params()

            # ----------------------------------------------------------
            # Stage 4: FR5 — Motion / blur gate (Laplacian variance)
            # ----------------------------------------------------------
            start_preprocess = time.perf_counter()
            variance = compute_variance(frame)

            if variance < params.blur_threshold:
                # Board is moving or out of focus.
                if self._running:
                    try:
                        self.new_frame.emit(bgr_frame_to_qimage(frame))
                    except ValueError:
                        pass  # Drop corrupted frames
                continue

            # ----------------------------------------------------------
            # Stage 5: Sharp frame — apply preprocessing pipeline (FR3, FR4)
            # ----------------------------------------------------------
            processed = apply_clahe(frame, params)  # FR3: CLAHE contrast enhance
            processed = apply_gamma(processed, params)  # FR4: Gamma shadow lift
            
            preprocess_duration_ms = (time.perf_counter() - start_preprocess) * 1000.0
            logger.debug("CameraWorker: Preprocessing took %.2fms (variance=%.1f)", 
                         preprocess_duration_ms, variance)

            # Emit to GUI (FR12 live display, FR19 real-time preprocessing preview).
            if self._running:
                try:
                    self.new_frame.emit(bgr_frame_to_qimage(processed))
                except ValueError:
                    pass  # Drop corrupted frames

            # Emit to InferenceWorker.
            if self._running:
                self.frame_ready_for_inference.emit(processed.copy())

        # ------------------------------------------------------------------
        # Cleanup on loop exit
        # ------------------------------------------------------------------
        if cap is not None:
            try:
                cap.release()
            except Exception as exc:  # pragma: no cover — DirectShow driver bug on disconnect
                logger.debug("CameraWorker: cap.release() raised (driver bug, safe to ignore): %s", exc)
            logger.info("CameraWorker: VideoCapture released on shutdown.")

        logger.info("CameraWorker.run() exited cleanly.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _snapshot_params(self) -> PreprocessParams:
        """
        Return a thread-safe immutable copy of the current PreprocessParams.

        Copying ensures the run loop uses a consistent param set for the
        entire frame, even if update_params() fires mid-processing.
        The lock is held only for the duration of the copy (microseconds).
        """
        with QMutexLocker(self._params_mutex):
            src = self._params
            return PreprocessParams(
                clahe_clip_limit=src.clahe_clip_limit,
                gamma=src.gamma,
                blur_threshold=src.blur_threshold,
            )

    def _wait_for_retry(self) -> None:
        """
        Non-blocking wait for CAMERA_RETRY_INTERVAL_S before the next
        reconnect attempt. Polls _running every CAMERA_RETRY_POLL_S so
        stop() takes effect within ~100ms even during a retry pause.

        UX UJ-03: "10-second retry cycle" — CAMERA_RETRY_INTERVAL_S = 10.0

        Note: This helper is retained for potential future retry-loop
        implementations but is not called in the current design, which
        delegates reconnection to MainWindow via re-enumeration.
        """
        elapsed: float = 0.0
        while elapsed < CAMERA_RETRY_INTERVAL_S and self._running:
            time.sleep(CAMERA_RETRY_POLL_S)
            elapsed += CAMERA_RETRY_POLL_S