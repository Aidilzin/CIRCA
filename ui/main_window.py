"""
ui/main_window.py
-----------------
MainWindow — root widget, thread lifecycle owner, and signal router.

Architecture role (architecture.md §Threading Architecture):
    MainWindow is the single authority that:
      1. Creates all workers and their QThreads.
      2. Wires ALL signals between workers and UI components.
      3. Starts the threads.
      4. Shuts them down safely on closeEvent.

    No worker connects to any UI widget directly.
    No UI widget holds a reference to any worker.
    All cross-thread data flows through this file.

Thread Model (4 threads total):
    ┌─────────────────────────────────────────────────────┐
    │ Main GUI Thread (Qt event loop)                      │
    │   MainWindow, VideoWidget, StatusFooter,             │
    │   WarningBanner, ControlPanel                        │
    └──────────────────┬──────────────────────────────────┘
                       │ QueuedConnection (automatic)
            ┌──────────┴──────────┬────────────────────────┐
            ▼                     ▼                        ▼
    ┌───────────────┐    ┌─────────────────┐    ┌──────────────────────┐
    │ Camera Thread │    │ Inference Thread │    │ Hardware Scan Thread │
    │ CameraWorker  │    │ InferenceWorker  │    │ HardwareScannerThread│
    └───────────────┘    └─────────────────┘    └──────────────────────┘

Signal catalogue (all connections below in _connect_signals):
    CameraWorker.new_frame(QImage)       → VideoWidget.set_frame
    CameraWorker.camera_error(str)       → MainWindow._on_camera_error
    CameraWorker.frame_ready_for_inference(object)
                                         → InferenceWorker.process_frame

    InferenceWorker.new_detections(object) → VideoWidget.set_detections
    InferenceWorker.new_detections(object) → MainWindow._on_new_detections
    InferenceWorker.model_loaded()        → MainWindow._on_model_loaded
    InferenceWorker.inference_error(str)  → MainWindow._on_inference_error

    ControlPanel.preprocessing_params_changed(object)
                                         → CameraWorker.update_params
    ControlPanel.inference_params_changed(object)
                                         → InferenceWorker.update_params
    ControlPanel.camera_selected(int)    → MainWindow._on_camera_selected

    HardwareScannerThread.cameras_found(list) → MainWindow._on_cameras_found

closeEvent safety contract (architecture.md §Worker Lifecycle Pattern):
    1. camera_worker.stop()          → sets _running = False
       (InferenceWorker has no run() loop; no stop() needed)
    2. camera_thread.quit()          → post Quit event to camera event loop
    3. camera_thread.wait(3000)      → block up to 3s for clean exit
    4. inference_thread.quit()       → post Quit event to inference event loop
    5. inference_thread.wait(3000)   → block up to 3s for clean exit
    Calling wait() without stop() FIRST on CameraWorker causes a deadlock
    because run() never exits its while loop.

Module import boundary:
    ALLOWED:   PyQt6.*, ui/*, workers/*, core/models.py, core/utils.py
    FORBIDDEN: core/preprocessor.py, core/inference_engine.py, cv2, openvino
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.debug import trace_execution
from core.models import DetectionResult, InferenceParams
from core.utils import enumerate_cameras
from ui.control_panel import ControlPanel
from ui.status_footer import StatusFooter
from ui.theme import (
    COLOR_BG_BASE,
    WINDOW_DEFAULT_HEIGHT,
    WINDOW_DEFAULT_WIDTH,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
    build_qss,
)
from ui.video_widget import VideoWidget
from ui.warning_banner import WarningBanner
from workers.camera_worker import CameraWorker
from workers.inference_worker import InferenceWorker

logger = logging.getLogger(__name__)

# Default model path — relative to project root when running with python main.py.
# PyInstaller bundles this under sys._MEIPASS; main.py resolves the full path.
DEFAULT_MODEL_PATH: str = "models/yolov12_int8.xml"

# Confidence threshold used for the FR15 WarningBanner.
# The ControlPanel confidence slider controls InferenceParams.confidence_threshold
# which gates detection. FR15 uses the same value for consistency.
_DEFAULT_CONFIDENCE_THRESHOLD: float = 0.50


class HardwareScannerThread(QThread):
    """
    Lightweight background thread for camera enumeration.
    
    Architecture: Prevents DirectShow COM timeouts from blocking the
    Main GUI Thread during hardware hotplug events or startup.
    """
    cameras_found = pyqtSignal(list)

    def __init__(self, active_device_index: int = -1, qt_names: Optional[List[str]] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._active_device_index = active_device_index
        self._qt_names = qt_names

    @trace_execution
    def run(self) -> None:
        """Execute enumerate_cameras in the background."""
        logger.debug("HardwareScannerThread: starting scan…")
        # Change 1: Pass active_device_index and qt_names to avoid collision and crash.
        cameras = enumerate_cameras(
            active_device_index=self._active_device_index,
            qt_names=self._qt_names
        )
        self.cameras_found.emit(cameras)
        logger.debug("HardwareScannerThread: scan complete (%d found).", len(cameras))


class MainWindow(QMainWindow):
    """
    Root application window — owns all widgets, workers, and threads.

    Responsibilities:
      - Layout assembly: WarningBanner / VideoWidget / StatusFooter on the left;
        ControlPanel on the right.
      - Thread lifecycle: create, start, stop.
      - Signal routing: all connections between workers and UI components.
      - FR15 mediation: receives new_detections, extracts avg_confidence,
        calls WarningBanner.update_confidence().

    Args:
        model_path: Absolute path to the OpenVINO .xml IR model file.
                    Passed in by main.py after resolving sys._MEIPASS.
        parent:     Qt parent (None for a top-level window).
    """

    @trace_execution
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._model_path = model_path

        # Live snapshot of the confidence threshold for FR15 routing.
        # Updated whenever ControlPanel emits inference_params_changed.
        self._confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD

        # Guards _on_camera_selected() during startup.
        # populate_cameras() emits camera_selected for the first detected camera,
        # but at that point the Camera Thread has not started yet. Without this
        # flag, _on_camera_selected() would attempt stop()/quit()/wait() on an
        # un-started thread, then call start() prematurely — before enumeration
        # is complete and before the new CameraWorker has been wired.
        self._camera_thread_started: bool = False

        # Persistent scanner thread instance to avoid repeated allocation.
        self._scanner_thread: Optional[HardwareScannerThread] = None

        logger.debug("[TRIPWIRE] MainWindow: About to call _setup_window()")
        self._setup_window()
        logger.debug("[TRIPWIRE] MainWindow: _setup_window() complete")
        
        logger.debug("[TRIPWIRE] MainWindow: About to call _build_ui()")
        self._build_ui()
        logger.debug("[TRIPWIRE] MainWindow: _build_ui() complete")
        
        logger.debug("[TRIPWIRE] MainWindow: About to call _create_workers()")
        self._create_workers()
        logger.debug("[TRIPWIRE] MainWindow: _create_workers() complete")
        
        logger.debug("[TRIPWIRE] MainWindow: About to call _connect_signals()")
        self._connect_signals()
        logger.debug("[TRIPWIRE] MainWindow: _connect_signals() complete")
        
        logger.debug("[TRIPWIRE] MainWindow: About to call _start_workers()")
        self._start_workers()
        logger.debug("[TRIPWIRE] MainWindow: _start_workers() complete")
        
        # Defer camera enumeration until AFTER app.exec() starts the Qt event
        # loop. Calling _post_start_init() directly here means DirectShow COM
        # calls happen before Windows has an active message pump, causing the
        # main thread to block on a COM message it cannot service — the classic
        # COM / Message Pump deadlock. 200 ms gives the event loop time to
        # start and the window to appear before any DSHOW probing begins.
        logger.debug("[TRIPWIRE] MainWindow: Scheduling _post_start_init delay")
        QTimer.singleShot(200, self._post_start_init)
        logger.debug("[TRIPWIRE] MainWindow: __init__ logic finished")


    # ------------------------------------------------------------------
    # Window configuration
    # ------------------------------------------------------------------

    @trace_execution
    def _setup_window(self) -> None:
        """Configure window geometry, title, minimum size constraints, and USB debounce timer."""
        self.setWindowTitle("CIRCA — PCB Defect Detection")
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resize(WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT)
        self.setStyleSheet(f"background-color: {COLOR_BG_BASE};")

        # USB hotplug debounce timer (single-shot, 500 ms).
        # Windows fires WM_DEVICECHANGE multiple times per device insertion;
        # the timer coalesces the burst into a single camera re-enumeration.
        self._usb_debounce_timer = QTimer(self)
        self._usb_debounce_timer.setSingleShot(True)
        self._usb_debounce_timer.timeout.connect(self._on_usb_timer_timeout)

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """
        Assemble the main window layout.

        Layout structure:
            QMainWindow
            └── central_widget (QHBoxLayout)
                ├── left_column (QVBoxLayout, stretch=1)
                │   ├── warning_banner    (32px, hidden by default)
                │   ├── video_widget      (stretch=1, fills remaining height)
                │   └── status_footer     (48px, always visible)
                └── control_panel         (fixed width 280px/28px)
        """
        # ── Central widget ────────────────────────────────────────────
        central = QWidget()
        central.setObjectName("CentralWidget")
        self.setCentralWidget(central)

        root_row = QHBoxLayout(central)
        root_row.setContentsMargins(0, 0, 0, 0)
        root_row.setSpacing(0)

        # ── Left column ───────────────────────────────────────────────
        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(0)

        # FR15: amber advisory bar (hidden until low confidence detected)
        self.warning_banner = WarningBanner()
        left_col.addWidget(self.warning_banner)

        # Live video viewport with QPainter bounding box overlay
        self.video_widget = VideoWidget()
        self.video_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        left_col.addWidget(self.video_widget, stretch=1)

        # Persistent system health bar — camera/model/FPS status
        self.status_footer = StatusFooter()
        left_col.addWidget(self.status_footer)

        # Pack left column into the root row with stretch
        left_container = QWidget()
        left_container.setLayout(left_col)
        root_row.addWidget(left_container, stretch=1)

        # ── Right column: collapsible control panel ───────────────────
        self.control_panel = ControlPanel()
        root_row.addWidget(self.control_panel)

    # ------------------------------------------------------------------
    # Worker construction (no threads started yet)
    # ------------------------------------------------------------------

    def _create_workers(self) -> None:
        """
        Instantiate workers and their QThreads without starting them.

        Workers are created as plain QObjects here and moved to their threads
        in _start_workers(). Separation of construction from start allows
        _connect_signals() to wire signals before threads begin running,
        which prevents a race where a signal fires before its slot is connected.
        """
        # ── Camera Thread ─────────────────────────────────────────────
        logger.debug("[TRIPWIRE] MainWindow: Instantiating CameraWorker and QThread")
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker(device_index=0)
        logger.debug("[TRIPWIRE] MainWindow: Moving CameraWorker to Camera Thread")
        self.camera_worker.moveToThread(self.camera_thread)
        logger.debug("[TRIPWIRE] MainWindow: CameraWorker moved to Thread")

        # ── Inference Thread ──────────────────────────────────────────
        logger.debug("[TRIPWIRE] MainWindow: Instantiating InferenceWorker and QThread")
        self.inference_thread = QThread()
        self.inference_worker = InferenceWorker()
        logger.debug("[TRIPWIRE] MainWindow: Moving InferenceWorker to Inference Thread")
        self.inference_worker.moveToThread(self.inference_thread)
        logger.debug("[TRIPWIRE] MainWindow: InferenceWorker moved to Thread")


    # ------------------------------------------------------------------
    # Signal routing
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        """
        Wire ALL signals between workers and UI components.

        Rule (architecture.md §Signal/Slot Connection Patterns):
            All connections are made here in MainWindow — never inside
            worker files, never inside widget files.

        Connection type: Qt automatically chooses QueuedConnection for
        cross-thread signal/slot pairs (worker → GUI). DirectConnection
        is used for same-thread pairs (QThread.started → worker.run).
        """
        # ── Thread lifecycle ──────────────────────────────────────────
        self.camera_thread.started.connect(self.camera_worker.run)
        # InferenceWorker has no run() loop — purely slot-driven.
        # inference_thread.started is NOT connected to any worker slot.

        # ── CameraWorker → VideoWidget ────────────────────────────────
        # Live frame display (FR12). Always emitted — blurry or sharp.
        self.camera_worker.new_frame.connect(self.video_widget.set_frame)

        # ── CameraWorker → InferenceWorker ────────────────────────────
        # Sharp preprocessed frames (variance >= threshold) forwarded for inference.
        # Qt QueuedConnection ensures the numpy array crosses threads safely.
        self.camera_worker.frame_ready_for_inference.connect(
            self.inference_worker.process_frame
        )

        # ── CameraWorker → StatusFooter (error reporting) ─────────────
        self.camera_worker.camera_error.connect(self._on_camera_error)

        # ── InferenceWorker → VideoWidget ─────────────────────────────
        # Bounding box overlay (FR13, FR14).
        self.inference_worker.new_detections.connect(self.video_widget.set_detections)

        # ── InferenceWorker → MainWindow (FR15 mediation) ─────────────
        # MainWindow extracts avg_confidence and calls warning_banner.update_confidence.
        # Using a private slot instead of a direct connection allows MainWindow
        # to read the current threshold from the confidence slider at call time.
        self.inference_worker.new_detections.connect(self._on_new_detections)

        # ── InferenceWorker → StatusFooter ────────────────────────────
        self.inference_worker.model_loaded.connect(self._on_model_loaded)
        self.inference_worker.inference_error.connect(self._on_inference_error)

        # ── ControlPanel sliders → Workers ────────────────────────────
        # Preprocessing params (CLAHE, Gamma, Blur) → CameraWorker
        self.control_panel.preprocessing_params_changed.connect(
            self.camera_worker.update_params
        )
        # Inference params (Confidence threshold) → InferenceWorker
        self.control_panel.inference_params_changed.connect(
            self.inference_worker.update_params
        )
        # Confidence threshold → MainWindow (for FR15 threshold tracking)
        self.control_panel.inference_params_changed.connect(
            self._on_inference_params_changed
        )

        # ── ControlPanel camera dropdown → MainWindow ──────────────────
        # Camera selection change requires stopping/restarting CameraWorker.
        # Handled by MainWindow._on_camera_selected() — not wired directly.
        self.control_panel.camera_selected.connect(self._on_camera_selected)

    # ------------------------------------------------------------------
    # Thread start
    # ------------------------------------------------------------------

    def _start_workers(self) -> None:
        """
        Start the Inference Thread ONLY.

        The Camera Thread is intentionally deferred to _post_start_init().

        ROOT CAUSE OF DEFERRAL (DirectShow race condition):
            Windows DirectShow is NOT thread-safe for concurrent
            cv2.VideoCapture.open() calls. If CameraWorker.run() opens the
            camera on the Camera Thread at the same moment enumerate_cameras()
            probes device indices on the Main Thread, both threads hit the
            DirectShow COM layer simultaneously. This raises an
            "unknown C++ exception" in OpenCV and can deadlock the DSHOW
            COM subsystem, preventing the window from showing at all.

        FIX:
            1. Start Inference Thread here (safe — it only waits for slots).
            2. enumerate_cameras() runs on the Main Thread in _post_start_init().
            3. After enumeration completes, _post_start_init() calls
               camera_thread.start() — by then the Main Thread is done with
               DirectShow, so the Camera Thread has exclusive DSHOW access.
        """
        # Both threads are now started inside _post_start_init() which is
        # scheduled 200 ms after app.exec() starts the Qt event loop.
        # Starting threads here (during __init__, before app.exec()) caused an
        # access violation: OpenVINO / DirectShow native DLLs require COM and
        # Windows message-pump services that do not exist until the event loop
        # is running. Set the footer to loading state but defer thread.start().
        self.status_footer.set_model_loading()
        logger.debug(
            "[TRIPWIRE] MainWindow: _start_workers() deferred — "
            "threads will start in _post_start_init() after event loop."
        )

    # ------------------------------------------------------------------
    # Post-start initialisation (happens synchronously after threads launch)
    # ------------------------------------------------------------------

    def _post_start_init(self) -> None:
        """
        Serialized startup: start threads, enumerate cameras — ALL after event loop.

        Asynchronous change: Instead of calling enumerate_cameras() directly,
        this now triggers the HardwareScannerThread to avoid blocking the UI.
        """
        logger.info("MainWindow: starting Inference Thread…")
        self.inference_thread.start()

        # Step 1: Start background camera scan.
        # Determining the device_index and starting the Camera Thread is now
        # handled in _on_cameras_found_startup.
        self._start_async_scan(is_startup=True)

    def _start_async_scan(self, is_startup: bool = False) -> None:
        """
        Launch the HardwareScannerThread if not already running.
        
        Args:
            is_startup: If True, connects results to the startup handler.
        """
        # 1. Prevent overlapping scans
        try:
            if self._scanner_thread and self._scanner_thread.isRunning():
                logger.debug("MainWindow: scanner thread already active; skipping.")
                return
        except RuntimeError:
            # Ghost wrapper: C++ object deleted but Python reference remains.
            self._scanner_thread = None

        # Change 1: Fix Probe Collision & Thread-Safety.
        qt_names: List[str] = []
        try:
            from PyQt6.QtMultimedia import QMediaDevices
            devices = QMediaDevices.videoInputs()
            qt_names = [d.description() for d in devices]
        except (ImportError, Exception):
            logger.debug("MainWindow: QtMultimedia not available for naming.")

        active_idx = -1
        if self._camera_thread_started:
            active_idx = self.camera_worker._device_index

        # 2. CRITICAL: Attach to 'self' and pass 'self' as parent to anchor the thread
        # This prevents the Garbage Collector from deleting the thread object while running.
        self._scanner_thread = HardwareScannerThread(
            active_device_index=active_idx,
            qt_names=qt_names,
            parent=self
        )
        
        if is_startup:
            self._scanner_thread.cameras_found.connect(self._on_cameras_found_startup)
        else:
            self._scanner_thread.cameras_found.connect(self._on_cameras_found_hotplug)
            
        # 3. Clean up C++ resources when finished, but Python ref remains until next scan
        self._scanner_thread.finished.connect(self._scanner_thread.deleteLater)
        
        # 4. Launch safely
        self._scanner_thread.start()

    @pyqtSlot(list)
    def _on_cameras_found_startup(self, cameras: List[Tuple[int, str]]) -> None:
        """Asynchronous callback for the first camera scan at launch."""
        self.control_panel.populate_cameras(cameras)

        combo_data = self.control_panel.camera_combo.currentData()
        device_index: int = (
            combo_data
            if (isinstance(combo_data, int) and combo_data >= 0)
            else -1
        )

        # Status text + footer.
        if cameras and device_index >= 0:
            logger.info("MainWindow: %d camera(s) found.", len(cameras))
            self.status_footer.set_camera_active()
            self.video_widget.set_status_text("Connecting…")
            
            # Rebuild CameraWorker if device_index ≠ default (0).
            if device_index != self.camera_worker._device_index:
                self.camera_worker = CameraWorker(device_index=device_index)
                self.camera_worker.moveToThread(self.camera_thread)
                # Re-wire signals for the new worker instance.
                self.camera_thread.started.connect(self.camera_worker.run)
                self.camera_worker.new_frame.connect(self.video_widget.set_frame)
                self.camera_worker.frame_ready_for_inference.connect(
                    self.inference_worker.process_frame
                )
                self.camera_worker.camera_error.connect(self._on_camera_error)
                self.control_panel.preprocessing_params_changed.connect(
                    self.camera_worker.update_params
                )
                logger.info("MainWindow: CameraWorker rebuilt for device_index=%d.", device_index)

            # Start Camera Thread.
            self._camera_thread_started = True
            logger.info("MainWindow: starting Camera Thread (device_index=%d)…", device_index)
            self.camera_thread.start()
        else:
            # Change 3: Video Feed Placeholder
            logger.warning("MainWindow: no cameras found during startup.")
            self.status_footer.set_camera_idle()
            self.video_widget.set_status_text("Please connect a camera")
            self._camera_thread_started = False

        # Trigger model load.
        if os.path.isfile(self._model_path):
            logger.info("MainWindow: queuing model load — %s", self._model_path)
            from PyQt6.QtCore import QMetaObject, Qt as QtNS
            QMetaObject.invokeMethod(
                self.inference_worker,
                "load_model",
                QtNS.ConnectionType.QueuedConnection,
                self._model_path,           # type: ignore[arg-type]
            )
        else:
            logger.warning(
                "MainWindow: model file not found at '%s'. Inference disabled.",
                self._model_path,
            )
            self.status_footer.set_model_error(f"Model not found: {self._model_path}")


    # ------------------------------------------------------------------
    # Worker signal handlers (Main GUI Thread slots)
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def _on_camera_error(self, message: str) -> None:
        """Red dot in StatusFooter; CameraWorker will retry automatically."""
        logger.warning("MainWindow: camera error — %s", message)
        self.status_footer.set_camera_error(message)
        # UX Spec update: show "camera unavailable" panel or clear feed
        if "disconnected" in message.lower():
            self.video_widget.clear_feed("Camera disconnected")
        else:
            self.video_widget.set_status_text("camera unavailable")

    @pyqtSlot()
    def _on_model_loaded(self) -> None:
        """Green dot in StatusFooter; inference is now active."""
        logger.info("MainWindow: model loaded — inference active.")
        self.status_footer.set_model_ready()

    @pyqtSlot(str)
    def _on_inference_error(self, message: str) -> None:
        """Red model dot; inference continues on the next frame (lock released)."""
        logger.error("MainWindow: inference error — %s", message)
        self.status_footer.set_model_error(message)

    @pyqtSlot(object)
    def _on_new_detections(self, result: DetectionResult) -> None:
        """
        FR15 mediation slot — called on every inference result.

        Extracts average_confidence from the DetectionResult and drives
        WarningBanner and StatusFooter detection count. All routing is
        centralised here so workers remain decoupled from each other.

        Args:
            result: DetectionResult from InferenceWorker.new_detections.
        """
        # ── StatusFooter detection count ──────────────────────────────
        self.status_footer.set_detection_count(len(result.boxes))

        # ── FR15: WarningBanner confidence routing ────────────────────
        # Pass the current threshold from _confidence_threshold (kept in sync
        # by _on_inference_params_changed). This avoids InferenceWorker
        # emitting a separate low_confidence signal — MainWindow mediates.
        self.warning_banner.update_confidence(
            result.average_confidence,
            self._confidence_threshold,
        )

    @pyqtSlot(object)
    def _on_inference_params_changed(self, params: InferenceParams) -> None:
        """
        Keep the local confidence threshold in sync with the control panel slider.

        This is the FR15 threshold source — ControlPanel.inference_params_changed
        is connected here so that when Leo adjusts the Min Confidence slider,
        the WarningBanner threshold updates immediately on the next frame.
        """
        self._confidence_threshold = params.confidence_threshold

    @pyqtSlot(int)
    def _on_camera_selected(self, device_index: int) -> None:
        """
        Handle camera dropdown selection change (runtime only).

        INIT GUARD: populate_cameras() emits camera_selected for the first
        camera found, but at that point the Camera Thread has not started yet
        (_camera_thread_started == False). If this slot were to proceed, it
        would call stop()/quit()/wait() on an un-started thread, add a
        duplicate camera_thread.started → run() connection, then call
        camera_thread.start() prematurely — before Step 4 of _post_start_init()
        has rebuilt the CameraWorker with the correct device index.

        The actual first start is handled by _post_start_init() itself.
        This slot only handles RUNTIME camera switches (user changes dropdown).
        """
        if device_index < 0:
            return

        if not self._camera_thread_started:
            logger.debug(
                "MainWindow: camera_selected(%d) suppressed during startup — "
                "deferred to _post_start_init().",
                device_index,
            )
            return

        logger.info("MainWindow: camera selected — device_index=%d", device_index)

        # Change 2: Immediate UI Wipe
        # Clear the frozen frame immediately before stopping the old thread.
        self.video_widget.set_status_text("Connecting…")

        # Stop the current run loop
        self.camera_worker.stop()
        self.camera_thread.quit()
        self.camera_thread.wait(3000)

        # Create a new CameraWorker with the selected device index.
        # The old worker is orphaned but will be garbage collected.
        self.camera_worker = CameraWorker(device_index=device_index)
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.camera_worker.run)

        # Reconnect signals for the new worker instance
        self.camera_worker.new_frame.connect(self.video_widget.set_frame)
        self.camera_worker.frame_ready_for_inference.connect(
            self.inference_worker.process_frame
        )
        self.camera_worker.camera_error.connect(self._on_camera_error)
        self.control_panel.preprocessing_params_changed.connect(
            self.camera_worker.update_params
        )

        self.camera_thread.start()
        self.status_footer.set_camera_active()
        logger.info("MainWindow: CameraWorker restarted on device_index=%d.", device_index)

    # ------------------------------------------------------------------
    # USB hotplug — Windows WM_DEVICECHANGE interceptor
    # ------------------------------------------------------------------

    def nativeEvent(self, eventType: bytes, message: object) -> tuple[bool, int]:
        """
        Intercept OS-level hardware change broadcasts (Windows only).

        WM_DEVICECHANGE (0x0219) is sent by Windows whenever a device is
        inserted or removed from a USB port. The OS fires it multiple times
        per event, so the signal is debounced with _usb_debounce_timer (500 ms).

        Non-Windows platforms: eventType will never be b'windows_generic_MSG',
        so this method falls through to the base class immediately — zero cost.

        Args:
            eventType: b'windows_generic_MSG' on Windows, other bytes on other OS.
            message:   Pointer to Windows MSG structure (sip.voidptr).

        Returns:
            (handled, result) tuple per Qt convention.
            False = pass to base class (we don't consume the event).
        """
        if eventType == b"windows_generic_MSG":
            import ctypes
            import ctypes.wintypes as wintypes

            class _MSG(ctypes.Structure):
                _fields_ = [
                    ("hWnd",    wintypes.HWND),
                    ("message", wintypes.UINT),
                    ("wParam",  wintypes.WPARAM),
                    ("lParam",  wintypes.LPARAM),
                    ("time",    wintypes.DWORD),
                    ("pt",      wintypes.POINT),
                ]

            WM_DEVICECHANGE = 0x0219
            try:
                msg = _MSG.from_address(int(message))  # type: ignore[arg-type]
                if msg.message == WM_DEVICECHANGE:
                    # Restart (coalesce) the debounce window.
                    self._usb_debounce_timer.start(500)
            except Exception:   # noqa: BLE001 — never crash on native event
                pass

        # Returning False, 0 bypasses super().nativeEvent() which causes an access
        # violation crash in some Windows PyQt6 builds when handling WM_PAINT.
        return False, 0

    @pyqtSlot()
    def _on_usb_timer_timeout(self) -> None:
        """
        Fired 500 ms after the last WM_DEVICECHANGE burst.

        Instead of polling hardware on the GUI thread, launch the background
        HardwareScannerThread.
        """
        logger.info("MainWindow: USB change detected — launching background scan.")
        self._start_async_scan(is_startup=False)

    @pyqtSlot(list)
    def _on_cameras_found_hotplug(self, cameras: List[Tuple[int, str]]) -> None:
        """Asynchronous callback for hotplug camera re-enumeration."""
        # Change 2: Safely compare new list with existing dropdown items.
        current_cameras = []
        for i in range(self.control_panel.camera_combo.count()):
            data = self.control_panel.camera_combo.itemData(i)
            if data is not None and data >= 0:
                current_cameras.append((data, self.control_panel.camera_combo.itemText(i)))

        if cameras == current_cameras:
            logger.debug("MainWindow: hotplug scan result matches current list; no UI update.")
            return

        logger.info("MainWindow: camera list changed; updating dropdown.")
        
        # Determine if the currently active camera is still present.
        active_idx = self.camera_worker._device_index if self._camera_thread_started else -1
        is_active_present = any(idx == active_idx for idx, _ in cameras)

        # Block signals so populate_cameras doesn't trigger a reboot via camera_selected emit.
        self.control_panel.camera_combo.blockSignals(True)
        self.control_panel.populate_cameras(cameras)
        
        # Restore selection if active camera is still present.
        if is_active_present and active_idx >= 0:
            for i in range(self.control_panel.camera_combo.count()):
                if self.control_panel.camera_combo.itemData(i) == active_idx:
                    self.control_panel.camera_combo.setCurrentIndex(i)
                    break
        self.control_panel.camera_combo.blockSignals(False)

        if cameras:
            self.status_footer.set_camera_active()
            # If the active camera was removed, stop the worker.
            if active_idx >= 0 and not is_active_present:
                # Change 3: Video Feed Placeholder
                logger.warning("MainWindow: active camera index %d removed.", active_idx)
                self.video_widget.set_status_text("Please connect a camera")
                self.camera_worker.stop()
                self.camera_thread.quit()
                self.camera_thread.wait(3000)
                self._camera_thread_started = False
            elif self.video_widget._status_text == "Please connect a camera":
                self.video_widget.set_status_text("Connecting\u2026")
            logger.info("MainWindow: background scan — %d camera(s) found.", len(cameras))
        else:
            # Change 3: Video Feed Placeholder
            self.status_footer.set_camera_idle()
            self.video_widget.set_status_text("Please connect a camera")
            if self._camera_thread_started:
                self.camera_worker.stop()
                self.camera_thread.quit()
                self.camera_thread.wait(3000)
                self._camera_thread_started = False
            logger.warning("MainWindow: background scan — no valid cameras found.")

    # ------------------------------------------------------------------
    # FPS update (called from a future FPS counter; placeholder slot)
    # ------------------------------------------------------------------

    @pyqtSlot(float)
    def update_fps(self, fps: float) -> None:
        """Relay FPS reading from CameraWorker to StatusFooter."""
        self.status_footer.set_fps(fps)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Safe shutdown sequence (architecture.md §Worker Lifecycle Pattern).

        MUST call camera_worker.stop() BEFORE camera_thread.quit().
        Without stop(), the CameraWorker.run() while loop never exits,
        camera_thread.wait() blocks forever, and the application hangs.

        InferenceWorker has no run() loop — quit() is enough.

        Sequence:
            1. camera_worker.stop()       → sets _running = False
            2. camera_thread.quit()       → posts Quit to event loop
            3. camera_thread.wait(3000)   → block up to 3s
            4. inference_thread.quit()    → posts Quit to event loop
            5. inference_thread.wait(3000)→ block up to 3s
            6. event.accept()             → window closes

        The 3-second wait() timeout prevents a hung thread from blocking the
        OS process exit indefinitely. After the timeout, Qt proceeds with
        shutdown anyway (threads are daemon-like on process exit).
        """
        logger.info("MainWindow: closeEvent — beginning safe shutdown.")

        # Stop the polling run() loop FIRST
        self.camera_worker.stop()
        logger.debug("MainWindow: camera_worker.stop() called.")

        self.camera_thread.quit()
        if not self.camera_thread.wait(3000):
            logger.warning(
                "MainWindow: camera_thread did not exit within 3s — forcing."
            )
        logger.debug("MainWindow: camera_thread stopped.")

        self.inference_thread.quit()
        if not self.inference_thread.wait(3000):
            logger.warning(
                "MainWindow: inference_thread did not exit within 3s — forcing."
            )
        logger.debug("MainWindow: inference_thread stopped.")

        logger.info("MainWindow: all threads stopped — accepting close event.")
        event.accept()
