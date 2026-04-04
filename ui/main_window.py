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
        # Add COM Safety for DirectShow on Windows.
        import ctypes
        try:
            ctypes.windll.ole32.CoInitialize(None)
        except Exception:
            pass

        logger.debug("HardwareScannerThread: starting scan…")
        cameras = enumerate_cameras(
            active_device_index=self._active_device_index,
            qt_names=self._qt_names
        )
        self.cameras_found.emit(cameras)
        logger.debug("HardwareScannerThread: scan complete (%d found).", len(cameras))

        try:
            ctypes.windll.ole32.CoUninitialize()
        except Exception:
            pass


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
        self._confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD

        # Guards _on_camera_selected() during startup.
        self._camera_thread_started: bool = False

        # Persistent scanner thread instance to avoid repeated allocation.
        self._scanner_thread: Optional[HardwareScannerThread] = None

        self._setup_window()
        self._build_ui()
        self._create_workers()
        self._connect_signals()
        self._start_workers()
        
        # Defer camera enumeration until AFTER app.exec() starts the Qt event loop.
        QTimer.singleShot(200, self._post_start_init)

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

        self._usb_debounce_timer = QTimer(self)
        self._usb_debounce_timer.setSingleShot(True)
        self._usb_debounce_timer.timeout.connect(self._on_usb_timer_timeout)

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Assemble the main window layout."""
        central = QWidget()
        central.setObjectName("CentralWidget")
        self.setCentralWidget(central)

        root_row = QHBoxLayout(central)
        root_row.setContentsMargins(0, 0, 0, 0)
        root_row.setSpacing(0)

        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(0)

        self.warning_banner = WarningBanner()
        left_col.addWidget(self.warning_banner)

        self.video_widget = VideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_col.addWidget(self.video_widget, stretch=1)

        self.status_footer = StatusFooter()
        left_col.addWidget(self.status_footer)

        left_container = QWidget()
        left_container.setLayout(left_col)
        root_row.addWidget(left_container, stretch=1)

        self.control_panel = ControlPanel()
        root_row.addWidget(self.control_panel)

    # ------------------------------------------------------------------
    # Worker construction
    # ------------------------------------------------------------------

    def _create_workers(self) -> None:
        """Instantiate workers and their QThreads without starting them."""
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker(device_index=0)
        self.camera_worker.moveToThread(self.camera_thread)

        self.inference_thread = QThread()
        self.inference_worker = InferenceWorker()
        self.inference_worker.moveToThread(self.inference_thread)

    # ------------------------------------------------------------------
    # Signal routing
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        """Wire ALL signals between workers and UI components."""
        # Initial connection for camera thread start. 
        # Future connections handled by _on_camera_selected.
        self.camera_thread.started.connect(self.camera_worker.run)

        self.camera_worker.new_frame.connect(self.video_widget.set_frame)
        self.camera_worker.frame_ready_for_inference.connect(self.inference_worker.process_frame)
        self.camera_worker.camera_error.connect(self._on_camera_error)

        self.inference_worker.new_detections.connect(self.video_widget.set_detections)
        self.inference_worker.new_detections.connect(self._on_new_detections)
        self.inference_worker.model_loaded.connect(self._on_model_loaded)
        self.inference_worker.inference_error.connect(self._on_inference_error)

        self.control_panel.preprocessing_params_changed.connect(self.camera_worker.update_params)
        self.control_panel.inference_params_changed.connect(self.inference_worker.update_params)
        self.control_panel.inference_params_changed.connect(self._on_inference_params_changed)
        self.control_panel.camera_selected.connect(self._on_camera_selected)

    # ------------------------------------------------------------------
    # Thread start
    # ------------------------------------------------------------------

    def _start_workers(self) -> None:
        """Start the Inference Thread ONLY. Camera Thread deferred."""
        self.status_footer.set_model_loading()
        logger.debug("MainWindow: _start_workers() deferred Camera Thread.")

    # ------------------------------------------------------------------
    # Post-start initialisation
    # ------------------------------------------------------------------

    @trace_execution
    def _post_start_init(self) -> None:
        """Start threads, enumerate cameras — ALL after event loop."""
        logger.info("MainWindow: starting Inference Thread…")
        self.inference_thread.start()
        self._start_async_scan(is_startup=True)

    def _disconnect_camera_worker(self) -> None:
        """
        Surgically disconnect ALL signals from the current camera worker.
        Prevents slot accumulation and stale reference crashes when workers
        are swapped during hotswitch or error recovery.
        """
        try:
            self.camera_worker.new_frame.disconnect()
        except (TypeError, RuntimeError):
            pass
        try:
            self.camera_worker.frame_ready_for_inference.disconnect()
        except (TypeError, RuntimeError):
            pass
        try:
            self.camera_worker.camera_error.disconnect()
        except (TypeError, RuntimeError):
            pass
        try:
            self.control_panel.preprocessing_params_changed.disconnect(self.camera_worker.update_params)
        except (TypeError, RuntimeError):
            pass
        try:
            self.camera_thread.started.disconnect()
        except (TypeError, RuntimeError):
            pass

    def _start_async_scan(self, is_startup: bool = False) -> None:
        """Launch the HardwareScannerThread if not already running."""
        try:
            if self._scanner_thread and self._scanner_thread.isRunning():
                return
        except RuntimeError:
            self._scanner_thread = None

        qt_names: List[str] = []
        try:
            from PyQt6.QtMultimedia import QMediaDevices
            devices = QMediaDevices.videoInputs()
            qt_names = [d.description() for d in devices]
        except Exception:
            pass

        active_idx = self.camera_worker._device_index if self._camera_thread_started else -1

        self._scanner_thread = HardwareScannerThread(
            active_device_index=active_idx,
            qt_names=qt_names,
            parent=self
        )
        
        if is_startup:
            self._scanner_thread.cameras_found.connect(self._on_cameras_found_startup)
        else:
            self._scanner_thread.cameras_found.connect(self._on_cameras_found_hotplug)
            
        self._scanner_thread.finished.connect(self._scanner_thread.deleteLater)
        self._scanner_thread.start()

    @pyqtSlot(list)
    def _on_cameras_found_startup(self, cameras: List[Tuple[int, str]]) -> None:
        """Asynchronous callback for the first camera scan at launch."""
        self.control_panel.populate_cameras(cameras)

        combo_data = self.control_panel.camera_combo.currentData()
        device_index: int = combo_data if (isinstance(combo_data, int) and combo_data >= 0) else -1

        if cameras and device_index >= 0:
            logger.info("MainWindow: %d camera(s) found.", len(cameras))
            self.status_footer.set_camera_active()
            self.video_widget.clear_feed("Connecting…")
            
            if device_index != self.camera_worker._device_index:
                self._disconnect_camera_worker()
                
                self.camera_worker = CameraWorker(device_index=device_index)
                self.camera_worker.moveToThread(self.camera_thread)
                self.camera_thread.started.connect(self.camera_worker.run)
                self.camera_worker.new_frame.connect(self.video_widget.set_frame)
                self.camera_worker.frame_ready_for_inference.connect(self.inference_worker.process_frame)
                self.camera_worker.camera_error.connect(self._on_camera_error)
                self.control_panel.preprocessing_params_changed.connect(self.camera_worker.update_params)

            self._camera_thread_started = True
            self.camera_thread.start()
        else:
            logger.warning("MainWindow: no cameras found during startup.")
            self.status_footer.set_camera_idle()
            self.video_widget.clear_feed("Please connect a camera")
            self._camera_thread_started = False

        if os.path.isfile(self._model_path):
            from PyQt6.QtCore import QMetaObject
            QMetaObject.invokeMethod(self.inference_worker, "load_model", Qt.ConnectionType.QueuedConnection, self._model_path)
        else:
            self.status_footer.set_model_error(f"Model not found: {self._model_path}")

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def _on_camera_error(self, message: str) -> None:
        """Handle camera disconnect or read failure."""
        logger.warning("MainWindow: camera error — %s", message)
        self.status_footer.set_camera_error(message)
        self._camera_thread_started = False
        if "disconnected" in message.lower():
            self.video_widget.clear_feed("Camera disconnected")
        else:
            self.video_widget.clear_feed("camera unavailable")

    @pyqtSlot()
    def _on_model_loaded(self) -> None:
        self.status_footer.set_model_ready()

    @pyqtSlot(str)
    def _on_inference_error(self, message: str) -> None:
        self.status_footer.set_model_error(message)

    @pyqtSlot(object)
    def _on_new_detections(self, result: DetectionResult) -> None:
        self.status_footer.set_detection_count(len(result.boxes))
        self.warning_banner.update_confidence(result.average_confidence, self._confidence_threshold)

    @pyqtSlot(object)
    def _on_inference_params_changed(self, params: InferenceParams) -> None:
        self._confidence_threshold = params.confidence_threshold

    @pyqtSlot(int)
    @trace_execution
    def _on_camera_selected(self, device_index: int) -> None:
        """Handle manual camera selection change."""
        if device_index < 0:
            return

        if not self._camera_thread_started:
            # If thread wasn't started (e.g. no cameras found at startup),
            # we need to allow selection to start it.
            pass

        logger.info("MainWindow: camera selected — device_index=%d", device_index)
        self.video_widget.clear_feed("Connecting…")

        self.camera_worker.stop()
        self.camera_thread.quit()
        self.camera_thread.wait(3000)

        self._disconnect_camera_worker()

        self.camera_worker = CameraWorker(device_index=device_index)
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.new_frame.connect(self.video_widget.set_frame)
        self.camera_worker.frame_ready_for_inference.connect(self.inference_worker.process_frame)
        self.camera_worker.camera_error.connect(self._on_camera_error)
        self.control_panel.preprocessing_params_changed.connect(self.camera_worker.update_params)

        self._camera_thread_started = True
        self.camera_thread.start()
        self.status_footer.set_camera_active()

    # ------------------------------------------------------------------
    # USB hotplug
    # ------------------------------------------------------------------

    def nativeEvent(self, eventType: bytes, message: object) -> tuple[bool, int]:
        if eventType == b"windows_generic_MSG":
            import ctypes
            import ctypes.wintypes as wintypes
            class _MSG(ctypes.Structure):
                _fields_ = [("hWnd", wintypes.HWND), ("message", wintypes.UINT), ("wParam", wintypes.WPARAM), ("lParam", wintypes.LPARAM), ("time", wintypes.DWORD), ("pt", wintypes.POINT)]
            WM_DEVICECHANGE = 0x0219
            try:
                msg = _MSG.from_address(int(message))
                if msg.message == WM_DEVICECHANGE:
                    self._usb_debounce_timer.start(500)
            except Exception:
                pass
        return False, 0

    @pyqtSlot()
    @trace_execution
    def _on_usb_timer_timeout(self) -> None:
        logger.info("MainWindow: USB change detected — launching background scan.")
        self._start_async_scan(is_startup=False)

    @pyqtSlot(list)
    def _on_cameras_found_hotplug(self, cameras: List[Tuple[int, str]]) -> None:
        """Handle hotplug re-enumeration result."""
        current_cameras = []
        for i in range(self.control_panel.camera_combo.count()):
            data = self.control_panel.camera_combo.itemData(i)
            if data is not None and data >= 0:
                current_cameras.append((data, self.control_panel.camera_combo.itemText(i)))

        if cameras == current_cameras:
            return

        logger.info("MainWindow: camera list changed; updating dropdown.")
        active_idx = self.camera_worker._device_index if self._camera_thread_started else -1
        is_active_present = any(idx == active_idx for idx, _ in cameras)

        # blockSignals(True) prevents ControlPanel from emitting camera_selected(0)
        # when we call populate_cameras. We handle the restart logic manually below.
        self.control_panel.camera_combo.blockSignals(True)
        self.control_panel.populate_cameras(cameras)
        
        if is_active_present and active_idx >= 0:
            for i in range(self.control_panel.camera_combo.count()):
                if self.control_panel.camera_combo.itemData(i) == active_idx:
                    self.control_panel.camera_combo.setCurrentIndex(i)
                    break
        self.control_panel.camera_combo.blockSignals(False)

        if cameras:
            self.status_footer.set_camera_active()
            needs_restart = not self._camera_thread_started or not is_active_present
            if needs_restart:
                new_device_idx = cameras[0][0]
                logger.info("MainWindow: Hot-recovery — starting camera index %d.", new_device_idx)
                # Sync dropdown to the new selection
                self.control_panel.camera_combo.blockSignals(True)
                self.control_panel.camera_combo.setCurrentIndex(0)
                self.control_panel.camera_combo.blockSignals(False)
                self._on_camera_selected(new_device_idx)
            else:
                if self.video_widget._status_text in ["Please connect a camera", "Camera disconnected"]:
                    self.video_widget.clear_feed("Connecting…")
        else:
            self.status_footer.set_camera_idle()
            self.video_widget.clear_feed("Please connect a camera")
            if self._camera_thread_started:
                self.camera_worker.stop()
                self.camera_thread.quit()
                self.camera_thread.wait(3000)
                self._camera_thread_started = False

    @pyqtSlot(float)
    def update_fps(self, fps: float) -> None:
        self.status_footer.set_fps(fps)

    def closeEvent(self, event: QCloseEvent) -> None:
        logger.info("MainWindow: closeEvent — beginning safe shutdown.")
        self.camera_worker.stop()
        self.camera_thread.quit()
        self.camera_thread.wait(3000)
        self.inference_thread.quit()
        self.inference_thread.wait(3000)
        event.accept()
