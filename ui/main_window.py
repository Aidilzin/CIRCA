"""
ui/main_window.py
-----------------
MainWindow — refactored for 2026 AI Studio Aesthetics with VS Code-style Sidebar.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QApplication,
)

from core.utils import enumerate_cameras
from ui.sidebar import ActivityBar, SidePanel
from ui.status_footer import StatusFooter
from ui.theme import (
    WINDOW_DEFAULT_HEIGHT,
    WINDOW_DEFAULT_WIDTH,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
    build_qss
)
from ui.video_widget import VideoWidget
from ui.warning_banner import WarningBanner
from workers.camera_worker import CameraWorker
from workers.inference_worker import InferenceWorker

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH: str = "models/yolov12_int8.xml"
_DEFAULT_CONFIDENCE_THRESHOLD: float = 0.50

class HardwareScannerThread(QThread):
    cameras_found = pyqtSignal(list)
    def __init__(self, active_idx=-1, names=None, parent=None):
        super().__init__(parent)
        self._active_idx, self._names = active_idx, names
    def run(self):
        import ctypes
        try:
            ctypes.windll.ole32.CoInitialize(None)
        except Exception:
            pass
        cameras = enumerate_cameras(active_device_index=self._active_idx, qt_names=self._names)
        self.cameras_found.emit(cameras)
        try:
            ctypes.windll.ole32.CoUninitialize()
        except Exception:
            pass

class MainWindow(QMainWindow):
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model_path = model_path
        self._confidence_threshold = _DEFAULT_CONFIDENCE_THRESHOLD
        self._camera_thread_started = False
        self._scanner_thread = None

        self._setup_window()
        self._build_ui()
        self._create_workers()
        self._connect_signals()
        self._start_workers()
        QTimer.singleShot(200, self._post_start_init)

    def _setup_window(self):
        self.setWindowTitle("CIRCA — PCB Defect Detection")
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resize(WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT)
        self._usb_debounce_timer = QTimer(self)
        self._usb_debounce_timer.setSingleShot(True)
        self._usb_debounce_timer.timeout.connect(self._on_usb_timer_timeout)

        # Monitor for USB Camera hot-plug events via QtMultimedia
        try:
            from PyQt6.QtMultimedia import QMediaDevices
            self._media_devices = QMediaDevices(self)
            self._media_devices.videoInputsChanged.connect(self._on_video_inputs_changed)
        except Exception as e:
            logger.error("Failed to initialize QMediaDevices: %s", e)

    @pyqtSlot()
    def _on_video_inputs_changed(self):
        """Triggered when a camera is plugged/unplugged."""
        logger.info("USB Video Input change detected — debouncing scan...")
        self._usb_debounce_timer.start(1000)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # Main horizontal layout: [ActivityBar] [SidePanel] [Central Workspace]
        self.main_layout = QHBoxLayout(central)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # 1. Activity Bar
        self.activity_bar = ActivityBar()
        self.main_layout.addWidget(self.activity_bar)

        # 2. Side Panel
        self.side_panel = SidePanel()
        self.main_layout.addWidget(self.side_panel)

        # 3. Central Workspace (Vertical: Banner -> Video -> Footer)
        self.workspace = QWidget()
        self.workspace_layout = QVBoxLayout(self.workspace)
        self.workspace_layout.setContentsMargins(0, 0, 0, 0)
        self.workspace_layout.setSpacing(0)

        self.warning_banner = WarningBanner()
        self.workspace_layout.addWidget(self.warning_banner)

        self.video_widget = VideoWidget()
        self.workspace_layout.addWidget(self.video_widget, stretch=1)

        self.status_footer = StatusFooter()
        self.workspace_layout.addWidget(self.status_footer)

        self.main_layout.addWidget(self.workspace, stretch=1)

    def _create_workers(self):
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker(device_index=0)
        self.camera_worker.moveToThread(self.camera_thread)
        self.inference_thread = QThread()
        self.inference_worker = InferenceWorker()
        self.inference_worker.moveToThread(self.inference_thread)

    def _connect_signals(self):
        # Sidebar interactions
        self.activity_bar.tab_selected.connect(self.side_panel.set_tab)
        self.side_panel.panel_toggled.connect(self.activity_bar.handle_panel_toggle)

        # Engine signals
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.new_frame.connect(self.video_widget.set_frame)
        self.camera_worker.frame_ready_for_inference.connect(self.inference_worker.process_frame)
        self.camera_worker.camera_error.connect(self._on_camera_error)
        self.inference_worker.new_detections.connect(self.video_widget.set_detections)
        self.inference_worker.new_detections.connect(self._on_new_detections)
        self.inference_worker.model_loaded.connect(self._on_model_loaded)
        self.inference_worker.inference_error.connect(self._on_inference_error)

        # Parameter signals from Side Panel
        self.side_panel.preprocessing_params_changed.connect(self.camera_worker.update_params)
        self.side_panel.inference_params_changed.connect(self.inference_worker.update_params)
        self.side_panel.inference_params_changed.connect(self._on_inference_params_changed)
        self.side_panel.camera_selected.connect(self._on_camera_selected)
        self.side_panel.theme_changed.connect(self._on_theme_changed)

    def _start_workers(self):
        self.status_footer.set_model_loading()

    @pyqtSlot(object)
    def _on_theme_changed(self, mode):
        QApplication.instance().setStyleSheet(build_qss(mode))

    def _post_start_init(self):
        self.inference_thread.start()
        self._start_async_scan(is_startup=True)

    def _start_async_scan(self, is_startup=False):
        try:
            if self._scanner_thread and self._scanner_thread.isRunning():
                return
        except Exception:
            self._scanner_thread = None
        qt_names = []
        try:
            from PyQt6.QtMultimedia import QMediaDevices
            qt_names = [d.description() for d in QMediaDevices.videoInputs()]
        except Exception:
            pass
        active_idx = self.camera_worker._device_index if self._camera_thread_started else -1
        self._scanner_thread = HardwareScannerThread(active_idx, qt_names, self)
        if is_startup:
            self._scanner_thread.cameras_found.connect(self._on_cameras_found_startup)
        else:
            self._scanner_thread.cameras_found.connect(self._on_cameras_found_hotplug)
        self._scanner_thread.finished.connect(self._scanner_thread.deleteLater)
        self._scanner_thread.start()

    @pyqtSlot(list)
    def _on_cameras_found_startup(self, cameras):
        self.side_panel.populate_cameras(cameras)
        device_index = self.side_panel.camera_combo.currentData()
        if cameras and isinstance(device_index, int) and device_index >= 0:
            self.status_footer.set_camera_active()
            self._camera_thread_started = True
            self.camera_thread.start()
        else:
            self.status_footer.set_camera_idle()
            self.video_widget.clear_feed("Please connect a camera")
        if os.path.isfile(self._model_path):
            path = self._model_path
            QTimer.singleShot(0, lambda: self.inference_worker.load_model(path))
        else:
            self.status_footer.set_model_error("Model missing")

    @pyqtSlot(list)
    def _on_cameras_found_hotplug(self, cameras):
        self.side_panel.populate_cameras(cameras)
        if not cameras:
            self.status_footer.set_camera_idle()
            self.video_widget.clear_feed("Please connect a camera")
        elif not self._camera_thread_started:
            # Auto-reconnect if we found a camera and we're currently idle
            logger.info("Hotplug: camera found while idle, attempting auto-connect...")
            idx = self.side_panel.camera_combo.currentData()
            if isinstance(idx, int) and idx >= 0:
                self.video_widget.set_status_text("Connecting…")
                self._on_camera_selected(idx)

    @pyqtSlot(str)
    def _on_camera_error(self, message: str) -> None:
        logger.error("Camera error signal received: %s", message)
        self.status_footer.set_camera_error(message)
        self._camera_thread_started = False
        self.video_widget.clear_feed("camera unavailable")

    @pyqtSlot()
    def _on_model_loaded(self) -> None:
        self.status_footer.set_model_ready()

    @pyqtSlot(str)
    def _on_inference_error(self, msg: str) -> None:
        self.status_footer.set_model_error(msg)

    @pyqtSlot(object)
    def _on_new_detections(self, res) -> None:
        self.status_footer.set_detection_count(len(res.boxes))
        self.warning_banner.update_confidence(res.average_confidence, self._confidence_threshold)

    @pyqtSlot(object)
    def _on_inference_params_changed(self, p) -> None:
        self._confidence_threshold = p.confidence_threshold

    @pyqtSlot(int)
    def _on_camera_selected(self, idx):
        if idx < 0:
            return
        logger.info("Switching to camera index %d", idx)

        # 1. Properly tear down existing worker and thread
        if self.camera_thread.isRunning():
            self.camera_worker.stop()
            self.camera_thread.quit()
            if not self.camera_thread.wait(2000):
                logger.warning("Camera thread timed out during switch; forcing termination.")
                self.camera_thread.terminate()

        # 2. Create new worker for the new index
        self.camera_worker = CameraWorker(device_index=idx)
        self.camera_worker.moveToThread(self.camera_thread)

        # 3. Re-wire core signals (mapping to the new worker instance)
        self.camera_thread.started.disconnect()  # Clear old connection
        self.camera_thread.started.connect(self.camera_worker.run)

        self.camera_worker.new_frame.connect(self.video_widget.set_frame)
        self.camera_worker.frame_ready_for_inference.connect(self.inference_worker.process_frame)
        self.camera_worker.camera_error.connect(self._on_camera_error)
        self.side_panel.preprocessing_params_changed.connect(self.camera_worker.update_params)
        # Push the current slider state into the new worker so it doesn't start
        # with default params regardless of where the UI sliders are positioned.
        self.side_panel._emit_preprocessing_params()

        # 4. Fire it up
        self._camera_thread_started = True
        self.camera_thread.start()
        self.status_footer.set_camera_active()

    def _on_usb_timer_timeout(self): self._start_async_scan()

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt C++ API override
        self.camera_worker.stop()
        self.camera_thread.quit()
        self.camera_thread.wait(3000)
        self.inference_thread.quit()
        self.inference_thread.wait(3000)
        event.accept()