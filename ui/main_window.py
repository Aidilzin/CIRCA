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
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QApplication,
    QSplitter,
    QPushButton,
    QLabel,
    QFrame,
)

from core.utils import enumerate_cameras
from ui.sidebar import SidePanel
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
from ui.help_dialog import HelpDialog
from workers.camera_worker import CameraWorker
from workers.inference_worker import InferenceWorker

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH: str = "models/yolov12_int8.xml"
_DEFAULT_CONFIDENCE_THRESHOLD: float = 0.50

class HardwareScannerThread(QThread):
    """
    Short-lived background thread for USB camera enumeration.

    Architecture note: This is an intentional exception to the project rule
    against subclassing QThread (documented in camera_worker.py). The exception
    is justified because:
      1. The scanner has no persistent run loop and runs exactly once.
      2. Spinning up a full QObject-in-QThread pair for a one-shot task adds
         boilerplate without safety benefit.
      3. The camera enumeration result is returned via a signal — correct
         Qt cross-thread communication is still maintained.
    For long-lived streaming workers (CameraWorker, InferenceWorker), the
    QObject-in-QThread pattern must still be used.
    """
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
        from ui.analytics_dashboard import AnalyticsDashboard

        central = QWidget()
        self.setCentralWidget(central)

        # Main horizontal layout: [ActivityBar] [Workspace (Feed + Analytics)] [SidePanel]
        self.main_layout = QHBoxLayout(central)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)



        # 2. Main Workspace (Vertical stack: TopBar -> Splitter -> Footer)
        self.workspace = QWidget()
        self.workspace_layout = QVBoxLayout(self.workspace)
        self.workspace_layout.setContentsMargins(0, 0, 0, 0)
        self.workspace_layout.setSpacing(0)

        # Top Navigation & Header Bar
        self.top_bar = QFrame()
        self.top_bar.setObjectName("TopNavBar")
        self.top_bar.setFixedHeight(50)
        tb_layout = QHBoxLayout(self.top_bar)
        tb_layout.setContentsMargins(16, 0, 16, 0)
        tb_layout.setSpacing(12)

        # Title/Logo
        logo = QLabel("CIRCA PCB VISION STUDIO")
        logo.setObjectName("TopBarLogo")
        logo.setFont(QFont("Inter", 11, QFont.Weight.Bold))
        tb_layout.addWidget(logo)

        # Armed status pill
        self.armed_status = QLabel("● DIAGNOSTICS ACTIVE")
        self.armed_status.setObjectName("ArmedStatus")
        self.armed_status.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        tb_layout.addWidget(self.armed_status)

        tb_layout.addStretch(1)

        # Helper function for icon buttons
        def create_icon_button(icon_name: str, tooltip: str) -> QPushButton:
            btn = QPushButton(self)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(tooltip)
            
            from ui.theme import ICONS_SVG, ThemeManager
            from PyQt6.QtGui import QIcon, QPixmap, QPainter
            from PyQt6.QtSvg import QSvgRenderer
            palette = ThemeManager().get_palette()
            svg_xml = ICONS_SVG.get(icon_name, "").format(color=palette["TEXT_PRIMARY"])
            if svg_xml:
                renderer = QSvgRenderer(svg_xml.encode('utf-8'))
                pixmap = QPixmap(16, 16)
                pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pixmap)
                renderer.render(painter)
                painter.end()
                btn.setIcon(QIcon(pixmap))
                btn.setIconSize(pixmap.size())
            return btn

        # Action Button: Help Onboarding
        self.top_help_btn = create_icon_button("info", "Quick Onboarding Guide & Help")
        self.top_help_btn.setObjectName("TopHelpButton")
        tb_layout.addWidget(self.top_help_btn)

        # Action Button: Quick Export
        self.top_export_btn = create_icon_button("export", "Export Diagnostics Log")
        self.top_export_btn.setObjectName("TopExportButton")
        tb_layout.addWidget(self.top_export_btn)

        # Action Button: Settings Toggle
        self.top_settings_btn = create_icon_button("settings", "Configure Parameters")
        self.top_settings_btn.setObjectName("TopSettingsButton")
        tb_layout.addWidget(self.top_settings_btn)

        self.workspace_layout.addWidget(self.top_bar)

        # Split Workspace (Horizontal Splitter between Live Feed and Analytics)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("WorkspaceSplitter")
        self.splitter.setHandleWidth(1)

        # Live Feed Panel (Left side of splitter)
        feed_panel = QWidget()
        feed_layout = QVBoxLayout(feed_panel)
        feed_layout.setContentsMargins(0, 0, 0, 0)
        feed_layout.setSpacing(0)

        self.warning_banner = WarningBanner()
        feed_layout.addWidget(self.warning_banner)

        self.video_widget = VideoWidget()
        feed_layout.addWidget(self.video_widget, stretch=1)
        self.splitter.addWidget(feed_panel)

        # Analytics Dashboard Panel (Right side of splitter)
        self.analytics_dashboard = AnalyticsDashboard()
        self.splitter.addWidget(self.analytics_dashboard)

        # Set ratio: Feed takes 65%, Analytics takes 35%
        self.splitter.setSizes([650, 350])
        self.workspace_layout.addWidget(self.splitter, stretch=1)

        # Footer
        self.status_footer = StatusFooter()
        self.workspace_layout.addWidget(self.status_footer)

        self.main_layout.addWidget(self.workspace, stretch=1)

        # 3. Side Panel (rightmost collapsible drawer - connects to toggle)
        self.side_panel = SidePanel()
        self.main_layout.addWidget(self.side_panel)

    def _create_workers(self):
        self.camera_thread = QThread()
        # device_index=0 is a placeholder — the real index is applied in
        # _on_cameras_found_startup() once the scan result is available.
        self.camera_worker = CameraWorker(device_index=0)
        self.camera_worker.moveToThread(self.camera_thread)
        self.inference_thread = QThread()
        self.inference_worker = InferenceWorker()
        self.inference_worker.moveToThread(self.inference_thread)

    def _connect_signals(self):


        # Top nav bar buttons
        self.top_help_btn.clicked.connect(self._show_onboarding_guide)
        self.top_settings_btn.clicked.connect(self.side_panel.toggle)
        self.top_export_btn.clicked.connect(self.analytics_dashboard._on_export_clicked)

        # Engine signals
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.new_frame.connect(self.video_widget.set_frame)
        self.camera_worker.frame_ready_for_inference.connect(self.inference_worker.process_frame)
        self.camera_worker.camera_error.connect(self._on_camera_error)
        self.inference_worker.new_detections.connect(self.video_widget.set_detections)
        self.inference_worker.new_detections.connect(self._on_new_detections)
        self.inference_worker.model_loaded.connect(self._on_model_loaded)
        self.inference_worker.inference_error.connect(self._on_inference_error)

        # Analytics connections
        self.video_widget.fps_updated.connect(self.status_footer.set_fps)
        self.video_widget.fps_updated.connect(self.analytics_dashboard.handle_fps)
        self.inference_worker.new_detections.connect(self.analytics_dashboard.handle_detections)
        self.analytics_dashboard.defect_hovered.connect(self.video_widget.set_highlight_index)

        # Parameter signals from Side Panel
        self.side_panel.preprocessing_params_changed.connect(self.camera_worker.update_params)
        self.side_panel.inference_params_changed.connect(self.inference_worker.update_params)
        self.side_panel.inference_params_changed.connect(self._on_inference_params_changed)
        self.side_panel.camera_selected.connect(self._on_camera_selected)
        self.side_panel.theme_changed.connect(self._on_theme_changed)
        self.side_panel.model_selected.connect(self._on_model_selected)
        self.side_panel.recommend_model_requested.connect(self._on_recommend_model_requested)
        self.camera_worker.auto_params_updated.connect(self.side_panel.update_auto_params_ui)
        self.inference_worker.benchmark_completed.connect(self._on_benchmark_completed)

    def _start_workers(self):
        self.status_footer.set_model_loading()

    @pyqtSlot(object)
    def _on_theme_changed(self, mode):
        QApplication.instance().setStyleSheet(build_qss(mode))
        self._update_icon_button(self.top_help_btn, "info")
        self._update_icon_button(self.top_export_btn, "export")
        self._update_icon_button(self.top_settings_btn, "settings")

    def _update_icon_button(self, btn: QPushButton, icon_name: str):
        from ui.theme import ICONS_SVG, ThemeManager
        from PyQt6.QtGui import QIcon, QPixmap, QPainter
        from PyQt6.QtSvg import QSvgRenderer
        palette = ThemeManager().get_palette()
        svg_xml = ICONS_SVG.get(icon_name, "").format(color=palette["TEXT_PRIMARY"])
        if svg_xml:
            renderer = QSvgRenderer(svg_xml.encode('utf-8'))
            pixmap = QPixmap(16, 16)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            btn.setIcon(QIcon(pixmap))
            btn.setIconSize(pixmap.size())

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
            # ── Bug fix: update the worker to the ACTUAL selected device index ──
            # _create_workers() used a placeholder index=0. Now that we know
            # which device the user has, reconfigure the worker before starting.
            if self.camera_worker._device_index != device_index:
                logger.info(
                    "Startup: updating CameraWorker device_index from %d to %d",
                    self.camera_worker._device_index,
                    device_index,
                )
                self.camera_worker._device_index = device_index
            self.status_footer.set_camera_active()
            self._camera_thread_started = True
            self.camera_thread.start()
        else:
            self.status_footer.set_camera_idle()
            self.video_widget.clear_feed("Please connect a camera")
        # Auto-detect best model on startup (on-by-default feature)
        self._on_recommend_model_requested()
        # Show onboarding guide on startup
        QTimer.singleShot(500, self._show_onboarding_guide)

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
        """Handle a camera_error signal from CameraWorker.

        The worker has already set _running=False and broken out of its loop,
        so it will not emit any more frames. We must still quit+wait the thread
        before marking ourselves as idle — otherwise the QThread gets destroyed
        while its internal thread is still unwinding, which causes Qt to print
        "QThread: Destroyed while thread is still running" and can crash.
        """
        logger.error("Camera error signal received: %s", message)
        self.status_footer.set_camera_error(message)

        # Stop and wait for the thread to finish before we clear state.
        # The worker already set _running=False, so quit() + wait() is safe.
        if self.camera_thread.isRunning():
            self.camera_thread.quit()
            if not self.camera_thread.wait(3000):
                logger.warning("_on_camera_error: camera thread did not stop in 3 s — forcing termination.")
                self.camera_thread.terminate()
                self.camera_thread.wait()

        self._camera_thread_started = False
        self.video_widget.clear_feed("Please connect a camera")

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

    @pyqtSlot(str)
    def _on_model_selected(self, model_path: str):
        if not os.path.isfile(model_path):
            self.status_footer.set_model_error("Model missing")
            return
        self.status_footer.set_model_loading()
        # Thread-safe load model call to worker
        QTimer.singleShot(0, lambda: self.inference_worker.load_model(model_path))

    @pyqtSlot()
    def _show_onboarding_guide(self):
        dialog = HelpDialog(self)
        dialog.exec()

    @pyqtSlot()
    def _on_recommend_model_requested(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        nano_path = os.path.join(project_root, "models", "yolov12_int8.xml")
        if not os.path.isfile(nano_path):
            self.status_footer.set_model_error("Prod model missing")
            return
        
        self.status_footer.set_status_text("Benchmarking hardware...")
        QTimer.singleShot(0, lambda: self.inference_worker.run_benchmark(nano_path))

    @pyqtSlot(str, float)
    def _on_benchmark_completed(self, model_path: str, avg_latency_ms: float):
        self.status_footer.set_model_ready()
        
        if avg_latency_ms < 12.0:
            recommended_substring = "Medium (FP16)"
            tier_name = "High-End (GPU)"
        elif avg_latency_ms < 28.0:
            recommended_substring = "Small (FP16)"
            tier_name = "Mid-Range (iGPU/CPU)"
        else:
            recommended_substring = "Nano (FP16"
            tier_name = "Budget/CPU-Only"

        matched_idx = -1
        for i in range(self.side_panel.model_combo.count()):
            text = self.side_panel.model_combo.itemText(i)
            if recommended_substring in text:
                matched_idx = i
                break

        if matched_idx >= 0:
            if self.side_panel.model_combo.currentIndex() == matched_idx:
                path = self.side_panel.model_combo.itemData(matched_idx)
                if path:
                    self._on_model_selected(path)
            else:
                self.side_panel.model_combo.setCurrentIndex(matched_idx)
            model_name = self.side_panel.model_combo.itemText(matched_idx)
            logger.info("Benchmark latency %.2f ms -> Auto-selected %s", avg_latency_ms, model_name)
            self.status_footer.set_status_text(f"Auto-selected {model_name} ({avg_latency_ms:.1f}ms - {tier_name})")
            QTimer.singleShot(4000, lambda: self.status_footer.set_model_ready())
        else:
            logger.warning("Could not find matching dropdown item for recommended model: %s", recommended_substring)

    def _on_usb_timer_timeout(self): self._start_async_scan()

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt C++ API override
        self.camera_worker.stop()
        self.camera_thread.quit()
        self.camera_thread.wait(3000)
        self.inference_thread.quit()
        self.inference_thread.wait(3000)
        event.accept()