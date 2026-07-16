"""
ui/main_window.py
-----------------
MainWindow — refactored for 2026 AI Studio Aesthetics with VS Code-style Sidebar.
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFileDialog,
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

from core.inference_engine import CLASS_LABELS
from core.pcb_guard import is_likely_pcb
from core.utils import bgr_frame_to_qimage, enumerate_cameras
from ui.sidebar import SidePanel
from ui.status_footer import StatusFooter
from ui.theme import (
    WINDOW_DEFAULT_HEIGHT,
    WINDOW_DEFAULT_WIDTH,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
    build_qss
)
from ui.image_inspect_widget import ImageInspectWidget
from ui.warning_banner import WarningBanner
from ui.help_dialog import HelpDialog
from workers.camera_worker import CameraWorker
from workers.inference_worker import InferenceWorker
from core.models import PreprocessParams, InferenceParams

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH: str = "models/yolov12_int8.xml"
_DEFAULT_CONFIDENCE_THRESHOLD: float = 0.50

class NavButton(QFrame):  # noqa: E302
    clicked = pyqtSignal()
    def __init__(self, icon_name: str, tooltip: str, text: str, parent=None):
        super().__init__(parent)
        self.setObjectName("NavButton")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tooltip)
        self._active = False
        self._icon_name = icon_name
        self._text = text
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 8)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.icon_label = QLabel()
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.icon_label)
        
        self.text_label = QLabel(text)
        self.text_label.setObjectName("NavButtonText")
        self.text_label.setFont(QFont("Inter", 8, QFont.Weight.Medium))
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.text_label)
        
        self._update_icon()

    def _update_icon(self):
        from ui.theme import ICONS_SVG, ThemeManager
        from PyQt6.QtGui import QPixmap, QPainter
        from PyQt6.QtSvg import QSvgRenderer
        palette = ThemeManager().get_palette()
        svg_xml = ICONS_SVG.get(self._icon_name, "").format(color=palette["TEXT_PRIMARY"])
        if svg_xml:
            renderer = QSvgRenderer(svg_xml.encode('utf-8'))
            pixmap = QPixmap(20, 20)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            self.icon_label.setPixmap(pixmap)

    def set_active(self, active: bool):
        self._active = active
        self.setProperty("active", "true" if active else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.text_label.setProperty("active", "true" if active else "false")
        self.text_label.style().unpolish(self.text_label)
        self.text_label.style().polish(self.text_label)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


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
        except (OSError, AttributeError):
            pass
        cameras = enumerate_cameras(active_device_index=self._active_idx, qt_names=self._names)
        self.cameras_found.emit(cameras)
        try:
            ctypes.windll.ole32.CoUninitialize()
        except (OSError, AttributeError):
            pass

class MainWindow(QMainWindow):
    # Cross-thread signal: routes inference requests from the GUI thread to the
    # InferenceWorker living in the Inference Thread via Qt's queued connection.
    # Using a signal instead of a direct method call prevents the ~40-200ms
    # synchronous OpenVINO inference from blocking the GUI thread (Fix #4).
    _request_inference: pyqtSignal = pyqtSignal(object, object)

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model_path = model_path
        self._confidence_threshold = _DEFAULT_CONFIDENCE_THRESHOLD
        self._camera_thread_started = False
        self._scanner_thread = None
        self._in_camera_preview = False
        self._current_raw_frame = None
        self._preprocess_params = PreprocessParams()
        self.onboarding_controller = None

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
            self._media_devices.videoInputsChanged.connect(self._on_camera_inputs_changed)
        except (ImportError, RuntimeError) as e:
            logger.error("Failed to initialize QMediaDevices: %s", e)

    @pyqtSlot()
    def _on_camera_inputs_changed(self):
        """Triggered when a camera is plugged/unplugged."""
        logger.info("USB Camera Input change detected — debouncing scan...")
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

        # 1. Left Navigation Sidebar (Static 80px width, stacked icon + text buttons)
        self.nav_sidebar = QFrame()
        self.nav_sidebar.setObjectName("NavSidebar")
        self.nav_sidebar.setFixedWidth(80)
        nav_layout = QVBoxLayout(self.nav_sidebar)
        nav_layout.setContentsMargins(4, 16, 4, 16)
        nav_layout.setSpacing(10)

        # Title/Logo Header
        self.logo_label = QLabel("CIRCA")
        self.logo_label.setObjectName("NavLogo")
        self.logo_label.setFont(QFont("Inter", 10, QFont.Weight.Bold))
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self.logo_label)

        # Active Status Label
        self.status_pill = QLabel("● ACTIVE")
        self.status_pill.setObjectName("NavStatus")
        self.status_pill.setFont(QFont("Inter", 7, QFont.Weight.Bold))
        self.status_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self.status_pill)

        # Helper function for custom NavButtons
        def create_icon_button(icon_name: str, tooltip: str, text: str = "") -> NavButton:
            btn = NavButton(icon_name, tooltip, text, self)
            return btn

        # Helper function to create visual separators
        def create_divider() -> QFrame:
            div = QFrame()
            div.setObjectName("ToolbarDivider")
            div.setProperty("class", "ToolbarDivider")
            div.setFrameShape(QFrame.Shape.HLine)
            div.setFrameShadow(QFrame.Shadow.Sunken)
            return div

        nav_layout.addWidget(create_divider())

        # Action Button: Load Image (primary workflow)
        self.top_load_btn = create_icon_button("folder_open", "Load PCB Image for Inspection", "Load Board")
        self.top_load_btn.setObjectName("TopLoadButton")
        self.top_load_btn.setProperty("active", "false")
        nav_layout.addWidget(self.top_load_btn)

        # Action Button: Capture from Camera
        self.top_capture_btn = create_icon_button("camera", "Capture Single Frame from Camera", "Capture")
        self.top_capture_btn.setObjectName("TopCaptureButton")
        self.top_capture_btn.setProperty("active", "false")
        nav_layout.addWidget(self.top_capture_btn)

        # Divider between Input and Execution Control
        nav_layout.addWidget(create_divider())

        # Action Button: Re-run Diagnostics / Reanalyze
        self.top_reanalyze_btn = create_icon_button("refresh", "Re-run Diagnostics (Reanalyze)", "Reanalyze")
        self.top_reanalyze_btn.setObjectName("TopReanalyzeButton")
        self.top_reanalyze_btn.setProperty("active", "false")
        nav_layout.addWidget(self.top_reanalyze_btn)

        # Divider between Execution Control and Output Logs
        nav_layout.addWidget(create_divider())

        # Action Button: Quick Export
        self.top_export_btn = create_icon_button("export", "Export Diagnostics Log", "Export Log")
        self.top_export_btn.setObjectName("TopExportButton")
        self.top_export_btn.setProperty("active", "false")
        nav_layout.addWidget(self.top_export_btn)

        # Divider between Output Logs and Interface Config
        nav_layout.addWidget(create_divider())

        # Action Button: Help Onboarding
        self.top_help_btn = create_icon_button("info", "Quick Onboarding Guide & Help", "Help")
        self.top_help_btn.setObjectName("TopHelpButton")
        self.top_help_btn.setProperty("active", "false")
        nav_layout.addWidget(self.top_help_btn)

        # Action Button: Settings Toggle
        self.top_settings_btn = create_icon_button("settings", "Configure Parameters", "Settings")
        self.top_settings_btn.setObjectName("TopSettingsButton")
        self.top_settings_btn.setProperty("active", "false")
        nav_layout.addWidget(self.top_settings_btn)

        nav_layout.addStretch(1)

        # Split Workspace (Horizontal Splitter between Inspection Panel and Analytics)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("WorkspaceSplitter")
        self.splitter.setHandleWidth(1)

        # Inspection Panel (Left side of splitter)
        feed_panel = QWidget()
        feed_layout = QVBoxLayout(feed_panel)
        feed_layout.setContentsMargins(0, 0, 0, 0)
        feed_layout.setSpacing(0)

        self.warning_banner = WarningBanner()
        feed_layout.addWidget(self.warning_banner)

        self.inspect_widget = ImageInspectWidget()
        feed_layout.addWidget(self.inspect_widget, stretch=1)
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

        self.main_layout.addWidget(self.nav_sidebar)

        # 3. Side Panel (collapsible drawer - positioned adjacent to navigation sidebar)
        self.side_panel = SidePanel()
        self.main_layout.addWidget(self.side_panel)

        self.main_layout.addWidget(self.workspace, stretch=1)

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
        self.top_load_btn.clicked.connect(self._on_load_image_clicked)
        self.top_load_btn.clicked.connect(lambda: self._set_active_nav_button(self.top_load_btn))
        self.top_capture_btn.clicked.connect(self._on_capture_clicked)
        self.top_capture_btn.clicked.connect(lambda: self._set_active_nav_button(self.top_capture_btn))
        self.top_reanalyze_btn.clicked.connect(self._reanalyze_current_image)
        self.top_reanalyze_btn.clicked.connect(lambda: self._set_active_nav_button(self.top_reanalyze_btn))
        self.top_help_btn.clicked.connect(self._show_onboarding_guide)
        self.top_help_btn.clicked.connect(lambda: self._set_active_nav_button(self.top_help_btn))
        self.top_settings_btn.clicked.connect(self.side_panel.toggle)
        self.top_settings_btn.clicked.connect(lambda: self._set_active_nav_button(self.top_settings_btn))
        self.top_export_btn.clicked.connect(self.analytics_dashboard._on_export_clicked)
        self.top_export_btn.clicked.connect(lambda: self._set_active_nav_button(self.top_export_btn))

        # Side panel toggle synchronization
        self.side_panel.panel_toggled.connect(self._on_side_panel_toggled)

        # Image inspect widget: user loaded/captured an image → run inference
        self.inspect_widget.image_loaded.connect(self._on_image_loaded)

        # Camera worker (used only for optional single-frame capture)
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.camera_error.connect(self._on_camera_error)

        # Inference results flow back to the inspect widget
        self.inference_worker.new_detections.connect(self.inspect_widget.set_detections)
        self.inference_worker.new_detections.connect(self._on_new_detections)
        self.inference_worker.model_loaded.connect(self._on_model_loaded)
        self.inference_worker.inference_error.connect(self._on_inference_error)

        # Analytics connections
        self.inspect_widget.inference_time_updated.connect(self.analytics_dashboard.handle_inference_time)
        self.inference_worker.new_detections.connect(self.analytics_dashboard.handle_detections)
        self.analytics_dashboard.defect_hovered.connect(self.inspect_widget.set_highlight_index)
        self.analytics_dashboard.defect_clicked.connect(self.inspect_widget.set_zoom_index)
        self.analytics_dashboard.fp_marked.connect(self._on_mark_false_positive)
        self.analytics_dashboard.next_board_requested.connect(self._on_next_board_requested)

        # Parameter signals from Side Panel
        self.side_panel.preprocessing_params_changed.connect(self.camera_worker.update_params)
        self.side_panel.preprocessing_params_changed.connect(self._on_preprocessing_params_changed)
        self.side_panel.inference_params_changed.connect(self.inference_worker.update_params)
        self.side_panel.inference_params_changed.connect(self._on_inference_params_changed)
        self.inference_worker.model_loaded.connect(self._reanalyze_current_image)
        self.side_panel.camera_selected.connect(self._on_camera_selected)
        self.side_panel.theme_changed.connect(self._on_theme_changed)
        self.side_panel.model_selected.connect(self._on_model_selected)
        self.side_panel.recommend_model_requested.connect(self._on_recommend_model_requested)
        self.camera_worker.auto_params_updated.connect(self.side_panel.update_auto_params_ui)
        self.inference_worker.benchmark_completed.connect(self._on_benchmark_completed)
        self.side_panel.reset_onboarding_requested.connect(self.start_onboarding_tour)

        # Cross-thread inference dispatch (Fix #4 — prevents GUI thread blocking)
        self._request_inference.connect(self.inference_worker.process_frame)

    def _start_workers(self):
        self._capture_pending: bool = False
        self.status_footer.set_model_loading()

    @pyqtSlot(object)
    def _on_theme_changed(self, mode):
        QApplication.instance().setStyleSheet(build_qss(mode))
        self._update_icon_button(self.top_help_btn, "info")
        self._update_icon_button(self.top_export_btn, "export")
        self._update_icon_button(self.top_settings_btn, "settings")

    def _update_icon_button(self, btn, icon_name: str):
        if hasattr(btn, "_update_icon"):
            btn._icon_name = icon_name
            btn._update_icon()

    def _post_start_init(self):
        self.inference_thread.start()
        self._start_async_scan(is_startup=True)

        from PyQt6.QtCore import QSettings
        settings = QSettings("CIRCA", "VisionStudio")
        has_run = settings.value("hasRunOnboarding", False, type=bool)
        if not has_run:
            QTimer.singleShot(500, self.start_onboarding_tour)

    def start_onboarding_tour(self):
        from ui.onboarding import OnboardingController
        if not self.onboarding_controller:
            self.onboarding_controller = OnboardingController(self)
        self.onboarding_controller.start_tour()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "onboarding_controller") and self.onboarding_controller and self.onboarding_controller.overlay.isVisible():
            self.onboarding_controller.overlay.setGeometry(self.rect())

    def _start_async_scan(self, is_startup=False):
        try:
            if self._scanner_thread and self._scanner_thread.isRunning():
                return
        except (AttributeError, RuntimeError):
            self._scanner_thread = None
        qt_names = []
        try:
            from PyQt6.QtMultimedia import QMediaDevices
            qt_names = [d.description() for d in QMediaDevices.videoInputs()]
        except (ImportError, RuntimeError):
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
        # Auto-detect best model on startup (on-by-default feature)
        self._on_recommend_model_requested()

    @pyqtSlot(list)
    def _on_cameras_found_hotplug(self, cameras):
        self.side_panel.populate_cameras(cameras)
        if not cameras:
            self.status_footer.set_camera_idle()
            self.inspect_widget.clear_feed("Please connect a camera")
        elif not self._camera_thread_started:
            # Auto-reconnect if we found a camera and we're currently idle
            logger.info("Hotplug: camera found while idle, attempting auto-connect...")
            idx = self.side_panel.camera_combo.currentData()
            if isinstance(idx, int) and idx >= 0:
                self.inspect_widget.set_status_text("Connecting…")
                self._on_camera_selected(idx)

    @pyqtSlot(object)
    def _on_image_loaded(self, bgr_frame: object) -> None:
        """Received from ImageInspectWidget when user loads or captures an image.
        Runs PCB guard, then dispatches to inference worker."""
        if not isinstance(bgr_frame, np.ndarray) or bgr_frame.size == 0:
            return

        ok, reason = is_likely_pcb(bgr_frame)
        if not ok:
            self.inspect_widget.set_rejected(reason)
            self.status_footer.set_status_text("PCB Guard: " + reason)
            return

        self._current_raw_frame = bgr_frame
        self._reanalyze_current_image()

    @pyqtSlot()
    def _on_load_image_clicked(self) -> None:
        """Open file picker and load selected image into inspect widget."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load PCB Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if path:
            self.analytics_dashboard.set_image_name(os.path.basename(path))
            self.inspect_widget.load_image_from_path(path)

    @pyqtSlot()
    def _on_capture_clicked(self) -> None:
        """Toggle live camera viewfinder preview. Snaps photo and stops camera on second click."""
        if not self._in_camera_preview:
            # 1. Start live preview viewfinder
            device_index = self.side_panel.camera_combo.currentData()
            if not isinstance(device_index, int) or device_index < 0:
                self.status_footer.set_status_text("No camera selected")
                return
            
            self._in_camera_preview = True
            self.inspect_widget.start_preview()
            self._on_camera_selected(device_index)
            
            # Connect viewfinder frame display
            self.camera_worker.new_frame.connect(self.inspect_widget.set_frame)
            self._update_icon_button(self.top_capture_btn, "check")
            self.top_capture_btn.setToolTip("Snap Photo and Close Camera")
            self.status_footer.set_status_text("Viewfinder Active")
        else:
            # 2. Snap photo and stop camera device
            self._in_camera_preview = False
            try:
                self.camera_worker.new_frame.disconnect(self.inspect_widget.set_frame)
            except (TypeError, RuntimeError):
                pass
            
            qimage = self.inspect_widget._qimage
            
            # Properly shut down camera device immediately (turns off webcam LED)
            if self.camera_thread.isRunning():
                self.camera_worker.stop()
                self.camera_thread.quit()
                self.camera_thread.wait(2000)
            self._camera_thread_started = False
            self.status_footer.set_camera_idle()
            
            if qimage and not qimage.isNull():
                qimage_rgb = qimage.convertToFormat(qimage.Format.Format_RGB888)
                w, h = qimage_rgb.width(), qimage_rgb.height()
                ptr = qimage_rgb.bits()
                ptr.setsize(h * w * 3)
                rgb = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                self.analytics_dashboard.set_image_name("Camera Capture")
                self.inspect_widget.load_image_from_array(bgr)
            else:
                self.inspect_widget.clear_feed("Capture failed — no frame received")
                
            self._update_icon_button(self.top_capture_btn, "camera")
            self.top_capture_btn.setToolTip("Capture Single Frame from Camera")

    @pyqtSlot()
    def _on_next_board_requested(self) -> None:
        """Reset the inspect widget and footer to empty state."""
        self._current_raw_frame = None
        self.inspect_widget.clear_feed()
        self.status_footer.set_detection_count(-1)
        self.status_footer.set_status_text("Ready")
        self.warning_banner.reset_board_state()

    @pyqtSlot(PreprocessParams)
    def _on_preprocessing_params_changed(self, params: PreprocessParams) -> None:
        self._preprocess_params = params
        if self._current_raw_frame is not None:
            self.inspect_widget.set_analyzing("Applying image enhancements…")
            self.status_footer.set_status_text("Applying enhancements…")
        # 50ms delay to let Qt render the loader overlay
        QTimer.singleShot(50, self._reanalyze_current_image)

    @pyqtSlot(InferenceParams)
    def _on_inference_params_changed(self, params: InferenceParams) -> None:
        if self._current_raw_frame is not None:
            self.inspect_widget.set_analyzing("Re-running neural diagnostics…")
            self.status_footer.set_status_text("Adjusting sensitivity…")
        QTimer.singleShot(50, self._reanalyze_current_image)

    @pyqtSlot()
    def _reanalyze_current_image(self) -> None:
        """Reprocess and run inference on the current loaded raw board frame."""
        if self._current_raw_frame is None or not isinstance(self._current_raw_frame, np.ndarray):
            return

        # Auto-reset zoom to prevent crop rendering issues during re-inference
        self.inspect_widget.set_zoom_index(-1)

        # Dispatch via cross-thread signal (Fix #4) — InferenceWorker lives in the
        # Inference Thread; emitting through _request_inference guarantees Qt's
        # queued connection delivers the call there, NOT on the GUI thread.
        self.inspect_widget.set_analyzing("Re-running neural diagnostics…")
        self.status_footer.set_status_text("Analyzing…")

        self._request_inference.emit(self._current_raw_frame, self._preprocess_params)

    @pyqtSlot(int)
    def _on_mark_false_positive(self, box_idx: int) -> None:
        """Handle user marking a detection bounding box as a false positive.
        Exclude it from active view/reports, update status, and log the feedback
        along with the image/coordinates to a local fine-tuning dataset folder."""
        if not self.inspect_widget._detections or not self.inspect_widget._detections.boxes:
            return
        
        if box_idx >= len(self.inspect_widget._detections.boxes):
            return

        if self._current_raw_frame is None or not isinstance(self._current_raw_frame, np.ndarray):
            return

        # Pop the box out
        boxes = self.inspect_widget._detections.boxes
        fp_box = boxes.pop(box_idx)
        


        # Reset zoom index to prevent crop misalignment on next click
        self.inspect_widget.set_zoom_index(-1)

        # Trigger fine-tuning data serialization
        image_name = self.analytics_dashboard._image_name
        try:
            base_filename = self._save_feedback_data(self._current_raw_frame, boxes, fp_box, image_name)
            logger.info("False positive marked and saved to training feedback folder: %s", base_filename)
            self.status_footer.set_status_text("Logged feedback for fine-tuning")
        except OSError as exc:
            logger.error("Failed to save feedback: %s", exc)

        # Redraw viewport and update checklist/stats in dashboard
        self.inspect_widget.update()
        self.analytics_dashboard.handle_detections(self.inspect_widget._detections)
        self._on_new_detections(self.inspect_widget._detections)

    def _save_feedback_data(self, image: np.ndarray, boxes: list, fp_box, image_name: str) -> str:
        """
        Save raw frame and corrected labels (excluding the false positive)
        to datasets/feedback_fine_tune/ to build a fine-tuning dataset over time.
        """
        feedback_dir = os.path.join("datasets", "feedback_fine_tune")
        images_dir = os.path.join(feedback_dir, "images")
        labels_dir = os.path.join(feedback_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_name = "".join([c if c.isalnum() else "_" for c in image_name])
        base_filename = f"feedback_{clean_name}_{timestamp}"

        # Write image file
        image_path = os.path.join(images_dir, f"{base_filename}.png")
        cv2.imwrite(image_path, image)

        # Write labels file in YOLO format
        h, w = image.shape[:2]
        reverse_labels = {v: k for k, v in CLASS_LABELS.items()}
        label_path = os.path.join(labels_dir, f"{base_filename}.txt")

        with open(label_path, "w") as f:
            for box in boxes:
                class_id = reverse_labels.get(box.class_name, -1)
                if class_id == -1:
                    continue
                # YOLO: class_id x_center y_center width height (normalized 0..1)
                x_center = (box.x + box.width / 2.0) / w
                y_center = (box.y + box.height / 2.0) / h
                norm_w = box.width / w
                norm_h = box.height / h
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        # Log history trace
        log_path = os.path.join(feedback_dir, "feedback_history.log")
        with open(log_path, "a") as f:
            f.write(f"[{datetime.datetime.now().isoformat()}] Image: {image_name} | FP: {fp_box.class_name.upper()} ({fp_box.confidence*100:.0f}%) | ImageSaved: {image_path}\n")

        return base_filename

    @pyqtSlot(str)
    def _on_camera_error(self, message: str) -> None:
        logger.error("Camera error signal received: %s", message)
        self.status_footer.set_camera_error(message)
        if self.camera_thread.isRunning():
            self.camera_thread.quit()
            if not self.camera_thread.wait(3000):
                self.camera_thread.terminate()
                self.camera_thread.wait()
        self._camera_thread_started = False

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

        # Display preprocessed frame returned from background thread
        if getattr(res, "preprocessed_frame", None) is not None:
            qimg = bgr_frame_to_qimage(res.preprocessed_frame)
            self.inspect_widget._qimage = qimg
            self.inspect_widget.update()

        # Dynamically reposition contrast & brightness sliders if auto-tuned
        if getattr(res, "auto_clahe", None) is not None and getattr(res, "auto_gamma", None) is not None:
            self.side_panel.update_auto_params_ui(res.auto_clahe, res.auto_gamma)

    @pyqtSlot(object)
    def _on_inference_params_changed(self, p) -> None:
        self._confidence_threshold = p.confidence_threshold

    @pyqtSlot(int)
    def _on_camera_selected(self, idx):
        if idx < 0:
            return
        logger.info("Switching to camera index %d", idx)

        # 1. Disconnect old worker signals BEFORE teardown to prevent dangling connections
        #    (Fix #7 — prevents double-firing into the dying worker during signal emission)
        try:
            self.side_panel.preprocessing_params_changed.disconnect(self.camera_worker.update_params)
        except (TypeError, RuntimeError):
            pass
        try:
            self.camera_worker.camera_error.disconnect(self._on_camera_error)
        except (TypeError, RuntimeError):
            pass

        # 2. Properly tear down existing worker and thread
        if self.camera_thread.isRunning():
            self.camera_worker.stop()
            self.camera_thread.quit()
            if not self.camera_thread.wait(2000):
                logger.warning("Camera thread timed out during switch; forcing termination.")
                self.camera_thread.terminate()

        # 3. Create new worker for the new index
        self.camera_worker = CameraWorker(device_index=idx)
        self.camera_worker.moveToThread(self.camera_thread)

        # 4. Re-wire core signals (mapping to the new worker instance)
        self.camera_thread.started.disconnect()  # Clear old connection
        self.camera_thread.started.connect(self.camera_worker.run)

        # In static image inspection mode, we do NOT connect camera streaming signals.
        # Captures are initiated on-demand by clicking "Capture", which dynamically
        # connects self._on_capture_frame to self.camera_worker.new_frame.
        self.camera_worker.camera_error.connect(self._on_camera_error)
        self.side_panel.preprocessing_params_changed.connect(self.camera_worker.update_params)
        # Push the current slider state into the new worker so it doesn't start
        # with default params regardless of where the UI sliders are positioned.
        self.side_panel._emit_preprocessing_params()

        # 5. Fire it up
        self._camera_thread_started = True
        self.camera_thread.start()
        self.status_footer.set_camera_active()

    @pyqtSlot(str)
    def _on_model_selected(self, model_path: str):
        if not os.path.isfile(model_path):
            self.status_footer.set_model_error("Model missing")
            return
        self.status_footer.set_model_loading()
        model_name = self.side_panel.model_combo.currentText()
        self.analytics_dashboard.set_model_name(model_name)
        if self._current_raw_frame is not None:
            self.inspect_widget.set_analyzing("Loading model engine…")
        # Thread-safe load model call to worker with 50ms delay to let UI paint loading state
        QTimer.singleShot(50, lambda: self.inference_worker.load_model(model_path))

    @pyqtSlot()
    def _show_onboarding_guide(self):
        dialog = HelpDialog(self)
        dialog.exec()

    @pyqtSlot()
    def _on_recommend_model_requested(self):
        # Fix #8: Resolve path from the module file location to handle both
        # development runs and PyInstaller frozen bundles correctly.
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        nano_path = os.path.join(project_root, "models", "yolov12_int8.xml")
        if not os.path.isfile(nano_path):
            self.status_footer.set_model_error("Prod model missing")
            return

        self.status_footer.set_status_text("Benchmarking hardware...")
        if self._current_raw_frame is not None:
            self.inspect_widget.set_analyzing("Benchmarking hardware…")
        QTimer.singleShot(50, lambda: self.inference_worker.run_benchmark(nano_path))

    @pyqtSlot(str, float)
    def _on_benchmark_completed(self, model_path: str, avg_latency_ms: float):
        self.status_footer.set_model_ready()
        
        if avg_latency_ms < 12.0:
            recommended_substring = "Medium (FP16)"
            tier_name = "High-End (GPU)"
        elif avg_latency_ms < 28.0:
            recommended_substring = "Small (FP16)"
            tier_name = "Mid-Range (GPU/CPU)"
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

    def _set_active_nav_button(self, active_btn: NavButton):
        for btn in [self.top_load_btn, self.top_capture_btn, self.top_reanalyze_btn, 
                    self.top_export_btn, self.top_help_btn, self.top_settings_btn]:
            is_active = btn == active_btn
            btn.set_active(is_active)

    @pyqtSlot(bool, str)
    def _on_side_panel_toggled(self, expanded: bool, key: str):
        if expanded:
            self._set_active_nav_button(self.top_settings_btn)
        else:
            self._set_active_nav_button(None)

    def _on_usb_timer_timeout(self): self._start_async_scan()

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt C++ API override
        self.camera_worker.stop()
        self.camera_thread.quit()
        self.camera_thread.wait(3000)
        self.inference_thread.quit()
        self.inference_thread.wait(3000)
        event.accept()