"""
tests/test_ui_main_window.py
-----------------------------
Unit tests for ui/main_window.py.

Testing strategy:
  MainWindow owns the signal wiring, layout assembly, and thread lifecycle
  but delegates all heavyweight work to workers. Tests verify:

  1. Layout structure — all expected child widgets present and correctly typed.
  2. Signal wiring — UI slots respond correctly to mocked worker signals.
  3. FR15 routing — _on_new_detections drives WarningBanner correctly.
  4. Confidence threshold tracking — ControlPanel slider change updates
     _confidence_threshold used for FR15.
  5. closeEvent safety — stop()/quit()/wait() called in correct order.
  6. StatusFooter state transitions — camera/model status updates land correctly.

Threading strategy:
  Tests use MainWindow with threads DISABLED. CameraWorker and InferenceWorker
  are replaced with MagicMock instances so:
    - cv2.VideoCapture is never opened (would block/fail without hardware)
    - InferenceEngine is never loaded (requires a real .xml model file)
    - All signals can be emitted manually via worker.signal.emit() in tests

  The fixture replaces self.camera_worker and self.inference_worker with mocks
  BEFORE _connect_signals() runs, by subclassing MainWindow and overriding
  _create_workers().

QApplication provided by tests/conftest.py (session scope).
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, patch, call

import pytest
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QMainWindow

from core.models import BoundingBox, DetectionResult, InferenceParams, PreprocessParams
from ui.control_panel import ControlPanel
from ui.main_window import MainWindow, _DEFAULT_CONFIDENCE_THRESHOLD
from ui.status_footer import StatusFooter
from ui.video_widget import VideoWidget
from ui.warning_banner import WarningBanner
from ui.theme import (
    WINDOW_MIN_WIDTH,
    WINDOW_MIN_HEIGHT,
    WINDOW_DEFAULT_WIDTH,
    WINDOW_DEFAULT_HEIGHT,
)


# ===========================================================================
# Stub workers (no cv2, no OpenVINO, no threading)
# ===========================================================================

class _StubCameraWorker(QObject):
    """Minimal QObject with the same signal/slot interface as CameraWorker."""
    new_frame = pyqtSignal(QImage)
    frame_ready_for_inference = pyqtSignal(object)
    camera_error = pyqtSignal(str)

    def __init__(self, device_index: int = 0, parent=None):
        super().__init__(parent)
        self._device_index = device_index
        self._running = False
        self.stop_called = False

    def stop(self):
        self.stop_called = True
        self._running = False

    def run(self):
        self._running = True

    def update_params(self, params):
        pass


class _StubInferenceWorker(QObject):
    """Minimal QObject with the same signal/slot interface as InferenceWorker."""
    new_detections = pyqtSignal(object)
    inference_error = pyqtSignal(str)
    model_loaded = pyqtSignal()

    def process_frame(self, frame): pass
    def update_params(self, params): pass
    def load_model(self, path): pass


class _HeadlessMainWindow(MainWindow):
    """
    MainWindow subclass that replaces real workers with stubs.

    Overrides _create_workers() to install stub workers and stub threads.
    Overrides _start_workers() and _post_start_init() to prevent actual
    thread.start() and camera enumeration (would call cv2).
    """

    def _create_workers(self) -> None:
        """Install stub workers with stub threads (never start())."""
        self.camera_thread = MagicMock(spec=QThread)
        self.camera_thread.wait.return_value = True   # Simulate clean exit
        self.camera_worker = _StubCameraWorker(device_index=0)
        self.camera_worker.moveToThread = MagicMock()   # Prevent actual move

        self.inference_thread = MagicMock(spec=QThread)
        self.inference_thread.wait.return_value = True
        self.inference_worker = _StubInferenceWorker()
        self.inference_worker.moveToThread = MagicMock()

    def _start_workers(self) -> None:
        """Void — no real threads started in tests."""
        pass

    def _post_start_init(self) -> None:
        """Void — prevents hardware scanner launch and QMetaObject.invokeMethod."""
        pass


@pytest.fixture()
def win() -> _HeadlessMainWindow:
    """Fresh headless MainWindow for each test."""
    return _HeadlessMainWindow(model_path="models/fake.xml")


# ===========================================================================
# Layout structure
# ===========================================================================

class TestMainWindowLayout:

    def test_is_qmainwindow(self, win):
        assert isinstance(win, QMainWindow)

    def test_has_video_widget(self, win):
        assert isinstance(win.video_widget, VideoWidget)

    def test_has_warning_banner(self, win):
        assert isinstance(win.warning_banner, WarningBanner)

    def test_warning_banner_starts_hidden(self, win):
        assert win.warning_banner.isHidden()

    def test_has_status_footer(self, win):
        assert isinstance(win.status_footer, StatusFooter)

    def test_has_control_panel(self, win):
        assert isinstance(win.control_panel, ControlPanel)

    def test_minimum_width(self, win):
        assert win.minimumWidth() == WINDOW_MIN_WIDTH

    def test_minimum_height(self, win):
        assert win.minimumHeight() == WINDOW_MIN_HEIGHT

    def test_default_confidence_threshold(self, win):
        """FR15: initial threshold must match _DEFAULT_CONFIDENCE_THRESHOLD."""
        assert abs(win._confidence_threshold - _DEFAULT_CONFIDENCE_THRESHOLD) < 1e-9

    def test_window_title_contains_circa(self, win):
        assert "CIRCA" in win.windowTitle()


# ===========================================================================
# Signal routing — CameraWorker → UI
# ===========================================================================

class TestCameraWorkerRouting:

    def test_camera_error_calls_status_footer(self, win):
        """camera_error signal → StatusFooter shows red dot."""
        win.camera_worker.camera_error.emit("USB unplugged")
        assert win.status_footer._camera_indicator.dot_color != "#4CAF50"  # Not green

    def test_camera_error_updates_video_widget_status_text(self, win):
        """camera_error -> VideoWidget shows 'camera unavailable' status text."""
        win.camera_worker.camera_error.emit("Device gone")
        assert win.video_widget._status_text == "camera unavailable"

    def test_new_frame_reaches_video_widget(self, win):
        """new_frame signal → VideoWidget stores a frame with matching dimensions."""
        from PyQt6.QtGui import QColor
        img = QImage(64, 64, QImage.Format.Format_RGB888)
        img.fill(QColor("#FF0000"))
        win.camera_worker.new_frame.emit(img)
        # Check that a frame was stored (not None) with the expected dimensions.
        # Object identity is not guaranteed across QueuedConnection delivery.
        assert win.video_widget._current_frame is not None
        assert win.video_widget._current_frame.width() == 64
        assert win.video_widget._current_frame.height() == 64


# ===========================================================================
# Signal routing — InferenceWorker → UI
# ===========================================================================

class TestInferenceWorkerRouting:

    def test_model_loaded_sets_model_ready_dot(self, win):
        """model_loaded signal → StatusFooter shows green model dot."""
        from ui.theme import COLOR_STATUS_OK
        win.inference_worker.model_loaded.emit()
        assert win.status_footer._model_indicator.dot_color == COLOR_STATUS_OK

    def test_inference_error_sets_model_error_dot(self, win):
        """inference_error signal → StatusFooter shows red model dot."""
        from ui.theme import COLOR_STATUS_ERROR
        win.inference_worker.inference_error.emit("OpenVINO crash")
        assert win.status_footer._model_indicator.dot_color == COLOR_STATUS_ERROR

    def test_new_detections_reaches_video_widget(self, win):
        """new_detections signal → VideoWidget stores the DetectionResult."""
        result = DetectionResult(boxes=[
            BoundingBox(x=10, y=10, width=50, height=50,
                        class_name="solder_bridge", confidence=0.9)
        ])
        win.inference_worker.new_detections.emit(result)
        assert win.video_widget._detections is result

    def test_new_detections_updates_detection_count(self, win):
        """new_detections → StatusFooter detection count updated."""
        from ui.theme import COLOR_STATUS_OK
        result = DetectionResult(boxes=[])
        win.inference_worker.new_detections.emit(result)
        # 0 detections → green dot
        assert win.status_footer._detection_indicator.dot_color == COLOR_STATUS_OK

    def test_new_detections_with_boxes_updates_count(self, win):
        """2 detections → StatusFooter shows '2 defects'."""
        result = DetectionResult(boxes=[
            BoundingBox(x=0, y=0, width=10, height=10,
                        class_name="burnt_area", confidence=0.8),
            BoundingBox(x=0, y=0, width=10, height=10,
                        class_name="solder_bridge", confidence=0.7),
        ])
        win.inference_worker.new_detections.emit(result)
        assert "2" in win.status_footer._detection_indicator.label_text


# ===========================================================================
# FR15 routing — _on_new_detections → WarningBanner
# ===========================================================================

class TestFR15Routing:

    def _make_result(self, avg_conf: float) -> DetectionResult:
        """Make a DetectionResult whose average_confidence equals avg_conf."""
        # average_confidence is computed as mean of box confidences.
        # Single box with target confidence gives exact average.
        return DetectionResult(boxes=[
            BoundingBox(x=0, y=0, width=10, height=10,
                        class_name="solder_bridge", confidence=avg_conf)
        ])

    def test_low_confidence_shows_warning_banner(self, win):
        """avg_confidence < threshold → WarningBanner visible."""
        win._confidence_threshold = 0.50
        result = self._make_result(0.30)
        win.inference_worker.new_detections.emit(result)
        assert not win.warning_banner.isHidden()

    def test_high_confidence_does_not_show_banner_immediately(self, win):
        """avg_confidence >= threshold → banner stays hidden (no cool-down elapsed)."""
        win._confidence_threshold = 0.50
        result = self._make_result(0.90)
        win.inference_worker.new_detections.emit(result)
        assert win.warning_banner.isHidden()

    def test_banner_auto_resets_after_cool_down(self, win):
        """30 consecutive high-confidence frames → banner auto-resets."""
        from ui.warning_banner import COOL_DOWN_FRAMES
        win._confidence_threshold = 0.50
        # First: show the banner
        win._on_new_detections(self._make_result(0.20))
        assert not win.warning_banner.isHidden()
        # Then: drive COOL_DOWN_FRAMES good frames through the slot
        for _ in range(COOL_DOWN_FRAMES):
            win._on_new_detections(self._make_result(0.90))
        assert win.warning_banner.isHidden()

    def test_zero_detections_clean_board_does_not_trigger_banner(self, win):
        """
        Clean board (no boxes) → DetectionResult.average_confidence == 1.0 (sentinel).
        This must NOT trigger the FR15 warning banner.
        """
        win._confidence_threshold = 0.50
        clean = DetectionResult(boxes=[])   # average_confidence == 1.0
        win._on_new_detections(clean)
        assert win.warning_banner.isHidden()

    def test_manual_dismiss_suppresses_banner(self, win):
        """Dismissed banner stays hidden even on subsequent low-confidence frames."""
        win._confidence_threshold = 0.50
        win._on_new_detections(self._make_result(0.20))
        assert not win.warning_banner.isHidden()
        win.warning_banner._dismiss_btn.click()
        win._on_new_detections(self._make_result(0.20))
        assert win.warning_banner.isHidden()


# ===========================================================================
# Confidence threshold tracking
# ===========================================================================

class TestConfidenceThresholdTracking:

    def test_initial_threshold_is_default(self, win):
        assert abs(win._confidence_threshold - _DEFAULT_CONFIDENCE_THRESHOLD) < 1e-9

    def test_inference_params_changed_updates_threshold(self, win):
        """ControlPanel confidence slider changes → _confidence_threshold updated."""
        new_params = InferenceParams(confidence_threshold=0.75)
        win._on_inference_params_changed(new_params)
        assert abs(win._confidence_threshold - 0.75) < 1e-9

    def test_threshold_change_affects_next_fr15_evaluation(self, win):
        """After threshold update, low-confidence is evaluated against new value."""
        # With threshold=0.80, confidence=0.70 is LOW
        win._on_inference_params_changed(InferenceParams(confidence_threshold=0.80))
        result = DetectionResult(boxes=[
            BoundingBox(x=0, y=0, width=10, height=10,
                        class_name="solder_bridge", confidence=0.70)
        ])
        win._on_new_detections(result)
        assert not win.warning_banner.isHidden()

    def test_threshold_raised_below_current_confidence(self, win):
        """Threshold lowered below current confidence → banner should not show."""
        # confidence = 0.70, threshold = 0.50 → good frame
        win._on_inference_params_changed(InferenceParams(confidence_threshold=0.50))
        result = DetectionResult(boxes=[
            BoundingBox(x=0, y=0, width=10, height=10,
                        class_name="solder_bridge", confidence=0.70)
        ])
        win._on_new_detections(result)
        # Good frame — banner stays hidden, counter increments
        assert win.warning_banner.isHidden()


# ===========================================================================
# closeEvent safety
# ===========================================================================

class TestCloseEvent:

    def test_close_event_calls_camera_worker_stop(self, win):
        """stop() MUST be called on CameraWorker before thread.quit()."""
        from PyQt6.QtGui import QCloseEvent
        event = QCloseEvent()
        win.closeEvent(event)
        assert win.camera_worker.stop_called is True

    def test_close_event_calls_camera_thread_quit(self, win):
        from PyQt6.QtGui import QCloseEvent
        event = QCloseEvent()
        win.closeEvent(event)
        win.camera_thread.quit.assert_called_once()

    def test_close_event_calls_camera_thread_wait(self, win):
        from PyQt6.QtGui import QCloseEvent
        event = QCloseEvent()
        win.closeEvent(event)
        win.camera_thread.wait.assert_called()

    def test_close_event_calls_inference_thread_quit(self, win):
        from PyQt6.QtGui import QCloseEvent
        event = QCloseEvent()
        win.closeEvent(event)
        win.inference_thread.quit.assert_called_once()

    def test_close_event_calls_inference_thread_wait(self, win):
        from PyQt6.QtGui import QCloseEvent
        event = QCloseEvent()
        win.closeEvent(event)
        win.inference_thread.wait.assert_called()

    def test_close_event_accepts_event(self, win):
        """closeEvent must call event.accept() — not event.ignore()."""
        from PyQt6.QtGui import QCloseEvent
        event = QCloseEvent()
        win.closeEvent(event)
        assert event.isAccepted()

    def test_stop_called_before_quit(self, win):
        """
        Critical ordering: camera_worker.stop() MUST precede camera_thread.quit().
        Without stop(), CameraWorker.run() never exits and wait() blocks forever.
        Verified by tracking call insertion order on a combined mock.
        """
        from PyQt6.QtGui import QCloseEvent
        from unittest.mock import MagicMock, call

        call_log: list[str] = []
        win.camera_worker.stop = lambda: call_log.append("stop")
        win.camera_thread.quit = MagicMock(side_effect=lambda: call_log.append("quit"))
        win.camera_thread.wait = MagicMock(return_value=True)

        event = QCloseEvent()
        win.closeEvent(event)

        assert call_log.index("stop") < call_log.index("quit"), (
            "camera_worker.stop() must be called BEFORE camera_thread.quit()"
        )


# ===========================================================================
# Default values and state
# ===========================================================================

class TestMainWindowDefaults:

    def test_camera_worker_created(self, win):
        assert isinstance(win.camera_worker, _StubCameraWorker)

    def test_inference_worker_created(self, win):
        assert isinstance(win.inference_worker, _StubInferenceWorker)

    def test_video_widget_initial_status_has_text(self, win):
        """VideoWidget starts with a non-empty status string."""
        assert len(win.video_widget._status_text) > 0

    def test_control_panel_clahe_default(self, win):
        """CLAHE slider default = 2.0 (UX spec)."""
        assert abs(win.control_panel.clahe_slider.value() - 2.0) < 0.05

    def test_control_panel_confidence_default(self, win):
        """Confidence slider default = 50% (UX spec)."""
        assert abs(win.control_panel.confidence_slider.value() - 50.0) < 0.5


# ===========================================================================
# ControlPanel pinned toggle button (Change 1)
# ===========================================================================

class TestControlPanelPinnedToggle:
    """Verify QHBoxLayout design: toggle button permanently on left edge."""

    def test_toggle_button_text_less_than_when_expanded(self, win):
        """Expanded state uses ASCII '<' not fancy '\u2039'."""
        assert win.control_panel._toggle_btn.text() == "<"

    def test_toggle_button_text_greater_than_when_collapsed(self, win):
        """Collapsed state uses ASCII '>' not fancy '\u203a'."""
        win.control_panel.toggle()
        assert win.control_panel._toggle_btn.text() == ">"

    def test_toggle_button_has_fixed_width_matching_collapsed_constant(self, win):
        """Button is set to CONTROL_PANEL_WIDTH_COLLAPSED via setFixedWidth().

        PyQt6 has no fixedWidth() getter — setFixedWidth(n) sets both
        minimumWidth and maximumWidth to n, so we verify via those.
        """
        from ui.theme import CONTROL_PANEL_WIDTH_COLLAPSED
        btn = win.control_panel._toggle_btn
        # setFixedWidth(28) sets both min and max to 28
        assert btn.minimumWidth() == CONTROL_PANEL_WIDTH_COLLAPSED
        assert btn.maximumWidth() == CONTROL_PANEL_WIDTH_COLLAPSED

    def test_toggle_button_stylesheet_font_weight_900(self, win):
        """QSS must include '900' for maximum glyph boldness."""
        qss = win.control_panel._toggle_btn.styleSheet()
        assert "900" in qss

    def test_toggle_button_stylesheet_font_size_16px(self, win):
        """QSS must specify font-size: 16px."""
        qss = win.control_panel._toggle_btn.styleSheet()
        assert "16px" in qss

    def test_toggle_button_stylesheet_border_none(self, win):
        """QSS must strip the default QPushButton frame."""
        qss = win.control_panel._toggle_btn.styleSheet()
        assert "border: none" in qss


# ===========================================================================
# USB hotplug via nativeEvent + QTimer debounce (Change 2)
# ===========================================================================

class TestUsbHotplug:
    """Verify WM_DEVICECHANGE intercept and debounce QTimer."""

    def test_usb_debounce_timer_exists_and_is_single_shot(self, win):
        from PyQt6.QtCore import QTimer
        assert hasattr(win, "_usb_debounce_timer")
        assert isinstance(win._usb_debounce_timer, QTimer)
        assert win._usb_debounce_timer.isSingleShot()

    def test_on_usb_timer_timeout_launches_scan(self, win):
        with patch.object(win, "_start_async_scan") as mock_scan:
            win._on_usb_timer_timeout()
        mock_scan.assert_called_once_with(is_startup=False)

    def test_on_cameras_found_hotplug_updates_dropdown(self, win):
        from unittest.mock import patch
        from ui.theme import COLOR_STATUS_OK
        # Mock currently empty dropdown
        win.control_panel.camera_combo.clear()
        
        new_cameras = [(0, "New Camera")]
        win._on_cameras_found_hotplug(new_cameras)
        
        assert win.control_panel.camera_combo.count() == 1
        assert win.control_panel.camera_combo.itemText(0) == "New Camera"
        assert win.status_footer._camera_indicator.dot_color == COLOR_STATUS_OK

    def test_on_cameras_found_hotplug_with_no_cameras_sets_placeholder(self, win):
        win._on_cameras_found_hotplug([])
        assert win.video_widget._status_text == "Please connect a camera"

    def test_on_cameras_found_hotplug_with_cameras_sets_connecting_text(self, win):
        # Start from "No camera found" state
        win.video_widget.set_status_text("Please connect a camera")
        
        win._on_cameras_found_hotplug([(0, "USB Cam")])
        assert "Connecting" in win.video_widget._status_text

    def test_native_event_non_windows_ignored(self, win):
        win._usb_debounce_timer.stop()
        win.nativeEvent(b"xcb_generic_event_t", None)
        assert not win._usb_debounce_timer.isActive()

    def test_native_event_non_devicechange_windows_msg_ignored(self, win):
        import ctypes, ctypes.wintypes as wt
        class _MSG(ctypes.Structure):
            _fields_ = [("hWnd", wt.HWND), ("message", wt.UINT), ("wParam", wt.WPARAM),
                        ("lParam", wt.LPARAM), ("time", wt.DWORD), ("pt", wt.POINT)]
        win._usb_debounce_timer.stop()
        msg = _MSG()
        msg.message = 0x0111   # WM_COMMAND — not WM_DEVICECHANGE
        win.nativeEvent(b"windows_generic_MSG", ctypes.addressof(msg))
        assert not win._usb_debounce_timer.isActive()

    def test_native_event_devicechange_starts_timer(self, win):
        """WM_DEVICECHANGE (0x0219) \u2192 timer becomes active."""
        import ctypes, ctypes.wintypes as wt
        class _MSG(ctypes.Structure):
            _fields_ = [("hWnd", wt.HWND), ("message", wt.UINT), ("wParam", wt.WPARAM),
                        ("lParam", wt.LPARAM), ("time", wt.DWORD), ("pt", wt.POINT)]
        win._usb_debounce_timer.stop()
        msg = _MSG()
        msg.message = 0x0219
        win.nativeEvent(b"windows_generic_MSG", ctypes.addressof(msg))
        assert win._usb_debounce_timer.isActive()

    def test_native_event_burst_coalesces_to_500ms(self, win):
        """5 consecutive WM_DEVICECHANGE events \u2192 single 500ms timer window."""
        import ctypes, ctypes.wintypes as wt
        class _MSG(ctypes.Structure):
            _fields_ = [("hWnd", wt.HWND), ("message", wt.UINT), ("wParam", wt.WPARAM),
                        ("lParam", wt.LPARAM), ("time", wt.DWORD), ("pt", wt.POINT)]
        win._usb_debounce_timer.stop()
        msg = _MSG()
        msg.message = 0x0219
        for _ in range(5):
            win.nativeEvent(b"windows_generic_MSG", ctypes.addressof(msg))
        assert win._usb_debounce_timer.isActive()
        assert win._usb_debounce_timer.interval() == 500


# ===========================================================================
# VideoWidget status text wiring (Change 3)
# ===========================================================================

class TestVideoWidgetStatusTextWiring:
    """Verify MainWindow drives set_status_text() at correct lifecycle moments."""

    def test_camera_error_status_is_unavailable(self, win):
        win._on_camera_error("USB dropout")
        assert win.video_widget._status_text == "camera unavailable"

    def test_usb_removal_no_cameras_shows_placeholder_text(self, win):
        win._on_cameras_found_hotplug([])
        assert win.video_widget._status_text == "Please connect a camera"

    def test_usb_insert_with_cameras_shows_connecting_text(self, win):
        # Force a "No camera" state first
        win.video_widget.set_status_text("Please connect a camera")
        
        win._on_cameras_found_hotplug([(0, "USB Cam")])
        assert "Connecting" in win.video_widget._status_text

    def test_clear_frame_and_set_status_text_independent(self, win):
        """clear_frame + set_status_text together produce correct idle state."""
        from PyQt6.QtGui import QImage, QColor
        img = QImage(4, 4, QImage.Format.Format_RGB888)
        img.fill(QColor("#00FF00"))
        win.video_widget.set_frame(img)
        win.video_widget.clear_frame()
        win.video_widget.set_status_text("Please connect a camera")
        assert win.video_widget._current_frame is None
        assert win.video_widget._status_text == "Please connect a camera"
