"""
tests/test_ui_wiring.py
-----------------------
Unit tests for MainWindow signal/slot wiring.

Focus:
  - CameraWorker -> UI routing
  - InferenceWorker -> UI routing
  - FR15 Low Confidence routing
  - Parameter synchronization
"""

from __future__ import annotations

import pytest
from PyQt6.QtGui import QImage, QColor

from core.models import BoundingBox, DetectionResult, InferenceParams
from ui.main_window import _DEFAULT_CONFIDENCE_THRESHOLD
# Import the shared fixture
from tests.ui_test_utils import win

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
        img = QImage(64, 64, QImage.Format.Format_RGB888)
        img.fill(QColor("#FF0000"))
        win.camera_worker.new_frame.emit(img)
        assert win.video_widget._current_frame is not None
        assert win.video_widget._current_frame.width() == 64
        assert win.video_widget._current_frame.height() == 64


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
        result = DetectionResult(
            boxes=[
                BoundingBox(
                    x=10, y=10, width=50, height=50,
                    class_name="solder_bridge", confidence=0.9,
                )
            ]
        )
        win.inference_worker.new_detections.emit(result)
        assert win.video_widget._detections is result

    def test_new_detections_updates_detection_count(self, win):
        """0 detections → green dot."""
        from ui.theme import COLOR_STATUS_OK
        result = DetectionResult(boxes=[])
        win.inference_worker.new_detections.emit(result)
        assert win.status_footer._detection_indicator.dot_color == COLOR_STATUS_OK

    def test_new_detections_with_boxes_updates_count(self, win):
        """2 detections → StatusFooter shows '2 defects'."""
        result = DetectionResult(
            boxes=[
                BoundingBox(x=0, y=0, width=10, height=10, class_name="burnt_area", confidence=0.8),
                BoundingBox(x=0, y=0, width=10, height=10, class_name="solder_bridge", confidence=0.7),
            ]
        )
        win.inference_worker.new_detections.emit(result)
        assert "2" in win.status_footer._detection_indicator.label_text


class TestFR15Routing:
    def _make_result(self, avg_conf: float) -> DetectionResult:
        return DetectionResult(
            boxes=[
                BoundingBox(x=0, y=0, width=10, height=10, class_name="solder_bridge", confidence=avg_conf)
            ]
        )

    def test_low_confidence_shows_warning_banner(self, win):
        win._confidence_threshold = 0.50
        result = self._make_result(0.30)
        win.inference_worker.new_detections.emit(result)
        assert not win.warning_banner.isHidden()

    def test_high_confidence_does_not_show_banner_immediately(self, win):
        win._confidence_threshold = 0.50
        result = self._make_result(0.90)
        win.inference_worker.new_detections.emit(result)
        assert win.warning_banner.isHidden()

    def test_banner_auto_resets_after_cool_down(self, win):
        from ui.warning_banner import COOL_DOWN_FRAMES
        win._confidence_threshold = 0.50
        win._on_new_detections(self._make_result(0.20))
        assert not win.warning_banner.isHidden()
        for _ in range(COOL_DOWN_FRAMES):
            win._on_new_detections(self._make_result(0.90))
        assert win.warning_banner.isHidden()

    def test_zero_detections_clean_board_does_not_trigger_banner(self, win):
        win._confidence_threshold = 0.50
        clean = DetectionResult(boxes=[])
        win._on_new_detections(clean)
        assert win.warning_banner.isHidden()

    def test_manual_dismiss_suppresses_banner(self, win):
        win._confidence_threshold = 0.50
        win._on_new_detections(self._make_result(0.20))
        assert not win.warning_banner.isHidden()
        win.warning_banner._dismiss_btn.click()
        win._on_new_detections(self._make_result(0.20))
        assert win.warning_banner.isHidden()


class TestConfidenceThresholdTracking:
    def test_initial_threshold_is_default(self, win):
        assert abs(win._confidence_threshold - _DEFAULT_CONFIDENCE_THRESHOLD) < 1e-9

    def test_inference_params_changed_updates_threshold(self, win):
        new_params = InferenceParams(confidence_threshold=0.75)
        win._on_inference_params_changed(new_params)
        assert abs(win._confidence_threshold - 0.75) < 1e-9

    def test_threshold_change_affects_next_fr15_evaluation(self, win):
        win._on_inference_params_changed(InferenceParams(confidence_threshold=0.80))
        result = DetectionResult(
            boxes=[
                BoundingBox(x=0, y=0, width=10, height=10, class_name="solder_bridge", confidence=0.70)
            ]
        )
        win._on_new_detections(result)
        assert not win.warning_banner.isHidden()


class TestVideoWidgetStatusTextWiring:
    def test_camera_error_status_is_unavailable(self, win):
        win._on_camera_error("USB dropout")
        assert win.video_widget._status_text == "camera unavailable"

    def test_usb_removal_no_cameras_shows_placeholder_text(self, win):
        win._on_cameras_found_hotplug([])
        assert win.video_widget._status_text == "Please connect a camera"

    def test_usb_insert_with_cameras_shows_connecting_text(self, win):
        win.video_widget.set_status_text("Please connect a camera")
        win._on_cameras_found_hotplug([(0, "USB Cam")])
        # _on_cameras_found_hotplug sets "Connecting…" before calling _on_camera_selected
        assert "Connecting" in win.video_widget._status_text
