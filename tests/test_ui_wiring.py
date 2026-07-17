import os
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from ui.main_window import MainWindow
from core.models import DetectionResult, BoundingBox

@pytest.fixture
def main_window(qapp):
    # Patch async scan to prevent real USB camera checking during initialization
    with patch("ui.main_window.MainWindow._start_async_scan"):
        win = MainWindow(model_path="models/yolov12_int8.xml")
        yield win
        # Clean up threads
        win.camera_thread.quit()
        win.camera_thread.wait()
        win.inference_thread.quit()
        win.inference_thread.wait()

def test_main_window_init(main_window):
    assert "CIRCA" in main_window.windowTitle()
    assert "Defect Detection" in main_window.windowTitle()
    assert main_window.inspect_widget is not None
    assert main_window.analytics_dashboard is not None
    assert main_window.side_panel is not None
    assert main_window.status_footer is not None

def test_main_window_image_loaded(main_window):
    """Verify that loading a valid PCB image triggers cross-thread inference dispatch.

    After Fix #4, inference is dispatched via _request_inference signal (not a direct
    process_frame call). We assert the signal was emitted with the correct arguments
    rather than patching the downstream slot, which is the correct architectural contract.
    """
    dummy_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    emitted_args = []

    # Spy on _request_inference signal emissions
    main_window._request_inference.connect(lambda frame, params: emitted_args.append((frame, params)))

    with patch("ui.main_window.is_likely_pcb", return_value=(True, "")):
        main_window.inspect_widget.image_loaded.emit(dummy_bgr)
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

    assert len(emitted_args) == 1, "Expected _request_inference to be emitted exactly once"
    emitted_frame, emitted_params = emitted_args[0]
    assert isinstance(emitted_frame, np.ndarray)
    assert emitted_frame.shape == (100, 100, 3)

def test_main_window_false_positive_logging(main_window, tmp_path):
    # Setup dummy loaded image and detections
    dummy_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
    box1 = BoundingBox(x=10, y=10, width=50, height=50, class_name="short", confidence=0.90)
    box2 = BoundingBox(x=80, y=80, width=40, height=40, class_name="missing_hole", confidence=0.80)
    res = DetectionResult(boxes=[box1, box2])
    
    main_window._current_raw_frame = dummy_bgr
    main_window.inspect_widget._detections = res
    main_window.analytics_dashboard._active_boxes = res.boxes
    main_window.analytics_dashboard._image_name = "test_board"
    
    feedback_dir = tmp_path / "datasets" / "feedback_fine_tune"
    images_dir = feedback_dir / "images"
    labels_dir = feedback_dir / "labels"
    
    orig_join = os.path.join
    with patch("os.path.join", side_effect=lambda *args: str(tmp_path / orig_join(*args) if args[0] == "datasets" else orig_join(*args))):
        # Trigger FP mark on first box (index 0 - short)
        main_window._on_mark_false_positive(0)
        
        # Verify box was popped from active detections
        assert len(main_window.inspect_widget._detections.boxes) == 1
        assert main_window.inspect_widget._detections.boxes[0].class_name == "missing_hole"
        
        # Verify serialization files were created
        assert os.path.isdir(images_dir)
        assert os.path.isdir(labels_dir)
        
        image_files = os.listdir(images_dir)
        label_files = os.listdir(labels_dir)
        
        assert len(image_files) == 1
        assert len(label_files) == 1
        assert image_files[0].endswith(".png")
        assert label_files[0].endswith(".txt")


def test_navigation_sidebar_and_settings_state(main_window):
    # Verify right side settings panel starts collapsed
    assert not main_window.side_panel._expanded
    assert main_window.side_panel.maximumWidth() == 0

    # Verify left navigation sidebar has static width 80px
    assert main_window.nav_sidebar is not None
    assert main_window.nav_sidebar.width() == 80 or main_window.nav_sidebar.maximumWidth() == 80

    # Verify custom NavButton properties
    assert main_window.logo_label.text() == "CIRCA"
    assert main_window.top_load_btn.text_label.text() == "Load Board"
    assert main_window.top_capture_btn.text_label.text() == "Capture"

    # Verify active highlighting works
    main_window._set_active_nav_button(main_window.top_load_btn)
    assert main_window.top_load_btn._active
    assert not main_window.top_capture_btn._active


def test_auto_optimise_sliders_behavior(main_window):
    # Retrieve widgets
    side = main_window.side_panel
    assert side.auto_opt_check.isChecked()

    # 1. By default, auto-optimise is true, so sliders should be disabled
    assert not side.clahe_slider.isEnabled()
    assert not side.gamma_slider.isEnabled()
    assert not side.clahe_slider._slider.isEnabled()
    assert not side.gamma_slider._slider.isEnabled()

    # 2. Toggle Auto-Optimise to False (Manual override state)
    side.auto_opt_check.setChecked(False)
    assert not side.auto_opt_check.isChecked()
    assert side.clahe_slider.isEnabled()
    assert side.gamma_slider.isEnabled()
    assert side.clahe_slider._slider.isEnabled()
    assert side.gamma_slider._slider.isEnabled()

    # 3. Toggle Auto-Optimise back to True
    side.auto_opt_check.setChecked(True)
    assert side.auto_opt_check.isChecked()
    assert not side.clahe_slider.isEnabled()
    assert not side.gamma_slider.isEnabled()
    assert not side.clahe_slider._slider.isEnabled()
    assert not side.gamma_slider._slider.isEnabled()




