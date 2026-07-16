import os
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
import cv2

from core.utils import bgr_frame_to_qimage, enumerate_cameras
from core.models import DetectionResult, BoundingBox
from ui.image_inspect_widget import ImageInspectWidget
from ui.analytics_dashboard import AnalyticsDashboard
from workers.inference_worker import InferenceWorker

def test_bgr_frame_to_qimage_invalid_input():
    # Null or empty frame
    with pytest.raises(ValueError, match="received an empty frame"):
        bgr_frame_to_qimage(None)
    
    with pytest.raises(ValueError, match="received an empty frame"):
        bgr_frame_to_qimage(np.zeros((0, 0, 3), dtype=np.uint8))
        
    # Invalid channel count
    with pytest.raises(ValueError, match="expects a \\(H, W, 3\\) BGR frame"):
        bgr_frame_to_qimage(np.zeros((100, 100), dtype=np.uint8))

    # Triggering cv2.error by invalid dtype
    with pytest.raises(ValueError, match="Color conversion failed"):
        bgr_frame_to_qimage(np.zeros((100, 100, 3), dtype=np.int32))

def test_enumerate_cameras_exception():
    # Verify enumerate_cameras handles cv2.VideoCapture exceptions gracefully
    with patch("cv2.VideoCapture") as mock_vc:
        mock_vc.side_effect = cv2.error("Probing failed")
        res = enumerate_cameras(max_index=0)
        assert len(res) == 0

def test_image_inspect_widget_load_path_exception(qapp):
    widget = ImageInspectWidget()
    # Loading non-existent path
    widget.load_image_from_path("non_existent_file_path.png")
    assert widget._state == widget._State.REJECTED
    assert "Failed to load image" in widget._guard_reason

def test_image_inspect_widget_load_array_exception(qapp):
    widget = ImageInspectWidget()
    # Loading invalid array
    widget.load_image_from_array(None)
    assert widget._state == widget._State.REJECTED
    assert "Failed to load capture" in widget._guard_reason

def test_image_inspect_widget_paint_event_fallback(qapp):
    widget = ImageInspectWidget()
    widget._state = widget._State.EMPTY
    
    # Mock _draw_drop_zone to raise ValueError
    with patch.object(widget, "_draw_drop_zone", side_effect=ValueError("Simulated Draw Error")):
        # Triggering paintEvent shouldn't crash the widget or program
        widget.repaint()
        # Fallback should be drawn or at least the function returns without crashing

def test_analytics_dashboard_metrics_fallback(qapp):
    dashboard = AnalyticsDashboard()
    
    # Send empty result or result missing expected attributes to trigger AttributeError/TypeError
    res = MagicMock()
    # We mock boxes to raise TypeError on len()
    type(res).boxes = property(lambda x: None)
    
    # Should handle gracefully, showing "Error" instead of crashing
    dashboard.handle_detections(res)
    assert dashboard.faults_val.text() == "Error"
    assert dashboard.conf_val.text() == "Error"

def test_analytics_dashboard_checklist_fallback(qapp):
    dashboard = AnalyticsDashboard()
    
    # Create result with valid boxes to trigger checklist repopulation
    box = BoundingBox(x=10, y=10, width=50, height=50, class_name="short", confidence=0.85)
    res = DetectionResult(boxes=[box])
    
    # Mock checklist repaint or checklist widget creation to raise an error
    with patch.object(dashboard, "_repopulate_checklist", side_effect=RuntimeError("Checklist repopulation error")):
        dashboard.handle_detections(res)
        
        # Verify fallback label was added
        found_err_lbl = False
        for i in range(dashboard.checklist_layout.count()):
            widget = dashboard.checklist_layout.itemAt(i).widget()
            if widget and "Checklist unavailable" in widget.text():
                found_err_lbl = True
                break
        assert found_err_lbl is True

def test_analytics_dashboard_advisory_fallback(qapp):
    dashboard = AnalyticsDashboard()
    box = BoundingBox(x=10, y=10, width=50, height=50, class_name="short", confidence=0.85)
    res = DetectionResult(boxes=[box])
    
    with patch.object(dashboard, "_update_advisory_ui", side_effect=RuntimeError("Advisory update error")):
        dashboard.handle_detections(res)
        
        assert dashboard.status_val.text() == "ADVISORY ERROR"
        assert "Rework advice is currently unavailable" in dashboard.status_desc.text()

def test_inference_worker_load_model_exception(qapp):
    worker = InferenceWorker()
    
    # Set up signal tracker
    errors = []
    worker.inference_error.connect(errors.append)
    
    # Load non-existent model
    worker.load_model("invalid_model_path.xml")
    
    assert len(errors) == 1
    assert "Model load failed" in errors[0]
