import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from core.utils import bgr_frame_to_qimage, enumerate_cameras

def test_enumerate_cameras_collision_avoidance():
    # If the camera index is equal to the active_device_index, it should bypass verification and be added.
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.random.randint(120, 130, (100, 100, 3), dtype=np.uint8))
    
    import cv2
    with patch("cv2.VideoCapture", return_value=mock_cap) as mock_vc:
        res = enumerate_cameras(max_index=2, active_device_index=1)
        # Bypassed index 1 verification; cv2.VideoCapture should not have been called for index 1
        mock_vc.assert_any_call(0, cv2.CAP_DSHOW)
        mock_vc.assert_any_call(2, cv2.CAP_DSHOW)
        # Verify index 1 is in the returned list
        indices = [idx for idx, label in res]
        assert 1 in indices

def test_enumerate_cameras_denylist_real_names():
    # Mock VideoCapture to return valid frames for indices 0, 1, 2
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.random.randint(120, 130, (100, 100, 3), dtype=np.uint8))
    
    with patch("cv2.VideoCapture", return_value=mock_cap):
        # Index 0: real name "Integrated Camera" (trusted, matches std/mean checks)
        # Index 1: real name "Camera 1" (denylisted)
        # Index 2: generic name (no name provided in qt_names, fallback "Camera 2 (USB)" created)
        qt_names = ["Integrated Camera", "Camera 1", None]
        
        res = enumerate_cameras(max_index=2, qt_names=qt_names)
        
        # Verify denylisted "Camera 1" is skipped
        # Fallback "Camera 2 (USB)" has "Camera 2" which doesn't match denylist, so it is returned
        labels = [label for idx, label in res]
        assert "Integrated Camera" in labels
        assert "Camera 1" not in labels
        assert "Camera 2 (USB)" in labels

def test_bgr_frame_to_qimage_contiguous_conversion():
    # Create non-contiguous BGR frame
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_slice = frame[::2, ::2]  # non-contiguous
    
    # Should convert cleanly without exception, making it contiguous under the hood
    qimg = bgr_frame_to_qimage(frame_slice)
    assert not qimg.isNull()
    assert qimg.width() == 50
    assert qimg.height() == 50
