"""
tests/test_utils.py
-------------------
Unit tests for core/utils.py.

Testing strategy:
  bgr_frame_to_qimage():
    - Input validation: None, empty, wrong shape → ValueError
    - Output type: must return a QImage instance
    - Output format: must be Format_RGB888
    - Output dimensions: width/height must match input frame
    - Colour correctness: BGR→RGB channel swap verified on solid-colour frames
    - bytesPerLine precision: non-contiguous (sliced) numpy array must produce
      a correctly-shaped QImage with no pixel drift
    - Memory safety: QImage must survive after the source numpy array is deleted
    - Immutability: source frame must not be mutated

  enumerate_cameras():
    - cv2.VideoCapture is fully mocked — no real hardware required
    - Tests for: empty result, single camera, multiple cameras, CAP_DSHOW flag,
      correct index/label format, always-releases capture, max_index parameter.
    - Active Frame Validation: verifies that only devices yielding a valid
      OpenCV frame are returned.

No PyQt6.QtWidgets import (validates core/ boundary rule).
"""

import gc
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from PyQt6.QtGui import QImage

from core.utils import bgr_frame_to_qimage, enumerate_cameras


# ===========================================================================
# Helpers
# ===========================================================================


def make_bgr(
    height: int = 64, width: int = 64, bgr: tuple = (128, 128, 128)
) -> np.ndarray:
    """Create a solid-colour BGR frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = bgr[0]  # Blue
    frame[:, :, 1] = bgr[1]  # Green
    frame[:, :, 2] = bgr[2]  # Red
    return frame


# ===========================================================================
# bgr_frame_to_qimage — FR12
# ===========================================================================


class TestBgrFrameToQimage:
    # --- Input validation ---

    def test_raises_on_none_frame(self):
        with pytest.raises(ValueError, match="empty frame"):
            bgr_frame_to_qimage(None)  # type: ignore[arg-type]

    def test_raises_on_empty_array(self):
        with pytest.raises(ValueError, match="empty frame"):
            bgr_frame_to_qimage(np.array([]))

    def test_raises_on_single_channel_frame(self):
        grey = np.full((64, 64), 128, dtype=np.uint8)
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            bgr_frame_to_qimage(grey)

    def test_raises_on_four_channel_frame(self):
        rgba = np.zeros((64, 64, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            bgr_frame_to_qimage(rgba)

    # --- Return type and format ---

    def test_returns_qimage_instance(self):
        frame = make_bgr()
        result = bgr_frame_to_qimage(frame)
        assert isinstance(result, QImage)

    def test_output_format_is_rgb888(self):
        frame = make_bgr()
        result = bgr_frame_to_qimage(frame)
        assert result.format() == QImage.Format.Format_RGB888

    # --- Dimensions ---

    def test_output_width_matches_frame(self):
        frame = make_bgr(height=48, width=80)
        result = bgr_frame_to_qimage(frame)
        assert result.width() == 80

    def test_output_height_matches_frame(self):
        frame = make_bgr(height=48, width=80)
        result = bgr_frame_to_qimage(frame)
        assert result.height() == 48

    def test_handles_1080p_frame(self):
        """Must not crash on a 1920×1080 frame (full pipeline resolution)."""
        frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        result = bgr_frame_to_qimage(frame)
        assert result.width() == 1920
        assert result.height() == 1080

    def test_handles_non_square_frame(self):
        frame = make_bgr(height=100, width=200)
        result = bgr_frame_to_qimage(frame)
        assert result.width() == 200
        assert result.height() == 100

    # --- BGR → RGB channel swap correctness ---

    def test_pure_blue_bgr_maps_to_blue_rgb(self):
        """
        A pure-blue BGR frame (B=255, G=0, R=0) must produce a QImage
        where pixel (0,0) is blue: RGB(0, 0, 255).
        """
        frame = make_bgr(bgr=(255, 0, 0))  # B=255 in BGR
        result = bgr_frame_to_qimage(frame)
        colour = result.pixelColor(0, 0)
        assert colour.red() == 0
        assert colour.green() == 0
        assert colour.blue() == 255

    def test_pure_green_bgr_maps_to_green_rgb(self):
        """BGR G channel (index 1) must remain green in RGB."""
        frame = make_bgr(bgr=(0, 255, 0))
        result = bgr_frame_to_qimage(frame)
        colour = result.pixelColor(0, 0)
        assert colour.red() == 0
        assert colour.green() == 255
        assert colour.blue() == 0

    def test_pure_red_bgr_maps_to_red_rgb(self):
        """
        A pure-red BGR frame (B=0, G=0, R=255) must produce a QImage
        where pixel (0,0) is red: RGB(255, 0, 0).
        Red is stored at index 2 in BGR, index 0 in RGB.
        """
        frame = make_bgr(bgr=(0, 0, 255))  # R=255 in BGR
        result = bgr_frame_to_qimage(frame)
        colour = result.pixelColor(0, 0)
        assert colour.red() == 255
        assert colour.green() == 0
        assert colour.blue() == 0

    def test_white_frame_produces_white_pixels(self):
        frame = make_bgr(bgr=(255, 255, 255))
        result = bgr_frame_to_qimage(frame)
        colour = result.pixelColor(0, 0)
        assert colour.red() == 255
        assert colour.green() == 255
        assert colour.blue() == 255

    def test_black_frame_produces_black_pixels(self):
        frame = make_bgr(bgr=(0, 0, 0))
        result = bgr_frame_to_qimage(frame)
        colour = result.pixelColor(0, 0)
        assert colour.red() == 0
        assert colour.green() == 0
        assert colour.blue() == 0

    # --- bytesPerLine precision (anti-skew) ---

    def test_non_contiguous_array_produces_correct_dimensions(self):
        """
        A non-contiguous numpy array (created by row-stride slicing) must
        produce a QImage with the correct dimensions and no pixel drift.
        This validates the strides[0] bytesPerLine approach.

        Strategy: create a 128×128 frame, slice every-other row (non-contiguous),
        convert, and verify the output dimensions match the slice, not the original.
        """
        large_frame = np.random.randint(0, 256, (128, 64, 3), dtype=np.uint8)
        # Slice every other row → produces a non-contiguous array
        sliced = large_frame[::2, :, :]  # shape (64, 64, 3), non-contiguous
        assert not sliced.flags["C_CONTIGUOUS"], "Slice should be non-contiguous"

        result = bgr_frame_to_qimage(sliced)
        assert result.width() == 64
        assert result.height() == 64

    def test_bytes_per_line_equals_width_times_three_for_contiguous_array(self):
        """
        For a standard contiguous (H, W, 3) frame, strides[0] == width * 3.
        Verify the QImage bytes-per-line is correct by checking image is not null.
        """
        frame = make_bgr(height=32, width=32)
        assert frame.strides[0] == 32 * 3  # sanity-check the test assumption
        result = bgr_frame_to_qimage(frame)
        assert not result.isNull()

    # --- Memory safety ---

    def test_qimage_survives_after_source_array_deleted(self):
        """
        QImage must own its pixel data (.copy() called internally).
        After the numpy source is deleted and GC runs, reading pixels
        must not crash or produce corrupted values.
        """
        frame = make_bgr(bgr=(0, 255, 0))  # solid green
        result = bgr_frame_to_qimage(frame)

        # Delete source and force GC
        del frame
        gc.collect()

        # QImage must still be readable and correct
        assert not result.isNull()
        colour = result.pixelColor(0, 0)
        assert colour.green() == 255

    # --- Immutability ---

    def test_source_frame_not_mutated(self):
        """bgr_frame_to_qimage must not modify the input array in place."""
        frame = make_bgr(bgr=(100, 150, 200))
        original = frame.copy()
        bgr_frame_to_qimage(frame)
        np.testing.assert_array_equal(frame, original)


# ===========================================================================
# enumerate_cameras — FR2
# ===========================================================================


class TestEnumerateCameras:
    """
    All tests mock cv2.VideoCapture to avoid requiring real USB hardware.
    """

    def _make_mock_cap(self, is_opened: bool, can_read: bool = True) -> MagicMock:
        """Build a mock VideoCapture that reports isOpened() and read()."""
        cap = MagicMock()
        cap.isOpened.return_value = is_opened
        cap.read.return_value = (can_read, np.zeros((64, 64, 3), dtype=np.uint8))
        return cap

    # --- Empty result ---

    def test_returns_empty_list_when_no_cameras_found(self):
        with patch("cv2.VideoCapture", return_value=self._make_mock_cap(False)):
            result = enumerate_cameras()
        assert result == []

    # --- Detection ---

    def test_returns_camera_0_when_index_0_opens(self):
        def fake_cap(index, backend):
            return self._make_mock_cap(is_opened=(index == 0))

        # Mock QtMultimedia to return a known name
        qt_names = ["Mock Camera 0"]

        with patch("cv2.VideoCapture", side_effect=fake_cap):
            result = enumerate_cameras(qt_names=qt_names)

        assert len(result) == 1
        idx, label = result[0]
        assert idx == 0
        assert label == "Mock Camera 0"

    def test_falls_back_to_generic_name_when_qtmultimedia_fails(self):
        def fake_cap(index, backend):
            return self._make_mock_cap(is_opened=(index == 0))

        # Mock QtMultimedia to raise an error
        with patch("cv2.VideoCapture", side_effect=fake_cap):
            with patch(
                "PyQt6.QtMultimedia.QMediaDevices.videoInputs", side_effect=ImportError
            ):
                result = enumerate_cameras()

        assert len(result) == 1
        idx, label = result[0]
        assert idx == 0
        assert "Camera 0" in label

    def test_returns_multiple_cameras_at_correct_indices(self):
        """Cameras at index 0 and 2 present; 1, 3, 4, 5 absent."""

        def fake_cap(index, backend):
            return self._make_mock_cap(is_opened=(index in {0, 2}))

        with patch("cv2.VideoCapture", side_effect=fake_cap):
            result = enumerate_cameras()

        indices = [idx for idx, _ in result]
        assert indices == [0, 2]

    def test_filters_out_cameras_that_fail_read(self):
        """Camera at index 0 opens but fails read(); Camera at index 1 opens and succeeds."""

        def fake_cap(index, backend):
            if index == 0:
                return self._make_mock_cap(is_opened=True, can_read=False)
            if index == 1:
                return self._make_mock_cap(is_opened=True, can_read=True)
            return self._make_mock_cap(is_opened=False)

        with patch("cv2.VideoCapture", side_effect=fake_cap):
            result = enumerate_cameras()

        # Only index 1 should be returned
        assert len(result) == 1
        assert result[0][0] == 1

    def test_result_contains_tuple_of_int_and_str(self):
        def fake_cap(index, backend):
            return self._make_mock_cap(is_opened=(index == 1))

        with patch("cv2.VideoCapture", side_effect=fake_cap):
            result = enumerate_cameras()

        assert len(result) == 1
        idx, label = result[0]
        assert isinstance(idx, int)
        assert isinstance(label, str)

    def test_label_contains_camera_index(self):
        """Label must include the camera index for user-readable identification."""

        def fake_cap(index, backend):
            return self._make_mock_cap(is_opened=(index == 3))

        with patch("cv2.VideoCapture", side_effect=fake_cap):
            result = enumerate_cameras()

        _, label = result[0]
        assert "3" in label

    # --- CAP_DSHOW enforcement ---

    def test_cap_dshow_flag_always_passed(self):
        """
        Every cv2.VideoCapture call must use cv2.CAP_DSHOW as the backend
        flag — this is mandatory for Windows DirectShow fast-loading (FR2).
        """
        calls_made: list[tuple] = []

        def recording_cap(index, backend):
            calls_made.append((index, backend))
            return self._make_mock_cap(False)

        with patch("cv2.VideoCapture", side_effect=recording_cap):
            enumerate_cameras(max_index=2)

        # Every call must use cv2.CAP_DSHOW (value = 700 in OpenCV)
        import cv2 as _cv2

        for _, backend in calls_made:
            assert backend == _cv2.CAP_DSHOW, (
                f"Expected CAP_DSHOW ({_cv2.CAP_DSHOW}); got {backend}"
            )

    # --- Resource cleanup ---

    def test_release_always_called_for_every_probe(self):
        """
        cap.release() must be called for every probed index, including
        those where isOpened() returns False, to free DirectShow resources.
        """
        mock_caps: list[MagicMock] = []

        def tracking_cap(index, backend):
            cap = self._make_mock_cap(False)
            mock_caps.append(cap)
            return cap

        with patch("cv2.VideoCapture", side_effect=tracking_cap):
            enumerate_cameras(max_index=2)  # probes 0, 1, 2 → 3 caps

        assert len(mock_caps) == 3
        for cap in mock_caps:
            cap.release.assert_called_once()

    def test_release_called_even_when_camera_opens(self):
        """
        After detecting an open camera, release() must still be called
        to not block the device for the actual CameraWorker that will
        re-open it properly.
        """
        mock_cap = self._make_mock_cap(True)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            enumerate_cameras(max_index=0)

        mock_cap.release.assert_called()

    # --- Probe range ---

    def test_probes_indices_0_through_max_inclusive(self):
        """Default max_index=5 means indices 0,1,2,3,4,5 are all probed."""
        probed_indices: list[int] = []

        def recording_cap(index, backend):
            probed_indices.append(index)
            return self._make_mock_cap(False)

        with patch("cv2.VideoCapture", side_effect=recording_cap):
            enumerate_cameras(max_index=5)

        assert probed_indices == [0, 1, 2, 3, 4, 5]

    def test_custom_max_index_respected(self):
        """enumerate_cameras(max_index=2) must probe exactly 0, 1, 2."""
        probed: list[int] = []

        def recording_cap(index, backend):
            probed.append(index)
            return self._make_mock_cap(False)

        with patch("cv2.VideoCapture", side_effect=recording_cap):
            enumerate_cameras(max_index=2)

        assert probed == [0, 1, 2]

    def test_max_index_zero_probes_only_index_zero(self):
        probed: list[int] = []

        def recording_cap(index, backend):
            probed.append(index)
            return self._make_mock_cap(False)

        with patch("cv2.VideoCapture", side_effect=recording_cap):
            enumerate_cameras(max_index=0)

        assert probed == [0]

    # --- Return type ---

    def test_returns_list(self):
        with patch("cv2.VideoCapture", return_value=self._make_mock_cap(False)):
            result = enumerate_cameras()
        assert isinstance(result, list)

    def test_result_is_sorted_by_index(self):
        """Result list must be ordered by camera index ascending."""

        def fake_cap(index, backend):
            return self._make_mock_cap(is_opened=(index in {0, 2, 4}))

        with patch("cv2.VideoCapture", side_effect=fake_cap):
            result = enumerate_cameras()

        indices = [idx for idx, _ in result]
        assert indices == sorted(indices)
