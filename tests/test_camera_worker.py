"""
tests/test_camera_worker.py
---------------------------
Unit tests for workers/camera_worker.py.

Testing strategy:
  CameraWorker.run() is an infinite while loop. We control termination by
  having the mocked cap.read() side_effect call worker.stop() after N frames.
  worker.run() is invoked DIRECTLY in the test thread (not in a QThread) —
  this is valid for unit testing because:
    1. pyqtSignal in the same thread uses direct connections, so connected
       Python lambdas are called synchronously.
    2. There is no thread boundary to cross in a single-thread test.
    3. This avoids race conditions and flakiness from real QThread timing.

  Execution model: mock → run() → signals fire → lists collect → assert.

  Tests that exercise the retry path patch time.sleep to keep them fast.

No PyQt6.QtWidgets import (validates workers/ boundary rule).
Real VideoCapture hardware is NOT required.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
from PyQt6.QtGui import QImage

from core.models import PreprocessParams
from workers.camera_worker import (
    CameraWorker,
)


# ===========================================================================
# Session-scoped QCoreApplication (needed for PyQt6 signal mechanics)
# ===========================================================================

# QApplication is provided by tests/conftest.py (session scope).
# No per-file fixture needed.


# ===========================================================================
# Helpers
# ===========================================================================


def make_flat_frame(value: int = 128, size: int = 64) -> np.ndarray:
    """Create a near-uniform (blurry) BGR frame.

    We add tiny gaussian noise (std=2) so the frame passes the dead-sensor
    guard (frame_std > 0.1) but still has very low Laplacian variance,
    causing it to be treated as a blurry/motion frame by the sharpness gate.
    """
    rng = np.random.default_rng(42)
    base = np.full((size, size, 3), value, dtype=np.float32)
    noisy = np.clip(base + rng.normal(0, 2, base.shape), 0, 255).astype(np.uint8)
    return noisy


def make_sharp_frame(size: int = 64) -> np.ndarray:
    """Create a high-contrast checkerboard BGR frame — high Laplacian variance."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if (x // 8 + y // 8) % 2 == 0:
                frame[y, x] = [255, 255, 255]
    return frame


def make_mock_cap(opened: bool = True) -> MagicMock:
    """Build a mock cv2.VideoCapture that reports isOpened() → opened."""
    cap = MagicMock()
    cap.isOpened.return_value = opened
    return cap


def make_stop_side_effect(worker: CameraWorker, n_frames: int, frame: np.ndarray):
    """
    Return a side_effect callable for cap.read() that:
      - Returns (True, frame.copy()) for the first n_frames calls
      - Calls worker.stop() and returns (False, None) thereafter.
    Prevents the infinite loop from running indefinitely in tests.
    """
    call_count = [0]

    def _read():
        call_count[0] += 1
        if call_count[0] > n_frames:
            worker.stop()
            return False, None
        return True, frame.copy()

    return _read


def make_default_params(**overrides) -> PreprocessParams:
    params = PreprocessParams()
    for k, v in overrides.items():
        setattr(params, k, v)
    return params


# ===========================================================================
# Initialisation & attribute tests
# ===========================================================================


class TestCameraWorkerInit:
    def test_default_device_index_is_zero(self):
        worker = CameraWorker()
        assert worker._device_index == 0

    def test_custom_device_index_stored(self):
        worker = CameraWorker(device_index=2)
        assert worker._device_index == 2

    def test_running_starts_false(self):
        worker = CameraWorker()
        assert worker._running is False

    def test_default_params_are_preprocess_params(self):
        worker = CameraWorker()
        assert isinstance(worker._params, PreprocessParams)

    def test_is_qobject_not_qthread(self):
        """Architecture rule: CameraWorker must be QObject, NOT QThread."""
        from PyQt6.QtCore import QThread, QObject

        worker = CameraWorker()
        assert isinstance(worker, QObject)
        assert not isinstance(worker, QThread)


# ===========================================================================
# Signal declarations
# ===========================================================================


class TestCameraWorkerSignals:
    def test_new_frame_signal_exists(self):
        worker = CameraWorker()
        assert hasattr(worker, "new_frame")

    def test_frame_ready_for_inference_signal_exists(self):
        worker = CameraWorker()
        assert hasattr(worker, "frame_ready_for_inference")

    def test_camera_error_signal_exists(self):
        worker = CameraWorker()
        assert hasattr(worker, "camera_error")

    def test_new_frame_can_be_connected(self):
        """Signals must be connectable without raising."""
        worker = CameraWorker()
        worker.new_frame.connect(lambda img: None)

    def test_frame_ready_for_inference_can_be_connected(self):
        worker = CameraWorker()
        worker.frame_ready_for_inference.connect(lambda arr: None)

    def test_camera_error_can_be_connected(self):
        worker = CameraWorker()
        worker.camera_error.connect(lambda msg: None)


# ===========================================================================
# update_params — thread-safe parameter slot
# ===========================================================================


class TestUpdateParams:
    def test_update_params_changes_stored_params(self):
        worker = CameraWorker()
        new_params = PreprocessParams(
            clahe_clip_limit=5.0, gamma=1.8, blur_threshold=200.0
        )
        worker.update_params(new_params)
        assert worker._params.clahe_clip_limit == 5.0
        assert worker._params.gamma == 1.8
        assert worker._params.blur_threshold == 200.0

    def test_update_params_replaces_entirely(self):
        worker = CameraWorker()
        worker.update_params(PreprocessParams(clahe_clip_limit=3.0))
        assert worker._params.clahe_clip_limit == 3.0
        # Other fields remain at their defaults
        assert worker._params.gamma == PreprocessParams().gamma

    def test_snapshot_params_returns_copy_not_reference(self):
        """
        _snapshot_params() must return a new object so mutations in run()
        don't race with update_params() calls from the GUI thread.
        """
        worker = CameraWorker()
        params = worker._params
        snapshot = worker._snapshot_params()
        assert snapshot is not params
        assert snapshot.clahe_clip_limit == params.clahe_clip_limit


# ===========================================================================
# stop()
# ===========================================================================


class TestStop:
    def test_stop_sets_running_false(self):
        worker = CameraWorker()
        worker._running = True
        worker.stop()
        assert worker._running is False

    def test_stop_is_idempotent(self):
        """Calling stop() twice must not raise."""
        worker = CameraWorker()
        worker.stop()
        worker.stop()
        assert worker._running is False


# ===========================================================================
# _wait_for_retry — non-blocking retry sleep
# ===========================================================================


class TestWaitForRetry:
    def test_exits_immediately_when_running_is_false(self):
        worker = CameraWorker()
        worker._running = False
        with patch("time.sleep") as mock_sleep:
            worker._wait_for_retry()
        # Should exit on first poll check without sleeping
        assert mock_sleep.call_count == 0

    def test_calls_sleep_in_poll_intervals(self):
        """With _running=True throughout, sleep should be called repeatedly."""
        worker = CameraWorker()
        worker._running = True
        poll_count = [0]

        def fake_sleep(duration):
            poll_count[0] += 1
            if poll_count[0] >= 3:
                worker._running = False  # Abort after 3 polls

        with patch("time.sleep", side_effect=fake_sleep):
            worker._wait_for_retry()

        assert poll_count[0] == 3

    def test_respects_running_false_mid_retry(self):
        """stop() during a retry should exit the wait within one poll interval."""
        worker = CameraWorker()
        worker._running = True
        poll_count = [0]

        def fake_sleep(duration):
            poll_count[0] += 1
            if poll_count[0] == 2:
                worker._running = False

        with patch("time.sleep", side_effect=fake_sleep):
            worker._wait_for_retry()

        # Must have exited after poll 2 set _running=False
        assert poll_count[0] == 2


# ===========================================================================
# run() — camera open failure path
# ===========================================================================


class TestRunCameraOpenFailure:
    def test_emits_camera_error_when_camera_fails_to_open(self):
        worker = CameraWorker(device_index=99)
        errors: list[str] = []
        worker.camera_error.connect(lambda msg: errors.append(msg))

        mock_cap = make_mock_cap(opened=False)
        mock_cap.release = MagicMock()

        with (
            patch("cv2.VideoCapture", return_value=mock_cap),
            patch("time.sleep", side_effect=lambda _: worker.stop()),
        ):
            worker.run()

        assert len(errors) >= 1
        assert "99" in errors[0] or "Cannot open" in errors[0]

    def test_camera_error_message_mentions_device_index(self):
        worker = CameraWorker(device_index=3)
        errors: list[str] = []
        worker.camera_error.connect(lambda msg: errors.append(msg))

        mock_cap = make_mock_cap(opened=False)
        with (
            patch("cv2.VideoCapture", return_value=mock_cap),
            patch("time.sleep", side_effect=lambda _: worker.stop()),
        ):
            worker.run()

        assert any("3" in e for e in errors)

    def test_cap_release_called_on_open_failure(self):
        worker = CameraWorker()
        mock_cap = make_mock_cap(opened=False)
        with (
            patch("cv2.VideoCapture", return_value=mock_cap),
            patch("time.sleep", side_effect=lambda _: worker.stop()),
        ):
            worker.run()

        mock_cap.release.assert_called()

    def test_uses_cap_dshow_backend(self):
        """CAP_DSHOW must be passed on every VideoCapture construction."""
        worker = CameraWorker(device_index=0)
        calls: list[tuple] = []

        def spy_cap(idx, backend):
            calls.append((idx, backend))
            cap = make_mock_cap(opened=False)

            def fake_release():
                worker.stop()

            cap.release = fake_release
            return cap

        with patch("cv2.VideoCapture", side_effect=spy_cap), patch("time.sleep"):
            worker.run()

        import cv2 as _cv2

        for _, backend in calls:
            assert backend == _cv2.CAP_DSHOW, (
                f"Expected CAP_DSHOW ({_cv2.CAP_DSHOW}), got {backend}"
            )


# ===========================================================================
# run() — camera read failure (disconnect) path (UJ-03)
# ===========================================================================


class TestRunReadFailure:
    def test_emits_camera_error_on_read_failure(self):
        worker = CameraWorker()
        errors: list[str] = []
        worker.camera_error.connect(lambda msg: errors.append(msg))

        mock_cap = make_mock_cap(opened=True)
        mock_cap.read.return_value = (False, None)

        with (
            patch("cv2.VideoCapture", return_value=mock_cap),
            patch("time.sleep", side_effect=lambda _: worker.stop()),
        ):
            worker.run()

        assert len(errors) >= 1

    def test_cap_released_on_read_failure(self):
        worker = CameraWorker()
        mock_cap = make_mock_cap(opened=True)
        mock_cap.read.return_value = (False, None)

        with (
            patch("cv2.VideoCapture", return_value=mock_cap),
            patch("time.sleep", side_effect=lambda _: worker.stop()),
        ):
            worker.run()

        mock_cap.release.assert_called()

    def test_no_new_frame_emitted_on_read_failure(self):
        worker = CameraWorker()
        frames: list[QImage] = []
        worker.new_frame.connect(lambda img: frames.append(img))

        mock_cap = make_mock_cap(opened=True)
        mock_cap.read.return_value = (False, None)

        with (
            patch("cv2.VideoCapture", return_value=mock_cap),
            patch("time.sleep", side_effect=lambda _: worker.stop()),
        ):
            worker.run()

        assert len(frames) == 0


# ===========================================================================
# run() — blurry frame path (variance < blur_threshold)
# ===========================================================================


class TestRunBlurryFrame:
    def test_blurry_frame_emits_new_frame(self):
        """Live feed must always be displayed, even for blurry/motion frames."""
        worker = CameraWorker()
        blurry = make_flat_frame()
        received: list[QImage] = []
        worker.new_frame.connect(lambda img: received.append(img))

        mock_cap = make_mock_cap(opened=True)
        mock_cap.read.side_effect = make_stop_side_effect(
            worker, n_frames=2, frame=blurry
        )

        with patch("cv2.VideoCapture", return_value=mock_cap):
            worker.run()

        assert len(received) >= 1
        assert all(isinstance(r, QImage) for r in received)

    def test_blurry_frame_does_not_emit_frame_ready_for_inference(self):
        """
        The inference signal must NOT be emitted for blurry frames —
        blurry frames produce useless detections and waste inference budget.
        """
        worker = CameraWorker()
        blurry = make_flat_frame()
        inference_frames: list = []
        worker.frame_ready_for_inference.connect(
            lambda arr: inference_frames.append(arr)
        )

        mock_cap = make_mock_cap(opened=True)
        mock_cap.read.side_effect = make_stop_side_effect(
            worker, n_frames=3, frame=blurry
        )

        with patch("cv2.VideoCapture", return_value=mock_cap):
            worker.run()

        assert len(inference_frames) == 0

    def test_blurry_frame_emits_raw_not_processed(self):
        """
        For blurry frames, the raw frame is converted to QImage directly —
        CLAHE and Gamma are not applied. Verified by checking pixel colour
        matches the input (a solid-green BGR frame should stay solid-green RGB).
        """
        worker = CameraWorker()
        # Solid green BGR: B=0, G=200, R=0 → in RGB: R=0, G=200, B=0
        green_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        green_bgr[:, :, 1] = 200  # G channel in BGR

        received: list[QImage] = []
        worker.new_frame.connect(lambda img: received.append(img))

        mock_cap = make_mock_cap(opened=True)
        mock_cap.read.side_effect = make_stop_side_effect(
            worker, n_frames=1, frame=green_bgr
        )

        with patch("cv2.VideoCapture", return_value=mock_cap):
            worker.run()

        assert len(received) >= 1
        # Green should remain green after BGR→RGB conversion
        colour = received[0].pixelColor(32, 32)
        assert colour.green() > 150  # Dominant channel should be green


# ===========================================================================
# run() — sharp frame path (full preprocessing + dual emit)
# ===========================================================================


class TestRunSharpFrame:
    def test_sharp_frame_emits_new_frame(self):
        worker = CameraWorker()
        sharp = make_sharp_frame()
        received: list[QImage] = []
        worker.new_frame.connect(lambda img: received.append(img))

        mock_cap = make_mock_cap(opened=True)
        mock_cap.read.side_effect = make_stop_side_effect(
            worker, n_frames=2, frame=sharp
        )

        with patch("cv2.VideoCapture", return_value=mock_cap):
            worker.run()

        assert len(received) >= 1

    def test_sharp_frame_emits_frame_ready_for_inference(self):
        """
        Sharp frames must be forwarded to InferenceWorker via
        frame_ready_for_inference signal (the core AI pipeline trigger).
        """
        worker = CameraWorker()
        sharp = make_sharp_frame(64)
        inference_arrays: list[np.ndarray] = []
        worker.frame_ready_for_inference.connect(
            lambda arr: inference_arrays.append(arr)
        )

        mock_cap = make_mock_cap(opened=True)
        mock_cap.read.side_effect = make_stop_side_effect(
            worker, n_frames=2, frame=sharp
        )

        with patch("cv2.VideoCapture", return_value=mock_cap):
            worker.run()

        assert len(inference_arrays) >= 1
        assert all(isinstance(a, np.ndarray) for a in inference_arrays)

    def test_sharp_frame_emitted_array_is_a_copy(self):
        """
        The numpy array emitted to InferenceWorker must be a copy —
        not a reference to the internal frame buffer. Verified by checking
        the emitted array is not the same object as the source sharp frame.
        """
        worker = CameraWorker()
        sharp = make_sharp_frame(64)
        inference_arrays: list[np.ndarray] = []
        worker.frame_ready_for_inference.connect(
            lambda arr: inference_arrays.append(arr)
        )

        mock_cap = make_mock_cap(opened=True)
        read_frames: list[np.ndarray] = []
        call_count = [0]

        def spy_read():
            call_count[0] += 1
            if call_count[0] > 1:
                worker.stop()
                return False, None
            read_frames.append(sharp.copy())
            return True, read_frames[-1]

        mock_cap.read.side_effect = spy_read

        with patch("cv2.VideoCapture", return_value=mock_cap):
            worker.run()

        assert len(inference_arrays) >= 1
        # The emitted array must NOT be the same object
        assert inference_arrays[0] is not read_frames[0]

    def test_sharp_frame_new_frame_emitted_before_inference(self):
        """
        new_frame must be emitted BEFORE frame_ready_for_inference on each
        sharp frame — this ensures the UI is never blocked by inference lag.
        """
        worker = CameraWorker()
        sharp = make_sharp_frame(64)
        emit_order: list[str] = []
        worker.new_frame.connect(lambda _: emit_order.append("display"))
        worker.frame_ready_for_inference.connect(
            lambda _: emit_order.append("inference")
        )

        mock_cap = make_mock_cap(opened=True)
        mock_cap.read.side_effect = make_stop_side_effect(
            worker, n_frames=1, frame=sharp
        )

        with patch("cv2.VideoCapture", return_value=mock_cap):
            worker.run()

        # Within each frame pair, display must always precede inference
        display_idx = emit_order.index("display")
        inference_idx = emit_order.index("inference")
        assert display_idx < inference_idx

    def test_sharp_frame_uses_preprocessing(self):
        """
        Verify that apply_clahe and apply_gamma are called on sharp frames
        (not on blurry frames). Uses mock to spy on the core functions.
        """
        worker = CameraWorker()
        sharp = make_sharp_frame(64)

        mock_cap = make_mock_cap(opened=True)
        mock_cap.read.side_effect = make_stop_side_effect(
            worker, n_frames=1, frame=sharp
        )

        with (
            patch("cv2.VideoCapture", return_value=mock_cap),
            patch(
                "workers.camera_worker.apply_clahe",
                wraps=__import__(
                    "core.preprocessor", fromlist=["apply_clahe"]
                ).apply_clahe,
            ) as mock_clahe,
            patch(
                "workers.camera_worker.apply_gamma",
                wraps=__import__(
                    "core.preprocessor", fromlist=["apply_gamma"]
                ).apply_gamma,
            ) as mock_gamma,
        ):
            worker.run()

        mock_clahe.assert_called_once()
        mock_gamma.assert_called_once()

    def test_blurry_frame_skips_preprocessing(self):
        """
        Verify that apply_clahe and apply_gamma are NOT called on blurry frames.
        """
        worker = CameraWorker()
        blurry = make_flat_frame()

        mock_cap = make_mock_cap(opened=True)
        mock_cap.read.side_effect = make_stop_side_effect(
            worker, n_frames=2, frame=blurry
        )

        with (
            patch("cv2.VideoCapture", return_value=mock_cap),
            patch("workers.camera_worker.apply_clahe") as mock_clahe,
            patch("workers.camera_worker.apply_gamma") as mock_gamma,
        ):
            worker.run()

        mock_clahe.assert_not_called()
        mock_gamma.assert_not_called()


# ===========================================================================
# run() — cleanup on exit
# ===========================================================================


class TestRunCleanup:
    def test_cap_released_on_clean_exit(self):
        """cap.release() must be called when run() exits normally."""
        worker = CameraWorker()
        sharp = make_sharp_frame()
        mock_cap = make_mock_cap(opened=True)
        mock_cap.read.side_effect = make_stop_side_effect(
            worker, n_frames=1, frame=sharp
        )

        with patch("cv2.VideoCapture", return_value=mock_cap):
            worker.run()

        mock_cap.release.assert_called()

    def test_running_is_true_during_run_and_false_after(self):
        """_running must be True while loop executes, False after return."""
        worker = CameraWorker()
        running_states: list[bool] = []

        sharp = make_sharp_frame()
        mock_cap = make_mock_cap(opened=True)

        def spy_read():
            running_states.append(worker._running)
            worker.stop()
            return True, sharp.copy()

        mock_cap.read.side_effect = spy_read

        assert worker._running is False  # Before run()
        with patch("cv2.VideoCapture", return_value=mock_cap):
            worker.run()

        assert True in running_states  # Was True during loop
        assert worker._running is False  # False after run() exits


# ===========================================================================
# Params-updated integration (motion gate threshold changes)
# ===========================================================================


class TestParamsIntegration:
    def test_blur_threshold_from_params_governs_gate(self):
        """
        A frame with non-zero variance should pass the gate when
        blur_threshold is set very low, but be dropped when set very high.
        """
        # Make a weakly-textured checkerboard that has SOME variance but not huge
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[::4, ::4] = [200, 200, 200]  # sparse pattern, low variance

        from core.preprocessor import compute_variance

        actual_variance = compute_variance(frame)

        # Frame should be sharp with very low threshold
        worker_pass = CameraWorker()
        worker_pass.update_params(PreprocessParams(blur_threshold=0.1))

        inference_received_pass: list = []
        worker_pass.frame_ready_for_inference.connect(
            lambda arr: inference_received_pass.append(arr)
        )

        mock_cap = make_mock_cap(opened=True)
        mock_cap.read.side_effect = make_stop_side_effect(
            worker_pass, n_frames=1, frame=frame
        )

        with patch("cv2.VideoCapture", return_value=mock_cap):
            worker_pass.run()

        if actual_variance >= 0.1:
            assert len(inference_received_pass) >= 1

        # Frame should be blurry with very high threshold
        worker_drop = CameraWorker()
        worker_drop.update_params(PreprocessParams(blur_threshold=1e9))

        inference_received_drop: list = []
        worker_drop.frame_ready_for_inference.connect(
            lambda arr: inference_received_drop.append(arr)
        )

        mock_cap2 = make_mock_cap(opened=True)
        mock_cap2.read.side_effect = make_stop_side_effect(
            worker_drop, n_frames=2, frame=frame
        )

        with patch("cv2.VideoCapture", return_value=mock_cap2):
            worker_drop.run()

        assert len(inference_received_drop) == 0
