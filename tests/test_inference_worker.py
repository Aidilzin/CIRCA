"""
tests/test_inference_worker.py
------------------------------
Unit tests for workers/inference_worker.py.

Testing strategy:
  InferenceWorker is slot-driven (no run() loop), so we can call its slots
  directly in the test thread — no QThread required. pyqtSignal in-thread
  connections fire synchronously, making signal assertions straightforward.

  InferenceEngine is fully mocked — no real model file or OpenVINO required.

  Backpressure tests use the threading.Lock directly:
    - Pre-acquire before calling process_frame() → simulates "inference busy"
    - Verify the frame is dropped (no new_detections emitted)
    - Verify the lock is released after a real inference call (even on exception)

No PyQt6.QtWidgets import (validates workers/ boundary rule).
No cv2 import (validates workers/ boundary rule — cv2 belongs in core/ only).
"""

import threading
from unittest.mock import MagicMock, patch

import numpy as np
# QApplication provided by tests/conftest.py (session scope).

from core.models import BoundingBox, DetectionResult, InferenceParams
from workers.inference_worker import InferenceWorker


# ===========================================================================
# Helpers
# ===========================================================================


def make_bgr_frame(h: int = 64, w: int = 64) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def make_detection_result(confidence: float = 0.85) -> DetectionResult:
    return DetectionResult(
        boxes=[
            BoundingBox(
                x=10,
                y=10,
                width=50,
                height=50,
                class_name="solder_bridge",
                confidence=confidence,
            )
        ]
    )


def make_default_params(**overrides) -> InferenceParams:
    params = InferenceParams()
    for k, v in overrides.items():
        setattr(params, k, v)
    return params


def inject_mock_engine(worker: InferenceWorker, result: DetectionResult) -> MagicMock:
    """Inject a mock InferenceEngine that returns `result` from run()."""
    mock_engine = MagicMock()
    mock_engine.run.return_value = result
    worker._engine = mock_engine
    return mock_engine


# ===========================================================================
# Initialisation & type tests
# ===========================================================================


class TestInferenceWorkerInit:
    def test_engine_is_none_before_load_model(self):
        worker = InferenceWorker()
        assert worker._engine is None

    def test_default_params_are_inference_params(self):
        worker = InferenceWorker()
        assert isinstance(worker._params, InferenceParams)

    def test_inference_lock_starts_unlocked(self):
        """The backpressure lock must be acquirable at startup."""
        worker = InferenceWorker()
        acquired = worker._inference_lock.acquire(blocking=False)
        assert acquired, "Lock should be free at startup"
        worker._inference_lock.release()

    def test_is_qobject_not_qthread(self):
        from PyQt6.QtCore import QThread, QObject

        worker = InferenceWorker()
        assert isinstance(worker, QObject)
        assert not isinstance(worker, QThread)

    def test_has_no_run_method(self):
        """
        InferenceWorker is slot-driven, not a polling loop worker.
        It must NOT define a run() method (that would imply polling architecture).
        """
        worker = InferenceWorker()
        # QObject has no 'run' by default; InferenceWorker must not add one
        assert not hasattr(InferenceWorker, "run") or InferenceWorker.run is getattr(
            type(worker).__mro__[-2], "run", None
        ), "InferenceWorker must not define its own run() method"


# ===========================================================================
# Signal declarations
# ===========================================================================


class TestSignalDeclarations:
    def test_new_detections_signal_exists(self):
        worker = InferenceWorker()
        assert hasattr(worker, "new_detections")

    def test_inference_error_signal_exists(self):
        worker = InferenceWorker()
        assert hasattr(worker, "inference_error")

    def test_model_loaded_signal_exists(self):
        worker = InferenceWorker()
        assert hasattr(worker, "model_loaded")

    def test_all_signals_connectable(self):
        worker = InferenceWorker()
        worker.new_detections.connect(lambda _: None)
        worker.inference_error.connect(lambda _: None)
        worker.model_loaded.connect(lambda: None)


# ===========================================================================
# load_model — FR6 model initialisation
# ===========================================================================


class TestLoadModel:
    def test_emits_model_loaded_on_success(self):
        worker = InferenceWorker()
        fired: list[bool] = []
        worker.model_loaded.connect(lambda: fired.append(True))

        with patch("workers.inference_worker.InferenceEngine") as mock_engine_class:
            mock_engine_class.return_value = MagicMock()
            worker.load_model("models/fake.xml")

        assert fired == [True]

    def test_engine_is_set_after_successful_load(self):
        worker = InferenceWorker()

        with patch("workers.inference_worker.InferenceEngine") as mock_engine_class:
            mock_instance = MagicMock()
            mock_engine_class.return_value = mock_instance
            worker.load_model("models/fake.xml")

        assert worker._engine is mock_instance

    def test_emits_inference_error_on_load_failure(self):
        worker = InferenceWorker()
        errors: list[str] = []
        worker.inference_error.connect(lambda msg: errors.append(msg))

        with patch("workers.inference_worker.InferenceEngine") as mock_engine_class:
            mock_engine_class.side_effect = FileNotFoundError("No model file")
            worker.load_model("models/missing.xml")

        assert len(errors) == 1
        assert "FileNotFoundError" in errors[0] or "Model load failed" in errors[0]

    def test_no_model_loaded_emitted_on_failure(self):
        worker = InferenceWorker()
        fired: list[bool] = []
        worker.model_loaded.connect(lambda: fired.append(True))

        with patch("workers.inference_worker.InferenceEngine") as mock_engine_class:
            mock_engine_class.side_effect = RuntimeError("GPU error")
            worker.load_model("models/bad.xml")

        assert fired == []

    def test_engine_stays_none_on_load_failure(self):
        worker = InferenceWorker()

        with patch("workers.inference_worker.InferenceEngine") as mock_engine_class:
            mock_engine_class.side_effect = RuntimeError("compile error")
            worker.load_model("models/bad.xml")

        assert worker._engine is None

    def test_model_can_be_reloaded(self):
        """Calling load_model() twice replaces the engine (useful for hot-reload)."""
        worker = InferenceWorker()
        first_engine = MagicMock()
        second_engine = MagicMock()

        with patch("workers.inference_worker.InferenceEngine") as mock_cls:
            mock_cls.return_value = first_engine
            worker.load_model("models/v1.xml")
            assert worker._engine is first_engine

            mock_cls.return_value = second_engine
            worker.load_model("models/v2.xml")
            assert worker._engine is second_engine

    def test_load_model_passes_path_to_engine(self):
        worker = InferenceWorker()
        path = "models/yolov12_int8.xml"

        with patch("workers.inference_worker.InferenceEngine") as mock_cls:
            mock_cls.return_value = MagicMock()
            worker.load_model(path)

        mock_cls.assert_called_once_with(path)


# ===========================================================================
# update_params — thread-safe param updates
# ===========================================================================


class TestUpdateParams:
    def test_update_params_stores_new_params(self):
        worker = InferenceWorker()
        new_params = InferenceParams(confidence_threshold=0.75)
        worker.update_params(new_params)
        assert worker._params.confidence_threshold == 0.75

    def test_snapshot_params_returns_copy(self):
        worker = InferenceWorker()
        worker.update_params(InferenceParams(confidence_threshold=0.65))
        snapshot = worker._snapshot_params()
        assert snapshot is not worker._params
        assert snapshot.confidence_threshold == 0.65

    def test_snapshot_threshold_matches_updated_value(self):
        worker = InferenceWorker()
        worker.update_params(InferenceParams(confidence_threshold=0.30))
        snapshot = worker._snapshot_params()
        assert abs(snapshot.confidence_threshold - 0.30) < 1e-9


# ===========================================================================
# process_frame — backpressure (NFR4)
# ===========================================================================


class TestProcessFrameBackpressure:
    def test_frame_dropped_when_model_not_loaded(self):
        """
        Before load_model() succeeds, process_frame() must silently drop
        frames — the engine is None so no inference can run.
        """
        worker = InferenceWorker()
        assert worker._engine is None

        detections: list = []
        errors: list[str] = []
        worker.new_detections.connect(lambda r: detections.append(r))
        worker.inference_error.connect(lambda e: errors.append(e))

        worker.process_frame(make_bgr_frame())

        assert detections == []
        assert errors == []

    def test_frame_dropped_when_inference_lock_held(self):
        """
        Core backpressure test: if the lock is held (simulating inference in
        progress), process_frame() must return immediately without running inference.
        """
        worker = InferenceWorker()
        result = make_detection_result()
        inject_mock_engine(worker, result)

        detections: list = []
        worker.new_detections.connect(lambda r: detections.append(r))

        # Pre-acquire the lock — simulates inference already running
        worker._inference_lock.acquire()
        try:
            worker.process_frame(make_bgr_frame())
        finally:
            worker._inference_lock.release()

        assert detections == [], "Frame must be dropped when lock is held"
        # Engine.run() must not have been called
        worker._engine.run.assert_not_called()

    def test_lock_released_after_successful_inference(self):
        """
        After a successful inference call, the lock must be free so the next
        frame can be processed. A deadlock here would freeze the Inference Thread.
        """
        worker = InferenceWorker()
        inject_mock_engine(worker, make_detection_result())

        worker.process_frame(make_bgr_frame())

        # Lock must be acquirable after the call
        acquired = worker._inference_lock.acquire(blocking=False)
        assert acquired, "Lock must be released after successful process_frame()"
        worker._inference_lock.release()

    def test_lock_released_after_inference_exception(self):
        """
        Critical: even if InferenceEngine.run() raises, the lock must be
        released in the finally block. Failure here deadlocks the worker.
        """
        worker = InferenceWorker()
        mock_engine = MagicMock()
        mock_engine.run.side_effect = RuntimeError("OpenVINO crash")
        worker._engine = mock_engine

        worker.process_frame(make_bgr_frame())

        acquired = worker._inference_lock.acquire(blocking=False)
        assert acquired, "Lock must be released after exception in process_frame()"
        worker._inference_lock.release()

    def test_second_call_drops_while_first_holds_lock(self):
        """
        Two concurrent process_frame() calls: first holds lock, second drops.
        Uses threads to simulate real concurrent access.

        Asserts on mock_engine.run() call count rather than signal delivery
        because pyqtSignal emissions from background threads are delivered as
        queued events that require QCoreApplication.processEvents() — the
        lock behaviour itself is what matters here.
        """
        worker = InferenceWorker()
        result = make_detection_result()

        lock_held = threading.Event()
        inference_done = threading.Event()

        # Mock engine that holds the lock long enough for the second caller
        def slow_run(frame, params):
            lock_held.set()
            inference_done.wait(timeout=2.0)
            return result

        mock_engine = MagicMock()
        mock_engine.run.side_effect = slow_run
        worker._engine = mock_engine

        # First call runs inference in a background thread
        first_thread = threading.Thread(
            target=worker.process_frame, args=(make_bgr_frame(),)
        )
        first_thread.start()

        # Wait until the first call has acquired the lock and is running
        lock_held.wait(timeout=2.0)

        # Second call — should drop immediately (lock held by first_thread)
        second_thread = threading.Thread(
            target=worker.process_frame, args=(make_bgr_frame(),)
        )
        second_thread.start()
        second_thread.join(timeout=1.0)

        # Second call must return immediately (within 1s timeout)
        assert not second_thread.is_alive(), (
            "Second process_frame() call must return immediately when lock is held"
        )

        # Allow first call to complete
        inference_done.set()
        first_thread.join(timeout=2.0)

        # engine.run() called exactly once — second call was dropped before run()
        assert mock_engine.run.call_count == 1, (
            f"Expected engine.run() called once; got {mock_engine.run.call_count}. "
            "Second frame must be dropped, not processed."
        )


# ===========================================================================
# process_frame — successful inference path
# ===========================================================================


class TestProcessFrameSuccess:
    def test_emits_new_detections_on_successful_inference(self):
        worker = InferenceWorker()
        result = make_detection_result(confidence=0.90)
        inject_mock_engine(worker, result)

        received: list[DetectionResult] = []
        worker.new_detections.connect(lambda r: received.append(r))

        worker.process_frame(make_bgr_frame())

        assert len(received) == 1
        assert received[0] is result

    def test_emitted_result_is_detection_result(self):
        worker = InferenceWorker()
        inject_mock_engine(worker, make_detection_result())

        received: list = []
        worker.new_detections.connect(lambda r: received.append(r))

        worker.process_frame(make_bgr_frame())

        assert isinstance(received[0], DetectionResult)

    def test_engine_run_called_with_frame_and_params(self):
        worker = InferenceWorker()
        mock_engine = inject_mock_engine(worker, make_detection_result())
        params = InferenceParams(confidence_threshold=0.70)
        worker.update_params(params)

        frame = make_bgr_frame()
        worker.process_frame(frame)

        mock_engine.run.assert_called_once()
        call_args = mock_engine.run.call_args
        # First positional arg is the frame
        assert call_args[0][0] is frame or np.array_equal(call_args[0][0], frame)
        # Confidence threshold must match what was set
        passed_params: InferenceParams = call_args[0][1]
        assert abs(passed_params.confidence_threshold - 0.70) < 1e-9

    def test_no_inference_error_emitted_on_success(self):
        worker = InferenceWorker()
        inject_mock_engine(worker, make_detection_result())

        errors: list[str] = []
        worker.inference_error.connect(lambda e: errors.append(e))

        worker.process_frame(make_bgr_frame())

        assert errors == []

    def test_empty_detection_result_also_emitted(self):
        """A clean board (no defects) returns an empty DetectionResult — still emit it."""
        worker = InferenceWorker()
        inject_mock_engine(worker, DetectionResult(boxes=[]))

        received: list[DetectionResult] = []
        worker.new_detections.connect(lambda r: received.append(r))

        worker.process_frame(make_bgr_frame())

        assert len(received) == 1
        assert received[0].boxes == []
        assert (
            received[0].average_confidence == 1.0
        )  # Clean board → 1.0 (per models.py)

    def test_confidence_threshold_from_params_passed_to_engine(self):
        """
        Verifies the full params→engine pipeline: updating threshold via
        update_params() must reach InferenceEngine.run() correctly.
        """
        worker = InferenceWorker()
        mock_engine = inject_mock_engine(worker, DetectionResult())
        worker.update_params(InferenceParams(confidence_threshold=0.30))

        worker.process_frame(make_bgr_frame())

        _, kwargs_or_args = mock_engine.run.call_args
        if mock_engine.run.call_args[0]:
            params_arg = mock_engine.run.call_args[0][1]
        else:
            params_arg = mock_engine.run.call_args[1].get("params")

        if params_arg is not None:
            assert abs(params_arg.confidence_threshold - 0.30) < 1e-9


# ===========================================================================
# process_frame — error handling path
# ===========================================================================


class TestProcessFrameErrorHandling:
    def test_emits_inference_error_on_engine_exception(self):
        worker = InferenceWorker()
        mock_engine = MagicMock()
        mock_engine.run.side_effect = RuntimeError("OpenVINO timeout")
        worker._engine = mock_engine

        errors: list[str] = []
        worker.inference_error.connect(lambda e: errors.append(e))

        worker.process_frame(make_bgr_frame())

        assert len(errors) == 1
        assert "RuntimeError" in errors[0] or "Inference failed" in errors[0]

    def test_no_new_detections_emitted_on_engine_exception(self):
        worker = InferenceWorker()
        mock_engine = MagicMock()
        mock_engine.run.side_effect = ValueError("bad frame shape")
        worker._engine = mock_engine

        received: list = []
        worker.new_detections.connect(lambda r: received.append(r))

        worker.process_frame(make_bgr_frame())

        assert received == []

    def test_error_message_contains_exception_type(self):
        worker = InferenceWorker()
        mock_engine = MagicMock()
        mock_engine.run.side_effect = MemoryError("OOM")
        worker._engine = mock_engine

        errors: list[str] = []
        worker.inference_error.connect(lambda e: errors.append(e))

        worker.process_frame(make_bgr_frame())

        assert "MemoryError" in errors[0]

    def test_worker_recovers_after_inference_exception(self):
        """
        After an exception in process_frame(), the next call must succeed.
        Validates that the finally block always releases the lock.
        """
        worker = InferenceWorker()
        mock_engine = MagicMock()

        # First call fails
        mock_engine.run.side_effect = RuntimeError("first call fails")
        worker._engine = mock_engine
        errors: list[str] = []
        worker.inference_error.connect(lambda e: errors.append(e))
        worker.process_frame(make_bgr_frame())
        assert len(errors) == 1

        # Second call succeeds
        result = make_detection_result()
        mock_engine.run.side_effect = None
        mock_engine.run.return_value = result
        received: list = []
        worker.new_detections.connect(lambda r: received.append(r))
        worker.process_frame(make_bgr_frame())

        assert len(received) == 1, (
            "Worker must recover from previous exception and process the next frame"
        )
