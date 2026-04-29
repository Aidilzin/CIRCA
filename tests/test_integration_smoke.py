"""
tests/test_integration_smoke.py
-------------------------------
[P0] Integration and Smoke tests for CIRCA backend components.

Features covered:
  - [P0] NFR5: Thread Decoupling (InferenceWorker in QThread)
  - [P0] FR6: OpenVINO Integration (Real InferenceEngine loading)
  - [P1] FR1: Real Hardware Ingest (cv2.VideoCapture smoke test)
  - [P0] NFR4: Backpressure gate verification (Lock-based frame dropping)
"""

import pytest
import cv2
import numpy as np
from PyQt6.QtCore import QThread, QObject, pyqtSignal
from PyQt6.QtTest import QSignalSpy
from workers.inference_worker import InferenceWorker
from core.inference_engine import InferenceEngine
from core.models import InferenceParams, DetectionResult

class TestIntegrationSmoke:
    """[P0] Integration and Smoke tests for CIRCA backend components."""

    @pytest.fixture
    def inference_setup(self, qapp_session):
        """
        Setup QThread and InferenceWorker for smoke testing.
        Ensures the worker is moved to a background thread to prove NFR5.
        """
        thread = QThread()
        worker = InferenceWorker()
        worker.moveToThread(thread)
        thread.start()
        yield thread, worker
        
        # Cleanup: Stop thread gracefully
        thread.quit()
        if not thread.wait(2000):
            thread.terminate()
            thread.wait()

    def test_inference_worker_thread_decoupling(self, qapp_session, inference_setup):
        """
        [P0] NFR5: Thread Decoupling. 
        Prove InferenceWorker emits signals from its own thread without blocking.
        Uses QSignalSpy to wait for signals across thread boundaries.
        """
        thread, worker = inference_setup
        
        # We expect an error because the model doesn't exist, 
        # but the signal emission proves the thread is alive and processing.
        spy_error = QSignalSpy(worker.inference_error)
        
        # Trigger model load (asynchronous via event loop in worker thread)
        # Using a connected signal ensures it runs in the worker thread via QueuedConnection.
        class Trigger(QObject):
            load = pyqtSignal(str)
        
        trigger = Trigger()
        trigger.load.connect(worker.load_model)
        trigger.load.emit("non_existent_model.xml")
        
        # QSignalSpy.wait() runs a local event loop in the main thread,
        # keeping the UI responsive while waiting for the background thread.
        assert spy_error.wait(2000), "InferenceWorker failed to emit inference_error within timeout"
        
        assert len(spy_error) > 0
        error_msg = str(spy_error[0][0])
        assert "OpenVINO model XML not found" in error_msg
        assert thread.isRunning()

    def test_inference_engine_openvino_loading(self):
        """
        [P0] FR6: OpenVINO Integration. 
        Verify InferenceEngine attempts real model loading and handles missing files via OpenVINO logic.
        No mocks used here to ensure the real library is reachable.
        """
        # Test that it correctly identifies missing XML
        with pytest.raises(FileNotFoundError) as exc:
            InferenceEngine("missing_model.xml")
        assert "OpenVINO model XML not found" in str(exc.value)

        # Verify OpenVINO runtime can be initialized
        try:
            from openvino.runtime import Core
            core = Core()
            assert core is not None
        except (ImportError, AttributeError):
            pytest.skip("openvino-runtime is not installed or incomplete")
        except Exception as e:
            pytest.skip(f"OpenVINO Core initialization failed: {e}")

    def test_camera_hardware_ingest_smoke(self):
        """
        [P1] FR1: Real Hardware Ingest.
        Verify cv2.VideoCapture(0) can be opened and read if hardware is present.
        Gracefully skips if no camera is detected (CI environment).
        """
        # We use CAP_DSHOW as per FR1 requirement in CameraWorker
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            cap.release()
            pytest.skip("No UVC camera detected at index 0 (DirectShow). Skipping hardware smoke test.")
            
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                pytest.skip("Camera found but failed to read frame (might be in use or physical switch off).")
            
            assert ret is True
            assert frame is not None
            assert frame.ndim == 3, "Expected BGR frame (H, W, 3)"
            assert frame.size > 0
        finally:
            cap.release()

    def test_inference_worker_backpressure_smoke(self, qapp_session, inference_setup):
        """
        [P0] NFR4: Backpressure Handling.
        Verify InferenceWorker uses the lock gate to drop frames when busy.
        This integration test ensures the threading.Lock works correctly in a multi-threaded context.
        """
        thread, worker = inference_setup
        
        # Manually acquire the lock to simulate "busy" state.
        # Since we are in the main thread and worker is in another, 
        # acquiring it here will cause worker.process_frame() to fail the acquire(blocking=False) check.
        worker._inference_lock.acquire()
        
        try:
            # Try to process a frame
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            
            # This should return immediately and NOT emit any signals because it's dropped by the gate.
            spy_detections = QSignalSpy(worker.new_detections)
            worker.process_frame(dummy_frame)
            
            # Wait a bit to ensure no signal is emitted (it should be nearly instant)
            assert not spy_detections.wait(200), "Frame was not dropped even though lock was held"
            assert len(spy_detections) == 0
            
        finally:
            worker._inference_lock.release()

    def test_inference_worker_parameter_update_smoke(self, qapp_session, inference_setup):
        """
        [P2] Verify thread-safe parameter updates across thread boundary.
        """
        thread, worker = inference_setup
        
        new_params = InferenceParams(confidence_threshold=0.88)
        worker.update_params(new_params)
        
        # Verify it updated in the worker's internal state
        assert worker._params.confidence_threshold == 0.88


