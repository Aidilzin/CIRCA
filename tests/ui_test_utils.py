"""
tests/ui_test_utils.py
-----------------------
Shared stubs and fixtures for UI unit tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QImage

from ui.main_window import MainWindow


class _StubCameraWorker(QObject):
    """Minimal QObject with the same signal/slot interface as CameraWorker."""

    new_frame = pyqtSignal(QImage)
    frame_ready_for_inference = pyqtSignal(object)
    camera_error = pyqtSignal(str)
    auto_params_updated = pyqtSignal(float, float)

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
    benchmark_completed = pyqtSignal(str, float)

    def process_frame(self, frame):
        pass

    def update_params(self, params):
        pass

    def load_model(self, path):
        pass

    def run_benchmark(self, path):
        pass

    def stop(self):
        pass


class _HeadlessMainWindow(MainWindow):
    """
    MainWindow subclass that replaces real workers with stubs.
    """

    def _create_workers(self) -> None:
        """Install stub workers with stub threads (never start())."""
        self.camera_thread = QThread()
        self.camera_thread.start = MagicMock()
        self.camera_thread.quit = MagicMock()
        self.camera_thread.wait = MagicMock(return_value=True)
        self.camera_worker = _StubCameraWorker(device_index=0)

        self.inference_thread = QThread()
        self.inference_thread.start = MagicMock()
        self.inference_thread.quit = MagicMock()
        self.inference_thread.wait = MagicMock(return_value=True)
        self.inference_worker = _StubInferenceWorker()

    def _start_workers(self) -> None:
        """Void — no real threads started in tests."""
        pass

    def _post_start_init(self) -> None:
        """Void — prevents hardware scanner launch and QMetaObject.invokeMethod."""
        pass


@pytest.fixture()
def win() -> _HeadlessMainWindow:
    """Fresh headless MainWindow for each test with proper thread cleanup."""
    w = _HeadlessMainWindow(model_path="models/fake.xml")
    yield w
    # Ensure threads are stopped and finished before destruction
    if hasattr(w, "camera_worker"):
        w.camera_worker.stop()
    if hasattr(w, "camera_thread"):
        w.camera_thread.quit()
        w.camera_thread.wait(1000)
    
    if hasattr(w, "inference_worker"):
        w.inference_worker.stop()
    if hasattr(w, "inference_thread"):
        w.inference_thread.quit()
        w.inference_thread.wait(1000)
