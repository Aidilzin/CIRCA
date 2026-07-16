import sys
from unittest.mock import MagicMock
import pytest
from PyQt6.QtWidgets import QApplication

# Mock OpenVINO modules before any test imports them
class MockCore:
    def __init__(self):
        pass
    def read_model(self, path):
        return MagicMock()
    def compile_model(self, model, device_name):
        mock_compiled = MagicMock()
        mock_compiled.input.return_value = MagicMock()
        
        # We need the output layer shape to be set correctly
        # Let's return shape [1, 11, 8400] for standard CIRCA NC: 7 model
        mock_output = MagicMock()
        mock_output.shape = [1, 11, 8400]
        mock_compiled.output.return_value = mock_output
        
        # Pre-create infer request
        mock_request = MagicMock()
        mock_request.get_output_tensor.return_value.data = MagicMock()
        mock_compiled.create_infer_request.return_value = mock_request
        
        return mock_compiled

# Inject mock OpenVINO into sys.modules
mock_ov_runtime = MagicMock()
mock_ov_runtime.Core = MockCore

sys.modules["openvino"] = mock_ov_runtime
sys.modules["openvino.runtime"] = mock_ov_runtime

@pytest.fixture(scope="session")
def qapp():
    """Fixture to initialize the QApplication once per session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
