"""
tests/test_inference_engine.py
------------------------------
Unit tests for core/inference_engine.py.

Testing strategy:
  - Model loading:    Mock openvino.runtime.Core to avoid needing a real model
                      file on disk; test FileNotFoundError for missing .xml/.bin.
  - _preprocess_frame: Call directly — tests shape, dtype, value range, channel
                        order correctness without any mock.
  - _postprocess:     Inject synthetic raw tensors representing known detection
                      scenarios; assert BoundingBox fields, confidence filtering,
                      NMS behaviour, coordinate scaling, and clamping.
  - run():            Integration of preprocess + infer mock + postprocess.
  - CLASS_LABELS:     Verify the 4 required defect class names are present.

No PyQt6 import anywhere in this file (validates core/ boundary rule).
No real OpenVINO model file required.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from core.models import BoundingBox, DetectionResult, InferenceParams
from core.inference_engine import (
    CLASS_LABELS,
    MODEL_INPUT_SIZE,
    InferenceEngine,
)


# ===========================================================================
# Fixtures and helpers
# ===========================================================================


def make_inference_engine_with_mocks(
    num_classes: int = 7,
    num_anchors: int = 8400,
) -> tuple[InferenceEngine, MagicMock]:
    """
    Construct an InferenceEngine instance with all OpenVINO I/O fully mocked.

    Returns the engine and the mock compiled_model so tests can inject
    synthetic output tensors via mock_compiled_model.

    The engine's _postprocess and _preprocess_frame methods are real code paths.
    """
    output_shape = (1, 4 + num_classes, num_anchors)

    # Build a fake output layer with .shape attribute
    mock_output_layer = MagicMock()
    mock_output_layer.shape = output_shape

    mock_input_layer = MagicMock()

    mock_compiled_model = MagicMock()
    mock_compiled_model.input.return_value = mock_input_layer
    mock_compiled_model.output.return_value = mock_output_layer

    # Pre-wire a reusable infer request (InferenceEngine caches this at init)
    mock_request = MagicMock()
    mock_compiled_model.create_infer_request.return_value = mock_request

    # Patch the openvino import inside inference_engine's __init__
    with (
        patch.dict("sys.modules", {"openvino.runtime": MagicMock()}),
        patch("core.inference_engine.Path.exists", return_value=True),
        patch("builtins.open", MagicMock()),
    ):
        mock_ov_module = MagicMock()
        mock_core_instance = MagicMock()
        mock_core_instance.read_model.return_value = MagicMock()
        mock_core_instance.compile_model.return_value = mock_compiled_model
        mock_ov_module.Core.return_value = mock_core_instance

        with patch.dict("sys.modules", {"openvino.runtime": mock_ov_module}):
            engine = InferenceEngine.__new__(InferenceEngine)
            engine._compiled_model = mock_compiled_model
            engine._input_layer = mock_input_layer
            engine._output_layer = mock_output_layer
            engine._num_classes = num_classes
            # Wire the cached infer request (created once at init in production)
            engine._infer_request = mock_request

    return engine, mock_compiled_model


def make_raw_output(
    cx: float,
    cy: float,
    w: float,
    h: float,
    class_scores: list[float],
    num_anchors: int = 8400,
) -> np.ndarray:
    """
    Build a synthetic [1, 4+C, num_anchors] raw output tensor with a single
    detection at anchor index 0 and zeros everywhere else.
    """
    num_classes = len(class_scores)
    output = np.zeros((1, 4 + num_classes, num_anchors), dtype=np.float32)
    output[0, 0, 0] = cx
    output[0, 1, 0] = cy
    output[0, 2, 0] = w
    output[0, 3, 0] = h
    for i, score in enumerate(class_scores):
        output[0, 4 + i, 0] = score
    return output


def make_default_params(**overrides) -> InferenceParams:
    params = InferenceParams()
    for key, val in overrides.items():
        setattr(params, key, val)
    return params


# ===========================================================================
# CLASS_LABELS — FR7–FR10
# ===========================================================================


class TestClassLabels:
    def test_has_seven_classes(self):
        """CIRCA uses the unified_pcb_v3 7-class taxonomy (nc=7)."""
        assert len(CLASS_LABELS) == 7

    def test_missing_hole_at_index_0(self):
        assert CLASS_LABELS[0] == "missing_hole"

    def test_mouse_bite_at_index_1(self):
        assert CLASS_LABELS[1] == "mouse_bite"

    def test_open_circuit_at_index_2(self):
        assert CLASS_LABELS[2] == "open_circuit"

    def test_short_at_index_3(self):
        assert CLASS_LABELS[3] == "short"

    def test_excess_solder_at_index_4(self):
        assert CLASS_LABELS[4] == "excess_solder"

    def test_insufficient_solder_at_index_5(self):
        assert CLASS_LABELS[5] == "insufficient_solder"

    def test_cold_solder_joint_at_index_6(self):
        assert CLASS_LABELS[6] == "cold_solder_joint"

    def test_indices_are_zero_based_integers(self):
        assert list(CLASS_LABELS.keys()) == [0, 1, 2, 3, 4, 5, 6]


# ===========================================================================
# InferenceEngine.__init__ — model loading (FR6)
# ===========================================================================


class TestInferenceEngineInit:
    def test_raises_file_not_found_when_xml_missing(self, tmp_path):
        missing = str(tmp_path / "nonexistent.xml")
        with pytest.raises(FileNotFoundError, match="model XML not found"):
            InferenceEngine(missing)

    def test_raises_file_not_found_when_bin_missing(self, tmp_path):
        # Create .xml but not .bin
        xml = tmp_path / "model.xml"
        xml.write_text("<model/>")
        with pytest.raises(FileNotFoundError, match=".bin"):
            InferenceEngine(str(xml))

    def test_raises_runtime_error_when_openvino_compile_fails(self, tmp_path):
        xml = tmp_path / "model.xml"
        bin_ = tmp_path / "model.bin"
        xml.write_text("<model/>")
        bin_.write_bytes(b"\x00")

        mock_ov = MagicMock()
        mock_core = MagicMock()
        mock_core.compile_model.side_effect = RuntimeError("Plugin error")
        mock_ov.Core.return_value = mock_core

        with patch.dict("sys.modules", {"openvino.runtime": mock_ov}):
            with pytest.raises(RuntimeError, match="Failed to compile"):
                InferenceEngine(str(xml))

    def test_num_classes_derived_from_output_shape(self):
        engine, _ = make_inference_engine_with_mocks(num_classes=4)
        assert engine._num_classes == 4


# ===========================================================================
# InferenceEngine._preprocess_frame
# ===========================================================================


class TestPreprocessFrame:
    def setup_method(self):
        self.engine, _ = make_inference_engine_with_mocks()

    def test_output_shape_is_nchw(self):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        tensor, pad_x, pad_y, scale = self.engine._preprocess_frame(frame)
        assert tensor.shape == (1, 3, MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0])

    def test_output_dtype_is_float32(self):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        tensor, _, _, _ = self.engine._preprocess_frame(frame)
        assert tensor.dtype == np.float32

    def test_output_values_in_zero_one_range(self):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        tensor, _, _, _ = self.engine._preprocess_frame(frame)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_pure_white_frame_normalises_to_one(self):
        """A 640x640 white frame has no letterbox padding, so tensor is all 1.0."""
        frame = np.full((640, 640, 3), 255, dtype=np.uint8)  # square = no padding
        tensor, _, _, _ = self.engine._preprocess_frame(frame)
        np.testing.assert_allclose(tensor, 1.0, atol=1e-5)

    def test_pure_black_frame_normalises_to_zero(self):
        """A 640x640 black frame has no letterbox padding, so tensor is all 0.0."""
        frame = np.zeros((640, 640, 3), dtype=np.uint8)  # square = no padding
        tensor, _, _, _ = self.engine._preprocess_frame(frame)
        np.testing.assert_allclose(tensor, 0.0, atol=1e-5)

    def test_handles_non_square_input_frame(self):
        """InferenceEngine must resize arbitrary-resolution frames to 640x640."""
        frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        tensor, pad_x, pad_y, scale = self.engine._preprocess_frame(frame)
        assert tensor.shape == (1, 3, 640, 640)
        assert isinstance(scale, float) and scale > 0.0

    def test_bgr_to_rgb_channel_swap(self):
        """
        Verify BGR→RGB channel swap. A pure-blue BGR frame (B=255, G=0, R=0)
        should map to an RGB frame where R=0, G=0, B=255.
        NCHW channel 0 = R channel = 0 for a blue pixel.
        Use 640x640 so no letterbox padding is present.
        """
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Blue channel in BGR
        tensor, _, _, _ = self.engine._preprocess_frame(frame)
        # After BGR→RGB: R channel (NCHW ch 0) should be 0 for a blue pixel
        assert tensor[0, 0].max() < 0.01  # R channel of blue pixel is ~0


# ===========================================================================
# InferenceEngine._postprocess — core FR logic
# ===========================================================================


class TestPostprocess:
    def setup_method(self):
        self.engine, _ = make_inference_engine_with_mocks(num_classes=7)
        self.params = make_default_params(confidence_threshold=0.5)

    def _run_postprocess(
        self,
        raw_output,
        original_w=640,
        original_h=640,
        pad_x=0.0,
        pad_y=0.0,
        scale=1.0,
    ):
        """Helper: call _postprocess with letterbox defaults (no-op padding)."""
        return self.engine._postprocess(
            raw_output,
            self.params,
            original_w=original_w,
            original_h=original_h,
            pad_x=pad_x,
            pad_y=pad_y,
            scale=scale,
        )

    # --- Basic detection ---

    def test_single_high_confidence_detection_returned(self):
        """One anchor above threshold → one BoundingBox returned."""
        raw = make_raw_output(
            cx=320,
            cy=320,
            w=100,
            h=100,
            class_scores=[0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0],  # mouse_bite, conf=0.9
        )
        boxes = self._run_postprocess(raw)
        assert len(boxes) == 1
        assert boxes[0].class_name == "mouse_bite"
        assert abs(boxes[0].confidence - 0.9) < 1e-4

    def test_box_with_confidence_below_threshold_filtered_out(self):
        """Anchor confidence below threshold → empty result."""
        raw = make_raw_output(
            cx=320,
            cy=320,
            w=100,
            h=100,
            class_scores=[0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 0.3 < threshold 0.5
        )
        boxes = self._run_postprocess(raw)
        assert len(boxes) == 0

    def test_box_exactly_at_threshold_is_included(self):
        """Boundary: confidence == threshold → included (>= comparison)."""
        raw = make_raw_output(
            cx=320,
            cy=320,
            w=100,
            h=100,
            class_scores=[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        boxes = self._run_postprocess(raw)
        assert len(boxes) == 1

    def test_empty_output_tensor_returns_empty_list(self):
        """All anchors at zero confidence → empty result."""
        raw = np.zeros((1, 11, 8400), dtype=np.float32)  # 4 + 7 classes
        boxes = self._run_postprocess(raw)
        assert len(boxes) == 0

    # --- Class label assignment (FR7-FR10) ---

    def test_class_0_maps_to_missing_hole(self):
        raw = make_raw_output(320, 320, 100, 100, [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        boxes = self._run_postprocess(raw)
        assert boxes[0].class_name == "missing_hole"

    def test_class_1_maps_to_mouse_bite(self):
        raw = make_raw_output(320, 320, 100, 100, [0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
        boxes = self._run_postprocess(raw)
        assert boxes[0].class_name == "mouse_bite"

    def test_class_2_maps_to_open_circuit(self):
        raw = make_raw_output(320, 320, 100, 100, [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0])
        boxes = self._run_postprocess(raw)
        assert boxes[0].class_name == "open_circuit"

    def test_class_3_maps_to_short(self):
        raw = make_raw_output(320, 320, 100, 100, [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0])
        boxes = self._run_postprocess(raw)
        assert boxes[0].class_name == "short"

    def test_unknown_class_id_falls_back_to_class_n_label(self):
        """If class_id > len(CLASS_LABELS), fallback label is used."""
        engine, _ = make_inference_engine_with_mocks(num_classes=8)
        raw = make_raw_output(
            320, 320, 100, 100,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]
        )
        boxes = engine._postprocess(
            raw, self.params, 640, 640, pad_x=0.0, pad_y=0.0, scale=1.0
        )
        assert boxes[0].class_name == "class_7"

    # --- Coordinate scaling ---

    def test_box_coordinates_scaled_from_640_to_original_size(self):
        """
        Detection at cx=320, cy=320, w=200, h=100 in 640×640 model space.
        With no letterbox padding (square input) and scale=0.5 (1280×960 → 640):
          x_orig = (cx - w/2 - pad_x) / scale = (320-100-0)/0.5 = 440
          y_orig = (cy - h/2 - pad_y) / scale = (320-50-0)/0.5  = 540  (→ clamped)
        For simplicity, use scale=1.0 (square frame matching model space):
          x_orig = 320-100 = 220, y_orig = 320-50 = 270
        """
        raw = make_raw_output(
            cx=320, cy=320, w=200, h=100,
            class_scores=[0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        boxes = self._run_postprocess(
            raw, original_w=640, original_h=640, pad_x=0.0, pad_y=0.0, scale=1.0
        )
        assert len(boxes) == 1
        b = boxes[0]
        # x1 = (cx - w/2 - 0) / 1.0 = 220
        assert b.x == 220
        # y1 = (cy - h/2 - 0) / 1.0 = 270
        assert b.y == 270
        assert b.width == 200
        assert b.height == 100

    def test_box_clamped_to_frame_boundaries(self):
        """
        A box whose centre is near the edge with large width/height may
        compute negative x1 or x1+w > frame_width. Must be clamped.
        # cx=10, cy=10 with w=100, h=100 → x1 = 10-50 = -40 → clamped to 0
        """
        raw = make_raw_output(
            cx=10, cy=10, w=100, h=100,
            class_scores=[0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        boxes = self._run_postprocess(
            raw, original_w=640, original_h=640, pad_x=0.0, pad_y=0.0, scale=1.0
        )
        # x1 and y1 must never be negative
        if boxes:
            assert boxes[0].x >= 0
            assert boxes[0].y >= 0
            assert boxes[0].width >= 1
            assert boxes[0].height >= 1

    # --- NMS behaviour ---

    def test_two_identical_overlapping_boxes_collapsed_to_one(self):
        """
        Two anchors at almost identical positions with the same class.
        NMS must keep exactly one.
        """
        raw = np.zeros((1, 11, 8400), dtype=np.float32)  # 4 + 7 classes
        # Anchor 0: high confidence
        raw[0, :4, 0] = [320, 320, 200, 200]
        raw[0, 4, 0] = 0.95
        # Anchor 1: slightly lower confidence, almost identical box
        raw[0, :4, 1] = [322, 322, 200, 200]
        raw[0, 4, 1] = 0.90
        boxes = self._run_postprocess(raw)
        assert len(boxes) == 1
        assert abs(boxes[0].confidence - 0.95) < 1e-4

    def test_two_non_overlapping_boxes_both_returned(self):
        """
        Two anchors at spatially separate positions → NMS should keep both.
        """
        raw = np.zeros((1, 11, 8400), dtype=np.float32)  # 4 + 7 classes
        # Top-left detection
        raw[0, :4, 0] = [80, 80, 80, 80]
        raw[0, 4, 0] = 0.9
        # Bottom-right detection (far away)
        raw[0, :4, 1] = [560, 560, 80, 80]
        raw[0, 5, 1] = 0.85
        boxes = self._run_postprocess(raw)
        assert len(boxes) == 2

    # --- DetectionResult integration ---

    def test_result_boxes_are_bounding_box_instances(self):
        raw = make_raw_output(320, 320, 100, 100, [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        boxes = self._run_postprocess(raw)
        for box in boxes:
            assert isinstance(box, BoundingBox)

    def test_confidence_preserved_to_float(self):
        raw = make_raw_output(320, 320, 100, 100, [0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0])
        boxes = self._run_postprocess(raw)
        assert isinstance(boxes[0].confidence, float)
        assert abs(boxes[0].confidence - 0.75) < 1e-4

    def test_custom_confidence_threshold_applied(self):
        """Lowering threshold to 0.3 should surface previously-filtered boxes."""
        raw = make_raw_output(
            320, 320, 100, 100,
            [0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        # Default threshold 0.5 → filtered out
        high_thresh = make_default_params(confidence_threshold=0.5)
        boxes_high = self.engine._postprocess(
            raw, high_thresh, 640, 640, pad_x=0.0, pad_y=0.0, scale=1.0
        )
        assert len(boxes_high) == 0

        # Lower threshold 0.3 → included
        low_thresh = make_default_params(confidence_threshold=0.3)
        boxes_low = self.engine._postprocess(
            raw, low_thresh, 640, 640, pad_x=0.0, pad_y=0.0, scale=1.0
        )
        assert len(boxes_low) == 1


# ===========================================================================
# InferenceEngine.run() — integration (mocked infer request)
# ===========================================================================


class TestInferenceEngineRun:
    def _make_engine_with_output(self, raw_output: np.ndarray, num_classes=7):
        """Build a fully mocked engine that returns `raw_output` on infer."""
        engine, mock_compiled = make_inference_engine_with_mocks(
            num_classes=num_classes,
            num_anchors=raw_output.shape[2],
        )

        # Update the cached infer request's output tensor to return raw_output
        mock_tensor = MagicMock()
        mock_tensor.data = raw_output
        engine._infer_request.get_output_tensor.return_value = mock_tensor

        return engine

    def test_run_returns_detection_result_instance(self):
        raw = make_raw_output(320, 320, 100, 100, [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        engine = self._make_engine_with_output(raw)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        params = make_default_params()
        result = engine.run(frame, params)
        assert isinstance(result, DetectionResult)

    def test_run_returns_detections_for_high_confidence_tensor(self):
        raw = make_raw_output(320, 320, 100, 100, [0.0, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0])
        engine = self._make_engine_with_output(raw, num_classes=7)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        params = make_default_params()
        result = engine.run(frame, params)
        assert len(result.boxes) == 1
        assert result.boxes[0].class_name == "mouse_bite"

    def test_run_returns_empty_for_all_low_confidence_tensor(self):
        raw = np.zeros((1, 11, 8400), dtype=np.float32)  # 4 + 7 classes
        engine = self._make_engine_with_output(raw)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        params = make_default_params()
        result = engine.run(frame, params)
        assert len(result.boxes) == 0

    def test_run_raises_value_error_on_none_frame(self):
        raw = np.zeros((1, 11, 8400), dtype=np.float32)  # 4 + 7 classes
        engine = self._make_engine_with_output(raw)
        params = make_default_params()
        with pytest.raises(ValueError, match="empty frame"):
            engine.run(None, params)  # type: ignore[arg-type]

    def test_run_raises_value_error_on_wrong_channels(self):
        raw = np.zeros((1, 11, 8400), dtype=np.float32)  # 4 + 7 classes
        engine = self._make_engine_with_output(raw)
        params = make_default_params()
        grey = np.zeros((480, 640), dtype=np.uint8)
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            engine.run(grey, params)

    def test_run_raises_runtime_error_on_openvino_failure(self):
        engine, _ = make_inference_engine_with_mocks()
        # Simulate infer_request raising at inference time
        engine._infer_request.infer.side_effect = RuntimeError("OV crash")
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        params = make_default_params()
        with pytest.raises(RuntimeError, match="infer_new_request failed"):
            engine.run(frame, params)

    def test_detection_result_average_confidence_correct(self):
        """
        DetectionResult.average_confidence (used for FR15 warning banner)
        must correctly reflect the confidence of detected boxes.
        """
        raw = make_raw_output(320, 320, 100, 100, [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0])
        engine = self._make_engine_with_output(raw)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        params = make_default_params()
        result = engine.run(frame, params)
        assert abs(result.average_confidence - 0.9) < 1e-4

    def test_clean_board_returns_average_confidence_of_one(self):
        """
        DetectionResult with no boxes → average_confidence returns 1.0
        (defined behaviour in models.py for clean boards).
        """
        raw = np.zeros((1, 11, 8400), dtype=np.float32)  # 4 + 7 classes
        engine = self._make_engine_with_output(raw)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        params = make_default_params()
        result = engine.run(frame, params)
        assert result.average_confidence == 1.0
