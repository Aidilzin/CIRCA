"""
core/inference_engine.py
------------------------
Wraps Intel OpenVINO 2024.x Runtime for synchronous INT8 IR model inference.

Architecture constraints (enforced):
  - NO PyQt6 / PySide6 imports — this module must remain UI-framework-agnostic.
  - Instantiated ONCE at application startup; CompiledModel cached for lifetime.
  - Called exclusively from workers/inference_worker.py inside the Inference Thread.
  - Never call infer_new_request() outside of this class.

Functional requirements covered:
  FR6  — INT8 synchronous OpenVINO inference          → InferenceEngine.run()
  FR7  — Detect solder bridge                         → CLASS_LABELS index 0
  FR8  — Detect missing component                     → CLASS_LABELS index 1
  FR9  — Detect misaligned component                  → CLASS_LABELS index 2
  FR10 — Detect burnt area                            → CLASS_LABELS index 3
  FR11 — Per-detection confidence score               → BoundingBox.confidence

Non-functional requirements:
  NFR2 — Total inference latency < 10s per frame (budgeted at ~40ms on consumer
          iGPU with INT8 IR; synchronous in dedicated thread satisfies NFR3).

Output tensor layout (YOLOv12 / YOLOv8-style ONNX→OpenVINO export):
  Shape: [1, 4 + num_classes, num_anchors]
         ═══   ═════════════  ════════════
         batch  bbox(cx,cy,   typically 8400 for 640×640 input
                w,h) + scores   at three YOLO detection heads
  Note: The architecture doc cites [1, 84, 8400] as the COCO baseline.
  For CIRCA's 4-class custom model: [1, 8, 8400].
  This implementation derives num_classes from the output shape at load time
  so it is robust to both layouts.

Post-processing pipeline (per frame):
  1. Resize + normalize + NCHW transpose → model input tensor
  2. infer_new_request() → raw output tensor [1, 4+C, 8400]
  3. Transpose to [8400, 4+C]; split into boxes (cx,cy,w,h) and class scores
  4. Per anchor: class_id = argmax(scores), confidence = max(scores)
  5. Confidence threshold filter (uses InferenceParams.confidence_threshold)
  6. Convert cx,cy,w,h → x,y,w,h (top-left corner, pixel units)
  7. Scale from model input size (640,640) → original frame size
  8. NMS via cv2.dnn.NMSBoxes (IOU_THRESHOLD = 0.45)
  9. Construct and return List[BoundingBox]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

from core.models import BoundingBox, DetectionResult, InferenceParams

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defect class label map — FR7 through FR10
# Index order MUST match the YOLOv12 training configuration class order.
# ---------------------------------------------------------------------------
CLASS_LABELS: dict[int, str] = {
    0: "solder_bridge",
    1: "missing_component",
    2: "misaligned_component",
    3: "burnt_area",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard YOLO input resolution. Must match the resolution used during
# model export. Change here if re-exported at a different size.
MODEL_INPUT_SIZE: tuple[int, int] = (640, 640)  # (width, height)

# IoU threshold for NMS. 0.45 is the standard YOLO default.
# Lower values → more aggressive suppression (fewer overlapping boxes).
IOU_THRESHOLD: float = 0.45


# ---------------------------------------------------------------------------
# InferenceEngine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Wraps an OpenVINO CompiledModel for synchronous YOLOv12 INT8 inference.

    Lifecycle:
      - Instantiated once in InferenceWorker.__init__().
      - Model is compiled and cached in self._compiled_model.
      - run() is called once per sharp, preprocessed frame from the queue.
      - No state is mutated between calls (thread-safe for single-consumer use).

    Usage:
        engine = InferenceEngine("models/yolov12_int8.xml")
        result: DetectionResult = engine.run(bgr_frame, params)
    """

    def __init__(self, model_xml_path: str) -> None:
        """
        Load and compile the OpenVINO IR model.

        Args:
            model_xml_path: Absolute or relative path to the .xml model file.
                            The companion .bin weights file must reside in the
                            same directory with the same stem name.

        Raises:
            FileNotFoundError: If the .xml or companion .bin file does not exist.
            RuntimeError: If OpenVINO fails to compile the model (e.g. plugin
                          error, corrupt model, unsupported op).
        """
        xml_path = Path(model_xml_path)
        bin_path = xml_path.with_suffix(".bin")

        if not xml_path.exists():
            raise FileNotFoundError(
                f"OpenVINO model XML not found: {xml_path.resolve()}\n"
                "Ensure 'models/yolov12_int8.xml' exists in the project root."
            )
        if not bin_path.exists():
            raise FileNotFoundError(
                f"OpenVINO model weights (.bin) not found: {bin_path.resolve()}\n"
                "The .xml and .bin files must reside in the same directory."
            )

        try:
            # Import here so the module can be imported even without openvino
            # installed (tests mock this path). Import at method scope also
            # avoids polluting the module namespace with openvino symbols.
            from openvino.runtime import Core  # type: ignore[import]

            ov_core = Core()
            model = ov_core.read_model(str(xml_path))
            # "CPU" device: consistent performance on any repair bench PC.
            # OpenVINO auto-selects iGPU if available via "AUTO" device,
            # but CPU is more predictable for INT8 thermal behaviour (NFR6).
            self._compiled_model = ov_core.compile_model(model, device_name="CPU")

        except Exception as exc:
            raise RuntimeError(
                f"Failed to compile OpenVINO model '{xml_path}': {exc}"
            ) from exc

        # Resolve and cache input/output layer references once at load time.
        self._input_layer = self._compiled_model.input(0)
        self._output_layer = self._compiled_model.output(0)

        # Derive num_classes from the output shape: [1, 4+C, 8400] → C
        output_shape = self._output_layer.shape
        # output_shape[1] = 4 (bbox) + num_classes
        self._num_classes = int(output_shape[1]) - 4
        if self._num_classes < 1:
            raise RuntimeError(
                f"Unexpected model output shape {output_shape}. "
                f"Expected [1, 4+num_classes, num_anchors]; got {list(output_shape)}."
            )

        logger.info(
            "OpenVINO model loaded: %s | classes=%d | device=CPU",
            xml_path.name,
            self._num_classes,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, frame: np.ndarray, params: InferenceParams) -> DetectionResult:
        """
        Run a single synchronous inference pass on a preprocessed BGR frame.

        This method is the sole caller of infer_new_request() in the entire
        codebase (architecture boundary rule).

        Args:
            frame:  Preprocessed BGR frame (H, W, 3) uint8 from CameraWorker.
                    Must not be None or empty; caller is responsible for gating
                    blurry frames via compute_variance() before calling here.
            params: Live inference parameters (confidence_threshold).

        Returns:
            DetectionResult containing zero or more BoundingBox objects,
            filtered by confidence_threshold and de-duplicated by NMS.

        Raises:
            ValueError: If frame is None, empty, or wrong shape.
            RuntimeError: If OpenVINO inference raises an exception.
        """
        if frame is None or frame.size == 0:
            raise ValueError("InferenceEngine.run() received an empty frame.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected (H, W, 3) BGR frame; got shape {frame.shape}."
            )

        original_h, original_w = frame.shape[:2]

        # Stage 1: Preprocess frame → model input tensor
        input_tensor = self._preprocess_frame(frame)

        # Stage 2: Synchronous inference
        try:
            infer_request = self._compiled_model.create_infer_request()
            infer_request.infer({self._input_layer: input_tensor})
            raw_output: np.ndarray = infer_request.get_output_tensor(0).data
        except Exception as exc:
            raise RuntimeError(f"OpenVINO infer_new_request failed: {exc}") from exc

        # Stage 3: Post-process raw tensor → DetectionResult
        boxes = self._postprocess(
            raw_output,
            params,
            original_w=original_w,
            original_h=original_h,
        )

        return DetectionResult(boxes=boxes)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize, normalize, and transpose a BGR frame to a model-ready NCHW tensor.

        Pipeline:
          BGR (H, W, 3) uint8
          → resize to MODEL_INPUT_SIZE (640, 640)
          → convert BGR → RGB (YOLO trained on RGB)
          → normalize: divide by 255.0 → float32 in [0, 1]
          → transpose HWC → CHW
          → add batch dimension → [1, 3, 640, 640]

        Args:
            frame: Raw BGR (H, W, 3) uint8 numpy array.

        Returns:
            float32 numpy array of shape [1, 3, 640, 640].
        """
        resized = cv2.resize(frame, MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))          # HWC → CHW
        return np.expand_dims(chw, axis=0)                  # CHW → NCHW [1,3,H,W]

    def _postprocess(
        self,
        raw_output: np.ndarray,
        params: InferenceParams,
        original_w: int,
        original_h: int,
    ) -> List[BoundingBox]:
        """
        Parse the raw YOLOv12 output tensor into a filtered, NMS-deduped
        list of BoundingBox objects scaled to the original frame dimensions.

        Tensor layout: [1, 4 + num_classes, num_anchors]
          - Axis 1, rows 0–3 : cx, cy, w, h  (pixel coords in 640×640 space)
          - Axis 1, rows 4+  : per-class confidence scores (no separate objectness
                               in YOLOv8/v12 — class score IS the final confidence)

        Args:
            raw_output:   Raw numpy array directly from the output tensor.
            params:       InferenceParams with live confidence_threshold.
            original_w/h: Frame dimensions BEFORE resize — used for coordinate scaling.

        Returns:
            Filtered, NMS-suppressed List[BoundingBox] in original pixel space.
        """
        # Squeeze batch dim: [1, 4+C, A] → [4+C, A] → transpose → [A, 4+C]
        output = np.squeeze(raw_output, axis=0).T   # shape: [num_anchors, 4+C]

        num_anchors = output.shape[0]
        if num_anchors == 0:
            return []

        # Split into bbox coordinates and class scores
        boxes_cxcywh = output[:, :4]                 # [A, 4]: cx, cy, w, h
        class_scores = output[:, 4:]                 # [A, C]: per-class confidence

        # Per anchor: best class index + its confidence score
        class_ids = np.argmax(class_scores, axis=1)  # [A]
        confidences = class_scores[
            np.arange(num_anchors), class_ids
        ]                                            # [A]

        # Confidence threshold filter (FR11 / InferenceParams.confidence_threshold)
        conf_mask = confidences >= params.confidence_threshold
        if not np.any(conf_mask):
            return []

        boxes_filtered = boxes_cxcywh[conf_mask]     # [K, 4]
        confs_filtered = confidences[conf_mask]       # [K]
        ids_filtered = class_ids[conf_mask]           # [K]

        # Scale factors: model input space (640×640) → original frame space
        scale_x = original_w / MODEL_INPUT_SIZE[0]
        scale_y = original_h / MODEL_INPUT_SIZE[1]

        # Convert cx,cy,w,h → x1,y1,w,h (top-left corner) for NMS input
        # Coordinates are still in model input (640×640) space at this point.
        cx, cy, bw, bh = (
            boxes_filtered[:, 0],
            boxes_filtered[:, 1],
            boxes_filtered[:, 2],
            boxes_filtered[:, 3],
        )
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0

        # Build list of [x, y, w, h] in model space for cv2.dnn.NMSBoxes
        nms_boxes = np.stack([x1, y1, bw, bh], axis=1).tolist()
        nms_scores = confs_filtered.tolist()

        # NMS per class — run NMS on the full set; class identity is preserved
        # because we pass per-anchor class_ids through. Standard YOLO practice
        # for small class counts (4) is class-agnostic NMS at this threshold.
        #
        # NOTE: cv2.dnn.NMSBoxes applies a strict `score > score_threshold`
        # filter internally (i.e. it EXCLUDES boxes at exactly the threshold).
        # Our upstream numpy mask already applied `>= confidence_threshold`, so
        # subtracting a tiny epsilon here ensures NMSBoxes does not re-filter
        # those boundary boxes, preserving the intended >= semantics.
        _nms_score_threshold = max(0.0, params.confidence_threshold - 1e-6)
        nms_indices = cv2.dnn.NMSBoxes(
            bboxes=nms_boxes,
            scores=nms_scores,
            score_threshold=_nms_score_threshold,
            nms_threshold=IOU_THRESHOLD,
        )

        if nms_indices is None or len(nms_indices) == 0:
            return []

        # cv2.dnn.NMSBoxes returns a (N, 1) array in older OpenCV; flatten.
        nms_indices = np.array(nms_indices).flatten()

        result_boxes: List[BoundingBox] = []
        for idx in nms_indices:
            class_id = int(ids_filtered[idx])
            class_name = CLASS_LABELS.get(class_id, f"class_{class_id}")
            confidence = float(confs_filtered[idx])

            # Scale from model input space → original frame space
            x_scaled = int(round(x1[idx] * scale_x))
            y_scaled = int(round(y1[idx] * scale_y))
            w_scaled = int(round(bw[idx] * scale_x))
            h_scaled = int(round(bh[idx] * scale_y))

            # Clamp to frame boundaries (handles floating-point overshoot)
            x_scaled = max(0, min(x_scaled, original_w - 1))
            y_scaled = max(0, min(y_scaled, original_h - 1))
            w_scaled = max(1, min(w_scaled, original_w - x_scaled))
            h_scaled = max(1, min(h_scaled, original_h - y_scaled))

            result_boxes.append(
                BoundingBox(
                    x=x_scaled,
                    y=y_scaled,
                    width=w_scaled,
                    height=h_scaled,
                    class_name=class_name,
                    confidence=confidence,
                )
            )

        return result_boxes
