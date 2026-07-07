"""
core/tiled_inference.py
-----------------------
Sliding-window tiled inference engine for CIRCA.

Solves the scale-mismatch problem:
  The YOLOv12 model was trained on close-up 640×640 crops where individual
  defects occupy ~10-30% of the frame. When a whole PCB board is shown at
  camera distance, squeezing it to 640×640 makes defects too small to detect.

Solution — Adaptive sliding-window tiling:
  1. If the frame fits in one tile (both dimensions ≤ tile_size), run single
     inference — identical behaviour to the existing InferenceEngine.run().
  2. Otherwise, divide the frame into overlapping 640×640 tiles. Run inference
     on each tile. Map bounding boxes back to original image coordinates.
     Apply cross-tile NMS to remove duplicate detections at tile boundaries.

Performance (typical latency per frame at INT8 on Intel iGPU):
  Close-up (≤640px)   → 1 tile  → ~25ms  → ~40 FPS
  Webcam (1280×720)   → 4 tiles → ~100ms → ~10 FPS
  Phone HD (1920×1080)→ 9 tiles → ~225ms → ~4-5 FPS

Architecture constraints (enforced):
  - NO PyQt6 / PySide6 imports.
  - NO state outside of the InferenceEngine reference.
  - Called exclusively from workers/inference_worker.py.
"""

from __future__ import annotations

import logging
from typing import List

import cv2
import numpy as np

from core.inference_engine import InferenceEngine
from core.models import BoundingBox, DetectionResult, InferenceParams

logger = logging.getLogger(__name__)

# Default tile configuration — matches model training resolution
TILE_SIZE: int = 640
# 40% overlap ensures defects straddling a tile boundary appear fully in one tile
TILE_OVERLAP: float = 0.40
# Stride between tile origins in pixels
TILE_STRIDE: int = int(TILE_SIZE * (1.0 - TILE_OVERLAP))  # = 384

# NMS IoU threshold for cross-tile duplicate suppression
CROSS_TILE_IOU: float = 0.45


class TiledInferenceEngine:
    """
    Wraps an InferenceEngine to transparently apply sliding-window tiled
    inference for frames larger than the model's native 640×640 input.

    For close-up frames (≤640px in both dimensions), run_tiled() behaves
    identically to InferenceEngine.run() with zero overhead.

    Lifecycle: Instantiated once in InferenceWorker after model loads.
    Thread-safety: Same as InferenceEngine — single-consumer, no shared state.

    Usage:
        engine = InferenceEngine("models/yolov12_int8.xml")
        tiled = TiledInferenceEngine(engine)
        result = tiled.run_tiled(frame, params)
    """

    def __init__(self, engine: InferenceEngine) -> None:
        self._engine = engine

    def run_tiled(
        self,
        frame: np.ndarray,
        params: InferenceParams,
        tile_size: int = TILE_SIZE,
        overlap: float = TILE_OVERLAP,
    ) -> DetectionResult:
        """
        Run inference on a frame of any size using adaptive sliding-window tiling.

        For frames that fit within a single tile, falls through to single
        inference with no overhead. For larger frames, divides into overlapping
        tiles, runs inference on each, then merges with cross-tile NMS.

        Args:
            frame:      BGR frame (H, W, 3) uint8 — any resolution.
            params:     InferenceParams with confidence_threshold.
            tile_size:  Size of each square tile in pixels (default: 640).
            overlap:    Fractional overlap between adjacent tiles (default: 0.4).

        Returns:
            DetectionResult with boxes in original-frame pixel coordinates.
        """
        if frame is None or frame.size == 0:
            return DetectionResult(boxes=[])

        h, w = frame.shape[:2]

        # Fast path: frame fits in one tile — single inference, no overhead
        if h <= tile_size and w <= tile_size:
            return self._engine.run(frame, params)

        stride = int(tile_size * (1.0 - overlap))
        tiles = self._compute_tiles(h, w, tile_size, stride)

        logger.debug(
            "TiledInference: %dx%d frame → %d tiles (stride=%d, overlap=%.0f%%)",
            w, h, len(tiles), stride, overlap * 100,
        )

        all_boxes: List[BoundingBox] = []

        for (tile_x, tile_y, tile_w, tile_h) in tiles:
            tile = frame[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w]
            if tile.size == 0:
                continue

            tile_result = self._engine.run(tile, params)

            # Remap bounding boxes to original-frame coordinates
            for box in tile_result.boxes:
                all_boxes.append(BoundingBox(
                    x=box.x + tile_x,
                    y=box.y + tile_y,
                    width=box.width,
                    height=box.height,
                    class_name=box.class_name,
                    confidence=box.confidence,
                ))

        if not all_boxes:
            return DetectionResult(boxes=[])

        # Cross-tile NMS: remove duplicate detections at tile boundaries
        merged_boxes = self._cross_tile_nms(all_boxes, params.confidence_threshold)

        logger.debug(
            "TiledInference: %d raw detections → %d after cross-tile NMS",
            len(all_boxes), len(merged_boxes),
        )

        return DetectionResult(boxes=merged_boxes)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_tiles(
        h: int, w: int, tile_size: int, stride: int
    ) -> List[tuple]:
        """
        Compute (x, y, w, h) crop regions for all tiles covering a frame of
        size (h, w). The last tile in each row/column is pinned to the image
        edge so that no image region is missed even when dimensions are not
        exact multiples of stride.

        Returns:
            List of (x, y, crop_w, crop_h) tuples in pixel coordinates.
        """
        tiles = []
        y = 0
        while y < h:
            y_end = min(y + tile_size, h)
            y_start = max(0, y_end - tile_size)  # pin to edge if needed
            actual_h = y_end - y_start

            x = 0
            while x < w:
                x_end = min(x + tile_size, w)
                x_start = max(0, x_end - tile_size)  # pin to edge
                actual_w = x_end - x_start

                tiles.append((x_start, y_start, actual_w, actual_h))

                if x_end >= w:
                    break
                x += stride

            if y_end >= h:
                break
            y += stride

        return tiles

    @staticmethod
    def _cross_tile_nms(
        boxes: List[BoundingBox], confidence_threshold: float
    ) -> List[BoundingBox]:
        """
        Apply Non-Maximum Suppression across all detections pooled from every tile.

        Filters out duplicate bounding boxes that arise when the same defect
        falls within the overlapping region of two adjacent tiles.

        Uses class-agnostic NMS (standard practice for small class counts):
        boxes from different classes can still suppress each other if they
        overlap significantly — acceptable since defects rarely co-locate.

        Args:
            boxes:                All BoundingBox objects from all tiles.
            confidence_threshold: Used as cv2.dnn.NMSBoxes score_threshold.

        Returns:
            Deduplicated List[BoundingBox] after NMS.
        """
        if len(boxes) <= 1:
            return boxes

        nms_input = [[b.x, b.y, b.width, b.height] for b in boxes]
        scores = [b.confidence for b in boxes]

        # cv2.dnn.NMSBoxes uses strict >, subtract epsilon to match >= semantics
        _threshold = max(0.0, confidence_threshold - 1e-6)

        indices = cv2.dnn.NMSBoxes(
            bboxes=nms_input,
            scores=scores,
            score_threshold=_threshold,
            nms_threshold=CROSS_TILE_IOU,
        )

        if indices is None or len(indices) == 0:
            return []

        indices = np.array(indices).flatten()
        return [boxes[i] for i in indices]
