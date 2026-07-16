from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BoundingBox:
    """Represents a single defect detected by the OpenVINO model."""

    x: int
    y: int
    width: int
    height: int
    class_name: str
    confidence: float


@dataclass
class DetectionResult:
    """Represents the complete inference result for a single frame."""

    boxes: list["BoundingBox"] = field(default_factory=list)
    inference_time_ms: float = -1.0
    tile_count: int = 1
    preprocessed_frame: Optional[object] = field(default=None, repr=False)
    auto_clahe: Optional[float] = None
    auto_gamma: Optional[float] = None

    @property
    def average_confidence(self) -> float:
        """Calculates the average confidence of all boxes for the FR15 Warning Banner."""
        if not self.boxes:
            return 1.0  # High confidence if the board is completely clean
        return sum(box.confidence for box in self.boxes) / len(self.boxes)


@dataclass
class PreprocessParams:
    """Holds the live, thread-safe parameters for the OpenCV preprocessing pipeline."""

    clahe_clip_limit: float = 2.0
    gamma: float = 1.0
    blur_threshold: float = 12.5
    auto_optimize: bool = True
    denoise: bool = True


@dataclass
class InferenceParams:
    """Holds the live parameters for the OpenVINO inference engine."""

    confidence_threshold: float = 0.50
