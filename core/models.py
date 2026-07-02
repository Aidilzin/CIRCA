from dataclasses import dataclass, field


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
