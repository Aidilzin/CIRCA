"""
capture_detection_screenshots.py
Headlessly runs the CIRCA inference pipeline on a few PCB images and saves
annotated output images (with bounding boxes drawn) to docs/assets/screenshots/.

Uses the correct CIRCA internal APIs:
  - core.inference_engine.InferenceEngine
  - core.tiled_inference.TiledInferenceEngine
  - core.models.InferenceParams, PreprocessParams, BoundingBox
  - core.preprocessor.apply_clahe, apply_gamma
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.inference_engine import InferenceEngine
from core.tiled_inference import TiledInferenceEngine
from core.models import InferenceParams, PreprocessParams
from core.preprocessor import apply_clahe, apply_gamma

# Output dir
OUT_DIR = ROOT / "docs" / "assets" / "screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_XML = ROOT / "models" / "yolov12_int8.xml"

# Class label → BGR colour for bounding box
CLASS_COLORS = {
    "missing_hole":         ( 50, 205,  50),  # lime green
    "insufficient_solder":  ( 30, 144, 255),  # dodger blue
    "cold_solder_joint":    (  0, 165, 255),  # amber/orange
    "excess_solder":        (  0, 215, 255),  # yellow-cyan
    "short":                (180,   0, 255),  # violet
    "open_circuit":         ( 50,  50, 255),  # red
    "mouse_bite":           (  0, 255, 180),  # teal
}
DEFAULT_COLOR = (200, 200, 200)


def draw_detections(image: np.ndarray, boxes: list) -> np.ndarray:
    """Draw bounding boxes + labels on a copy of image."""
    vis = image.copy()
    for box in boxes:
        x1 = int(box.x)
        y1 = int(box.y)
        x2 = int(box.x + box.width)
        y2 = int(box.y + box.height)
        colour = CLASS_COLORS.get(box.class_name, DEFAULT_COLOR)

        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 3)

        text = f"{box.class_name} {box.confidence:.2f}"
        font_scale = 0.65
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Label pill background
        cv2.rectangle(vis, (x1, y1 - th - 12), (x1 + tw + 8, y1), colour, -1)
        cv2.putText(vis, text, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return vis


def preprocess(img: np.ndarray, params: PreprocessParams) -> np.ndarray:
    """Apply CLAHE + gamma correction — mirrors camera_worker pipeline."""
    img = apply_clahe(img, params)
    img = apply_gamma(img, params)
    return img


def main():
    print("[capture] Loading InferenceEngine...")
    try:
        engine = InferenceEngine(str(MODEL_XML))
        tiled = TiledInferenceEngine(engine)
        print("[capture] Engine loaded OK.")
    except Exception as e:
        print(f"[capture] ERROR loading engine: {e}")
        sys.exit(1)

    prep_params = PreprocessParams(clahe_clip_limit=2.0, gamma=1.2, auto_optimize=True)
    inf_params = InferenceParams(confidence_threshold=0.10)  # low threshold = show more boxes

    # Find validation images
    candidate_dirs = [
        ROOT / "datasets" / "unified_pcb_v3" / "valid" / "images",
        ROOT / "datasets" / "unified_pcb_v3" / "test" / "images",
        ROOT / "datasets" / "unified_pcb_v3_preproc" / "valid" / "images",
        ROOT / "datasets" / "benchmark_real_pcbs",
    ]

    test_images: list[Path] = []
    for d in candidate_dirs:
        if d.exists():
            found = sorted(d.glob("*.jpg"))[:6] + sorted(d.glob("*.png"))[:6]
            test_images.extend(found)
            print(f"[capture] Found {len(found)} images in {d}")
        if len(test_images) >= 6:
            break

    if not test_images:
        print("[capture] No validation images found. Check dataset paths.")
        sys.exit(1)

    # Pick images that produce at least one detection, up to 4 saved
    saved: list[Path] = []
    for img_path in test_images:
        if len(saved) >= 4:
            break

        print(f"\n[capture] Processing: {img_path.name}")
        img = cv2.imread(str(img_path))
        if img is None:
            print("  Cannot read — skipping.")
            continue

        # Resize if very large
        h, w = img.shape[:2]
        if max(h, w) > 3000:
            scale = 3000 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        processed = preprocess(img, prep_params)

        try:
            result = tiled.run_tiled(processed, inf_params)
        except Exception as e:
            print(f"  Inference error: {e}")
            continue

        print(f"  Detections: {len(result.boxes)}")
        for b in result.boxes:
            print(f"    {b.class_name} conf={b.confidence:.3f} @ ({b.x},{b.y},{b.width},{b.height})")

        # Save all images (even zero detections — useful for "clean board" demo)
        annotated = draw_detections(img, result.boxes)  # draw on unprocessed for better colour
        out_path = OUT_DIR / f"detection_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved.append(out_path)
        print(f"  Saved: {out_path}")

    print(f"\n[capture] Done. {len(saved)} images saved to {OUT_DIR}")
    for p in saved:
        print(f"  → {p.name}")


if __name__ == "__main__":
    main()
