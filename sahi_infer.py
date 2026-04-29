"""
sahi_infer.py — CIRCA SAHI Sliced Inference
============================================
Recovers small-defect detection precision when the model is trained at imgsz=640.

At inference, the full-resolution PCB image is sliced into overlapping 640×640 tiles.
Each tile is run through the detector independently, then results are merged and
de-duplicated with NMS. This gives effectively 2–4× the resolution sensitivity
of a single global-pass inference at the same imgsz.

Install dependency:
    pip install sahi

Usage:
    python sahi_infer.py --model runs/detect/<EXPERIMENT>/weights/best.pt --source path/to/image.jpg
    python sahi_infer.py --model runs/detect/<EXPERIMENT>/weights/best.pt --source path/to/folder/
"""

import argparse
import os
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="CIRCA SAHI Sliced Inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained YOLO weights (e.g. runs/detect/.../weights/best.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to a single image or a directory of images",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=640,
        help="Tile size in pixels — should match the imgsz used during training (default: 640)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        help="Fractional overlap between tiles — 0.2 prevents missed edge defects (default: 0.2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Inference device: 'cuda:0' or 'cpu' (default: cuda:0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sahi_output",
        help="Directory to save annotated output images (default: sahi_output)",
    )
    return parser.parse_args()


def collect_images(source: str) -> list[Path]:
    """Return a list of image paths from a file or directory."""
    src = Path(source)
    if src.is_file():
        return [src]
    if src.is_dir():
        return [p for p in src.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    raise FileNotFoundError(f"Source not found: {source}")


def run_inference(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[SAHI] Loading model: {args.model}")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.model,
        confidence_threshold=args.conf,
        device=args.device,
    )

    images = collect_images(args.source)
    print(f"[SAHI] Processing {len(images)} image(s) | "
          f"tile={args.slice_size}px | overlap={args.overlap:.0%} | conf={args.conf}")

    for img_path in images:
        print(f"  → {img_path.name}", end=" ... ", flush=True)

        result = get_sliced_prediction(
            str(img_path),
            detection_model,
            slice_height=args.slice_size,
            slice_width=args.slice_size,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
        )

        result.export_visuals(
            export_dir=args.output_dir,
            file_name=img_path.stem,
        )

        n = len(result.object_prediction_list)
        print(f"{n} defect(s) found → {args.output_dir}/{img_path.stem}.png")

    print(f"\n[SAHI] Done. Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
