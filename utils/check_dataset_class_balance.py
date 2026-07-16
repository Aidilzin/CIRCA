"""
utils/check_dataset_class_balance.py
======================================
Reusable utility module to compute and return detailed split and class distributions
across CIRCA datasets. Parses YOLO annotation text files to build structured telemetry.

Can be imported as a Python module or executed directly from the command line.
"""

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("circa.utils.dataset_balance")

DEFAULT_CLASS_NAMES: List[str] = [
    "missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper",
    "excess_solder", "insufficient_solder", "solder_spike", "cold_solder_joint", "scratch", "pinhole"
]


def get_dataset_class_distribution(dataset_path: str, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Computes split distribution and class annotation frequency for a YOLO format dataset.

    Args:
        dataset_path: Root folder of the dataset containing train/val/test subdirectories.
        class_names: List of defect category names ordered by class ID. Falls back to DEFAULT_CLASS_NAMES.

    Returns:
        Dict[str, Any] containing distribution metrics and status details.
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    result = {
        "success": False,
        "error": None,
        "total_images": 0,
        "total_annotations": 0,
        "split_distribution": {},
        "class_distribution": {}
    }

    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        result["error"] = f"Dataset path does not exist or is not a directory: {dataset_path}"
        return result

    splits = {
        'train': dataset_dir / 'train' / 'labels',
        'val': dataset_dir / 'val' / 'labels',
        'test': dataset_dir / 'test' / 'labels'
    }

    total_images = 0
    split_counts = {}
    class_counts = Counter()

    try:
        # 1. Parse all splits
        for split_name, labels_dir in splits.items():
            if not labels_dir.exists():
                logger.warning("Split labels directory not found: %s", labels_dir)
                split_counts[split_name] = 0
                continue
                
            files = list(labels_dir.glob('*.txt'))
            split_counts[split_name] = len(files)
            total_images += len(files)
            
            for file_path in files:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            try:
                                class_id = int(parts[0])
                                class_counts[class_id] += 1
                            except ValueError:
                                continue

        # 2. Build structured splits metadata
        result["total_images"] = total_images
        for split_name, count in split_counts.items():
            percentage = (count / total_images * 100.0) if total_images > 0 else 0.0
            result["split_distribution"][split_name] = {
                "count": count,
                "percentage": round(percentage, 2)
            }

        # 3. Build structured class distribution metadata
        total_annotations = sum(class_counts.values())
        result["total_annotations"] = total_annotations

        for i, name in enumerate(class_names):
            count = class_counts.get(i, 0)
            percentage = (count / total_annotations * 100.0) if total_annotations > 0 else 0.0
            result["class_distribution"][i] = {
                "name": name,
                "count": count,
                "percentage": round(percentage, 2)
            }

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        logger.error("Failed to calculate dataset balance", exc_info=True)

    return result


if __name__ == "__main__":
    # Standalone CLI Guard
    parser = argparse.ArgumentParser(description="Analyze split balance and label distributions of CIRCA datasets.")
    parser.add_argument(
        "dataset_path", nargs="?",
        default="datasets/unified_pcb_v2_preproc",
        help="Path to YOLO dataset root (default: datasets/unified_pcb_v2_preproc)"
    )
    args = parser.parse_args()

    # Configure root logging for standalone console feedback
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    print(f"[*] Analyzing dataset at: {args.dataset_path}...")
    res = get_dataset_class_distribution(args.dataset_path)

    if res["success"]:
        print(f"\n=== Dataset Split Distribution ===")
        print(f"Total Images: {res['total_images']}")
        for name, metrics in res["split_distribution"].items():
            print(f"  - {name.capitalize():<8}: {metrics['count']:<5} images ({metrics['percentage']:.1f}%)")

        print(f"\n=== Class Annotation Distribution ===")
        print(f"Total Labels: {res['total_annotations']}")
        for cid, metrics in res["class_distribution"].items():
            print(f"  - Class {cid:02d} ({metrics['name']:<22}): {metrics['count']:<5} instances ({metrics['percentage']:.1f}%)")
    else:
        print(f"[X] FAILED: {res['error']}", file=sys.stderr)
        sys.exit(1)
