#!/usr/bin/env python
"""
scripts/generate_copypaste_data.py
----------------------------------
Exhibition Prep: Copy-Paste Augmentation Pipeline.

Extracts annotated defect crops from close-up dataset sources and pastes them
onto full-board PCB backgrounds (such as MiracleFactory motherboard photos,
stock PCB flat lays, or existing PXL images) to create composite training images.

This addresses the domain gap by presenting defect patterns on realistic full-board
backgrounds across all 7 target classes.
"""

import argparse
import logging
import os
import random
import shutil
from typing import List, Dict, Tuple

import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("CopyPasteGenerator")

# Class ID to name mapping
CLASS_NAMES = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "excess_solder",
    5: "insufficient_solder",
    6: "cold_solder_joint"
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract defect crops and paste them onto full-board backgrounds."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets/unified_pcb_v3",
        help="Path to the original unified dataset."
    )
    parser.add_argument(
        "--backgrounds-dir",
        type=str,
        default="",
        help="Path to a directory containing downloaded full-board PCB images (optional)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/unified_pcb_v3_copypaste",
        help="Directory to save the new copy-paste augmented dataset."
    )
    parser.add_argument(
        "--num-composites",
        type=int,
        default=10,
        help="Number of composite images to generate per background."
    )
    parser.add_argument(
        "--min-defects",
        type=int,
        default=2,
        help="Minimum number of defects to paste per composite image."
    )
    parser.add_argument(
        "--max-defects",
        type=int,
        default=6,
        help="Maximum number of defects to paste per composite image."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    return parser.parse_args()


def compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Union (IoU) of two YOLO format boxes (x_c, y_c, w, h)."""
    # Convert to x1, y1, x2, y2 coordinates
    b1_x1, b1_x2 = box1[0] - box1[2] / 2.0, box1[0] + box1[2] / 2.0
    b1_y1, b1_y2 = box1[1] - box1[3] / 2.0, box1[1] + box1[3] / 2.0
    b2_x1, b2_x2 = box2[0] - box2[2] / 2.0, box2[0] + box2[2] / 2.0
    b2_y1, b2_y2 = box2[1] - box2[3] / 2.0, box2[1] + box2[3] / 2.0

    # Intersection rectangle
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    # Union
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def extract_defect_crops(
    train_images_dir: str, train_labels_dir: str
) -> Dict[int, List[np.ndarray]]:
    """
    Extract defect bounding box crops from all training images that are close-up crops.
    """
    logger.info("Extracting defect crops from training set...")
    crops: Dict[int, List[np.ndarray]] = {cls_id: [] for cls_id in CLASS_NAMES}

    image_files = [
        f for f in os.listdir(train_images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    total_images_processed = 0
    total_crops_extracted = 0

    for img_name in image_files:
        # Skip backgrounds to avoid circular crops (only crop from close-ups)
        if img_name.startswith("PXL_"):
            continue

        img_path = os.path.join(train_images_dir, img_name)
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(train_labels_dir, lbl_name)

        if not os.path.exists(lbl_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        total_images_processed += 1

        with open(lbl_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            try:
                cls_id = int(parts[0])
                x_c, y_c, w_n, h_n = map(float, parts[1:5])
            except ValueError:
                continue

            if cls_id not in CLASS_NAMES:
                continue

            # Convert YOLO normalized coords to pixel boundaries
            px_x_c = x_c * w
            px_y_c = y_c * h
            px_w = w_n * w
            px_h = h_n * h

            # Calculate box bounds with a tiny 5% padding to capture edge context
            pad_w = int(px_w * 0.05)
            pad_h = int(px_h * 0.05)

            x1 = max(0, int(px_x_c - px_w / 2.0 - pad_w))
            y1 = max(0, int(px_y_c - px_h / 2.0 - pad_h))
            x2 = min(w, int(px_x_c + px_w / 2.0 + pad_w))
            y2 = min(h, int(px_y_c + px_h / 2.0 + pad_h))

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crops[cls_id].append(crop)
            total_crops_extracted += 1

    logger.info(
        "Extracted %d crops from %d close-up images.",
        total_crops_extracted, total_images_processed
    )
    for cls_id, cls_crops in crops.items():
        logger.info("  Class %d (%s): %d crops", cls_id, CLASS_NAMES[cls_id], len(cls_crops))

    return crops


def augment_crop(crop: np.ndarray) -> np.ndarray:
    """Apply minor visual augmentations to a crop to improve generalization."""
    h, w = crop.shape[:2]

    # 1. Random rotation (±15 degrees)
    angle = random.uniform(-15.0, 15.0)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    crop = cv2.warpAffine(crop, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 2. Random brightness jitter (±10%)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    brightness_factor = random.uniform(0.9, 1.1)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
    hsv = hsv.astype(np.uint8)
    crop = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 3. Slight blur
    if random.random() < 0.3:
        crop = cv2.GaussianBlur(crop, (3, 3), 0)

    return crop


def feather_mask(crop_shape: Tuple[int, int], border_px: int = 3) -> np.ndarray:
    """Create a 2D float32 mask with feathered boundaries to blend edges smoothly."""
    h, w = crop_shape
    mask = np.ones((h, w), dtype=np.float32)

    if h > 2 * border_px and w > 2 * border_px:
        mask[:border_px, :] = 0.0
        mask[-border_px:, :] = 0.0
        mask[:, :border_px] = 0.0
        mask[:, -border_px:] = 0.0
        kernel_size = 2 * border_px + 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    return mask


def generate_composites(
    crops: Dict[int, List[np.ndarray]],
    background_paths: List[str],
    output_images_dir: str,
    output_labels_dir: str,
    num_composites: int,
    min_defects: int,
    max_defects: int
) -> int:
    """
    Generate composite training images by pasting crops onto backgrounds.
    """
    logger.info("Generating composite full-board images...")

    # Exclude class 0 (missing_hole) from copy-paste as requested by the user.
    available_classes = [c for c, l in crops.items() if len(l) > 0 and c != 0]
    if not available_classes:
        logger.error("No crops available for copy-paste augmentation!")
        return 0

    class_weights = {}
    for c in available_classes:
        if c in [0, 1, 2, 6]:
            class_weights[c] = 4.0
        elif c == 3:
            class_weights[c] = 2.0
        else:
            class_weights[c] = 1.0

    weight_sum = sum(class_weights.values())
    classes_list = list(class_weights.keys())
    weights_list = [w / weight_sum for w in class_weights.values()]

    composites_count = 0

    for bg_idx, bg_path in enumerate(background_paths):
        bg_name_raw = os.path.basename(bg_path)
        bg_base = os.path.splitext(bg_name_raw)[0]
        bg_img_raw = cv2.imread(bg_path)

        if bg_img_raw is None:
            logger.warning("Could not read background image: %s", bg_path)
            continue

        bg_h, bg_w = bg_img_raw.shape[:2]

        for comp_idx in range(num_composites):
            bg_img = bg_img_raw.copy()
            num_defects = random.randint(min_defects, max_defects)
            composite_labels: List[Tuple[int, float, float, float, float]] = []

            for _ in range(num_defects):
                cls_id = random.choices(classes_list, weights=weights_list, k=1)[0]
                crop = random.choice(crops[cls_id])

                crop_scale = random.uniform(0.03, 0.08)
                target_crop_w = int(bg_w * crop_scale)
                orig_crop_h, orig_crop_w = crop.shape[:2]
                target_crop_h = int(orig_crop_h * (target_crop_w / orig_crop_w))

                if target_crop_w < 8 or target_crop_h < 8:
                    continue

                resized_crop = cv2.resize(
                    crop, (target_crop_w, target_crop_h), interpolation=cv2.INTER_LINEAR
                )

                augmented_crop = augment_crop(resized_crop)
                mask = feather_mask((target_crop_h, target_crop_w), border_px=3)

                placed = False
                for _ in range(30):
                    margin_x = int(bg_w * 0.05)
                    margin_y = int(bg_h * 0.05)
                    
                    max_x = bg_w - target_crop_w - margin_x
                    max_y = bg_h - target_crop_h - margin_y

                    if max_x <= margin_x or max_y <= margin_y:
                        break

                    x_start = random.randint(margin_x, max_x)
                    y_start = random.randint(margin_y, max_y)

                    c_x_c = (x_start + target_crop_w / 2.0) / bg_w
                    c_y_c = (y_start + target_crop_h / 2.0) / bg_h
                    c_w_n = target_crop_w / bg_w
                    c_h_n = target_crop_h / bg_h
                    candidate_box = (c_x_c, c_y_c, c_w_n, c_h_n)

                    overlap = False
                    for _, prev_x, prev_y, prev_w, prev_h in composite_labels:
                        iou = compute_iou(candidate_box, (prev_x, prev_y, prev_w, prev_h))
                        if iou > 0.05:
                            overlap = True
                            break

                    if not overlap:
                        roi = bg_img[y_start:y_start + target_crop_h, x_start:x_start + target_crop_w]
                        mask_3d = np.expand_dims(mask, axis=2)
                        blended = augmented_crop * mask_3d + roi * (1.0 - mask_3d)
                        bg_img[y_start:y_start + target_crop_h, x_start:x_start + target_crop_w] = blended.astype(np.uint8)

                        composite_labels.append((cls_id, c_x_c, c_y_c, c_w_n, c_h_n))
                        placed = True
                        break

            if composite_labels:
                out_name = f"composite_{bg_base}_{comp_idx}"
                img_out_path = os.path.join(output_images_dir, f"{out_name}.jpg")
                lbl_out_path = os.path.join(output_labels_dir, f"{out_name}.txt")

                cv2.imwrite(img_out_path, bg_img)

                with open(lbl_out_path, "w") as f_lbl:
                    for label in composite_labels:
                        f_lbl.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

                composites_count += 1

        if (bg_idx + 1) % 5 == 0 or (bg_idx + 1) == len(background_paths):
            logger.info("Processed %d/%d background images...", bg_idx + 1, len(background_paths))

    return composites_count


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_images_dir = os.path.join(args.dataset_dir, "train", "images")
    train_labels_dir = os.path.join(args.dataset_dir, "train", "labels")

    if not os.path.exists(train_images_dir) or not os.path.exists(train_labels_dir):
        logger.error("Dataset train folders not found. Checked: %s", train_images_dir)
        return

    crops = extract_defect_crops(train_images_dir, train_labels_dir)

    background_paths: List[str] = []
    if args.backgrounds_dir and os.path.exists(args.backgrounds_dir):
        logger.info("Scanning backgrounds from: %s", args.backgrounds_dir)
        background_paths = [
            os.path.join(args.backgrounds_dir, f) for f in os.listdir(args.backgrounds_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    else:
        logger.info("No backgrounds directory specified. Scanning dataset for PXL images...")
        background_paths = [
            os.path.join(train_images_dir, f) for f in os.listdir(train_images_dir)
            if f.startswith("PXL_") and f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    background_paths = sorted(list(set(background_paths)))

    if not background_paths:
        logger.error("No background images found. Aborting.")
        return

    logger.info("Found %d unique background images.", len(background_paths))
    logger.info("Preparing output directory: %s", args.output_dir)
    
    if os.path.exists(args.output_dir):
        logger.info("Output directory already exists. Removing to rebuild cleanly...")
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir)
    shutil.copytree(os.path.join(args.dataset_dir, "valid"), os.path.join(args.output_dir, "valid"))
    shutil.copytree(os.path.join(args.dataset_dir, "test"), os.path.join(args.output_dir, "test"))
    shutil.copytree(os.path.join(args.dataset_dir, "train"), os.path.join(args.output_dir, "train"))
    shutil.copy2(os.path.join(args.dataset_dir, "data.yaml"), os.path.join(args.output_dir, "data.yaml"))

    data_yaml_path = os.path.join(args.output_dir, "data.yaml")
    with open(data_yaml_path, "r") as f_yaml:
        yaml_lines = f_yaml.readlines()
    with open(data_yaml_path, "w") as f_yaml:
        for line in yaml_lines:
            if line.strip().startswith("path:"):
                f_yaml.write(f"path: {args.output_dir}\n")
            else:
                f_yaml.write(line)

    output_images_dir = os.path.join(args.output_dir, "train", "images")
    output_labels_dir = os.path.join(args.output_dir, "train", "labels")

    composites_created = generate_composites(
        crops=crops,
        background_paths=background_paths,
        output_images_dir=output_images_dir,
        output_labels_dir=output_labels_dir,
        num_composites=args.num_composites,
        min_defects=args.min_defects,
        max_defects=args.max_defects
    )

    logger.info(
        "Augmentation completed successfully! Generated %d composite images inside '%s'.",
        composites_created, args.output_dir
    )


if __name__ == "__main__":
    main()
