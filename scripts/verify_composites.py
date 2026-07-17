import os
import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CompositeVerifier")

def main():
    dataset_dir = Path("d:/FYP/CIRCA/datasets/unified_pcb_v3_copypaste/train")
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    
    if not images_dir.exists() or not labels_dir.exists():
        logger.error("Dataset folders do not exist!")
        return
        
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
    
    logger.info("Found %d images and %d labels.", len(image_files), len(label_files))
    
    corrupted_images = 0
    empty_labels = 0
    out_of_bounds = 0
    total_boxes = 0
    
    widths = []
    heights = []
    classes = {}
    boxes_per_image = []
    
    for img_file in image_files[:100]: # Check first 100 images for speed
        img_path = images_dir / img_file
        img = cv2.imread(str(img_path))
        if img is None:
            corrupted_images += 1
            continue
            
        h_img, w_img = img.shape[:2]
        
        # Corresponding label file
        lbl_file = img_file.rsplit('.', 1)[0] + '.txt'
        lbl_path = labels_dir / lbl_file
        
        if not lbl_path.exists():
            empty_labels += 1
            boxes_per_image.append(0)
            continue
            
        lines = open(lbl_path).read().strip().split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        boxes_per_image.append(len(lines))
        if len(lines) == 0:
            empty_labels += 1
            
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
                
            cls_id = int(parts[0])
            classes[cls_id] = classes.get(cls_id, 0) + 1
            total_boxes += 1
            
            x, y, w, h = map(float, parts[1:5])
            
            # Check bounding box sizes relative to image
            widths.append(w)
            heights.append(h)
            
            # Check coordinates out of bounds
            if x - w/2 < 0 or x + w/2 > 1 or y - h/2 < 0 or y + h/2 > 1:
                out_of_bounds += 1

    logger.info("Evaluation Summary:")
    logger.info("  - Corrupted Images: %d", corrupted_images)
    logger.info("  - Empty Label Files: %d", empty_labels)
    logger.info("  - Out-of-Bounds Bounding Boxes: %d", out_of_bounds)
    logger.info("  - Total Bounding Boxes Evaluated: %d", total_boxes)
    if total_boxes > 0:
        logger.info("  - Avg boxes per image: %.2f", np.mean(boxes_per_image))
        logger.info("  - Avg bbox width (relative): %.4f (%.1f pixels at 640x640)", np.mean(widths), np.mean(widths)*640)
        logger.info("  - Avg bbox height (relative): %.4f (%.1f pixels at 640x640)", np.mean(heights), np.mean(heights)*640)
        logger.info("  - Class Distribution: %s", sorted(classes.items()))
        
if __name__ == "__main__":
    main()
