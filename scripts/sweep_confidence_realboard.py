import os
import sys
import time
import logging
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.models import BoundingBox, DetectionResult, InferenceParams
from core.tiled_inference import TiledInferenceEngine
from scripts.compare_tuned_vs_untuned import PyTorchYOLOEngine, match_boxes, load_ground_truth

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("sweep_confidence_realboard")

def evaluate_model_at_thresholds_optimized(
    model_path: str, 
    image_paths: List[str], 
    thresholds: List[float]
) -> Dict[float, Dict[str, Any]]:
    # Initialize engine with baseline lowest conf (0.01) to fetch all candidates
    engine = PyTorchYOLOEngine(model_path)
    tiled_engine = TiledInferenceEngine(engine)
    base_params = InferenceParams(confidence_threshold=0.01)
    
    # 1. Run inference once per image and gather detections + ground truth
    logger.info("  Running baseline inference at conf=0.01 to cache candidate boxes...")
    inferences = []
    total_gt = 0
    
    for i, path in enumerate(image_paths):
        raw_image = cv2.imread(path)
        if raw_image is None or raw_image.size == 0:
            continue
        h, w = raw_image.shape[:2]
        gt_boxes = load_ground_truth(path, w, h)
        total_gt += len(gt_boxes)
        
        # Run inference once
        res = tiled_engine.run_tiled(raw_image, base_params)
        inferences.append((gt_boxes, res.boxes))
        
    # 2. Perform the sweep in Python by filtering the cached candidate boxes
    results = {}
    for conf in thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for gt_boxes, candidate_boxes in inferences:
            # Filter boxes locally
            filtered_boxes = [box for box in candidate_boxes if box.confidence >= conf]
            
            matched_gt, _, unmatched_det = match_boxes(gt_boxes, filtered_boxes, iou_threshold=0.3)
            tp = len(matched_gt)
            fp = len(unmatched_det)
            fn = len(gt_boxes) - tp
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / total_gt if total_gt > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[conf] = {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    return results

def main():
    images_dir = os.path.join(PROJECT_ROOT, "datasets", "pcb_evaluation_images")
    if not os.path.exists(images_dir):
        logger.error(f"pcb_evaluation_images directory not found: {images_dir}")
        sys.exit(1)
        
    image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_paths:
        logger.error("No evaluation images found.")
        sys.exit(1)
        
    logger.info(f"Found {len(image_paths)} images for real-board evaluation.")
    
    models = {
        "Nano Untuned": "runs/detect/CIRCA_V12N_004_TRAIN_Phase4_Nano/weights/best.pt",
        "Nano Fine-Tuned (CP)": "runs/detect/CIRCA_V12N_007_TRAIN_copypaste_exhibition/weights/best.pt",
        "Small Untuned": "runs/detect/CIRCA_V12S_005_TRAIN_Phase4_Small/weights/best.pt",
        "Small Fine-Tuned (CP)": "runs/detect/CIRCA_V12S_008_TRAIN_copypaste_small/weights/best.pt",
        "Medium Untuned": "runs/detect/CIRCA_V12M_006_TRAIN_Phase4_Medium/weights/best.pt",
        "Medium Fine-Tuned (CP)": "runs/detect/CIRCA_V12M_009_TRAIN_copypaste_medium/weights/best.pt",
    }
    
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    sweep_data = {}
    
    for name, path in models.items():
        full_path = os.path.join(PROJECT_ROOT, path)
        if not os.path.exists(full_path):
            logger.warning(f"Weights file not found for {name} at: {full_path}")
            continue
            
        logger.info(f"Starting sweep for model: {name}")
        start_time = time.time()
        sweep_data[name] = evaluate_model_at_thresholds_optimized(full_path, image_paths, thresholds)
        logger.info(f"Finished sweep for model {name} in {time.time() - start_time:.1f}s")
        
    # Generate report
    report_path = os.path.join(PROJECT_ROOT, "docs", "realboard_confidence_sweep.md")
    logger.info(f"Writing sweep report to: {report_path}...")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# CIRCA Real-Board Confidence Threshold Sweep Report\n\n")
        f.write(f"> **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"> **Dataset:** 50 Staged Real-Board PCB Images (146 ground-truth defects)\n\n")
        
        f.write("## 1. Optimal Configurations (Highest F1-Score for each model)\n\n")
        f.write("| Model Variant | Best Conf | TP | FP | FN | Precision | Recall | Max F1-Score |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
        for name in models.keys():
            if name not in sweep_data:
                continue
            model_sweep = sweep_data[name]
            best_conf = max(model_sweep.keys(), key=lambda conf: model_sweep[conf]["f1"])
            r = model_sweep[best_conf]
            f.write(f"| **{name}** | {best_conf:.2f} | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision'] * 100.0:.2f}% | {r['recall'] * 100.0:.2f}% | **{r['f1'] * 100.0:.2f}%** |\n")
            
        f.write("\n## 2. Full Sweep Details\n\n")
        for name in models.keys():
            if name not in sweep_data:
                continue
            f.write(f"### {name}\n\n")
            f.write("| Conf Threshold | TP | FP | FN | Precision | Recall | F1-Score |\n")
            f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for conf in thresholds:
                r = sweep_data[name][conf]
                f.write(f"| {conf:.2f} | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision'] * 100.0:.2f}% | {r['recall'] * 100.0:.2f}% | {r['f1'] * 100.0:.2f}% |\n")
            f.write("\n")
            
    logger.info("Confidence sweep complete. Report saved.")

if __name__ == "__main__":
    main()
