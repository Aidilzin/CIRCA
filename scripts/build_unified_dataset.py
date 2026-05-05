import os
import shutil
import random
import logging
from pathlib import Path
from PIL import Image
import imagehash
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
random.seed(42)

# --- MAPPINGS ---
MAPPINGS = {
    'pku_jr': {0:0, 1:1, 2:2, 3:3, 4:4, 5:5},
    'pku_rahul': {0:0, 1:1, 2:2, 3:3, 4:4, 5:5},
    'bare_pcb': {0:5, 1:0, 2:1, 3:2, 4:11, 5:10, 6:3, 7:4},
    'soldef': {0:6, 3:7, 4:8},
    'kydra': {0:9, 1:6, 2:7},
    'hue': {0:7, 2:3},
    'xnbzp': {0:6},
    's89jo': {0:9, 1:6, 2:7}
}

CLASSES = [
    "missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper",
    "excess_solder", "insufficient_solder", "solder_spike", "cold_solder_joint", "scratch", "pinhole"
]

def get_image_info(img_path):
    """Returns phash, width, height, and valid labels for an image."""
    try:
        with Image.open(img_path) as img:
            w, h = img.size
            if min(w, h) < 320:
                return None, img_path, "too_small", []
            phash_val = str(imagehash.phash(img))
    except Exception as e:
        return None, img_path, f"corrupted: {e}", []
    
    label_path = None
    if img_path.parent.name == "images":
        label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
    else:
        label_path = img_path.with_suffix(".txt")
        
    if not label_path or not label_path.exists():
        alt_label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if alt_label_path.exists():
            label_path = alt_label_path
            
    if not label_path or not label_path.exists():
        return None, img_path, "missing_label", []
        
    # --- SUBSTRING PATH MATCHING ---
    path_str = str(img_path).replace("\\", "/").lower()
    
    if "jr-mqqnk" in path_str:
        mapping, source = MAPPINGS['pku_jr'], 'pku_jr'
    elif "rahul-jhj03" in path_str:
        mapping, source = MAPPINGS['pku_rahul'], 'pku_rahul'
    elif "bare-pcb-defects" in path_str:
        mapping, source = MAPPINGS['bare_pcb'], 'bare_pcb'
    elif "soldef" in path_str: 
        mapping, source = MAPPINGS['soldef'], 'soldef'
    elif "kydra" in path_str:
        mapping, source = MAPPINGS['kydra'], 'kydra'
    elif "hue-" in path_str:
        mapping, source = MAPPINGS['hue'], 'hue'
    elif "xnbzp" in path_str:
        mapping, source = MAPPINGS['xnbzp'], 'xnbzp'
    elif "s89jo" in path_str:
        mapping, source = MAPPINGS['s89jo'], 's89jo'
    else:
        return None, img_path, "unknown_source", []

    valid_lines = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            try:
                cls_id = int(parts[0])
            except ValueError: continue
            
            if cls_id in mapping:
                valid_lines.append(f"{mapping[cls_id]} {' '.join(parts[1:])}")

    if not valid_lines:
        return None, img_path, "no_valid_defects", []
        
    return phash_val, img_path, source, valid_lines

def apply_enhancements(img_path, out_path):
    img = cv2.imread(str(img_path))
    if img is None: return False
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)
    
    cv2.imwrite(str(out_path), enhanced)
    return True

def main():
    base_dir = Path("d:/FYP/CIRCA/datasets")
    target_v2 = base_dir / "unified_pcb_v2"
    target_preproc = base_dir / "unified_pcb_v2_preproc"
    neg_dir = base_dir / "negatives_reserve"
    
    if target_v2.exists(): shutil.rmtree(target_v2)
    if target_preproc.exists(): shutil.rmtree(target_preproc)
    
    neg_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Scanning for images...")
    image_paths = []
    
    scan_dirs = [
        base_dir / "httpsuniverse.roboflow.comjr-mqqnkpcb-defects-detection-anddldataset1",
        base_dir / "httpsuniverse.roboflow.comrahul-jhj03pcb-defects-dataset",
        base_dir / "https___universe.roboflow.com_bare-pcb-defects_obj-detection-pcb-defects-yolov8",
        base_dir / "httpswww.kaggle.comdatasetsmauriziocalabresesoldef-ai-pcb-dataset-for-defect-detection",
        base_dir / "https___universe.roboflow.com_pcb-defect-detection-emmts_excessive-solder-kydra",
        base_dir / "https___universe.roboflow.com_pcb-defect-detection-emmts_hue-dbgbs-reqtv",
        base_dir / "https___universe.roboflow.com_pcb-defect-detection-emmts_solder-f8m5i-xnbzp",
        base_dir / "https___universe.roboflow.com_pcb-defect-detection-emmts_pcb-solder-defect-detection-v2-s89jo"
    ]
    
    for scan_dir in scan_dirs:
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_paths.extend(list(scan_dir.rglob(ext)))

    logger.info(f"Found {len(image_paths)} potential images.")

    with Pool(cpu_count()) as pool:
        results = pool.map(get_image_info, image_paths)

    logger.info("Deduplicating via pHash (Threshold=6). This may take a moment...")
    
    unique_data = []
    valid_results = [res for res in results if res[0] is not None]
    
    for res in results:
        if res[0] is None and res[2] == "no_valid_defects":
            shutil.copy2(res[1], neg_dir / f"neg_{res[1].name}")

    total_valid = len(valid_results)
    for idx, (h_str, path, source, valid_lines) in enumerate(valid_results):
        current_hash = imagehash.hex_to_hash(h_str)
        is_duplicate = False
        
        for unique_item in unique_data:
            unique_hash = imagehash.hex_to_hash(unique_item[0])
            if current_hash - unique_hash <= 6:
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_data.append((h_str, path, source, valid_lines))
            
        if idx > 0 and idx % 5000 == 0:
            logger.info(f"Deduplication progress: {idx}/{total_valid} items checked...")

    logger.info(f"Deduplication complete. {len(unique_data)} valid unique images remain.")

    random.shuffle(unique_data)
    total = len(unique_data)
    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)
    
    splits = {
        'train': unique_data[:train_end],
        'val': unique_data[train_end:val_end],
        'test': unique_data[val_end:]
    }

    for target in [target_v2, target_preproc]:
        for split_name in ['train', 'val', 'test']:
            (target / split_name / "images").mkdir(parents=True, exist_ok=True)
            (target / split_name / "labels").mkdir(parents=True, exist_ok=True)

    def write_pair(target_root, split_name, img_path, label_text, prefix, do_enhance=False):
        img_out = target_root / split_name / "images" / f"{prefix}{img_path.name}"
        lbl_out = target_root / split_name / "labels" / f"{img_out.stem}.txt"
        
        with open(lbl_out, 'w') as f:
            f.write("\n".join(label_text) + "\n")
            
        if do_enhance:
            apply_enhancements(img_path, img_out)
        else:
            shutil.copy2(img_path, img_out)

    logger.info("Writing files and applying enhancements...")
    
    for split_name, items in splits.items():
        for _, img_path, source, label_text in items:
            write_pair(target_v2, split_name, img_path, label_text, f"{source}_")
            write_pair(target_preproc, split_name, img_path, label_text, f"{source}_", do_enhance=True)

    yaml_data = {
        'path': '.',
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(CLASSES),
        'names': {i: name for i, name in enumerate(CLASSES)}
    }
    
    for target in [target_v2, target_preproc]:
        with open(target / "data.yaml", 'w') as f:
            yaml.dump(yaml_data, f, sort_keys=False)

    logger.info("Compilation successfully finished!")

if __name__ == "__main__":
    main()