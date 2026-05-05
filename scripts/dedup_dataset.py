import os
import argparse
import imagehash
from PIL import Image
from pathlib import Path
import logging
from multiprocessing import Pool, cpu_count
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def get_dhash(image_path):
    """Computes dHash for a single image."""
    try:
        with Image.open(image_path) as img:
            return str(imagehash.dhash(img)), image_path
    except Exception as e:
        logger.warning(f"Could not process {image_path}: {e}")
        return None, image_path

def run_dedup(dataset_path, threshold=6, dry_run=True):
    dataset_path = Path(dataset_path)
    image_paths = list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.png")) + list(dataset_path.rglob("*.jpeg"))
    
    logger.info(f"Scanning {len(image_paths)} images in {dataset_path}...")
    
    # Use multiprocessing to speed up hashing
    with Pool(cpu_count()) as pool:
        results = pool.map(get_dhash, image_paths)
    
    hashes = defaultdict(list)
    for h, path in results:
        if h:
            hashes[h].append(path)
            
    duplicates = []
    unique_hashes = list(hashes.keys())
    
    # Check for near-duplicates using Hamming distance
    # This is O(N^2) on unique hashes, but N is usually small after exact hash grouping
    logger.info(f"Checking {len(unique_hashes)} unique hashes for near-duplicates (Hamming distance <= {threshold})...")
    
    processed_hashes = set()
    for i, h1_str in enumerate(unique_hashes):
        if h1_str in processed_hashes:
            continue
            
        h1 = imagehash.hex_to_hash(h1_str)
        group = hashes[h1_str]
        
        for j in range(i + 1, len(unique_hashes)):
            h2_str = unique_hashes[j]
            if h2_str in processed_hashes:
                continue
                
            h2 = imagehash.hex_to_hash(h2_str)
            if h1 - h2 <= threshold:
                group.extend(hashes[h2_str])
                processed_hashes.add(h2_str)
        
        if len(group) > 1:
            # We found a group of duplicates. Keep the first one, mark others for removal.
            duplicates.extend(group[1:])
            logger.info(f"Group found: keeping {group[0].name}, found {len(group)-1} duplicates.")

    if not duplicates:
        logger.info("No duplicates found.")
        return

    logger.info(f"Found {len(duplicates)} duplicate images.")
    
    if dry_run:
        logger.info("[DRY RUN] No files were deleted. Run without --dry-run to purge.")
        for d in duplicates[:10]:
            logger.info(f"[DRY RUN] Would delete: {d}")
        if len(duplicates) > 10:
            logger.info(f"... and {len(duplicates)-10} more.")
    else:
        for d in duplicates:
            try:
                # Remove image
                d.unlink()
                # Remove corresponding label if exists
                label_path = d.parent.parent / "labels" / (d.stem + ".txt")
                if label_path.exists():
                    label_path.unlink()
                logger.info(f"Deleted: {d}")
            except Exception as e:
                logger.error(f"Failed to delete {d}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate dataset using perceptual hashing (dHash).")
    parser.add_argument("--path", type=str, default="datasets/unified_pcb_v2", help="Path to the dataset root.")
    parser.add_argument("--threshold", type=int, default=6, help="Hamming distance threshold for duplicates.")
    parser.add_argument("--no-dry-run", action="store_true", help="Actually delete the files.")
    
    args = parser.parse_args()
    run_dedup(args.path, args.threshold, not args.no_dry_run)
