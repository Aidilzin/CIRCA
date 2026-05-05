import os
from pathlib import Path
from collections import Counter

def count_dataset_instances():
    base_dir = Path("d:/FYP/CIRCA/datasets")
    
    # List of all your raw datasets (you can add/remove from here as needed)
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
    
    print("=== Raw Dataset Instance Counter ===")
    print("Scanning directories...\n")
    
    total_datasets_scanned = 0
    
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            print(f"[WARNING] Directory not found, skipping: {scan_dir.name}")
            continue
            
        total_datasets_scanned += 1
        class_counts = Counter()
        file_count = 0
        
        # Recursively find all .txt files
        for txt_file in scan_dir.rglob("*.txt"):
            # Skip readme or classes metadata files
            if txt_file.name.lower() in ["classes.txt", "readme.txt", "readme.dataset.txt"]:
                continue
                
            try:
                with open(txt_file, 'r') as f:
                    file_has_labels = False
                    for line in f:
                        parts = line.strip().split()
                        # Check if line is valid and starts with an integer (the class ID)
                        if parts and parts[0].isdigit():
                            cls_id = int(parts[0])
                            class_counts[cls_id] += 1
                            file_has_labels = True
                            
                    if file_has_labels:
                        file_count += 1
            except Exception as e:
                print(f"  [ERROR] Could not read {txt_file.name}: {e}")
        
        # Print the report for this dataset
        print("-" * 60)
        # Simplify the folder name for cleaner output
        short_name = scan_dir.name.split('_')[-1] if '_' in scan_dir.name else scan_dir.name[:30] + "..."
        print(f"Dataset: {short_name}")
        print(f"Total labeled files parsed: {file_count}")
        
        if not class_counts:
            print("  No YOLO formatted labels found.")
        else:
            print("  Raw Class Breakdown:")
            # Sort by class ID for neatness
            for cls_id in sorted(class_counts.keys()):
                print(f"    Raw ID {cls_id}: {class_counts[cls_id]} instances")
    
    print("-" * 60)
    print(f"\nScan complete. Processed {total_datasets_scanned} directories.")

if __name__ == "__main__":
    count_dataset_instances()