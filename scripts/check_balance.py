import os
from pathlib import Path
from collections import Counter
import yaml

def check_balance(dataset_path):
    dataset_path = Path(dataset_path)
    
    splits = {
        'train': dataset_path / 'train' / 'labels',
        'val': dataset_path / 'val' / 'labels',
        'test': dataset_path / 'test' / 'labels'
    }
    
    class_names = [
        "missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper",
        "excess_solder", "insufficient_solder", "solder_spike", "cold_solder_joint", "scratch", "pinhole"
    ]
    
    total_images = 0
    split_counts = {}
    class_counts = Counter()
    
    for split_name, labels_dir in splits.items():
        if not labels_dir.exists():
            print(f"Directory not found: {labels_dir}")
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
                        class_id = int(parts[0])
                        class_counts[class_id] += 1

    print("=== Dataset Split Distribution ===")
    if total_images == 0:
        print("No images found.")
        return
        
    for split_name, count in split_counts.items():
        percentage = (count / total_images) * 100
        print(f"{split_name.capitalize()}: {count} images ({percentage:.1f}%)")
        
    print(f"\nTotal Images: {total_images}")
    
    print("\n=== Class Distribution ===")
    for i in range(12):
        count = class_counts.get(i, 0)
        print(f"Class {i:2d} ({class_names[i]:<30}): {count} instances")
        
if __name__ == "__main__":
    check_balance("datasets/unified_pcb_v2")
