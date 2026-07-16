#!/usr/bin/env python
"""
CIRCA RunPod Verification & Diagnostic Tool
===========================================
This script runs a comprehensive sanity check on the RunPod cloud instance to
ensure the environment is correctly configured, all training dependencies are
installed, CUDA is active with the GPU, and the uploaded dataset is intact.
"""

import os
import sys
from pathlib import Path

# Try importing common standard libraries
try:
    import yaml
    import torch
    import cv2
    import numpy as np
except ImportError as e:
    print(f"\n[!] WARNING: Some primary libraries are missing: {e}")
    print("Please run: pip install -r requirements_runpod.txt\n")

def print_banner(title):
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60)

def check_system():
    print_banner("System Diagnostics")
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"Operating System: {sys.platform}")
    print(f"Working Dir     : {os.getcwd()}")
    
    # Check GPU
    cuda_available = torch.cuda.is_available() if 'torch' in sys.modules else False
    print(f"CUDA Available  : {'YES' if cuda_available else 'NO'}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"GPU Count       : {device_count}")
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU [{i}]        : {name} ({vram:.1f} GB VRAM)")
    else:
        print("[!] WARNING: PyTorch cannot find a CUDA-enabled GPU. Training will run on CPU!")

def check_dependencies():
    print_banner("Dependency Health Check")
    deps = {
        "torch": "PyTorch",
        "torchvision": "Torchvision",
        "ultralytics": "Ultralytics YOLO",
        "wandb": "Weights & Biases",
        "cv2": "OpenCV Python",
        "numpy": "NumPy",
        "yaml": "PyYAML",
        "imagehash": "ImageHash",
        "PIL": "Pillow"
    }
    
    missing = []
    for module_name, friendly_name in deps.items():
        try:
            if module_name == "PIL":
                import PIL
                ver = PIL.__version__
            elif module_name == "cv2":
                import cv2
                ver = cv2.__version__
            elif module_name == "yaml":
                import yaml
                ver = yaml.__version__
            else:
                mod = __import__(module_name)
                ver = getattr(mod, "__version__", "unknown")
            print(f"  [+] {friendly_name:<18}: Installed ({ver})")
        except ImportError:
            print(f"  [X] {friendly_name:<18}: MISSING")
            missing.append(module_name)
            
    if missing:
        print("\n[!] CRITICAL: Missing dependencies! Run:")
        print("    pip install -r requirements_runpod.txt")
        return False
    else:
        print("\n[+] All core dependencies are successfully installed.")
        return True

def verify_dataset(dataset_root="datasets/unified_pcb_v3"):
    print_banner("Dataset Health Check")
    root_path = Path(dataset_root)
    
    if not root_path.exists():
        print(f"[!] CRITICAL: Dataset root folder not found at: {root_path.resolve()}")
        print("Please check your unzip path and directory structure.")
        return False
        
    data_yaml_path = root_path / "data.yaml"
    if not data_yaml_path.exists():
        print(f"[!] CRITICAL: data.yaml is missing from {root_path}")
        return False
        
    print(f"Dataset Folder  : {root_path.resolve()}")
    print(f"data.yaml Path  : {data_yaml_path.resolve()}")
    
    # Parse data.yaml
    try:
        with open(data_yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        print("\n--- data.yaml Contents ---")
        for k, v in cfg.items():
            if k == "names":
                print("names:")
                for cid, name in v.items():
                    print(f"  {cid}: {name}")
            else:
                print(f"{k}: {v}")
        print("--------------------------\n")
    except Exception as e:
        print(f"[!] ERROR: Failed to parse data.yaml: {e}")
        return False

    # Check directories and file counts
    splits = ["train", "valid", "test"]
    all_ok = True
    
    for split in splits:
        split_dir = root_path / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        
        if not split_dir.exists():
            print(f"  [X] Split directory '{split}' is MISSING")
            all_ok = False
            continue
            
        if not img_dir.exists():
            print(f"  [X] Images directory '{img_dir}' is MISSING")
            all_ok = False
            
        if not lbl_dir.exists():
            print(f"  [X] Labels directory '{lbl_dir}' is MISSING")
            all_ok = False
            
        if img_dir.exists() and lbl_dir.exists():
            img_files = list(img_dir.glob("*"))
            img_exts = {p.suffix.lower() for p in img_files if p.is_file()}
            images = [p for p in img_files if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
            labels = list(lbl_dir.glob("*.txt"))
            
            print(f"  [+] Split '{split:<5}': {len(images):>5} images | {len(labels):>5} label files")
            
            # Warn if mismatch
            if len(images) != len(labels):
                print(f"      [!] WARNING: Image and label counts do not match ({len(images)} vs {len(labels)})")
                # Count files with no labels
                lbl_names = {p.stem for p in labels}
                missing_lbls = [p.name for p in images if p.stem not in lbl_names]
                if missing_lbls:
                    print(f"      [!] First 5 images missing label files: {missing_lbls[:5]}")
                    
            if not images:
                print(f"      [!] WARNING: No images found in split '{split}'")
                all_ok = False

    return all_ok

def print_launch_playbook():
    print_banner("Launch Commands Playbook")
    print("Use the following commands to launch experiments inside your RunPod terminal.\n")
    print("NOTE: If you ran setup-runpod-environment.sh, dataset prep (oversampling + preproc) is already done.\n")

    print("--- Step 0: Weights & Biases Authentication (First Step) ---")
    print("  wandb login <YOUR_WANDB_API_KEY>")
    print("*(Skipping login auto-disables W&B to prevent terminal hang)*\n")

    print("--- Phase 1: Vanilla Baseline (100 Epochs, Raw) ---")
    print("Ablation control -- raw unified_pcb_v3 (oversampled, no CLAHE).")
    print("  python train_engine.py --mode train --variant s --id 001 --desc Baseline_Vanilla \\")
    print("    --epochs 100 --batch 24 --data datasets/unified_pcb_v3/data.yaml\n")

    print("--- Phase 2: CIRCA Preprocessing Baseline (100 Epochs, CLAHE+Gamma) ---")
    print("Uses unified_pcb_v3_preproc (already created + oversampled by setup.sh).")
    print("  python train_engine.py --mode train --variant s --id 002 --desc Baseline_CIRCA \\")
    print("    --epochs 100 --batch 24 --data datasets/unified_pcb_v3_preproc/data.yaml\n")

    print("--- Phase 3: Hyperparameter Tuning / HPO (50 Iterations x 50 Epochs) ---")
    print("Runs on unified_pcb_v3_preproc (preprocessed + oversampled -- already done by setup.sh).")
    print("DO NOT use --preproc flag here.")
    print("  python train_engine.py --mode tune --variant s --id 003 --desc HPO_7class \\")
    print("    --epochs 50 --iterations 50 --fraction 0.5 --batch 24 \\")
    print("    --data datasets/unified_pcb_v3_preproc/data.yaml\n")

    print("=" * 60)

if __name__ == "__main__":
    print_banner("CIRCA RUNPOD SETUP DIAGNOSTIC")
    
    check_system()
    deps_ok = check_dependencies()
    dataset_ok = verify_dataset()
    
    if deps_ok and dataset_ok:
        print_banner("DIAGNOSTIC STATUS: READY")
        print("[+] SUCCESS: Your RunPod environment and dataset are fully validated!")
        print("[+] You are ready to start training.")
    else:
        print_banner("DIAGNOSTIC STATUS: ISSUES FOUND")
        print("[!] WARNING: Some issues were detected. Please review the details above.")
        
    print_launch_playbook()
