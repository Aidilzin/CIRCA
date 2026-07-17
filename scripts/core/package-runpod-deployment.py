#!/usr/bin/env python
"""
CIRCA Streamlined RunPod Upload Package Builder
================================================
This script packages only the core dataset (unified_pcb_v3) and training scripts.
To save bandwidth, it compresses and resizes the close-up training images to 640x640
on the fly during ZIP packaging, reducing the size from 830MB to ~150MB.

Background motherboard images will be downloaded directly on the RunPod instance.
"""

import os
import sys
import zipfile
import cv2
from pathlib import Path

def print_banner(title):
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60)

def main():
    print_banner("CIRCA Streamlined RunPod Package Builder")
    
    root_path = Path("d:/FYP/CIRCA").resolve()
    zip_path = root_path / "CIRCA_runpod.zip"
    
    dataset_source = root_path / "datasets" / "unified_pcb_v3"
    train_engine_source = root_path / "training" / "train_engine.py"
    reqs_source = root_path / "requirements_runpod.txt"
    setup_py_source = root_path / "scripts" / "core" / "setup-runpod-environment.py"
    setup_sh_source = root_path / "scripts" / "core" / "setup-runpod-environment.sh"
    
    # Validations
    targets = [
        (dataset_source, "Dataset directory"),
        (train_engine_source, "Training engine script"),
        (reqs_source, "Streamlined requirements file"),
        (setup_py_source, "RunPod verification script"),
        (setup_sh_source, "RunPod setup shell script")
    ]
    
    missing = False
    for path, desc in targets:
        if not path.exists():
            print(f"  [X] ERROR: {desc} not found at {path}")
            missing = True
            
    if missing:
        print("\n[!] CRITICAL: Cannot package. Some required files are missing.\n")
        sys.exit(1)
        
    if zip_path.exists():
        zip_path.unlink()
        
    print(f"[*] Compressing and streamlining files into '{zip_path.name}'...")
    print("    (Resizing training images to 640x640 on the fly; please wait)...")
    
    files_added = 0
    images_compressed = 0
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add files from root
            zipf.write(train_engine_source, arcname="train_engine.py")
            zipf.write(reqs_source, arcname="requirements_runpod.txt")
            files_added += 2
            
            # Add scripts folder
            zipf.write(setup_py_source, arcname="scripts/core/setup-runpod-environment.py")
            zipf.write(setup_sh_source, arcname="scripts/core/setup-runpod-environment.sh")
            # Include generators and downloaders for remote execution
            zipf.write(root_path / "scripts" / "download_exhibition_backgrounds.py", arcname="scripts/download_exhibition_backgrounds.py")
            zipf.write(root_path / "scripts" / "generate_copypaste_data.py", arcname="scripts/generate_copypaste_data.py")
            files_added += 4
            
            # Include Phase 4 best weights as starting weights for remote fine-tuning
            nano_source = root_path / "runs" / "detect" / "CIRCA_V12N_004_TRAIN_Phase4_Nano" / "weights" / "best.pt"
            small_source = root_path / "runs" / "detect" / "CIRCA_V12S_005_TRAIN_Phase4_Small" / "weights" / "best.pt"
            medium_source = root_path / "runs" / "detect" / "CIRCA_V12M_006_TRAIN_Phase4_Medium" / "weights" / "best.pt"
            
            if nano_source.exists():
                zipf.write(nano_source, arcname="models/yolo12n.pt")
                files_added += 1
            if small_source.exists():
                zipf.write(small_source, arcname="models/yolo12s.pt")
                files_added += 1
            if medium_source.exists():
                zipf.write(medium_source, arcname="models/yolo12m.pt")
                files_added += 1
            
            # Add dataset recursively with on-the-fly compression
            dataset_files = list(dataset_source.rglob("*"))
            for file_path in dataset_files:
                if file_path.is_file():
                    rel_path = file_path.relative_to(root_path)
                    arcname = str(rel_path).replace("\\", "/")
                    
                    ext = file_path.suffix.lower()
                    if ext in {'.jpg', '.jpeg', '.png', '.bmp'}:
                        # On-the-fly resizing and compression
                        img = cv2.imread(str(file_path))
                        if img is not None:
                            h, w = img.shape[:2]
                            if h > 640 or w > 640:
                                img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
                            # Encode as JPEG with quality 85
                            success, encoded_img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if success:
                                zipf.writestr(arcname, encoded_img.tobytes())
                                images_compressed += 1
                                files_added += 1
                                continue
                                
                    # Default write for labels and other files
                    zipf.write(file_path, arcname=arcname)
                    files_added += 1
                    
        print(f"  [+] Compression successful! Added {files_added} files ({images_compressed} images compressed).")
    except Exception as e:
        print(f"\n[!] CRITICAL ERROR during compression: {e}\n")
        sys.exit(1)
        
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 60)
    print("                STREAMLINED PACKAGE CREATED!             ")
    print("=" * 60)
    print(f"File Name     : {zip_path.name}")
    print(f"File Location : {zip_path.resolve()}")
    print(f"File Size     : {zip_size_mb:.2f} MB")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
