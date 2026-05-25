#!/usr/bin/env python
"""
CIRCA Cross-Platform RunPod Upload Package Builder
==================================================
This script packages exactly the required dataset and training scripts for RunPod
using Python's zipfile module. This guarantees that all directory paths in the ZIP
use forward slashes (/), which prevents flat extraction bugs on Linux/RunPod.
"""

import os
import sys
import zipfile
from pathlib import Path

def print_banner(title):
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60)

def main():
    print_banner("CIRCA Python RunPod Package Builder")
    
    # 1. Path Configurations
    root_path = Path("d:/FYP/CIRCA").resolve()
    zip_path = root_path / "CIRCA_runpod.zip"
    
    dataset_source = root_path / "datasets" / "unified_pcb_v3"
    train_engine_source = root_path / "train_engine.py"
    reqs_source = root_path / "requirements_runpod.txt"
    setup_py_source = root_path / "scripts" / "runpod_setup.py"
    setup_sh_source = root_path / "scripts" / "runpod_setup.sh"
    
    # 2. Validations
    print("[*] Validating required files...")
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
        else:
            print(f"  [+] Found: {desc}")
            
    if missing:
        print("\n[!] CRITICAL: Cannot package. Some required files are missing.\n")
        sys.exit(1)
        
    # 3. Clean up previous ZIP if it exists
    if zip_path.exists():
        print("[*] Removing previous ZIP package...")
        zip_path.unlink()
        
    # 4. Compile ZIP
    print(f"[*] Compressing files into '{zip_path.name}'...")
    print("    (This handles thousands of images; please wait)...")
    
    files_added = 0
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add files from root
            zipf.write(train_engine_source, arcname="train_engine.py")
            zipf.write(reqs_source, arcname="requirements_runpod.txt")
            files_added += 2
            
            # Add scripts folder
            zipf.write(setup_py_source, arcname="scripts/runpod_setup.py")
            zipf.write(setup_sh_source, arcname="scripts/runpod_setup.sh")
            files_added += 2
            
            # Add dataset recursively
            dataset_files = list(dataset_source.rglob("*"))
            for file_path in dataset_files:
                if file_path.is_file():
                    # Calculate relative path from CIRCA root to keep 'datasets/unified_pcb_v3/...'
                    rel_path = file_path.relative_to(root_path)
                    # Convert to forward slashes for Linux compatibility
                    arcname = str(rel_path).replace("\\", "/")
                    zipf.write(file_path, arcname=arcname)
                    files_added += 1
                    
        print(f"  [+] Compression successful! Added {files_added} files.")
    except Exception as e:
        print(f"\n[!] CRITICAL ERROR during compression: {e}\n")
        sys.exit(1)
        
    # 5. Success Report
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 60)
    print("                PACKAGE CREATED SUCCESSFULLY!             ")
    print("=" * 60)
    print(f"File Name     : {zip_path.name}")
    print(f"File Location : {zip_path.resolve()}")
    print(f"File Size     : {zip_size_mb:.2f} MB")
    print("=" * 60)
    print("\nNext Steps on RunPod:")
    print("1. Delete your current flat CIRCA extraction folder in Jupyter Lab.")
    print("2. Upload the newly generated 'CIRCA_runpod.zip'.")
    print("3. Unzip standardly into your terminal:")
    print("   python -m zipfile -e CIRCA_runpod.zip CIRCA")
    print("   cd CIRCA")
    print("   bash scripts/runpod_setup.sh")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
