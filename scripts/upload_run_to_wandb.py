#!/usr/bin/env python
"""
CIRCA WandB Run Restorer & Uploader
==================================
This utility manually synchronizes a completed local training run folder 
to Weights & Biases, fully reconstructing the training epoch curves, 
attaching the correct configuration hyperparameters, and uploading 
the model weights as versioned artifacts.
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def print_banner(title):
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60)

def parse_run_name(folder_name):
    """
    Parse CIRCA run name (e.g. CIRCA_V12S_001_TRAIN_Baseline_Vanilla)
    to extract metadata.
    """
    parts = folder_name.split('_')
    metadata = {
        "variant": "s",
        "id": "000",
        "job_type": "train",
        "description": folder_name
    }
    
    if len(parts) >= 3:
        # e.g., CIRCA, V12S, 001, TRAIN, Baseline, Vanilla
        # parts[1] is V12S or similar
        variant_part = parts[1].lower()
        if "v12" in variant_part:
            metadata["variant"] = variant_part.replace("v12", "")
        
        # parts[2] is ID (001)
        metadata["id"] = parts[2]
        
        # parts[3] is TRAIN or TUNE
        metadata["job_type"] = parts[3].lower()
        
        # remaining is description
        metadata["description"] = "_".join(parts[4:]) if len(parts) > 4 else parts[3]
        
    return metadata

def main():
    print_banner("CIRCA WandB Run Restorer & Uploader")
    
    parser = argparse.ArgumentParser(description="Manually upload a completed local training run folder to Weights & Biases.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the training run directory (e.g., runs/detect/CIRCA_V12S_001_TRAIN_Baseline_Vanilla)")
    parser.add_argument("--project", type=str, default="circa-yolov12", help="W&B Project name")
    parser.add_argument("--notes", type=str, default="", help="Optional notes for this run in W&B")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        print(f"[!] ERROR: Run directory does not exist or is not a folder: {run_dir}")
        sys.exit(1)
        
    results_csv = run_dir / "results.csv"
    args_yaml = run_dir / "args.yaml"
    best_weights = run_dir / "weights" / "best.pt"
    
    if not results_csv.exists():
        print(f"[!] ERROR: Could not find 'results.csv' in {run_dir}. Is this a valid Ultralytics run folder?")
        sys.exit(1)
        
    # Check if W&B API Key is available
    if not os.environ.get("WANDB_API_KEY"):
        print("[!] ERROR: WANDB_API_KEY environment variable not found in .env or system environment.")
        print("Please ensure your .env file in the project root contains:")
        print("WANDB_API_KEY=your_key_here")
        sys.exit(1)
        
    # Parse run name metadata
    folder_name = run_dir.name
    meta = parse_run_name(folder_name)
    print(f"[*] Parsed Run Name: {folder_name}")
    print(f"    - Variant    : {meta['variant'].upper()}")
    print(f"    - Run ID     : {meta['id']}")
    print(f"    - Job Type   : {meta['job_type'].upper()}")
    print(f"    - Description: {meta['description']}")
    
    # Load training configurations from args.yaml if available
    config_dict = {
        "run_folder": folder_name,
        "variant": meta["variant"],
        "id": meta["id"],
        "description": meta["description"],
        "manual_upload": True
    }
    
    if args_yaml.exists():
        try:
            with open(args_yaml, "r") as f:
                yolo_args = yaml.safe_load(f)
                if isinstance(yolo_args, dict):
                    # Merge YOLO args into our config
                    config_dict.update(yolo_args)
                    print("[+] Successfully loaded and merged training hyperparameters from args.yaml")
        except Exception as e:
            print(f"[!] WARNING: Failed to parse args.yaml: {e}")
            
    # Import wandb safely
    try:
        import wandb
    except ImportError:
        print("[!] ERROR: wandb package is not installed in your current environment.")
        print("Please run: pip install wandb")
        sys.exit(1)
        
    # Initialize WandB Run
    print(f"[*] Initializing W&B run in project '{args.project}'...")
    run_notes = args.notes if args.notes else f"Manual offline upload of run {folder_name}"
    
    run = wandb.init(
        project=args.project,
        name=folder_name,
        job_type=meta["job_type"],
        notes=run_notes,
        config=config_dict
    )
    
    # Read metrics CSV and upload
    print(f"[*] Reading metrics from {results_csv.name}...")
    df = pd.read_csv(results_csv)
    
    # Clean column names (strip spaces if any)
    df.columns = [c.strip() for c in df.columns]
    
    print(f"[*] Uploading {len(df)} epochs to W&B...")
    for idx, row in df.iterrows():
        # Clean row data
        clean_row = row.dropna().to_dict()
        
        # Convert scientific notation or weird types to floats where applicable
        metrics_to_log = {}
        for k, v in clean_row.items():
            # Standardize column naming for W&B formatting
            # e.g., metrics/mAP50(B) -> metrics/mAP50
            clean_key = k.replace("(B)", "")
            metrics_to_log[clean_key] = float(v) if not isinstance(v, str) else v
            
        # Log epoch data
        epoch_num = int(metrics_to_log.get("epoch", idx + 1))
        
        # Log metrics to W&B
        wandb.log(metrics_to_log, step=epoch_num)
        
    print(f"[+] Successfully logged {len(df)} epochs.")
    
    # Upload training charts/plots if they exist
    charts = [
        "results.png", "confusion_matrix_normalized.png", 
        "BoxPR_curve.png", "BoxF1_curve.png"
    ]
    for chart in charts:
        chart_path = run_dir / chart
        if chart_path.exists():
            print(f"[*] Logging chart: {chart}...")
            # Log image directly
            run.log({chart.replace(".png", ""): wandb.Image(str(chart_path))})
            
    # Upload best.pt as a model artifact if it exists
    if best_weights.exists():
        print(f"[*] Uploading best model weights: {best_weights.name} ({best_weights.stat().st_size / (1024*1024):.2f} MB)...")
        artifact_name = f"{folder_name.lower().replace('_', '-')}-model"
        artifact = wandb.Artifact(artifact_name, type="model", description=f"Best weights for run {folder_name}")
        artifact.add_file(str(best_weights))
        run.log_artifact(artifact)
        print(f"[+] Successfully logged model artifact: {artifact_name}")
    else:
        print("[!] WARNING: 'weights/best.pt' not found. Skipping weight artifact upload.")
        
    run.finish()
    print("\n" + "=" * 60)
    print("               UPLOAD COMPLETED SUCCESSFULLY!             ")
    print("=" * 60)
    print("Check your W&B dashboard for the synchronized run.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
