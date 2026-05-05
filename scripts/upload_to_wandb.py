import wandb
import pandas as pd
from pathlib import Path
import argparse

def upload_run(run_dir: str, project: str = "circa-yolov12", run_name: str = None):
    run_path = Path(run_dir)
    if not run_path.exists():
        print(f"Directory not found: {run_dir}")
        return

    if run_name is None:
        run_name = run_path.name + "_retroactive"

    print(f"Initializing W&B run: {run_name}")
    run = wandb.init(project=project, name=run_name, job_type="upload_offline")

    csv_file = run_path / "results.csv"
    if csv_file.exists():
        print("Uploading metrics from results.csv...")
        df = pd.read_csv(csv_file)
        # Clean up whitespace in column names
        df.columns = df.columns.str.strip()  
        
        for _, row in df.iterrows():
            # Convert row to dictionary and remove NaNs if any
            metrics = {col: row[col] for col in df.columns if pd.notna(row[col])}
            
            # W&B usually tracks by an internal step, but we can pass epoch
            epoch = int(metrics.get("epoch", _))
            wandb.log(metrics, step=epoch)
            
    print("Uploading plots and images (Confusion Matrix, PR curves, etc)...")
    for img_path in run_path.glob("*.jpg"):
        wandb.log({img_path.stem: wandb.Image(str(img_path))})
    for img_path in run_path.glob("*.png"):
        wandb.log({img_path.stem: wandb.Image(str(img_path))})

    best_pt = run_path / "weights" / "best.pt"
    if best_pt.exists():
        print(f"Uploading best.pt model from {best_pt}...")
        # Save as a W&B Artifact
        art = wandb.Artifact(name=f"run_{run.id}_model", type="model")
        art.add_file(str(best_pt))
        wandb.log_artifact(art)

    wandb.finish()
    print("Upload complete! You can view it on your W&B dashboard.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload offline YOLO run to W&B")
    parser.add_argument("run_dir", type=str, help="Path to the training run directory")
    parser.add_argument("--project", type=str, default="circa-yolov12", help="W&B Project name")
    args = parser.parse_args()
    
    upload_run(args.run_dir, args.project)
