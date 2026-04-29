import os
import logging
import argparse
import shutil
import torch
from ultralytics import YOLO

# --- Configure file-based logging for crash resilience ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)


def run_experiment():
    parser = argparse.ArgumentParser(description="CIRCA High-Performance Training Engine")
    parser.add_argument("--variant", type=str, default="s", choices=["n", "s", "m", "l", "x"], help="Model variant")
    parser.add_argument("--id", type=str, required=True, help="Experiment ID (e.g., 001)")
    parser.add_argument("--desc", type=str, required=True, help="Description (e.g., HPO_Baseline)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (640 recommended for stability)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (8 recommended for 6GB VRAM)")
    parser.add_argument("--data", type=str, default="datasets/PKU-Market-PCB-ver1/data.yaml", help="Path to data.yaml")
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM (requires 16GB+ free RAM)")
    parser.add_argument("--clear", action="store_true", help="Clear existing experiment folder")
    parser.add_argument("--device", type=str, default="0", help="Device (0 or cpu)")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers (2 recommended for Windows 16GB RAM)")
    
    args = parser.parse_args()

    # 1. Standardized Naming
    folder_name = f"CIRCA_V12{args.variant.upper()}_{args.id}_{args.desc}"
    exp_dir = os.path.join("runs", "detect", folder_name)
    
    # 2. Dataset Validation
    if not os.path.exists(args.data):
        log.error("--- [ERROR] Dataset config not found: %s ---", args.data)
        raise FileNotFoundError(args.data)

    # 3. Cleanup Logic
    if args.clear and os.path.exists(exp_dir):
        if os.path.exists(os.path.join(exp_dir, "weights", "last.pt")):
            log.warning("--- [WARNING] Clearing existing experiment with checkpoints: %s ---", folder_name)
        log.info("--- [CLEANUP] Removing: %s ---", exp_dir)
        shutil.rmtree(exp_dir)

    # 4. Hardware Capability Check & Logic Fix
    # Handle 'cuda' or '0' correctly
    device_id = args.device
    if device_id.isdigit():
        device_id = int(device_id)
        
    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    
    if device == "cuda":
        # Parse physical index for properties check
        idx = int(args.device) if args.device.isdigit() else 0
        try:
            props = torch.cuda.get_device_properties(idx)
            log.info("--- [HARDWARE] Detected: %s (%dMB VRAM) ---", props.name, props.total_memory / 1024**2)
        except Exception as e:
            log.warning("--- [HARDWARE] Could not query GPU properties: %s ---", e)

    # 5. Resume Detection
    checkpoint_path = os.path.join(exp_dir, "weights", "last.pt")
    resume = os.path.exists(checkpoint_path)
    model_source = checkpoint_path if resume else f"yolo12{args.variant.lower()}.pt"

    # 6. Load Model
    log.info("--- [LOAD] Initializing model from: %s ---", model_source)
    model = YOLO(model_source)

    # 7. Training Execution
    log.info("--- [START] %s | Variant: %s | Resolution: %d ---", folder_name, args.variant, args.imgsz)
    
    try:
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            resume=resume,
            name=folder_name,
            device=args.device,
            
            # --- Windows Stability Optimizations ---
            amp=True,               # Automatic Mixed Precision
            cache=args.cache,       # Defaults to False (prevents WinError 1455)
            workers=args.workers,   # Defaults to 2 (lowers paging file pressure)
            
            # --- Small Object Strategy ---
            multi_scale=True,       # Improve scale robustness
            overlap_mask=True,      
            rect=False,             
            
            # --- HPO / Stability Defaults ---
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            box=7.5,
            cls=0.5,
            patience=50,
            cos_lr=True,
            exist_ok=True,
            plots=True
        )
    except Exception as e:
        log.error("--- [CRASH] Training interrupted: %s ---", e)
        return

    # 8. Standardized Export
    log.info("--- [EXPORT] Compiling to OpenVINO INT8 ---")
    model.export(format="openvino", int8=True, data=args.data)
    log.info("--- [COMPLETE] Experiment %s finalized. ---", args.id)


if __name__ == "__main__":
    run_experiment()
