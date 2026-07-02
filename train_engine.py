"""
CIRCA High-Performance Training Engine
======================================
Thesis-aligned training & HPO engine for YOLOv12 PCB defect detection.

Key features:
- Optional CIRCA preprocessing pipeline (CLAHE + Gamma) applied once to disk
- HPO via Ultralytics genetic tuner with playbook-aligned search space
- Stability patches for low-VRAM training
- INT8 OpenVINO export from best.pt (not last epoch)
- Reproducibility: seeded runs, full version logging
"""

import argparse
import logging
import os
import shutil
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

# --- Weights & Biases Configuration (Playbook §1.6) ---
# Set BEFORE importing ultralytics so its W&B callback picks these up.
os.environ.setdefault("WANDB_PROJECT", "circa-yolov12")
os.environ.setdefault("WANDB_LOG_MODEL", "end")  # 'end' = upload final best.pt only (saves bandwidth)

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
import ultralytics

# Windows-specific: ANSI support for progress bar
if sys.platform == "win32":
    try:
        from colorama import init
        init(autoreset=True)
    except ImportError:
        pass

# --- File-based logging for crash resilience ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# Image extensions (case-insensitive on case-sensitive filesystems)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}



# ---------------------------------------------------------------------------
# Weights & Biases bootstrap
# ---------------------------------------------------------------------------
def ensure_wandb_login() -> bool:
    """
    Ensure W&B is usable for this run.

    Behaviour:
    - Already authenticated (env var, ~/.netrc, ~/.config/wandb/settings, or resolved api_key): proceed.
    - Not authenticated AND running interactively (TTY): prompt for login.
    - Not authenticated AND headless: disable W&B and warn (prevents hang).

    Returns True if W&B is active, False if disabled.
    """
    try:
        import wandb
    except ImportError:
        log.warning("[W&B] wandb not installed; pip install wandb. Continuing without cloud logging.")
        os.environ["WANDB_MODE"] = "disabled"
        return False

    # Check for existing credentials using multiple mechanisms
    has_key = bool(os.environ.get("WANDB_API_KEY"))
    has_netrc = Path.home().joinpath(".netrc").exists()
    has_config = Path.home().joinpath(".config", "wandb", "settings").exists()
    
    # Use wandb's internal api_key resolver as the ultimate source of truth
    has_resolved = False
    try:
        has_resolved = bool(wandb.api.api_key)
    except Exception:
        pass

    if has_key or has_netrc or has_config or has_resolved:
        log.info("[W&B] Existing credentials detected")
        # Ensure we are not overriding to disabled if we have credentials
        if os.environ.get("WANDB_MODE") == "disabled":
            del os.environ["WANDB_MODE"]
        return True

    # Headless / non-interactive: refuse to prompt to avoid hanging
    if not sys.stdin.isatty():
        os.environ["WANDB_MODE"] = "disabled"
        log.warning(
            "[W&B] No credentials and stdin is not a TTY -- running with "
            "WANDB_MODE=disabled to avoid an interactive hang. "
            "Run `wandb login` once, or set WANDB_API_KEY, to enable cloud logging."
        )
        return False

    log.info("[W&B] No credentials found. Launching interactive login...")
    try:
        ok = wandb.login()  # blocking; opens browser / prompts for API key
    except Exception as e:
        log.warning("[W&B] Login failed (%s). Running with WANDB_MODE=disabled.", e)
        os.environ["WANDB_MODE"] = "disabled"
        return False

    if not ok:
        log.warning("[W&B] Login was not completed. Running with WANDB_MODE=disabled.")
        os.environ["WANDB_MODE"] = "disabled"
        return False

    log.info("[W&B] Login successful")
    return True


# ---------------------------------------------------------------------------
# CIRCA Thesis Preprocessing (CLAHE + Gamma)
# ---------------------------------------------------------------------------
def apply_circa_preproc(image_path: Path, output_path: Path) -> bool:
    """Apply CLAHE on L-channel of LAB, then Gamma=1.2. Returns True on success."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        log.warning("[PREPROC] Unreadable image skipped: %s", image_path)
        return False

    # 1. CLAHE on L channel of LAB
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. Gamma Correction (1.2)
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")
    out = cv2.LUT(out, table)

    cv2.imwrite(str(output_path), out)
    return True


def _preproc_one(args):
    """Worker for multiprocessing pool."""
    src, dst = args
    return apply_circa_preproc(src, dst)


def _list_images(directory: Path) -> list:
    """List images in directory, case-insensitive on extension."""
    if not directory.exists():
        return []
    return [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS]


def preprocess_dataset(data_yaml_path: str, force: bool = False) -> str:
    """
    Pre-process the entire dataset once and return the path to the new data.yaml.

    Resolves label paths from each image's actual location (image_dir/../labels/...),
    so non-default dataset layouts are handled correctly.
    """
    data_yaml_path = Path(data_yaml_path)
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    base_path = Path(data["path"]).resolve()
    preproc_base = base_path.parent / (base_path.name + "_preproc")

    if preproc_base.exists() and not force:
        log.info("[PREPROC] Reusing existing preprocessed dataset: %s", preproc_base)
        return str(preproc_base / "data.yaml")

    if force and preproc_base.exists():
        log.warning("[PREPROC] --force-preproc set, removing %s", preproc_base)
        shutil.rmtree(preproc_base)

    log.info("[PREPROC] Starting CIRCA preprocessing (CLAHE + Gamma) at %s", preproc_base)

    # Build canonical split mapping using whatever keys are present in the YAML
    yaml_split_keys = {}
    for canonical in ("train", "val", "test"):
        if canonical in data:
            yaml_split_keys[canonical] = canonical
        elif canonical == "val" and "valid" in data:
            yaml_split_keys[canonical] = "valid"

    # Rule 2 (AGENTS.md): cap workers at 8 on local runs to prevent IDE crashes.
    # On RunPod (detected via RUNPOD_POD_ID env var), use full CPU count.
    _on_runpod = bool(os.environ.get("RUNPOD_POD_ID"))
    if _on_runpod:
        n_workers = max(1, cpu_count() - 1)
        log.info("[PREPROC] RunPod detected — using %d workers (uncapped)", n_workers)
    else:
        n_workers = min(8, max(1, cpu_count() - 1))
        log.info("[PREPROC] Local run — workers capped at 8 (using %d)", n_workers)

    for canonical, yaml_key in yaml_split_keys.items():
        split_rel = data[yaml_key]
        src_img_dir = (base_path / split_rel).resolve()
        dst_img_dir = preproc_base / canonical / "images"
        dst_lbl_dir = preproc_base / canonical / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        images = _list_images(src_img_dir)
        if not images:
            log.warning("[PREPROC] No images found for split %s at %s", canonical, src_img_dir)
            continue

        log.info("[PREPROC] %s: processing %d images with %d workers", canonical, len(images), n_workers)

        # Parallel image preprocessing
        tasks = [(img, dst_img_dir / img.name) for img in images]
        with Pool(n_workers) as pool:
            results = pool.map(_preproc_one, tasks)
        n_ok = sum(results)
        log.info("[PREPROC] %s: %d/%d images written", canonical, n_ok, len(images))

        # Copy labels (resolved relative to each image, not a hardcoded path)
        n_labels = 0
        for img in images:
            # Standard YOLO layout: <split>/images/x.jpg <-> <split>/labels/x.txt
            src_lbl = img.parent.parent / "labels" / (img.stem + ".txt")
            if src_lbl.exists():
                shutil.copy(src_lbl, dst_lbl_dir / src_lbl.name)
                n_labels += 1
            else:
                log.warning("[PREPROC] Missing label for %s", img.name)
        log.info("[PREPROC] %s: %d/%d labels copied", canonical, n_labels, len(images))

    # Write new data.yaml mirroring the original structure
    new_data = data.copy()
    try:
        # Compute relative path to workspace root (Cwd) to keep path relative
        rel_path = os.path.relpath(preproc_base, Path.cwd())
        new_data["path"] = rel_path.replace("\\", "/")
    except ValueError:
        try:
            # Fallback: make it relative to the parent of the input yaml if on a different drive (e.g. windows pytest tmp_path)
            rel_path = os.path.relpath(preproc_base, Path(data_yaml_path).parent)
            new_data["path"] = rel_path.replace("\\", "/")
        except ValueError:
            # Absolute fallback
            new_data["path"] = str(preproc_base.resolve()).replace("\\", "/")

    if "train" in yaml_split_keys:
        new_data["train"] = "train/images"
    if "val" in yaml_split_keys:
        # Preserve original key (val vs valid) for downstream tooling
        new_data[yaml_split_keys["val"]] = "val/images"
        new_data["val"] = "val/images"  # Ultralytics always reads `val`
    if "test" in yaml_split_keys:
        new_data["test"] = "test/images"

    new_yaml_path = preproc_base / "data.yaml"
    with open(new_yaml_path, "w") as f:
        yaml.dump(new_data, f)

    log.info("[PREPROC] Complete. New data.yaml: %s", new_yaml_path)
    return str(new_yaml_path)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiment():
    parser = argparse.ArgumentParser(description="CIRCA High-Performance Training Engine")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "tune"], help="Execution mode")
    parser.add_argument("--variant", type=str, default="s", choices=["n", "s", "m", "l", "x"], help="Model variant")
    parser.add_argument("--id", type=str, required=True, help="Experiment ID (e.g., 001)")
    parser.add_argument("--desc", type=str, required=True, help="Description (e.g., HPO_Baseline)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs per trial/run")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations for tuning mode")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (640 recommended for stability)")
    parser.add_argument("--batch", type=int, default=12, help="Batch size (12 recommended for 6GB VRAM)")
    parser.add_argument("--data", type=str, default="datasets/unified_pcb_v3/data.yaml", help="Path to data.yaml (v3 = 7-class IPC corpus, unified_pcb_v3)")
    parser.add_argument("--cfg", type=str, default=None, help="Optional path to best_hyperparameters.yaml from HPO")
    parser.add_argument("--preproc", action="store_true", help="Apply CIRCA preprocessing (CLAHE + Gamma)")
    parser.add_argument("--force-preproc", action="store_true", help="Regenerate preprocessed dataset even if cached")
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM (requires 16GB+ free RAM)")
    parser.add_argument("--clear", action="store_true", help="Clear existing experiment folder")
    parser.add_argument("--device", type=str, default="0", help="Device (0 or cpu)")
    parser.add_argument("--workers", type=int,
                        default=(max(1, cpu_count() - 1) if os.environ.get("RUNPOD_POD_ID") else 4),
                        help="DataLoader workers (auto: full CPU on RunPod, 4 on local)")
    parser.add_argument("--multi_scale", action="store_true", help="Enable multi-scale training (risky on 6GB VRAM)")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial LR (ignored if --cfg provided)")
    parser.add_argument("--warmup-epochs", type=float, default=5.0, help="Warmup epochs (ignored if --cfg provided)")
    parser.add_argument("--nbs", type=int, default=64, help="Nominal batch size for loss scaling")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of training data to use per HPO trial (default: 1.0 = full dataset). "
                             "Use 0.5 on balanced data to halve tuning time. Ignored in train mode.")
    parser.add_argument("--cls-pw", type=float, default=1.0,
                        help="Power for inverse-frequency class weighting (Ultralytics fraction key, range [0.0, 1.0]). "
                             "0.0 = disabled; 1.0 = full inverse-frequency weighting (recommended for "
                             "bare-board/solder class imbalance in unified_pcb_v3). Ignored when --cfg is provided.")

    args = parser.parse_args()

    # --- Version logging (Playbook §9 guardrail) ---
    log.info(
        "[VERSIONS] torch=%s | ultralytics=%s | opencv=%s | cuda=%s",
        torch.__version__, ultralytics.__version__, cv2.__version__,
        torch.version.cuda if torch.cuda.is_available() else "n/a",
    )

    # --- W&B login (interactive prompt if needed; safe-disable if headless) ---
    wandb_active = ensure_wandb_login()

    # 0. Optional Preprocessing
    data_path = args.data
    if args.preproc:
        data_path = preprocess_dataset(args.data, force=args.force_preproc)

    # 1. Standardized Naming
    mode_prefix = "TUNE" if args.mode == "tune" else "TRAIN"
    folder_name = f"CIRCA_V12{args.variant.upper()}_{args.id}_{mode_prefix}_{args.desc}"
    exp_dir = Path("runs") / "detect" / folder_name

    # --- W&B run metadata: tags + config snapshot for filtering in the UI ---
    if wandb_active:
        os.environ["WANDB_NAME"] = folder_name
        os.environ["WANDB_TAGS"] = ",".join([
            f"v12{args.variant}",
            args.mode,
            f"id-{args.id}",
            "preproc" if args.preproc else "vanilla",
        ])
        os.environ["WANDB_NOTES"] = args.desc

    # 2. Dataset Validation
    if not Path(data_path).exists():
        log.error("[ERROR] Dataset config not found: %s", data_path)
        raise FileNotFoundError(data_path)

    # 3. Cleanup Logic
    if args.clear and exp_dir.exists():
        if (exp_dir / "weights" / "last.pt").exists():
            log.warning("[WARNING] Clearing existing experiment with checkpoints: %s", folder_name)
        log.info("[CLEANUP] Removing: %s", exp_dir)
        shutil.rmtree(exp_dir)

    # 4. Hardware Capability Check
    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    if use_cuda:
        idx = int(args.device) if args.device.isdigit() else 0
        try:
            props = torch.cuda.get_device_properties(idx)
            log.info("[HARDWARE] %s (%dMB VRAM)", props.name, props.total_memory / 1024**2)
        except Exception as e:
            log.warning("[HARDWARE] Could not query GPU properties: %s", e)
    else:
        log.info("[HARDWARE] Running on CPU")

    # 5. Resume Detection (train mode only)
    resume = False
    model_source = f"yolo12{args.variant.lower()}.pt"
    if args.mode == "train":
        checkpoint_path = exp_dir / "weights" / "last.pt"
        if checkpoint_path.exists():
            resume = True
            model_source = str(checkpoint_path)
            log.info("[RESUME] Found checkpoint: %s", checkpoint_path)

    if not resume:
        # Fallback to models/ if not in root
        if not Path(model_source).exists():
            alt_path = Path("models") / model_source
            if alt_path.exists():
                model_source = str(alt_path)

    # 6. Load Model
    log.info("[LOAD] Initializing model from: %s", model_source)
    model = YOLO(model_source)

    # Respect the CLI batch argument for the local 6GB VRAM constraint
    batch_size = args.batch

    # 7. Training/Tuning Execution
    log.info(
        "[START] %s | Mode: %s | Variant: %s | Resolution: %d | Batch: %d | Epochs: %d",
        folder_name, args.mode, args.variant, args.imgsz, batch_size, args.epochs,
    )

    common_kwargs = {
        "data": data_path,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": batch_size,
        "name": folder_name,
        "device": args.device,
        "amp": True,
        "cache": args.cache,
        "workers": args.workers,
        "exist_ok": True,
        "plots": True,
        "seed": 42,            # Playbook: reproducibility — applied to BOTH modes
    }

    try:
        if args.mode == "tune":
            # Playbook v2 search space — 17 parameters aligned with thesis Ch.3 §3.6.2.1.
            # Notes:
            #   - hsv_h / hsv_s removed: PCB colour (solder-mask hue, copper) is diagnostically
            #     useful; perturbing hue can confuse copper vs. background.
            #   - cls_pw added: classification positive-weight, used to combat the bare-board /
            #     solder class imbalance documented in Dataset Expansion Plan §5.
            #   - copy_paste added: small-object augmentation for solder-defect classes.
            #   - degrees capped at 10°: PCBs are roughly aligned in frame.
            search_space = {
                "lr0": (1e-5, 1e-2),
                "lrf": (0.01, 1.0),
                "momentum": (0.7, 0.98),
                "weight_decay": (0.0, 1e-3),
                "warmup_epochs": (0.0, 5.0),
                "box": (0.02, 0.2),
                "cls": (0.2, 4.0),
                "cls_pw": (0.1, 1.0),   # Ultralytics fraction key — must stay within [0, 1]
                "dfl": (0.4, 12.0),
                "hsv_v": (0.0, 0.9),
                "degrees": (0.0, 10.0),
                "translate": (0.0, 0.3),
                "scale": (0.0, 0.9),
                "fliplr": (0.0, 0.5),
                "mosaic": (0.0, 1.0),
                "mixup": (0.0, 0.2),
                "copy_paste": (0.0, 0.3),
            }
            log.info("[TUNE] Genetic search: %d iterations | fraction=%.2f | epochs=%d",
                     args.iterations, args.fraction, args.epochs)
            model.tune(
                iterations=args.iterations,
                optimizer="AdamW",
                space=search_space,
                save=False,
                val=True,
                fraction=args.fraction,
                close_mosaic=10,   # Playbook
                **common_kwargs,
            )
        else:
            train_kwargs = dict(
                resume=resume,
                optimizer="AdamW",
                patience=args.patience,
                cos_lr=True,
                close_mosaic=10,
                multi_scale=args.multi_scale,
                nbs=args.nbs,
                **common_kwargs,
            )

            if args.cfg:
                # HPO-tuned hyperparameters take priority. Do NOT override lr0/warmup/cls_pw.
                cfg_path = Path(args.cfg)
                if not cfg_path.exists():
                    raise FileNotFoundError(f"--cfg not found: {cfg_path}")
                log.info("[CFG] Loading tuned hyperparameters from %s", cfg_path)
                train_kwargs["cfg"] = str(cfg_path)
                # cls_pw is ignored when an HPO config overrides all hyperparameters.
                log.warning("[CFG] --cls-pw=%s ignored because --cfg overrides it", args.cls_pw)
            else:
                # Manual stability patches (only when no HPO config provided)
                train_kwargs["lr0"] = args.lr0
                train_kwargs["warmup_epochs"] = args.warmup_epochs
                # cls_pw must satisfy 0.0 <= cls_pw <= 1.0 (Ultralytics CFG_FRACTION_KEYS).
                # 0.0 disables class weighting; 1.0 = full inverse-frequency weighting.
                if not (0.0 <= args.cls_pw <= 1.0):
                    raise ValueError(f"--cls-pw must be in [0.0, 1.0], got {args.cls_pw}")
                train_kwargs["cls_pw"] = args.cls_pw
                log.info("[CFG] Using manual lr0=%s, warmup_epochs=%s, cls_pw=%s (inverse-freq power)",
                         args.lr0, args.warmup_epochs, args.cls_pw)

            model.train(**train_kwargs)

            # Final metrics summary (train mode only)
            try:
                metrics = model.metrics
                map50   = float(metrics.box.map50)
                map5095 = float(metrics.box.map)
                prec    = float(metrics.box.mp)
                rec     = float(metrics.box.mr)
                log.info(
                    "[RESULT] mAP@0.5=%.4f | mAP@0.5:0.95=%.4f | precision=%.4f | recall=%.4f",
                    map50, map5095, prec, rec,
                )
                # Write a machine-readable summary for downstream Phase 5–7 scripts.
                summary = {
                    "experiment_id": args.id,
                    "description": args.desc,
                    "variant": args.variant,
                    "mode": args.mode,
                    "data": data_path,
                    "mAP50": round(map50, 4),
                    "mAP50_95": round(map5095, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                }
                summary_path = exp_dir / "run_summary.yaml"
                with open(summary_path, "w") as f:
                    yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
                log.info("[RESULT] Summary written to %s", summary_path)
            except Exception as e:
                log.warning("[RESULT] Could not extract final metrics: %s", e)

    except Exception:
        log.exception("[CRASH] Execution interrupted")
        return

    # 8. Standardized Export — always export from best.pt, not the in-memory last-epoch weights
    if args.mode == "train":
        best_pt = exp_dir / "weights" / "best.pt"
        if not best_pt.exists():
            log.error("[EXPORT] best.pt not found at %s; skipping OpenVINO export", best_pt)
            return

        log.info("[EXPORT] Loading %s for OpenVINO INT8 export", best_pt)
        try:
            export_model = YOLO(str(best_pt))
            export_model.export(format="openvino", int8=True, data=data_path, imgsz=args.imgsz)
            log.info("[COMPLETE] Experiment %s finalized. Best weights: %s", args.id, best_pt)
        except Exception:
            log.exception("[EXPORT] OpenVINO INT8 export failed")
    else:
        log.info(
            "[COMPLETE] Tuning %s finalized. Best hyperparameters: %s",
            args.id, exp_dir / "best_hyperparameters.yaml",
        )


if __name__ == "__main__":
    run_experiment()

