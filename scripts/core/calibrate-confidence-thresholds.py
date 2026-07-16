"""
scripts/calibrate_thresholds.py
---------------------------------
Phase 7 — Confidence Threshold Calibration (Validation Split Only)

Sweeps the confidence threshold from 0.10 to 0.90 in steps of 0.05,
computing per-class precision and recall at each threshold on the
VALIDATION split (NEVER the test split).

Selects per-class thresholds as defined in Chapter 3 §3.6.5:
  - Display threshold : min confidence where per-class precision >= 0.90
  - Warning threshold : min confidence where per-class recall >= 0.95

Outputs:
    circa_thresholds.yaml                          — production threshold file
    docs/assets/fig4_9_threshold_sweep.png         — P/R vs threshold plot

Usage:
    python scripts/calibrate_thresholds.py \\
        --model runs/detect/CIRCA_V12S_005_TRAIN_Phase4_Small/weights/best.pt \\
        --data datasets/unified_pcb_v3_preproc/data.yaml

    # With exported OpenVINO IR:
    python scripts/calibrate_thresholds.py \\
        --model runs/detect/CIRCA_V12S_005_TRAIN_Phase4_Small/weights/best_int8_openvino_model \\
        --data datasets/unified_pcb_v3_preproc/data.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("calibrate")

# ---------------------------------------------------------------------------
# Chapter 3 §3.6.5 threshold selection targets
# ---------------------------------------------------------------------------
DISPLAY_PRECISION_TARGET = 0.90   # minimum precision for display threshold
WARNING_RECALL_TARGET    = 0.95   # minimum recall for warning threshold
CONF_SWEEP_START  = 0.10
CONF_SWEEP_END    = 0.60
CONF_SWEEP_STEP   = 0.10

VAL_SPLIT         = "val"         # NEVER sweep on test split
IMGSZ             = 640
SEED              = 42

CLASS_LABELS = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "excess_solder",
    5: "insufficient_solder",
    6: "cold_solder_joint",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 7 — CIRCA Confidence Threshold Calibration (val split only)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to best.pt or OpenVINO IR directory",
    )
    parser.add_argument(
        "--data", type=str, default="datasets/unified_pcb_v3_preproc/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--output", type=str, default="config/circa_thresholds.yaml",
        help="Output path for thresholds YAML (default: config/circa_thresholds.yaml)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="docs",
        help="Directory for the threshold sweep plot (default: docs/)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to run validation on (cpu, or 0, 1 etc.)",
    )
    parser.add_argument(
        "--fraction", type=float, default=1.0,
        help="Fraction of validation dataset to use (default: 1.0)",
    )
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    return parser.parse_args()


def run_val_at_conf(model, data_path: str, conf: float, imgsz: int, device: str = "cpu", fraction: float = 1.0) -> dict:
    """
    Run model.val() at a single confidence threshold.
    Returns per-class precision and recall arrays.
    """
    import numpy as np
    metrics = model.val(
        data=data_path,
        split=VAL_SPLIT,
        conf=conf,
        imgsz=imgsz,
        iou=0.6,           # standard NMS IoU
        seed=SEED,
        verbose=False,
        device=device,
        fraction=fraction,
    )

    # metrics.box.p, metrics.box.r are arrays of shape (num_classes,)
    # metrics.box.ap_class_index maps array positions to class IDs
    per_class_p = {}
    per_class_r = {}
    if hasattr(metrics.box, "p") and hasattr(metrics.box, "ap_class_index"):
        for idx, class_id in enumerate(metrics.box.ap_class_index):
            cid = int(class_id)
            per_class_p[cid] = float(metrics.box.p[idx]) if idx < len(metrics.box.p) else 0.0
            per_class_r[cid] = float(metrics.box.r[idx]) if idx < len(metrics.box.r) else 0.0
    return {"precision": per_class_p, "recall": per_class_r}


def calibrate(args: argparse.Namespace) -> None:
    from ultralytics import YOLO
    import numpy as np

    model_path = Path(args.model)
    if not model_path.exists():
        log.error("Model not found: %s", model_path)
        sys.exit(1)

    # Load model — supports both .pt and OpenVINO IR directory
    log.info("[LOAD] Loading model: %s", model_path)
    model = YOLO(str(model_path))


    log.info("=" * 60)
    log.info("Confidence threshold sweep: %.2f → %.2f (step %.2f)", CONF_SWEEP_START, CONF_SWEEP_END, CONF_SWEEP_STEP)
    log.info("Split: %s (test split is NEVER touched during calibration)", VAL_SPLIT)
    log.info("=" * 60)

    # Build threshold list
    thresholds = []
    t = CONF_SWEEP_START
    while t <= CONF_SWEEP_END + 1e-9:
        thresholds.append(round(t, 2))
        t += CONF_SWEEP_STEP

    # Accumulate per-class P and R at each threshold
    # sweep_data[class_id][threshold] = {precision, recall}
    sweep_data: dict[int, dict[float, dict]] = {cid: {} for cid in CLASS_LABELS}

    for conf in thresholds:
        log.info("[SWEEP] conf=%.2f ...", conf)
        result = run_val_at_conf(model, args.data, conf, args.imgsz, device=args.device, fraction=args.fraction)
        for cid in CLASS_LABELS:
            sweep_data[cid][conf] = {
                "precision": result["precision"].get(cid, 0.0),
                "recall":    result["recall"].get(cid, 0.0),
            }

    # Select thresholds per class
    display_thresholds: dict[str, float] = {}
    warning_thresholds: dict[str, float] = {}

    for cid, class_name in CLASS_LABELS.items():
        display_t = None
        warning_t = None
        for t in thresholds:
            p = sweep_data[cid][t]["precision"]
            r = sweep_data[cid][t]["recall"]
            if display_t is None and p >= DISPLAY_PRECISION_TARGET:
                display_t = t
            if warning_t is None and r >= WARNING_RECALL_TARGET:
                warning_t = t
        # Default to sweep endpoints if targets not met
        display_thresholds[class_name] = display_t if display_t is not None else CONF_SWEEP_END
        warning_thresholds[class_name]  = warning_t  if warning_t  is not None else CONF_SWEEP_START
        log.info(
            "[CLASS] %-22s  display_threshold=%.2f  warning_threshold=%.2f",
            class_name, display_thresholds[class_name], warning_thresholds[class_name],
        )

    # Global trigger config (Ch3 §3.5.3)
    thresholds_yaml = {
        "_meta": {
            "generated_by": "scripts/calibrate_thresholds.py",
            "split_used": VAL_SPLIT,
            "model": str(model_path),
            "display_target_precision": DISPLAY_PRECISION_TARGET,
            "warning_target_recall": WARNING_RECALL_TARGET,
        },
        "display_thresholds": display_thresholds,
        "warning_thresholds": warning_thresholds,
        "global_trigger": {
            "mean_confidence_below": 0.50,
            "no_detection_timeout_s": 1.0,
            "note": "Banner fires when mean conf < 0.50 OR Laplacian variance < blur_threshold OR no boxes for >= 1s",
        },
    }

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(thresholds_yaml, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    log.info("[OUTPUT] Thresholds written → %s", output_path)

    # Generate per-class P-R vs threshold sweep plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        assets_dir = Path(args.output_dir) / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
        axes = axes.flatten()
        colors = plt.cm.tab10.colors  # type: ignore

        for idx, (cid, class_name) in enumerate(CLASS_LABELS.items()):
            ax = axes[idx]
            prec_vals = [sweep_data[cid][t]["precision"] for t in thresholds]
            rec_vals  = [sweep_data[cid][t]["recall"]    for t in thresholds]
            ax.plot(thresholds, prec_vals, color=colors[0], linewidth=1.5, label="Precision")
            ax.plot(thresholds, rec_vals,  color=colors[1], linewidth=1.5, label="Recall", linestyle="--")
            dt = display_thresholds[class_name]
            wt = warning_thresholds[class_name]
            ax.axvline(dt, color=colors[0], linestyle=":", alpha=0.7, label=f"Display ({dt:.2f})")
            ax.axvline(wt, color=colors[1], linestyle=":", alpha=0.7, label=f"Warning ({wt:.2f})")
            ax.axhline(DISPLAY_PRECISION_TARGET, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_title(class_name.replace("_", " ").title(), fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(CONF_SWEEP_START, CONF_SWEEP_END)
            ax.grid(alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=7)

        # Hide unused subplot
        if len(CLASS_LABELS) < len(axes):
            for ax in axes[len(CLASS_LABELS):]:
                ax.set_visible(False)

        fig.suptitle("CIRCA Confidence Threshold Sweep — Per-Class Precision & Recall (Val Split)", fontsize=11)
        fig.supxlabel("Confidence Threshold", fontsize=10)
        fig.tight_layout()

        plot_path = assets_dir / "fig4_9_threshold_sweep.png"
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("[PLOT] Threshold sweep saved → %s", plot_path)
    except ImportError:
        log.warning("[PLOT] matplotlib not available — skipping sweep plot")

    log.info("\nCalibration complete. Load %s in the CIRCA application.", output_path)


if __name__ == "__main__":
    calibrate(parse_args())
