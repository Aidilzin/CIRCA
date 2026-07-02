"""
scripts/evaluate_quantization.py
---------------------------------
Phase 5 — OpenVINO Quantisation Validation

Exports a trained YOLOv12 best.pt to three precision levels (FP32, FP16, INT8)
and validates each against the CIRCA fallback rule defined in Chapter 3 §3.6.4:

  PASS condition (use INT8):
    - INT8 mAP@0.5 on validation split >= 90.0%
    - AND delta(FP32 - INT8) mAP@0.5 <= 1.0 percentage point

  FAIL condition (fall back to FP16):
    - INT8 mAP@0.5 < 90.0%
    - OR delta > 1.0 pp

Outputs:
    docs/quantization_report.md   — full results table + fallback decision

Usage (run locally after Phase 4 weights are downloaded):
    python scripts/evaluate_quantization.py \\
        --weights runs/detect/CIRCA_V12S_005_TRAIN_Phase4_Small/weights/best.pt \\
        --data datasets/unified_pcb_v3_preproc/data.yaml \\
        --variant s
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("quant_eval")

# ---------------------------------------------------------------------------
# Constants — match Chapter 3 §3.6.4 acceptance thresholds
# ---------------------------------------------------------------------------
INT8_MIN_MAP50 = 90.0            # INT8 mAP@0.5 must be >= 90% (in %)
INT8_MAX_DELTA_PP = 1.0          # max allowed degradation vs FP32 (in pp)
VAL_SPLIT = "val"
IMGSZ = 640
SEED = 42

VARIANT_ID_MAP = {
    "n": ("004", "Nano"),
    "s": ("005", "Small"),
    "m": ("006", "Medium"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5 — OpenVINO FP32/FP16/INT8 Quantisation Evaluation"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to best.pt from Phase 4 (e.g. runs/detect/.../weights/best.pt)",
    )
    parser.add_argument(
        "--data", type=str, default="datasets/unified_pcb_v3_preproc/data.yaml",
        help="Path to data.yaml (default: unified_pcb_v3_preproc)",
    )
    parser.add_argument(
        "--variant", type=str, choices=["n", "s", "m"], required=True,
        help="Model variant: n=Nano, s=Small, m=Medium",
    )
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="docs")
    return parser.parse_args()


def export_model(weights_path: Path, precision: str, data_path: str, imgsz: int) -> Optional[Path]:
    """
    Export best.pt to OpenVINO IR at the specified precision level.
    Returns the path to the exported IR directory, or None on failure.
    """
    from ultralytics import YOLO
    log.info("[EXPORT] Loading %s for %s export", weights_path, precision.upper())
    model = YOLO(str(weights_path))
    export_kwargs: dict = dict(format="openvino", imgsz=imgsz)
    if precision == "fp16":
        export_kwargs["half"] = True
    elif precision == "int8":
        export_kwargs["int8"] = True
        export_kwargs["data"] = data_path
    try:
        t0 = time.perf_counter()
        result = model.export(**export_kwargs)
        log.info("[EXPORT] %s done in %.1fs → %s", precision.upper(), time.perf_counter() - t0, result)
        return Path(result) if result else None
    except Exception:
        log.exception("[EXPORT] %s export failed", precision.upper())
        return None


def validate_ir(ir_dir: Path, data_path: str, imgsz: int, batch: int) -> dict:
    """
    Run val on an exported OpenVINO IR directory.
    Returns dict with keys: map50, map50_95, precision, recall (all in %, or None on failure).
    """
    from ultralytics import YOLO
    xml_files = list(ir_dir.glob("*.xml"))
    if not xml_files:
        log.error("[VAL] No .xml found in %s", ir_dir)
        return {"map50": None, "map50_95": None, "precision": None, "recall": None}
    log.info("[VAL] Validating %s on split=%s", xml_files[0].name, VAL_SPLIT)
    try:
        model = YOLO(str(ir_dir))
        t0 = time.perf_counter()
        m = model.val(data=data_path, split=VAL_SPLIT, imgsz=imgsz, batch=batch, seed=SEED, verbose=False, device="cpu")
        log.info("[VAL] %.1fs — mAP@50: %.4f | P: %.4f | R: %.4f", time.perf_counter() - t0, m.box.map50, m.box.mp, m.box.mr)
        return {
            "map50":    float(m.box.map50) * 100,
            "map50_95": float(m.box.map)   * 100,
            "precision": float(m.box.mp)   * 100,
            "recall":   float(m.box.mr)    * 100,
        }
    except Exception:
        log.exception("[VAL] Validation failed for %s", ir_dir)
        return {"map50": None, "map50_95": None, "precision": None, "recall": None}


def get_size_mb(ir_dir: Path) -> float:
    return sum(f.stat().st_size for f in ir_dir.iterdir() if f.suffix in (".xml", ".bin")) / 1_048_576


def apply_fallback_rule(fp32_map50: Optional[float], int8_map50: Optional[float]) -> tuple[str, str]:
    """Apply Chapter 3 §3.6.4 fallback rule. Returns (decision, reason)."""
    if int8_map50 is None or fp32_map50 is None:
        return "FP16", "INT8 export or validation failed"
    delta = fp32_map50 - int8_map50
    if int8_map50 < INT8_MIN_MAP50:
        return "FP16", f"INT8 mAP@0.5 ({int8_map50:.2f}%) < {INT8_MIN_MAP50}% threshold"
    if delta > INT8_MAX_DELTA_PP:
        return "FP16", f"delta ({delta:.2f} pp) > {INT8_MAX_DELTA_PP} pp tolerance"
    return "INT8", f"INT8 mAP@0.5 ({int8_map50:.2f}%) ≥ 90% and delta ({delta:.2f} pp) ≤ 1 pp"


def _f(val: Optional[float]) -> str:
    return f"{val:.2f}" if val is not None else "—"


def write_report(output_dir: Path, rows: list[dict]) -> Path:
    """Generate docs/quantization_report.md from accumulated per-precision rows using JSON to merge."""
    import json
    report_path = output_dir / "quantization_report.md"
    json_path = output_dir / "quantization_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing results
    all_rows = []
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                all_rows = json.load(f)
        except Exception as e:
            log.warning("Could not read %s: %s", json_path, e)

    # Filter out existing rows for the current variant to prevent duplicates
    if rows:
        current_variant = rows[0]["variant_label"]
        all_rows = [r for r in all_rows if r.get("variant_label") != current_variant]

    # Append new rows
    all_rows.extend(rows)

    # Sort rows by variant (Nano, Small, Medium) and precision (FP32, FP16, INT8)
    def sort_key(r):
        v = r.get("variant_label", "")
        p = r.get("prec_label", "")
        # Order: Nano -> Small -> Medium
        v_order = 0 if "Nano" in v else (1 if "Small" in v else 2)
        # Order: FP32 -> FP16 -> INT8
        p_order = 0 if p == "FP32" else (1 if p == "FP16" else 2)
        return (v_order, p_order)

    all_rows.sort(key=sort_key)

    # Save to JSON backing store
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2)
    except Exception as e:
        log.warning("Could not write %s: %s", json_path, e)

    # Generate Markdown lines
    lines = [
        "# CIRCA Phase 5 — OpenVINO Quantisation Report\n\n",
        "> Auto-generated by `scripts/evaluate_quantization.py`  \n",
        f"> Validation split: `{VAL_SPLIT}` | imgsz: {IMGSZ} | seed: {SEED}  \n",
        "> Fallback rule (Ch3 §3.6.4): INT8 mAP@0.5 ≥ 90% AND Δ ≤ 1 pp vs FP32\n\n---\n\n",
        "**Table: FP32 / FP16 / INT8 Validation Results**\n\n",
        "| Variant | Precision | mAP@0.5 (%) | mAP@0.5:0.95 (%) | P (%) | R (%) | Size (MB) | Δ vs FP32 (pp) | Fallback? |\n",
        "|:--|:--|--:|--:|--:|--:|--:|--:|:--|\n",
    ]
    for r in all_rows:
        delta_str = f"{r['delta_pp']:+.2f}" if r.get("delta_pp") is not None else "—"
        fb = ("✅ No" if r["decision"] == "INT8" else "⚠️ → FP16") if r["prec_label"] == "INT8" else "—"
        lines.append(
            f"| {r['variant_label']} | {r['prec_label']} "
            f"| {_f(r.get('map50'))} | {_f(r.get('map50_95'))} "
            f"| {_f(r.get('precision'))} | {_f(r.get('recall'))} "
            f"| {_f(r.get('size_mb'))} | {delta_str} | {fb} |\n"
        )
    lines += [
        "\n---\n\n**Per-Variant Fallback Decisions**\n\n",
        "| Variant | Chosen Format | Reason |\n|:--|:--|:--|\n",
    ]
    for r in all_rows:
        if r["prec_label"] == "INT8":
            icon = "✅" if r["decision"] == "INT8" else "⚠️"
            lines.append(f"| {r['variant_label']} | {icon} **{r['decision']}** | {r['reason']} |\n")
    lines.append("\n---\n*Source: `scripts/evaluate_quantization.py`*\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    log.info("[REPORT] Written → %s", report_path)
    return report_path



def run() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        log.error("Weights not found: %s", weights_path)
        sys.exit(1)

    _, variant_suffix = VARIANT_ID_MAP[args.variant]
    variant_label = f"YOLOv12-{args.variant.upper()} ({variant_suffix})"
    output_dir = Path(args.output_dir)
    rows = []
    results: dict = {}

    log.info("=" * 60)
    log.info("Phase 5 — Quantisation Evaluation: %s", variant_label)
    log.info("Weights: %s | Data: %s", weights_path, args.data)
    log.info("=" * 60)

    # FP32
    fp32_dir = export_model(weights_path, "fp32", args.data, args.imgsz)
    if not (fp32_dir and fp32_dir.exists()):
        log.error("FP32 export failed — cannot continue")
        sys.exit(1)
    m = validate_ir(fp32_dir, args.data, args.imgsz, args.batch)
    results["fp32"] = {**m, "size_mb": get_size_mb(fp32_dir), "ir_dir": str(fp32_dir)}
    rows.append({**m, "prec_label": "FP32", "variant_label": variant_label,
                 "size_mb": results["fp32"]["size_mb"], "delta_pp": 0.0, "decision": "—", "reason": "Baseline"})

    # FP16
    fp16_dir = export_model(weights_path, "fp16", args.data, args.imgsz)
    if fp16_dir and fp16_dir.exists():
        m = validate_ir(fp16_dir, args.data, args.imgsz, args.batch)
        sz = get_size_mb(fp16_dir)
        results["fp16"] = {**m, "size_mb": sz, "ir_dir": str(fp16_dir)}
        delta = (results["fp32"]["map50"] or 0) - (m["map50"] or 0)
        rows.append({**m, "prec_label": "FP16", "variant_label": variant_label,
                     "size_mb": sz, "delta_pp": delta, "decision": "—", "reason": "Reference"})

    # INT8
    int8_dir = export_model(weights_path, "int8", args.data, args.imgsz)
    if int8_dir and int8_dir.exists():
        m = validate_ir(int8_dir, args.data, args.imgsz, args.batch)
        sz = get_size_mb(int8_dir)
        decision, reason = apply_fallback_rule(results["fp32"].get("map50"), m.get("map50"))
        results["int8"] = {**m, "size_mb": sz, "ir_dir": str(int8_dir), "decision": decision}
        delta = (results["fp32"]["map50"] or 0) - (m["map50"] or 0)
        rows.append({**m, "prec_label": "INT8", "variant_label": variant_label,
                     "size_mb": sz, "delta_pp": delta, "decision": decision, "reason": reason})
    else:
        results["int8"] = {"decision": "FP16", "reason": "INT8 export failed"}
        rows.append({"prec_label": "INT8", "variant_label": variant_label, "map50": None, "map50_95": None,
                     "precision": None, "recall": None, "size_mb": None, "delta_pp": None,
                     "decision": "FP16", "reason": "INT8 export failed"})

    dec = results["int8"]["decision"]
    log.info("\n=== VERDICT: %s → Deploy as %s ===", variant_label, dec)
    write_report(output_dir, rows)


if __name__ == "__main__":
    run()
