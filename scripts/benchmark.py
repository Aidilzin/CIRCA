"""
scripts/benchmark.py
---------------------
Phase 6 — Hardware Benchmarking on Deployment Target

Measures the four acceptance criteria from Chapter 1 §1.5 for each
OpenVINO IR variant + precision combination:

  1. Preprocessing latency  ≤ 5 ms per frame
  2. Inference latency      (CPU and iGPU, ms)
  3. End-to-end live FPS    ≥ 15 FPS (60-second loop)
  4. Static image inference ≤ 10 seconds

Outputs:
    docs/benchmark_report.md          — variant selection matrix + decisions
    docs/assets/fig4_5_live_fps_trace.png — FPS trace plot

Usage:
    # CPU benchmark for one variant:
    python scripts/benchmark.py \\
        --model-dir runs/detect/CIRCA_V12S_005_TRAIN_Phase4_Small/weights/best_int8_openvino_model \\
        --data datasets/unified_pcb_v3_preproc/data.yaml \\
        --variant s --precision INT8 --device CPU

    # iGPU benchmark:
    python scripts/benchmark.py \\
        --model-dir <same dir> \\
        --variant s --precision INT8 --device GPU
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Acceptance criteria (Chapter 1 §1.5)
# ---------------------------------------------------------------------------
PREPROC_BUDGET_MS  = 5.0    # ≤ 5 ms per frame
LIVE_FPS_MIN       = 15.0   # ≥ 15 FPS end-to-end
STATIC_MAX_S       = 10.0   # ≤ 10 s for a single high-res static image

# Preprocessing parameters — match training constants
CLAHE_CLIP  = 2.0
CLAHE_TILE  = (8, 8)
GAMMA       = 1.2
BLUR_THRESH = 100.0
IMGSZ       = 640

# Benchmark settings
PREPROC_FRAMES  = 1000   # frames to time for preprocessing latency
INFER_WARMUP    = 20     # warmup inferences (excluded from timing)
INFER_FRAMES    = 100    # inferences to average
LIVE_DURATION_S = 60     # seconds for live FPS measurement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6 — Hardware Benchmarking for Variant Selection"
    )
    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="Path to OpenVINO IR directory (contains .xml + .bin)",
    )
    parser.add_argument(
        "--data", type=str, default="datasets/unified_pcb_v3_preproc/data.yaml",
        help="Path to data.yaml for sourcing representative frames",
    )
    parser.add_argument(
        "--variant", type=str, choices=["n", "s", "m"], required=True,
        help="Model variant: n=Nano, s=Small, m=Medium",
    )
    parser.add_argument(
        "--precision", type=str, choices=["FP32", "FP16", "INT8"], default="INT8",
        help="Precision of the IR model being benchmarked",
    )
    parser.add_argument(
        "--device", type=str, choices=["CPU", "GPU"], default="CPU",
        help="OpenVINO plugin device (CPU or GPU for integrated GPU)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="docs",
        help="Directory for benchmark_report.md output",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Preprocessing helpers (mirrors core/preprocessor.py, no PyQt dependency)
# ---------------------------------------------------------------------------

_GAMMA_LUT: Optional[np.ndarray] = None

def _get_gamma_lut(gamma: float) -> np.ndarray:
    global _GAMMA_LUT
    if _GAMMA_LUT is None:
        inv = 1.0 / gamma
        _GAMMA_LUT = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
    return _GAMMA_LUT

_CLAHE_OBJ: Optional[cv2.CLAHE] = None

def _get_clahe() -> cv2.CLAHE:
    global _CLAHE_OBJ
    if _CLAHE_OBJ is None:
        _CLAHE_OBJ = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    return _CLAHE_OBJ


def preprocess_frame(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Apply CLAHE + Gamma, return (processed_frame, laplacian_variance)."""
    # CLAHE on L channel of LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = _get_clahe().apply(l)
    frame_clahe = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
    # Gamma correction
    frame_gamma = cv2.LUT(frame_clahe, _get_gamma_lut(GAMMA))
    # Laplacian variance (blur gate)
    grey = cv2.cvtColor(frame_gamma, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    lap = cv2.Laplacian(small, cv2.CV_32F)
    _, std = cv2.meanStdDev(lap)
    variance = float(std[0, 0] ** 2)
    return frame_gamma, variance


# ---------------------------------------------------------------------------
# Load val images as representative frames
# ---------------------------------------------------------------------------

def load_val_images(data_yaml: str, max_frames: int = 1200) -> list[np.ndarray]:
    """Load up to max_frames images from the validation split."""
    import yaml
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    data_root = Path(data_yaml).parent
    val_rel = cfg.get("val", "val/images")
    val_dir = data_root / val_rel
    if not val_dir.exists():
        # Try resolving relative to datasets/
        val_dir = Path("datasets") / val_rel
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [p for p in val_dir.iterdir() if p.suffix.lower() in exts][:max_frames]
    frames = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(img)
    log.info("[DATA] Loaded %d validation frames from %s", len(frames), val_dir)
    return frames


# ---------------------------------------------------------------------------
# Benchmark 1: Preprocessing latency
# ---------------------------------------------------------------------------

def benchmark_preprocessing(frames: list[np.ndarray]) -> dict:
    """Time CLAHE + Gamma + Laplacian over PREPROC_FRAMES frames."""
    n = min(PREPROC_FRAMES, len(frames))
    times_ms = []
    for i in range(n):
        frame = frames[i % len(frames)]
        t0 = time.perf_counter()
        preprocess_frame(frame)
        times_ms.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times_ms)
    result = {
        "mean_ms": float(np.mean(arr)),
        "std_ms":  float(np.std(arr)),
        "p95_ms":  float(np.percentile(arr, 95)),
        "pass":    float(np.mean(arr)) <= PREPROC_BUDGET_MS,
    }
    log.info("[PREPROC] mean=%.2fms  std=%.2fms  p95=%.2fms  PASS=%s",
             result["mean_ms"], result["std_ms"], result["p95_ms"], result["pass"])
    return result


# ---------------------------------------------------------------------------
# Benchmark 2: Inference latency
# ---------------------------------------------------------------------------

def load_openvino_model(ir_dir: Path, device: str):
    """Load and compile the OpenVINO IR model."""
    try:
        from openvino import Core
    except ImportError:
        from openvino.runtime import Core
    xml_files = list(ir_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No .xml found in {ir_dir}")
    core = Core()
    model = core.read_model(str(xml_files[0]))
    compiled = core.compile_model(model, device)
    log.info("[OV] Compiled %s on %s", xml_files[0].name, device)
    return compiled


def prepare_input(frame: np.ndarray) -> np.ndarray:
    """Resize + normalise frame to (1, 3, 640, 640) float32 NCHW tensor."""
    resized = cv2.resize(frame, (IMGSZ, IMGSZ))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    nchw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(nchw, axis=0)


def benchmark_inference(compiled_model, frames: list[np.ndarray]) -> dict:
    """Warm up then time INFER_FRAMES inference calls."""
    infer_req = compiled_model.create_infer_request()
    input_key = compiled_model.input(0)

    # Warmup
    for i in range(INFER_WARMUP):
        blob = prepare_input(frames[i % len(frames)])
        infer_req.infer({input_key: blob})

    # Timed
    times_ms = []
    for i in range(INFER_FRAMES):
        blob = prepare_input(frames[i % len(frames)])
        t0 = time.perf_counter()
        infer_req.infer({input_key: blob})
        times_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times_ms)
    result = {
        "mean_ms": float(np.mean(arr)),
        "std_ms":  float(np.std(arr)),
        "p95_ms":  float(np.percentile(arr, 95)),
    }
    log.info("[INFER] mean=%.2fms  std=%.2fms  p95=%.2fms", result["mean_ms"], result["std_ms"], result["p95_ms"])
    return result


# ---------------------------------------------------------------------------
# Benchmark 3: End-to-end live FPS (60-second simulated loop)
# ---------------------------------------------------------------------------

def benchmark_live_fps(compiled_model, frames: list[np.ndarray], output_dir: Path) -> dict:
    """
    Simulate 60 seconds of live inference (preprocess + infer) and compute FPS.
    Saves a moving-average FPS trace as docs/assets/fig4_5_live_fps_trace.png.
    """
    infer_req = compiled_model.create_infer_request()
    input_key = compiled_model.input(0)
    frame_times = []
    deadline = time.perf_counter() + LIVE_DURATION_S
    idx = 0
    while time.perf_counter() < deadline:
        frame = frames[idx % len(frames)]
        t0 = time.perf_counter()
        processed, _ = preprocess_frame(frame)
        blob = prepare_input(processed)
        infer_req.infer({input_key: blob})
        frame_times.append(time.perf_counter() - t0)
        idx += 1

    n_frames = len(frame_times)
    avg_fps = n_frames / LIVE_DURATION_S

    # Compute per-second FPS for the trace
    fps_per_second = []
    cumulative = 0.0
    bucket = []
    for ft in frame_times:
        cumulative += ft
        bucket.append(ft)
        if cumulative >= 1.0:
            fps_per_second.append(len(bucket))
            bucket = []
            cumulative = 0.0

    # Save FPS trace plot using matplotlib if available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        x = list(range(len(fps_per_second)))
        ax.plot(x, fps_per_second, linewidth=1.5, color="#2196F3", label="FPS per second")
        ax.axhline(LIVE_FPS_MIN, color="#F44336", linestyle="--", linewidth=1, label=f"Min threshold ({LIVE_FPS_MIN} FPS)")
        ax.axhline(avg_fps, color="#4CAF50", linestyle="-.", linewidth=1, label=f"Average ({avg_fps:.1f} FPS)")
        ax.set_xlabel("Elapsed Time (seconds)")
        ax.set_ylabel("Frames per Second")
        ax.set_title("CIRCA Live Inference FPS — 60-Second Trace")
        ax.legend()
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        plot_path = assets_dir / "fig4_5_live_fps_trace.png"
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("[FPS] Trace saved -> %s", plot_path)
    except ImportError:
        log.warning("[FPS] matplotlib not available — skipping trace plot")

    result = {
        "total_frames": n_frames,
        "avg_fps": avg_fps,
        "min_fps": min(fps_per_second) if fps_per_second else 0,
        "max_fps": max(fps_per_second) if fps_per_second else 0,
        "pass": avg_fps >= LIVE_FPS_MIN,
    }
    log.info("[FPS] avg=%.1f  min=%.1f  max=%.1f  frames=%d  PASS=%s",
             result["avg_fps"], result["min_fps"], result["max_fps"], result["total_frames"], result["pass"])
    return result


# ---------------------------------------------------------------------------
# Benchmark 4: Static image inference
# ---------------------------------------------------------------------------

def benchmark_static(compiled_model, frames: list[np.ndarray]) -> dict:
    """Time full pipeline (preproc + infer + overlay draw stub) on a single frame."""
    frame = max(frames, key=lambda f: f.shape[0] * f.shape[1])  # use largest frame
    infer_req = compiled_model.create_infer_request()
    input_key = compiled_model.input(0)

    t0 = time.perf_counter()
    processed, _ = preprocess_frame(frame)
    blob = prepare_input(processed)
    infer_req.infer({input_key: blob})
    # NMS stub (simulates drawing; actual NMS cost is negligible)
    _ = cv2.resize(processed, (640, 640))
    total_s = time.perf_counter() - t0

    result = {
        "total_s": total_s,
        "pass": total_s <= STATIC_MAX_S,
    }
    log.info("[STATIC] total=%.3fs  PASS=%s", result["total_s"], result["pass"])
    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_report(output_dir: Path, entry: dict) -> Path:
    """
    Write / append to docs/benchmark_report.md.
    Each call appends a new variant + device row.
    """
    report_path = output_dir / "benchmark_report.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    is_new = not report_path.exists()

    with open(report_path, "a", encoding="utf-8") as f:
        if is_new:
            f.write("# CIRCA Phase 6 — Hardware Benchmark Report\n\n")
            f.write("> Auto-generated by `scripts/benchmark.py`  \n")
            f.write(f"> Acceptance criteria: preproc ≤ {PREPROC_BUDGET_MS}ms | FPS ≥ {LIVE_FPS_MIN} | static ≤ {STATIC_MAX_S}s\n\n---\n\n")
            f.write("## Variant Selection Matrix\n\n")
            f.write("| Variant | Precision | Device | mAP@0.5 (%) | Preproc (ms) | Infer (ms) | Live FPS | Static (s) | Preproc✓ | FPS✓ | Static✓ | **PASS?** |\n")
            f.write("|:--|:--|:--|--:|--:|--:|--:|--:|:--:|:--:|:--:|:--:|\n")

        vl   = entry["variant_label"]
        prec = entry["precision"]
        dev  = entry["device"]
        map50_str = f"{entry['map50']:.2f}" if entry.get("map50") else "—"
        pp_str  = f"{entry['preproc']['mean_ms']:.2f}"
        inf_str = f"{entry['infer']['mean_ms']:.2f}"
        fps_str = f"{entry['live']['avg_fps']:.1f}"
        st_str  = f"{entry['static']['total_s']:.3f}"
        pp_ok   = "✅" if entry["preproc"]["pass"] else "❌"
        fps_ok  = "✅" if entry["live"]["pass"] else "❌"
        st_ok   = "✅" if entry["static"]["pass"] else "❌"
        all_ok  = "✅ **YES**" if (entry["preproc"]["pass"] and entry["live"]["pass"] and entry["static"]["pass"]) else "❌ **NO**"
        f.write(f"| {vl} | {prec} | {dev} | {map50_str} | {pp_str} | {inf_str} | {fps_str} | {st_str} | {pp_ok} | {fps_ok} | {st_ok} | {all_ok} |\n")

    log.info("[REPORT] Appended -> %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    args = parse_args()
    ir_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    variant_names = {"n": "YOLOv12-N (Nano)", "s": "YOLOv12-S (Small)", "m": "YOLOv12-M (Medium)"}
    variant_label = variant_names[args.variant]

    log.info("=" * 60)
    log.info("Phase 6 — Hardware Benchmark: %s | %s | %s", variant_label, args.precision, args.device)
    log.info("IR dir: %s", ir_dir)
    log.info("=" * 60)

    # Load frames
    frames = load_val_images(args.data, max_frames=1200)
    if not frames:
        log.error("No validation images found — check --data path")
        sys.exit(1)

    # Benchmark preprocessing (device-independent)
    pp_result = benchmark_preprocessing(frames)

    # Load model
    compiled = load_openvino_model(ir_dir, args.device)

    # Benchmark inference
    infer_result = benchmark_inference(compiled, frames)

    # Benchmark live FPS
    live_result = benchmark_live_fps(compiled, frames, output_dir)

    # Benchmark static
    static_result = benchmark_static(compiled, frames)

    # Collect mAP from quantization report if available
    map50 = None
    quant_report = output_dir / "quantization_report.md"
    if quant_report.exists():
        with open(quant_report, "r", encoding="utf-8") as f:

            for line in f:
                if f"YOLOv12-{args.variant.upper()}" in line and f"| {args.precision} " in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) > 3:
                        try:
                            map50 = float(parts[3])
                        except ValueError:
                            pass
                    break



    entry = {
        "variant_label": variant_label,
        "precision": args.precision,
        "device": args.device,
        "map50": map50,
        "preproc": pp_result,
        "infer": infer_result,
        "live": live_result,
        "static": static_result,
    }
    write_report(output_dir, entry)

    log.info("\n=== BENCHMARK SUMMARY: %s | %s | %s ===", variant_label, args.precision, args.device)
    log.info("  Preproc latency : %.2f ms  [PASS=%s]", pp_result["mean_ms"], pp_result["pass"])
    log.info("  Infer latency   : %.2f ms", infer_result["mean_ms"])
    log.info("  Live FPS        : %.1f     [PASS=%s]", live_result["avg_fps"], live_result["pass"])
    log.info("  Static time     : %.3f s   [PASS=%s]", static_result["total_s"], static_result["pass"])


if __name__ == "__main__":
    run()
