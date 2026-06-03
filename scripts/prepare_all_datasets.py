"""
prepare_all_datasets.py
========================
CIRCA -- Full local dataset preparation pipeline.

Runs in sequence:
  1.   Rebuild unified_pcb_v3 from source images (clean slate)
  1.5  Cap dominant-only images (reduce 9.9:1 ratio → ~5.7:1)
  2.   Oversample unified_pcb_v3 (tiered minority-class duplication)
  3.   Build unified_pcb_v3_preproc (CLAHE + Gamma on all images)
  4.   Oversample unified_pcb_v3_preproc

After this script completes, both datasets are ready to be zipped
and uploaded to RunPod without any further setup on the pod.

Usage (run from project root d:/FYP/CIRCA):
    python scripts/prepare_all_datasets.py
    python scripts/prepare_all_datasets.py --force        # force rebuild of preproc even if exists
    python scripts/prepare_all_datasets.py --skip-rebuild # skip step 1, keep existing unified_pcb_v3
    python scripts/prepare_all_datasets.py --cap 700      # use a lower cap for more aggressive balancing
"""

import sys
import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_V3   = PROJECT_ROOT / "datasets" / "unified_pcb_v3"
DATASET_PREPROC = PROJECT_ROOT / "datasets" / "unified_pcb_v3_preproc"
DATA_YAML    = DATASET_V3 / "data.yaml"


def run(cmd: list, step: str) -> None:
    """Run a subprocess command, raise on failure."""
    log.info("[%s] Running: %s", step, " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        log.error("[%s] FAILED with exit code %d", step, result.returncode)
        sys.exit(result.returncode)
    log.info("[%s] Done.", step)


def step_rebuild_dataset() -> None:
    print("\n" + "=" * 60)
    print("  STEP 1 -- Rebuilding unified_pcb_v3 from source")
    print("=" * 60)
    run([sys.executable, "scripts/build_unified_pcb_v3.py"], step="1/4")


def step_cap_dominant(cap: int) -> None:
    print("\n" + "=" * 60)
    print(f"  STEP 1.5 -- Capping dominant-only images (cap={cap})")
    print("=" * 60)
    run(
        [sys.executable, "scripts/cap_dominant_classes.py",
         "--dataset-root", str(DATASET_V3),
         "--cap", str(cap),
         "--seed", "42"],
        step="1.5/5"
    )


def step_oversample_raw() -> None:
    print("\n" + "=" * 60)
    print("  STEP 2 -- Oversampling unified_pcb_v3")
    print("=" * 60)
    run(
        [sys.executable, "scripts/oversample_minority_classes.py",
         "--dataset-root", str(DATASET_V3)],
        step="2/5"
    )


def get_python_exe() -> str:
    """Return the venv Python if it exists, otherwise fall back to the current interpreter."""
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        log.info("[preproc] Using venv Python: %s", venv_python)
        return str(venv_python)
    log.warning("[preproc] Venv not found at %s, falling back to sys.executable: %s", venv_python, sys.executable)
    return sys.executable


def step_build_preproc(force: bool) -> None:
    print("\n" + "=" * 60)
    print("  STEP 3 -- Building unified_pcb_v3_preproc (CLAHE+Gamma)")
    print("=" * 60)

    if DATASET_PREPROC.exists() and not force:
        log.info("[3/4] unified_pcb_v3_preproc already exists -- skipping (use --force to rebuild).")
        return

    if DATASET_PREPROC.exists() and force:
        import shutil
        log.info("[3/4] --force flag set, removing existing %s", DATASET_PREPROC)
        shutil.rmtree(DATASET_PREPROC)

    # Use the venv Python so cv2/ultralytics deps are available
    python_exe = get_python_exe()

    # Run preprocess_dataset via a small inline script to avoid a full training epoch
    inline_script = (
        "import sys; sys.path.insert(0, r'{root}'); "
        "from train_engine import preprocess_dataset; "
        "result = preprocess_dataset(r'{yaml}', force=False); "
        "print('[3/4] Preprocessed data.yaml:', result)"
    ).format(root=str(PROJECT_ROOT), yaml=str(DATA_YAML))

    import os
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    log.info("[3/4] Starting CLAHE+Gamma preprocessing pipeline...")
    result = subprocess.run(
        [python_exe, "-c", inline_script],
        cwd=PROJECT_ROOT,
        env=env,
    )
    if result.returncode != 0:
        log.error("[3/5] Preprocessing FAILED with exit code %d", result.returncode)
        sys.exit(result.returncode)
    log.info("[3/5] Done.")


def step_oversample_preproc() -> None:
    print("\n" + "=" * 60)
    print("  STEP 4 -- Oversampling unified_pcb_v3_preproc")
    print("=" * 60)
    run(
        [sys.executable, "scripts/oversample_minority_classes.py",
         "--dataset-root", str(DATASET_PREPROC)],
        step="4/5"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CIRCA -- Full local dataset preparation pipeline (steps 1-4)."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force rebuild of unified_pcb_v3_preproc even if it already exists."
    )
    parser.add_argument(
        "--skip-rebuild", action="store_true",
        help="Skip step 1 (dataset rebuild). Use if unified_pcb_v3 sources haven't changed."
    )
    parser.add_argument(
        "--cap", type=int, default=1000,
        help="Max dominant-only images to keep in step 1.5 (default: 1000 → ~5.7:1 ratio). Set to 0 to skip capping."
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  CIRCA Dataset Preparation Pipeline")
    print("  Steps: rebuild -> cap -> oversample -> preproc -> oversample")
    print("=" * 60)

    if not args.skip_rebuild:
        step_rebuild_dataset()
    else:
        log.info("[1/5] SKIPPED (--skip-rebuild). Using existing unified_pcb_v3.")

    if args.cap > 0:
        step_cap_dominant(args.cap)
    else:
        log.info("[1.5/5] SKIPPED (--cap 0). Dominant-class capping disabled.")

    step_oversample_raw()
    step_build_preproc(force=args.force)
    step_oversample_preproc()

    print("\n" + "=" * 60)
    print("  ALL STEPS COMPLETE")
    print(f"  Raw dataset    : {DATASET_V3}")
    print(f"  Preproc dataset: {DATASET_PREPROC}")
    print("  Both datasets are ready to package for RunPod.")
    print("  Next: powershell -File scripts/package_runpod.ps1")
    print("=" * 60 + "\n")
    print(f"  Cap used: {args.cap} dominant-only images kept")


if __name__ == "__main__":
    main()
