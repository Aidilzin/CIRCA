#!/bin/bash
# CIRCA RunPod setup script (simplified)
# -------------------------------------------------------
# Both datasets (unified_pcb_v3 and unified_pcb_v3_preproc)
# are pre-baked into the zip -- no preprocessing needed on the pod.
#
# Steps:
#   1. Install requirements
#   2. Run diagnostic (verify GPU, deps, dataset)
#
# Run from /workspace/CIRCA:
#   bash scripts/runpod_setup.sh
# -------------------------------------------------------

set -e   # exit on first error

echo "=========================================================="
echo "       CIRCA RunPod Environment Setup (Simplified)        "
echo "=========================================================="
echo ""

# 1. Install requirements
echo "[1/2] Installing requirements from requirements_runpod.txt..."
pip install --upgrade pip --quiet
pip install -r requirements_runpod.txt --quiet
echo "      Done."

# 2. Run diagnostic
echo ""
echo "[2/2] Running CIRCA verification diagnostic..."
python scripts/runpod_setup.py

echo ""
echo "=========================================================="
echo "  Setup complete. Both datasets are ready:"
echo "    datasets/unified_pcb_v3          (raw + oversampled)"
echo "    datasets/unified_pcb_v3_preproc  (CLAHE+Gamma + oversampled)"
echo ""
echo "  Next steps:"
echo "    wandb login <YOUR_API_KEY>"
echo ""
echo "  Phase 1 (Vanilla Baseline):"
echo "    python train_engine.py --mode train --variant s \\"
echo "      --id 001 --desc Baseline_Vanilla \\"
echo "      --epochs 100 --batch 24 \\"
echo "      --data datasets/unified_pcb_v3/data.yaml"
echo ""
echo "  Phase 2 (CIRCA Preprocessed):"
echo "    python train_engine.py --mode train --variant s \\"
echo "      --id 002 --desc Baseline_CIRCA \\"
echo "      --epochs 100 --batch 24 \\"
echo "      --data datasets/unified_pcb_v3_preproc/data.yaml"
echo "=========================================================="
