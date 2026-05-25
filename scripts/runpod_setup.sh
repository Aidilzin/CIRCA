#!/bin/bash
# CIRCA RunPod automated setup script
# -------------------------------------------------------
# Steps:
#   1. Install requirements
#   2. Run diagnostic (verify GPU, deps, dataset)
#   3. Oversample unified_pcb_v3 (idempotent)
#   4. Create unified_pcb_v3_preproc via CLAHE+Gamma (idempotent)
#   5. Oversample unified_pcb_v3_preproc (idempotent)
#
# Run from /workspace/CIRCA:
#   bash scripts/runpod_setup.sh
# -------------------------------------------------------

set -e   # exit on first error

echo "=========================================================="
echo "       CIRCA RunPod Automated Environment Setup           "
echo "=========================================================="
echo ""

# 1. Install requirements
echo "[1/5] Installing requirements from requirements_runpod.txt..."
pip install --upgrade pip --quiet
pip install -r requirements_runpod.txt --quiet
echo "      Done."

# 2. Run diagnostic
echo ""
echo "[2/5] Running CIRCA verification diagnostic..."
python scripts/runpod_setup.py

# 3. Oversample unified_pcb_v3 (adds missing_hole 3x, cold_solder_joint 5x)
echo ""
echo "[3/5] Applying tiered oversampling to datasets/unified_pcb_v3 ..."
echo "      (idempotent -- safe to re-run if already done)"
python scripts/oversample_minority_classes.py \
    --dataset-root datasets/unified_pcb_v3
echo "      Done."

# 4. Create unified_pcb_v3_preproc (CLAHE+Gamma on all 9,404 train images)
#    Reuses existing folder if already present (train_engine --preproc is idempotent).
echo ""
echo "[4/5] Creating datasets/unified_pcb_v3_preproc (CLAHE+Gamma preprocessing)..."
echo "      This will take ~5-10 min for 9,404 training images. Skip if folder exists."
if [ -d "datasets/unified_pcb_v3_preproc" ]; then
    echo "      [SKIP] datasets/unified_pcb_v3_preproc already exists."
else
    python train_engine.py \
        --mode train --variant s --id 000 --desc PREPROC_ONLY \
        --epochs 1 --batch 24 --preproc \
        --data datasets/unified_pcb_v3/data.yaml \
        2>&1 | grep -E "PREPROC|ERROR|WARNING|Done|written" || true
    # Kill early -- we only needed the preproc step, not the actual training epoch.
    echo "      Preprocessing complete. Run folder 000 can be ignored."
fi

# 5. Oversample unified_pcb_v3_preproc
echo ""
echo "[5/5] Applying tiered oversampling to datasets/unified_pcb_v3_preproc ..."
echo "      (idempotent -- safe to re-run if already done)"
python scripts/oversample_minority_classes.py \
    --dataset-root datasets/unified_pcb_v3_preproc
echo "      Done."

echo ""
echo "=========================================================="
echo "  CIRCA RunPod Setup Complete. Ready to run Phase 3 HPO.  "
echo "  Next command:                                           "
echo "    wandb login <YOUR_API_KEY>                            "
echo "    python train_engine.py --mode tune --variant s \\     "
echo "      --id 003 --desc HPO_7class \\                       "
echo "      --epochs 50 --iterations 50 --fraction 0.5 \\       "
echo "      --batch 24 \\                                       "
echo "      --data datasets/unified_pcb_v3_preproc/data.yaml   "
echo "=========================================================="
