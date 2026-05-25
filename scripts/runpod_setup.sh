#!/bin/bash
# CIRCA RunPod automated setup script

echo "=========================================================="
echo "          CIRCA RunPod Automated Environment Setup        "
echo "=========================================================="
echo ""

# 1. Update pip and install requirements
echo "[1/2] Installing requirements from requirements_runpod.txt..."
pip install --upgrade pip
pip install -r requirements_runpod.txt

# 2. Run Python diagnostic tool
echo ""
echo "[2/2] Running CIRCA verification diagnostic tool..."
python scripts/runpod_setup.py
