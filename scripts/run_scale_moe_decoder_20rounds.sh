#!/usr/bin/env bash
# ==============================================================================
# Scale-Aware Dual-Decoder MoE (20 Rounds) Runner
#
# Runs:
# SAM-FedGS + Composite Loss + Scale-Aware Dual-Decoder MoE (D_small, D_large)
# - Clean evaluation without DFT-HT to protect small-polyp frequency cues
# - beta = 0.5 (Scale specialization weight)
# - lambda_gate = 0.1 (Gating supervision cross-entropy weight)
# - 20 FL rounds, LoRA r=8, RTX 5090
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs checkpoints/runs
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}"

CONFIG="federated/configs/sam_fedgs_scale_decoder_moe_r8_20rounds.yaml"
LOG="logs/sam_fedgs_scale_decoder_moe_r8_20rounds.log"

echo "==========================================================================="
echo "🔬 Starting Experiment: Scale-Aware Dual-Decoder MoE (20 Rounds)"
echo "   Config:   ${CONFIG}"
echo "   Live Log: ${LOG}"
echo "==========================================================================="

RESUME_FLAG=""
if [ -f "checkpoints/runs/sam_fedgs_scale_decoder_moe_r8_20rounds/FedFGS_SAM_last.pth" ]; then
    echo "🔁 Found existing checkpoint from previous rounds! Automatically resuming..."
    RESUME_FLAG="--resume true"
fi

conda run --no-capture-output -n polypsamfl python -u federated/flower_sim.py --config "${CONFIG}" ${RESUME_FLAG} "$@" 2>&1 | tee -a "${LOG}"

echo ""
echo "==========================================================================="
echo "🎉 Scale-Aware Dual-Decoder MoE Experiment Complete!"
echo "   Log saved to: ${LOG}"
echo "==========================================================================="
