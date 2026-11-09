#!/usr/bin/env bash
# ==============================================================================
# Sequential Ablation Runner:
# 1. SAM + FedAvg + Pure Dice Loss (20 Rounds) - fsm_fedavg_20round.yaml
# 2. SAM + FedAvg + Composite Loss (20 Rounds) - ablation_sam_fedavg_composite_loss_r8_20rounds.yaml
# 3. SAM + FedAvg + DFT-HT (20 Rounds)        - ablation_sam_fedavg_dftht_r8_20rounds.yaml
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs checkpoints/runs

CONFIG_BASE="federated/configs/fsm_fedavg_20round.yaml"
CONFIG_COMPOSITE="federated/configs/ablation_sam_fedavg_composite_loss_r8_20rounds.yaml"
CONFIG_DFTHT="federated/configs/ablation_sam_fedavg_dftht_r8_20rounds.yaml"

# If user passes arguments, run those; otherwise run the full pipeline
RUN_ALL=true
if [[ "${1:-}" == "--composite-and-dftht" ]]; then
    RUN_ALL=false
fi

if [[ "${RUN_ALL}" == "true" ]]; then
    echo "==========================================================================="
    echo "🔬 [STAGE 1/3] Starting Run 1: SAM + FedAvg (Pure Dice Loss, 20 Rounds)"
    echo "   Config: ${CONFIG_BASE}"
    echo "==========================================================================="
    ./run_fed_fgs.sh --config "${CONFIG_BASE}"
    echo "✅ Stage 1 Completed Successfully!"
fi

echo ""
echo "==========================================================================="
echo "🔬 Starting Stage: SAM + FedAvg + Composite Loss (20 Rounds)"
echo "   Config: ${CONFIG_COMPOSITE}"
echo "==========================================================================="
./run_fed_fgs.sh --config "${CONFIG_COMPOSITE}"
echo "✅ Stage Completed Successfully!"

echo ""
echo "==========================================================================="
echo "🔬 Starting Stage: SAM + FedAvg + DFT-HT (Pure Dice Loss, 20 Rounds)"
echo "   Config: ${CONFIG_DFTHT}"
echo "==========================================================================="
./run_fed_fgs.sh --config "${CONFIG_DFTHT}"
echo "✅ Stage Completed Successfully!"

echo ""
echo "==========================================================================="
echo "🎉 ALL ABLATION RUNS COMPLETED SUCCESSFULLY!"
echo "==========================================================================="
