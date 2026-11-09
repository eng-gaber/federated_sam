#!/usr/bin/env bash
# ==============================================================================
# Run Fed-FGS with SAM ViT-B + LoRA (r=8, 30 Rounds Default)
# Paper: "Fed-FGS: Domain-Generalized Federated Polyp Segmentation via
#        Fourier Hard-Thresholding and Gradient Scaling" (Gaber et al.)
#
# Usage:
#   bash scripts/run_fed_fgs.sh                                  # Default: 30 rounds, r=8
#   bash scripts/run_fed_fgs.sh --config path/to/config.yaml     # Custom YAML config
#   bash scripts/run_fed_fgs.sh 30 8 my_run_name                 # Positional: ROUNDS RANK RUN_NAME
# ==============================================================================

set -euo pipefail

# Ensure we are in project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs checkpoints/runs

DEFAULT_CONFIG="federated/configs/fed_fgs_r8_30rounds.yaml"
export PYTHONUNBUFFERED=1

if [[ "${1:-}" == "--config" && -n "${2:-}" ]]; then
    CONFIG_FILE="$2"
    LOG_FILE="logs/fed_fgs_$(basename "${CONFIG_FILE}" .yaml).log"
    echo "==========================================================================="
    echo "🚀 Starting Fed-FGS with Config File: ${CONFIG_FILE}"
    echo "   Live Log: ${LOG_FILE}"
    echo "==========================================================================="
    conda run --no-capture-output -n polypsamfl python -u flower_sim.py --config "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}"

elif [[ -f "${DEFAULT_CONFIG}" && $# -eq 0 ]]; then
    LOG_FILE="logs/fed_fgs_r8_30rounds.log"
    echo "==========================================================================="
    echo "🚀 Starting Fed-FGS (SAM ViT-B + LoRA r=8, 30 Rounds)"
    echo "   Using Default Config: ${DEFAULT_CONFIG}"
    echo "   Live Log:             ${LOG_FILE}"
    echo "==========================================================================="
    conda run --no-capture-output -n polypsamfl python -u flower_sim.py --config "${DEFAULT_CONFIG}" 2>&1 | tee "${LOG_FILE}"

else
    ROUNDS="${1:-30}"
    RANK="${2:-8}"
    RUN_NAME="${3:-fed_fgs_sam_vit_b_r${RANK}_${ROUNDS}rounds}"
    LOCAL_EPOCHS="${4:-2}"
    BATCH_SIZE="${5:-2}"
    LR="${6:-0.0002}"
    DFTHT_PROB="${7:-0.3}"
    FOLD_NR="${8:-1}"
    LOG_FILE="logs/${RUN_NAME}.log"

    echo "==========================================================================="
    echo "🚀 Starting Fed-FGS (SAM ViT-B + LoRA r=${RANK})"
    echo "   Run Name:        ${RUN_NAME}"
    echo "   FL Rounds:       ${ROUNDS}"
    echo "   LoRA Rank:       ${RANK}"
    echo "   Local Epochs:    ${LOCAL_EPOCHS}"
    echo "   Batch Size:      ${BATCH_SIZE}"
    echo "   Learning Rate:   ${LR} (Cosine decay to 1e-5)"
    echo "   DFT-HT Prob:     ${DFTHT_PROB}"
    echo "   Fold:            Fold ${FOLD_NR}"
    echo "   Output Run Dir:  checkpoints/runs/${RUN_NAME}"
    echo "   Live Log:        ${LOG_FILE}"
    echo "==========================================================================="

    conda run --no-capture-output -n polypsamfl python -u flower_sim.py \
        --run_name "${RUN_NAME}" \
        --num_rounds "${ROUNDS}" \
        --lora_rank "${RANK}" \
        --local_epochs "${LOCAL_EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${LR}" \
        --dftht_prob "${DFTHT_PROB}" \
        --num_clients 5 \
        --out_center 5 \
        --split_type fold \
        --fold_nr "${FOLD_NR}" \
        --if_update_encoder True \
        --if_encoder_lora_layer True \
        --if_decoder_lora_layer True \
        --lr_schedule cosine \
        --min_lr 1e-5 2>&1 | tee "${LOG_FILE}"
fi
