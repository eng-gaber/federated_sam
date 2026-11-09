#!/usr/bin/env bash
# ==============================================================================
# Master Automated Sequential Runner:
# Scale-Aware Dual-Decoder & Hierarchical Bi-Level MoE (20 Rounds Each, LoRA r=8)
#
# Sequential Execution:
# 1. SAM-FedGS + Scale-Aware Dual Decoder (No DFT-HT) [Resumes from Rd 4 -> 20]
# 2. Hierarchical Bi-Level MoE (Encoder MoDE-4-CP + Decoder Scale-MoE, No DFT-HT, 20 Rds)
# 3. Full Fed-FGS + Hierarchical Bi-Level MoE (Encoder MoDE-4-CP + Decoder Scale-MoE, With DFT-HT, 20 Rds)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs checkpoints/runs
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}"

# Configuration Files
CFG_SCALE_ALONE="federated/configs/sam_fedgs_scale_decoder_moe_r8_20rounds.yaml"
CFG_HIERARCHICAL_NO_DFT="federated/configs/hierarchical_bi_level_moe_no_dftht_r8_20rounds.yaml"

run_experiment() {
    local title="$1"
    local cfg="$2"
    local run_name
    run_name="$(basename "${cfg}" .yaml)"
    local log="logs/${run_name}.log"
    local ckpt_dir="checkpoints/runs/${run_name}"
    local resume_flag=""

    if [ -f "${ckpt_dir}/FedFGS_SAM_last.pth" ]; then
        echo "🔁 Existing checkpoint detected in ${ckpt_dir}! Resuming..."
        resume_flag="--resume true"
    fi

    echo ""
    echo "==========================================================================="
    echo "🔬 STARTING: ${title}"
    echo "   Config:   ${cfg}"
    echo "   Live Log: ${log}"
    echo "==========================================================================="

    conda run --no-capture-output -n polypsamfl python -u federated/flower_sim.py \
        --config "${cfg}" ${resume_flag} 2>&1 | tee -a "${log}"

    echo ""
    echo "✅ Completed: ${title}"
    echo "==========================================================================="
}

# ------------------------------------------------------------------------------
# 1. Scale-Aware Dual-Decoder Alone (Completes remaining rounds 4-20)
# ------------------------------------------------------------------------------
run_experiment "1. Scale-Aware Dual-Decoder Alone (No DFT-HT, 20 Rounds)" "${CFG_SCALE_ALONE}"

# ------------------------------------------------------------------------------
# 2. Hierarchical Bi-Level MoE: MoDE-4-CP Encoder + Scale Dual-Decoder (No DFT-HT)
# ------------------------------------------------------------------------------
run_experiment "2. Hierarchical Bi-Level MoE (Domain MoDE-4-CP + Scale-MoE, No DFT-HT, 20 Rounds)" "${CFG_HIERARCHICAL_NO_DFT}"

echo ""
echo "==========================================================================="
echo "🎉 BOTH SCALE & HIERARCHICAL MoE EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "==========================================================================="
