#!/usr/bin/env bash
# ==============================================================================
# Sequential MoDE (Mixture of Domain Experts) Experiment Runner
#
# Experiments Executed Sequentially:
# 1. MoDE-2: SAM + MoDE in last 2 blocks (10, 11) + FedGS (Mode A: Token-Driven)
# 2. MoDE-4: SAM + MoDE in last 4 blocks (8, 9, 10, 11) + FedGS (Mode A: Token-Driven)
# 3. MoDE-2 Center Prior: SAM + MoDE in blocks 10, 11 + FedGS (Mode B: lambda=0.2)
#
# Optional:
#   --with-baseline   Prepend baseline SAM-LoRA + FedGS (default_fed_fgs.yaml)
#   --mode2-only      Run only MoDE-2
#   --mode4-only      Run only MoDE-4
#   --prior-only      Run only MoDE-2 Center Prior
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs checkpoints/runs
export PYTHONUNBUFFERED=1

CONFIG_BASELINE="federated/configs/default_fed_fgs.yaml"
CONFIG_MODE2="federated/configs/mode2_sam_fedgs.yaml"
CONFIG_MODE4="federated/configs/mode4_sam_fedgs.yaml"
CONFIG_MODE2_PRIOR="federated/configs/mode2_center_prior_sam_fedgs.yaml"

RUN_BASELINE=false
RUN_MODE2=true
RUN_MODE4=true
RUN_MODE2_PRIOR=true

# Parse command line flags
for arg in "$@"; do
    case "${arg}" in
        --with-baseline)
            RUN_BASELINE=true
            ;;
        --mode2-only)
            RUN_MODE2=true
            RUN_MODE4=false
            RUN_MODE2_PRIOR=false
            ;;
        --mode4-only)
            RUN_MODE2=false
            RUN_MODE4=true
            RUN_MODE2_PRIOR=false
            ;;
        --prior-only)
            RUN_MODE2=false
            RUN_MODE4=false
            RUN_MODE2_PRIOR=true
            ;;
        --help|-h)
            echo "Usage: $0 [--with-baseline] [--mode2-only] [--mode4-only] [--prior-only]"
            exit 0
            ;;
    esac
done

run_experiment() {
    local stage_title="$1"
    local config_file="$2"
    local log_file="logs/$(basename "${config_file}" .yaml).log"

    echo ""
    echo "==========================================================================="
    echo "🔬 ${stage_title}"
    echo "   Config:   ${config_file}"
    echo "   Live Log: ${log_file}"
    echo "==========================================================================="

    conda run --no-capture-output -n polypsamfl python -u flower_sim.py --config "${config_file}" 2>&1 | tee "${log_file}"
    echo "✅ Completed: ${stage_title}"
}

TOTAL_EXPERIMENTS=0
[[ "${RUN_BASELINE}" == "true" ]] && ((TOTAL_EXPERIMENTS++)) || true
[[ "${RUN_MODE2}" == "true" ]] && ((TOTAL_EXPERIMENTS++)) || true
[[ "${RUN_MODE4}" == "true" ]] && ((TOTAL_EXPERIMENTS++)) || true
[[ "${RUN_MODE2_PRIOR}" == "true" ]] && ((TOTAL_EXPERIMENTS++)) || true

CURRENT_EXP=1

echo "==========================================================================="
echo "🚀 STARTING SEQUENTIAL MoDE EXPERIMENT PIPELINE"
echo "   Total experiments queued: ${TOTAL_EXPERIMENTS}"
echo "   Environment: polypsamfl"
echo "   Hardware: RTX 5090 (BF16 enabled)"
echo "==========================================================================="

if [[ "${RUN_BASELINE}" == "true" ]]; then
    run_experiment "[${CURRENT_EXP}/${TOTAL_EXPERIMENTS}] Baseline: SAM-LoRA + FedGS" "${CONFIG_BASELINE}"
    ((CURRENT_EXP++))
fi

if [[ "${RUN_MODE2}" == "true" ]]; then
    run_experiment "[${CURRENT_EXP}/${TOTAL_EXPERIMENTS}] MoDE-2: Last 2 Blocks (10, 11) + FedGS (Mode A)" "${CONFIG_MODE2}"
    ((CURRENT_EXP++))
fi

if [[ "${RUN_MODE4}" == "true" ]]; then
    run_experiment "[${CURRENT_EXP}/${TOTAL_EXPERIMENTS}] MoDE-4: Last 4 Blocks (8, 9, 10, 11) + FedGS (Mode A)" "${CONFIG_MODE4}"
    ((CURRENT_EXP++))
fi

if [[ "${RUN_MODE2_PRIOR}" == "true" ]]; then
    run_experiment "[${CURRENT_EXP}/${TOTAL_EXPERIMENTS}] MoDE-2 with Center Prior (lambda=0.2, Mode B)" "${CONFIG_MODE2_PRIOR}"
    ((CURRENT_EXP++))
fi

echo ""
echo "==========================================================================="
echo "🎉 ALL QUEUED EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "==========================================================================="

# Print Summary Table from Run JSONs
conda run -n polypsamfl python -c "
import os, json, glob

runs = [
    ('Baseline SAM + FedGS', 'checkpoints/runs/fed_fgs_vit_b_r4_default/run_summary.json' if os.path.exists('checkpoints/runs/fed_fgs_vit_b_r4_default/run_summary.json') else 'checkpoints/runs/fed_fgs_vit_b_r8_30rounds/run_summary.json'),
    ('MoDE-2 (Blocks 10, 11, r=8)', 'checkpoints/runs/mode2_sam_fedgs_r8/run_summary.json'),
    ('MoDE-4 (Blocks 8-11, r=8)', 'checkpoints/runs/mode4_sam_fedgs_r8/run_summary.json'),
    ('MoDE-2 Center Prior (r=8, lam=0.2)', 'checkpoints/runs/mode2_center_prior_sam_fedgs_r8/run_summary.json'),
]

print(f'\n{chr(61)*112}')
print(f'📊 MoDE EXPERIMENT COMPARISON SUMMARY (Center 6 Unseen Domain Benchmark)')
print(f'{chr(61)*112}')
print(f'{\"Experiment\":<34} | {\"DSC\":<8} | {\"DSC_S\":<8} | {\"IoU\":<8} | {\"Recall\":<8} | {\"HD95 (px)\":<10} | {\"Best Round\":<10}')
print(f'{chr(45)*34}-|-{chr(45)*8}-|-{chr(45)*8}-|-{chr(45)*8}-|-{chr(45)*8}-|-{chr(45)*10}-|-{chr(45)*10}')

for name, path in runs:
    if os.path.exists(path):
        try:
            with open(path) as f:
                d = json.load(f)
            best_info = d.get('best_overall_unseen_c6', {})
            dsc = f\"{best_info.get('dice', d.get('best_c6_dsc', 0.0)):.4f}\"
            small = f\"{best_info.get('dice_small', d.get('best_c6_small_dsc', 0.0)):.4f}\"
            iou = f\"{best_info.get('iou', 0.0):.4f}\" if 'iou' in best_info else \"-\"
            rec = f\"{best_info.get('recall', 0.0):.4f}\" if 'recall' in best_info else \"-\"
            hd = f\"{best_info.get('hd95', 0.0):.2f}\" if 'hd95' in best_info else \"-\"
            rd = f\"Round {best_info.get('round', d.get('best_c6_dsc_round', '-'))}\"
            print(f'{name:<34} | {dsc:<8} | {small:<8} | {iou:<8} | {rec:<8} | {hd:<10} | {rd:<10}')
        except Exception as err:
            print(f'{name:<34} | {\"Error\":<8} | {\"-\":<8} | {\"-\":<8} | {\"-\":<8} | {\"-\":<10} | {\"-\":<10}')
    else:
        print(f'{name:<34} | {\"Pending\":<8} | {\"-\":<8} | {\"-\":<8} | {\"-\":<8} | {\"-\":<10} | {\"-\":<10}')

print(f'{chr(61)*112}\n')
" || true
