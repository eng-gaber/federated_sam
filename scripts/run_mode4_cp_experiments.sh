#!/usr/bin/env bash
# ==============================================================================
# MoDE-4-CP Priority Runner:
#
# Experiments Executed Sequentially:
# 1. SAM-FedGS + Composite + MoDE-4-CP (No DFT-HT), 20 rounds
#    Compare to: SAM-FedGS + Composite (0.8006 DSC)
# 2. SAM-FedGS + Composite + MoDE-4-CP (No DFT-HT), 30 rounds
#    Compare to: SAM-FedGS + Composite
# 3. Full Fed-FGS SAM + MoDE-4-CP (With DFT-HT), 30 rounds
#    Compare to: Full Fed-FGS SAM (0.7958 DSC, HD95 101.26 px)
#
# Optional Flags:
#   --no-dftht-20-only   Run only Experiment 1 (20 rounds, no DFT-HT)
#   --no-dftht-30-only   Run only Experiment 2 (30 rounds, no DFT-HT)
#   --full-30-only       Run only Experiment 3 (30 rounds, with DFT-HT)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs checkpoints/runs
export PYTHONUNBUFFERED=1

CONFIG_1="federated/configs/sam_fedgs_composite_mode4_cp_r8_20rounds.yaml"
CONFIG_2="federated/configs/sam_fedgs_composite_mode4_cp_r8_30rounds.yaml"
CONFIG_3="federated/configs/full_fed_fgs_sam_mode4_cp_r8_30rounds.yaml"

RUN_1=true
RUN_2=true
RUN_3=true

for arg in "$@"; do
    case "${arg}" in
        --no-dftht-20-only)
            RUN_1=true
            RUN_2=false
            RUN_3=false
            ;;
        --no-dftht-30-only)
            RUN_1=false
            RUN_2=true
            RUN_3=false
            ;;
        --full-30-only)
            RUN_1=false
            RUN_2=false
            RUN_3=true
            ;;
        --help|-h)
            echo "Usage: $0 [--no-dftht-20-only] [--no-dftht-30-only] [--full-30-only]"
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
[[ "${RUN_1}" == "true" ]] && ((TOTAL_EXPERIMENTS++)) || true
[[ "${RUN_2}" == "true" ]] && ((TOTAL_EXPERIMENTS++)) || true
[[ "${RUN_3}" == "true" ]] && ((TOTAL_EXPERIMENTS++)) || true

CURRENT_EXP=1

echo "==========================================================================="
echo "🚀 STARTING MoDE-4-CP PRIORITY EXPERIMENTS PIPELINE"
echo "   Total experiments queued: ${TOTAL_EXPERIMENTS}"
echo "   Environment: polypsamfl"
echo "   Hardware: RTX 5090 (BF16 enabled)"
echo "==========================================================================="

if [[ "${RUN_1}" == "true" ]]; then
    run_experiment "[${CURRENT_EXP}/${TOTAL_EXPERIMENTS}] SAM-FedGS + Comp + MoDE-4-CP (No DFT-HT, 20 Rounds)" "${CONFIG_1}"
    ((CURRENT_EXP++))
fi

if [[ "${RUN_2}" == "true" ]]; then
    run_experiment "[${CURRENT_EXP}/${TOTAL_EXPERIMENTS}] SAM-FedGS + Comp + MoDE-4-CP (No DFT-HT, 30 Rounds)" "${CONFIG_2}"
    ((CURRENT_EXP++))
fi

if [[ "${RUN_3}" == "true" ]]; then
    run_experiment "[${CURRENT_EXP}/${TOTAL_EXPERIMENTS}] Full Fed-FGS SAM + MoDE-4-CP (With DFT-HT, 30 Rounds)" "${CONFIG_3}"
    ((CURRENT_EXP++))
fi

echo ""
echo "==========================================================================="
echo "🎉 ALL QUEUED MoDE-4-CP EXPERIMENTS COMPLETED!"
echo "==========================================================================="

conda run -n polypsamfl python -c "
import os, json

comparisons = [
    ('SAM-FedGS + Composite (Baseline)', 'checkpoints/runs/ablation_sam_fedgs_composite_loss_r8_20rounds/run_summary.json', 'No', '20', '0.8006 (Table 2)'),
    ('SAM-FedGS + Comp + MoDE-4-CP (20r)', 'checkpoints/runs/sam_fedgs_composite_mode4_cp_r8_20rounds/run_summary.json', 'No', '20', 'Direct Match'),
    ('SAM-FedGS + Comp + MoDE-4-CP (30r)', 'checkpoints/runs/sam_fedgs_composite_mode4_cp_r8_30rounds/run_summary.json', 'No', '30', 'Extended'),
    ('Full Fed-FGS SAM (Baseline)', 'checkpoints/runs/fed_fgs_vit_b_r8_30rounds/run_summary.json', 'Yes', '30', '0.7958 (HD95 101.26)'),
    ('Full Fed-FGS SAM + MoDE-4-CP (30r)', 'checkpoints/runs/full_fed_fgs_sam_mode4_cp_r8_30rounds/run_summary.json', 'Yes', '30', 'Direct Match'),
]

print(f'\n{chr(61)*130}')
print('📊 MoDE-4-CP PRIORITY BENCHMARK COMPARISON (Center 6 Unseen Domain)')
print(f'{chr(61)*130}')
print(f'{\"Experiment\":<38} | {\"DFT-HT\":<6} | {\"Rounds\":<6} | {\"DSC\":<8} | {\"DSC_S\":<8} | {\"IoU\":<8} | {\"Recall\":<8} | {\"HD95 (px)\":<10} | {\"Best Round\":<10}')
print(f'{chr(45)*38}-|-{chr(45)*6}-|-{chr(45)*6}-|-{chr(45)*8}-|-{chr(45)*8}-|-{chr(45)*8}-|-{chr(45)*8}-|-{chr(45)*10}-|-{chr(45)*10}')

for name, path, dftht, rds, note in comparisons:
    if os.path.exists(path):
        try:
            with open(path) as f:
                d = json.load(f)
            best_info = d.get('best_overall_unseen_c6', {})
            dsc = f\"{best_info.get('dice', d.get('best_c6_dsc', 0.0)):.4f}\"
            small = f\"{best_info.get('dice_small', d.get('best_c6_small_dsc', 0.0)):.4f}\"
            iou = f\"{best_info.get('iou', 0.0):.4f}\" if 'iou' in best_info else '—'
            rec = f\"{best_info.get('recall', 0.0):.4f}\" if 'recall' in best_info else '—'
            hd = f\"{best_info.get('hd95', 0.0):.2f}\" if 'hd95' in best_info else '—'
            pk = f\"Rd {best_info.get('round', d.get('best_c6_dsc_round', '-'))}\"
            print(f'{name:<38} | {dftht:<6} | {rds:<6} | {dsc:<8} | {small:<8} | {iou:<8} | {rec:<8} | {hd:<10} | {pk:<10}')
        except Exception:
            print(f'{name:<38} | {dftht:<6} | {rds:<6} | {\"Error\":<8} | {\"—\":<8} | {\"—\":<8} | {\"—\":<8} | {\"—\":<10} | {\"—\":<10}')
    else:
        print(f'{name:<38} | {dftht:<6} | {rds:<6} | {\"Pending\":<8} | {\"—\":<8} | {\"—\":<8} | {\"—\":<8} | {\"—\":<10} | {\"—\":<10}')

print(f'{chr(61)*130}\n')
" || true
