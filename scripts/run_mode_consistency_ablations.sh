#!/usr/bin/env bash
# ==============================================================================
# Master Table Consistency Ablation Runner for MoDE:
#
# Experiments in this pipeline:
# 1. SAM-FedGS + Composite Loss + MoDE-2 (No DFT-HT), 20 rounds
# 2. SAM-FedGS + Composite Loss + MoDE-4 (No DFT-HT), 20 rounds
#
# Complemented with already completed:
# - Full Fed-FGS SAM + MoDE-2 (with DFT-HT, 20 rounds) -> checkpoints/runs/mode2_sam_fedgs_r8/
# - Full Fed-FGS SAM + MoDE-4 (with DFT-HT, 20 rounds) -> checkpoints/runs/mode4_sam_fedgs_r8/
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs checkpoints/runs
export PYTHONUNBUFFERED=1

CONFIG_NO_DFTHT_MODE2="federated/configs/sam_fedgs_composite_loss_mode2_r8_20rounds.yaml"
CONFIG_NO_DFTHT_MODE4="federated/configs/sam_fedgs_composite_loss_mode4_r8_20rounds.yaml"
CONFIG_FULL_MODE2="federated/configs/mode2_sam_fedgs.yaml"
CONFIG_FULL_MODE4="federated/configs/mode4_sam_fedgs.yaml"

RUN_MODE2_NO_DFTHT=true
RUN_MODE4_NO_DFTHT=true
RERUN_FULL_MODE=false

for arg in "$@"; do
    case "${arg}" in
        --mode2-only)
            RUN_MODE2_NO_DFTHT=true
            RUN_MODE4_NO_DFTHT=false
            ;;
        --mode4-only)
            RUN_MODE2_NO_DFTHT=false
            RUN_MODE4_NO_DFTHT=true
            ;;
        --rerun-full)
            RERUN_FULL_MODE=true
            ;;
        --table-only)
            RUN_MODE2_NO_DFTHT=false
            RUN_MODE4_NO_DFTHT=false
            ;;
    esac
done

run_experiment() {
    local title="$1"
    local cfg="$2"
    local log="logs/$(basename "${cfg}" .yaml).log"

    echo ""
    echo "==========================================================================="
    echo "🔬 Starting: ${title}"
    echo "   Config:   ${cfg}"
    echo "   Live Log: ${log}"
    echo "==========================================================================="

    conda run --no-capture-output -n polypsamfl python -u flower_sim.py --config "${cfg}" 2>&1 | tee "${log}"
    echo "✅ Completed: ${title}"
}

if [[ "${RUN_MODE2_NO_DFTHT}" == "true" ]]; then
    run_experiment "SAM-FedGS + Composite Loss + MoDE-2 (No DFT-HT, 20 Rounds)" "${CONFIG_NO_DFTHT_MODE2}"
fi

if [[ "${RUN_MODE4_NO_DFTHT}" == "true" ]]; then
    run_experiment "SAM-FedGS + Composite Loss + MoDE-4 (No DFT-HT, 20 Rounds)" "${CONFIG_NO_DFTHT_MODE4}"
fi

if [[ "${RERUN_FULL_MODE}" == "true" ]]; then
    run_experiment "Full Fed-FGS SAM + MoDE-2 (FedGS + Composite Loss + DFT-HT, 20 Rounds)" "${CONFIG_FULL_MODE2}"
    run_experiment "Full Fed-FGS SAM + MoDE-4 (FedGS + Composite Loss + DFT-HT, 20 Rounds)" "${CONFIG_FULL_MODE4}"
fi

echo ""
echo "==========================================================================="
echo "🎉 ALL TARGET EXPERIMENTS COMPLETE! GENERATING MASTER ABLATION TABLE..."
echo "==========================================================================="

conda run -n polypsamfl python -c "
import os, json

table_rows = [
    # Baseline non-MoDE rows
    ('SAM-FedGS + Composite Loss (No DFT-HT)', 'checkpoints/runs/ablation_sam_fedgs_composite_loss_r8_20rounds/run_summary.json', 'No', 'No'),
    # MoDE without DFT-HT
    ('SAM-FedGS + Comp Loss + MoDE-2 (No DFT-HT)', 'checkpoints/runs/sam_fedgs_composite_loss_mode2_r8_20rounds/run_summary.json', 'No', 'MoDE-2'),
    ('SAM-FedGS + Comp Loss + MoDE-4 (No DFT-HT)', 'checkpoints/runs/sam_fedgs_composite_loss_mode4_r8_20rounds/run_summary.json', 'No', 'MoDE-4'),
    # Baseline with DFT-HT
    ('Full Fed-FGS SAM (Standard Baseline)', 'checkpoints/runs/ablation_sam_fedgs_r8_20rounds/run_summary.json', 'Yes', 'No'),
    # MoDE with DFT-HT
    ('Full Fed-FGS SAM + MoDE-2', 'checkpoints/runs/mode2_sam_fedgs_r8/run_summary.json', 'Yes', 'MoDE-2'),
    ('Full Fed-FGS SAM + MoDE-4', 'checkpoints/runs/mode4_sam_fedgs_r8/run_summary.json', 'Yes', 'MoDE-4'),
    ('Full Fed-FGS SAM + MoDE-2 (Center Prior)', 'checkpoints/runs/mode2_center_prior_sam_fedgs_r8/run_summary.json', 'Yes', 'MoDE-2 (Prior)'),
]

print(f'\n{chr(61)*125}')
print('📊 MASTER TABLE CONSISTENCY ABLATION (Center 6 Out-of-Distribution Generalization, 20 Rounds, r=8)')
print(f'{chr(61)*125}')
print(f'{\"Method / Configuration\":<42} | {\"DFT-HT\":<6} | {\"MoDE\":<14} | {\"DSC\":<7} | {\"DSC_S\":<7} | {\"IoU\":<7} | {\"Recall\":<7} | {\"HD95 (px)\":<9} | {\"Peak Rd\":<7}')
print(f'{chr(45)*42}-|-{chr(45)*6}-|-{chr(45)*14}-|-{chr(45)*7}-|-{chr(45)*7}-|-{chr(45)*7}-|-{chr(45)*7}-|-{chr(45)*9}-|-{chr(45)*7}')

for name, path, dftht, mode in table_rows:
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
            rd = f\"Rd {best_info.get('round', d.get('best_c6_dsc_round', '-'))}\"
            print(f'{name:<42} | {dftht:<6} | {mode:<14} | {dsc:<7} | {small:<7} | {iou:<7} | {rec:<7} | {hd:<9} | {rd:<7}')
        except Exception:
            print(f'{name:<42} | {dftht:<6} | {mode:<14} | {\"Error\":<7} | {\"—\":<7} | {\"—\":<7} | {\"—\":<7} | {\"—\":<9} | {\"—\":<7}')
    else:
        print(f'{name:<42} | {dftht:<6} | {mode:<14} | {\"Pending\":<7} | {\"—\":<7} | {\"—\":<7} | {\"—\":<7} | {\"—\":<9} | {\"—\":<7}')

print(f'{chr(61)*125}\n')
" || true
