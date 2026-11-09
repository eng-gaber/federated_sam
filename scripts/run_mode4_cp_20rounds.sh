#!/usr/bin/env bash
# ==============================================================================
# Priority MoDE-4-CP (20 Rounds) Runner:
#
# Runs ONLY the 2 requested experiments sequentially:
# 1. SAM-FedGS + Composite + MoDE-4-CP (No DFT-HT), 20 rounds
#    Direct comparison to: SAM-FedGS + Composite (0.8006 DSC)
# 2. Full Fed-FGS SAM + MoDE-4-CP (With DFT-HT), 20 rounds
#    Direct comparison to: Full Fed-FGS SAM (0.7958 DSC, HD95 101.26 px)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs checkpoints/runs
export PYTHONUNBUFFERED=1

CONFIG_NO_DFTHT="federated/configs/sam_fedgs_composite_mode4_cp_no_dftht_20rounds.yaml"
CONFIG_FULL="federated/configs/full_fed_fgs_sam_mode4_cp_20rounds.yaml"

RUN_1=true
RUN_2=true

for arg in "$@"; do
    case "${arg}" in
        --no-dftht-only)
            RUN_1=true
            RUN_2=false
            ;;
        --full-only)
            RUN_1=false
            RUN_2=true
            ;;
        --table-only)
            RUN_1=false
            RUN_2=false
            ;;
        --help|-h)
            echo "Usage: $0 [--no-dftht-only] [--full-only] [--table-only]"
            exit 0
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

TOTAL=0
[[ "${RUN_1}" == "true" ]] && ((TOTAL++)) || true
[[ "${RUN_2}" == "true" ]] && ((TOTAL++)) || true
CURRENT=1

echo "==========================================================================="
echo "🚀 STARTING PRIORITY MoDE-4-CP PIPELINE (20 ROUNDS EACH)"
echo "   Queued experiments: ${TOTAL}"
echo "   Hardware: RTX 5090 (BF16 enabled)"
echo "==========================================================================="

if [[ "${RUN_1}" == "true" ]]; then
    run_experiment "[${CURRENT}/${TOTAL}] SAM-FedGS + Composite + MoDE-4-CP (No DFT-HT, 20 Rounds)" "${CONFIG_NO_DFTHT}"
    ((CURRENT++))
fi

if [[ "${RUN_2}" == "true" ]]; then
    run_experiment "[${CURRENT}/${TOTAL}] Full Fed-FGS SAM + MoDE-4-CP (With DFT-HT, 20 Rounds)" "${CONFIG_FULL}"
    ((CURRENT++))
fi

echo ""
echo "==========================================================================="
echo "🎉 BOTH MoDE-4-CP EXPERIMENTS COMPLETED! GENERATING COMPARISON TABLE..."
echo "==========================================================================="

conda run -n polypsamfl python -c "
import os, json

rows = [
    ('SAM-FedGS + Composite (Baseline)', 'checkpoints/runs/ablation_sam_fedgs_composite_loss_r8_20rounds/run_summary.json', 'No', '0.8006', '0.5100', '0.7343', '0.8240', '114.96'),
    ('SAM-FedGS + Comp + MoDE-4-CP', 'checkpoints/runs/sam_fedgs_composite_mode4_cp_no_dftht_20rounds/run_summary.json', 'No', None, None, None, None, None),
    ('Full Fed-FGS SAM (Baseline)', 'checkpoints/runs/fed_fgs_vit_b_r8_30rounds/run_summary.json', 'Yes', '0.7958', '0.4955', '0.7359', '0.8277', '101.26'),
    ('Full Fed-FGS SAM + MoDE-4-CP', 'checkpoints/runs/full_fed_fgs_sam_mode4_cp_20rounds/run_summary.json', 'Yes', None, None, None, None, None),
]

print(f'\n{chr(61)*126}')
print('📊 MoDE-4-CP DIRECT COMPARISON (Center 6 Unseen Domain Benchmark, 20 Rounds, r=8)')
print(f'{chr(61)*126}')
print(f'{\"Experiment\":<36} | {\"DFT-HT\":<6} | {\"DSC\":<8} | {\"DSC_S\":<8} | {\"IoU\":<8} | {\"Recall\":<8} | {\"HD95 (px)\":<10} | {\"Best Round\":<10}')
print(f'{chr(45)*36}-|-{chr(45)*6}-|-{chr(45)*8}-|-{chr(45)*8}-|-{chr(45)*8}-|-{chr(45)*8}-|-{chr(45)*10}-|-{chr(45)*10}')

for name, path, dftht, fixed_dsc, fixed_small, fixed_iou, fixed_rec, fixed_hd in rows:
    if fixed_dsc is not None and not os.path.exists(path):
        # Printed cached baseline from Table 2
        print(f'{name:<36} | {dftht:<6} | {fixed_dsc:<8} | {fixed_small:<8} | {fixed_iou:<8} | {fixed_rec:<8} | {fixed_hd:<10} | {\"Table 2\":<10}')
    elif os.path.exists(path):
        try:
            with open(path) as f:
                d = json.load(f)
            best_info = d.get('best_overall_unseen_c6', {})
            dsc = f\"{best_info.get('dice', d.get('best_c6_dsc', 0.0)):.4f}\"
            small = f\"{best_info.get('dice_small', d.get('best_c6_small_dsc', 0.0)):.4f}\"
            iou = f\"{best_info.get('iou', fixed_iou if fixed_iou else 0.0):.4f}\" if 'iou' in best_info or fixed_iou else '—'
            rec = f\"{best_info.get('recall', fixed_rec if fixed_rec else 0.0):.4f}\" if 'recall' in best_info or fixed_rec else '—'
            hd = f\"{best_info.get('hd95', fixed_hd if fixed_hd else 0.0):.2f}\" if 'hd95' in best_info or fixed_hd else '—'
            pk = f\"Rd {best_info.get('round', d.get('best_c6_dsc_round', '-'))}\"
            print(f'{name:<36} | {dftht:<6} | {dsc:<8} | {small:<8} | {iou:<8} | {rec:<8} | {hd:<10} | {pk:<10}')
        except Exception:
            print(f'{name:<36} | {dftht:<6} | {\"Error\":<8} | {\"—\":<8} | {\"—\":<8} | {\"—\":<8} | {\"—\":<10} | {\"—\":<10}')
    else:
        print(f'{name:<36} | {dftht:<6} | {\"Pending\":<8} | {\"—\":<8} | {\"—\":<8} | {\"—\":<8} | {\"—\":<10} | {\"—\":<10}')

print(f'{chr(61)*126}\n')
" || true
