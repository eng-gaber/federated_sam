#!/usr/bin/env bash
# ==============================================================================
# MedSAM Experiments Runner — Full Ablation Suite
# Runs 4 experiments sequentially to measure each component's contribution:
#
#   1. MedSAM + FedAvg (Dice+BCE)            → pure backbone baseline
#   2. MedSAM + FedGS + Composite Loss       → +FedGS, +Focal loss
#   3. MedSAM + FedGS + Composite + MoDE4-CP → +encoder domain experts
#   4. MedSAM + FedGS + Scale-Decoder MoE    → +dual-decoder architecture
#
# Compare against SAM counterparts:
#   SAM-FedAvg:              0.7918 DSC
#   SAM-FedGS-Composite:     0.8006 DSC
#   SAM-MoDE4-CP:            0.7973 DSC
#   SAM-Scale-MoE:           0.8064 DSC
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

echo "============================================="
echo "  MedSAM Full Ablation Suite (4 Experiments, 30 Rounds Each)"
echo "  Project: $PROJECT_ROOT"
echo "============================================="

# ── Experiment 1: MedSAM + FedAvg (Pure Baseline, Resuming from Rd 14) ──
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [1/4] MedSAM-FedAvg (Pure Backbone Baseline, 30 Rounds)"
echo "  Config: baseline_medsam_fedavg_r8_20rounds.yaml (Resuming from Rd 14)"
echo "  Compare: SAM-FedAvg (0.7918 DSC)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python "$PROJECT_ROOT/flower_sim.py" \
    --config "$PROJECT_ROOT/federated/configs/baseline_medsam_fedavg_r8_20rounds.yaml" \
    --resume true

echo ""
echo "✅ [1/4] MedSAM-FedAvg DONE"
echo ""

# ── Experiment 2: MedSAM + FedGS + Composite Loss (30 Rounds) ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/4] MedSAM-FedGS-Composite (+FedGS +Focal, 30 Rounds)"
echo "  Config: medsam_fedgs_composite_loss_r8_20rounds.yaml"
echo "  Compare: SAM-FedGS-Composite (0.8006 DSC)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python "$PROJECT_ROOT/flower_sim.py" \
    --config "$PROJECT_ROOT/federated/configs/medsam_fedgs_composite_loss_r8_20rounds.yaml" \
    --resume true

echo ""
echo "✅ [2/4] MedSAM-FedGS-Composite DONE"
echo ""

# ── Experiment 3: MedSAM + FedGS + Composite + MoDE-4-CP (30 Rounds) ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [3/4] MedSAM-FedGS-Composite-MoDE4-CP (+Encoder Experts, 30 Rounds)"
echo "  Config: medsam_fedgs_composite_mode4_cp_no_dftht_20rounds.yaml"
echo "  Compare: SAM-MoDE4-CP (0.7973 DSC)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python "$PROJECT_ROOT/flower_sim.py" \
    --config "$PROJECT_ROOT/federated/configs/medsam_fedgs_composite_mode4_cp_no_dftht_20rounds.yaml" \
    --resume true

echo ""
echo "✅ [3/4] MedSAM-FedGS-Composite-MoDE4-CP DONE"
echo ""

# ── Experiment 4: MedSAM + FedGS + Scale-Aware Dual-Decoder MoE (30 Rounds) ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [4/4] MedSAM-FedGS-Scale-Decoder-MoE (+Dual Decoder, 30 Rounds)"
echo "  Config: medsam_fedgs_scale_decoder_moe_r8_20rounds.yaml"
echo "  Compare: SAM-Scale-MoE (0.8064 DSC)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python "$PROJECT_ROOT/flower_sim.py" \
    --config "$PROJECT_ROOT/federated/configs/medsam_fedgs_scale_decoder_moe_r8_20rounds.yaml" \
    --resume true

echo ""
echo "✅ [4/4] MedSAM-FedGS-Scale-Decoder-MoE DONE"
echo ""

echo "============================================="
echo "  All 4 MedSAM experiments completed (30 Rounds)!"
echo "  Check checkpoints/runs/ for results."
echo "============================================="
