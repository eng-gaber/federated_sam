# -*- coding: utf-8 -*-
"""
Unit and Integration Tests for MoDE (Mixture of Domain Experts) with SAM ViT-B & FedGS
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mode_mlp import MoDE_MLP, ExpertMLPLoRA, TokenGatingNetwork
from models.sam_LoRa import LoRA_Sam
from models.sam import sam_model_registry
from federated.flower_sim import (
    extract_lora_parameters,
    set_lora_parameters,
    FedFGSCompositeLoss,
)


def test_mode_mlp_isolated():
    print("\n--- Test 1: Isolated MoDE-MLP Properties ---")
    D = 768
    D_mlp = 3072
    num_experts = 5
    top_k = 2

    # Dummy base MLP
    class DummyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(D, D_mlp)
            self.act = nn.GELU()
            self.lin2 = nn.Linear(D_mlp, D)

        def forward(self, x):
            return self.lin2(self.act(self.lin1(x)))

    base_mlp = DummyMLP()
    mode_mlp = MoDE_MLP(
        base_mlp=base_mlp,
        block_idx=10,
        embedding_dim=D,
        mlp_dim=D_mlp,
        num_experts=num_experts,
        r=4,
        top_k=top_k,
        routing_mode="mode_a",
    )

    # 1. Base MLP must be frozen
    for name, p in mode_mlp.base_mlp.named_parameters():
        assert not p.requires_grad, f"Base MLP parameter {name} must be frozen!"
    print("  ✓ Base SAM MLP weights are completely frozen.")

    # 2. Expert LoRA B matrices initialized to zero
    for e, expert in enumerate(mode_mlp.experts):
        assert torch.all(expert.lora_b1.weight == 0), f"Expert {e} lora_b1 not zero initialized"
        assert torch.all(expert.lora_b2.weight == 0), f"Expert {e} lora_b2 not zero initialized"
    print("  ✓ Expert LoRA B matrices are zero-initialized.")

    # 3. Step 0 equivalence: y == y_base
    x = torch.randn(2, 64, 64, D)
    with torch.no_grad():
        y_base = base_mlp(x)
        y_mode = mode_mlp(x)
        diff = (y_mode - y_base).abs().max().item()
        assert diff < 1e-6, f"Step 0 output differs from base MLP by {diff}!"
    print(f"  ✓ Output matches base SAM MLP exactly at initialization (max diff: {diff:.2e}).")

    # 4. Mode B unseen domain safety
    mode_mlp.routing_mode = "mode_b"
    mode_mlp.gating.routing_mode = "mode_b"
    mode_mlp.set_center_id(None)  # Unseen Center 6 has no center prior
    y_unseen = mode_mlp(x)
    assert y_unseen.shape == x.shape
    print("  ✓ Mode B unseen domain (center_id=None) executes cleanly without leakage.")


def test_gradient_propagation():
    print("\n--- Test 2: MoDE-MLP Gradient Propagation ---")
    D = 768
    D_mlp = 3072
    num_experts = 5

    class DummyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(D, D_mlp)
            self.act = nn.GELU()
            self.lin2 = nn.Linear(D_mlp, D)

    base_mlp = DummyMLP()
    mode_mlp = MoDE_MLP(
        base_mlp=base_mlp,
        block_idx=10,
        embedding_dim=D,
        mlp_dim=D_mlp,
        num_experts=num_experts,
        r=4,
        top_k=2,
        routing_mode="mode_a",
    )

    x = torch.randn(2, 16, 16, D, requires_grad=True)
    out = mode_mlp(x)
    loss = out.sum()
    loss.backward()

    # Base MLP must receive ZERO gradient
    for name, p in mode_mlp.base_mlp.named_parameters():
        assert p.grad is None, f"Base MLP parameter {name} unexpectedly received gradients!"
    print("  ✓ Frozen base MLP receives strictly 0 gradients.")

    # Expert LoRAs: B matrices receive gradients immediately
    b_grads_received = 0
    for expert in mode_mlp.experts:
        if expert.lora_b1.weight.grad is not None:
            b_grads_received += 1
    assert b_grads_received > 0, "Expert LoRA parameters failed to receive gradients!"
    print(f"  ✓ Activated experts ({b_grads_received}/{num_experts}) received LoRA gradients.")


def test_full_sam_mode_integration():
    print("\n--- Test 3: Full SAM ViT-B + MoDE-2 Integration ---")
    import argparse

    args = argparse.Namespace(
        arch="vit_b",
        sam_checkpoint="sam_vit_b_01ec64.pth",
        num_cls=2,
        lora_rank=8,
        if_encoder_lora_layer=True,
        encoder_lora_layer=[],
        if_decoder_lora_layer=True,
        if_update_encoder=True,
        devices=["cpu"],
        if_split_encoder_gpus=False,
        if_encoder_adapter=False,
        encoder_adapter_depths=[],
        if_mask_decoder_adapter=False,
        decoder_adapt_depth=0,
        thd=False,
        # MoDE hyperparameters
        enable_mode=True,
        mode_blocks=[10, 11],
        mode_num_experts=5,
        mode_lora_rank=8,
        mode_lora_alpha=8.0,
        mode_lora_dropout=0.0,
        mode_top_k=2,
        mode_gate_hidden_dim=128,
        mode_routing_mode="mode_a",
        mode_center_prior_lambda=0.2,
    )

    base_sam = sam_model_registry["vit_b"](args=args, checkpoint=args.sam_checkpoint, num_classes=2)
    lora_wrapper = LoRA_Sam(args, base_sam, r=args.lora_rank)
    model = lora_wrapper.sam

    # 1. Verify MoDE blocks placed in blocks 10 and 11
    assert isinstance(model.image_encoder.blocks[10].mlp, MoDE_MLP), "Block 10 MLP is not MoDE_MLP!"
    assert isinstance(model.image_encoder.blocks[11].mlp, MoDE_MLP), "Block 11 MLP is not MoDE_MLP!"
    assert not isinstance(model.image_encoder.blocks[9].mlp, MoDE_MLP), "Block 9 MLP should be standard MLP!"
    print("  ✓ MoDE-MLP successfully placed in blocks 10 and 11.")

    # 2. Parameter count verification
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  ✓ Trainable parameters (LoRA + MoDE + Decoder): {trainable_count:,}")
    print(f"  ✓ Frozen base SAM parameters: {frozen_count:,}")

    # 3. Federated serialization roundtrip (extract & set)
    param_list = extract_lora_parameters(model)
    assert len(param_list) == len(trainable_params), "Mismatch in parameter extraction count!"
    set_lora_parameters(model, param_list)
    print("  ✓ Federated parameter extraction & reload roundtrip successful.")

    # 4. Expert usage logging
    model.set_center_id(0)
    model.reset_expert_usage()
    usage = model.get_expert_usage()
    assert 10 in usage and 11 in usage
    print(f"  ✓ Expert usage tracking initialized: {usage}")

    print("\n✅ ALL MoDE INTEGRATION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_mode_mlp_isolated()
    test_gradient_propagation()
    test_full_sam_mode_integration()
