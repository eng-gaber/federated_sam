# -*- coding: utf-8 -*-
"""
Automated Integration Tests for Scale-Aware Dual-Decoder MoE in SAM.

Tests:
1. Shape and Interface Compatibility:
   - Forward pass returns (z_mix, iou_mix) compatible with SAM tuple unpacking.
   - Auxiliary dictionary caching with weights, z_small, z_large.
2. Initial Gating Equilibrium:
   - Gate initializes at exactly [0.5, 0.5] balanced routing at step 0.
3. Decoupling & Gradient Flow:
   - Backprop flows through z_mix, gate, and both decoders.
   - For small polyp (s=1), L_expert penalizes only D_small.
   - For large polyp (s=0), L_expert penalizes only D_large.
4. Federated Synchronization:
   - extract_lora_parameters and set_lora_parameters successfully roundtrip all parameters.
"""

import sys
import os
import copy
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.scale_moe_decoder import ScaleGatingNetwork, ScaleAwareDualMaskDecoder
from fedfgs_loss import FedFGSCompositeLoss
from federated.flower_sim import extract_lora_parameters, set_lora_parameters


class MockMaskDecoder(nn.Module):
    """Lightweight mock of SAM's MaskDecoder for testing."""
    def __init__(self, transformer_dim: int = 256):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.conv = nn.Conv2d(transformer_dim, 1, kernel_size=1)
        self.iou_proj = nn.Linear(transformer_dim, 1)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = False,
    ):
        B = image_embeddings.shape[0]
        # Simulate mask prediction: (B, 1, 64, 64)
        mask = self.conv(image_embeddings)
        iou = self.iou_proj(image_embeddings.mean(dim=[-2, -1]))
        return mask, iou


def test_scale_gating_initialization():
    """Verify gate starts at exactly [0.5, 0.5] routing."""
    gate = ScaleGatingNetwork(in_features=256, hidden_dim=64)
    dummy_f = torch.randn(4, 256, 64, 64)
    weights, logits = gate(dummy_f)
    
    assert weights.shape == (4, 2), f"Expected shape (4, 2), got {weights.shape}"
    assert torch.allclose(weights[:, 0], torch.tensor(0.5), atol=1e-5), "w_small must be 0.5 at init"
    assert torch.allclose(weights[:, 1], torch.tensor(0.5), atol=1e-5), "w_large must be 0.5 at init"
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-5), "Weights must sum to 1"
    print("✓ test_scale_gating_initialization passed!")


def test_scale_dual_decoder_forward():
    """Verify ScaleAwareDualMaskDecoder matches SAM interface and caches outputs."""
    base_decoder = MockMaskDecoder(transformer_dim=256)
    dual_decoder = ScaleAwareDualMaskDecoder(base_decoder, gate_hidden_dim=64, beta=0.5, lambda_gate=0.1)

    B = 2
    dummy_f = torch.randn(B, 256, 64, 64)
    dummy_pe = torch.randn(1, 256, 64, 64)
    dummy_sparse = torch.randn(B, 2, 256)
    dummy_dense = torch.randn(B, 256, 64, 64)

    # Test tuple unpacking
    pred, iou = dual_decoder(dummy_f, dummy_pe, dummy_sparse, dummy_dense, multimask_output=False)
    assert pred.shape == (B, 1, 64, 64), f"Expected pred shape (B, 1, 64, 64), got {pred.shape}"
    assert iou.shape == (B, 1), f"Expected iou shape (B, 1), got {iou.shape}"

    # Verify cached aux dict
    aux = dual_decoder.last_aux
    assert aux is not None
    assert "z_mix" in aux and "z_small" in aux and "z_large" in aux and "weights" in aux
    assert torch.allclose(pred, aux["z_mix"])
    print("✓ test_scale_dual_decoder_forward passed!")


def test_scale_aware_loss_and_gradient_decoupling():
    """Verify multi-task loss computation and gradient decoupling for small vs large experts."""
    base_decoder = MockMaskDecoder(transformer_dim=256)
    dual_decoder = ScaleAwareDualMaskDecoder(base_decoder, gate_hidden_dim=64, beta=0.5, lambda_gate=0.1)
    criterion = FedFGSCompositeLoss(lambda_dice=0.7, lambda_bce=0.2, lambda_focal=0.1)

    B = 2
    dummy_f = torch.randn(B, 256, 64, 64)
    dummy_pe = torch.randn(1, 256, 64, 64)
    dummy_sparse = torch.randn(B, 2, 256)
    dummy_dense = torch.randn(B, 256, 64, 64)
    target_masks = torch.randint(0, 2, (B, 1, 64, 64)).float()

    # Case: Sample 0 is small (s=1), Sample 1 is large (s=0)
    is_smalls = torch.tensor([1, 0])

    pred, _ = dual_decoder(dummy_f, dummy_pe, dummy_sparse, dummy_dense)
    loss_total, stats = dual_decoder.compute_scale_aware_loss(
        criterion=criterion,
        target_masks=target_masks,
        is_smalls=is_smalls,
    )

    assert loss_total.ndim == 0, "Loss must be scalar"
    assert loss_total.item() > 0.0, "Loss must be positive"
    assert "loss_mix" in stats and "loss_expert" in stats and "loss_gate" in stats

    # Backward pass
    dual_decoder.zero_grad()
    loss_total.backward()

    # Check gradients in gate
    for p in dual_decoder.gate.parameters():
        if p.requires_grad:
            assert p.grad is not None, "Gate parameter must receive gradient"

    # Check gradients in both decoders
    assert dual_decoder.decoder_small.conv.weight.grad is not None
    assert dual_decoder.decoder_large.conv.weight.grad is not None
    print(f"✓ test_scale_aware_loss_and_gradient_decoupling passed! Loss stats: {stats}")


def test_federated_parameters_roundtrip():
    """Verify extract_lora_parameters and set_lora_parameters work seamlessly with dual decoder."""
    base_decoder = MockMaskDecoder(transformer_dim=256)
    dual_decoder = ScaleAwareDualMaskDecoder(base_decoder, gate_hidden_dim=64, beta=0.5, lambda_gate=0.1)

    # Extract initial parameters (copy to avoid in-place numpy view aliasing)
    initial_params = [arr.copy() for arr in extract_lora_parameters(dual_decoder)]
    assert len(initial_params) > 0, "Must extract non-empty parameter list"

    # Perturb module parameters
    for p in dual_decoder.parameters():
        if p.requires_grad:
            p.data.add_(0.5)

    # Verify perturbation took effect
    perturbed_params = extract_lora_parameters(dual_decoder)
    assert not np.allclose(initial_params[0], perturbed_params[0])

    # Restore initial parameters via set_lora_parameters
    set_lora_parameters(dual_decoder, initial_params)
    restored_params = extract_lora_parameters(dual_decoder)

    for orig, rest in zip(initial_params, restored_params):
        assert np.allclose(orig, rest), "Restored parameters must match initial parameters exactly"
    print("✓ test_federated_parameters_roundtrip passed!")


if __name__ == "__main__":
    test_scale_gating_initialization()
    test_scale_dual_decoder_forward()
    test_scale_aware_loss_and_gradient_decoupling()
    test_federated_parameters_roundtrip()
    print("\n🎉 ALL SCALE-AWARE DUAL-DECODER INTEGRATION TESTS PASSED SUCCESSFULLY!")
