"""
Quick functional verification script for Fed-FGS modules:
1. DFT-HT augmentation
2. FedFGSCompositeLoss
3. Metadata loading
4. Dataset sample loading with DFT-HT
"""
import torch
import numpy as np
from feddg_dftht import extract_amp_spectrum, freq_space_interpolation
from fedfgs_loss import FedFGSCompositeLoss
from dataset_polyp import PolypDataset, load_polyp_metadata

def test_pipeline():
    print("--- 1. Testing DFT-HT ---")
    img1 = torch.randn(3, 1024, 1024)
    img2 = torch.randn(3, 1024, 1024)
    amp2 = extract_amp_spectrum(img2)
    aug = freq_space_interpolation(img1, amp2, L=0.1, ratio=0.5, alpha=0.05)
    assert aug.shape == img1.shape
    print(f"DFT-HT success! Output shape: {aug.shape}")

    print("\n--- 2. Testing FedFGSCompositeLoss ---")
    loss_fn = FedFGSCompositeLoss(lambda_dice=0.7, lambda_bce=0.2, lambda_focal=0.1)
    pred_logits = torch.randn(2, 1, 256, 256, requires_grad=True)
    target_mask = (torch.rand(2, 1, 256, 256) > 0.7).long()
    loss = loss_fn(pred_logits, target_mask)
    loss.backward()
    assert pred_logits.grad is not None
    print(f"FedFGSCompositeLoss success! Loss: {loss.item():.4f}, Grad norm: {pred_logits.grad.norm().item():.4f}")

    print("\n--- 3. Testing Metadata Loading ---")
    meta = load_polyp_metadata("polyp_db_dataset/metadata")
    print(f"Loaded {len(meta)} metadata entries.")
    assert len(meta) > 0

    print("\n--- 4. Testing Dataset with Amplitude Pool ---")
    dummy_pool = [torch.randn(3, 1024, 1024)]
    ds = PolypDataset(
        data_root="polyp_db_dataset/data",
        client_id="0",
        mode="train",
        image_size=1024,
        target_amp_pool=dummy_pool,
        dftht_prob=1.0,
        metadata_dict=meta,
    )
    assert len(ds) > 0
    sample = ds[0]
    print(f"Sample loaded: {sample['image_name']}, img shape: {sample['image'].shape}, mask shape: {sample['mask'].shape}")
    print(f"Small polyp: {sample['small_polyp'].item()}, Inv area: {sample['inv_area'].item():.2f}")
    print("\n ALL UNIT TESTS PASSED SUCCESSFULLY! ")

if __name__ == "__main__":
    test_pipeline()
