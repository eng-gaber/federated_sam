"""
Validation & Test Evaluation Script for Fed-FGS SAM ViT-B + LoRA Checkpoints.
Adapted from fed-dftht-main/src/evaluate_polyp_sizes.py and evaluate_centers.py.

Evaluates:
1. In-Domain Validation & Test Performance across training centers C1-C5.
2. Out-of-Domain Unseen Generalization on strictly held-out Center 6.
3. Size-stratified metrics: Overall DSC, Small Polyp DSC (DSC_S), Large Polyp DSC (DSC_L).
4. All 8 Paper Table I metrics: DSC, IoU, PA, Recall, FNR, FPR, Youden Index, HD95.
"""
import argparse
import os
import sys

# Ensure repository root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.sam import sam_model_registry
from models.sam_LoRa import LoRA_Sam
from dataset_polyp import PolypDataset, load_polyp_metadata, CLIENT_MAP

try:
    from monai.metrics import compute_hausdorff_distance
    HAS_MONAI_HD95 = True
except ImportError:
    HAS_MONAI_HD95 = False


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fed-FGS SAM Checkpoint")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/fed_fgs/FedFGS_SAM_best.pth",
        help="Path to .pth checkpoint",
    )
    parser.add_argument("--arch", type=str, default="vit_b")
    parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_b_01ec64.pth")
    parser.add_argument("--num_cls", type=int, default=2)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--data_root", type=str, default="polyp_db_dataset/data")
    parser.add_argument("--metadata_root", type=str, default="polyp_db_dataset/metadata")
    parser.add_argument("--out_csv", type=str, default="eval_results_detailed.csv")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples per center for quick check")

    args = parser.parse_args()
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.devices = [torch.device(device_name)]
    args.if_encoder_lora_layer = True
    args.encoder_lora_layer = []
    args.if_decoder_lora_layer = False
    args.if_split_encoder_gpus = False
    args.if_encoder_adapter = False
    args.encoder_adapter_depths = []
    args.if_mask_decoder_adapter = False
    args.decoder_adapt_depth = 0
    args.thd = False
    return args


def load_model(args, ckpt_path: str, device: torch.device):
    print(f"\n[Model] Loading base SAM ViT-B with LoRA (r={args.lora_rank})...")
    base_sam = sam_model_registry[args.arch](
        args=args,
        checkpoint=args.sam_checkpoint,
        num_classes=args.num_cls,
    )
    lora_wrapper = LoRA_Sam(args, base_sam, r=args.lora_rank)
    model = lora_wrapper.sam.to(device)

    print(f"[Model] Loading checkpoint weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "lora_parameters" in checkpoint:
        lora_params = checkpoint["lora_parameters"]
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        for p, new_val in zip(trainable_params, lora_params):
            p.data = torch.from_numpy(new_val).to(device=p.device, dtype=p.dtype)
        print(f"[Model] Successfully loaded {len(lora_params)} LoRA parameter tensors!")
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("[Model] Successfully loaded model_state_dict!")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print("[Model] Loaded state dict directly!")

    model.eval()
    return model


def evaluate_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    center_label: str,
    split_name: str,
    max_samples: int = None,
    args = None,
) -> Tuple[Dict[str, float], List[dict]]:
    """
    Evaluates dataloader, returns summary metrics and individual sample records.
    """
    records = []
    overall_dices = []
    small_dices = []
    large_dices = []
    ious = []
    pas = []
    recalls = []
    fnrs = []
    fprs = []
    hd95s = []

    samples_processed = 0

    use_amp = getattr(args, "use_amp", True) and device.type == "cuda"
    with torch.inference_mode():
        for batch in dataloader:
            if max_samples and samples_processed >= max_samples:
                break

            imgs = batch["image"].to(device, non_blocking=True)
            msks = batch["mask"].to(device, non_blocking=True)
            img_names = batch["image_name"]
            is_smalls = batch["small_polyp"]
            inv_areas = batch["inv_area"]
            B = imgs.shape[0]

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                img_emb = model.image_encoder(imgs)
                sparse_emb, dense_emb = model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )
                pred, _ = model.mask_decoder(
                    image_embeddings=img_emb,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )

            # Upsample prediction to full resolution (1024x1024)
            if pred.shape[-2:] != msks.shape[-2:]:
                pred_full = F.interpolate(pred, size=msks.shape[-2:], mode="bilinear", align_corners=False)
            else:
                pred_full = pred

            pred_bin = (torch.sigmoid(pred_full) > 0.5).float()
            gt_bin = (msks > 0).float()

            pred_flat = pred_bin.flatten(1)
            gt_flat = gt_bin.flatten(1)

            tp = (pred_flat * gt_flat).sum(dim=1)
            fp = (pred_flat * (1.0 - gt_flat)).sum(dim=1)
            fn = ((1.0 - pred_flat) * gt_flat).sum(dim=1)
            tn = ((1.0 - pred_flat) * (1.0 - gt_flat)).sum(dim=1)

            sample_dices = (2.0 * tp + 1e-5) / (2.0 * tp + fp + fn + 1e-5)
            sample_ious = (tp + 1e-5) / (tp + fp + fn + 1e-5)
            sample_pas = (tp + tn) / (tp + tn + fp + fn + 1e-5)
            sample_recalls = (tp + 1e-5) / (tp + fn + 1e-5)
            sample_fnrs = (fn + 1e-5) / (tp + fn + 1e-5)
            sample_fprs = (fp + 1e-5) / (fp + tn + 1e-5)

            # HD95
            sample_hd95s = [0.0] * B
            if HAS_MONAI_HD95:
                for i in range(B):
                    try:
                        if gt_flat[i].sum() > 0 and pred_flat[i].sum() > 0:
                            hd = compute_hausdorff_distance(
                                pred_bin[i:i+1],
                                gt_bin[i:i+1],
                                percentile=95.0,
                                include_background=False
                            )
                            hd_val = hd.item()
                            if np.isfinite(hd_val):
                                sample_hd95s[i] = hd_val
                    except Exception:
                        sample_hd95s[i] = 0.0

            for i in range(B):
                dsc_i = sample_dices[i].item()
                iou_i = sample_ious[i].item()
                pa_i = sample_pas[i].item()
                rec_i = sample_recalls[i].item()
                fnr_i = sample_fnrs[i].item()
                fpr_i = sample_fprs[i].item()
                hd_i = sample_hd95s[i]
                yi_i = rec_i - fpr_i

                is_sm_i = bool(is_smalls[i].item() == 1)
                inv_area_i = float(inv_areas[i].item())
                has_polyp = bool(gt_flat[i].sum().item() > 0)

                overall_dices.append(dsc_i)
                ious.append(iou_i)
                pas.append(pa_i)
                recalls.append(rec_i)
                fnrs.append(fnr_i)
                fprs.append(fpr_i)
                if hd_i > 0:
                    hd95s.append(hd_i)

                if has_polyp:
                    if is_sm_i:
                        small_dices.append(dsc_i)
                    else:
                        large_dices.append(dsc_i)

                records.append({
                    "center": center_label,
                    "split": split_name,
                    "image_name": img_names[i],
                    "dsc": dsc_i,
                    "iou": iou_i,
                    "pa": pa_i,
                    "recall": rec_i,
                    "fnr": fnr_i,
                    "fpr": fpr_i,
                    "yi": yi_i,
                    "hd95": hd_i,
                    "is_small": is_sm_i,
                    "inv_area": inv_area_i,
                    "has_polyp": has_polyp,
                })

            samples_processed += B

    summary = {
        "count": len(overall_dices),
        "dsc": float(np.mean(overall_dices)) if overall_dices else 0.0,
        "dsc_small": float(np.mean(small_dices)) if small_dices else 0.0,
        "dsc_large": float(np.mean(large_dices)) if large_dices else 0.0,
        "num_small": len(small_dices),
        "num_large": len(large_dices),
        "iou": float(np.mean(ious)) if ious else 0.0,
        "pa": float(np.mean(pas)) if pas else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "fnr": float(np.mean(fnrs)) if fnrs else 0.0,
        "fpr": float(np.mean(fprs)) if fprs else 0.0,
        "yi": float(np.mean(recalls) - np.mean(fprs)) if recalls and fprs else 0.0,
        "hd95": float(np.mean(hd95s)) if hd95s else 0.0,
    }
    return summary, records


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    metadata_dict = load_polyp_metadata(args.metadata_root)
    print(f"Loaded {len(metadata_dict)} metadata entries from {args.metadata_root}")

    model = load_model(args, args.ckpt, device)

    all_records = []
    center_summaries = {}

    centers = ["0", "1", "2", "3", "4", "5"]  # C1-C6

    print("\n" + "="*80)
    print(f"🔬 RUNNING BENCHMARK EVALUATION ON CHECKPOINT: {os.path.basename(args.ckpt)}")
    print("="*80)

    for cid in centers:
        c_name = CLIENT_MAP[cid]
        is_unseen = (cid == "5")
        tag = "[OOD UNSEEN]" if is_unseen else "[IN-DOMAIN]"

        print(f"\n--- Evaluating Center {c_name} {tag} ---")

        # 1. Validation Split
        val_loader = DataLoader(
            PolypDataset(
                data_root=args.data_root,
                client_id=cid,
                mode="val",
                image_size=args.image_size,
                metadata_dict=metadata_dict,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_sum, val_rec = evaluate_dataloader(model, val_loader, device, c_name, "val", args.max_samples, args=args)
        all_records.extend(val_rec)

        # 2. Test Split
        test_loader = DataLoader(
            PolypDataset(
                data_root=args.data_root,
                client_id=cid,
                mode="test",
                image_size=args.image_size,
                metadata_dict=metadata_dict,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_sum, test_rec = evaluate_dataloader(model, test_loader, device, c_name, "test", args.max_samples, args=args)
        all_records.extend(test_rec)

        center_summaries[c_name] = {"val": val_sum, "test": test_sum}
        print(f"  {c_name} Test DSC: {test_sum['dsc']:.4f} (Small: {test_sum['dsc_small']:.4f} [N={test_sum['num_small']}], Large: {test_sum['dsc_large']:.4f} [N={test_sum['num_large']}]) | IoU: {test_sum['iou']:.4f}")

    # =========================================================================
    # SUMMARY TABLES
    # =========================================================================
    print("\n" + "="*85)
    print("📊 TABLE 1: PER-CENTER TEST PERFORMANCE (Paper Format)")
    print("="*85)
    print(f"{'Center':<10} | {'Status':<14} | {'Overall DSC':<12} | {'DSC_S (Small)':<14} | {'DSC_L (Large)':<14} | {'IoU':<8} | {'PA':<8} | {'Recall':<8}")
    print("-" * 85)

    in_domain_dscs = []
    in_domain_smalls = []
    in_domain_larges = []
    in_domain_ious = []
    in_domain_pas = []
    in_domain_recalls = []

    for cid in centers:
        c_name = CLIENT_MAP[cid]
        s = center_summaries[c_name]["test"]
        is_unseen = (cid == "5")
        status = "OOD Unseen" if is_unseen else "In-Domain"

        print(f"{c_name:<10} | {status:<14} | {s['dsc']:<12.4f} | {s['dsc_small']:<14.4f} | {s['dsc_large']:<14.4f} | {s['iou']:<8.4f} | {s['pa']:<8.4f} | {s['recall']:<8.4f}")

        if not is_unseen:
            in_domain_dscs.append(s['dsc'])
            if s['dsc_small'] > 0: in_domain_smalls.append(s['dsc_small'])
            if s['dsc_large'] > 0: in_domain_larges.append(s['dsc_large'])
            in_domain_ious.append(s['iou'])
            in_domain_pas.append(s['pa'])
            in_domain_recalls.append(s['recall'])

    print("-" * 85)
    mean_in_dsc = np.mean(in_domain_dscs)
    mean_in_small = np.mean(in_domain_smalls) if in_domain_smalls else 0.0
    mean_in_large = np.mean(in_domain_larges) if in_domain_larges else 0.0
    mean_in_iou = np.mean(in_domain_ious)
    mean_in_pa = np.mean(in_domain_pas)
    mean_in_rec = np.mean(in_domain_recalls)
    print(f"{'C1-C5 Mean':<10} | {'In-Domain':<14} | {mean_in_dsc:<12.4f} | {mean_in_small:<14.4f} | {mean_in_large:<14.4f} | {mean_in_iou:<8.4f} | {mean_in_pa:<8.4f} | {mean_in_rec:<8.4f}")

    c6 = center_summaries["C6"]["test"]
    print(f"{'C6 (OOD)':<10} | {'Unseen Domain':<14} | {c6['dsc']:<12.4f} | {c6['dsc_small']:<14.4f} | {c6['dsc_large']:<14.4f} | {c6['iou']:<8.4f} | {c6['pa']:<8.4f} | {c6['recall']:<8.4f}")
    print("=" * 85)

    # Save detailed CSV
    df = pd.DataFrame(all_records)
    df.to_csv(args.out_csv, index=False)
    print(f"\n💾 Saved individual sample evaluation records to: {args.out_csv}")


if __name__ == "__main__":
    main()
