"""
Fed-FGS: Domain-Generalized Federated Polyp Segmentation
via Fourier Hard-Thresholding (DFT-HT) and Difficulty-Aware Gradient Scaling (FedGS)
Using SAM (Segment Anything Model) ViT-B + LoRA Fine-Tuning.

Paper Specifications Adherence:
- Backbone: SAM ViT-B + LoRA (rank=4) on Image Encoder attention layers.
- Strict Out-of-Distribution Setup: Centers 1-5 participate in federated training.
  Center 6 is strictly held out as the unseen test domain and excluded from amplitude pools.
- Composite Loss (Paper Eq. 5): 0.7 * L_Dice + 0.2 * L_BCE + 0.1 * L_Focal.
- Frequency Augmentation (DFT-HT): L=0.1, alpha=0.05, prob=0.3, ratio in [0, 1].
- Difficulty-Aware Gradient Scaling (FedGS): Dynamic inverse-area scaling eta_t for small polyps.
- Size-Stratified Evaluation: Overall DSC, Small Polyp DSC (DSC_S), Large Polyp DSC (DSC_L).
"""
import argparse
import os
import sys
import warnings
from typing import Dict, List, Tuple, Optional

# Ensure project root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Suppress noisy library and registry overwrite warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import flwr as fl

from models.sam import sam_model_registry
from models.sam_LoRa import LoRA_Sam
from dataset_polyp import (
    PolypDataset,
    get_client_dataloader,
    build_amplitude_pools_for_clients,
    load_polyp_metadata,
    CLIENT_MAP,
)
from fedfgs_loss import FedFGSCompositeLoss
from federated.config import load_yaml_config, setup_experiment_run
from federated.tracker import ExperimentTracker

try:
    from monai.metrics import compute_hausdorff_distance
    HAS_MONAI_HD95 = True
except ImportError:
    HAS_MONAI_HD95 = False


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_lora_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract only trainable (LoRA) parameters as a list of numpy arrays."""
    return [p.detach().cpu().numpy() for p in model.parameters() if p.requires_grad]


def set_lora_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load trainable (LoRA) parameters into model."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) != len(parameters):
        raise ValueError(
            f"Parameter length mismatch: model has {len(trainable_params)} trainable tensors, "
            f"received {len(parameters)}."
        )
    for p, new_val in zip(trainable_params, parameters):
        p.data = torch.from_numpy(new_val).to(device=p.device, dtype=p.dtype)


def compute_batch_metrics(
    pred_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    is_smalls: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute all 8 evaluation metrics from Paper Table I:
    1. Dice Score (DSC)
    2. Intersection over Union (IoU / mIoU)
    3. Pixel Accuracy (PA)
    4. Recall (Sensitivity / TPR)
    5. False Negative Rate (FNR = 1 - Recall)
    6. False Positive Rate (FPR)
    7. Youden Index (YI = Recall - FPR)
    8. 95% Hausdorff Distance (HD95)
    Plus size-stratified DSC_S (small) and DSC_L (large).
    """
    with torch.no_grad():
        if pred_logits.shape[1] == 1:
            pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
        elif pred_logits.shape[1] == 2:
            pred_bin = (pred_logits.argmax(dim=1, keepdim=True) == 1).float()
        else:
            pred_bin = (torch.sigmoid(pred_logits[:, 0:1, :, :]) > 0.5).float()

        gt_bin = (gt_masks > 0).float()
        if gt_bin.shape[-2:] != pred_bin.shape[-2:]:
            gt_bin = F.interpolate(gt_bin, size=pred_bin.shape[-2:], mode="nearest")

        pred_flat = pred_bin.flatten(1)
        gt_flat = gt_bin.flatten(1)

        tp = (pred_flat * gt_flat).sum(dim=1)
        fp = (pred_flat * (1.0 - gt_flat)).sum(dim=1)
        fn = ((1.0 - pred_flat) * gt_flat).sum(dim=1)
        tn = ((1.0 - pred_flat) * (1.0 - gt_flat)).sum(dim=1)

        sample_dices = (2.0 * tp + 1e-5) / (2.0 * tp + fp + fn + 1e-5)
        dice = sample_dices.mean().item()
        iou = ((tp + 1e-5) / (tp + fp + fn + 1e-5)).mean().item()
        pa = ((tp + tn) / (tp + tn + fp + fn + 1e-5)).mean().item()
        recall = ((tp + 1e-5) / (tp + fn + 1e-5)).mean().item()
        fnr = ((fn + 1e-5) / (tp + fn + 1e-5)).mean().item()
        fpr = ((fp + 1e-5) / (fp + tn + 1e-5)).mean().item()
        yi = recall - fpr

        # Stratified Small vs Large Dice
        small_dice_list = []
        large_dice_list = []
        if is_smalls is not None:
            for i in range(len(sample_dices)):
                # Only evaluate size when polyp is present in ground truth
                if gt_flat[i].sum() > 0:
                    if is_smalls[i].item() == 1:
                        small_dice_list.append(sample_dices[i].item())
                    else:
                        large_dice_list.append(sample_dices[i].item())

        hd95_val = 0.0
        if HAS_MONAI_HD95:
            try:
                hd = compute_hausdorff_distance(pred_bin, gt_bin, percentile=95.0, include_background=False)
                hd = hd[torch.isfinite(hd)]
                if len(hd) > 0:
                    hd95_val = hd.mean().item()
            except Exception:
                hd95_val = 0.0

    result = {
        "dice": dice,
        "iou": iou,
        "pa": pa,
        "recall": recall,
        "fnr": fnr,
        "fpr": fpr,
        "yi": yi,
        "hd95": hd95_val,
        "small_dices": small_dice_list,
        "large_dices": large_dice_list,
    }
    return result


class PolypSamClient(fl.client.NumPyClient):
    """
    Client for Fed-FGS training and evaluation.
    Supports DFT-HT augmentation, FedGS difficulty-aware gradient scaling, and composite loss.
    """
    def __init__(
        self,
        cid: str,
        args: argparse.Namespace,
        amp_pool: Optional[List[torch.Tensor]] = None,
        metadata_dict: Optional[Dict[str, dict]] = None,
    ):
        self.cid = str(cid)
        self.args = args
        self.device = get_device()
        self.client_name = CLIENT_MAP.get(self.cid, f"C{int(self.cid)+1}")
        self.amp_pool = amp_pool or []
        self.metadata_dict = metadata_dict or {}

        # Build SAM ViT-B model
        base_sam = sam_model_registry[args.arch](
            args=args,
            checkpoint=args.sam_checkpoint,
            num_classes=args.num_cls,
        )

        # Apply LoRA surgery (r=4)
        self.lora_wrapper = LoRA_Sam(args, base_sam, r=args.lora_rank)
        self.model = self.lora_wrapper.sam.to(self.device)

        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_count = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        print(f"[Client {self.cid} - {self.client_name}] Trainable LoRA params: {trainable_count:,} | Frozen params: {frozen_count:,}")

        # Paper Optimizer: Adam with β1=0.9, β2=0.99, lr=0.001, weight_decay=1e-5
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            betas=(0.9, 0.99),
            weight_decay=args.weight_decay,
        )

        # Composite Loss: 0.7 * Dice + 0.2 * BCE + 0.1 * Focal (Paper Eq. 5)
        self.criterion = FedFGSCompositeLoss(
            lambda_dice=args.lambda_dice,
            lambda_bce=args.lambda_bce,
            lambda_focal=args.lambda_focal,
        ).to(self.device)

        # Training Dataloader with DFT-HT frequency augmentation
        self.train_loader = get_client_dataloader(
            client_id=self.cid,
            mode="train",
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            data_root=args.data_root,
            image_size=args.image_size,
            target_amp_pool=self.amp_pool,
            dftht_prob=args.dftht_prob,
            dftht_L=args.dftht_L,
            dftht_alpha=args.dftht_alpha,
            metadata_dict=self.metadata_dict,
            split_type=getattr(args, "split_type", "fold"),
            fold_nr=getattr(args, "fold_nr", 1),
            splits_root=getattr(args, "splits_root", None),
        )

        # Validation Dataloader (no frequency augmentation)
        self.val_loader = get_client_dataloader(
            client_id=self.cid,
            mode="val",
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            data_root=args.data_root,
            image_size=args.image_size,
            metadata_dict=self.metadata_dict,
            split_type=getattr(args, "split_type", "fold"),
            fold_nr=getattr(args, "fold_nr", 1),
            splits_root=getattr(args, "splits_root", None),
        )

        # Test Dataloader (for held-out client OOD evaluation)
        self.test_loader = get_client_dataloader(
            client_id=self.cid,
            mode="test",
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            data_root=args.data_root,
            image_size=args.image_size,
            metadata_dict=self.metadata_dict,
            split_type=getattr(args, "split_type", "fold"),
            fold_nr=getattr(args, "fold_nr", 1),
            splits_root=getattr(args, "splits_root", None),
        )

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return extract_lora_parameters(self.model)

    def fit(
        self,
        parameters: List[np.ndarray],
        round_idx: int = 1,
        config: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        set_lora_parameters(self.model, parameters)
        self.model.train()
        if hasattr(self.model, "set_center_id"):
            try:
                self.model.set_center_id(int(self.cid))
            except Exception:
                self.model.set_center_id(None)
        if hasattr(self.model, "reset_expert_usage"):
            self.model.reset_expert_usage()

        # Cosine learning rate scheduling across rounds
        current_lr = self.args.learning_rate
        if getattr(self.args, "lr_schedule", "cosine") == "cosine" and self.args.num_rounds > 1:
            min_lr = getattr(self.args, "min_lr", 1e-5)
            max_lr = self.args.learning_rate
            current_lr = min_lr + 0.5 * (max_lr - min_lr) * (
                1.0 + np.cos(np.pi * (round_idx - 1) / self.args.num_rounds)
            )
            for pg in self.optimizer.param_groups:
                pg["lr"] = current_lr

        num_samples = len(self.train_loader.dataset)
        if num_samples == 0:
            return extract_lora_parameters(self.model), 0, {}

        epochs = self.args.local_epochs
        total_loss = 0.0
        p_factors_logged = []

        print(f"[Client {self.cid} - {self.client_name}] Local training: {epochs} epoch(s) | LR: {current_lr:.6f} | Samples: {num_samples}...")

        for epoch in range(epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(self.train_loader):
                if self.args.max_steps_per_epoch and step >= self.args.max_steps_per_epoch:
                    break

                imgs = batch["image"].to(self.device)
                msks = batch["mask"].to(self.device)
                is_smalls = batch["small_polyp"]
                inv_areas = batch["inv_area"]
                B = imgs.shape[0]

                # 1. Forward pass with mixed precision (BF16 on RTX 5090)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if self.args.if_update_encoder:
                        img_emb = self.model.image_encoder(imgs)
                    else:
                        with torch.no_grad():
                            img_emb = self.model.image_encoder(imgs)

                    sparse_emb, dense_emb = self.model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                    )

                    # Single binary mask prediction (multimask_output=False)
                    pred, _ = self.model.mask_decoder(
                        image_embeddings=img_emb,
                        image_pe=self.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=False,
                    )

                    # 2. Base Composite Loss (0.7 Dice + 0.2 BCE + 0.1 Focal) or Scale-Aware Dual-Decoder MoE Loss
                    if getattr(self.args, "enable_scale_decoder_moe", False) and hasattr(self.model.mask_decoder, "compute_scale_aware_loss"):
                        loss, _ = self.model.mask_decoder.compute_scale_aware_loss(
                            criterion=self.criterion,
                            target_masks=msks,
                            is_smalls=is_smalls,
                        )
                    else:
                        loss = self.criterion(pred, msks)

                # 3. FedGS Difficulty-Aware Gradient Scaling (eta_t)
                if self.args.enable_fedgs:
                    small_factors = []
                    for i in range(B):
                        if is_smalls[i].item() == 1:
                            ia = float(inv_areas[i].item())
                            if ia <= 1.0:
                                small_factors.append(1.0)
                            else:
                                log_val = np.log(ia) / np.log(100.0)
                                small_factors.append(np.tanh(log_val ** 2))

                    if len(small_factors) > 0:
                        eta_t = 1.0 + float(sum(small_factors) * (2.0 / B))
                    else:
                        eta_t = 1.0

                    p_factors_logged.append(eta_t)
                    loss = loss * eta_t

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  [Client {self.cid}] Warning: skipping batch with NaN/Inf loss")
                    continue

                # 4. Backward & Gradient Clipping
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / max(1, step + 1)
            total_loss += avg_epoch_loss

        avg_loss = total_loss / max(1, epochs)
        mean_eta = np.mean(p_factors_logged) if p_factors_logged else 1.0
        print(f"  ✓ [{self.client_name}] Completed {epochs} local epochs ({num_samples} samples) | Loss: {avg_loss:.4f} | Gradient Scaling (eta_t): {mean_eta:.3f}")

        expert_usage = {}
        if getattr(self.args, "enable_mode", False) and hasattr(self.model, "get_expert_usage"):
            expert_usage = self.model.get_expert_usage()

        return extract_lora_parameters(self.model), num_samples, {
            "loss": avg_loss,
            "eta_t": mean_eta,
            "expert_usage": expert_usage,
        }

    def evaluate_loader(self, loader: DataLoader) -> Tuple[float, int, Dict[str, float]]:
        self.model.eval()
        if hasattr(self.model, "set_center_id"):
            self.model.set_center_id(None)
        num_samples = len(loader.dataset)
        if num_samples == 0:
            return 0.0, 0, {"dice": 0.0, "iou": 0.0}

        val_loss = 0.0
        val_metrics_accum = {}
        all_small_dices = []
        all_large_dices = []
        all_gate_w_smalls = []
        all_gate_w_larges = []
        batches = 0

        with torch.inference_mode():
            for step, batch in enumerate(loader):
                if self.args.max_eval_steps and step >= self.args.max_eval_steps:
                    break

                imgs = batch["image"].to(self.device, non_blocking=True)
                msks = batch["mask"].to(self.device, non_blocking=True)
                is_smalls = batch["small_polyp"]

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    img_emb = self.model.image_encoder(imgs)
                    sparse_emb, dense_emb = self.model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                    )
                    pred, _ = self.model.mask_decoder(
                        image_embeddings=img_emb,
                        image_pe=self.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=False,
                    )

                    loss = self.criterion(pred, msks)

                    if getattr(self.args, "enable_scale_decoder_moe", False) and hasattr(self.model.mask_decoder, "last_aux"):
                        aux = self.model.mask_decoder.last_aux
                        if aux is not None and "weights" in aux:
                            w = aux["weights"].detach().cpu()
                            all_gate_w_smalls.extend(w[:, 0].tolist())
                            all_gate_w_larges.extend(w[:, 1].tolist())

                # Upsample prediction to full resolution (1024x1024) for evaluation
                if pred.shape[-2:] != msks.shape[-2:]:
                    pred_full = F.interpolate(pred, size=msks.shape[-2:], mode="bilinear", align_corners=False)
                else:
                    pred_full = pred

                metrics = compute_batch_metrics(pred_full, msks, is_smalls=is_smalls)

                for k, v in metrics.items():
                    if k == "small_dices":
                        all_small_dices.extend(v)
                    elif k == "large_dices":
                        all_large_dices.extend(v)
                    else:
                        val_metrics_accum[k] = val_metrics_accum.get(k, 0.0) + v

                val_loss += loss.item()
                batches += 1

        avg_loss = val_loss / max(1, batches)
        avg_metrics = {k: v / max(1, batches) for k, v in val_metrics_accum.items()}
        avg_metrics["dice_small"] = float(np.mean(all_small_dices)) if len(all_small_dices) > 0 else 0.0
        avg_metrics["dice_large"] = float(np.mean(all_large_dices)) if len(all_large_dices) > 0 else 0.0
        avg_metrics["num_small_samples"] = len(all_small_dices)
        avg_metrics["num_large_samples"] = len(all_large_dices)
        if len(all_gate_w_smalls) > 0:
            avg_metrics["gate_w_small"] = float(np.mean(all_gate_w_smalls))
            avg_metrics["gate_w_large"] = float(np.mean(all_gate_w_larges))

        return avg_loss, num_samples, avg_metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict[str, float]]:
        set_lora_parameters(self.model, parameters)
        loss, num_samples, metrics = self.evaluate_loader(self.val_loader)
        return loss, num_samples, metrics


def weighted_average_eval_metrics(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Compute sample-weighted average across clients."""
    total_samples = sum(num_samples for num_samples, _ in metrics)
    if total_samples == 0:
        return {}

    aggregated = {}
    metric_keys = metrics[0][1].keys()
    for k in metric_keys:
        aggregated[k] = sum(num_samples * m.get(k, 0.0) for num_samples, m in metrics) / total_samples

    return aggregated


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_arguments():
    # Pass 1: Parse --config if provided
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--config", type=str, default=None, help="Path to YAML or JSON experiment config file")
    conf_args, _ = conf_parser.parse_known_args()

    config_dict = {}
    if conf_args.config and os.path.isfile(conf_args.config):
        raw_config = load_yaml_config(conf_args.config)
        for section, values in raw_config.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    config_dict[k] = v
            else:
                config_dict[section] = values

    # Pass 2: Main parser with defaults from config
    parser = argparse.ArgumentParser(description="Fed-FGS: Federated Polyp Segmentation with SAM + LoRA")

    # Federated Setup
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of FL rounds")
    parser.add_argument("--num_clients", type=int, default=5, help="Number of training clients (Centers 1-5)")
    parser.add_argument("--out_center", type=str, default="5", help="Client ID of unseen held-out center (5 = C6)")
    parser.add_argument("--local_epochs", type=int, default=6, help="Local epochs per round")
    parser.add_argument("--batch_size", type=int, default=2, help="Local batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="LoRA learning rate (paper UTNet was 1e-4)")
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "constant"], help="LR schedule")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum LR for cosine schedule")
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # SAM & LoRA
    parser.add_argument("--arch", type=str, default="vit_b")
    parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_b_01ec64.pth")
    parser.add_argument("--num_cls", type=int, default=2)
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank r=4")
    parser.add_argument("--if_encoder_lora_layer", type=str2bool, default=True)
    parser.add_argument("--encoder_lora_layer", type=list, default=[])
    parser.add_argument("--if_decoder_lora_layer", type=str2bool, default=True)
    parser.add_argument("--if_update_encoder", type=str2bool, default=True)
    parser.add_argument("--devices", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--if_split_encoder_gpus", type=str2bool, default=False)
    parser.add_argument("--if_encoder_adapter", type=str2bool, default=False)
    parser.add_argument("--encoder_adapter_depths", type=list, default=[])
    parser.add_argument("--if_mask_decoder_adapter", type=str2bool, default=False)
    parser.add_argument("--decoder_adapt_depth", type=int, default=0)
    parser.add_argument("--thd", type=str2bool, default=False)

    # MoDE (Mixture of Domain Experts) Architecture
    parser.add_argument("--enable_mode", type=str2bool, default=False, help="Enable MoDE in selected encoder blocks")
    parser.add_argument("--mode_blocks", default=[10, 11], help="Encoder block indices where MoDE is placed")
    parser.add_argument("--mode_num_experts", type=int, default=5, help="Number of experts per MoDE block (default: 5)")
    parser.add_argument("--mode_lora_rank", type=int, default=4, help="LoRA rank r for expert adapters (default: 4)")
    parser.add_argument("--mode_lora_alpha", type=float, default=4.0, help="LoRA alpha for expert adapters")
    parser.add_argument("--mode_lora_dropout", type=float, default=0.0, help="Dropout for expert LoRA")
    parser.add_argument("--mode_top_k", type=int, default=2, help="Top-k expert routing (default: 2)")
    parser.add_argument("--mode_gate_hidden_dim", type=int, default=128, help="Hidden dimension of gating network (default: 128)")
    parser.add_argument("--mode_routing_mode", type=str, default="mode_a", choices=["mode_a", "mode_b"], help="Routing mode: 'mode_a' (token-driven) or 'mode_b' (center-prior)")
    parser.add_argument("--mode_center_prior_lambda", type=float, default=0.2, help="Lambda weight for center prior in mode_b")

    # Scale-Aware Dual-Decoder MoE
    parser.add_argument("--enable_scale_decoder_moe", type=str2bool, default=False, help="Enable Scale-Aware Dual-Decoder MoE")
    parser.add_argument("--scale_moe_gate_hidden_dim", type=int, default=64, help="Hidden dimension of scale gating MLP")
    parser.add_argument("--scale_moe_beta", type=float, default=0.5, help="Scale specialization loss weight beta (default: 0.5)")
    parser.add_argument("--scale_moe_lambda_gate", type=float, default=0.1, help="Gating supervision loss weight lambda_gate (default: 0.1)")

    # Fed-FGS Hyperparameters
    parser.add_argument("--dftht_prob", type=float, default=0.3, help="Probability of applying DFT-HT augmentation")
    parser.add_argument("--dftht_L", type=float, default=0.1, help="Low-frequency radius ratio L=0.1")
    parser.add_argument("--dftht_alpha", type=float, default=0.05, help="Hard-threshold ratio alpha=0.05")
    parser.add_argument("--amp_pool_max_samples", type=int, default=25, help="Max amplitude spectra per foreign center")
    parser.add_argument("--enable_fedgs", type=str2bool, default=True, help="Enable difficulty-aware gradient scaling")

    # Paper Composite Loss Weights (Eq. 5)
    parser.add_argument("--lambda_dice", type=float, default=0.7)
    parser.add_argument("--lambda_bce", type=float, default=0.2)
    parser.add_argument("--lambda_focal", type=float, default=0.1)

    # Dataset & Runtime (Fold 1 default)
    parser.add_argument("--data_root", type=str, default="polyp_db_dataset/data")
    parser.add_argument("--metadata_root", type=str, default="polyp_db_dataset/metadata")
    parser.add_argument("--split_type", type=str, default="fold", help="Dataset split type: fold or center")
    parser.add_argument("--fold_nr", type=int, default=1, help="Fold index (1 to 5) matching paper")
    parser.add_argument("--splits_root", type=str, default="fed-dftht-main/data_polypgen/splits/fold_splits")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--config", type=str, default=conf_args.config, help="Path to YAML or JSON experiment config file")
    parser.add_argument("--run_name", type=str, default=None, help="Custom experiment run name")
    parser.add_argument("--output_root", type=str, default="checkpoints/runs", help="Directory where run folders are created")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional direct output directory override")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--save_checkpoints", type=str2bool, default=True)
    parser.add_argument("--resume", type=str2bool, default=False, help="Resume training from existing checkpoint in run_dir")

    if config_dict:
        parser.set_defaults(**config_dict)

    args = parser.parse_args()
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.devices = [torch.device(device_name)]
    return args


def run_fed_fgs_simulation(args, tracker=None):
    """
    Direct Federated Simulation for Fed-FGS with SAM + LoRA.
    - Trains on Centers 1-5.
    - Evaluates strictly held-out Center 6 (Out-of-Distribution).
    - Logs size-stratified metrics and saves best overall & best small models.
    """
    run_dir, clean_config = setup_experiment_run(args)
    if tracker is None:
        tracker = ExperimentTracker(run_dir=run_dir, run_name=args.run_name, config=clean_config)

    print(f"\n{'='*75}")
    print(f"🚀 Launching Fed-FGS Simulation: SAM ViT-B + LoRA (r={args.lora_rank})")
    print(f"   Split: {args.split_type.upper()} {args.fold_nr} | Image Size: {args.image_size}")
    print(f"   Training Centers: C1-C5 ({args.num_clients} clients)")
    print(f"   Held-out Unseen Domain: Center 6 (cid={args.out_center})")
    print(f"   Trainable: Image Encoder LoRA={args.if_encoder_lora_layer} (update={args.if_update_encoder}) | Mask Decoder LoRA={args.if_decoder_lora_layer}")
    if getattr(args, "enable_mode", False):
        print(f"   MoDE Architecture: ENABLED | Blocks={args.mode_blocks} | Experts={args.mode_num_experts} | Top-k={args.mode_top_k} | Rank={args.mode_lora_rank} | Mode={args.mode_routing_mode}")
    print(f"   Frequency Augmentation: DFT-HT (L={args.dftht_L}, alpha={args.dftht_alpha}, p={args.dftht_prob})")
    print(f"   Difficulty Scaling: FedGS (eta_t dynamic inverse area)")
    print(f"   Composite Loss: {args.lambda_dice}*Dice + {args.lambda_bce}*BCE + {args.lambda_focal}*Focal")
    print(f"   FL Rounds: {args.num_rounds} | Local Epochs: {args.local_epochs} | Base LR: {args.learning_rate} (Schedule: {args.lr_schedule})")
    print(f"{'='*75}\n")

    # 1. Load PolypGen metadata for difficulty scaling & size stratification
    metadata_dict = load_polyp_metadata(args.metadata_root)
    print(f"Loaded {len(metadata_dict)} metadata entries from {args.metadata_root}")

    # 2. Build foreign amplitude pools across training centers C1-C5 (excluding Center 6)
    train_cids = [str(i) for i in range(args.num_clients)]
    amp_pools = build_amplitude_pools_for_clients(
        client_ids=train_cids,
        out_center=args.out_center,
        data_root=args.data_root,
        max_samples_per_center=args.amp_pool_max_samples,
        image_size=args.image_size,
        split_type=args.split_type,
        fold_nr=args.fold_nr,
        splits_root=args.splits_root,
    )

    # 3. Initialize training clients (C1-C5)
    clients: List[PolypSamClient] = []
    for cid in train_cids:
        c = PolypSamClient(
            cid=cid,
            args=args,
            amp_pool=amp_pools.get(cid, []),
            metadata_dict=metadata_dict,
        )
        clients.append(c)

    # 4. Initialize held-out Center 6 client for Out-of-Distribution evaluation
    print(f"\n[Unseen Domain] Initializing strictly held-out Center 6 client (cid={args.out_center})...")
    c6_evaluator = PolypSamClient(
        cid=args.out_center,
        args=args,
        amp_pool=[],
        metadata_dict=metadata_dict,
    )

    # Initial global LoRA parameters
    global_parameters = clients[0].get_parameters(config={})
    best_c6_dice = tracker.best_c6_dsc
    best_c6_dice_small = tracker.best_c6_small_dsc
    start_round = 1

    # Check for resume
    if getattr(args, "resume", False):
        last_ckpt = os.path.join(args.run_dir, "FedFGS_SAM_last.pth")
        if os.path.isfile(last_ckpt):
            print(f"\n🔁 Resuming training from checkpoint: {last_ckpt}")
            ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
            global_parameters = ckpt["lora_parameters"]
            start_round = int(ckpt["round"]) + 1
            print(f"   ✓ Loaded {len(global_parameters)} LoRA tensors from Round {ckpt['round']}.")
            print(f"   ✓ Continuing from Round {start_round}/{args.num_rounds}.")
            print(f"   ✓ Best C6 DSC so far: {best_c6_dice:.4f} | Best Small DSC so far: {best_c6_dice_small:.4f}")
        else:
            print(f"\n⚠️  Resume requested but no checkpoint found at {last_ckpt}. Starting from Round 1.")

    # 5. Communication Rounds Loop
    for round_idx in range(start_round, args.num_rounds + 1):
        print(f"\n{'#'*75}")
        print(f"🔄 [Communication Round {round_idx}/{args.num_rounds}]")
        print(f"{'#'*75}")

        round_updates = []
        total_fit_samples = 0

        # Step A: Local Training across C1-C5
        client_usages = {}
        for client in clients:
            updated_params, num_samples, fit_metrics = client.fit(global_parameters, round_idx=round_idx, config={})
            if num_samples > 0:
                round_updates.append((updated_params, num_samples))
                total_fit_samples += num_samples
                if fit_metrics.get("expert_usage"):
                    client_usages[client.client_name] = fit_metrics["expert_usage"]

        # Log expert routing frequencies per client and per selected block
        if client_usages and getattr(args, "enable_mode", False):
            print(f"\n┌{'─'*76}┐")
            print(f"│ 🧭 MoDE EXPERT UTILIZATION PER CLIENT (Round {round_idx:>2}){' '*32}│")
            print(f"├{'─'*76}┤")
            for c_name, blocks_usage in client_usages.items():
                print(f"│  • Client {c_name:<5}:{' '*64}│")
                for blk_idx, dist in sorted(blocks_usage.items()):
                    dist_str = " | ".join(f"E{e}: {dist.get(e, 0.0)*100:4.1f}%" for e in range(args.mode_num_experts))
                    line_content = f"      Block {blk_idx:>2}:  {dist_str}"
                    print(f"│{line_content:<76}│")
            print(f"└{'─'*76}┘")

        # Step B: FedAvg LoRA Aggregation
        if round_updates and total_fit_samples > 0:
            new_global_params = [np.zeros_like(p) for p in global_parameters]
            for client_params, n_k in round_updates:
                weight = n_k / total_fit_samples
                for i in range(len(new_global_params)):
                    new_global_params[i] += weight * client_params[i]
            global_parameters = new_global_params

        # Step C: In-Domain Evaluation (mean over C1-C5)
        in_domain_metrics = []
        for client in clients:
            loss, num_val, m = client.evaluate(global_parameters, config={})
            if num_val > 0:
                in_domain_metrics.append((num_val, m))
        in_domain_agg = weighted_average_eval_metrics(in_domain_metrics)

        # Step D: Out-of-Domain Unseen Center 6 Evaluation
        c6_val_loss, c6_val_samples, c6_val_m = c6_evaluator.evaluate(global_parameters, config={})
        c6_test_loss, c6_test_samples, c6_test_m = c6_evaluator.evaluate_loader(c6_evaluator.test_loader)

        # Primary test metrics from C6 test split
        c6_dice = c6_test_m.get("dice", 0.0)
        c6_small = c6_test_m.get("dice_small", 0.0)
        c6_large = c6_test_m.get("dice_large", 0.0)
        c6_iou = c6_test_m.get("iou", 0.0)
        c6_pa = c6_test_m.get("pa", 0.0)
        c6_recall = c6_test_m.get("recall", 0.0)
        c6_hd95 = c6_test_m.get("hd95", 0.0)

        is_new_best = c6_dice > best_c6_dice
        is_new_best_small = c6_small > best_c6_dice_small
        best_star = "🌟 [NEW BEST OVERALL!]" if is_new_best else ""
        small_fire = "🔥 [NEW BEST SMALL!]" if is_new_best_small else ""

        print(f"\n┌{'─'*76}┐")
        print(f"│ 📊 ROUND [{round_idx:>2}/{args.num_rounds}] EVALUATION & METRICS SUMMARY{' '*30}│")
        print(f"├{'─'*76}┤")
        print(f"│  • In-Domain Training Centers (C1-C5 Mean):{' '*32}│")
        print(f"│      Dice (DSC): {in_domain_agg.get('dice', 0.0):.4f}   |  IoU: {in_domain_agg.get('iou', 0.0):.4f}{' '*35}│")
        print(f"│      Pixel Acc:  {in_domain_agg.get('pa', 0.0):.4f}   |  Recall: {in_domain_agg.get('recall', 0.0):.4f}{' '*32}│")
        print(f"│                                                                            │")
        print(f"│  • Strictly Held-Out Unseen Domain (Center 6 Test):{' '*23}│")
        print(f"│      Overall Dice (DSC):  {c6_dice:.4f}  {best_star:<28}│")
        print(f"│      Small Polyp DSC:     {c6_small:.4f}  {small_fire:<28}│")
        print(f"│      Large Polyp DSC:     {c6_large:.4f}{' '*45}│")
        print(f"│      IoU / mIoU:          {c6_iou:.4f}{' '*45}│")
        print(f"│      Recall / Sens.:      {c6_recall:.4f}{' '*45}│")
        print(f"│      Pixel Accuracy:      {c6_pa:.4f}{' '*45}│")
        print(f"│      HD95 Distance:       {c6_hd95:.2f} px{' '*42}│")
        if getattr(args, "enable_scale_decoder_moe", False):
            c6_gw_small = c6_test_m.get("gate_w_small", 0.5)
            c6_gw_large = c6_test_m.get("gate_w_large", 0.5)
            print(f"│      Scale Router w:      Small: {c6_gw_small:.3f} | Large: {c6_gw_large:.3f}{' '*30}│")
        print(f"└{'─'*76}┘\n")

        # Step E: Checkpoint Saving & Real-Time Tracking
        round_record = tracker.log_round(
            round_idx=round_idx,
            in_domain_metrics=in_domain_agg,
            c6_metrics=c6_test_m,
        )

        if args.save_checkpoints:
            last_ckpt = os.path.join(args.run_dir, "FedFGS_SAM_last.pth")
            torch.save({
                "round": round_idx,
                "lora_parameters": global_parameters,
                "in_domain_metrics": in_domain_agg,
                "c6_metrics": c6_test_m,
            }, last_ckpt)

            # Best overall unseen DSC
            if c6_dice > best_c6_dice:
                best_c6_dice = c6_dice
                best_ckpt = os.path.join(args.run_dir, "FedFGS_SAM_best.pth")
                torch.save({
                    "round": round_idx,
                    "lora_parameters": global_parameters,
                    "c6_metrics": c6_test_m,
                }, best_ckpt)
                print(f"🌟 Best Overall Model Updated! Unseen C6 DSC: {best_c6_dice:.4f} -> {best_ckpt}")

            # Best small polyp DSC
            if c6_small > best_c6_dice_small:
                best_c6_dice_small = c6_small
                best_small_ckpt = os.path.join(args.run_dir, "FedFGS_SAM_best_small.pth")
                torch.save({
                    "round": round_idx,
                    "lora_parameters": global_parameters,
                    "c6_metrics": c6_test_m,
                }, best_small_ckpt)
                print(f"🔥 Best Small Polyp Model Updated! DSC_S: {best_c6_dice_small:.4f} -> {best_small_ckpt}")

    tracker.save_summary(status="completed")
    print(f"\n{'='*75}")
    print(f"🎉 Fed-FGS simulation completed successfully!")
    print(f"   Run Name:                       {args.run_name}")
    print(f"   Best Unseen Center 6 DSC:       {tracker.best_c6_dsc:.4f} (Round {tracker.best_c6_dsc_round})")
    print(f"   Best Unseen Center 6 Small DSC: {tracker.best_c6_small_dsc:.4f} (Round {tracker.best_c6_small_round})")
    print(f"   Checkpoints & Config Saved In:  {args.run_dir}")
    print(f"   Metrics History CSV:            {tracker.csv_path}")
    print(f"   Run Summary JSON:               {tracker.summary_path}")
    print(f"{'='*75}\n")


def main():
    args = parse_arguments()
    run_fed_fgs_simulation(args)


if __name__ == "__main__":
    main()
