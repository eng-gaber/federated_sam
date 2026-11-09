# -*- coding: utf-8 -*-
"""
Scale-Aware Dual-Decoder Mixture of Experts (MoE) for SAM.

This module introduces two specialized MaskDecoders:
  1. D_small: Specialized expert for diminutive / small polyps (fine edges, low contrast).
  2. D_large: Specialized expert for medium / large polyps (global region coverage, consistency).

An Image-Level Scale Gating Network takes the global average pooled token representation
of the image embedding F, passing it through a 2-layer MLP to predict dynamic routing weights:
    [w_small, w_large] = Softmax(MLP(GAP(F)))

Final predicted mask logits are a weighted mixture:
    Z_mix = w_small * Z_small + w_large * Z_large

During training, multi-task scale supervision is applied:
    L_total = L_mix + beta * L_expert + lambda_gate * L_gate
where:
  - L_mix is the end-to-end composite segmentation loss on Z_mix.
  - L_expert enforces scale specialization using the ground-truth scale label s_i:
        s_i * L_comp(Z_small) + (1 - s_i) * L_comp(Z_large)
    Notice that the non-selected expert is NOT penalized by L_expert, though it still
    receives smoothed gradient updates through L_mix (scaled by its routing weight).
  - L_gate is the binary cross-entropy supervising the scale router with ground-truth scale s_i.
"""

from typing import Dict, Optional, Tuple, Union
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleGatingNetwork(nn.Module):
    """
    Image-Level Scale Gating Network.
    
    Takes image embedding F of shape (B, C, H', W') where C=256, H'=W'=64 for SAM-Base.
    Performs Global Average Pooling (GAP) -> LayerNorm -> Linear -> GELU -> Linear -> Softmax.
    Outputs: [w_small, w_large] of shape (B, 2), normalized to sum to 1.
    """
    def __init__(self, in_features: int = 256, hidden_dim: int = 64):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        
        self.norm = nn.LayerNorm(in_features)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize hidden layer with Kaiming normal
        nn.init.kaiming_normal_(self.mlp[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.mlp[0].bias)
        # Initialize final classification layer to zero weights & bias
        # This ensures exactly [0.5, 0.5] balanced initial routing at step 0
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, image_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_embeddings: (B, C, H', W') image feature map from SAM image encoder.
        Returns:
            weights: (B, 2) softmax routing weights [w_small, w_large].
            logits:  (B, 2) unnormalized gating logits.
        """
        # Global Average Pooling over spatial dimensions
        f_bar = image_embeddings.mean(dim=[-2, -1])  # (B, 256)
        normed_f = self.norm(f_bar)
        logits = self.mlp(normed_f)  # (B, 2)
        weights = F.softmax(logits, dim=-1)
        return weights, logits


class ScaleAwareDualMaskDecoder(nn.Module):
    """
    Dual-Decoder Mask Prediction Head with Scale-Aware Mixture of Experts.
    
    Wraps two cloned instances of SAM's MaskDecoder (D_small and D_large) along with
    the ScaleGatingNetwork. Seamlessly drop-in compatible with SAM's standard mask decoder interface.
    """
    def __init__(
        self,
        base_decoder: nn.Module,
        gate_hidden_dim: int = 64,
        beta: float = 0.5,
        lambda_gate: float = 0.1,
    ):
        super().__init__()
        # Deepcopy creates two independent, identical decoder heads from the base SAM decoder
        self.decoder_small = copy.deepcopy(base_decoder)
        self.decoder_large = copy.deepcopy(base_decoder)
        
        # Determine embedding dimension (typically 256 in SAM-Base)
        transformer_dim = getattr(base_decoder, "transformer_dim", 256)
        self.gate = ScaleGatingNetwork(in_features=transformer_dim, hidden_dim=gate_hidden_dim)
        
        self.beta = beta
        self.lambda_gate = lambda_gate
        
        # Cache for latest forward pass results (accessible during loss computation)
        self.last_aux: Optional[Dict[str, torch.Tensor]] = None

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = False,
        return_dict: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass predicting masks using both experts and image-level scale gating.
        
        Args:
            image_embeddings: (B, 256, 64, 64)
            image_pe: (1, 256, 64, 64) positional encoding
            sparse_prompt_embeddings: (B, N, 256)
            dense_prompt_embeddings: (B, 256, 64, 64)
            multimask_output: whether to output 3 masks or 1 mask (False for polyp segmentation)
            return_dict: if True, returns detailed auxiliary dictionary
        Returns:
            Standard mode: (z_mix, iou_mix) matching SAM's MaskDecoder return signature
            Dict mode: dictionary with z_mix, z_small, z_large, weights, iou_mix
        """
        # 1. Compute gating weights [w_small, w_large]
        weights, gate_logits = self.gate(image_embeddings)  # (B, 2)
        w_small = weights[:, 0].view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        w_large = weights[:, 1].view(-1, 1, 1, 1)

        # 2. Forward through small expert decoder
        z_small, iou_small = self.decoder_small(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
        )

        # 3. Forward through large expert decoder
        z_large, iou_large = self.decoder_large(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
        )

        # 4. Mixture of predicted logits and IoU scores
        z_mix = w_small * z_small + w_large * z_large
        w_small_iou = weights[:, 0].unsqueeze(1)
        w_large_iou = weights[:, 1].unsqueeze(1)
        iou_mix = w_small_iou * iou_small + w_large_iou * iou_large

        # Cache forward outputs for client-side loss computation
        self.last_aux = {
            "z_mix": z_mix,
            "z_small": z_small,
            "z_large": z_large,
            "weights": weights,         # (B, 2)
            "gate_logits": gate_logits, # (B, 2)
            "iou_mix": iou_mix,
        }

        if return_dict:
            return self.last_aux

        # Preserves identical tuple unpacking signature: pred, _ = model.mask_decoder(...)
        return z_mix, iou_mix

    def compute_scale_aware_loss(
        self,
        criterion: nn.Module,
        target_masks: torch.Tensor,
        is_smalls: torch.Tensor,
        aux_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task Scale-Aware Dual-Decoder Loss:
            L_total = L_mix + beta * L_expert + lambda_gate * L_gate

        Args:
            criterion: FedFGSCompositeLoss module (0.7 Dice + 0.2 BCE + 0.1 Focal)
            target_masks: Ground truth masks (B, 1, H, W)
            is_smalls: Ground truth scale indicator (B,) where 1 = small polyp, 0 = large polyp
            aux_dict: Optional cached dictionary from forward pass (uses self.last_aux if None)
        Returns:
            loss_total: scalar loss tensor for backpropagation
            loss_stats: dictionary of individual loss components for logging
        """
        if aux_dict is None:
            aux_dict = self.last_aux
        if aux_dict is None:
            raise RuntimeError("No forward output cached. Run forward() before compute_scale_aware_loss().")

        z_mix = aux_dict["z_mix"]
        z_small = aux_dict["z_small"]
        z_large = aux_dict["z_large"]
        weights = aux_dict["weights"]  # (B, 2), col 0: w_small, col 1: w_large
        B = z_mix.shape[0]

        # 1. Ensemble Mixture Loss L_mix
        loss_mix = criterion(z_mix, target_masks)

        # 2. Scale-Specialization Auxiliary Loss L_expert
        # L_expert = (1/B) * sum_i [ s_i * L_comp(Z_small, Y) + (1 - s_i) * L_comp(Z_large, Y) ]
        # NOTE: Non-selected experts are not penalized by L_expert, eliminating gradient tug-of-war.
        # They still receive subtle, scaled gradients through L_mix proportional to their gate weight.
        expert_losses = []
        is_smalls_flat = is_smalls.view(B).long()
        for i in range(B):
            s_i = is_smalls_flat[i].item()
            pred_i = z_small[i:i+1] if s_i == 1 else z_large[i:i+1]
            loss_i = criterion(pred_i, target_masks[i:i+1])
            expert_losses.append(loss_i)
        loss_expert = torch.stack(expert_losses).mean() if len(expert_losses) > 0 else torch.tensor(0.0, device=z_mix.device)

        # 3. Gating Supervision Loss L_gate
        # Binary cross entropy between gate prediction w_small and ground-truth s_i
        # s_i = 1 -> target w_small = 1, target w_large = 0
        eps = 1e-7
        w_small_token = weights[:, 0].clamp(min=eps, max=1.0 - eps)
        w_large_token = weights[:, 1].clamp(min=eps, max=1.0 - eps)
        s_float = is_smalls.view(B).float().to(device=weights.device)

        loss_gate = - (s_float * torch.log(w_small_token) + (1.0 - s_float) * torch.log(w_large_token)).mean()

        # 4. Total Multi-Task Combination
        loss_total = loss_mix + self.beta * loss_expert + self.lambda_gate * loss_gate

        loss_stats = {
            "loss_total": float(loss_total.detach().item()),
            "loss_mix": float(loss_mix.detach().item()),
            "loss_expert": float(loss_expert.detach().item()),
            "loss_gate": float(loss_gate.detach().item()),
            "mean_w_small": float(weights[:, 0].detach().mean().item()),
            "mean_w_large": float(weights[:, 1].detach().mean().item()),
        }

        return loss_total, loss_stats
