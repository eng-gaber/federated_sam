# -*- coding: utf-8 -*-
"""
Composite loss function for Fed-FGS (Gaber et al., Eq. 5):
    L_Total = 0.7 * L_Dice + 0.2 * L_BCE + 0.1 * L_Focal
Adapted directly from fed-dftht-main/src/models/losses.py for SAM models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_fun(pred: torch.Tensor, target: torch.Tensor, smooth: float = 0.1) -> torch.Tensor:
    """Dice loss with smoothing."""
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1).float()
    intersection = (iflat * tflat).sum()
    loss = (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    return 1.0 - loss.mean()


def bce_loss_fun(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Numerically stable BCE loss directly from logits."""
    logits = pred_logits.contiguous().view(-1).float()
    t = target.contiguous().view(-1).float()
    return F.binary_cross_entropy_with_logits(logits, t)


def focal_loss_fun(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Numerically stable Focal loss directly from logits."""
    logits = pred_logits.contiguous().view(-1).float()
    t = target.contiguous().view(-1).float()
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    focal_weight = t * (alpha * (1.0 - p) ** gamma) + (1.0 - t) * ((1.0 - alpha) * p ** gamma)
    return (focal_weight * bce).mean()


class FedFGSCompositeLoss(nn.Module):
    """
    Paper Eq. (5) Composite Loss:
        L_Total = lambda_dice * L_Dice + lambda_bce * L_BCE + lambda_focal * L_Focal
    Defaults:
        lambda_dice = 0.7
        lambda_bce = 0.2
        lambda_focal = 0.1
    """
    def __init__(
        self,
        lambda_dice: float = 0.7,
        lambda_bce: float = 0.2,
        lambda_focal: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super(FedFGSCompositeLoss, self).__init__()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.lambda_focal = lambda_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, pred_logits: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """
        pred_logits: (B, C, H, W) raw logits from SAM
        target_masks: (B, 1, H, W) or (B, H, W) binary ground truth {0, 1}
        """
        # Ensure target matches spatial shape of prediction
        if target_masks.ndim == 3:
            target_masks = target_masks.unsqueeze(1)
        if target_masks.shape[-2:] != pred_logits.shape[-2:]:
            target_masks = F.interpolate(
                target_masks.float(),
                size=pred_logits.shape[-2:],
                mode="nearest"
            ).long()

        # Extract foreground logits & probabilities
        if pred_logits.shape[1] == 1:
            raw_fg_logits = pred_logits[:, 0, :, :]
        elif pred_logits.shape[1] == 2:
            raw_fg_logits = pred_logits[:, 1, :, :] - pred_logits[:, 0, :, :]
        else:
            raw_fg_logits = pred_logits[:, 0, :, :]

        pred_fg_prob = torch.sigmoid(raw_fg_logits)
        target_fg = (target_masks[:, 0, :, :] > 0).float()

        l_dice = dice_loss_fun(pred_fg_prob, target_fg)
        l_bce = bce_loss_fun(raw_fg_logits, target_fg)
        l_focal = focal_loss_fun(raw_fg_logits, target_fg, alpha=self.focal_alpha, gamma=self.focal_gamma)

        total_loss = self.lambda_dice * l_dice + self.lambda_bce * l_bce + self.lambda_focal * l_focal
        return total_loss
