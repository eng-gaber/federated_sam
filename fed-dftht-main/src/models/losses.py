import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from monai.losses import DiceLoss, FocalLoss

def KL_divergence(logvar, mu):
    logvar = torch.clamp(logvar, -10.0, 10.0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kld.mean()

def dice_loss_fun(pred, target):
    smooth = 0.1

    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    loss = ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)).mean()

    return 1 - loss

def bce_loss_fun(pred, target):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    eps = 1e-15
    pred = torch.clamp(pred, eps, 1.0 - eps)
    bce = - (target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
    return bce.mean()


def focal_loss_fun(pred, target, alpha=0.25, gamma=2.0):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    eps = 1e-15
    pred = torch.clamp(pred, eps, 1.0 - eps)
    focal_weight = target * (alpha * (1.0 - pred) ** gamma) + (1.0 - target) * ((1.0 - alpha) * pred ** gamma)
    bce = - (target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
    focal_loss = focal_weight * bce
    return focal_loss.mean()


def compute_segmentation_loss(loss_module, seg_pred, y_true):
    """Compute segmentation loss based on loss_type config.
    
    Returns:
        tuple: (loss_value, is_composite) where is_composite indicates whether
               the composite loss was used (so callers can skip their own dice_w
               multiplier, since composite already includes balanced weights).
    """
    pred_fg = seg_pred[:, 1, :, :]
    
    loss_type = "default"
    lambda_dice = 1.0
    lambda_bce = 1.0
    lambda_focal = 1.0
    
    config = getattr(loss_module, 'config', None)
    if config is not None:
        client_cfg = getattr(config, 'client', None)
        if client_cfg is not None:
            loss_type = getattr(client_cfg, 'loss_type', 'default')
            lambda_dice = getattr(client_cfg, 'lambda_dice', 1.0)
            lambda_bce = getattr(client_cfg, 'lambda_bce', 1.0)
            lambda_focal = getattr(client_cfg, 'lambda_focal', 1.0)
            
    if loss_type == "composite":
        dice = dice_loss_fun(pred_fg, y_true)
        bce = bce_loss_fun(pred_fg, y_true)
        focal = focal_loss_fun(pred_fg, y_true)
        return lambda_dice * dice + lambda_bce * bce + lambda_focal * focal, True
    else:
        return dice_loss_fun(pred_fg, y_true), False


class UNetLoss(nn.Module):
    def __init__(self, train, dice_w=1.0, bce_w=0.0):
        super().__init__()
        self.train_mode = train

        self.dice_w = dice_w
        self.bce_w = bce_w

    def forward(self, y_true, y_pred):
        seg_pred = F.softmax(y_pred, dim=1)
        d_loss, is_composite = compute_segmentation_loss(self, seg_pred, y_true)

        if self.train_mode:
            loss = d_loss

            return {'train_loss': loss, 'dice_loss': d_loss}
        else:
            return {'dice_loss': d_loss}


class SDNetLoss(nn.Module):
    def __init__(self, train, k1_w=0.01, regress_w=1.0, focal_w=0.0, dice_w=10.0, reco_w=1.0, aug_w=1.0):
        super().__init__()
        self.train_mode = train

        self.k1_loss = KL_divergence
        self.regress_loss = nn.L1Loss()
        self.reco_loss = nn.L1Loss()

        self.k1_w = k1_w
        self.regress_w = regress_w
        self.focal_w = focal_w
        self.dice_w = dice_w
        self.reco_w = reco_w

    def forward(self, y_true, seg_pred, reco, z_out, z_out_tilde, mu, logvar, images):
        d_loss, is_composite = compute_segmentation_loss(self, seg_pred, y_true)

        if self.train_mode:
            k_loss = self.k1_loss(logvar, mu)
            r1_loss = self.regress_loss(z_out_tilde, z_out)
            r2_loss = self.reco_loss(reco, images)

            loss = d_loss * self.dice_w + \
                k_loss * self.k1_w + \
                r1_loss * self.regress_w + \
                r2_loss * self.reco_w 

            return {'train_loss': loss, 'dice_loss': d_loss, 'k1_loss': k_loss, 'regress_loss': r1_loss, 'reco_loss': r2_loss}
        else:
            return {'dice_loss': d_loss}
        
    def forward_aug(self, y_true, seg_pred, reco, z_out, mu_tilde, mu, logvar, aug_loss, images):
        d_loss, is_composite = compute_segmentation_loss(self, seg_pred, y_true)

        if self.train_mode:
            k_loss = self.k1_loss(logvar, mu)
            r1_loss = self.regress_loss(mu_tilde, z_out)
            r2_loss = self.reco_loss(reco, images)

            loss = d_loss * self.dice_w + \
                k_loss * self.k1_w + \
                r1_loss * self.regress_w + \
                r2_loss * self.reco_w 

            return {'train_loss': loss, 'dice_loss': d_loss, 'k1_loss': k_loss, 'regress_loss': r1_loss, 'reco_loss': r2_loss, 'aug_loss': aug_loss}
        else:
            return {'dice_loss': d_loss}

class FedDisLoss(nn.Module):
    def __init__(self, train, alpha=0.2, beta=0.5, dice_w=5.0):
        super().__init__()
        self.train_mode = train

        self.reco_loss = nn.L1Loss()
        self.lol_loss = nn.CosineSimilarity()

        self.alpha = alpha
        self.beta = beta
        self.dice_w = dice_w

    def forward(self, y_true, seg, x_, z_s, z_a, z_s_shift, z_s_proj, images):
        d_loss, is_composite = compute_segmentation_loss(self, seg, y_true)

        if self.train_mode:
            r_loss = self.reco_loss(x_, images)
            l_loss = self.lol_loss(z_a.flatten(start_dim=1), z_s_proj.flatten(start_dim=1))
            l_loss = torch.mean(l_loss.flatten())

            loss =  r_loss * self.alpha + \
                    l_loss * (1-self.alpha) + \
                    d_loss * self.dice_w
            
            return {'train_loss': loss, 'dice_loss': d_loss, 'reco_loss': r_loss, 'lol_loss': l_loss}
        else:
            return {'dice_loss': d_loss}