# -*- coding: utf-8 -*-
"""
FedDG-DFTHT: Frequency-Based Federated Domain Generalization using DFT Hard-Thresholding.
Adapted for SAM input pipeline on PyTorch 2.7+ / CUDA 12.8.
"""
import numpy as np
import torch
from typing import List, Optional, Union


def hard_threshold(x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Zero-out frequency components whose magnitude is below threshold T."""
    return torch.where(torch.abs(x) >= T, x, torch.zeros_like(x))


def extract_amp_spectrum(img: torch.Tensor) -> torch.Tensor:
    """
    Compute the amplitude spectrum of image tensor via 2-D FFT.
    img: (C, H, W) or (B, C, H, W)
    Returns: amplitude spectrum with same shape.
    """
    return torch.abs(torch.fft.fft2(img))


def amp_spectrum_swap(
    amp_local: torch.Tensor,
    amp_target: torch.Tensor,
    L: float = 0.1,
    ratio: float = 0.0,
    alpha: float = 0.05,
) -> torch.Tensor:
    """
    Replace the low-frequency band of amp_local with hard-thresholded amp_target.
    L: low-frequency window radius ratio (default 0.1 = 10%)
    ratio: interpolation weight in [0, 1]. 0 -> full replacement, 1 -> keep local
    alpha: hard-threshold ratio (default 0.05 = 5%)
    """
    a_local = torch.fft.fftshift(amp_local.clone())
    a_trg = torch.fft.fftshift(amp_target)

    h = a_local.shape[-2]
    w = a_local.shape[-1]
    b = int(np.floor(np.amin((h, w)) * L))
    c_h = int(np.floor(h / 2.0))
    c_w = int(np.floor(w / 2.0))

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    # Per-channel threshold: alpha * max of target amplitude
    threshold = torch.amax(a_trg, dim=[-1, -2], keepdim=True) * alpha

    a_local[..., h1:h2, w1:w2] = (
        a_local[..., h1:h2, w1:w2] * ratio
        + hard_threshold(a_trg[..., h1:h2, w1:w2], threshold) * (1.0 - ratio)
    )
    a_local = torch.fft.ifftshift(a_local)
    return a_local


def freq_space_interpolation(
    local_img: torch.Tensor,
    amp_target: torch.Tensor,
    L: float = 0.1,
    ratio: float = 0.0,
    alpha: float = 0.05,
) -> torch.Tensor:
    """
    Apply frequency-space style transfer with DFT-HT to an image tensor.
    local_img: (C, H, W) or (B, C, H, W)
    amp_target: amplitude spectrum of target domain (same spatial shape)
    """
    fft_local = torch.fft.fft2(local_img)
    amp_local = torch.abs(fft_local)
    pha_local = torch.angle(fft_local)

    amp_local_ = amp_spectrum_swap(amp_local, amp_target, L=L, ratio=ratio, alpha=alpha)

    fft_local_ = amp_local_ * torch.exp(1j * pha_local)
    local_in_trg = torch.fft.ifft2(fft_local_)
    local_in_trg = torch.real(local_in_trg)

    # Clamp to original range to prevent extreme values and numerical instability
    local_in_trg = torch.clamp(local_in_trg, local_img.min(), local_img.max())
    local_in_trg = torch.clamp(local_in_trg, -3.5, 3.5)

    return local_in_trg
