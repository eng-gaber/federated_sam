# -*- coding: utf-8 -*-
"""
FedDG-DFTHT: Frequency-Based Federated Domain Generalization
using DFT Hard-Thresholding.

Adapted from the paper's source code (freq_space_interpolation_DFTHT.py
and threshold.py).  This module is designed to be used as a **data-level
augmentation** inside the existing SD-UTNet federated pipeline.

Key design decisions
--------------------
* ``extract_amp_spectrum`` is deliberately isolated as a public function so
  it can later be called inside the federated communication layer (e.g.
  each client extracts its amplitude spectra and shares them with others).
* All operations are pure PyTorch and work on GPU tensors if provided.
* The hard-threshold ratio ``alpha`` defaults to 5 % (0.05) as in the paper.
"""

import numpy as np
import torch
from typing import List, Optional, Union


# ── threshold helpers ───────────────────────────────────────────────────
def hard_threshold(x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Zero-out frequency components whose magnitude is below *T*."""
    return torch.where(torch.abs(x) >= T, x, torch.zeros_like(x))


# ── amplitude extraction (isolated for federated communication) ─────────
def extract_amp_spectrum(img: torch.Tensor) -> torch.Tensor:
    """
    Compute the amplitude spectrum of *img* via 2-D FFT.

    Parameters
    ----------
    img : torch.Tensor
        Image tensor of shape ``(C, H, W)`` **or** ``(B, C, H, W)``.

    Returns
    -------
    torch.Tensor
        Amplitude spectrum with the same shape as *img*.

    Notes
    -----
    This function is intentionally kept as a thin, standalone helper so
    that it can be reused in the federated communication layer (e.g.
    clients extract their amplitude spectra and broadcast them to peers).
    """
    return torch.abs(torch.fft.fft2(img))


# ── low-frequency amplitude swap with hard-thresholding ─────────────────
def amp_spectrum_swap(
    amp_local: torch.Tensor,
    amp_target: torch.Tensor,
    L: float = 0.1,
    ratio: float = 0.0,
    alpha: float = 0.05,
) -> torch.Tensor:
    """
    Replace the low-frequency band of *amp_local* with a hard-thresholded
    version of *amp_target*, controlled by ``ratio`` and ``alpha``.

    Parameters
    ----------
    amp_local  : amplitude spectrum of the local image.
    amp_target : amplitude spectrum of the target (other-domain) image.
    L          : fraction of the shortest spatial dimension that defines the
                 low-frequency window radius (default 0.1).
    ratio      : interpolation weight in [0, 1].  0 → full replacement,
                 1 → keep local.
    alpha      : hard-threshold ratio – the threshold equals
                 ``alpha * max(amp_target)`` per channel.  Default 5 %.
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
        + hard_threshold(a_trg[..., h1:h2, w1:w2], threshold) * (1 - ratio)
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
    Apply frequency-space style transfer with DFTHT to a single image
    (or a batch).

    Parameters
    ----------
    local_img  : source image tensor  ``(C,H,W)`` or ``(B,C,H,W)``.
    amp_target : amplitude spectrum of the target domain image
                 (same shape as *local_img*).
    L, ratio, alpha : see ``amp_spectrum_swap``.

    Returns
    -------
    torch.Tensor – the augmented image, same shape and dtype as *local_img*.
    """
    fft_local = torch.fft.fft2(local_img)
    amp_local = torch.abs(fft_local)
    pha_local = torch.angle(fft_local)

    amp_local_ = amp_spectrum_swap(amp_local, amp_target, L=L, ratio=ratio, alpha=alpha)

    fft_local_ = amp_local_ * torch.exp(1j * pha_local)
    local_in_trg = torch.fft.ifft2(fft_local_)
    local_in_trg = torch.real(local_in_trg)

    # Clamp to the range of the original image to prevent extreme values
    # that can cause NaN in downstream loss functions. Also clamp to a safe
    # normalized range [-3.0, 3.0] to prevent extreme pixel values.
    local_in_trg = torch.clamp(local_in_trg, local_img.min(), local_img.max())
    local_in_trg = torch.clamp(local_in_trg, -3.0, 3.0)

    return local_in_trg


# ── batch-level helper (used inside the DataLoader) ─────────────────────
def freq_space_interpolation_batch(
    local_img: torch.Tensor,
    target_amp_pool: List[torch.Tensor],
    L: float = 0.1,
    ratio: Union[float, List[float]] = (0.0, 1.0),
    alpha: float = 0.05,
) -> torch.Tensor:
    """
    For every image in the batch, randomly pick one amplitude spectrum
    from *target_amp_pool* (a list of ``(C, H, W)`` tensors) and apply
    the DFTHT augmentation.

    Parameters
    ----------
    local_img       : ``(B, C, H, W)`` batch of normalised images.
    target_amp_pool : list of amplitude spectra from other domains.
                      Each element is ``(C, H, W)``.
    L, ratio, alpha : forwarded to ``freq_space_interpolation``.

    Returns
    -------
    torch.Tensor – the augmented batch ``(B, C, H, W)``.
    """
    B = local_img.shape[0]
    n_targets = len(target_amp_pool)
    if n_targets == 0:
        return local_img  # nothing to augment with

    selected = []
    for _ in range(B):
        idx = np.random.randint(n_targets)
        selected.append(target_amp_pool[idx])
    amp_target = torch.stack(selected).to(local_img.device)

    # sample a random interpolation ratio if a range is given
    if isinstance(ratio, (list, tuple)):
        r = np.random.uniform(ratio[0], ratio[1])
    else:
        r = ratio

    return freq_space_interpolation(local_img, amp_target, L=L, ratio=r, alpha=alpha)
