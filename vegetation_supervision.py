"""Training-only discrimination of ISPRS low vegetation (2) and trees (3)."""

import torch
from torch import nn
from torch.nn import functional as F


class VegetationAuxHead(nn.Module):
    """Predict two raw logits from the geometry-corrected 1/8 DSM features."""

    def __init__(self, in_channels=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 2, 1),
        )

    def forward(self, features):
        return self.head(features)


def vegetation_cross_entropy(logits, target, ignore_index=6):
    """Average CE over vegetation pixels only; all other labels are ignored.

    The two outputs mean low vegetation (0) and trees (1), respectively.
    This loss uses training labels only; they are never inputs to the model.
    """
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("Vegetation logits must have shape [B,2,H,W].")
    if target.ndim != 3 or target.shape[0] != logits.shape[0]:
        raise ValueError("Vegetation targets must have shape [B,H,W] with matching batch size.")
    valid = ((target == 2) | (target == 3)) & (target != ignore_index)
    if not valid.any():
        # Preserve the graph so empty-vegetation batches also support backward.
        return logits.float().sum() * 0
    binary_target = torch.full_like(target, -100)
    binary_target[valid] = target[valid] - 2
    logits = F.interpolate(logits.float(), size=target.shape[-2:],
                           mode="bilinear", align_corners=False)
    return F.cross_entropy(logits, binary_target, ignore_index=-100)
