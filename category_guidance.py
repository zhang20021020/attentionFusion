"""Supervised category guidance for optical/DSM fusion and PAN."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def normalize_dsm(dsm):
    """Normalize a full tile before cropping; constant/invalid tiles stay finite."""
    dsm = np.asarray(dsm, dtype=np.float32)
    finite = np.isfinite(dsm)
    if not finite.any():
        return np.zeros_like(dsm)
    low, high = dsm[finite].min(), dsm[finite].max()
    clean = np.where(finite, dsm, low)
    return (clean - low) / max(float(high - low), 1e-6)


class DSMGeometryInput(nn.Module):
    """[height, signed horizontal gradient, signed vertical gradient]."""

    def __init__(self, enabled=True):
        super().__init__()
        self.enabled = enabled
        kernel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
        self.register_buffer("sobel", torch.stack([kernel, kernel.t()])[:, None],
                             persistent=False)

    def forward(self, dsm):
        if dsm.ndim == 3:
            dsm = dsm.unsqueeze(1)
        if dsm.ndim != 4 or dsm.shape[1] != 1:
            raise ValueError("DSM must have shape [B,H,W] or [B,1,H,W].")
        if not self.enabled:
            return dsm.repeat(1, 3, 1, 1)
        # Replicate padding avoids artificial height cliffs at patch borders.
        gradients = F.conv2d(F.pad(dsm, (1, 1, 1, 1), mode="replicate"),
                             self.sobel.to(dtype=dsm.dtype))
        return torch.cat([dsm, gradients], dim=1)


def category_confidence(probabilities):
    """Normalized entropy confidence multiplied by the top-two margin."""
    if probabilities.shape[1] < 2:
        raise ValueError("Category confidence requires at least two classes.")
    p = probabilities.float()
    entropy = -(p * p.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
    entropy = entropy / math.log(p.shape[1])
    top2 = p.topk(2, dim=1).values
    return (1 - entropy).clamp(0, 1) * (top2[:, :1] - top2[:, 1:2])


class CategoryPrediction(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        hidden = max(channels // 4, 16)
        self.head = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_classes, 1),
        )

    def forward(self, features):
        return self.head(features)


def calibrate_categories(rgb_logits, dsm_logits):
    # Train confidence through supervised logits, not by rewarding overconfidence
    # in the final segmentation loss. The auxiliary loss still trains both heads.
    rgb = rgb_logits.detach().float().softmax(dim=1)
    dsm = dsm_logits.detach().float().softmax(dim=1)
    rgb_conf, dsm_conf = category_confidence(rgb), category_confidence(dsm)
    rgb_support = rgb_conf * rgb
    dsm_support = dsm_conf * dsm
    # Retain an equal-modality fallback when both predictions are uncertain.
    rgb_share = (rgb_support + 1e-6) / (rgb_support + dsm_support + 2e-6)
    categories = ((rgb_conf + 1e-6) * rgb + (dsm_conf + 1e-6) * dsm)
    categories = categories / (rgb_conf + dsm_conf + 2e-6)
    return categories, rgb_share


class ConfidenceFeatureFusion(nn.Module):
    """Category-specific modality selection, added residually to SE fusion."""

    def __init__(self, channels, num_classes):
        super().__init__()
        self.channel_categories = nn.Parameter(torch.zeros(channels, num_classes))
        self.strength = nn.Parameter(torch.tensor(-2.0))

    def forward(self, rgb, dsm, rgb_share):
        associations = self.channel_categories.softmax(dim=1)
        weights = torch.einsum("ck,bkhw->bchw", associations, rgb_share)
        weights = weights.to(dtype=rgb.dtype)
        baseline = rgb + dsm
        selected = 2 * (weights * rgb + (1 - weights) * dsm)
        return baseline + self.strength.sigmoid() * (selected - baseline)


class CategoryGuidedPAN(nn.Module):
    """Top-down category propagation and semantic gating of bottom-up PAN."""

    def __init__(self, channels, num_classes, levels=4):
        super().__init__()
        self.mix_logits = nn.Parameter(torch.zeros(levels - 1))
        self.gates = nn.ModuleList([
            nn.Conv2d(num_classes, channels, 1) for _ in range(levels)
        ])
        self.strengths = nn.Parameter(torch.full((levels,), -2.0))

    def forward(self, categories):
        if len(categories) != len(self.gates):
            raise ValueError("PAN needs one category distribution per scale.")
        guidance = list(categories)
        for i in range(len(guidance) - 2, -1, -1):
            deeper = F.interpolate(guidance[i + 1], size=guidance[i].shape[-2:],
                                   mode="bilinear", align_corners=False)
            local_weight = self.mix_logits[i].sigmoid()
            guidance[i] = local_weight * guidance[i] + (1 - local_weight) * deeper
        return guidance

    def gate(self, features, categories, level):
        confidence = category_confidence(categories).to(dtype=features.dtype)
        modulation = self.gates[level](categories.to(dtype=features.dtype)).tanh()
        return features * (1 + self.strengths[level].sigmoid() * confidence * modulation)
