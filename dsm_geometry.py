"""Five-channel DSM geometry and optical-guided residual correction.

This is separate from the legacy [D, Sobel-x, Sobel-y] input adapter.
Inputs are the finite, tile-normalized DSM crops produced by the data loader.
Relative height here is a local height difference, not nDSM or metric tree height.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DSMGeometryDescriptors(nn.Module):
    """Return [D, H_small, H_large, S_small, S_large] after augmentation."""

    def __init__(self, windows=(9, 25), eps=1e-6):
        super().__init__()
        windows = tuple(windows)
        if (len(windows) != 2 or any(type(k) is not int or k < 3 or k % 2 == 0
                                     for k in windows) or windows[0] >= windows[1]):
            raise ValueError("Geometry windows must be two increasing odd integers >= 3.")
        if eps <= 0:
            raise ValueError("Geometry epsilon must be positive.")
        self.windows = windows
        self.eps = eps

    def forward(self, dsm):
        if dsm.ndim == 3:
            dsm = dsm.unsqueeze(1)
        if dsm.ndim != 4 or dsm.shape[1] != 1:
            raise ValueError("DSM must have shape [B,H,W] or [B,1,H,W].")
        # Local variance is sensitive to cancellation in fp16/bfloat16.
        with torch.autocast(device_type=dsm.device.type, enabled=False):
            height = dsm.float()
            moments = torch.cat([height, height.square()], dim=1)
            relative, roughness = [], []
            for window in self.windows:
                radius = window // 2
                pooled = F.avg_pool2d(
                    F.pad(moments, (radius,) * 4, mode="replicate"), window, stride=1)
                mean, second = pooled[:, :1], pooled[:, 1:]
                relative.append(height - mean)
                variance = (second - mean.square()).clamp_min(0)
                roughness.append((variance + self.eps).sqrt())
            return torch.cat([height, *relative, *roughness], dim=1)


def _separable_down(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1,
                  groups=in_channels, bias=False),
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.GELU(),
    )


class DSMGeometryEncoder(nn.Module):
    """Learn geometry directly from DSM; emit 1/8 and 1/16 features."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(5, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(),
        )
        self.down4 = _separable_down(32, 48)
        self.down8 = _separable_down(48, 64)
        self.down16 = _separable_down(64, 96)

    def forward(self, descriptors):
        features = self.down4(self.stem(descriptors))
        at8 = self.down8(features)
        return at8, self.down16(at8)


class OpticalGuidedGeometryResidual(nn.Module):
    """Correct DSM features with a spatial gate; identity at initialization."""

    def __init__(self, geometry_channels, feature_channels=256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(feature_channels + geometry_channels, 32, 1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid(),
        )
        self.projection = nn.Conv2d(geometry_channels, feature_channels, 1, bias=False)
        self.strength = nn.Parameter(torch.tensor(-2.0))
        # Only the output projection is zero: it receives gradients immediately.
        # Encoder/gate gradients start after the first optimizer update.
        nn.init.zeros_(self.projection.weight)

    def forward(self, optical, dsm_features, geometry):
        if optical.shape != dsm_features.shape:
            raise ValueError("Optical and DSM feature shapes must match for geometry correction.")
        if geometry.shape[-2:] != dsm_features.shape[-2:]:
            geometry = F.interpolate(geometry, size=dsm_features.shape[-2:],
                                     mode="bilinear", align_corners=False)
        geometry = geometry.to(dtype=dsm_features.dtype)
        gate = self.gate(torch.cat([optical, geometry], dim=1))
        residual = self.projection(geometry)
        return dsm_features + self.strength.sigmoid() * gate * residual


class DSMGeometryBranch(nn.Module):
    """Correct only levels 1 and 2 (res2y/res3y); leave res1y/res4y intact."""

    def __init__(self, feature_channels=256, windows=(9, 25)):
        super().__init__()
        self.descriptors = DSMGeometryDescriptors(windows)
        self.encoder = DSMGeometryEncoder()
        self.injections = nn.ModuleList([
            OpticalGuidedGeometryResidual(64, feature_channels),
            OpticalGuidedGeometryResidual(96, feature_channels),
        ])

    def forward(self, dsm, optical_features, dsm_features):
        if len(optical_features) != 4 or len(dsm_features) != 4:
            raise ValueError("Geometry correction requires four optical and DSM feature scales.")
        geometry_features = self.encoder(self.descriptors(dsm))
        corrected = list(dsm_features)
        for level, injection, geometry in zip((1, 2), self.injections, geometry_features):
            corrected[level] = injection(optical_features[level], dsm_features[level], geometry)
        return corrected
