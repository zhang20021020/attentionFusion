import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F
from einops import rearrange
#
class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
# class Adapter(nn.Module):
#     """
#     Adapter module with pre-normalization and a learnable scaling factor.
#     Implements: x_out = x + s * Adapter(LN(x))
#     """
#     def __init__(
#         self,
#         D_features: int,
#         mlp_ratio: float = 0.25,
#         act_layer: nn.Module = nn.GELU,
#         skip_connect: bool = True,
#     ):
#         super().__init__()
#         self.skip_connect = skip_connect
#
#         # LayerNorm on last dimension
#         self.norm = nn.LayerNorm(D_features)
#
#         # Learnable scaling parameter, initialized to 0 so adapter starts as identity
#         self.scale = nn.Parameter(torch.zeros(1))
#
#         # Bottleneck hidden dimension
#         D_hidden = int(D_features * mlp_ratio)
#
#         # Down- and up-projection
#         self.fc1 = nn.Linear(D_features, D_hidden)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(D_hidden, D_features)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: Tensor of shape (..., D_features)
#         """
#         # Pre-normalize
#         x_norm = self.norm(x)
#
#         # Bottleneck MLP
#         h = self.fc1(x_norm)
#         h = self.act(h)
#         h = self.fc2(h)
#
#         # Apply scaling and residual
#         if self.skip_connect:
#             return x + self.scale * h
#         else:
#             return self.scale * h
# class Adapter(nn.Module):
#     def __init__(self, D_features, mlp_ratio=0.5, act_layer=nn.GELU, skip_connect=True):
#         super().__init__()
#         self.skip_connect = skip_connect
#         D_hidden_features = int(D_features * mlp_ratio)
#         self.act = act_layer()
#
#         # 下采样层
#         self.downsample1 = nn.Linear(D_features, D_hidden_features)
#         self.downsample2 = nn.Linear(D_hidden_features, D_hidden_features // 2)
#
#         # 上采样层
#         self.upsample1 = nn.Linear(D_hidden_features // 2, D_hidden_features)
#         self.upsample2 = nn.Linear(D_hidden_features, D_features)
#
#     def forward(self, x):
#         # 下采样
#         x = self.downsample1(x)
#         x = self.act(x)
#         x = self.downsample2(x)
#         x = self.act(x)
#
#         # 上采样
#         x = self.upsample1(x)
#         x = self.act(x)
#         x = self.upsample2(x)
#
#         # 跳跃连接
#         if self.skip_connect:
#             x = x + self.upsample2(x)
#         return x