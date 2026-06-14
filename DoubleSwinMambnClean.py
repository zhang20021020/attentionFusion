import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from sympy import false
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from safetensors.torch import load_file
import cv2
from mamba_ssm import Mamba
from PyramidMamba import ManBaBlock
import copy

class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.ws - 1
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        pad_w = (0, (ps - W % ps) % ps)
        pad_h = (0, (ps - H % ps) % ps)
        # 4值padding：(left, right, top, bottom)
        x = F.pad(x, pad=(pad_w[0], pad_w[1], pad_h[0], pad_h[1]), mode='reflect')
        return x

    def pad_out(self, x):
        # 输出多加1列和1行，保证窗口计算对齐
        return F.pad(x, pad=(0, 1, 0, 1), mode='reflect')

    def forward(self, x):
        B, C, H, W = x.shape

        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]

        return out

# class GlobalLocalAttention(nn.Module):
#     def __init__(self,
#                  dim=256,
#                  num_heads=16,
#                  qkv_bias=False,
#                  window_size=8,
#                  relative_pos_embedding=True
#                  ):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // self.num_heads
#         self.scale = head_dim ** -0.5
#         self.ws = window_size
#
#         self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
#         self.local1 = ConvBN(dim, dim, kernel_size=3)
#         self.local2 = ConvBN(dim, dim, kernel_size=1)
#         self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)
#
#         self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
#         self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))
#
#         self.relative_pos_embedding = relative_pos_embedding
#
#         if self.relative_pos_embedding:
#             # define a parameter table of relative position bias
#             self.relative_position_bias_table = nn.Parameter(
#                 torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
#
#             # get pair-wise relative position index for each token inside the window
#             coords_h = torch.arange(self.ws)
#             coords_w = torch.arange(self.ws)
#             coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#             coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#             relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#             relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#             relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
#             relative_coords[:, :, 1] += self.ws - 1
#             relative_coords[:, :, 0] *= 2 * self.ws - 1
#             relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#             self.register_buffer("relative_position_index", relative_position_index)
#
#             trunc_normal_(self.relative_position_bias_table, std=.02)
#
#     def pad(self, x, ps):
#         _, _, H, W = x.size()
#         if W % ps != 0:
#             x = F.pad(x, (0, ps - W % ps), mode='reflect')
#         if H % ps != 0:
#             x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
#         return x
#
#     def pad_out(self, x):
#         x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
#         return x
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#
#         local = self.local2(x) + self.local1(x)
#
#         x = self.pad(x, self.ws)
#         B, C, Hp, Wp = x.shape
#         qkv = self.qkv(x)
#
#         q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
#                             d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws)
#
#         dots = (q @ k.transpose(-2, -1)) * self.scale
#
#         if self.relative_pos_embedding:
#             relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#                 self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
#             relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#             dots += relative_position_bias.unsqueeze(0)
#
#         attn = dots.softmax(dim=-1)
#         attn = attn @ v
#
#         attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
#                          d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)
#
#         attn = attn[:, :, :H, :W]
#
#         out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
#               self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))
#
#         out = out + local
#         out = self.pad_out(out)
#         out = self.proj(out)
#         # print(out.size())
#         out = out[:, :, :H, :W]
#
#         return out


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class WF_single(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF_single, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.post_conv(x)
        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class SEFusion(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SEFusion, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_depth = SqueezeAndExcitation(channels_in,
                                             activation=activation)

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = rgb + depth
        return out

class CBAMFusion(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 通道注意力（跟 SEFusion2 一样）
        self.channel_mlp = nn.Sequential(
            nn.Linear(2*channels, channels//reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels//reduction, channels, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力：1×1 卷积 + Sigmoid
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, rgb, dsm):
        B, C, H, W = rgb.shape
        # —— 通道注意力部分 ——
        rgb_pool = rgb.view(B, C, -1).mean(-1)
        dsm_pool = dsm.view(B, C, -1).mean(-1)
        ch_cat = torch.cat([rgb_pool, dsm_pool], dim=1)            # [B,2C]
        ch_gate = self.channel_mlp(ch_cat).view(B, C, 1, 1)        # [B,C,1,1]
        rgb_c = ch_gate * rgb;   dsm_c = (1-ch_gate) * dsm

        # —— 空间注意力部分 ——
        # 在通道维度上拼二路特征的“最大池”+“平均池”
        max_pool, _ = torch.max(torch.stack([rgb, dsm], dim=1), dim=1)  # [B,C,H,W]
        avg_pool = (rgb + dsm) * 0.5                                   # [B,C,H,W]
        pool_cat = torch.cat([max_pool.mean(1,True), avg_pool.mean(1,True)], dim=1)  # [B,2,H,W]
        sp_gate = self.spatial_conv(pool_cat)                         # [B,1,H,W]
        out = sp_gate * rgb_c + (1-sp_gate) * dsm_c
        return out

class DynamicConvFusion(nn.Module):
    def __init__(self, channels, kernel_size=3, reduction=16):
        super().__init__()
        hidden = channels//reduction
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1), nn.ReLU(),
            nn.Conv2d(hidden, channels*kernel_size*kernel_size, 1)
        )
        self.ks = kernel_size
        self.padding = kernel_size//2

    def forward(self, rgb, dsm):
        B, C, H, W = rgb.shape
        # 1) 用 DSM 生成动态卷积核
        kernels = self.kernel_gen(dsm)            # [B, C*K*K, 1,1]
        kernels = kernels.view(B*C, 1, self.ks, self.ks)
        # 2) 对 RGB 做分组卷积：每个通道一组
        rgb_flat = rgb.view(1, B*C, H, W)
        out = F.conv2d(rgb_flat, weight=kernels,
                       groups=B*C, padding=self.padding)
        out = out.view(B, C, H, W)
        # 3) 简单融合
        return 0.5*rgb + 0.5*out

class FeatureRefinementHead_single(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
            nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels // 16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels // 16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # weights = nn.ReLU()(self.weights)
        # fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        # x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
            nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels // 16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels // 16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


class Decoder_single(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder_single, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF_single(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF_single(encoder_channels[-3], decode_channels)

        self.p1 = FeatureRefinementHead_single(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res4, h, w):
        x = self.b4(self.pre_conv(res4))
        x = self.p3(x)
        x = self.b3(x)

        x = self.p2(x)
        x = self.b2(x)

        x = self.p1(x)

        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        x = self.b4(self.pre_conv(res4))
        x = self.p3(x, res3)
        x = self.b3(x)

        x = self.p2(x, res2)
        x = self.b2(x)

        x = self.p1(x, res1)

        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def draw_features(feature, savename=''):
    H = W = 256
    visualize = F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=False)
    visualize = visualize.detach().cpu().numpy()
    visualize = np.mean(visualize, axis=1).reshape(H, W)
    visualize = (((visualize - np.min(visualize)) / (np.max(visualize) - np.min(visualize))) * 255).astype(np.uint8)
    # fvis = np.fft.fft2(visualize)
    # fshift = np.fft.fftshift(fvis)
    # fshift = 20*np.log(np.abs(fshift))
    savedir = savename
    visualize = cv2.applyColorMap(visualize, cv2.COLORMAP_JET)
    cv2.imwrite(savedir, visualize)

class LightFusion(nn.Module):  # 替换原SEFusion
    def __init__(self, ch):
        super().__init__()
        # 保持原SE模块
        self.se_x = SqueezeAndExcitation(ch)
        self.se_y = SqueezeAndExcitation(ch)
        # 新增轻量交叉注意力
        self.cross_attn = nn.Sequential(
            nn.Conv2d(ch, ch // 16, 1),
            nn.ReLU(),
            nn.Conv2d(ch // 16, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x = self.se_x(x)
        y = self.se_y(y)
        # 增加交叉注意力权重
        xy_weight = self.cross_attn(x * y)  # 计算交互权重
        return x + y + xy_weight * (x + y)  # 加权增强


 # 可学习分辨率对齐层
class ResolutionAlign(nn.Module):
    def __init__(self, in_ch, scale_factor):
        super().__init__()
        self.scale = scale_factor
        self.conv  = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1)
    def forward(self, x):
        x = F.interpolate(x,
                         scale_factor=self.scale,
                         mode='bilinear',
                         align_corners=False)
        return self.conv(x)

class CrossAttentionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 将 rgb → q，dsm → k,v
        self.to_q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.scale = channels ** -0.5

    def forward(self, rgb, dsm):
        B, C, H, W = rgb.shape
        # [B, C, H*W] → [B, H*W, C]
        q = self.to_q(rgb).flatten(2).transpose(1, 2)
        k = self.to_k(dsm).flatten(2).transpose(1, 2)
        v = self.to_v(dsm).flatten(2).transpose(1, 2)

        # QK^T -> [B, HW, HW]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # attn @ V -> [B, HW, C] → [B, C, H, W]
        out = (attn @ v).transpose(1, 2).view(B, C, H, W)
        # 线性投射 + 残差回 RGB
        return rgb + self.proj(out)



class GeoMeanFusion(nn.Module):
    """
    对多个特征图按元素级几何平均融合
    """

    """
      对多个特征图按元素级几何平均融合，但在对数域累加，避免溢出/下溢。
      out = exp( ( sum_i log(feat_i + eps) ) / N )
      """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        # 对每一路特征先 clamp，避免 log(0)
        logs = [torch.log(f.clamp(min=self.eps)) for f in features]
        # 对数域平均
        log_mean = torch.stack(logs, dim=0).mean(dim=0)
        # 指数映射回原域
        return torch.exp(log_mean)


class TriLightFusion(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        # 对三路分别做 SE
        self.se1 = SqueezeAndExcitation(ch, reduction)
        self.se2 = SqueezeAndExcitation(ch, reduction)
        self.se3 = SqueezeAndExcitation(ch, reduction)

        # 用三路交叉特征生成 3 通道的注意力图
        # 输入通道 ch，输出 3（对应三路权重），空间大小跟输入一致
        self.gate = nn.Sequential(
            nn.Conv2d(ch, ch // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, 3, kernel_size=1, bias=False),
            nn.Softmax(dim=1)    # 在通道维度做归一化
        )

    def forward(self, x, y, z):
        # 1) SE 模块
        x1 = self.se1(x)
        y1 = self.se2(y)
        z1 = self.se3(z)

        # 2) 交叉特征（这里用逐元素相乘作为交互表示）
        cross = x1 * y1 * z1   # B×C×H×W

        # 3) 根据交叉特征生成三路权重图
        #    gate_map[:,0,:,:] 对应 x1 的权重，
        #    gate_map[:,1,:,:] 对应 y1 的权重，
        #    gate_map[:,2,:,:] 对应 z1 的权重
        gate_map = self.gate(cross)

        # 4) 加权融合
        w1 = gate_map[:, 0:1, :, :]
        w2 = gate_map[:, 1:2, :, :]
        w3 = gate_map[:, 2:3, :, :]

        out = w1 * x1 + w2 * y1 + w3 * z1
        return out

# ———— MambaLayer：只在最深分辨率上做全局序列建模 ————
class MambaLayer(nn.Module):
    def __init__(self,
                 in_chs: int = 256,
                 dim: int = 128,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 last_feat_size: int = 16):
        super().__init__()
        pool_scales = list(range(1, last_feat_size, last_feat_size // 4))
        self.pool_layers = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_chs, dim, 1, bias=False),
                nn.ReLU(inplace=True)
            ) for s in pool_scales
        ])
        self.mamba = Mamba(
            d_model=dim * len(pool_scales) + in_chs,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        # ✅ 新增：将输出通道映射回原 in_chs (256)
        self.proj = nn.Conv2d(dim * len(pool_scales) + in_chs, in_chs, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        ppm = [x]
        for p in self.pool_layers:
            y = p(x)
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
            ppm.append(y)
        cat = torch.cat(ppm, dim=1)
        seq = rearrange(cat, 'b c h w -> b (h w) c')
        out = self.mamba(seq)
        out = out.transpose(2, 1).view(B, cat.shape[1], H, W)
        out = self.proj(out)  # ✅ 通道数回到 256
        return out


class DecoderBlock(nn.Module):
    def __init__(self, ch, heads=8, window_size=8):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBNReLU(ch, ch)
        self.attn = GlobalLocalAttention(ch, num_heads=heads, window_size=window_size)
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = x + self.attn(x)
        return x

class SpatialGatedSkipFusion(nn.Module):
    """
    轻量门控 skip 融合：
    gate 越大越偏向 skip（浅层细节），越小越偏向 up（深层语义）
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 16)

        self.up_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, up_feat, skip_feat):
        up = self.up_proj(up_feat)
        skip = self.skip_proj(skip_feat)

        gate = self.gate(torch.cat([up, skip], dim=1))   # [B,1,H,W]
        fused = gate * skip + (1.0 - gate) * up

        residual = 0.5 * (up_feat + skip_feat)
        out = fused + self.res_scale * residual
        out = self.out(out)
        return out

class PMDecoder(nn.Module):
    """
    只保留两项有效改动：
    1) 增加中间尺度
    2) skip 连接改为门控融合
    """
    def __init__(self,
                 in_chs_low: int = 256,
                 in_chs_mid: int = 256,
                 in_chs_high: int = 256,
                 decoder_channels: int = 128,
                 num_classes: int = 6,
                 last_feat_size: int = 16):
        super().__init__()

        # 三层特征先统一到 decoder_channels
        self.low_proj = nn.Sequential(
            nn.Conv2d(in_chs_low, decoder_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU()
        )
        self.mid_proj = nn.Sequential(
            nn.Conv2d(in_chs_mid, decoder_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU()
        )
        self.high_proj = nn.Sequential(
            nn.Conv2d(in_chs_high, decoder_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU()
        )

        # 保留你当前残差版 ManBaBlock
        self.b3 = ManBaBlock(
            in_chs=decoder_channels,
            dim=decoder_channels,
            hidden_ch=decoder_channels * 4,
            out_ch=decoder_channels,
            last_feat_size=last_feat_size
        )

        # high -> mid
        self.up_high_to_mid = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU()
        )

        # mid -> low
        self.up_mid_to_low = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU()
        )

        # 用升级后的门控 skip 融合替代简单相加
        self.fuse_mid = SpatialGatedSkipFusion(decoder_channels)
        self.fuse_low = SpatialGatedSkipFusion(decoder_channels)

        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        )

        self.apply(self._init_weights)

    def forward(self, x_low, x_mid, x_high):
        x_low  = self.low_proj(x_low)
        x_mid  = self.mid_proj(x_mid)
        x_high = self.high_proj(x_high)

        # deepest feature
        x = self.b3(x_high)

        # high -> mid
        x = self.up_high_to_mid(x)
        if x.shape[-2:] != x_mid.shape[-2:]:
            x = F.interpolate(x, size=x_mid.shape[-2:], mode='bilinear', align_corners=False)
        x = self.fuse_mid(x, x_mid)

        # mid -> low
        x = self.up_mid_to_low(x)
        if x.shape[-2:] != x_low.shape[-2:]:
            x = F.interpolate(x, size=x_low.shape[-2:], mode='bilinear', align_corners=False)
        x = self.fuse_low(x, x_low)

        out = self.seg_head(x)
        return out

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class TwoBranchBackbone(nn.Module):
    """
    RGB / DSM 两路骨干，输出 three scales:
        low  : 浅层特征
        mid  : 中间层特征
        high : 深层特征

    这里只保留你原来的同尺度静态融合（CBAMFusion），
    不再引入编码端额外门控实验。
    """
    def __init__(self,
                 backbone_name='convnext_base',
                 pretrained=True,
                 out_indices=(0, 1, 3),
                 out_ch=256,
                 weight_path='/home/zhangben/mamba/attentionFusion/weights/convnext/convnext_base.pth'):
        super().__init__()

        self.rgb_backbone = timm.create_model(
            backbone_name,
            features_only=True,
            pretrained=pretrained,
            pretrained_cfg_overlay=dict(file=weight_path) if weight_path else None,
            out_indices=out_indices,
            in_chans=3
        )

        self.dsm_backbone = timm.create_model(
            backbone_name,
            features_only=True,
            pretrained=pretrained,
            pretrained_cfg_overlay=dict(file=weight_path) if weight_path else None,
            out_indices=out_indices,
            in_chans=1
        )

        rgb_c1, rgb_c2, rgb_c3 = self.rgb_backbone.feature_info.channels()
        dsm_c1, dsm_c2, dsm_c3 = self.dsm_backbone.feature_info.channels()

        # 统一到 out_ch
        self.to256_rgb_low  = nn.Conv2d(rgb_c1, out_ch, 1, bias=False) if rgb_c1 != out_ch else nn.Identity()
        self.to256_rgb_mid  = nn.Conv2d(rgb_c2, out_ch, 1, bias=False) if rgb_c2 != out_ch else nn.Identity()
        self.to256_rgb_high = nn.Conv2d(rgb_c3, out_ch, 1, bias=False) if rgb_c3 != out_ch else nn.Identity()

        self.to256_dsm_low  = nn.Conv2d(dsm_c1, out_ch, 1, bias=False) if dsm_c1 != out_ch else nn.Identity()
        self.to256_dsm_mid  = nn.Conv2d(dsm_c2, out_ch, 1, bias=False) if dsm_c2 != out_ch else nn.Identity()
        self.to256_dsm_high = nn.Conv2d(dsm_c3, out_ch, 1, bias=False) if dsm_c3 != out_ch else nn.Identity()

        # 继续保留你原来的融合策略，不再额外改编码端
        self.fuse = CBAMFusion(out_ch)

    def _adapt_img_size(self, backbone: nn.Module, x: torch.Tensor):
        m = getattr(backbone, 'model', backbone)
        if not hasattr(m, 'patch_embed'):
            return
        pe = m.patch_embed
        if not hasattr(pe, 'img_size'):
            return
        H, W = x.shape[-2:]
        if pe.img_size != (H, W):
            pe.img_size = (H, W)
            if hasattr(pe, 'grid_size'):
                ph, pw = pe.patch_size
                pe.grid_size = (H // ph, W // pw)

    def _to_nchw(self, feat: torch.Tensor):
        if feat.ndim == 4 and feat.shape[1] < feat.shape[-1]:
            feat = feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def forward(self, rgb: torch.Tensor, dsm: torch.Tensor):
        if dsm.ndim == 3:
            dsm = dsm.unsqueeze(1)
        elif dsm.ndim == 2:
            dsm = dsm.unsqueeze(0).unsqueeze(0)
        elif dsm.ndim == 5:
            dsm = dsm.squeeze(1)

        self._adapt_img_size(self.rgb_backbone, rgb)
        self._adapt_img_size(self.dsm_backbone, dsm)

        rgb_low, rgb_mid, rgb_high = self.rgb_backbone(rgb)
        dsm_low, dsm_mid, dsm_high = self.dsm_backbone(dsm)

        rgb_low, rgb_mid, rgb_high = map(self._to_nchw, (rgb_low, rgb_mid, rgb_high))
        dsm_low, dsm_mid, dsm_high = map(self._to_nchw, (dsm_low, dsm_mid, dsm_high))

        rgb_low  = self.to256_rgb_low(rgb_low)
        rgb_mid  = self.to256_rgb_mid(rgb_mid)
        rgb_high = self.to256_rgb_high(rgb_high)

        dsm_low  = self.to256_dsm_low(dsm_low)
        dsm_mid  = self.to256_dsm_mid(dsm_mid)
        dsm_high = self.to256_dsm_high(dsm_high)

        low  = self.fuse(rgb_low,  dsm_low)
        mid  = self.fuse(rgb_mid,  dsm_mid)
        high = self.fuse(rgb_high, dsm_high)

        return low, mid, high

class UNetFormer_TwoModal(nn.Module):
    """
    最终网络：
    保留残差版 ManBaBlock，
    只增加：
    1) 中间尺度
    2) skip 门控融合
    """
    def __init__(self,
                 num_classes: int = 6,
                 decode_channels: int = 128,
                 backbone_name: str = 'convnext_base',
                 last_feat_size: int = 16,
                 out_ch: int = 256):
        super().__init__()

        self.encoder = TwoBranchBackbone(
            backbone_name=backbone_name,
            pretrained=True,
            out_indices=(0, 1, 3),   # 三尺度
            out_ch=out_ch
        )

        self.decoder = PMDecoder(
            in_chs_low=out_ch,
            in_chs_mid=out_ch,
            in_chs_high=out_ch,
            decoder_channels=decode_channels,
            num_classes=num_classes,
            last_feat_size=last_feat_size
        )

    def forward(self, rgb: torch.Tensor, dsm: torch.Tensor, mode=None):
        low, mid, high = self.encoder(rgb, dsm)
        out = self.decoder(low, mid, high)
        return F.interpolate(out, size=rgb.shape[-2:], mode='bilinear', align_corners=False)