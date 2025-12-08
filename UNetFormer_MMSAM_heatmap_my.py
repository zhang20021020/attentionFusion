
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
import cv2
import torch.autograd as autograd
from MedSAM.models.sam import sam_model_registry
import MedSAM.cfg as cfg
import matplotlib.pyplot as plt


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
            nn.ReLU()
        )
class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU6, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6(inplace=True)
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
            nn.ReLU(inplace=True)
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

# 引入的 Decoder 部分
class MMSConv(nn.Module):
    def __init__(self, in_channels=128, out_channels=48, eps=1e-8, ca_num_heads=6, norm_layer=nn.BatchNorm2d):
        super(MMSConv, self).__init__()
        self.out_channels = out_channels
        self.ca_num_heads = ca_num_heads
        self.act = nn.ReLU(inplace=True)
        self.convcat = ConvBN(in_channels * 3, out_channels, kernel_size=1, stride=1)
        self.conv11 = ConvBNReLU6(in_channels, out_channels, kernel_size=3, stride=1)
        self.conv12 = ConvBNReLU6(in_channels, out_channels, kernel_size=3, stride=1)
        self.conv13 = ConvBNReLU6(in_channels, out_channels, kernel_size=3, stride=1)

        self.conv22 = ConvBNReLU6(in_channels, out_channels, kernel_size=3, stride=1)
        self.conv23 = ConvBNReLU6(in_channels, out_channels, kernel_size=3, stride=1)

        self.conv33 = ConvBNReLU6(in_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, X):
        shortcut = X
        x11 = self.conv11(X)
        x12 = self.conv12(X)
        x13 = self.conv13(X)
        x112 = x11 + x13 + x12

        x22 = self.conv22(x112)
        x23 = self.conv23(x112)
        x223 = x22 + x23

        x33 = self.conv33(x223)
        x_out = torch.cat([x112, x223, x33], 1)
        x_out = self.convcat(x_out)
        x_out = x_out + shortcut
        x_out = self.act(x_out)

        return x_out


class preAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 norm_layer=nn.BatchNorm2d,
                 relative_pos_embedding=True,
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size
        self.qk = ConvBNReLU(dim, 2 * dim, kernel_size=1, bias=qkv_bias)
        self.avgpool = nn.AvgPool2d((1, self.num_heads))
        self.avgpool_arrange = nn.AvgPool2d((window_size * window_size, 1))
        # self.sigmo = nn.Sigmoid()

        self.relative_pos_embedding = relative_pos_embedding
        self.act = nn.ReLU()

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        x = self.pad(x, self.ws)

        B, C, Hp, Wp = x.shape
        qk = self.qk(x)
        num = Hp // self.ws
        q, k = rearrange(qk, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                         d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=2, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

            dots += relative_position_bias.unsqueeze(0)
        avgdots_normal = dots.permute(0, 2, 3, 1).contiguous()
        avgdots_normal = self.avgpool(avgdots_normal)  # b*64*64*1
        avgdots_normal = torch.squeeze(avgdots_normal, dim=-1).contiguous()
        avgdots_normal = self.avgpool_arrange(avgdots_normal)  # avgpool_arrange   b*64*1
        avgdots_normal = avgdots_normal.permute(0, 2, 1).contiguous()
        attn = dots.softmax(dim=-1)
        return attn, avgdots_normal


class ClassAttention(nn.Module):
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
        self.ws = window_size
        self.wv = ConvBNReLU(dim, dim, kernel_size=1, bias=qkv_bias)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x, attn_i):
        B, C, H, W = x.shape
        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        v = self.wv(x)
        v = rearrange(v, 'b (h d) (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                      d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)
        attn = attn_i @ v
        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]
        out = self.pad_out(attn)
        out = out[:, :, :H, :W]

        return out

class Block(nn.Module):
    def __init__(self, dim=256, num_classes=6, device=0, mlp_ratio=4., qkv_bias=False, drop=0.,
                 norm_layer=nn.BatchNorm2d, window_size=8, ):
        super().__init__()
        self.device = device
        self.ca_num_heads = num_classes
        self.heads = 4
        self.ca_num_headsdim = dim // num_classes
        mlp_hidden_dim = int(self.ca_num_headsdim * mlp_ratio)
        self.MAX = nn.MaxPool2d((1, self.ca_num_heads))
        self.act = nn.ReLU()
        for i in range(self.ca_num_heads):
            preattn = preAttention(dim=self.ca_num_headsdim, num_heads=self.heads, window_size=window_size,
                                   qkv_bias=qkv_bias
                                   )
            setattr(self, f"pre_attn{i + 1}", preattn)

            Cattn = ClassAttention(dim=self.ca_num_headsdim, num_heads=self.heads, window_size=window_size,
                                   qkv_bias=qkv_bias
                                   )
            setattr(self, f"C_attn{i + 1}", Cattn)
            mlp = MMSConv(self.ca_num_headsdim, self.ca_num_headsdim)
            setattr(self, f"C_mlp{i + 1}", mlp)

    def forward(self, x):
        for i in range(self.ca_num_heads):
            preattn = getattr(self, f"pre_attn{i + 1}")
            prex_in = x[i]
            s_i, mask_i = preattn(prex_in)
            if i == 0:
                s_out = s_i
                amask_out = mask_i
            else:
                s_out = torch.cat([s_out, s_i], 1)  # b*6h*64*64
                amask_out = torch.cat([amask_out, mask_i], -1)  # b*64*64*6
        amask_dia_max = self.MAX(amask_out)
        amask_dia_max = amask_dia_max.repeat(1, 1, self.ca_num_heads)
        mask_good = torch.where(amask_out == amask_dia_max, torch.Tensor([1]).cuda(self.device),
                                torch.Tensor([-1]).cuda(self.device))
        mask_good = mask_good.permute(0, 2, 1).contiguous()
        mask_good = torch.chunk(mask_good, self.ca_num_heads, dim=1)
        s_out = torch.chunk(s_out, self.ca_num_heads, dim=1)  # 6*b*64*64
        for i in range(self.ca_num_heads):
            Cattn = getattr(self, f"C_attn{i + 1}")
            Cmlp = getattr(self, f"C_mlp{i + 1}")
            s_out_i = s_out[i]
            Cmask_i = mask_good[i]

            Cmask_i = Cmask_i.transpose(-2, -1) @ Cmask_i
            Cmask_i = torch.unsqueeze(Cmask_i, dim=1)
            attn_i = s_out_i * Cmask_i
            s_in = x[i]
            attn_out = Cattn(s_in, attn_i)
            attn_out = s_in + attn_out
            attn_out = Cmlp(attn_out)
            if i == 0:
                Cattn_out = attn_out
            else:
                Cattn_out = torch.cat([Cattn_out, attn_out], 1)  # 6*b*h*64*64
        return Cattn_out


class MSConv(nn.Module):
    def __init__(self, in_channels=128, out_channels=48, eps=1e-8, ca_num_heads=6, norm_layer=nn.BatchNorm2d):
        super(MSConv, self).__init__()
        self.out_channels = out_channels
        self.ca_num_heads = ca_num_heads
        self.act = nn.ReLU(inplace=True)
        self.convcat = ConvBN(in_channels * 3, out_channels, kernel_size=1, stride=1)
        self.conv11 = ConvBNReLU6(in_channels, out_channels, kernel_size=3, stride=1)
        self.conv12 = ConvBNReLU6(in_channels, out_channels, kernel_size=3, stride=1)
        self.conv13 = ConvBNReLU6(in_channels, out_channels, kernel_size=3, stride=1)

        self.conv22 = ConvBNReLU6(in_channels, out_channels, kernel_size=3, stride=1)
        self.conv23 = ConvBNReLU6(in_channels, out_channels, kernel_size=3, stride=1)

        self.conv33 = ConvBNReLU6(in_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, X):
        shortcut = X
        x11 = self.conv11(X)
        x12 = self.conv12(X)
        x13 = self.conv13(X)
        x112 = x11 + x13 + x12

        x22 = self.conv22(x112)
        x23 = self.conv23(x112)
        x223 = x22 + x23

        x33 = self.conv33(x223)
        x_out = torch.cat([x112, x223, x33], 1)
        x_out = self.convcat(x_out)
        x_out = x_out + shortcut
        x_out = self.act(x_out)

        return x_out


class Detail_Recovery(nn.Module):
    def __init__(self,
                 decode_channels=256,
                 num_classes=6,
                 device=2,
                 norm_layer=nn.BatchNorm2d, mlp_ratio=4., drop=0., act_layer=nn.ReLU6,
                 ):
        super().__init__()
        self.device = device
        self.ca_num_heads = num_classes
        self.conv_channels = decode_channels // self.ca_num_heads
        self.wv = ConvBNReLU(decode_channels, self.conv_channels, kernel_size=1)
        for i in range(self.ca_num_heads):
            local_conv = MSConv(self.conv_channels, self.conv_channels)
            setattr(self, f"local_conv_{i + 1}", local_conv)

            normi = norm_layer(self.conv_channels)
            setattr(self, f"C_normi{i + 1}", normi)

    def forward(self, res, attn):
        attns = attn
        qp = self.wv(res)
        B, C, H, W = qp.shape
        point_center = qp.reshape(B, C, H * W).permute(0, 2, 1).contiguous().reshape(B, H * W, 1, C)


        feature_map_pad = F.pad(qp, (4, 4, 4, 4), 'constant', 0)  # 5*5    dialation=2
        unfolded = feature_map_pad.unfold(2, 9, 1).unfold(3, 9, 1)
        unfolded = unfolded.reshape(B, C, H * W, 9, 9)
        unfolded = torch.index_select(unfolded, -1, torch.tensor([0, 2, 4, 6, 8]).cuda(self.device))
        unfolded = torch.index_select(unfolded, -2, torch.tensor([0, 2, 4, 6, 8]).cuda(self.device))
        q = unfolded.reshape(B, C, H * W, 25).permute(0, 2, 3, 1).contiguous()  # B*(HW)*(9)*C


        point_aware = point_center @ q.transpose(-2, -1)  # 3*3=4  5*5=12 7*7=24
        point_aware = point_aware.index_fill_(-1, torch.LongTensor([12]).cuda(self.device),
                                              0)  # !!!!!!!!!!!!!!!!!!!!!!!!
        point_aware_attn = point_aware.softmax(dim=-1)
        point_center_aware = point_aware_attn @ q  # B*(HW)*1*C
        point_center = point_center + point_center_aware

        k = attns.reshape(B, self.ca_num_heads, C, H * W, ).permute(0, 3, 1, 2).contiguous().reshape(B, H * W,
                                                                                                     self.ca_num_heads,
                                                                                                     C)  # B*HW*6*C
        CLASS_attn = point_center @ k.transpose(-2, -1)  # B*HW*1*6
        CLASS_attn = CLASS_attn.softmax(dim=-1)
        CLASS_attn_out = CLASS_attn.transpose(-2, -1) @ point_center  # B*HW*6*C
        CLASS_attn_out = CLASS_attn_out.reshape(B, H * W, self.ca_num_heads * C).permute(0, 2, 1).contiguous().reshape(
            B, self.ca_num_heads * C, H, W)

        attn_in = torch.chunk(attn, self.ca_num_heads, dim=1)
        CLASS_attn_in = torch.chunk(CLASS_attn_out, self.ca_num_heads, dim=1)
        for i in range(self.ca_num_heads):
            Cmlp = getattr(self, f"local_conv_{i + 1}")
            C_norm = getattr(self, f"C_normi{i + 1}")
            CLASS_attn_i = CLASS_attn_in[i]
            CLASS_attn_norm = C_norm(CLASS_attn_i)
            attn_i = attn_in[i] + CLASS_attn_norm
            attn_out = Cmlp(attn_i)
            if i == 0:
                out = attn_out
            else:
                out = torch.cat([out, attn_out], 1)  # 6*b*h*64*64
        return out


class PREaux(nn.Module):
    def __init__(self, decode_channels=64, ca_num_heads=6, ):
        super().__init__()
        self.decode_channels = decode_channels
        self.ca_num_heads = ca_num_heads

        conv_channels = self.decode_channels // self.ca_num_heads

        for i in range(self.ca_num_heads):
            local_conv = nn.Sequential(
                ConvBNReLU(conv_channels, conv_channels),
                nn.Conv2d(conv_channels, 1, kernel_size=1, stride=1, padding=0),
            )
            setattr(self, f"local_conv_{i + 1}", local_conv)

    def forward(self, s4_out, s3_out, s2_out):
        s4_outup = F.interpolate(s4_out, scale_factor=2, mode='bilinear', align_corners=False)
        x = s4_outup + s3_out + s2_out

        x_in = torch.chunk(x, self.ca_num_heads, dim=1)
        for i in range(self.ca_num_heads):
            local_conv = getattr(self, f"local_conv_{i + 1}")
            x_i = x_in[i]
            s_i = local_conv(x_i)
            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out, s_i], 1)
        s_out = F.interpolate(s_out, scale_factor=8, mode='bilinear', align_corners=False)

        return s_out


class WS(nn.Module):
    def __init__(self, decode_channels=64, ca_num_heads=6, ):
        super().__init__()
        self.decode_channels = decode_channels
        self.ca_num_heads = ca_num_heads
        conv_channels = self.decode_channels // self.ca_num_heads
        for i in range(self.ca_num_heads):
            local_conv = ConvBNReLU(conv_channels * 2, conv_channels, kernel_size=1)
            setattr(self, f"local_conv_{i + 1}", local_conv)

    def forward(self, xl, xh):
        xl_in = torch.chunk(xl, self.ca_num_heads, dim=1)
        xh_in = torch.chunk(xh, self.ca_num_heads, dim=1)
        for i in range(self.ca_num_heads):
            local_conv = getattr(self, f"local_conv_{i + 1}")
            xl_i = xl_in[i]
            xh_i = xh_in[i]
            x_i = torch.cat([xl_i, xh_i], dim=1)
            s_i = local_conv(x_i)
            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out, s_i], 1)
        return s_out

class BAM(nn.Module):
    """Bottleneck Attention Module"""
    def __init__(self, channels, reduction=16, dilation=4):
        super(BAM, self).__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, kernel_size=3, stride=1,
                      padding=dilation, dilation=dilation, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_att(x)
        sa = self.spatial_att(x)
        out = x * ca * sa
        return out


class CIAPNet_Head(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=192,
                 todevice=0,
                 window_size=8,
                 num_classes=6,
                 eps=1e-8):
        super(CIAPNet_Head, self).__init__()
        self.todevice = todevice
        self.ca_num_heads = num_classes

        self.ws3 = WS(decode_channels, num_classes)
        self.ws2 = WS(decode_channels, num_classes)

        self.bottleneck1 = ConvBNReLU(encoder_channels[0], decode_channels, kernel_size=3)
        self.bottleneck2 = ConvBNReLU(encoder_channels[1], decode_channels, kernel_size=3)
        self.bottleneck3 = ConvBNReLU(encoder_channels[2], decode_channels, kernel_size=3)
        self.bottleneck4 = ConvBNReLU(encoder_channels[3], decode_channels, kernel_size=3)
        self.b1 = Detail_Recovery(decode_channels, num_classes=num_classes, device=todevice, )
        self.CaTTn4 = Block(dim=decode_channels, num_classes=num_classes, window_size=window_size,
                            device=self.todevice, )
        self.CaTTn3 = Block(dim=decode_channels, num_classes=num_classes, window_size=window_size,
                            device=self.todevice, )
        self.CaTTn2 = Block(dim=decode_channels, num_classes=num_classes, window_size=window_size,
                            device=self.todevice, )
        if self.training:
            self.aux = PREaux(decode_channels, num_classes)

        self.segmentation_head = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels),
            Conv(decode_channels, num_classes, kernel_size=1))

        self.init_weight()

# decoder解码部分，多尺度特征融合，金字塔结构

    def forward(self, x_list):
        if self.training:
            res1, res2, res3, res4 = x_list[0], x_list[1], x_list[2], x_list[3]
            x1 = self.bottleneck1(res1)
            x2 = self.bottleneck2(res2)
            x3 = self.bottleneck3(res3)
            x4 = self.bottleneck4(res4)


            x4_in = torch.chunk(x4, self.ca_num_heads, dim=1)
            s4_out = self.CaTTn4(x4_in)
            s4_out = F.interpolate(s4_out, scale_factor=2, mode='bilinear', align_corners=False)


            x3_list = self.ws3(x3, s4_out)
            x3_list = torch.chunk(x3_list, self.ca_num_heads, dim=1)
            s3_out = self.CaTTn3(x3_list)
            s3_out = F.interpolate(s3_out, scale_factor=2, mode='bilinear', align_corners=False)


            x2_list = self.ws2(x2, s3_out)
            x2_list = torch.chunk(x2_list, self.ca_num_heads, dim=1)
            s2_out = self.CaTTn2(x2_list)

            sh = self.aux(s4_out, s3_out, s2_out)

            s2_out = F.interpolate(s2_out, scale_factor=2, mode='bilinear', align_corners=False)
            s2_out = self.b1(x1, s2_out)
            x = self.segmentation_head(s2_out)
            x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

            return x, sh
        else:
            res1, res2, res3, res4 = x_list[0], x_list[1], x_list[2], x_list[3]
            x1 = self.bottleneck1(res1)
            x2 = self.bottleneck2(res2)
            x3 = self.bottleneck3(res3)
            x4 = self.bottleneck4(res4)


            x4_in = torch.chunk(x4, self.ca_num_heads, dim=1)
            s4_out = self.CaTTn4(x4_in)
            s4_out = F.interpolate(s4_out, scale_factor=2, mode='bilinear', align_corners=False)


            x3_list = self.ws3(x3, s4_out)
            x3_list = torch.chunk(x3_list, self.ca_num_heads, dim=1)
            s3_out = self.CaTTn3(x3_list)
            s3_out = F.interpolate(s3_out, scale_factor=2, mode='bilinear', align_corners=False)


            x2_list = self.ws2(x2, s3_out)
            x2_list = torch.chunk(x2_list, self.ca_num_heads, dim=1)
            s2_out = self.CaTTn2(x2_list)

            s2_out = F.interpolate(s2_out, scale_factor=2, mode='bilinear', align_corners=False)
            s2_out = self.b1(x1, s2_out)  # oudim=decoder*3
            x = self.segmentation_head(s2_out)

            x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

            return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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


class FeatureRefinementHead_single(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU6(decode_channels, decode_channels, kernel_size=3)

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
        self.post_conv = ConvBNReLU6(decode_channels, decode_channels, kernel_size=3)

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

class UNetFormer(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6
                 ):
        super().__init__()
        args = cfg.parse_args()
        self.sam = sam_model_registry["vit_l"](args, checkpoint='weights/sam_vit_l_0b3195.pth')
        self.image_encoder = self.sam.image_encoder
        encoder_channels = (256, 256, 256, 256)
        # encoder侧的金字塔结构
        self.fpn1x = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            Norm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.fpn2x = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.fpn3x = nn.Identity()
        self.fpn4x = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fpn1y = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.fpn2y = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.fpn3y = nn.Identity()
        self.fpn4y = nn.MaxPool2d(kernel_size=2, stride=2)
        #  新增：Encoder 金字塔 BAM
        self.bam1x = BAM(encoder_channels[0])
        self.bam2x = BAM(encoder_channels[1])
        self.bam3x = BAM(encoder_channels[2])
        self.bam4x = BAM(encoder_channels[3])

        self.bam1y = BAM(encoder_channels[0])
        self.bam2y = BAM(encoder_channels[1])
        self.bam3y = BAM(encoder_channels[2])
        self.bam4y = BAM(encoder_channels[3])

        self.fusion1 = SEFusion(encoder_channels[0])
        self.fusion2 = SEFusion(encoder_channels[1])
        self.fusion3 = SEFusion(encoder_channels[2])
        self.fusion4 = SEFusion(encoder_channels[3])


        for n, value in self.image_encoder.named_parameters():
            if 'lora_' not in n:
                value.requires_grad = False
            else:
                value.requires_grad = True

        # 【修改1】使用 CIAPNet_Head 替换旧 decoder
        self.decoder = CIAPNet_Head(
            encoder_channels=encoder_channels,
            decode_channels=num_classes * 32,  # CIAPNet 默认设置
            todevice=0,
            window_size=window_size,
            num_classes=num_classes
        )

        # （旧版 decoder 已注释保留）
        # self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)

    def forward(self, x, y, mode='Train'):
        """
        x: RGB, 形状通常是 (B, 3, H, W)
        y: DSM, 可能是 (B, H, W) 或 (B, 1, H, W)
        mode: 'Train' / 'Test' / 'Heatmap'
        """
        # ---------- 统一 y 的形状 ----------
        # 支持 (B,H,W) 或 (B,1,H,W) 或 (B,3,H,W)
        if y.dim() == 3:
            # (B, H, W) -> (B, 1, H, W)
            y = y.unsqueeze(1)

        if y.dim() != 4:
            raise ValueError(f"y 维度错误，期望 3D 或 4D，得到 {y.dim()}D, shape={y.shape}")

        # 此时 y: (B, C, H, W)
        if y.size(1) == 1:
            # (B,1,H,W) -> (B,3,H,W)
            y = y.repeat(1, 3, 1, 1)
        elif y.size(1) == 3:
            # 已经是 3 通道，保持不变
            pass
        else:
            raise ValueError(f"y 通道数错误，期望 1 或 3，得到 {y.size(1)}, shape={y.shape}")

        h, w = x.size()[-2:]

        # ---------- Encoder ----------
        deepx, deepy = self.image_encoder(x, y)  # 这里的 deepx/deepy 需要参与梯度
        heatmaps = [deepx, deepy]

        # ---------- FPN + BAM ----------
        res1x = self.bam1x(self.fpn1x(deepx))
        res2x = self.bam2x(self.fpn2x(deepx))
        res3x = self.bam3x(self.fpn3x(deepx))
        res4x = self.bam4x(self.fpn4x(deepx))

        res1y = self.bam1y(self.fpn1y(deepy))
        res2y = self.bam2y(self.fpn2y(deepy))
        res3y = self.bam3y(self.fpn3y(deepy))
        res4y = self.bam4y(self.fpn4y(deepy))

        res1 = self.fusion1(res1x, res1y)
        res2 = self.fusion2(res2x, res2y)
        res3 = self.fusion3(res3x, res3y)
        res4 = self.fusion4(res4x, res4y)

        x_list = [res1, res2, res3, res4]

        # ---------- Train 模式 ----------
        if mode == 'Train':
            # CIAPNet_Head：训练时返回 (x, sh)
            x, sh = self.decoder(x_list)
            return x, sh

        # ---------- Test 模式 ----------
        if mode == 'Test':
            # 推理时只需要语义分割输出
            x = self.decoder(x_list)
            return x

        # ---------- Heatmap 模式 ----------
        if mode == 'Heatmap':
            # Heatmap 建议强制 batch=1
            assert x.size(0) == 1, "Heatmap 模式必须 batch=1"

            # decoder 正常前向（保持与 Test 一致）
            x = self.decoder(x_list)  # shape: (1, num_classes, H, W)

            # 你原来是固定取 (1, 100, 65)，这里保持不变
            # 也可以改成 argmax 的位置，这里先不动
            pred = x[0, 1, 100, 65]

            heatmap_outputs = []
            for feature in heatmaps:
                # feature: (B, C, Hf, Wf)
                grads = autograd.grad(pred, feature, retain_graph=True)[0]  # (B, C, Hf, Wf)
                # GAP over Hf, Wf
                pooled_grads = torch.mean(grads, dim=(2, 3))[0]  # (C,)
                feature_map = feature[0]  # (C, Hf, Wf)

                # 每个通道乘以对应权重
                feature_map = feature_map * pooled_grads[:, None, None]

                # 转成 numpy
                heatmap = feature_map.detach().cpu().numpy()
                heatmap = np.mean(heatmap, axis=0)  # (Hf, Wf)

                heatmap = np.maximum(heatmap, 0)
                maxv = np.max(heatmap)
                if maxv > 0:
                    heatmap /= maxv

                heatmap_outputs.append(heatmap)

            heatmap1 = heatmap_outputs[0]
            heatmap2 = heatmap_outputs[1] if len(heatmap_outputs) > 1 else None

            return x, heatmap1, heatmap2

        # ---------- 兜底 ----------
        raise ValueError("mode 必须为 'Train' / 'Test' / 'Heatmap'")

