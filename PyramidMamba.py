import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from mamba_ssm import Mamba


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
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
            norm_layer(in_channels),
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


# class MambaLayer(nn.Module):
#     def __init__(self, in_chs=512, dim=128, d_state=16, d_conv=4, expand=2, last_feat_size=16):
#         super().__init__()
#         pool_scales = self.generate_arithmetic_sequence(1, last_feat_size, last_feat_size // 4)
#         self.pool_len = len(pool_scales)
#         self.pool_layers = nn.ModuleList()
#         self.pool_layers.append(nn.Sequential(
#                     ConvBNReLU(in_chs, dim, kernel_size=1),
#                     nn.AdaptiveAvgPool2d(1)
#                     ))
#         for pool_scale in pool_scales[1:]:
#             self.pool_layers.append(
#                 nn.Sequential(
#                     nn.AdaptiveAvgPool2d(pool_scale),
#                     ConvBNReLU(in_chs, dim, kernel_size=1)
#                     ))
#         self.mamba = Mamba(
#             d_model=dim*self.pool_len+in_chs,  # Model dimension d_model
#             d_state=d_state,  # SSM state expansion factor
#             d_conv=d_conv,  # Local convolution width
#             expand=expand # Block expansion factor
#         )
#
#     def forward(self, x): # B, C, H, W
#         res = x
#         B, C, H, W = res.shape
#         ppm_out = [res]
#         for p in self.pool_layers:
#             pool_out = p(x)
#             pool_out = F.interpolate(pool_out, (H, W), mode='bilinear', align_corners=False)
#             ppm_out.append(pool_out)
#         x = torch.cat(ppm_out, dim=1)
#         _, chs, _, _ = x.shape
#         x = rearrange(x, 'b c h w -> b (h w) c', b=B, c=chs, h=H, w=W)
#         x = self.mamba(x)
#         x = x.transpose(2, 1).view(B, chs, H, W)
#         return x
#
#     def generate_arithmetic_sequence(self, start, stop, step):
#         sequence = []
#         for i in range(start, stop, step):
#             sequence.append(i)
#         return sequence

class MultiHeadMambaLayer(nn.Module):
    def __init__(self,
                 in_chs: int = 512,
                 dim: int = 128,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 last_feat_size: int = 16,
                 heads: int = 4):
        super().__init__()
        # 1) 构造多尺度池化 + 1×1→dim
        pool_scales = list(range(1, last_feat_size, max(1, last_feat_size // 4)))
        self.pool_layers = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_chs, dim, 1, bias=False),
                nn.ReLU(inplace=True)
            )
            for s in pool_scales
        ])
        self.pool_len = len(pool_scales)
        total_ch = in_chs + dim * self.pool_len

        assert total_ch % heads == 0, "total_ch 必须能被 heads 整除"
        head_dim = total_ch // heads

        # 2) 为每个 head 各自准备一个 Mamba
        self.heads = heads
        self.mambas = nn.ModuleList([
            Mamba(
              d_model=head_dim,
              d_state=d_state,
              d_conv=d_conv,
              expand=expand
            )
            for _ in range(heads)
        ])

        # 3) 最后拼回后投射回原通道
        self.proj = nn.Conv2d(total_ch, in_chs, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        # —— 多尺度拼接 ——
        feats = [x]
        for p in self.pool_layers:
            y = p(x)
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
            feats.append(y)
        cat = torch.cat(feats, dim=1)        # [B, total_ch, H, W]
        total_ch = cat.shape[1]

        # —— 按 head 切分 ——
        cat = cat.view(B, self.heads, total_ch // self.heads, H, W)
        head_outs = []
        for i in range(self.heads):
            h_feat = cat[:, i]                # [B, head_dim, H, W]
            seq = rearrange(h_feat, 'b c h w -> b (h w) c')  # [B, H*W, head_dim]
            out = self.mambas[i](seq)                      # [B, H*W, head_dim]
            out = out.transpose(2,1).view(B, total_ch//self.heads, H, W)
            head_outs.append(out)

        fused = torch.cat(head_outs, dim=1)  # [B, total_ch, H, W]
        return self.proj(fused)              # [B, in_chs, H, W]

class ChannelSpatialMambaLayer(nn.Module):
    def __init__(self,
                 in_chs: int = 512,
                 embed_dim: int = 128,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        # 空间 SSM 用的 embed 投射
        self.spatial_proj = nn.Conv2d(in_chs, embed_dim, kernel_size=1, bias=False)
        self.spatial_mamba = Mamba(d_model=embed_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.spatial_out = nn.Conv2d(embed_dim, in_chs, kernel_size=1, bias=False)

        # 通道 SSM 用的 embed 投射（先降维到 embed_dim，然后再还原）
        self.channel_proj = nn.Conv2d(in_chs, embed_dim, kernel_size=1, bias=False)
        self.channel_mamba = Mamba(d_model=embed_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.channel_out = nn.Conv2d(embed_dim, in_chs, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        # —— 空间 SSM ——
        sp = self.spatial_proj(x)                     # [B, E, H, W]
        seq_sp = rearrange(sp, 'b e h w -> b (h w) e')# [B, H*W, E]
        out_sp = self.spatial_mamba(seq_sp)           # [B, H*W, E]
        out_sp = rearrange(out_sp, 'b (h w) e -> b e h w', h=H, w=W)
        out_sp = self.spatial_out(out_sp)             # [B, C, H, W]

        # —— 通道 SSM ——
        ch = self.channel_proj(x)                     # [B, E, H, W]
        seq_ch = rearrange(ch, 'b e h w -> b (h w) e')# [B, H*W, E]
        # 反向把 H*W 当作“特征维”，C 作为序列长度
        seq_ch = seq_ch.transpose(1,2)                # [B, E, H*W]
        out_ch = self.channel_mamba(seq_ch)           # [B, E, H*W]
        out_ch = out_ch.transpose(1,2).view(B, self.embed_dim, H, W)
        out_ch = self.channel_out(out_ch)             # [B, C, H, W]

        # —— 融合 ——
        return out_sp + out_ch
class MambaLayer(nn.Module):
    def __init__(self,
                 in_chs: int = 512,
                 dim: int = 128,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 last_feat_size: int = 16):
        super().__init__()
        # 1) 决定池化尺度
        pool_scales = list(range(1, last_feat_size, max(1, last_feat_size // 4)))
        # 2) 构造对应的 pool+1×1→dim
        self.pool_layers = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                ConvBNReLU(in_chs, dim, kernel_size=1),
            )
            for s in pool_scales
        ])
        # pool_len 用于 ManBaBlock 里 conv_ffn 的通道计算
        self.pool_len = len(pool_scales)

        # 3) 正确的拼接后总通道数
        total_ch = in_chs + dim * self.pool_len

        # 4) 传给 Mamba 的 d_model 一定要 = total_ch
        self.mamba = Mamba(
            d_model=total_ch,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # 5) 投射回原始 in_chs
        self.proj = nn.Conv2d(total_ch, in_chs, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        outs = [x]
        for p in self.pool_layers:
            y = p(x)  # [B, dim, s, s]
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(y)

        cat = torch.cat(outs, dim=1)               # [B, total_ch, H, W]
        seq = rearrange(cat, 'b c h w -> b (h w) c')  # [B, H*W, total_ch]
        ssm_out = self.mamba(seq)                  # [B, H*W, total_ch]

        out = ssm_out.transpose(2,1).view(B, -1, H, W)  # [B, total_ch, H, W]
        out = self.proj(out)                         # [B, in_chs, H, W]
        return out
class ConvFFN(nn.Module):
    def __init__(self, in_ch=128, hidden_ch=512, out_ch=128, drop=0.):
        super(ConvFFN, self).__init__()
        self.conv = ConvBNReLU(in_ch, in_ch, kernel_size=3)
        self.fc1 = Conv(in_ch, hidden_ch, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = Conv(hidden_ch, out_ch, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

#
# class ManBaBlock(nn.Module):
#     def __init__(self, in_chs=512, dim=128, hidden_ch=512, out_ch=128, drop=0.1, d_state=16, d_conv=4, expand=2, last_feat_size=16):
#         super(ManBaBlock, self).__init__()
#         self.mamba = MambaLayer(in_chs=in_chs, dim=dim, d_state=d_state, d_conv=d_conv, expand=expand, last_feat_size=last_feat_size)
#         self.conv_ffn = ConvFFN(in_ch=dim*self.mamba.pool_len+in_chs, hidden_ch=hidden_ch, out_ch=out_ch, drop=drop)
#
#     def forward(self, x):
#         x = self.mamba(x)
#         x = self.conv_ffn(x)
#
#         return x

class ManBaBlock(nn.Module):
    def __init__(self,
                 in_chs: int = 512,
                 dim: int = 128,
                 hidden_ch: int = 512,
                 out_ch: int = 128,
                 drop: float = 0.1,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 last_feat_size: int = 16):
        super().__init__()
        # 先做 SSM + 多尺度拼接 + 投射回 in_chs
        self.mamba = MultiHeadMambaLayer(
            in_chs=in_chs,
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            last_feat_size=last_feat_size
        )
        # 这里 conv_ffn 的输入通道一定是 in_chs（因为 MambaLayer.proj 输出 in_chs）
        self.conv_ffn = nn.Sequential(
            nn.Conv2d(in_chs, hidden_ch, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=False),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x = self.mamba(x)    # -> [B, in_chs, H, W]
        x = self.conv_ffn(x) # -> [B, out_ch, H, W]
        return x



class Decoder(nn.Module):
    def __init__(self, encoder_channels=(64, 128, 256, 512), decoder_channels=128, num_classes=6, last_feat_size=16):
        super().__init__()
        self.b3 = ManBaBlock(in_chs=encoder_channels[-1], dim=decoder_channels, last_feat_size=last_feat_size)
        self.up_conv = nn.Sequential(ConvBNReLU(decoder_channels, decoder_channels),
                                     nn.Upsample(scale_factor=2),
                                     ConvBNReLU(decoder_channels, decoder_channels),
                                     nn.Upsample(scale_factor=2),
                                     ConvBNReLU(decoder_channels, decoder_channels),
                                     nn.Upsample(scale_factor=2),
                                     )
        self.pre_conv = ConvBNReLU(encoder_channels[0], decoder_channels)
        self.head = nn.Sequential(ConvBNReLU(decoder_channels, decoder_channels // 2),
                                  nn.Upsample(scale_factor=2, mode='bilinear'),
                                  ConvBNReLU(decoder_channels // 2, decoder_channels // 2),
                                  nn.Upsample(scale_factor=2, mode='bilinear'),
                                  Conv(decoder_channels // 2, num_classes, kernel_size=1))
        self.apply(self._init_weights)

    def forward(self, x0, x3):
        x3 = self.b3(x3)
        x3 = self.up_conv(x3)
        x = x3 + self.pre_conv(x0)
        x = self.head(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class EfficientPyramidMamba(nn.Module):
    def __init__(self,
                 backbone_name='swsl_resnet18',
                 pretrained=True,
                 num_classes=6,
                 decoder_channels=128,
                 last_feat_size=16  # last_feat_size=input_img_size // 32
                 ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()
        self.decoder = Decoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels, num_classes=num_classes, last_feat_size=last_feat_size)

    def forward(self, x):
        x0, x3 = self.backbone(x)
        x = self.decoder(x0, x3)

        return x


class PyramidMamba(nn.Module):
    def __init__(self,
                 backbone_name='swin_base_patch4_window12_384.ms_in22k_ft_in1k',
                 pretrained=True,
                 num_classes=6,
                 decoder_channels=128,
                 last_feat_size=32,
                 img_size=1024
                 ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32, img_size=img_size,
                                          out_indices=(-4, -1), pretrained=pretrained)

        encoder_channels = self.backbone.feature_info.channels()
        self.decoder = Decoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels, num_classes=num_classes, last_feat_size=last_feat_size)

    def forward(self, x):
        x0, x3 = self.backbone(x)
        x0 = x0.permute(0, 3, 1, 2)
        x3 = x3.permute(0, 3, 1, 2)
        x = self.decoder(x0, x3)

        return x
