import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

from .mamba_fusion import MambaFusionBlock

class PositionalEncoding2D(nn.Module):
    """
    2D Sinusoidal Positional Encoding for AIFI.
    """
    def __init__(self, hidden_dim, temperature=10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=x.device)
        
        y_embed = y_embed.unsqueeze(1).repeat(1, W)
        x_embed = x_embed.unsqueeze(0).repeat(H, 1)

        dim_t = torch.arange(self.hidden_dim // 2, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.hidden_dim // 2))

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)
        
        return x + pos

class AIFI(nn.Module):
    """
    Attention-based Intra-scale Feature Interaction.
    """
    def __init__(self, hidden_dim=256, num_heads=8, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.pos_encoder = PositionalEncoding2D(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_pos = self.pos_encoder(x)
        
        # Flatten for attention
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_pos_flat = x_pos.flatten(2).transpose(1, 2) # (B, H*W, C)
        
        # MHSA with Pre-Norm
        attn_out, _ = self.attn(self.norm1(x_pos_flat), self.norm1(x_pos_flat), self.norm1(x_flat))
        x_flat = x_flat + attn_out
        
        # FFN with Pre-Norm
        ffn_out = self.ffn(self.norm2(x_flat))
        x_flat = x_flat + ffn_out
        
        # Reshape back
        out = x_flat.transpose(1, 2).reshape(B, C, H, W)
        return out

class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class RepVggBlock(nn.Module):
    """RT-DETR's RepVGG block (training form): 3x3 conv-BN + 1x1 conv-BN, summed, SiLU."""
    def __init__(self, channels):
        super().__init__()
        self.conv3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, bias=False), nn.BatchNorm2d(channels))
        self.conv1 = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0, bias=False), nn.BatchNorm2d(channels))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv3(x) + self.conv1(x))


class CSPRepLayer(nn.Module):
    """RT-DETR's CCFF fusion block (RepC3 / CSPRepLayer, expansion 1.0, 3 RepVGG blocks): the plain-RT-DETR
    baseline for MambaFusionBlock (v2: encoder.fusion = 'repc3')."""
    def __init__(self, in_channels, out_channels, num_blocks=3):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, 1)
        self.conv2 = ConvNormAct(in_channels, out_channels, 1)
        self.bottlenecks = nn.Sequential(*[RepVggBlock(out_channels) for _ in range(num_blocks)])

    def forward(self, x):
        return self.bottlenecks(self.conv1(x)) + self.conv2(x)


class HybridEncoder(nn.Module):
    """
    RT-DETR Hybrid Encoder with AIFI and Mamba CCFF.

    Operates on P3, P4 and P5 (strides 8, 16, 32). The full FPN-style
    top-down path fuses P5→P4→P3, and the bottom-up path fuses O3→O4→O5,
    with Mamba CCFF fusion at each step. This gives the decoder a high-res
    feature level for small object detection.
    """
    def __init__(self, in_channels=[512, 1024, 2048], hidden_dim=256, num_heads=8,
                 dim_feedforward=1024, d_state=16, d_conv=4, expand=2, 
                 mlp_ratio=2, num_scans=4, dropout=0.0, use_checkpoint=False, fusion='mamba'):
        super().__init__()
        self.hidden_dim = hidden_dim
        # v2: fusion = 'mamba' (MambaFusionBlock, as in the 2026-07-29 run) or 'repc3' (RT-DETR's CSPRepLayer: the
        # same encoder graph with the original fusion block, for a like-for-like baseline). The attribute names
        # (td_mamba1, ...) are kept for both, so train.build_optimizer's lr_mamba applies to the fusion blocks
        # either way and the training recipe is identical.
        assert fusion in ('mamba', 'repc3'), fusion
        self.fusion = fusion

        def fuse():
            if fusion == 'repc3':
                return CSPRepLayer(hidden_dim * 2, hidden_dim)
            return MambaFusionBlock(
                hidden_dim * 2, hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand,
                mlp_ratio=mlp_ratio, num_scans=num_scans, dropout=dropout, use_checkpoint=use_checkpoint,
            )
        
        # Input projections (P3, P4, P5)
        self.proj3 = nn.Conv2d(in_channels[0], hidden_dim, 1)
        self.proj4 = nn.Conv2d(in_channels[1], hidden_dim, 1)
        self.proj5 = nn.Conv2d(in_channels[2], hidden_dim, 1)
        
        # AIFI on lowest resolution (P5)
        self.aifi = AIFI(hidden_dim, num_heads, dim_feedforward, dropout)
        
        # CCFF Top-down path: P5_attn + P4 -> T4, then T4 + P3 -> O3
        self.td_mamba1 = fuse()
        self.td_mamba2 = fuse()
        
        # CCFF Bottom-up path: down(O3) + T4 -> O4, then down(O4) + P5_attn -> O5
        self.bu_down1 = ConvNormAct(hidden_dim, hidden_dim, 3, 2, 1)
        self.bu_mamba1 = fuse()
        self.bu_down2 = ConvNormAct(hidden_dim, hidden_dim, 3, 2, 1)
        self.bu_mamba2 = fuse()

    def forward(self, features):
        """
        features: OrderedDict {'C3': ..., 'C4': ..., 'C5': ...}
        Returns: list of 3 tensors [O3, O4, O5] each (B, hidden_dim, Hi, Wi)
        """
        c3, c4, c5 = features['C3'], features['C4'], features['C5']
        
        # Projections
        p3 = self.proj3(c3)
        p4 = self.proj4(c4)
        p5 = self.proj5(c5)
        
        # AIFI on P5
        p5_attn = self.aifi(p5)
        
        # Top-down path: P5_attn + P4 -> T4
        p5_up = F.interpolate(p5_attn, size=p4.shape[-2:], mode='nearest')
        t4_in = torch.cat([p5_up, p4], dim=1)
        t4 = self.td_mamba1(t4_in)
        
        # Top-down path: T4 + P3 -> O3
        t4_up = F.interpolate(t4, size=p3.shape[-2:], mode='nearest')
        o3_in = torch.cat([t4_up, p3], dim=1)
        o3 = self.td_mamba2(o3_in)
        
        # Bottom-up path: down(O3) + T4 -> O4
        o3_down = self.bu_down1(o3)
        o4_in = torch.cat([o3_down, t4], dim=1)
        o4 = self.bu_mamba1(o4_in)
        
        # Bottom-up path: down(O4) + P5_attn -> O5
        o4_down = self.bu_down2(o4)
        o5_in = torch.cat([o4_down, p5_attn], dim=1)
        o5 = self.bu_mamba2(o5_in)
        
        return [o3, o4, o5]
