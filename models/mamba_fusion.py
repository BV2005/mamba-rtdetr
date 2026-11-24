import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

try:
    from mamba_ssm import Mamba
    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False
# v2: MAMBA_FALLBACK=1 forces the pure-PyTorch SelectiveSSM (identical parameters, see below), e.g. for the CPU
# checks in test_v2_fixes.py on a machine where the CUDA-only mamba_ssm is installed
import os as _os
if _os.environ.get('MAMBA_FALLBACK') == '1':
    HAS_MAMBA_SSM = False

class SelectiveSSM(nn.Module):
    """Pure PyTorch implementation of Mamba's selective scan.
    
    This acts as a fallback when the CUDA-compiled mamba-ssm is not available.
    v2: same parameters, shapes and initialisation as mamba_ssm.Mamba (dt_rank = ceil(d_model / 16), not 1), so a
    CPU run builds the same model as the CUDA run and checkpoints load in both.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.dt_rank = math.ceil(self.d_model / 16)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # mamba_ssm's dt initialisation: weights ~ U(+-dt_rank^-0.5), bias = softplus^-1(dt), dt log-uniform in
        # [dt_min, dt_max]
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):  
        # (B, L, D) -> (B, L, D)
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        # 1D Conv
        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Projections
        x_dbl = self.x_proj(x_conv)
        delta, B_proj, C_proj = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        delta = F.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log)

        # Discretization and Recurrent Scan
        y = self.selective_scan(x_conv, delta, A, B_proj, C_proj, self.D)
        
        # Gating and Output Projection
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out

    def selective_scan(self, u, delta, A, B, C, D):
        # u: (B, L, D_inner)
        # delta: (B, L, D_inner)
        # A: (D_inner, D_state)
        # B: (B, L, D_state)
        # C: (B, L, D_state)
        # D: (D_inner)
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[1]

        h = torch.zeros(batch_size, d_inner, d_state, device=u.device)
        ys = []

        for i in range(seq_len):
            delta_i = delta[:, i]
            u_i = u[:, i]
            B_i = B[:, i]
            C_i = C[:, i]

            # Zero-order hold discretization
            # A_bar: (B, D_inner, D_state)
            A_bar = torch.exp(delta_i.unsqueeze(-1) * A.unsqueeze(0))
            # B_bar: (B, D_inner, D_state)
            B_bar = (delta_i.unsqueeze(-1) * B_i.unsqueeze(1))
            
            # Recurrent update
            h = A_bar * h + B_bar * u_i.unsqueeze(-1)
            
            # Compute output
            y_i = torch.sum(h * C_i.unsqueeze(1), dim=-1) + D * u_i
            ys.append(y_i)

        return torch.stack(ys, dim=1)


class CrossScanMamba2D(nn.Module):
    """Cross-scan Mamba for 2D feature maps (row-forward + row-backward by
    default; set num_scans=4 to add the two column-wise directions back)."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, num_scans=2):
        super().__init__()
        self.num_scans = num_scans
        self.d_model = d_model
        
        self.mamba_layers = nn.ModuleList()
        for _ in range(num_scans):
            if HAS_MAMBA_SSM:
                self.mamba_layers.append(
                    Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                )
            else:
                self.mamba_layers.append(
                    SelectiveSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                )

    def forward(self, x):  
        # (B, C, H, W) -> (B, C, H, W)
        B, C, H, W = x.shape
        
        # Flatten for scanning
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Scan 1: Row forward
        x1 = self.mamba_layers[0](x_flat)
        
        if self.num_scans > 1:
            # Scan 2: Row backward
            x_bwd = torch.flip(x_flat, dims=[1])
            x2 = self.mamba_layers[1](x_bwd)
            x2 = torch.flip(x2, dims=[1])
        else:
            x2 = 0
            
        if self.num_scans > 2:
            # Scan 3: Col forward
            x_col = x.transpose(2, 3).flatten(2).transpose(1, 2) # (B, W*H, C)
            x3 = self.mamba_layers[2](x_col)
            # Revert to row-major
            x3 = x3.transpose(1, 2).view(B, C, W, H).transpose(2, 3).flatten(2).transpose(1, 2)
        else:
            x3 = 0
            
        if self.num_scans > 3:
            # Scan 4: Col backward
            x_col = x.transpose(2, 3).flatten(2).transpose(1, 2)
            x_col_bwd = torch.flip(x_col, dims=[1])
            x4 = self.mamba_layers[3](x_col_bwd)
            x4 = torch.flip(x4, dims=[1])
            # Revert to row-major
            x4 = x4.transpose(1, 2).view(B, C, W, H).transpose(2, 3).flatten(2).transpose(1, 2)
        else:
            x4 = 0

        # Merge scans
        y = x1 + x2 + x3 + x4
        y = y / self.num_scans
        
        # Reshape back to (B, C, H, W)
        return y.transpose(1, 2).view(B, C, H, W)


class FusionMLP(nn.Module):
    """Feed-forward MLP with GELU activation."""
    def __init__(self, dim, expand_ratio=2, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * expand_ratio)
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  
        # (B, C, H, W) -> (B, C, H, W)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNorm2d(nn.Module):
    """LayerNorm specifically for 2D spatial features (B, C, H, W)."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x


class MambaFusionBlock(nn.Module):
    """Replaces RepC3 in RT-DETR's CCFF module.
    
    Architecture:
        concat_features (2C) 
          → 1×1 Conv + 3×3 DWConv + BN + SiLU  (channel compression, local context)
          → LayerNorm → CrossScanMamba → + residual  (global context)
          → LayerNorm → MLP → + residual  (non-linear transform)
    """
    def __init__(self, in_channels, out_channels, d_state=16, d_conv=4, expand=2, 
                 mlp_ratio=2, num_scans=2, dropout=0.0, use_checkpoint=False):
        super().__init__()

        # Gradient checkpointing: trades compute for memory by not storing the
        # Mamba/MLP activations for backward and recomputing them instead.
        # Useful since CrossScanMamba2D activations are the main memory driver
        # for this block (multiple scan directions over the full H*W sequence).
        self.use_checkpoint = use_checkpoint

        # Channel compression and local spatial context
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.dwconv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, 
                                   groups=out_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        
        # Mamba module with Pre-Norm
        self.norm1 = LayerNorm2d(out_channels)
        self.mamba = CrossScanMamba2D(d_model=out_channels, d_state=d_state, d_conv=d_conv, 
                                      expand=expand, num_scans=num_scans)
        
        # MLP module with Pre-Norm
        self.norm2 = LayerNorm2d(out_channels)
        self.mlp = FusionMLP(dim=out_channels, expand_ratio=mlp_ratio, dropout=dropout)

    def _mamba_branch(self, x):
        return self.mamba(self.norm1(x))

    def _mlp_branch(self, x):
        return self.mlp(self.norm2(x))

    def forward(self, x):  
        # (B, 2C, H, W) -> (B, C, H, W)
        
        # 1. 1x1 Conv + 3x3 DWConv + BN + SiLU
        x = self.conv1x1(x)
        x = self.dwconv3x3(x)
        x = self.bn(x)
        x = self.act(x)
        
        # 2. Pre-Norm + CrossScanMamba + Residual
        if self.use_checkpoint and self.training:
            x = x + cp.checkpoint(self._mamba_branch, x, use_reentrant=False)
        else:
            x = x + self._mamba_branch(x)
        
        # 3. Pre-Norm + MLP + Residual
        if self.use_checkpoint and self.training:
            x = x + cp.checkpoint(self._mlp_branch, x, use_reentrant=False)
        else:
            x = x + self._mlp_branch(x)
        
        return x
