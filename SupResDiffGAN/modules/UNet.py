import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.unets import UNet2DModel
from mamba_ssm import Mamba

def dwt_init(x):
    """
    Discrete Wavelet Transform (DWT) to extract sub-pixel high-frequency info.
    Splits spatial resolution into channel depth.
    """
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    return torch.cat([x1, x2, x3, x4], dim=1)

class SGFN(nn.Module):
    """
    Spatial Gated Feed-Forward Network.
    Captures spatial dependencies more effectively than standard MLPs for SR.
    """
    def __init__(self, dim, dw_expansion=2):
        super().__init__()
        dw_dim = dim * dw_expansion
        self.conv1 = nn.Conv2d(dim, dw_dim * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(dw_dim * 2, dw_dim * 2, kernel_size=3, padding=1, groups=dw_dim * 2)
        self.conv2 = nn.Conv2d(dw_dim, dim, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        # Gating mechanism for spatial feature selection
        x = x1 * torch.sigmoid(x2)
        return self.conv2(x)

class MHSS_SSSM_Block(nn.Module):
    """
    Multi-Head Selective Scan (MHSS) & Scalable State Space Model (SSSM).
    Processes images in 4 directions to eliminate sequence bias.
    """
    def __init__(self, dim, scale_factor=4):
        super().__init__()
        self.dim = dim
        
        # 4-Directional Scanning (Standard, Reversed, Width-Standard, Width-Reversed)
        self.mamba_heads = nn.ModuleList([
            Mamba(d_model=dim, d_state=16, d_conv=4, expand=1) for _ in range(4)
        ])
        
        # SSSM modulation: Adjusts state transitions based on Super-Resolution scale
        self.scale_modulator = nn.Parameter(torch.ones(1) * (1.0 / scale_factor))

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).view(B, -1, C) # (B, L, C)
        
        # 1. Horizontal Forward
        out = self.mamba_heads[0](x_flat)
        
        # 2. Horizontal Backward
        out += self.mamba_heads[1](x_flat.flip(dims=[1])).flip(dims=[1])
        
        # 3. Vertical Forward (Transpose spatial dims)
        x_spatial = x_flat.view(B, H, W, C)
        x_v = x_spatial.permute(0, 2, 1, 3).reshape(B, -1, C)
        v_out = self.mamba_heads[2](x_v).view(B, W, H, C).permute(0, 2, 1, 3).reshape(B, -1, C)
        out += v_out
        
        # 4. Vertical Backward
        v_out_rev = self.mamba_heads[3](x_v.flip(dims=[1])).flip(dims=[1])
        out += v_out_rev.view(B, W, H, C).permute(0, 2, 1, 3).reshape(B, -1, C)
        
        return (out / 4) * self.scale_modulator

class HiMambaBlock(nn.Module):
    """
    Hierarchical Mamba Block.
    Combines local window modeling with global regional scanning.
    """
    def __init__(self, dim, window_size=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        
        # Local Window-based SSM for fine textures
        self.local_ssm = Mamba(d_model=dim, d_state=8, d_conv=2, expand=1)
        # Regional Global SSM for structural consistency
        self.region_ssm = MHSS_SSSM_Block(dim)
        
        self.norm = nn.LayerNorm(dim)
        self.sgfn = SGFN(dim)
        self.gamma = nn.Parameter(torch.zeros(1)) # Learnable residual connection

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        
        # Normalization and Spatial-to-Feature conversion
        x_norm = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Regional / Global MHSS Path
        x_mamba = self.region_ssm(x_norm).view(B, H, W, C).permute(0, 3, 1, 2)
        
        # Local Window Path
        # Simple window partition for local context
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x_padded = F.pad(x_norm, (0, pad_w, 0, pad_h))
        
        Hp, Wp = x_padded.shape[2:]
        x_win = x_padded.view(B, C, Hp//self.window_size, self.window_size, Wp//self.window_size, self.window_size)
        x_win = x_win.permute(0, 2, 4, 3, 5, 1).reshape(-1, self.window_size * self.window_size, C)
        
        x_local = self.local_ssm(x_win).reshape(B, Hp//self.window_size, Wp//self.window_size, self.window_size, self.window_size, C)
        x_local = x_local.permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)
        x_local = x_local[:, :, :H, :W] # Unpad
        
        # Fusion
        x_fused = x_mamba + x_local
        
        # SGFN Refinement Path
        x_refined = x_fused + self.sgfn(x_fused)
        
        return shortcut + self.gamma * x_refined

class HybridUNet(nn.Module):
    """
    The complete Hybrid UNet for SupResDiffGAN.
    Integrates Diffusers UNet with HiMamba Bottleneck and DWT Prior.
    """
    def __init__(self, channels: list[int] = [64, 96, 128, 256]):
        super().__init__()
        self.unet = UNet2DModel(
            in_channels=8, # Concat of noisy latent (4) and LR latent (4)
            out_channels=4,
            block_out_channels=channels,
            layers_per_block=2,
            down_block_types=("ResnetDownsampleBlock2D", "ResnetDownsampleBlock2D", "ResnetDownsampleBlock2D", "ResnetDownsampleBlock2D"),
            up_block_types=("ResnetUpsampleBlock2D", "ResnetUpsampleBlock2D", "ResnetUpsampleBlock2D", "ResnetUpsampleBlock2D"),
            add_attention=[False, False, True, True],
        )
        
        # DWT head to inject high-frequency sub-pixel information
        self.dwt_projection = nn.Conv2d(channels[-1] * 4, channels[-1], kernel_size=1)
        
        # Main Bottleneck Refiner
        self.hi_mamba_bottleneck = HiMambaBlock(dim=channels[-1])

    def forward(self, lr_img, x_t, t):
        # Concatenate noisy image and LR condition
        x_in = torch.cat([x_t, lr_img], dim=1)
        
        # 1. Standard UNet forward pass (Base structure)
        # Note: return_dict=False is used for compatibility across diffusers versions
        unet_out = self.unet(x_in, timestep=t.float(), return_dict=False)[0]
        
        # 2. DWT High-Frequency Injection
        # We apply DWT to the bottleneck features to find sub-pixel priors
        dwt_feats = dwt_init(unet_out)
        dwt_projected = self.dwt_projection(dwt_feats)
        
        # 3. Apply Hierarchical Mamba Bottleneck
        # Fusing standard UNet features with DWT-aware features
        refined = self.hi_mamba_bottleneck(unet_out + dwt_projected)
        
        return refined

class UNet(nn.Module):
    """
    Main entry point for scripts/model_config_imports.py.
    """
    def __init__(self, cfg_unet):
        super().__init__()
        # Extract channels from Hydra config or use defaults
        if isinstance(cfg_unet, list):
            channels = cfg_unet
        elif hasattr(cfg_unet, 'channels'):
            channels = list(cfg_unet.channels)
        else:
            channels = [64, 96, 128, 256]
            
        self.model = HybridUNet(channels=channels)

    def forward(self, lr_img, x_t, t):
        return self.model(lr_img, x_t, t)