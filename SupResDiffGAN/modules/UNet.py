import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.unets import UNet2DModel
from mamba_ssm import Mamba

class SGFN(nn.Module):
    """
    Spatial Gated Feed-Forward Network.
    Optimized for spatial awareness in high-resolution reconstruction.
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
        # Spatial gating mechanism
        return self.conv2(x1 * torch.sigmoid(x2))

class MHSS_SSSM_Block(nn.Module):
    """
    Stabilized Multi-Head Selective Scan & Scalable State Space Model.
    Processes features bi-directionally to capture global latent context.
    """
    def __init__(self, dim, scale_factor=4):
        super().__init__()
        self.dim = dim
        self.mamba_heads = nn.ModuleList([
            Mamba(d_model=dim, d_state=16, d_conv=4, expand=1) for _ in range(2)
        ])
        self.scale_modulator = nn.Parameter(torch.ones(1) * (1.0 / scale_factor))

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).view(B, -1, C)
        
        # Bi-directional selective scan
        out = self.mamba_heads[0](x_flat)
        out += self.mamba_heads[1](x_flat.flip(dims=[1])).flip(dims=[1])
        
        return (out.view(B, H, W, C).permute(0, 3, 1, 2)) * self.scale_modulator

class HiMambaBlock(nn.Module):
    """
    Hierarchical Mamba Refiner.
    Combines MHSS global scanning with SGFN spatial gating.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.region_ssm = MHSS_SSSM_Block(dim)
        self.norm = nn.LayerNorm(dim)
        self.sgfn = SGFN(dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        shortcut = x
        # Apply LayerNorm on the channel dimension
        x_norm = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Mamba Global Scan
        x_mamba = self.region_ssm(x_norm)
        
        # Spatial Gated Refinement
        x_refined = x_mamba + self.sgfn(x_mamba)
        
        return shortcut + self.gamma * x_refined

class HybridUNet(nn.Module):
    """
    Hybrid UNet that injects Hi-Mamba logic into the UNet2DModel bottleneck.
    """
    def __init__(self, channels: list[int] = [64, 96, 128, 256]):
        super().__init__()
        self.unet = UNet2DModel(
            in_channels=8,
            out_channels=4,
            block_out_channels=channels,
            layers_per_block=2,
            add_attention=[False, False, True, True],
        )
        
        # Initialize Mamba refiner for the bottleneck (256 channels)
        self.mamba_refiner = HiMambaBlock(dim=channels[-1])
        
        # --- Bottleneck Injection Logic ---
        # We wrap the internal mid_block of the UNet to include Mamba
        original_mid_block = self.unet.mid_block
        
        class WrappedMidBlock(nn.Module):
            def __init__(self, mid_block, mamba):
                super().__init__()
                self.mid_block = mid_block
                self.mamba = mamba
            def forward(self, *args, **kwargs):
                # Run the original mid_block (bottleneck)
                hidden_states = self.mid_block(*args, **kwargs)
                # Apply Mamba global refinement
                return self.mamba(hidden_states)
                
        self.unet.mid_block = WrappedMidBlock(original_mid_block, self.mamba_refiner)

        self.unet.enable_gradient_checkpointing()

    def forward(self, lr_img, x_t, t):
        # Concatenate noisy image and condition
        x_in = torch.cat([x_t, lr_img], dim=1)
        # UNet now runs with Mamba inside its bottleneck!
        return self.unet(x_in, timestep=t.float()).sample

class UNet(nn.Module):
    """
    Entry point for the project configuration.
    """
    def __init__(self, cfg_unet):
        super().__init__()
        if isinstance(cfg_unet, list):
            channels = cfg_unet
        elif hasattr(cfg_unet, 'channels'):
            channels = list(cfg_unet.channels)
        else:
            channels = [64, 96, 128, 256]
        self.model = HybridUNet(channels=channels)

    def forward(self, lr_img, x_t, t):
        return self.model(lr_img, x_t, t)