"""Key Publication-Level Enhancements:
nn.ModuleDict Integration: Replaced the standard Python dictionary with nn.ModuleDict. This ensures that when you call .to(device) or .cuda() on the FeatureExtractor, all the VGG layers are moved correctly without manual iteration.

register_buffer for Normalization: Using register_buffer for mean and std is the best practice. It ensures these constants are part of the model state (saved in .ckpt) but are correctly excluded from the optimizer's parameter list.

Modern Weights API: Updated to VGG19_Weights.IMAGENET1K_V1 to comply with the latest torchvision standards, replacing the deprecated pretrained=True.

Implicit Eval Enforcement: Overrode the train() method. Perceptual extractors should never be in training mode (which would affect Dropout or BatchNorm behavior); this safeguard ensures consistency regardless of the LightningModule state.

Optimized Forward Pass: The previous implementation re-processed the images from the start in every loop iteration. This version uses a sequential forward pass, where each block builds on the output of the previous one, significantly reducing redundant computations and GPU memory overhead.
"""

import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from torch.utils.checkpoint import checkpoint
from typing import Dict

class FeatureExtractor(nn.Module):
    """
    Perceptual Feature Extractor based on VGG-19.
    
    This module computes multi-scale perceptual loss by extracting intermediate 
    feature maps from a pre-trained VGG-19 network. It is designed to guide 
    Hi-MambaSR toward natural image manifolds by minimizing distance in 
    semantic feature space.
    """

    def __init__(self) -> None:
        super(FeatureExtractor, self).__init__()
        
        # Load pre-trained VGG19 features with modern Weights API
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        
        # Define hierarchical feature blocks (Conv1_2, Conv2_2, Conv3_4, Conv4_4, Conv5_4)
        # These layers are standard for perceptual loss in super-resolution research
        self.blocks = nn.ModuleDict({
            "block1": nn.Sequential(*list(vgg.children())[:4]),
            "block2": nn.Sequential(*list(vgg.children())[4:9]),
            "block3": nn.Sequential(*list(vgg.children())[9:18]),
            "block4": nn.Sequential(*list(vgg.children())[18:27]),
            "block5": nn.Sequential(*list(vgg.children())[27:36]),
        })

        # Register ImageNet normalization constants as buffers (non-trainable)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Criterion: L1 is typically preferred for perceptual loss to ensure sharpness
        self.criterion = nn.L1Loss()

        # Weighted contribution for each hierarchical scale
        self.feature_weights = {
            "block1": 0.1,
            "block2": 0.1,
            "block3": 1.0,
            "block4": 1.0,
            "block5": 1.0,
        }

        # Freeze all layers to prevent weights from drifting during training
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, sr_img: torch.Tensor, hr_img: torch.Tensor) -> torch.Tensor:
        """
        Computes the multi-scale perceptual loss between SR and HR images.

        Args:
            sr_img (torch.Tensor): Super-resolved image in range [-1, 1].
            hr_img (torch.Tensor): Ground-truth image in range [-1, 1].

        Returns:
            torch.Tensor: Scalar perceptual loss.
        """
        # Use the modern torch.amp API to enforce fp32 computation
        with torch.amp.autocast('cuda', enabled=False):
            # Efficient range normalization from [-1, 1] to [0, 1]
            sr = (sr_img.float() + 1.0) * 0.5
            hr = (hr_img.float() + 1.0) * 0.5

            # Apply ImageNet standardization
            sr = (sr - self.mean.to(sr.device, dtype=sr.dtype)) / self.std.to(sr.device, dtype=sr.dtype)
            hr = (hr - self.mean.to(hr.device, dtype=hr.dtype)) / self.std.to(hr.device, dtype=hr.dtype)

            perceptual_loss = 0.0
            curr_sr, curr_hr = sr, hr
            
            for name, block in self.blocks.items():
                # Use gradient checkpointing for large tensors to prevent 6GB VRAM OOM
                if curr_sr.shape[-1] >= 256:
                    curr_sr = checkpoint(block, curr_sr, use_reentrant=False)
                    curr_hr = checkpoint(block, curr_hr, use_reentrant=False)
                else:
                    curr_sr = block(curr_sr)
                    curr_hr = block(curr_hr)
                
                # Weighted L1 distance at the current hierarchical depth
                perceptual_loss += self.feature_weights[name] * self.criterion(curr_sr, curr_hr)

        return perceptual_loss

    def train(self, mode: bool = True):
        """Override train mode to ensure the extractor always stays in eval mode."""
        super().train(False)
        return self