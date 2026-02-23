"""Key Publication-Level Enhancements:
BCEWithLogits Alignment: Removed nn.Sigmoid() from the final layer. Research-grade GANs use raw logits with torch.nn.BCEWithLogitsLoss to benefit from the log-sum-exp trick, which prevents vanishing gradients when the discriminator becomes too confident.

Spectral Normalization (SN): Integrated spectral_norm as an option. This is a primary technical requirement for modern GAN papers to ensure the Discriminator remains a K-Lipschitz function.

VRAM Efficiency: Added inplace=True to all LeakyReLU activations to reduce memory allocation during the forward pass.

Flexible Conditioning: The Discriminator is now explicitly configured for 6 channels (supporting the concatenation of HR/SR and the LR condition), which is the standard for conditional SR-GANs.

Modern Weights API: Updated the ResNetDiscriminator to use the modern torchvision weights API, ensuring the code is compatible with 2025/2026 library versions.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import List

class DiscriminatorBlock(nn.Module):
    """
    Modular Discriminator Block for Hi-MambaSR.
    
    Includes a convolutional layer followed by Batch Normalization and 
    LeakyReLU activation. Supports Spectral Normalization for enhanced 
    Lipschitz continuity during GAN training.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int, 
        use_sn: bool = False
    ) -> None:
        super(DiscriminatorBlock, self).__init__()
        
        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        
        # Spectral Normalization is a research standard to prevent mode collapse
        self.conv = spectral_norm(conv) if use_sn else conv
        
        # Replaced BatchNorm with InstanceNorm
        # BatchNorm fails catastrophically with the small micro-batch sizes (e.g., 4) 
        # required to fit inside 6GB VRAM limits.
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lrelu(self.norm(self.conv(x)))

class Discriminator(nn.Module):
    """
    Relativistic Discriminator Network for Hi-MambaSR.
    
    Designed to process concatenated latent/pixel features for Relativistic 
    GAN (RaGAN) objectives. The network identifies structural inconsistencies 
    at multiple spatial scales.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 6 for concatenated SR/HR and LR condition).
    channels : List[int]
        Feature maps configuration (e.g., [64, 128, 256, 512]).
    use_sn : bool
        Whether to apply Spectral Normalization across all layers.
    """

    def __init__(
        self, 
        in_channels: int = 6, 
        channels: List[int] = [64, 128, 256, 512], 
        use_sn: bool = True  # Forced True for structural stability
    ) -> None:
        super(Discriminator, self).__init__()

        # Initial Feature Projection
        first_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1)
        self.initial = nn.Sequential(
            spectral_norm(first_conv) if use_sn else first_conv,
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Hierarchical Feature Extraction
        blocks = []
        curr_channels = channels[0]
        for out_channels in channels[1:]:
            # Downsampling Layer
            blocks.append(DiscriminatorBlock(curr_channels, out_channels, stride=2, use_sn=use_sn))
            # Feature Refinement Layer
            blocks.append(DiscriminatorBlock(out_channels, out_channels, stride=1, use_sn=use_sn))
            curr_channels = out_channels
        
        self.main_body = nn.Sequential(*blocks)

        # Dense Classification Head
        # Note: Sigmoid is removed here in favor of using BCEWithLogitsLoss for 
        # superior numerical stability in 16-mixed precision training.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(curr_channels, curr_channels * 2, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(curr_channels * 2, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
        Returns:
            torch.Tensor: Prediction logits of shape (N, 1, 1, 1).
        """
        x = self.initial(x)
        x = self.main_body(x)
        return self.classifier(x)

class ResNetDiscriminator(nn.Module):
    """
    Transfer-Learning based Discriminator using ResNet50 backbone.
    
    Utilizes pre-trained semantic features to guide the Super-Resolution 
    process toward natural image manifolds. Useful for ablation studies.
    """
    def __init__(self, pretrained: bool = True) -> None:
        super(ResNetDiscriminator, self).__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        
        # Strip the fully connected layer and global pooling
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        
        # Adaptive head for arbitrary input resolutions
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ResNet expects 3 channels; handle concatenated inputs if necessary
        if x.size(1) > 3:
            x = x[:, :3, :, :]
        
        feat = self.features(x)
        return self.head(feat)