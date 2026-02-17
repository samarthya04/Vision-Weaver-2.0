# In scripts/model_config_imports.py

"""File containing imports for SupResDiffGAN models only."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from diffusers import AutoencoderKL, AutoencoderTiny # Add AutoencoderTiny

# SupResDiffGAN imports only
from SupResDiffGAN.modules.Diffusion import Diffusion as Diffusion_supresdiffgan
from SupResDiffGAN.modules.Discriminator import (
    Discriminator as Discriminator_supresdiffgan,
)
from SupResDiffGAN.modules.FeatureExtractor import (
    FeatureExtractor as FeatureExtractor_supresdiffgan,
)
from SupResDiffGAN.modules.UNet import UNet as UNet_supresdiffgan
from SupResDiffGAN.modules.VggLoss import VGGLoss as VGGLoss_supresdiffgan
from SupResDiffGAN.SupResDiffGAN import SupResDiffGAN
from SupResDiffGAN.SupResDiffGAN_simple_gan import SupResDiffGAN_simple_gan
from SupResDiffGAN.SupResDiffGAN_without_adv import SupResDiffGAN_without_adv