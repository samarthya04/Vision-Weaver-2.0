# In scripts/model_config.py

from .model_config_imports import *
from diffusers import AutoencoderTiny 
import os
import torch
from SupResDiffGAN.modules.UNet import HybridUNet as UNet_hybrid


def model_selection(cfg, device):
    """Select and initialize model based on configuration."""
    if cfg.model.name == "SupResDiffGAN":
        return initialize_supresdiffgan(
            cfg, device, SupResDiffGAN, use_discriminator=True
        )

    elif cfg.model.name == "SupResDiffGAN_Hybrid":
        return initialize_supresdiffgan(
            cfg, device, SupResDiffGAN, use_discriminator=True, use_hybrid=True
        )

    elif cfg.model.name == "SupResDiffGAN_without_adv":
        return initialize_supresdiffgan(
            cfg, device, SupResDiffGAN_without_adv, use_discriminator=False
        )

    else:
        raise ValueError(f"Model '{cfg.model.name}' not supported.")


def initialize_supresdiffgan(cfg, device, model_class, use_discriminator=True, use_hybrid=False):
    """Helper to initialize variants including the hybrid architecture."""
    if cfg.autoencoder == "VAE":
        model_id = "madebyollin/taesd"
        autoencoder = AutoencoderTiny.from_pretrained(model_id).to(device)

    discriminator = (
        Discriminator_supresdiffgan(
            in_channels=cfg.discriminator.in_channels,
            channels=cfg.discriminator.channels,
        ) if use_discriminator else None
    )

    # Instantiate the requested UNet architecture
    if use_hybrid:
        unet = UNet_hybrid(channels=cfg.unet)
    else:
        unet = UNet_supresdiffgan(cfg.unet)

    diffusion = Diffusion_supresdiffgan(
        timesteps=cfg.diffusion.timesteps,
        beta_type=cfg.diffusion.beta_type,
        posterior_type=cfg.diffusion.posterior_type,
    )

    # Initialize loss modules
    vgg_loss = None
    if cfg.use_perceptual_loss:
        vgg_loss = FeatureExtractor_supresdiffgan(device) if cfg.feature_extractor else VGGLoss_supresdiffgan(device)

    model = model_class(
        ae=autoencoder,
        discriminator=discriminator,
        unet=unet,
        diffusion=diffusion,
        learning_rate=cfg.model.lr,
        alfa_perceptual=cfg.model.alfa_perceptual,
        alfa_adv=cfg.model.alfa_adv,
        vgg_loss=vgg_loss,
    )

    # Checkpoint loading logic
    if cfg.model.load_model:
        model.load_state_dict(torch.load(cfg.model.load_model, map_location=device))

    return model