"""
Hi-MambaSR Model Configuration Imports.
Consolidates all architectural components for hierarchical State-Space refinement.
Handles path resolution for the updated 'Hi-MambaSR' directory structure.
"""

import os
import sys
import torch
from diffusers import AutoencoderKL, AutoencoderTiny

# ==============================================================================
# PATH RESOLUTION LOGIC
# ==============================================================================
# 1. Get the absolute path of the directory containing 'train_model.py'
current_script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_script_dir, ".."))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 2. Append the internal 'Hi-MambaSR' package directory
internal_pkg_dir = os.path.join(root_dir, "Hi-MambaSR")
if internal_pkg_dir not in sys.path:
    sys.path.insert(0, internal_pkg_dir)

# ==============================================================================
# COMPONENT IMPORTS
# ==============================================================================
try:
    # Hi-MambaSR Core Engine Imports (from internal_pkg_dir/modules)
    from modules.Diffusion import Diffusion as Diffusion_engine
    from modules.Discriminator import Discriminator as Discriminator_engine
    from modules.FeatureExtractor import FeatureExtractor as FeatureExtractor_engine
    from modules.UNet import UNet as HybridUNet_backbone
    from modules.VggLoss import VGGLoss as VGGLoss_engine

    # Main Research Class (from internal_pkg_dir/HiMambaSR.py)
    from HiMambaSR import HiMambaSR

except ImportError as e:
    print(f"Critical Import Error: {e}")
    print("Check if filenames match HiMambaSR.py and the 'modules' folder exists.")
    raise

# Legacy/Ablation Variants
try:
    from SupResDiffGAN_simple_gan import SupResDiffGAN_simple_gan
    from SupResDiffGAN_without_adv import SupResDiffGAN_without_adv
except ImportError:
    SupResDiffGAN_simple_gan = None
    SupResDiffGAN_without_adv = None

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def get_vae(vae_type: str = "VAE", device: str = "cuda"):
    """
    Utility to load the latent manifold projector.
    Ensures the standard scaling factor (0.18215) is registered via the config object.
    """
    try:
        if vae_type == "TinyVAE":
            # TAESD for high-speed research iterations
            vae = AutoencoderTiny.from_pretrained("madebyollin/taesd")
        else:
            # StabilityAI's MSE-tuned VAE for maximum fidelity
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        
        # ----------------------------------------------------------------------
        # PUBLICATION-CRITICAL: Scaling Factor Calibration (Config-Aware)
        # ----------------------------------------------------------------------
        # Accessing via .config suppresses the FutureWarning and ensures 
        # consistency with the HiMambaSR main module.
        if not hasattr(vae.config, "scaling_factor") or vae.config.scaling_factor is None:
            # Manually injecting for VAEs that do not include it in metadata
            vae.config.scaling_factor = 0.18215
            
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
            
        return vae
    except Exception as e:
        print(f"Error loading VAE ({vae_type}): {e}")
        raise