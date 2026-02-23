# Hi-MambaSR: Hierarchical State-Space Refinement for Latent Diffusion

<div align="center">
  <p><strong>Fusing Latent Diffusion, Generative Adversarial Networks (GANs), Swin Transformers, and Mamba State-Space Models for High-Fidelity Super-Resolution on 6GB Consumer GPUs.</strong></p>
</div>

<div align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c?logo=pytorch" alt="PyTorch"></a>
  <a href="https://lightning.ai/docs/pytorch/stable/"><img src="https://img.shields.io/badge/Lightning-2.2%2B-792ee5?logo=pytorch-lightning" alt="PyTorch Lightning"></a>
  <a href="https://hydra.cc/"><img src="https://img.shields.io/badge/Config-Hydra-89b8cd" alt="Hydra"></a>
  <a href="https://wandb.ai/"><img src="https://img.shields.io/badge/Logged-W%26B-yellowgreen" alt="Weights & Biases"></a>
</div>

**Hi-MambaSR** is a cutting-edge super-resolution architecture that implements a hybrid approach combining **Latent Denoising Diffusion Models**, **Relativistic GANs**, **Swin Transformers**, and **Mamba State-Space Models (SSMs)**. By performing diffusion in the latent space of a pre-trained autoencoder, injecting precise localized pixel-attention via Swin Transformers, and providing linear-time global context via Mamba blocks, the model achieves unprecedented perceptual quality and structural realism.

Crucially, this entire framework has been heavily mathematically and structurally optimized to run end-to-end training and inference on **6GB VRAM Consumer GPUs**.

---

## 🌟 Key Research Features & Innovations

- **6GB VRAM Hardened Pipeline**: 
  - Reduced physical batch sizes (4) synthesized with high gradient accumulation (16) to fit massive architectures on small GPUs without unstable math.
  - Generative PyTorch gradient checkpointing injected into colossal VGG feature extraction layers to prevent Out-Of-Memory (OOM) crashes during hierarchical perceptual loss calculations.
- **Mamba State-Space Stability Engine**: 
  - Substituted native `LayerNorm` with custom **RMSNorm** to stabilize Mamba mixed-precision variables.
  - Implemented strict 1.0 Global Gradient Norm Clipping and a Linear Learning Rate Warmup scheduler (10% epoch phase) to prevent state-matrix exploding gradients.
- **Pixel-Space Residual Skip**: Bypass the Autoencoder (VAE) blurring bottleneck entirely via bicubic upsampling + rendering VAE decodings as pure high-frequency residuals.
- **Small-Batch GAN Physics**: Replaced `BatchNorm` with `InstanceNorm` to prevent batch statistics from collapsing during 4-image micro-batch processing, stabilized by default Spectral Normalization.
- **Dynamic Fast-Sampling Math**: Corrected continuous 1000-step DDPM to 20-step DDIM subsampling to guarantee the neural network interpolates over valid `alpha_bar` trajectories.

---

## 📊 Evaluation & Benchmarking

The architecture leverages automated testing pipelines that scrub PyTorch Lightning logs, generate CSV analytics, and natively port qualitative results (such as inference speed vs. PSNR charts) directly into Weights & Biases dashboards.

> Example benchmarks will be populated in `evaluation_results/hi_mambasr_benchmarks.csv` after executing the testing suite.

---

## ⚙️ Getting Started

### Prerequisites
- **Python**: `3.9+` (tested on `3.10`)
- **PyTorch 2.0+** (Required for `torch.compile` speed enhancements)
- **NVIDIA GPU** (Minimum 6GB VRAM)

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/samarthya04/Hi-MambaSR.git
cd Hi-MambaSR

# 2. Create and activate conda environment
conda create -n mamba_sr python=3.10
conda activate mamba_sr

# 3. Install dependencies
pip install -r requirements-data.txt
pip install -r requirements-gpu.txt

# 4. Login to Weights & Biases
wandb login
```

---

## 💿 Datasets

This project uses **CelebA-HQ** by default. To automate the download and processing:

```bash
bash get_data.sh -c
```

This script will:
1. Interface with Kaggle to download the HQ datasets.
2. Unzip and properly split into `train`/`val`/`test` subdirectories.
3. Automatically generate the target 4x Bicubic down-sampled **LR images**.

---

## 🚀 Usage

### Training / Fine-tuning

The architecture leverages Hydra for extreme declarative configuration management. The default 6GB VRAM-optimized YAML is `config_mamba`.

```bash
python train_model.py -cn config_mamba
```

#### To **resume** training or **fine-tune**:
1. Open `conf/config_mamba.yaml`:
   ```yaml
   mode: 'train'
   trainer:
     resume_from_checkpoint: "models/checkpoints/Hi-MambaSR-best.ckpt"
   ```
2. Adjust `model.lr` (e.g., lower it to `1e-5` for advanced fine-tuning).
3. Execute the script. The system automatically supports resuming from older ablation targets (via `strict=False` loading).

---

### Inference & Benchmarking

```bash
python evaluate_model.py -cn config_mamba
```

#### Before running:
1. Open `conf/config_mamba.yaml` and configure the multi-step testing suite:
   ```yaml
   model:
     load_model: "models/checkpoints/<your_best_checkpoint>.ckpt"
   evaluation:
     steps: [25, 50]    # Will evaluate at both step configurations
     posteriors: ["ddim", "ddpm"] 
     results_file: "evaluation_results/hi_mambasr_benchmarks.csv"
   ```

---

## 🏗️ Architecture Stack

| Component | Description |
|-----------|-------------|
| **Autoencoder** | `AutoencoderTiny`/`AutoencoderKL` projecting RGB into a compressed manifold, augmented with a pixel-space residual connection. |
| **U-Net Generator** | Hybrid UNet mixing localized **Swin Transformer V2 Attention Arrays** (for high-frequency texture precision) with global linear-time **MultiHead Selective Scan (Mamba)** bottlenecks (for wide-range structural coherence). |
| **Discriminator** | Relativistic Patch-GAN running `InstanceNorm` + `SpectralNorm` driven by `BCEWithLogitsLoss`. |
| **Perceptual Loss** | Memory checkpointing Multi-scale VGG19 feature extraction (forced to fp32 calculation to prevent NaNs). |

See:
- `Hi-MambaSR/HiMambaSR.py`
- `Hi-MambaSR/modules/UNet.py`
- `Hi-MambaSR/modules/Discriminator.py`
- `Hi-MambaSR/modules/Diffusion.py`

---

## 📜 Acknowledgements

This architecture builds upon foundations provided by:
- **Swin Transformers**: Liu et al.
- **State Space Models (Mamba)**: Gu & Dao
- **Stable Diffusion (Latent Physics)**: Rombach et al. 
- Original Hybrid PyTorch architectures such as SupResDiffGAN/Real-ESRGAN.

---

## ⚖️ License

Currently Unlicensed. Please adhere to the proprietary distribution constraints of the foundational codebases utilized within (such as `diffusers`, `mamba_ssm`, and `torchvision`).
