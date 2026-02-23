import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class HiMambaSR(pl.LightningModule):
    """
    Hi-MambaSR: Hierarchical State-Space Refinement for Latent Diffusion Super-Resolution.
    Hardened for 6GB VRAM with Latent Regularization and Micro-Batch Decoding.
    """

    def __init__(
        self,
        ae: nn.Module,
        discriminator: nn.Module,
        unet: nn.Module,
        diffusion: nn.Module,
        learning_rate: float = 1e-4,
        alfa_perceptual: float = 2e-2, # Increased to 0.02 for structural anchoring
        alfa_adv: float = 5e-3,
        vgg_loss: nn.Module | None = None,
    ) -> None:
        super(HiMambaSR, self).__init__()
        self.save_hyperparameters(ignore=["ae", "discriminator", "unet", "diffusion", "vgg_loss"])
        
        self.ae = ae
        self.discriminator = discriminator
        self.generator = unet
        self.diffusion = diffusion
        self.vgg_loss = vgg_loss

        self.content_loss = nn.MSELoss()
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)

        self.lr = learning_rate
        self.alfa_adv = alfa_adv
        self.alfa_perceptual = alfa_perceptual
        self.betas = (0.9, 0.999)

        # Freeze VAE, unfreeze decoder for high-frequency fine-tuning
        for param in self.ae.parameters():
            param.requires_grad = False
        if hasattr(self.ae, 'decoder'):
            for param in self.ae.decoder.parameters():
                param.requires_grad = True

        # Compile U-Net/Mamba Generator for Faster Inference
        if hasattr(torch, 'compile'):
            self.generator = torch.compile(self.generator)

        self.automatic_optimization = False
        self.test_step_outputs = []
        
        # Noise difficulty progression (EMA)
        self.ema_weight = 0.99
        self.ema_mean = 0.5
        self.s = 0

        # VAE VRAM Optimizations
        self.ae.enable_slicing()
        self.ae.enable_tiling()
        
        self._register_sobel_kernels()

    def _register_sobel_kernels(self):
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_kx', kx)
        self.register_buffer('sobel_ky', ky)

    def normalize_for_lpips(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x.float() + 1) / 2, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference loop: Encode -> Refine -> Decode."""
        with torch.no_grad():
            posterior = self.ae.encode(x).latent_dist
            x_lat = posterior.mode() * self.ae.config.scaling_factor
            
        x_gen = self.diffusion.sample(self.generator, x_lat, x_lat.shape)
        
        with torch.no_grad():
            # Force fp32 during decoding to prevent NaNs/black images in mixed precision
            x_out = self.ae.decode(x_gen.to(torch.float32) / self.ae.config.scaling_factor).sample.to(x_lat.dtype)
            
        # High-Frequency Skip Connection (Pixel-Space Residual)
        x_upsampled = F.interpolate(x, size=x_out.shape[-2:], mode='bicubic', align_corners=False)
        return torch.clamp(x_out + x_upsampled, -1, 1)

    def micro_batch_decode(self, latent_tensor: torch.Tensor, micro_batch_size: int = 4) -> torch.Tensor:
        """Sequential decoding to handle batch size 16 on limited VRAM."""
        scale = self.ae.config.scaling_factor
        decoded_chunks = []
        for i in range(0, latent_tensor.shape[0], micro_batch_size):
            chunk = latent_tensor[i : i + micro_batch_size]
            with torch.no_grad():
                # Force fp32 during decoding to prevent NaNs/black images in mixed precision
                decoded = self.ae.decode(chunk.to(torch.float32) / scale).sample.to(latent_tensor.dtype)
            decoded_chunks.append(decoded)
        return torch.cat(decoded_chunks, dim=0)

    def calculate_edge_loss(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        def get_gradients(x):
            x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')
            c = x.size(1)
            gx = F.conv2d(x_padded, self.sobel_kx.repeat(c, 1, 1, 1), groups=c)
            gy = F.conv2d(x_padded, self.sobel_ky.repeat(c, 1, 1, 1), groups=c)
            return torch.sqrt(gx**2 + gy**2 + 1e-8)
        return self.content_loss(get_gradients(sr), get_gradients(hr))

    def training_step(self, batch: dict, batch_idx: int):
        lr_img, hr_img = batch["lr"], batch["hr"]
        optimizer_g, optimizer_d = self.optimizers()
        scale = self.ae.config.scaling_factor

        # 1. Feature Extraction (Latent Space)
        with torch.no_grad():
            lr_lat = self.ae.encode(lr_img).latent_dist.sample().detach() * scale
            x0_lat = self.ae.encode(hr_img).latent_dist.sample().detach() * scale

        # 2. Diffusion logic
        t = torch.randint(0, self.diffusion.timesteps, (x0_lat.shape[0],), device=self.device)
        x_t = self.diffusion.forward(x0_lat, t)
        alfa_bars = self.diffusion.alpha_bars_torch.to(self.device)[t]
        
        # 3. Generator Core Pass
        x_gen_0 = self.generator(lr_lat, x_t, alfa_bars)

        if batch_idx % 2 == 0:
            # --- GENERATOR UPDATE ---
            self.toggle_optimizer(optimizer_g)
            optimizer_g.zero_grad(set_to_none=True)
            
            l_mse = self.content_loss(x_gen_0, x0_lat)
            
            # --- LATENT REGULARIZATION ---
            # Prevents latent drift that causes hallucinations/artifacts
            l_lat_reg = torch.mean(torch.pow(x_gen_0, 2)) 
            
            # Micro-Batch Decode for Perceptual Loss (with Skip Connection)
            sr_decoded = self.micro_batch_decode(x_gen_0)
            lr_upsampled = F.interpolate(lr_img, size=sr_decoded.shape[-2:], mode='bicubic', align_corners=False)
            sr_img = torch.clamp(sr_decoded + lr_upsampled, -1, 1)
            
            l_per = self.lpips(self.normalize_for_lpips(sr_img), self.normalize_for_lpips(hr_img))
            l_edge = self.calculate_edge_loss(sr_img, hr_img)
            if self.vgg_loss: l_per += self.vgg_loss(sr_img, hr_img)

            # Relativistic GAN logic with residual targeting
            s_tensor = torch.full((x0_lat.shape[0],), self.s, device=self.device, dtype=torch.long)
            
            sr_s_decoded = self.micro_batch_decode(self.diffusion.forward(x_gen_0, s_tensor))
            sr_s_img = torch.clamp(sr_s_decoded + lr_upsampled, -1, 1)
            
            hr_s_img = torch.clamp(self.micro_batch_decode(self.diffusion.forward(x0_lat, s_tensor).detach()), -1, 1)
            
            y = torch.randint(0, 2, (hr_s_img.shape[0],), device=self.device).view(-1, 1, 1, 1)
            first = torch.where(y == 0, sr_s_img, hr_s_img)
            second = torch.where(y == 0, hr_s_img, sr_s_img)
            
            prediction = self.discriminator(torch.cat([first, second], dim=1))
            l_adv = self.adversarial_loss(prediction, 1.0 - y.float())

            # Composite Loss (Increased Structural Weights)
            g_loss = l_mse + (self.hparams.alfa_perceptual * l_per) + \
                     (self.hparams.alfa_adv * l_adv) + (0.1 * l_edge) + (1e-3 * l_lat_reg)
            
            self.manual_backward(g_loss)
            # Manual Gradient Clipping: Crucial for protecting Mamba State-Matrix from explosions
            self.clip_gradients(optimizer_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)
            
            self.log_dict({
                "train/g_loss": g_loss, 
                "train/l_lat_reg": l_lat_reg,
                "train/g_lpips": l_per
            }, prog_bar=True, batch_size=lr_img.shape[0])
            self.calculate_ema_noise_step(prediction.detach(), 1.0 - y.float())

        else:
            # --- DISCRIMINATOR UPDATE ---
            self.toggle_optimizer(optimizer_d)
            optimizer_d.zero_grad(set_to_none=True)
            
            with torch.no_grad():
                s_tensor = torch.full((x0_lat.shape[0],), self.s, device=self.device, dtype=torch.long)
                
                sr_s_decoded = self.micro_batch_decode(self.diffusion.forward(x_gen_0.detach(), s_tensor))
                lr_upsampled = F.interpolate(lr_img, size=sr_s_decoded.shape[-2:], mode='bicubic', align_corners=False)
                sr_s_img = torch.clamp(sr_s_decoded + lr_upsampled, -1, 1)
                
                hr_s_img = torch.clamp(self.micro_batch_decode(self.diffusion.forward(x0_lat, s_tensor)), -1, 1)
            
            y = torch.randint(0, 2, (hr_s_img.shape[0],), device=self.device).view(-1, 1, 1, 1)
            first = torch.where(y == 0, hr_s_img, sr_s_img)
            second = torch.where(y == 0, sr_s_img, hr_s_img)
            
            prediction = self.discriminator(torch.cat([first, second], dim=1))
            d_loss = self.adversarial_loss(prediction, y.float())
            
            self.manual_backward(d_loss)
            # Manual Gradient Clipping for Discriminator stability
            self.clip_gradients(optimizer_d, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            optimizer_d.step()
            self.untoggle_optimizer(optimizer_d)
            self.log("train/d_loss", d_loss, prog_bar=True, batch_size=lr_img.shape[0])

        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

    def calculate_ema_noise_step(self, pred: torch.Tensor, y: torch.Tensor) -> None:
        acc = ((torch.sigmoid(pred) > 0.5).float() == y).float().mean().item()
        self.ema_mean = acc * (1 - self.ema_weight) + self.ema_mean * self.ema_weight
        self.s = int(np.clip((self.ema_mean - 0.5) * 2 * self.diffusion.timesteps, 0, self.diffusion.timesteps - 1))
        self.log_dict({"train/ema_s": self.s, "train/ema_acc": acc}, on_epoch=True, batch_size=pred.shape[0])

    def configure_optimizers(self):
        # Include ae.decoder in generator optimizer if unfrozen
        gen_params = list(self.generator.parameters())
        if hasattr(self.ae, 'decoder') and any(p.requires_grad for p in self.ae.decoder.parameters()):
            gen_params += list(self.ae.decoder.parameters())
            
        opt_g = torch.optim.Adam(gen_params, lr=self.lr, betas=self.betas, weight_decay=1e-6)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas, weight_decay=1e-6)
        
        t_max = self.trainer.max_epochs 
        warmup_epochs = max(1, t_max // 10) # 10% Warming up
        
        warmup_g = torch.optim.lr_scheduler.LinearLR(opt_g, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=t_max - warmup_epochs, eta_min=1e-7)
        sch_g = torch.optim.lr_scheduler.SequentialLR(opt_g, schedulers=[warmup_g, cosine_g], milestones=[warmup_epochs])

        warmup_d = torch.optim.lr_scheduler.LinearLR(opt_d, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=t_max - warmup_epochs, eta_min=1e-7)
        sch_d = torch.optim.lr_scheduler.SequentialLR(opt_d, schedulers=[warmup_d, cosine_d], milestones=[warmup_epochs])

        return [opt_g, opt_d], [sch_g, sch_d]

    def on_train_epoch_end(self):
        sch_g, sch_d = self.lr_schedulers()
        self.log("lr/gen", sch_g.get_last_lr()[0], on_epoch=True, prog_bar=True)
        sch_g.step()
        sch_d.step()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        lr_img, hr_img = batch["lr"], batch["hr"]
        sr_img = self(lr_img)
        padding_info = {"lr": batch["padding_data_lr"], "hr": batch["padding_data_hr"]}

        # Cast to float32 for metric calculations
        hr_np = hr_img.float().cpu().numpy()
        sr_np = sr_img.float().cpu().numpy()

        val_psnr = peak_signal_noise_ratio(hr_np, sr_np, data_range=2.0)
        val_lpips = self.lpips(self.normalize_for_lpips(hr_img), self.normalize_for_lpips(sr_img)).cpu().item()
        
        if batch_idx == 0:
            per_metrics = [(val_psnr, 0, val_lpips)]
            img_grid = self.plot_images_with_metrics(hr_img.float(), lr_img.float(), sr_img.float(), padding_info, 
                                                   f"Hi-MambaSR | Epoch {self.current_epoch}", per_metrics)
            self.logger.experiment.log({"val_samples": wandb.Image(img_grid)})

        self.log_dict({"val/PSNR": val_psnr, "val/LPIPS": val_lpips}, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=hr_img.shape[0])

    def plot_images_with_metrics(self, hr_img, lr_img, sr_img, padding_info, title, per_image_metrics) -> np.ndarray:
        fig, axs = plt.subplots(3, 3, figsize=(10, 10), dpi=120)
        fig.patch.set_facecolor("#121212")
        for i in range(min(3, hr_img.shape[0])):
            img_sets = [(hr_img[i], "Ground Truth", axs[0, i]), 
                        (lr_img[i], "Bicubic Input", axs[1, i]), 
                        (sr_img[i], "Hi-MambaSR Prediction", axs[2, i])]
            for img_t, lbl, ax in img_sets:
                img_np = np.clip((img_t.detach().cpu().float().numpy().transpose(1, 2, 0) + 1) / 2, 0, 1)
                ax.imshow(img_np)
                ax.set_title(lbl, fontsize=10, color="white")
                ax.axis('off')
        plt.tight_layout()
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img_array

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        lr_img, hr_img = batch["lr"].float(), batch["hr"].float()
        start = time.perf_counter()
        sr_img = self(lr_img).float()
        inf_time = time.perf_counter() - start
        
        psnr = peak_signal_noise_ratio(hr_img.cpu().numpy(), sr_img.cpu().numpy(), data_range=2.0)
        lpips = self.lpips(self.normalize_for_lpips(hr_img), self.normalize_for_lpips(sr_img)).cpu().item()
        
        result = {"test/PSNR": psnr, "test/LPIPS": lpips, "test/inference_time": inf_time}
        self.test_step_outputs.append(result)
        return result

    def on_test_epoch_end(self) -> None:
        if not self.test_step_outputs: return
        avg_res = {k: np.mean([x[k] for x in self.test_step_outputs]) for k in self.test_step_outputs[0].keys()}
        self.log_dict(avg_res, on_epoch=True, prog_bar=True)
        self.test_step_outputs.clear()