import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import os
import wandb
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import time


class SupResDiffGAN(pl.LightningModule):
    """SupResDiffGAN class for Super-Resolution Diffusion Generative Adversarial Network.

    This model combines the best features from both the samarthya04 and dawir7 repositories.

    Parameters
    ----------
    ae : nn.Module
        Autoencoder model.
    discriminator : nn.Module
        Discriminator model.
    unet : nn.Module
        UNet generator model.
    diffusion : nn.Module
        Diffusion model.
    learning_rate : float, optional
        Learning rate for the optimizers (default is 1e-4).
    alfa_perceptual : float, optional
        Weight for the perceptual loss (default is 1e-3).
    alfa_adv : float, optional
        Weight for the adversarial loss (default is 1e-2).
    vgg_loss : nn.Module | None, optional
        The VGG loss module for perceptual loss (default is None).
        If None, the perceptual loss will not be used.
    """

    def __init__(
        self,
        ae: nn.Module,
        discriminator: nn.Module,
        unet: nn.Module,
        diffusion: nn.Module,
        learning_rate: float = 1e-4,
        alfa_perceptual: float = 1e-3,
        alfa_adv: float = 1e-2,
        vgg_loss: nn.Module | None = None,
    ) -> None:
        super(SupResDiffGAN, self).__init__()
        self.ae = ae
        self.discriminator = discriminator
        self.generator = unet
        self.diffusion = diffusion

        self.vgg_loss = vgg_loss
        self.content_loss = nn.MSELoss()
        self.adversarial_loss = nn.BCEWithLogitsLoss()  # From samarthya04 for numerical stability
        
        # --- FIX 1: Removed .to(self.device) ---
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        # --- END FIX 1 ---

        self.lr = learning_rate
        self.alfa_adv = alfa_adv
        self.alfa_perceptual = alfa_perceptual
        self.betas = (0.9, 0.999)

        for param in self.ae.parameters():
            param.requires_grad = False

        self.automatic_optimization = False

        self.test_step_outputs = []
        self.ema_weight = 0.97
        self.ema_mean = 0.5
        self.s = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SupResDiffGAN model."""
        with torch.no_grad():
            x_lat = (
                self.ae.encode(x).latents.detach()
                * self.ae.scaling_factor
            )

        x = self.diffusion.sample(self.generator, x_lat, x_lat.shape)

        with torch.no_grad():
            x_out = self.ae.decode(x / self.ae.scaling_factor).sample
        x_out = torch.clamp(x_out, -1, 1)
        return x_out

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Training step for the SupResDiffGAN model."""
        lr_img, hr_img = batch["lr"], batch["hr"]
        optimizer_g, optimizer_d = self.optimizers()

        with torch.no_grad():
            lr_lat = (
                self.ae.encode(lr_img).latents.detach()
                * self.ae.scaling_factor
            )
            x0_lat = (
                self.ae.encode(hr_img).latents.detach()
                * self.ae.scaling_factor
            )

        timesteps = torch.randint(
            0,
            self.diffusion.timesteps,
            (x0_lat.shape[0],),
            device=x0_lat.device,
            dtype=torch.long,
        )
        x_t = self.diffusion.forward(x0_lat, timesteps)
        alfa_bars = self.diffusion.alpha_bars_torch.to(timesteps.device)[
            timesteps]
        x_gen_0 = self.generator(lr_lat, x_t, alfa_bars)

        s_tensor = torch.tensor(self.s, device=x0_lat.device, dtype=torch.long).expand(
            x0_lat.shape[0]
        )
        x_s = self.diffusion.forward(x0_lat, s_tensor)
        x_gen_s = self.diffusion.forward(x_gen_0, s_tensor)

        with torch.no_grad():
            sr_img = torch.clamp(self.ae.decode(x_gen_0 / self.ae.scaling_factor).sample, -1, 1)
            hr_s_img = torch.clamp(self.ae.decode(x_s / self.ae.scaling_factor).sample, -1, 1)
            sr_s_img = torch.clamp(self.ae.decode(x_gen_s / self.ae.scaling_factor).sample, -1, 1)


        if batch_idx % 2 == 0:
            # Generator training
            optimizer_g.zero_grad()
            g_loss = self.generator_loss(
                x0_lat, x_gen_0, hr_img, sr_img, hr_s_img, sr_s_img
            )
            self.manual_backward(g_loss)
            optimizer_g.step()
            self.log(
                "train/g_loss",
                g_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            return {"g_loss": g_loss}
        else:
            # Discriminator training
            optimizer_d.zero_grad()
            d_loss = self.discriminator_loss(hr_s_img, sr_s_img.detach())
            self.manual_backward(d_loss)
            optimizer_d.step()
            self.log(
                "train/d_loss",
                d_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            return {"d_loss": d_loss}

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Validation step for the SupResDiffGAN model."""
        lr_img, hr_img = batch["lr"], batch["hr"]
        lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
        sr_img = self(lr_img)
        padding_info = {"lr": batch["padding_data_lr"],
                        "hr": batch["padding_data_hr"]}

        if batch_idx < 1:
            print(
                f"Generating validation images for Epoch {self.current_epoch}, Batch {batch_idx}..."
            )
            try:
                title = f"Validation Epoch {self.current_epoch} - Batch {batch_idx}"
                per_image_metrics = []
                for i in range(min(3, lr_img.shape[0])):
                    hr_img_np = hr_img[i].detach(
                    ).cpu().numpy().transpose(1, 2, 0)
                    sr_img_np = sr_img[i].detach(
                    ).cpu().numpy().transpose(1, 2, 0)
                    hr_img_np = (hr_img_np + 1) / 2
                    sr_img_np = (sr_img_np + 1) / 2
                    hr_img_np = hr_img_np[
                        : padding_info["hr"][i][1], : padding_info["hr"][i][0], :
                    ]
                    sr_img_np = sr_img_np[
                        : padding_info["hr"][i][1], : padding_info["hr"][i][0], :
                    ]
                    psnr = peak_signal_noise_ratio(
                        hr_img_np, sr_img_np, data_range=1.0)
                    ssim = structural_similarity(
                        hr_img_np, sr_img_np, channel_axis=-1, data_range=1.0
                    )
                    lpips_val = (
                        self.lpips(hr_img[i: i + 1],
                                  sr_img[i: i + 1]).cpu().item()
                    )
                    per_image_metrics.append((psnr, ssim, lpips_val))

                img_array = self.plot_images_with_metrics(
                    hr_img, lr_img, sr_img, padding_info, title, per_image_metrics
                )
                self.logger.experiment.log({
                    "validation_images": wandb.Image(img_array, caption=f"Epoch {self.current_epoch} Batch {batch_idx}")
                })
                print(
                    f"Successfully logged validation images for Epoch {self.current_epoch}, Batch {batch_idx}"
                )
            except Exception as e:
                print(
                    f"Visualization error for Epoch {self.current_epoch}, Batch {batch_idx}: {str(e)}"
                )
                os.makedirs("outputs/logs", exist_ok=True)
                with open("outputs/logs/validation_errors.txt", "a") as f:
                    f.write(f"Epoch {self.current_epoch} Batch {batch_idx}: {str(e)}\n")

        metrics = {"PSNR": [], "SSIM": [], "MSE": []}
        for i in range(lr_img.shape[0]):
            hr_img_np = hr_img[i].detach().cpu().numpy().transpose(1, 2, 0)
            sr_img_np = sr_img[i].detach().cpu().numpy().transpose(1, 2, 0)
            hr_img_np = (hr_img_np + 1) / 2
            sr_img_np = (sr_img_np + 1) / 2
            hr_img_np = hr_img_np[
                : padding_info["hr"][i][1], : padding_info["hr"][i][0], :
            ]
            sr_img_np = sr_img_np[
                : padding_info["hr"][i][1], : padding_info["hr"][i][0], :
            ]
            psnr = peak_signal_noise_ratio(
                hr_img_np, sr_img_np, data_range=1.0)
            ssim = structural_similarity(
                hr_img_np, sr_img_np, channel_axis=-1, data_range=1.0
            )
            mse = np.mean((hr_img_np - sr_img_np) ** 2)
            metrics["PSNR"].append(psnr)
            metrics["SSIM"].append(ssim)
            metrics["MSE"].append(mse)

        lpips = self.lpips(hr_img, sr_img).cpu().item()
        self.log_dict({
            "val/PSNR": np.mean(metrics["PSNR"]),
            "val/SSIM": np.mean(metrics["SSIM"]),
            "val/MSE": np.mean(metrics["MSE"]),
            "val/LPIPS": lpips
        }, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def plot_images_with_metrics(
        self,
        hr_img: torch.Tensor,
        lr_img: torch.Tensor,
        sr_img: torch.Tensor,
        padding_info: dict,
        title: str,
        per_image_metrics: list,
    ) -> np.ndarray:
        fig, axs = plt.subplots(
            3, 3, figsize=(9, 9), dpi=100
        )
        fig.suptitle(title, fontsize=14, fontweight="bold", color="white")
        fig.patch.set_facecolor("#1e1e1e")
        for ax in axs.flat:
            ax.set_facecolor("#2e2e2e")

        for i in range(3):
            num = np.random.randint(0, len(per_image_metrics))
            psnr, ssim, lpips_val = per_image_metrics[num]

            sr_img_plot = (
                torch.clamp(sr_img[num], -1,
                            1).cpu().float().numpy().transpose(1, 2, 0)
            )
            sr_img_plot = (sr_img_plot + 1) / 2
            sr_img_plot = np.clip(sr_img_plot, 0, 1)[
                : padding_info["hr"][num][1], : padding_info["hr"][num][0], :
            ]

            hr_img_true = hr_img[num].cpu().float().numpy().transpose(1, 2, 0)
            hr_img_true = (hr_img_true + 1) / 2
            hr_img_true = np.clip(hr_img_true, 0, 1)[
                : padding_info["hr"][num][1], : padding_info["hr"][num][0], :
            ]

            lr_img_true = lr_img[num].cpu().float().numpy().transpose(1, 2, 0)
            lr_img_true = (lr_img_true + 1) / 2
            lr_img_true = np.clip(lr_img_true, 0, 1)[
                : padding_info["lr"][num][1], : padding_info["lr"][num][0], :
            ]

            axs[0, i].imshow(hr_img_true)
            axs[0, i].set_title("HR Ground Truth", fontsize=9, color="white")
            axs[0, i].set_xticks([])
            axs[0, i].set_yticks([])

            axs[1, i].imshow(lr_img_true)
            axs[1, i].set_title("LR Input", fontsize=9, color="white")
            axs[1, i].set_xticks([])
            axs[1, i].set_yticks([])

            axs[2, i].imshow(sr_img_plot)
            axs[2, i].set_title(
                f"SR Predicted\nPSNR: {psnr:.2f}\nSSIM: {ssim:.3f}\nLPIPS: {lpips_val:.3f}",
                fontsize=7,
                color="limegreen",
                pad=2,
            )
            axs[2, i].set_xticks([])
            axs[2, i].set_yticks([])

        plt.tight_layout(pad=0.5)
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(
            fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img_array

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict:
        lr_img, hr_img = batch["lr"], batch["hr"]
        lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
        padding_info = {"lr": batch["padding_data_lr"],
                        "hr": batch["padding_data_hr"]}

        start_time = time.perf_counter()
        sr_img = self(lr_img)
        elapsed_time = time.perf_counter() - start_time

        if batch_idx == 0:
            per_image_metrics = []
            for i in range(min(3, lr_img.shape[0])):
                hr_img_np = hr_img[i].detach().cpu().numpy().transpose(1, 2, 0)
                sr_img_np = sr_img[i].detach().cpu().numpy().transpose(1, 2, 0)
                hr_img_np = (hr_img_np + 1) / 2
                sr_img_np = (sr_img_np + 1) / 2
                hr_img_np = hr_img_np[: padding_info["hr"][i][1], : padding_info["hr"][i][0], :]
                sr_img_np = sr_img_np[: padding_info["hr"][i][1], : padding_info["hr"][i][0], :]
                psnr = peak_signal_noise_ratio(hr_img_np, sr_img_np, data_range=1.0)
                ssim = structural_similarity(hr_img_np, sr_img_np, channel_axis=-1, data_range=1.0)
                lpips_val = (self.lpips(hr_img[i: i + 1], sr_img[i: i + 1]).cpu().item())
                per_image_metrics.append((psnr, ssim, lpips_val))

            img_array = self.plot_images_with_metrics(
                hr_img,
                lr_img,
                sr_img,
                padding_info,
                title=f"Test Images: Timesteps {self.diffusion.timesteps}, Posterior {self.diffusion.posterior_type}",
                per_image_metrics=per_image_metrics
            )
            os.makedirs("outputs/test_images", exist_ok=True)
            from PIL import Image
            img_pil = Image.fromarray(img_array)
            img_pil.save(f"outputs/test_images/test_results_timesteps_{self.diffusion.timesteps}_posterior_{self.diffusion.posterior_type}.png")
            self.logger.experiment.log(
                {"test_images": wandb.Image(img_pil, caption="Test Images")}
            )

        metrics = {"PSNR": [], "SSIM": [], "MSE": []}
        for i in range(lr_img.shape[0]):
            hr_img_np = hr_img[i].detach().cpu().numpy().transpose(1, 2, 0)
            sr_img_np = sr_img[i].detach().cpu().numpy().transpose(1, 2, 0)
            hr_img_np = (hr_img_np + 1) / 2
            sr_img_np = (sr_img_np + 1) / 2
            
            # --- FIX 2: Corrected array slicing ---
            hr_img_np = hr_img_np[
                : padding_info["hr"][i][1], : padding_info["hr"][i][0], :
            ]
            sr_img_np = sr_img_np[
                : padding_info["hr"][i][1], : padding_info["hr"][i][0], :
            ]
            # --- END FIX 2 ---
            
            psnr = peak_signal_noise_ratio(
                hr_img_np, sr_img_np, data_range=1.0)
            ssim = structural_similarity(
                hr_img_np, sr_img_np, channel_axis=-1, data_range=1.0
            )
            mse = np.mean((hr_img_np - sr_img_np) ** 2)
            metrics["PSNR"].append(psnr)
            metrics["SSIM"].append(ssim)
            metrics["MSE"].append(mse)

        lpips = self.lpips(hr_img, sr_img).cpu().item()
        result = {
            "PSNR": np.mean(metrics["PSNR"]),
            "SSIM": np.mean(metrics["SSIM"]),
            "MSE": np.mean(metrics["MSE"]),
            "LPIPS": lpips,
            "time": elapsed_time,
        }
        self.test_step_outputs.append(result)
        return result

    def on_test_epoch_end(self) -> None:
        """Aggregate the metrics for all batches at the end of the test epoch."""
        avg_psnr = np.mean([x["PSNR"] for x in self.test_step_outputs])
        avg_ssim = np.mean([x["SSIM"] for x in self.test_step_outputs])
        avg_mse = np.mean([x["MSE"] for x in self.test_step_outputs])
        avg_lpips = np.mean([x["LPIPS"] for x in self.test_step_outputs])
        avg_time = np.mean([x["time"] for x in self.test_step_outputs])

        self.log_dict({
            "test/PSNR": avg_psnr,
            "test/SSIM": avg_ssim,
            "test/MSE": avg_mse,
            "test/LPIPS": avg_lpips,
            "test/time": avg_time,
        }, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.test_step_outputs.clear()

    def discriminator_loss(
        self, hr_s_img: torch.Tensor, sr_s_img: torch.Tensor
    ) -> torch.Tensor:
        y = torch.randint(0, 2, (hr_s_img.shape[0],), device=hr_s_img.device)
        y = y.view(-1, 1, 1, 1)
        y_expanded = y.expand(
            -1, hr_s_img.shape[1], hr_s_img.shape[2], hr_s_img.shape[3]
        )
        first = torch.where(y_expanded == 0, hr_s_img, sr_s_img)
        second = torch.where(y_expanded == 0, sr_s_img, hr_s_img)
        x = torch.cat([first, second], dim=1)
        prediction = self.discriminator(x)
        d_loss = self.adversarial_loss(prediction, y.float())
        self.calculate_ema_noise_step(prediction, y.float())
        return d_loss

    def generator_loss(
        self,
        x0: torch.Tensor,
        x_gen: torch.Tensor,
        hr_img: torch.Tensor,
        sr_img: torch.Tensor,
        hr_s_img: torch.Tensor,
        sr_s_img: torch.Tensor,
    ) -> torch.Tensor:
        content_loss = self.content_loss(x_gen, x0)
        perceptual_loss = (
            self.vgg_loss(sr_img, hr_img) if self.vgg_loss is not None else 0
        )
        y = torch.randint(0, 2, (hr_s_img.shape[0],), device=hr_s_img.device)
        y = y.view(-1, 1, 1, 1)
        y_expanded = y.expand(
            -1, hr_s_img.shape[1], hr_s_img.shape[2], hr_s_img.shape[3]
        )
        first = torch.where(y_expanded == 0, sr_s_img, hr_s_img)
        second = torch.where(y_expanded == 0, hr_s_img, sr_s_img)
        x = torch.cat([first, second], dim=1)
        prediction = self.discriminator(x)
        adversarial_loss = self.adversarial_loss(prediction, y.float())
        y_reversed = 1 - y.float()
        self.calculate_ema_noise_step(prediction, y_reversed)
        g_loss = (
            content_loss
            + self.alfa_perceptual * perceptual_loss
            + self.alfa_adv * adversarial_loss
        )
        self.log(
            "train/g_content_loss",
            content_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/g_adv_loss",
            adversarial_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if self.vgg_loss is not None:
            self.log(
                "train/g_perceptual_loss",
                perceptual_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        return g_loss

    def configure_optimizers(
        self,
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=self.betas
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=self.betas
        )
        return [opt_g, opt_d]

    def calculate_ema_noise_step(self, pred: torch.Tensor, y: torch.Tensor) -> None:
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        acc = (pred_binary == y).float().mean().cpu().item()
        self.ema_mean = acc * (1 - self.ema_weight) + \
            self.ema_mean * self.ema_weight
        self.s = int(
            torch.clamp(
                torch.tensor((self.ema_mean - 0.5) * 2 *
                             self.diffusion.timesteps),
                0,
                self.diffusion.timesteps - 1,
            ).item()
        )
        self.log_dict({
            "train/ema_noise_step": self.s,
            "train/ema_accuracy": acc,
        }, on_epoch=True, logger=True, sync_dist=True)