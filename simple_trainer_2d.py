import math
import os
import time
import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import tyro
from PIL import Image
from torch import Tensor, optim

from gsplat import rasterization, rasterization_2dgs
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import os
from IPython.display import display, clear_output
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from guidance import SDSLoss3DGS, SDILoss3DGS
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as VF
from PIL import Image
from config import Config


class OneImageDataset(Dataset):
    def __init__(
        self,
        image_path,
        patch_size=64,
        dataset_len=1000,
        resize=True,
        train=True,
        use_generated_img=False,
    ):
        super().__init__()
        self.train = train

        self.len = dataset_len
        self.img = Image.open(image_path)
        if resize:
            self.img = self.img.resize((256, 256))
        to_tensor = transforms.ToTensor()
        self.img = to_tensor(self.img)
        self.patch_size = patch_size
        self.training_img = None
        self.generated_img = None
        self.use_generated_img = use_generated_img

    def __getitem__(self, idx):
        # C, H, W -> H, W, C
        i, j, h, w = transforms.RandomCrop.get_params(
            self.img, output_size=(self.patch_size, self.patch_size)
        )
        if self.train:
            img = self.img if not self.use_generated_img else self.generated_img
            patch_real = VF.crop(img, i, j, h, w).permute(1, 2, 0)
            patch_pred = VF.crop(self.training_img, i, j, h, w).permute(1, 2, 0)
            return patch_pred, patch_real
        # training img already in correct shape
        return self.training_img, self.img.permute(1, 2, 0)

    def __len__(self):
        return self.len if self.train else 1


class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        self.device = torch.device("cuda:0")
        print(f"Loading dataset...")
        self.one_image_dataset = OneImageDataset(image_path=cfg.img_path)
        self.num_points = self.cfg.num_points
        self.iter = 0
        self.frames = []
        print(f"Creating directories...")
        self.ckpt_dir = f"{self.cfg.results_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{self.cfg.results_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{self.cfg.results_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        fov_x = math.pi / 2.0
        self.H, self.W = self.one_image_dataset.img.size(
            1
        ), self.one_image_dataset.img.size(2)
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        if self.cfg.ckpt_path:
            self._load_gaussians(self.cfg.ckpt_path)
        else:
            self._init_gaussians()

        self.optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats],
            self.cfg.lr,
        )

        if self.cfg.ckpt_path:
            print("Loading optimizer state...")
            self.optimizer.load_state_dict(self.optimizer_state_dict)
        if self.cfg.use_sds_loss or self.cfg.use_sdi_loss:
            self.sds_loss = (
                SDSLoss3DGS(prompt=self.cfg.prompt)
                if self.cfg.use_sds_loss
                else SDILoss3DGS(prompt=self.cfg.prompt)
            )
            self.dataloader = DataLoader(
                self.one_image_dataset, batch_size=cfg.batch_size, num_workers=0
            )
        self.mse_loss = torch.nn.MSELoss()
        if self.cfg.use_noise_scheduler:
            self.noise_scheduler = self.set_linear_time_strategy(
                self.cfg.iterations, self.cfg.min_noise_step, self.cfg.max_noise_step
            )
        self.lr_scheduler = None
        if self.cfg.use_lr_scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                total_steps=self.cfg.iterations,
                max_lr=self.cfg.lr,
                pct_start=0.5,
            )

    def set_linear_time_strategy(
        self, output_shape, min_diffusion_steps=20, max_diffusion_steps=980
    ):
        return (
            torch.linspace(min_diffusion_steps, max_diffusion_steps, output_shape)
            .flip(0)
            .to(torch.long)
        )

    def compute_step(self, min_step, max_step, iter_frac):
        step = max_step - (max_step - min_step) * math.sqrt(iter_frac)
        return int(step)

    def _load_gaussians(self, ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        print(f"Loading checkpoint {ckpt_path}")
        self.opacities = ckpt["opacities"]
        self.quats = ckpt["quats"]
        self.rgbs = ckpt["rgbs"]
        self.scales = ckpt["scales"]
        self.means = ckpt["means"]
        self.iter = ckpt["iter"]
        self.frames = ckpt["frames"]

        self.optimizer_state_dict = ckpt["optimizer"]
        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )

    def _init_gaussians(self):
        """Random gaussians"""
        bd = 2

        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        self.scales = torch.rand(self.num_points, 3, device=self.device)
        d = 3
        self.rgbs = torch.rand(self.num_points, d, device=self.device)

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.opacities = torch.ones((self.num_points), device=self.device)

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.background = torch.zeros(d, device=self.device)

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

    # Calculate gradient norm from optimizer's parameters
    def calculate_grad_norm(self):
        total_norm = 0.0
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is not None:  # Check if gradient exists
                    param_norm = param.grad.norm(2)  # L2 norm of the gradient
                    total_norm += param_norm.item() ** 2
        return total_norm**0.5  # Return the overall gradient norm

    def train(self):
        frames = self.frames
        times = [0] * 2  # rasterization, backward
        K = torch.tensor(
            [
                [self.focal, 0, self.W / 2],
                [0, self.focal, self.H / 2],
                [0, 0, 1],
            ],
            device=self.device,
        )

        if self.cfg.model_type == "3dgs":
            rasterize_fnc = rasterization
        elif self.cfg.model_type == "2dgs":
            rasterize_fnc = rasterization_2dgs

        begin = 0
        end = self.cfg.iterations
        losses = []
        psnrs = []
        grad_norms = []
        learning_rates = []
        self.one_image_dataset.img = self.one_image_dataset.img.to(self.device)

        base_render = None

        for i in range(begin, end):
            start = time.time()

            renders = rasterize_fnc(
                self.means,
                self.quats / self.quats.norm(dim=-1, keepdim=True),
                self.scales,
                torch.sigmoid(self.opacities),
                torch.sigmoid(self.rgbs),
                self.viewmat[None],
                K[None],
                self.W,
                self.H,
                packed=False,
            )[0]
            out_img = renders[0]
            if base_render is None:
                base_render = out_img.detach().clone()
                base_render.requires_grad = True

            torch.cuda.synchronize()
            times[0] += time.time() - start
            if self.cfg.use_classic_mse_loss:
                # calculate loss
                mse_loss = self.mse_loss(
                    out_img, self.one_image_dataset.img.permute(1, 2, 0)
                )
            elif self.cfg.use_downscaled_mse_loss:
                # downscale the base image and rendering
                # compute mse
                resolution = (64, 64)
                # interpolate expects b, c, h, w, while we have h, w, c
                downscaled_out_img = F.interpolate(
                    out_img.permute(1, 2, 0).unsqueeze(0),
                    resolution,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                downscaled_base_render = F.interpolate(
                    base_render.permute(1, 2, 0).unsqueeze(0),
                    resolution,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                mse_loss = self.mse_loss(downscaled_out_img, downscaled_base_render)

            if self.cfg.use_sds_loss or self.cfg.use_sdi_loss:
                # H, W, C -> C, H, W
                if self.cfg.base_render_as_cond:
                    self.dataloader.dataset.generated_img = base_render.permute(2, 0, 1)
                    self.dataloader.dataset.use_generated_img = True
                self.dataloader.dataset.training_img = out_img.permute(2, 0, 1)
                pred, real = next(iter(self.dataloader))
                pred = pred.to(self.device).permute(0, 3, 1, 2)
                real = real.to(self.device).permute(0, 3, 1, 2)
                if self.cfg.collapsing_noise_scheduler:
                    min_step = self.compute_step(200, 300, i / self.cfg.iterations)
                    max_step = self.compute_step(500, 980, i / self.cfg.iterations)
                else:
                    min_step = self.cfg.min_noise_step
                    max_step = self.cfg.max_noise_step
                sds = (
                    self.sds_loss(
                        images=pred,
                        original=real,
                        min_step=min_step,
                        max_step=max_step,
                        lowres_noise_level=self.cfg.lowres_noise_level,
                        scheduler_timestep=self.noise_scheduler[i]
                        if self.cfg.use_noise_scheduler
                        else None,
                        downscale_condition=cfg.downscale_condition,
                        guidance_scale=cfg.guidance_scale,
                    )
                    * self.cfg.lmbd
                )

            if self.cfg.use_fused_loss and (
                self.cfg.use_sds_loss or self.cfg.use_sdi_loss
            ):
                loss = mse_loss + sds
            elif self.cfg.use_sds_loss or self.cfg.use_sdi_loss:
                loss = sds
            else:
                loss = mse_loss

            self.optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[1] += time.time() - start
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            psnr = self.psnr(out_img, self.one_image_dataset.img.permute(1, 2, 0))
            # stats
            losses.append(loss.item())
            psnrs.append(psnr.item())
            grad_norms.append(self.calculate_grad_norm())
            learning_rates.append(self.optimizer.param_groups[0]["lr"])

            print(f"Iteration {i + 1}/{end}, Loss: {loss.item()}, PSNR: {psnr.item()}")

            if i % self.cfg.show_steps == 0 or i == end - 1:
                if self.cfg.show_plots:
                    clear_output(wait=True)
                base_render_rgb = (base_render.detach().cpu().numpy() * 255).astype(
                    np.uint8
                )
                pred = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
                orig = (
                    self.one_image_dataset.img.permute(1, 2, 0).detach().cpu().numpy()
                    * 255
                ).astype(np.uint8)

                # Create the figure with an additional row for the new plot
                fig, axes = plt.subplots(4, 2, figsize=(12, 20))

                # Plot 1: Original Image
                axes[0, 0].imshow(orig)
                axes[0, 0].set_title("Original Image")
                axes[0, 0].axis("off")

                # Plot 2: Predicted Image
                axes[0, 1].imshow(pred)
                axes[0, 1].set_title("Predicted Image")
                axes[0, 1].axis("off")

                # Plot 3: PSNR Evolution
                axes[1, 0].plot(psnrs, label="PSNR")
                axes[1, 0].set_title("PSNR")
                axes[1, 0].set_xlabel("Epoch")
                axes[1, 0].set_ylabel("PSNR (dB)")
                axes[1, 0].grid(True)
                axes[1, 0].legend()

                # Plot 4: Loss Evolution
                axes[1, 1].plot(losses, label="Loss", color="red")
                axes[1, 1].set_title("Loss")
                axes[1, 1].set_xlabel("Epoch")
                axes[1, 1].set_ylabel("Loss")
                axes[1, 1].grid(True)
                axes[1, 1].legend()

                # Plot 5: Gradient Norm Evolution
                axes[2, 0].plot(grad_norms, label="Gradient Norm", color="purple")
                axes[2, 0].set_title("Gradient Norm")
                axes[2, 0].set_xlabel("Epoch")
                axes[2, 0].set_ylabel("Gradient Norm")
                axes[2, 0].grid(True)
                axes[2, 0].legend()

                # Plot 6: Base Render Image
                axes[2, 1].imshow(base_render_rgb)
                axes[2, 1].set_title("Base Render from Start of Training")
                axes[2, 1].axis("off")

                # Plot 7: Learning Rate Evolution
                axes[3, 0].plot(learning_rates, label="Learning Rate", color="green")
                axes[3, 0].set_title("Learning Rate Evolution")
                axes[3, 0].set_xlabel("Epoch")
                axes[3, 0].set_ylabel("Learning Rate")
                axes[3, 0].grid(True)
                axes[3, 0].legend()

                # Adjust layout
                plt.tight_layout()

                if self.cfg.show_plots:
                    plt.show()
                else:
                    plt.savefig(
                        f"{self.stats_dir}/training_plots.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close(fig)
                if self.cfg.save_imgs:
                    frame = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(frame).save(f"{self.render_dir}/image_{i}.png")
            if i in [idx - 1 for idx in self.cfg.save_steps] or i == end - 1:
                print(f"Saving checkpoint at: {i}")
                to_save = {
                    "optimizer": self.optimizer.state_dict(),
                    "iter": i,
                    "means": self.means,
                    "quats": self.quats,
                    "opacities": self.opacities,
                    "rgbs": self.rgbs,
                    "scales": self.scales,
                    "frames": frames,
                }
                frame = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
                # also save the last rendering
                Image.fromarray(frame).save(f"{self.render_dir}/image_{i}.png")
                torch.save(to_save, f"{self.ckpt_dir}/ckpt_{i}.pt")
                with open(f"{self.stats_dir}/step{i}.json", "w") as f:
                    json.dump({"psnr": psnr.item()}, f)

        print(f"Total(s):\nRasterization: {times[0]:.3f}, Backward: {times[1]:.3f}")
        print(
            f"Per step(s):\nRasterization: {times[0]/self.cfg.iterations:.5f}, Backward: {times[1]/self.cfg.iterations:.5f}"
        )


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(
    cfg: Config,
) -> None:

    trainer = SimpleTrainer(cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
