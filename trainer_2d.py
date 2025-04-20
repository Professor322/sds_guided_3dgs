# baseline code was taken from https://github.com/nerfstudio-project/gsplat/blob/main/examples/image_fitting.py

import math
import os
import time
import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.parameter
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
from configs import Config2D
import tqdm
from gsplat.strategy import DefaultStrategy
from enum import Enum
from fused_ssim import fused_ssim


class LossType(Enum):
    MSE: int = 0
    SDS: int = 1
    UNKNOWN: int = 2


class OneImageDataset(Dataset):
    def __init__(
        self,
        image_path,
        patch_size=64,
        dataset_len=1000,
        training_width=256,
        training_height=256,
        validation_width=256,
        validation_height=256,
        train=True,
        use_generated_img=False,
    ):
        super().__init__()
        self.train = train

        self.len = dataset_len
        self.img = Image.open(image_path)
        to_tensor = transforms.ToTensor()
        img = to_tensor(self.img)

        self.img = F.interpolate(
            img.unsqueeze(0),
            (training_width, training_height),
            align_corners=False,
            antialias=True,
            mode="bilinear",
        ).squeeze(0)
        self.validation_img = F.interpolate(
            img.unsqueeze(0),
            (validation_width, validation_height),
            align_corners=False,
            antialias=True,
            mode="bilinear",
        ).squeeze(0)
        self.patch_size = patch_size
        self.training_img = None
        self.generated_img = None
        self.use_generated_img = use_generated_img

    def __getitem__(self, idx):
        # if self.train:
        # C, H, W -> H, W, C
        # i, j, h, w = transforms.RandomCrop.get_params(
        #     self.img, output_size=(self.patch_size, self.patch_size)
        # )
        # img = self.img if not self.use_generated_img else self.generated_img
        # patch_real = VF.crop(img, i, j, h, w).permute(1, 2, 0)
        # patch_pred = VF.crop(self.training_img, i, j, h, w).permute(1, 2, 0)
        # return patch_pred, patch_real
        # training img already in correct shape
        return self.training_img, self.img

    def __len__(self):
        return self.len if self.train else 1


class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(self, cfg: Config2D = Config2D()):
        self.cfg = cfg
        if self.cfg.use_strategy:
            self.cfg.strategy = DefaultStrategy(
                verbose=True, dropout=self.cfg.densification_dropout
            )
        else:
            self.cfg.strategy = NotImplementedError
        self.strategy_state = None
        self.device = torch.device("cuda:0")
        print(f"Loading dataset...")
        self.one_image_dataset = OneImageDataset(
            image_path=cfg.img_path,
            training_height=cfg.height,
            training_width=cfg.width,
            validation_height=cfg.render_height,
            validation_width=cfg.render_width,
        )
        self.num_points = self.cfg.num_points
        print(f"Creating directories...")
        self.ckpt_dir = f"{self.cfg.results_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{self.cfg.results_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{self.cfg.results_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        fov_x = math.pi / 2.0
        self.focal = 0.5 * float(self.cfg.render_width) / math.tan(0.5 * fov_x)

        if self.cfg.ckpt_path:
            self.splats, self.optimizers = self.load_splats_with_optimizers()
        else:
            self.splats, self.optimizers = self.create_splats_with_optimizers()
        if self.cfg.use_strategy:
            self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        print("Model initialized. Number of GS:", len(self.splats["means"]))

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
        self.mae_loss = torch.nn.L1Loss()
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

        if self.cfg.model_type == "3dgs":
            self.rasterize_fnc = rasterization
        elif self.cfg.model_type == "2dgs":
            self.rasterize_fnc = rasterization_2dgs

    def rasterize_splats(self, i) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        quats = quats / quats.norm(dim=-1, keepdim=True)
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        colors = torch.sigmoid(self.splats["rgbs"])

        render_colors, render_alphas, info = self.rasterize_fnc(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=self.viewmat[None],  # [C, 4, 4]
            Ks=self.K[None],  # [C, 3, 3]
            width=self.cfg.render_width,
            height=self.cfg.render_height,
            packed=False,
        )
        return render_colors, render_alphas, info

    def create_splats_with_optimizers(self):
        """Random gaussians"""
        bd = 2

        means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        scales = torch.rand(self.num_points, 3, device=self.device)
        d = 3
        rgbs = torch.rand(self.num_points, d, device=self.device)

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        opacities = torch.ones((self.num_points), device=self.device)

        params = [
            # name, value, lr
            ("means", torch.nn.Parameter(means), self.cfg.lr),
            ("scales", torch.nn.Parameter(scales), self.cfg.lr),
            ("quats", torch.nn.Parameter(quats), self.cfg.lr),
            ("opacities", torch.nn.Parameter(opacities), self.cfg.lr),
            ("rgbs", torch.nn.Parameter(rgbs), self.cfg.lr),
        ]
        splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(self.device)
        optimizers = {
            name: torch.optim.Adam([{"params": splats[name], "lr": lr, "name": name}])
            for name, _, lr in params
        }

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.K = torch.tensor(
            [
                [self.focal, 0, self.cfg.render_width / 2],
                [0, self.focal, self.cfg.render_height / 2],
                [0, 0, 1],
            ],
            device=self.device,
        )
        self.K.requires_grad = False
        self.viewmat.requires_grad = False
        return splats, optimizers

    def load_splats_with_optimizers(self):
        print(f"Loading checkpoint {self.cfg.ckpt_path}")
        ckpt = torch.load(self.cfg.ckpt_path, weights_only=False)

        splats, _ = self.create_splats_with_optimizers()

        for k in splats.keys():
            splats[k].data = ckpt["splats"][k]

        optimizers = {
            name: torch.optim.Adam(
                [{"params": splats[name], "lr": self.cfg.lr, "name": name}]
            )
            for name in splats.keys()
        }

        return splats, optimizers

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

    # Calculate gradient norm from optimizer's parameters
    def calculate_grad_norm(self):
        total_norm = 0.0
        for optimizer in self.optimizers.values():
            for param_group in optimizer.param_groups:
                for param in param_group["params"]:
                    if param.grad is not None:  # Check if gradient exists
                        param_norm = param.grad.norm(2)  # L2 norm of the gradient
                        total_norm += param_norm.item() ** 2
        return total_norm**0.5  # Return the overall gradient norm

    def validate(self):
        print("Validating...")
        with torch.no_grad():
            self.one_image_dataset.img = self.one_image_dataset.img.to(self.device)
            renders, _, _ = self.rasterize_splats()
            out_img = renders[0]
            frame = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(frame).save(f"{self.render_dir}/render.png")
            psnr_with_original = self.psnr(
                out_img, self.one_image_dataset.img.permute(1, 2, 0)
            )
            print(f"PSNR with original: {psnr_with_original.item()}")
            with open(f"{self.stats_dir}/render.json", "w") as f:
                json.dump({"psnr": psnr_with_original.item()}, f)

    def train(self):
        times = [0] * 2  # rasterization, backward

        begin = 0
        end = self.cfg.iterations
        losses = []
        psnrs = []
        ssims = []
        grad_norms = []
        learning_rates = []
        self.one_image_dataset.img = self.one_image_dataset.img.to(self.device)
        self.one_image_dataset.validation_img = (
            self.one_image_dataset.validation_img.to(self.device)
        )
        prev_render = None
        diff = None

        base_render = None
        if self.cfg.use_strategy and self.strategy_state is None:
            self.strategy_state = self.cfg.strategy.initialize_state()

        current_loss_type = LossType.UNKNOWN
        if self.cfg.use_altering_loss:
            # we start with SDS
            current_loss_type = LossType.SDS

        if cfg.debug_training:
            # clone params
            param_info = {
                name: {"previous": param.detach().clone(), "diff": 0.0, "updates": 0}
                for name, param in self.splats.items()
            }

        pbar = tqdm.tqdm(range(begin, end))
        for i in pbar:
            start = time.time()
            renders, _, info = self.rasterize_splats(i)
            if self.cfg.use_strategy:
                self.cfg.strategy.step_pre_backward(
                    self.splats, self.optimizers, self.strategy_state, i, info
                )
            out_img = renders[0]
            # how much pixels have changed compared to previous iteration
            if prev_render is not None:
                diff = (prev_render - out_img).clamp(0.0, 1.0)

            if base_render is None:
                base_render = out_img.detach().clone()
                base_render.requires_grad = True

            torch.cuda.synchronize()
            times[0] += time.time() - start
            if self.cfg.use_classic_mse_loss:
                # calculate loss
                mse_loss = self.mse_loss(
                    out_img.permute(2, 0, 1), self.one_image_dataset.img
                )
            elif self.cfg.use_downscaled_mse_loss:
                # downscale the base image and rendering
                # compute mse
                resolution = (64, 64)
                # interpolate expects b, c, h, w, while we have h, w, c
                downscaled_out_img = F.interpolate(
                    out_img.permute(2, 0, 1).unsqueeze(0),
                    resolution,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                downscaled_base_render = F.interpolate(
                    base_render.permute(2, 0, 1).unsqueeze(0),
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
                pred = pred.to(self.device)  # .permute(0, 3, 1, 2)
                real = real.to(self.device)  # .permute(0, 3, 1, 2)
                if self.cfg.collapsing_noise_scheduler:
                    min_step = self.compute_step(200, 300, i / self.cfg.iterations)
                    max_step = self.compute_step(500, 980, i / self.cfg.iterations)
                elif self.cfg.noise_step_anealing > 0:
                    min_step = int(
                        min(
                            self.cfg.min_noise_step,
                            self.cfg.max_noise_step - i / self.cfg.noise_step_anealing,
                        )
                    )
                    max_step = self.cfg.max_noise_step
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
            if cfg.use_gaussian_sr:
                # do not use base renders anymore
                # render image in HR 256x256
                # apply noise to it and condition on LR (64x64)
                # original image (probably do not apply any noise to condition)
                # calculate SDS
                # downscale rendering into 256x256->64x64 and calculate MSE
                # final loss MSE + lmbd * SDS (how to do it properly?)
                resolution = (64, 64)
                downscaled_render = F.interpolate(
                    out_img.permute(2, 0, 1).unsqueeze(0),
                    resolution,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                if cfg.use_mae_loss:
                    loss = self.mae_loss(
                        downscaled_render, self.dataloader.dataset.img.unsqueeze(0)
                    )
                else:
                    loss = self.mse_loss(
                        downscaled_render, self.dataloader.dataset.img.unsqueeze(0)
                    )
                if cfg.use_sds_loss or cfg.use_sdi_loss:
                    loss += sds.squeeze()
                if cfg.use_ssim_loss:
                    loss += cfg.ssim_lambda * (
                        1.0
                        - fused_ssim(
                            downscaled_render,
                            self.dataloader.dataset.img.unsqueeze(0),
                        )
                    )

            elif self.cfg.use_altering_loss:
                if current_loss_type == LossType.MSE:
                    loss = mse_loss
                    current_loss_type = LossType.SDS
                elif current_loss_type == LossType.SDS:
                    loss = sds
                    current_loss_type = LossType.MSE
            elif self.cfg.use_fused_loss and (
                self.cfg.use_sds_loss or self.cfg.use_sdi_loss
            ):
                ssim_loss = 0.0
                if self.cfg.use_ssim_loss:
                    # we need to minimize
                    # this one wants [B, C, H, W]
                    ssim_loss = 1.0 - fused_ssim(
                        out_img.permute(2, 0, 1).unsqueeze(0),
                        base_render.permute(2, 0, 1).unsqueeze(0),
                    )
                loss = mse_loss + sds + ssim_loss
            elif self.cfg.use_sds_loss or self.cfg.use_sdi_loss:
                ssim_loss = 0.0
                if self.cfg.use_ssim_loss:
                    # we need to minimize
                    # this one wants [B, C, H, W]
                    ssim_loss = 1.0 - fused_ssim(
                        out_img.permute(2, 0, 1).unsqueeze(0),
                        base_render.permute(2, 0, 1).unsqueeze(0),
                    )
                loss = sds + ssim_loss
            else:
                # this is classical 3DGS case
                ssim_loss = 0.0
                if self.cfg.use_ssim_loss:
                    # we need to minimize
                    # this one wants [B, C, H, W]
                    ssim_loss = 1.0 - fused_ssim(
                        out_img.permute(2, 0, 1).unsqueeze(0),
                        self.one_image_dataset.img.unsqueeze(0),
                    )
                loss = mse_loss + ssim_loss

            start = time.time()
            loss.backward()

            torch.cuda.synchronize()
            times[1] += time.time() - start

            if self.cfg.use_strategy:
                self.cfg.strategy.step_post_backward(
                    self.splats, self.optimizers, self.strategy_state, i, info
                )

            # this one wants [H, W, C]
            psnr = self.psnr(
                out_img, self.one_image_dataset.validation_img.permute(1, 2, 0)
            )
            # this one wants [B, C, H, W]
            ssim = self.ssim(
                out_img.permute(2, 0, 1).unsqueeze(0),
                self.one_image_dataset.validation_img.unsqueeze(0),
            )

            if self.cfg.grad_clipping > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in self.splats.values()], self.cfg.grad_clipping
                )

            # stats
            losses.append(loss.item())
            psnrs.append(psnr.item())
            grad_norms.append(self.calculate_grad_norm())
            ssims.append(ssim.item())
            # learning rate and scheduling is the same for all params
            learning_rates.append(self.optimizers["means"].param_groups[0]["lr"])

            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if self.cfg.debug_training:
                for name, param in self.splats.items():
                    if name == "opacities":
                        # l2 norm manually
                        count_changed = (
                            ((param - param_info[name]["previous"]) ** 2 > 0.0)
                            .sum()
                            .item()
                        )
                    else:
                        count_changed = (
                            (
                                torch.norm(param - param_info[name]["previous"], dim=-1)
                                > 0.0
                            )
                            .sum()
                            .item()
                        )
                    param_info[name]["updates"] += count_changed
                    param_info[name]["previous"] = param.detach().clone()

            pbar.set_description(
                f"Iteration {i + 1}/{end}, Loss: {loss.item()}, PSNR: {psnr.item()} SSIM: {ssim.item()}"
            )
            prev_render = out_img.detach().clone()
            if i % self.cfg.show_steps == 0 or i == end - 1 or i == begin + 1:
                if self.cfg.show_plots:
                    clear_output(wait=True)
                base_render_rgb = (base_render.detach().cpu().numpy() * 255).astype(
                    np.uint8
                )
                pred = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
                orig = (
                    self.one_image_dataset.validation_img.permute(1, 2, 0)
                    .detach()
                    .cpu()
                    .numpy()
                    * 255
                ).astype(np.uint8)

                # Create the figure with an additional row for the new plot
                fig, axes = plt.subplots(3, 3, figsize=(20, 20))

                # Plot 1: Original Image
                axes[0, 0].imshow(orig)
                axes[0, 0].set_title("Original Image")
                axes[0, 0].axis("off")

                # Plot 2: Predicted Image
                axes[0, 1].imshow(pred)
                axes[0, 1].set_title("Predicted Image")
                axes[0, 1].axis("off")

                # Plot 3: Base Render Image
                axes[0, 2].imshow(base_render_rgb)
                axes[0, 2].set_title("Base Render from Start of Training")
                axes[0, 2].axis("off")

                # Plot 4: PSNR Evolution
                axes[1, 0].plot(psnrs, label="PSNR")
                axes[1, 0].set_title("PSNR with original image")
                axes[1, 0].set_xlabel("Epoch")
                axes[1, 0].set_ylabel("PSNR (dB)")
                axes[1, 0].grid(True)
                axes[1, 0].legend()

                # Plot 5: Loss Evolution
                axes[1, 1].plot(losses, label="Loss", color="red")
                axes[1, 1].set_title("Loss")
                axes[1, 1].set_xlabel("Epoch")
                axes[1, 1].set_ylabel("Loss")
                axes[1, 1].grid(True)
                axes[1, 1].legend()

                # Plot 6: Gradient Norm Evolution
                axes[1, 2].plot(grad_norms, label="Gradient Norm", color="purple")
                axes[1, 2].set_title("Gradient Norm")
                axes[1, 2].set_xlabel("Epoch")
                axes[1, 2].set_ylabel("Gradient Norm")
                axes[1, 2].grid(True)
                axes[1, 2].legend()

                # Plot 7: Learning Rate Evolution
                axes[2, 0].plot(learning_rates, label="Learning Rate", color="green")
                axes[2, 0].set_title("Learning Rate Evolution")
                axes[2, 0].set_xlabel("Epoch")
                axes[2, 0].set_ylabel("Learning Rate")
                axes[2, 0].grid(True)
                axes[2, 0].legend()

                # Plot 8: SSIM with original image
                axes[2, 1].plot(ssims, label="SSIM", color="green")
                axes[2, 1].set_title("SSIM with original image")
                axes[2, 1].set_xlabel("Epoch")
                axes[2, 1].set_ylabel("SSIM")
                axes[2, 1].grid(True)
                axes[2, 1].legend()

                # Plot 9: Pixel difference between last 2 iterations
                if diff is not None:
                    diff_render_rgb = (diff.detach().cpu().numpy() * 255).astype(
                        np.uint8
                    )
                    axes[2, 2].imshow(diff_render_rgb)
                    axes[2, 2].set_title("Pixel difference between last 2 iterations")
                    axes[2, 2].axis("off")
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
                    if diff is not None:
                        Image.fromarray(diff_render_rgb).save(
                            f"{self.render_dir}/diff_image_{i}.png"
                        )

            if i in [idx - 1 for idx in self.cfg.save_steps] or i == end - 1:
                print(f"Saving checkpoint at: {i}")
                to_save = {
                    "splats": self.splats.state_dict(),
                }
                frame = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
                # also save the last rendering
                Image.fromarray(frame).save(f"{self.render_dir}/image_{i}.png")
                torch.save(to_save, f"{self.ckpt_dir}/ckpt_{i}.pt")
                with open(f"{self.stats_dir}/step{i}.json", "w") as f:
                    json.dump({"psnr": psnr.item(), "ssim": ssim.item()}, f)

        print(f"Total(s):\nRasterization: {times[0]:.3f}, Backward: {times[1]:.3f}")
        print(
            f"Per step(s):\nRasterization: {times[0]/self.cfg.iterations:.5f}, Backward: {times[1]/self.cfg.iterations:.5f}"
        )
        if cfg.debug_training:
            for name, info in param_info.items():
                print(
                    f"Parameter {name} had {info['updates'] / self.cfg.iterations} updates per iteration on average"
                )


def main(
    cfg: Config2D,
) -> None:

    trainer = SimpleTrainer(cfg=cfg)
    if cfg.validate:
        trainer.validate()
    else:
        trainer.train()


if __name__ == "__main__":
    cfg = tyro.cli(Config2D)
    main(cfg)
