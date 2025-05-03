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
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from guidance import SDSLoss3DGS, SDILoss3DGS, SDSLoss3DGS_StableSR
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as VF
from PIL import Image
from configs import Config2D
import tqdm
from gsplat.strategy import DefaultStrategy
from enum import Enum
from fused_ssim import fused_ssim
from utils import (
    set_linear_noise_schedule,
    compute_collapsing_noise_step,
    calculate_grad_norm,
)
from StableSR.scripts.wavelet_color_fix import (
    adaptive_instance_normalization,
    wavelet_reconstruction,
)


def process_image(image_path: str, image_shape: Tuple[int, int]) -> torch.Tensor:
    image = Image.open(image_path)
    to_tensor = transforms.ToTensor()
    image = to_tensor(image)
    if image_shape != (0, 0):
        image = F.interpolate(
            image.unsqueeze(0),
            image_shape,
            mode="bicubic",
        ).clamp(0.0, 1.0)
    return image.squeeze(0)


class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(self, cfg: Config2D = Config2D()):
        self.cfg = cfg
        if self.cfg.use_strategy:
            self.cfg.strategy = DefaultStrategy(
                verbose=True, dropout=self.cfg.densification_dropout
            )
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            self.cfg.strategy = NotImplementedError
        self.device = torch.device("cuda:0")
        self.num_points = self.cfg.num_points
        self.training_img = process_image(
            self.cfg.training_image_path, (self.cfg.height, self.cfg.width)
        )
        self.height = self.training_img.size(1)
        self.width = self.training_img.size(2)
        self.render_width = int(self.width * self.cfg.scale_factor)
        self.render_height = int(self.height * self.cfg.scale_factor)
        self.validation_image = process_image(
            self.cfg.validation_image_path, (self.render_height, self.render_width)
        )

        self.training_img = self.training_img.to(self.device)
        self.validation_image = self.validation_image.to(self.device)
        print(
            f"Training image: {self.cfg.training_image_path} resolution: h{self.height}xw{self.width}"
        )
        print(
            f"Validation image: {self.cfg.validation_image_path}, resolution: h{self.render_height}xw{self.render_width}"
        )

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
        self.focal = 0.5 * float(self.render_width) / math.tan(0.5 * fov_x)

        if self.cfg.ckpt_path:
            self.splats, self.optimizers = self.load_splats_with_optimizers()
        else:
            self.splats, self.optimizers = self.create_splats_with_optimizers()
        if self.cfg.use_strategy:
            self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        if self.cfg.use_gaussian_sr:
            if self.cfg.sds_loss_type == "deepfloyd_sdi":
                self.sds_loss = SDILoss3DGS(prompt=self.cfg.prompt)
            elif self.cfg.sds_loss_type == "deepfloyd_sds":
                self.sds_loss = SDSLoss3DGS(prompt=self.cfg.prompt)
            elif self.cfg.sds_loss_type == "stable_sr_sds":
                self.sds_loss = SDSLoss3DGS_StableSR(
                    model_checkpoint_path=self.cfg.stable_sr_checkpoint_path,
                    model_config_path=self.cfg.stable_sr_config_path,
                    # these are mainly for debugging
                    encoder_checkpoint_path=self.cfg.encoder_checkpoint_path,
                    encoder_config_path=self.cfg.encoder_configh_path,
                    render_dir=self.render_dir,
                )

        if self.cfg.classic_loss_type == "l2loss":
            self.mse_loss = torch.nn.MSELoss()
        elif self.cfg.classic_loss_type == "l1loss":
            self.mae_loss = torch.nn.L1Loss()

        if self.cfg.noise_scheduler_type == "linear":
            self.noise_scheduler = set_linear_noise_schedule(
                self.cfg.iterations, self.cfg.min_noise_step, self.cfg.max_noise_step
            )

        if self.cfg.model_type == "3dgs":
            self.rasterize_fnc = rasterization
        elif self.cfg.model_type == "2dgs":
            self.rasterize_fnc = rasterization_2dgs

    def rasterize_splats(self) -> Tuple[Tensor, Tensor, Dict]:
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
            width=self.render_width,
            height=self.render_height,
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
                [self.focal, 0, self.render_width / 2],
                [0, self.focal, self.render_height / 2],
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

    def do_color_correction(self, out_img: Tensor):
        if cfg.color_correction_mode == "adain":
            out_img = adaptive_instance_normalization(
                out_img.permute(2, 0, 1).unsqueeze(0),
                self.training_img.unsqueeze(0),
            )
            out_img = out_img.squeeze(0).permute(1, 2, 0)
        elif cfg.color_correction_mode == "wavelet":
            out_img = wavelet_reconstruction(
                out_img.permute(2, 0, 1).unsqueeze(0),
                F.interpolate(
                    self.training_img.unsqueeze(0),
                    (out_img.size(0), out_img.size(1)),
                    mode="bicubic",
                ),
            )
            out_img = out_img.squeeze(0).permute(1, 2, 0)
        return out_img.clamp(0.0, 1.0)

    def validate(self, iteration):
        print("Validating...")
        with torch.no_grad():
            renders, _, _ = self.rasterize_splats()
            out_img = self.do_color_correction(renders[0])
            frame = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(frame).save(f"{self.render_dir}/val_render_{iteration}.png")
            psnr = self.psnr(out_img, self.validation_image.permute(1, 2, 0))
            ssim = self.ssim(
                out_img.permute(2, 0, 1).unsqueeze(0),
                self.validation_image.unsqueeze(0),
            )
            print(f"Val PSNR: {psnr.item()}, Val SSIM: {ssim.item()}")
            with open(f"{self.stats_dir}/val_{iteration}.json", "w") as f:
                json.dump({"psnr": psnr.item(), "ssim": ssim.item()}, f)

    def train(self):
        times = [0] * 2  # rasterization, backward

        begin = 0
        end = self.cfg.iterations
        losses = []
        psnrs = []
        ssims = []
        grad_norms = []
        learning_rates = []
        prev_render = None
        diff = None

        if cfg.debug_training:
            # clone params
            param_info = {
                name: {"previous": param.detach().clone(), "diff": 0.0, "updates": 0}
                for name, param in self.splats.items()
            }

        pbar = tqdm.tqdm(range(begin, end))
        for i in pbar:
            start = time.time()
            renders, _, info = self.rasterize_splats()
            if self.cfg.use_strategy:
                self.cfg.strategy.step_pre_backward(
                    self.splats, self.optimizers, self.strategy_state, i, info
                )
            out_img = self.do_color_correction(renders[0])

            # how much pixels have changed compared to previous iteration
            if prev_render is not None:
                diff = (prev_render - out_img).clamp(0.0, 1.0)

            torch.cuda.synchronize()
            times[0] += time.time() - start

            if cfg.use_gaussian_sr:
                resolution = (self.height, self.width)
                downscaled_render = F.interpolate(
                    out_img.permute(2, 0, 1).unsqueeze(0),
                    resolution,
                    mode="bicubic",
                ).clamp(0.0, 1.0)
                if cfg.classic_loss_type == "l1loss":
                    # we need to minimize
                    # this one wants [B, C, H, W]
                    ssim_loss = (
                        1.0
                        - fused_ssim(downscaled_render, self.training_img.unsqueeze(0))
                        * self.cfg.ssim_lambda
                    )
                    loss = (
                        self.mae_loss(downscaled_render, self.training_img.unsqueeze(0))
                        + ssim_loss
                    )
                elif cfg.classic_loss_type == "l2loss":
                    loss = self.mse_loss(
                        downscaled_render, self.training_img.unsqueeze(0)
                    )
                sds = 0.0
                if self.cfg.sds_loss_type != "none":
                    if self.cfg.noise_scheduler_type == "collapsing":
                        min_step = compute_collapsing_noise_step(
                            200, 300, i / self.cfg.iterations
                        )
                        max_step = compute_collapsing_noise_step(
                            500, 980, i / self.cfg.iterations
                        )
                    elif self.cfg.noise_scheduler_type == "annealing":
                        assert self.cfg.noise_step_anealing > 0
                        min_step = int(
                            max(
                                self.cfg.min_noise_step,
                                (self.cfg.max_noise_step - 1)
                                - i / self.cfg.noise_step_anealing,
                            )
                        )
                        max_step = self.cfg.max_noise_step
                    else:
                        min_step = self.cfg.min_noise_step
                        max_step = self.cfg.max_noise_step
                    if self.cfg.sds_loss_type in ("deepfloyd_sds", "deepfloyd_sdi"):
                        sds = self.sds_loss(
                            images=out_img.permute(2, 0, 1).unsqueeze(0),
                            original=self.training_img.unsqueeze(0),
                            min_step=min_step,
                            max_step=max_step,
                            lowres_noise_level=self.cfg.lowres_noise_level,
                            scheduler_timestep=self.noise_scheduler[i]
                            if self.cfg.noise_scheduler_type == "linear"
                            else None,
                            downscale_condition=False,
                            guidance_scale=cfg.guidance_scale,
                        ).squeeze()
                    elif self.cfg.sds_loss_type == "stable_sr_sds":
                        sds = self.sds_loss(
                            render=out_img.permute(2, 0, 1).unsqueeze(0),
                            condition=self.training_img.unsqueeze(0),
                            min_noise_step=min_step,
                            max_noise_step=max_step,
                            iteration=i,
                        ).squeeze()
                loss += sds * self.cfg.sds_lambda

            else:
                if cfg.classic_loss_type == "l1loss":
                    # this one wants [B, C, H, W]
                    ssim_loss = (
                        1.0
                        - fused_ssim(
                            out_img.permute(2, 0, 1).unsqueeze(0),
                            self.training_img.unsqueeze(0),
                        )
                        * self.cfg.ssim_lambda
                    )
                    loss = (
                        self.mae_loss(
                            out_img.permute(2, 0, 1).unsqueeze(0),
                            self.training_img.unsqueeze(0),
                        )
                        + ssim_loss
                    )
                elif cfg.classic_loss_type == "l2loss":
                    loss = self.mse_loss(
                        out_img.permute(2, 0, 1).unsqueeze(0),
                        self.training_img.unsqueeze(0),
                    )

            start = time.time()
            loss.backward()

            torch.cuda.synchronize()
            times[1] += time.time() - start

            if self.cfg.use_strategy:
                self.cfg.strategy.step_post_backward(
                    self.splats, self.optimizers, self.strategy_state, i, info
                )

            # this one wants [H, W, C]
            psnr = self.psnr(out_img, self.validation_image.permute(1, 2, 0))
            # this one wants [B, C, H, W]
            ssim = self.ssim(
                out_img.permute(2, 0, 1).unsqueeze(0),
                self.validation_image.unsqueeze(0),
            )

            if self.cfg.grad_clipping > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in self.splats.values()], self.cfg.grad_clipping
                )

            # stats
            losses.append(loss.item())
            psnrs.append(psnr.item())
            grad_norms.append(calculate_grad_norm(self.optimizers))
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
                pred = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
                orig = (
                    self.validation_image.permute(1, 2, 0).detach().cpu().numpy() * 255
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
                    json.dump(
                        {
                            "psnr": psnr.item(),
                            "ssim": ssim.item(),
                            "num_splats": self.splats["means"].size(0),
                        },
                        f,
                    )

            if i in [idx - 1 for idx in self.cfg.valiation_steps] or i == end - 1:
                self.validate(i)

        print(f"Total(s):\nRasterization: {times[0]:.3f}, Backward: {times[1]:.3f}")
        print(
            f"Per step(s):\nRasterization: {times[0]/self.cfg.iterations:.5f}, Backward: {times[1]/self.cfg.iterations:.5f}"
        )
        print(f"Num splats: {self.splats['means'].size(0)}")
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
        trainer.validate(0)
    else:
        trainer.train()


if __name__ == "__main__":
    cfg = tyro.cli(Config2D)
    main(cfg)
