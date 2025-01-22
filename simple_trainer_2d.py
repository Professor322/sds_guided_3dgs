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
from guidance import SDSLoss3DGS
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as VF
from PIL import Image


@dataclass
class Config:
    width: int = 256
    height: int = 256
    num_points: int = 100_000
    save_imgs: bool = True
    iterations: int = 1_000
    lr: float = 0.01
    model_type: Literal["3dgs", "2dgs"] = "3dgs"
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    img_path: str = ""
    ckpt_path: str = ""
    results_dir: str = "results_2d"
    show_steps: int = 10
    use_sds_loss: bool = False
    use_fused_loss: bool = False
    lmbd: float = 1.0
    save_images: bool = False
    patch_image: bool = False
    batch_size: int = 64
    # noise level for conditional image
    lowres_noise_level: float = 0.75
    # minimum step for forward diffusion process
    min_noise_step: int = 20
    # maximum step for forward diffusion process
    max_noise_step: int = 980
    # instead randomly sampling noise we can
    # linearly changing applied noise
    use_noise_scheduler: bool = False
    show_plots: bool = False
    current_rendering_as_condition: bool = False


class OneImageDataset(Dataset):
    def __init__(
        self, image_path, patch_size=64, dataset_len=1000, resize=True, train=True
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

    def __getitem__(self, idx):
        # C, H, W -> H, W, C
        i, j, h, w = transforms.RandomCrop.get_params(
            self.img, output_size=(self.patch_size, self.patch_size)
        )
        if self.train:
            patch_real = VF.crop(self.img, i, j, h, w).permute(1, 2, 0)
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
        self.one_image_dataset = OneImageDataset(image_path=cfg.img_path)
        self.num_points = self.cfg.num_points
        self.iter = 0
        self.frames = []
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
            self.optimizer.load_state_dict(self.optimizer_state_dict)
        if self.cfg.use_sds_loss:
            self.sds_loss = SDSLoss3DGS()
            self.dataloader = DataLoader(
                self.one_image_dataset, batch_size=cfg.batch_size, num_workers=0
            )
        self.mse_loss = torch.nn.MSELoss()
        if self.cfg.use_noise_scheduler:
            self.noise_scheduler = self.set_linear_time_strategy(
                self.cfg.iterations, self.cfg.min_noise_step, self.cfg.max_noise_step
            )

    def set_linear_time_strategy(
        self, output_shape, min_diffusion_steps=20, max_diffusion_steps=980
    ):
        return (
            torch.linspace(min_diffusion_steps, max_diffusion_steps, output_shape)
            .flip(0)
            .to(torch.long)
        )

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

        begin = self.iter
        end = begin + self.cfg.iterations
        losses = []
        psnrs = []
        grad_norms = []
        self.one_image_dataset.img = self.one_image_dataset.img.to(self.device)

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
            torch.cuda.synchronize()
            times[0] += time.time() - start
            # calculate loss
            mse_loss = self.mse_loss(
                out_img, self.one_image_dataset.img.permute(1, 2, 0)
            )

            if self.cfg.use_sds_loss:
                # H, W, C -> C, H, W
                self.dataloader.dataset.training_img = out_img.permute(2, 0, 1)
                pred, real = next(iter(self.dataloader))
                pred = pred.to(self.device).permute(0, 3, 1, 2)
                real = real.to(self.device).permute(0, 3, 1, 2)
                sds = self.sds_loss(
                    images=pred,
                    original=pred if self.cfg.current_rendering_as_condition else real,
                    min_step=self.cfg.min_noise_step,
                    max_step=self.cfg.max_noise_step,
                    lowres_noise_level=self.cfg.lowres_noise_level,
                    scheduler_timestep=self.noise_scheduler[i]
                    if self.cfg.use_noise_scheduler
                    else None,
                )

            if self.cfg.use_fused_loss and self.cfg.use_sds_loss:
                loss = mse_loss + sds
            elif self.cfg.use_sds_loss:
                loss = sds
            else:
                loss = mse_loss

            self.optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[1] += time.time() - start
            self.optimizer.step()
            psnr = self.psnr(out_img, self.one_image_dataset.img.permute(1, 2, 0))
            losses.append(loss.item())
            psnrs.append(psnr.item())
            grad_norms.append(self.calculate_grad_norm())
            print(f"Iteration {i + 1}/{end}, Loss: {loss.item()}, PSNR: {psnr.item()}")

            if i % self.cfg.show_steps == 0:
                if self.cfg.show_plots:
                    clear_output(wait=True)
                pred = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
                orig = (
                    self.one_image_dataset.img.permute(1, 2, 0).detach().cpu().numpy()
                    * 255
                ).astype(np.uint8)
                # Create the figure with additional row for gradient norm plot
                fig, axes = plt.subplots(3, 2, figsize=(12, 15))

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

                # Hide the last unused subplot (bottom-right corner)
                axes[2, 1].axis("off")
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
                    frames.append(pred)
            if i in [idx - 1 for idx in self.cfg.save_steps] or i == end - 1:
                print(f"Saving checkpoint at: {i}")
                to_save = {
                    "optimizer": self.optimizer.state_dict(),
                    "iter": iter,
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

        if self.cfg.save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            frames[0].save(
                f"{self.render_dir}/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
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
