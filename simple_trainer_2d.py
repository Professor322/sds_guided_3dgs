import math
import os
import time
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


class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 2000,
        ckpt_path: str = "",
        lr: float = 0.01,
        results_dir: str = "",
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points
        self.iter = 0
        self.frames = []
        self.ckpt_dir = f"{results_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        # self.stats_dir = f"{results_dir}/stats"
        # os.makedirs(self.stats_dir, exist_ok=True)
        # self.render_dir = f"{results_dir}/renders"
        # os.makedirs(self.render_dir, exist_ok=True)

        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        if ckpt_path:
            self._load_gaussians(ckpt_path)
        else:
            self._init_gaussians()

        self.optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )

        if ckpt_path:
            self.optimizer.load_state_dict(self.optimizer_state_dict)

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

    def train(
        self,
        iterations: int = 1000,
        save_imgs: bool = False,
        model_type: Literal["3dgs", "2dgs"] = "3dgs",
    ):
        mse_loss = torch.nn.MSELoss()
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

        if model_type == "3dgs":
            rasterize_fnc = rasterization
        elif model_type == "2dgs":
            rasterize_fnc = rasterization_2dgs

        begin = self.iter
        end = begin + iterations + 1
        for iter in range(begin, end):
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
            loss = mse_loss(out_img, self.gt_image)
            self.optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[1] += time.time() - start
            self.optimizer.step()
            print(f"Iteration {iter + 1}/{end}, Loss: {loss.item()}")

            if save_imgs and iter % 50 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
            if iter in [i - 1 for i in cfg.save_steps] or iter == end - 1:
                print(f"Saving checkpoint at: {iter}")
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
                torch.save(to_save, f"{self.ckpt_dir}/ckpt_{iter}.pt")

        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
        print(f"Total(s):\nRasterization: {times[0]:.3f}, Backward: {times[1]:.3f}")
        print(
            f"Per step(s):\nRasterization: {times[0]/iterations:.5f}, Backward: {times[1]/iterations:.5f}"
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
    if cfg.img_path:
        gt_image = image_path_to_tensor(cfg.img_path)
    else:
        gt_image = torch.ones((cfg.height, cfg.width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: cfg.height // 2, : cfg.width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[cfg.height // 2 :, cfg.width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

    trainer = SimpleTrainer(
        gt_image=gt_image,
        num_points=cfg.num_points,
        lr=cfg.lr,
        ckpt_path=cfg.ckpt_path,
        results_dir=cfg.results_dir,
    )
    trainer.train(
        iterations=cfg.iterations,
        save_imgs=cfg.save_imgs,
        model_type=cfg.model_type,
    )


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
