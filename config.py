from dataclasses import dataclass, field
from typing import List, Literal
from gsplat.strategy import DefaultStrategy


@dataclass
class Config:
    render_width: int = 256
    render_height: int = 256
    # this is training width and height
    width: int = 256
    height: int = 256
    num_points: int = 100_000
    save_imgs: bool = True
    iterations: int = 1_000
    lr: float = 0.01
    model_type: Literal["3dgs", "2dgs"] = "3dgs"
    save_steps: List[int] = field(
        default_factory=lambda: [500, 700, 1_000, 3_000, 7_000, 30_000]
    )
    img_path: str = ""
    ckpt_path: str = ""
    results_dir: str = "results_2d"
    show_steps: int = 200
    use_sds_loss: bool = False
    use_sdi_loss: bool = False
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
    # this one will gradually collaps t_min and t_max
    collapsing_noise_scheduler: bool = False
    show_plots: bool = False
    base_render_as_cond: bool = False
    use_lr_scheduler: bool = False
    downscale_condition: bool = False
    prompt: str = ""
    guidance_scale: float = 10.0
    use_classic_mse_loss: bool = False
    use_downscaled_mse_loss: bool = False
    use_strategy: bool = False
    # for pruning
    strategy: DefaultStrategy = field(default_factory=DefaultStrategy)
    validate: bool = False
    # in altering fashion: one iteration of sds,
    # then one iteration of mse
    use_altering_loss: bool = False
    use_ssim_loss: bool = False

    debug_training: bool = False
    grad_clipping: float = 0.0
    # implementing method from the paper
    use_gaussian_sr: bool = False
    noise_step_anealing: int = 0
