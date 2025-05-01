from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from typing_extensions import Literal, assert_never


@dataclass
class Config3D:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    # When gaussian SR is enabled, checkpoint should be provided
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(
        default_factory=lambda: [3_000, 7_000, 10_000, 15_000, 30_000]
    )
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [3_000, 7_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2
    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = True

    lpips_net: Literal["vgg", "alex"] = "alex"

    # Method to enhance quality of the pretrained
    # low resolution gaussian splats
    gaussian_sr: bool = False
    # Scale factor to enhance gaussian splats
    scale_factor: float = 0.0
    # Type of Score Disitllation Sampling loss
    sds_loss_type: Literal["sdi", "sds", "none"] = "none"
    # Controls influence of Score Distillation Sampling
    sds_lambda: float = 0.001
    # prompt for diffussion model
    prompt: str = ""
    # minimum noise step for forward diffusion process
    min_noise_step: int = 20
    # maximum noise step for forward diffusion process
    max_noise_step: int = 980
    # amount of noise applied to the low resolution condition image
    condition_noise: float = 0.0
    # type of noise scheduling for forward diffusion process
    noise_scheduler_type: Literal["collapsing", "linear", "annealing", "none"] = "none"
    # guidance scale for denoising process
    guidance_scale: float = 10.0
    # annealing coefficient for "annealing" type of noise scheduler
    noise_step_annealing: int = 100
    # dropout of splats during densification process
    # applicable right now only to default strategy
    densification_dropout: float = 0.7
    # general loss type, gaussian sr uses l2 loss in their paper
    # without any D-SSIM coefficients
    loss_type: Literal["l2loss", "l1loss"] = "l1loss"
    # upscale suffix for validation and training
    upscale_suffix: Literal["bilinear", "bicubic", "sr", ""] = ""
    # maximum allowed number of splats
    max_splats: int = 1_000_000

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


@dataclass
class Config2D:
    # input image width, leave it equal to 0
    # in case you want to preserve original size
    width: int = 0
    # input image height, leave it equal to 0
    # in case you want to preserve original size
    height: int = 0
    # number of splats to fit
    num_points: int = 100_000
    # whether to store images during saving epoch
    save_imgs: bool = True
    # number of iterawtions to run
    iterations: int = 1_000
    # learning rate for training
    lr: float = 0.01
    # type of splat model to run
    model_type: Literal["3dgs", "2dgs"] = "3dgs"
    # checkpoint save steps
    save_steps: List[int] = field(default_factory=lambda: [1_000, 3_000, 7_000, 30_000])
    # validation steps
    valiation_steps: List[int] = field(
        default_factory=lambda: [1_000, 3_000, 7_000, 30_000]
    )
    # path to image used for training
    training_image_path: str = ""
    # path to image used for validation
    validation_image_path: str = ""
    # checkpoint path
    ckpt_path: str = ""
    # directory for saving results
    results_dir: str = "results_2d"
    # how often to update training meta data
    show_steps: int = 200
    # coefficient for SDS loss
    sds_lambda: float = 1.0
    # noise level for conditional image
    lowres_noise_level: float = 0.0
    # minimum step for forward diffusion process
    min_noise_step: int = 20
    # maximum step for forward diffusion process
    max_noise_step: int = 980
    # prompt for diffusion model in case
    # of deepfloyd sds
    prompt: str = ""
    # classic loss type to use, for l1loss we will loss add lambda * d-ssim
    classic_loss_type: Literal["l2loss", "l1loss"] = "l2loss"
    # type of sds loss in case of gaussianSR
    sds_loss_type: Literal[
        "deepfloyd_sds", "deepfloyd_sdi", "stable_sr_sds", "none"
    ] = "none"
    # guidance scale for the diffusion model incase of deepfloyd
    guidance_scale: float = 10.0
    # whether to use densification process or not
    use_strategy: bool = False
    # noise scheduler type
    noise_scheduler_type: Literal["collapsing", "linear", "annealing", "none"] = "none"
    # for densification process
    strategy: DefaultStrategy = field(default_factory=DefaultStrategy)
    # whether to use checkpoint for validation
    validate: bool = False
    # in altering fashion: one iteration of sds,
    # then one iteration of mse
    use_altering_loss: bool = False
    # this is to track if parameters are being changed
    debug_training: bool = False
    # gradient clipping option 0.0 - means disabled
    grad_clipping: float = 0.0
    # implementing method from the paper
    use_gaussian_sr: bool = False
    # scale factor for gaussian sr
    scale_factor: int = 1
    # noise step annealing when annealing
    # noise scheduler is enabled
    noise_step_anealing: int = 0
    # lambda for ssim coefficient
    ssim_lambda: float = 0.2
    # splat dropout during during default
    # densification process
    densification_dropout: float = 0.0
    # color correction mode
    color_correction_mode: Literal["adain", "wavelet", "none"] = "none"
    # path to StableSR config
    stable_sr_config_path: str = (
        "StableSR/configs/stableSRNew/v2-finetune_text_T_512.yaml"
    )
    # path to StableSR checkpoint
    stable_sr_checkpoint_path: str = "StableSR/stablesr_000117.ckpt"


@dataclass
class DatasetAugmentationConfig:
    data_dir: str = "./data/360_v2/my_bicycle"
    data_factor: int = 4
    upscale_type: Literal["bilinear", "bicubic", "sr"] = "bicubic"
    scale_factor: int = 4
    prompt: str = ""
    condtion_noise_level_steps: int = 50
