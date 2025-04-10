from config import Config
import tyro
import os
import json
import glob

IMG_PATH = "data/360_v2/bicycle/images_8/_DSC8679.JPG"
# IMG_PATH = "render_bicycle_hard_prompt.png"
ITERATIONS = 1_000
# sometimes can hit oom, so we have to reduce it
BATCH_SIZE = 1
DEBUG = False
GET_PLOTS = False
TOP_PSNRS = False

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=train_2d
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1

module purge
module load Python gnu10

source deactivate
source activate gsplat_sds

nvidia-smi
echo "starting 2d training"
"""

SBATCH_FILENAME = "2d_training_generated.sbatch"


def classic_splats_with_validation_2d(cfg: Config):
    # validate on the original image downscaled to 256x256

    validataion_width = 256
    validation_height = 256
    validation_img_path = "data/360_v2/bicycle/images_8/_DSC8679.JPG"
    num_points = [10_000, 20_000, 30_000, 40_000]

    iterations = 30_000
    params = [
        # to train downscaled bicycle
        {
            "resolution": (64, 64),
            "img_path": "data/360_v2/bicycle/images_8/_DSC8679.JPG",
        },
        # # to train original one
        # {
        #     "resolution": (256, 256),
        #     "img_path": "data/360_v2/bicycle/images_8/_DSC8679.JPG",
        # },
        # # to train on upscaled bicycle 64->256 using SR model
        # {"resolution": (256, 256), "img_path": "render_bicycle_hard_prompt_25.png"},
        # # to train on upscaled bicycle 64->256 using interpolation
        # {"resolution": (256, 256), "img_path": "interpolated_bicycle.png"},
    ]
    classic_run_args = [
        "python3 simple_trainer_2d.py",
        f"--iterations {iterations}",
        f"--use-classic-mse_loss",
    ]
    result_dirs = []
    # first start a batch of training
    for param in params:
        for num_point in num_points:
            # execution
            current_run_args = classic_run_args.copy()
            width, height = param["resolution"]
            img_path = param["img_path"]
            if "interpolated" in img_path:
                image_type = "upscale_interpolated"
            elif "render" in img_path:
                image_type = "upscale_sr"
            else:
                image_type = "original"
            result_dir = f"results_2d_classic_{width}x{width}_{image_type}_num_points_{num_point}"

            current_run_args.append(f"--num-points {num_point}")
            current_run_args.append(f"--width {width}")
            current_run_args.append(f"--height {height}")
            current_run_args.append(f"--img-path {img_path}")
            current_run_args.append(f"--results-dir {result_dir}")
            result_dirs.append(result_dir)
            file_content = (
                SBATCH_TEMPLATE
                + "\n"
                + f"echo '{result_dir}'\n"
                + "srun "
                + " ".join(current_run_args)
            )

            # validatation
            if cfg.validate:
                validation_run_args = classic_run_args.copy()
                checkpoint_path = f"{result_dir}/ckpts/ckpt_{iterations - 1}.pt"
                validation_run_args.append(f"--ckpt-path {checkpoint_path}")
                validation_run_args.append(f"--width {validataion_width}")
                validation_run_args.append(f"--height {validation_height}")
                validation_run_args.append(f"--img-path {validation_img_path}")
                validation_run_args.append(f"--results-dir {result_dir}")
                validation_run_args.append("--validate")
                file_content += "\n" + "srun " + " ".join(validation_run_args)

            if DEBUG:
                print(file_content)
            with open(SBATCH_FILENAME, "w") as file:
                file.write(file_content)
            if not DEBUG and not GET_PLOTS and not TOP_PSNRS:
                os.system(f"sbatch {SBATCH_FILENAME}")

    return result_dirs


def sds_experiments_2d(cfg: Config, default_run_args):
    noise_levels = [0.0]
    # checkpoints = [2999, 6999, 29999]
    checkpoints = [29999]
    # grad_clipping_values = [1.0, 20.0, 50.0, 200.0]
    grad_clipping_values = [0.0]
    num_points = [10_000]
    # num_points = [10_000, 20_000]
    easy_prompt = "bicycle near the bench"
    hard_prompt = (
        "A surreal outdoor scene featuring a "
        "white bicycle seamlessly blending into "
        "a black park bench, creating an optical illusion. "
        "The front wheel of the bike rests on the grass, "
        "while the rear wheel appears to be aligned perfectly "
        "with the backrest of the bench, making it look as if "
        "the bicycle is embedded into the structure. The setting "
        "is a peaceful park with lush green grass, a paved pathway, "
        "and dense foliage in the background. The atmosphere is calm "
        "and natural, with soft, diffused lighting enhancing the realism of the scene."
    )
    prompts = [easy_prompt]
    guidance_scales = [10.0]
    result_dirs = []
    min_step = 20
    max_step = 980
    lmbds = [0.001, 0.01, 0.1]
    for lmbd in lmbds:
        for noise_level in noise_levels:
            for checkpoint in checkpoints:
                for grad_clipping_value in grad_clipping_values:
                    for num_point in num_points:
                        for prompt in prompts:
                            for guidance_scale in guidance_scales:
                                current_run_args = default_run_args.copy()
                                checkpoint_path = (
                                    f"results_2d_classic_{cfg.width}x{cfg.height}"
                                )
                                checkpoint_path += f"_original_num_points_{num_point}/ckpts/ckpt_{checkpoint}.pt"
                                current_run_args.append(
                                    f"--ckpt-path {checkpoint_path}"
                                )
                                result_dir = f"results_2d_low_res_noise_level_{str(noise_level).replace('.', '_')}_{checkpoint}_min{min_step}_max{max_step}"
                                if cfg.use_sdi_loss:
                                    current_run_args.append("--use-sdi-loss")
                                    result_dir += "_sdi_loss"
                                if cfg.use_sds_loss:
                                    current_run_args.append("--use-sds-loss")
                                    result_dir += "_sds_loss"
                                if cfg.base_render_as_cond:
                                    current_run_args.append("--base-render-as-cond")
                                    result_dir += "_base_render_as_cond"
                                if cfg.use_downscaled_mse_loss:
                                    current_run_args.append(
                                        f"--use-downscaled-mse-loss"
                                    )
                                    result_dir += "_downscaled_mse_loss"
                                if cfg.use_fused_loss:
                                    current_run_args.append("--use-fused-loss")
                                    result_dir += "_fused_loss"
                                if cfg.use_altering_loss:
                                    current_run_args.append("--use-altering-loss")
                                    result_dir += "_altering_loss"
                                if cfg.use_ssim_loss:
                                    current_run_args.append("--use-ssim-loss")
                                    result_dir += "_ssim_loss"
                                if cfg.debug_training:
                                    current_run_args.append("--debug-training")
                                    result_dir += "_debug"
                                if grad_clipping_value > 0.0:
                                    result_dir += f"_grad_clip_{str(grad_clipping_value).replace('.', '_')}"
                                    current_run_args.append(
                                        f"--grad-clipping {grad_clipping_value}"
                                    )
                                if lmbd > 0.0:
                                    result_dir += f"_lmbd_{str(lmbd).replace('.', '_')}"
                                    current_run_args.append(f"--lmbd {lmbd}")
                                if cfg.noise_step_anealing > 0:
                                    result_dir += f"_anealing_{cfg.noise_step_anealing}"
                                    current_run_args.append(
                                        f"--noise-step-anealing {cfg.noise_step_anealing}"
                                    )
                                if cfg.use_gaussian_sr:
                                    result_dir += f"_gaussian_sr"
                                    current_run_args.append(f"--use-gaussian-sr")
                                if prompt != "":
                                    current_run_args.append(f'--prompt "{prompt}"')
                                    result_dir += f"_{'easy' if prompt == easy_prompt else 'hard'}_prompt"
                                if guidance_scale > 0.0:
                                    current_run_args.append(
                                        f"--guidance-scale {guidance_scale}"
                                    )
                                    result_dir += f"_guidance_scale_{str(guidance_scale).replace('.', '_')}"
                                if cfg.use_strategy:
                                    result_dir += "_pruning"
                                    current_run_args.append(f"--use-strategy")

                                result_dir += f"_num_points_{num_point}"
                                current_run_args.append(f"--num-points {num_point}")
                                current_run_args.append(
                                    f"--lowres-noise-level {noise_level}"
                                )
                                current_run_args.append(f"--results-dir {result_dir}")
                                current_run_args.append(f"--min-noise-step {min_step}")
                                current_run_args.append(f"--max-noise-step {max_step}")
                                current_run_args.append(f"--width {cfg.width}")
                                current_run_args.append(f"--height {cfg.height}")
                                current_run_args.append(
                                    f"--render-width {cfg.render_width}"
                                )
                                current_run_args.append(
                                    f"--render-height {cfg.render_height}"
                                )
                                file_content = (
                                    SBATCH_TEMPLATE
                                    + "\n"
                                    + f"echo '{result_dir}'\n"
                                    + " ".join(current_run_args)
                                )
                                result_dirs.append(result_dir)
                                if DEBUG:
                                    print(file_content)
                                with open(SBATCH_FILENAME, "w") as file:
                                    file.write(file_content)
                                if not DEBUG and not GET_PLOTS and not TOP_PSNRS:
                                    os.system(f"sbatch {SBATCH_FILENAME}")
    return result_dirs


def main(
    cfg: Config,
) -> None:
    # modify parameters for testing
    cfg.base_render_as_cond = False
    cfg.use_sds_loss = True
    cfg.width = 64
    cfg.height = 64
    cfg.render_height = 256
    cfg.render_width = 256
    cfg.use_gaussian_sr = True
    # cfg.use_fused_loss = True
    # cfg.use_downscaled_mse_loss = True
    cfg.use_strategy = True
    cfg.noise_step_anealing = 100
    # cfg.use_ssim_loss = True
    # cfg.use_altering_loss = True
    # cfg.collapsing_noise_scheduler = True
    # cfg.use_lr_scheduler = True
    # cfg.use_sdi_loss = True
    do_sds_experiments = True
    do_classic_experiments = False

    default_run_args = [
        "python3 simple_trainer_2d.py",
        f"--img-path {IMG_PATH}",
        f"--iterations {ITERATIONS}",
        f"--num-points {cfg.num_points}",
        f"--batch-size {BATCH_SIZE}",
    ]
    result_dirs = []
    if do_sds_experiments:
        result_dirs += sds_experiments_2d(cfg, default_run_args)
    if do_classic_experiments:
        result_dirs += classic_splats_with_validation_2d(cfg)

    result_dirs = glob.glob("/home/nskochetkov/sds_guided_3dgs/results_2d_low*")
    if GET_PLOTS:
        print("Getting plots...")
        for result_dir in result_dirs:
            print(f"Getting plots for {result_dir}")
            os.system(
                f"scp nskochetkov@cluster.hpc.hse.ru:/home/nskochetkov/sds_guided_3dgs/{result_dir}/stats/training_plots.png {result_dir}.png"
            )
    if TOP_PSNRS:
        print("Getting psnrs...")
        psnrs_to_dirs = []
        result_dirs = glob.glob("/home/nskochetkov/sds_guided_3dgs/results_2d_low*")
        for result_dir in result_dirs:
            filename = f"{result_dir}/stats/step999.json"
            if not os.path.exists(filename):
                continue
            with open(
                filename,
                "r",
            ) as file:
                # mitigation, as I messed up json format
                data = file.read().split("}")[0] + "}"
                results = json.loads(data)
                psnrs_to_dirs.append((results["psnr"], result_dir))

        psnrs_to_dirs = sorted(psnrs_to_dirs, reverse=True)
        for psnr_to_dir in psnrs_to_dirs:
            print(psnr_to_dir)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
