from config import Config
import tyro
import os
import json
import glob

CHECKPOINT = 2999
IMG_PATH = "data/360_v2/bicycle/images_8/_DSC8679.JPG"
# IMG_PATH = "render_bicycle_hard_prompt.png"
CHECKPOINT_PATH = (
    f"/home/nskochetkov/sds_guided_3dgs/results_2d/ckpts/ckpt_{CHECKPOINT}.pt"
)
MAX_STEP = 980
MIN_STEP = 20
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
#SBATCH --cpus-per-task=2

module purge
module load Python gnu10

source deactivate
source activate gsplat_sds

nvidia-smi
echo "starting 2d training"
"""

SBATCH_FILENAME = "2d_training_generated.sbatch"


def different_checkpoints_exp(cfg: Config, default_run_args):
    # checkpoints = [999, 2999, 6999, 29999]
    checkpoints = [499, 699]
    easy_prompt = "bicycle"
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
    # noise_levels = [0.25, 0.5, 0.75]
    # guidance_scales = [10.0, 25.0, 50.0, 100.0]
    noise_levels = [0.25]
    guidance_scales = [10.0, 25.0]
    result_dirs = []
    for checkpoint in checkpoints:
        for prompt in [easy_prompt, hard_prompt]:
            for noise_level in noise_levels:
                for guidance_scale in guidance_scales:
                    current_run_args = default_run_args.copy()
                    checkpoint_path = f"results_2d_classic_{cfg.width}x{cfg.height}_{'upscaled' if 'render' in IMG_PATH else 'original'}/ckpts/ckpt_{checkpoint}.pt"
                    current_run_args.append(f"--ckpt-path {checkpoint_path}")
                    is_easy_prompt = len(prompt.split(" ")) == 1
                    result_dir = f"results_2d_low_res_noise_level_{str(noise_level).replace('.', '_')}_{checkpoint}_{cfg.height}x{cfg.width}_upscaled"
                    result_dir += f'_{"easy" if is_easy_prompt else "hard"}'
                    result_dir += (
                        f"_prompt_guidance_{str(guidance_scale).replace('.', '_')}"
                    )
                    if cfg.use_strategy:
                        result_dir += "_pruning" if cfg.use_strategy else ""
                        current_run_args.append("--use-strategy")
                    if cfg.use_sdi_loss:
                        result_dir += "_sdi_loss"
                    if cfg.use_downscaled_mse_loss:
                        current_run_args.append(f"--use-downscaled-mse-loss")
                        result_dir += "_downscaled_mse_loss"
                    if cfg.use_fused_loss:
                        result_dir += "_fused_loss"
                    current_run_args.append(f"--lowres-noise-level {noise_level}")
                    current_run_args.append(f'--prompt "{prompt}"')
                    current_run_args.append(f"--guidance-scale {guidance_scale}")
                    current_run_args.append(f"--width {cfg.width}")
                    current_run_args.append(f"--height {cfg.height}")
                    if cfg.base_render_as_cond:
                        current_run_args.append("--base-render-as-cond")
                        result_dir += "_base_render_as_cond"
                    current_run_args.append(f"--results-dir {result_dir}")
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


def classic_splats_with_validation(cfg: Config):
    # validate on the original image downscaled to 256x256
    validataion_width = 256
    validation_height = 256
    validation_img_path = "data/360_v2/bicycle/images_8/_DSC8679.JPG"

    iterations = 30_000
    params = [
        # to train downscaled bicycle
        {
            "resolution": (64, 64),
            "img_path": "data/360_v2/bicycle/images_8/_DSC8679.JPG",
        },
        # to train original one
        {
            "resolution": (256, 256),
            "img_path": "data/360_v2/bicycle/images_8/_DSC8679.JPG",
        },
        # to train on upscaled bicycle 64->256 using SR model
        {"resolution": (256, 256), "img_path": "render_bicycle_hard_prompt_25.png"},
        # to train on upscaled bicycle 64->256 using interpolation
        {"resolution": (256, 256), "img_path": "interpolated_bicycle.png"},
    ]
    classic_run_args = [
        "python3 simple_trainer_2d.py",
        f"--iterations {iterations}",
        f"--use-classic-mse_loss",
    ]
    result_dirs = []
    # first start a batch of training
    for param in params:
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
        result_dir = f"results_2d_classic_{width}x{width}_{image_type}"

        current_run_args.append(f"--width {width}")
        current_run_args.append(f"--height {height}")
        current_run_args.append(f"--img-path {img_path}")
        current_run_args.append(f"--results-dir {result_dir}")
        result_dirs.append(result_dir)

        # validatation
        validation_run_args = classic_run_args.copy()
        checkpoint_path = f"{result_dir}/ckpts/ckpt_{iterations - 1}.pt"
        validation_run_args.append(f"--ckpt-path {checkpoint_path}")
        validation_run_args.append(f"--width {validataion_width}")
        validation_run_args.append(f"--height {validation_height}")
        validation_run_args.append(f"--img-path {validation_img_path}")
        validation_run_args.append(f"--results-dir {result_dir}")
        validation_run_args.append("--validate")

        file_content = (
            SBATCH_TEMPLATE
            + "\n"
            + f"echo '{result_dir}'\n"
            + "srun "
            + " ".join(current_run_args)
            + "\n"
            + "srun "
            + " ".join(validation_run_args)
        )
        if DEBUG:
            print(file_content)
        with open(SBATCH_FILENAME, "w") as file:
            file.write(file_content)
        if not DEBUG and not GET_PLOTS and not TOP_PSNRS:
            os.system(f"sbatch {SBATCH_FILENAME}")

    return result_dirs


def classic_splat_exps(cfg: Config):
    result_dir = f"results_2d_classic_{cfg.width}x{cfg.height}"
    result_dir += f"{'_pruning' if cfg.use_strategy else ''}"
    if cfg.use_ssim_loss:
        result_dir += "_ssim_loss"
    if cfg.debug_training:
        result_dir += "_debug"
    iterations = ITERATIONS
    checkpoint = 2999
    checkpoint_path = f"results_2d_classic_{cfg.width}x{cfg.height}"
    checkpoint_path += f"_original/ckpts/ckpt_{checkpoint}.pt"

    classic_run_args = [
        "python3 simple_trainer_2d.py",
        f"--img-path {IMG_PATH}",
        f"--iterations {iterations}",
        f"--width {cfg.width}",
        f"--height {cfg.height}",
        f"--results-dir {result_dir}",
        f"--use-classic-mse_loss",
        "--use-strategy" if cfg.use_strategy else "",
        "--use-ssim-loss" if cfg.use_ssim_loss else "",
        "--debug-training" if cfg.debug_training else "",
        f"--ckpt-path {checkpoint_path}",
    ]
    file_content = (
        SBATCH_TEMPLATE + "\n" + f"echo '{result_dir}'\n" + " ".join(classic_run_args)
    )
    if DEBUG:
        print(file_content)
    with open(SBATCH_FILENAME, "w") as file:
        file.write(file_content)
    if not DEBUG and not GET_PLOTS and not TOP_PSNRS:
        os.system(f"sbatch {SBATCH_FILENAME}")

    return [result_dir]


def new_noise_levels_exps(cfg: Config, default_run_args):
    noise_levels = [0.01]
    checkpoints = [2999, 6999, 29999]
    grad_clipping_values = [1.0, 20.0, 50.0]
    result_dirs = []
    min_step = 10
    max_step = 50
    for noise_level in noise_levels:
        for checkpoint in checkpoints:
            for grad_clipping_value in grad_clipping_values:
                current_run_args = default_run_args.copy()
                checkpoint_path = f"results_2d_classic_{cfg.width}x{cfg.height}"
                checkpoint_path += f"_original/ckpts/ckpt_{checkpoint}.pt"
                current_run_args.append(f"--ckpt-path {checkpoint_path}")
                result_dir = f"results_2d_low_res_noise_level_{str(noise_level).replace('.', '_')}_{checkpoint}_min{min_step}_max{max_step}"
                if cfg.use_sdi_loss:
                    result_dir += "_sdi_loss"
                if cfg.base_render_as_cond:
                    current_run_args.append("--base-render-as-cond")
                    result_dir += "_base_render_as_cond"
                if cfg.use_downscaled_mse_loss:
                    current_run_args.append(f"--use-downscaled-mse-loss")
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
                result_dir += f"_grad_clip_{str(grad_clipping_value).replace('.', '_')}"
                current_run_args.append(f"--grad-clipping {grad_clipping_value}")
                current_run_args.append(f"--lowres-noise-level {noise_level}")
                current_run_args.append(f"--results-dir {result_dir}")
                current_run_args.append(f"--min-noise-step {min_step}")
                current_run_args.append(f"--max-noise-step {max_step}")
                current_run_args.append(f"--width {cfg.width}")
                current_run_args.append(f"--height {cfg.height}")
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


def noise_levels_exps(cfg: Config, default_run_args):
    noise_levels = [0.25, 0.5, 0.75]
    coefs_for_sds = [0.001, 0.01, 0.1, 1.0]
    # just noise levels
    result_dirs = []
    for noise_level in noise_levels:
        for coef in coefs_for_sds:
            current_run_args = default_run_args.copy()
            current_run_args.append(f"--ckpt-path {CHECKPOINT_PATH}")
            result_dir = f"results_2d_low_res_noise_level_{str(noise_level).replace('.', '_')}_{CHECKPOINT}"
            if cfg.use_sdi_loss:
                result_dir += "_sdi_loss"
            if cfg.use_downscaled_mse_loss:
                current_run_args.append(f"--use-downscaled-mse-loss")
                result_dir += "_downscaled_mse_loss"
            if cfg.use_fused_loss:
                result_dir += "_fused_loss"
            result_dir += f"_coef_{str(coef).replace('.', '_')}"

            if cfg.base_render_as_cond:
                current_run_args.append("--base-render-as-cond")
                result_dir += "_base_render_as_cond"
            current_run_args.append(f"--lmbd {coef}")
            current_run_args.append(f"--lowres-noise-level {noise_level}")
            current_run_args.append(f"--results-dir {result_dir}")
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


def prompts_and_guidance_exps(cfg: Config, default_run_args):
    result_dirs = []
    noise_levels = [0.25, 0.5, 0.75]
    coefs_for_sds = [0.001, 0.01, 0.1, 1.0]
    # noise levels and condition and prompts
    easy_prompt = "bicycle"
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
    for noise_level in noise_levels:
        for prompt in [easy_prompt, hard_prompt]:
            for guidance_scale in [10.0, 25.0, 50.0, 100.0]:
                for coef in coefs_for_sds:
                    current_run_args = default_run_args.copy()
                    current_run_args.append(f"--ckpt-path {CHECKPOINT_PATH}")
                    is_easy_prompt = len(prompt.split(" ")) == 1
                    result_dir = f"results_2d_low_res_noise_level_{str(noise_level).replace('.', '_')}_{CHECKPOINT}"
                    result_dir += f'_{"easy" if is_easy_prompt else "hard"}'
                    result_dir += (
                        f"_prompt_guidance_{str(guidance_scale).replace('.', '_')}"
                    )
                    if cfg.use_sdi_loss:
                        result_dir += "_sdi_loss"
                    if cfg.use_downscaled_mse_loss:
                        current_run_args.append(f"--use-downscaled-mse-loss")
                        result_dir += "_downscaled_mse_loss"
                    if cfg.use_fused_loss:
                        result_dir += "_fused_loss"
                    result_dir += f"_coef_{str(coef).replace('.', '_')}"
                    current_run_args.append(f"--lmbd {coef}")
                    current_run_args.append(f"--lowres-noise-level {noise_level}")
                    current_run_args.append(f'--prompt "{prompt}"')
                    current_run_args.append(f"--guidance-scale {guidance_scale}")
                    if cfg.base_render_as_cond:
                        current_run_args.append(" --base-render-as-cond")
                        result_dir += "_base_render_as_cond"
                    current_run_args.append(f"--results-dir {result_dir}")
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


def simple_experiments(cfg: Config, default_run_args):
    result_dirs = []
    noise_levels = [0.25]
    guidance_scales = [10, 25]
    easy_prompt = "bicycle"
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
    for noise_level in noise_levels:
        for prompt in [easy_prompt, hard_prompt]:
            for guidance_scale in guidance_scales:
                current_run_args = default_run_args.copy()
                current_run_args.append(f"--ckpt-path {CHECKPOINT_PATH}")
                is_easy_prompt = len(prompt.split(" ")) == 1
                result_dir = f"results_2d_low_res_noise_level_{str(noise_level).replace('.', '_')}_{CHECKPOINT}_iter_{ITERATIONS}"
                result_dir += f'_{"easy" if is_easy_prompt else "hard"}'
                result_dir += (
                    f"_prompt_guidance_{str(guidance_scale).replace('.', '_')}"
                )
                if cfg.use_lr_scheduler:
                    current_run_args.append("--use-lr-scheduler")
                    result_dir += "_lr_scheduler"
                if cfg.collapsing_noise_scheduler:
                    current_run_args.append("--collapsing-noise-scheduler")
                    result_dir += "_noise_scheduler"

                if cfg.use_sdi_loss:
                    result_dir += "_sdi_loss"
                if cfg.use_downscaled_mse_loss:
                    current_run_args.append(f"--use-downscaled-mse-loss")
                    result_dir += "_downscaled_mse_loss"
                if cfg.use_fused_loss:
                    result_dir += "_fused_loss"
                current_run_args.append(f"--lowres-noise-level {noise_level}")
                current_run_args.append(f'--prompt "{prompt}"')
                current_run_args.append(f"--guidance-scale {guidance_scale}")
                if cfg.base_render_as_cond:
                    current_run_args.append(" --base-render-as-cond")
                    result_dir += "_base_render_as_cond"
                current_run_args.append(f"--results-dir {result_dir}")
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
    cfg.base_render_as_cond = True
    cfg.use_sds_loss = True
    cfg.width = 64
    cfg.height = 64
    # cfg.use_fused_loss = True
    # cfg.use_downscaled_mse_loss = True
    cfg.use_strategy = False
    cfg.use_ssim_loss = True
    # cfg.use_altering_loss = True
    # cfg.collapsing_noise_scheduler = True
    # cfg.use_lr_scheduler = True
    # cfg.use_sdi_loss = True

    default_run_args = [
        "python3 simple_trainer_2d.py",
        f"--img-path {IMG_PATH}",
        f"--iterations {ITERATIONS}",
        f"--num-points {cfg.num_points}",
        # f"--max-noise-step {MAX_STEP}",
        # f"--min-noise-step {MIN_STEP}",
        f"--batch-size {BATCH_SIZE}",
        "--use-sds-loss" if cfg.use_sds_loss else "--use-sdi-loss",
        # "--use-fused-loss",
    ]
    result_dirs = []
    # result_dirs += classic_splat_exps(cfg)
    result_dirs += new_noise_levels_exps(cfg, default_run_args)
    # result_dirs += classic_splats_with_validation(cfg)
    # result_dirs += different_checkpoints_exp(cfg, default_run_args)
    # result_dirs += classic_splat_exps(cfg)
    # result_dirs += simple_experiments(cfg, default_run_args)
    # result_dirs += noise_levels_exps(cfg, default_run_args)
    # result_dirs += prompts_and_guidance_exps(cfg, default_run_args)

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
