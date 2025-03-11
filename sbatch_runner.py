from config import Config
import tyro
import os
import json
import glob

CHECKPOINT = 2999
IMG_PATH = "data/360_v2/bicycle/images_8/_DSC8679.JPG"
CHECKPOINT_PATH = (
    f"/home/nskochetkov/sds_guided_3dgs/results_2d/ckpts/ckpt_{CHECKPOINT}.pt"
)
MAX_STEP = 980
MIN_STEP = 20
ITERATIONS = 10_000
# sometimes can hit oom, so we have to reduce it
BATCH_SIZE = 24
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


def classic_splat_exps():

    result_dir = "results_2d_classic_64x64"
    classic_run_args = [
        "python3 simple_trainer_2d.py",
        f"--img-path {IMG_PATH}",
        f"--iterations 30000",
        f"--width 64",
        f"--height 64",
        f"--results-dir {result_dir}",
        f"--use_classic_mse_loss",
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


def noise_levels_exps(cfg: Config, default_run_args):
    noise_levels = [0.25, 0.5, 0.75]
    coefs_for_sds = [0.001, 0.01, 0.1, 1.0]
    # just noise levels
    result_dirs = []
    for noise_level in noise_levels:
        for coef in coefs_for_sds:
            current_run_args = default_run_args.copy()
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
    cfg.use_fused_loss = False
    cfg.use_downscaled_mse_loss = False
    cfg.collapsing_noise_scheduler = True
    cfg.use_lr_scheduler = True
    # cfg.use_sdi_loss = True

    default_run_args = [
        "python3 simple_trainer_2d.py",
        f"--img-path {IMG_PATH}",
        f"--iterations {ITERATIONS}",
        f"--num-points {cfg.num_points}",
        f"--max-noise-step {MAX_STEP}",
        f"--min-noise-step {MIN_STEP}",
        f"--ckpt-path {CHECKPOINT_PATH}",
        f"--batch-size {BATCH_SIZE}",
        "--use-sds-loss" if cfg.use_sds_loss else "--use-sdi-loss",
        # "--use-fused-loss",
    ]
    result_dirs = []
    result_dirs += classic_splat_exps()
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
                data = file.read()
                results = json.loads(data)
                psnrs_to_dirs.append((results["psnr"], result_dir))

        psnrs_to_dirs = sorted(psnrs_to_dirs, reverse=True)
        for psnr_to_dir in psnrs_to_dirs:
            print(psnr_to_dir)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
