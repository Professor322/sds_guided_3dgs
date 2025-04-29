from configs import Config2D
import tyro
import os
import json
import glob
from typing import List

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


def classic_splats_with_validation_2d(cfg: Config2D, default_run_args: List[str]):
    result_dir = "results_2d_classic"
    cfg.iterations = 30_000
    cfg.num_points = 10_000
    cfg.classic_loss_type = "l1loss"
    training_scale = 16
    cfg.validation_image_path = (
        f"data/360_v2/bicycle/images_{training_scale}/_DSC8679.JPG"
    )
    cfg.training_image_path = (
        f"data/360_v2/bicycle/images_{training_scale}/_DSC8679.JPG"
    )
    cfg.use_strategy = True

    current_run_args = default_run_args.copy()
    current_run_args.append(f"--validation-image-path {cfg.validation_image_path}")
    current_run_args.append(f"--training-image-path {cfg.training_image_path}")

    current_run_args.append(f"--num-points {cfg.num_points}")
    result_dir += f"_num_points_{cfg.num_points}"

    current_run_args.append(f"--classic-loss-type {cfg.classic_loss_type}")
    result_dir += f"_{cfg.classic_loss_type}"

    if (cfg.width, cfg.height) != (0, 0):
        result_dir += f"_{cfg.width}x{cfg.height}"
    else:
        result_dir += "_original"

    if cfg.use_strategy:
        result_dir += "_strategy"
        current_run_args.append(f"--use-strategy")
    result_dir += f"_scale_{training_scale}"

    current_run_args.append(f"--results-dir {result_dir}")

    file_content = (
        SBATCH_TEMPLATE + "\n" + f"echo '{result_dir}'\n" + " ".join(current_run_args)
    )
    if DEBUG:
        print(file_content)
    with open(SBATCH_FILENAME, "w") as file:
        file.write(file_content)
    if not DEBUG and not GET_PLOTS and not TOP_PSNRS:
        os.system(f"sbatch {SBATCH_FILENAME}")

    return [result_dir]


def sds_experiments_2d(cfg: Config2D, default_run_args: List[str]):
    checkpopint_num = 999
    cfg.num_points = 10_000
    cfg.use_gaussian_sr = True
    cfg.scale_factor = 2
    cfg.sds_loss_type = "none"
    cfg.classic_loss_type = "l1loss"
    training_scale = 16
    validation_scale = training_scale // cfg.scale_factor
    cfg.validation_image_path = (
        f"data/360_v2/bicycle/images_{training_scale}/_DSC8679.JPG"
    )
    cfg.training_image_path = (
        f"data/360_v2/bicycle/images_{validation_scale}/_DSC8679.JPG"
    )

    current_run_args = default_run_args.copy()

    current_run_args.append(f"--validation-image-path {cfg.validation_image_path}")
    current_run_args.append(f"--training-image-path {cfg.training_image_path}")

    checkpoint_path = f"results_2d_classic"
    checkpoint_path += f"_num_points_{cfg.num_points}_{cfg.classic_loss_type}"
    if (cfg.width, cfg.height) != (0, 0):
        checkpoint_path += f"_{cfg.width}x{cfg.height}"
    else:
        checkpoint_path += "_original"
    checkpoint_path += "_strategy"
    checkpoint_path += f"_scale_{training_scale}"
    checkpoint_path += f"/ckpts/ckpt_{checkpopint_num}.pt"
    current_run_args.append(f"--ckpt-path {checkpoint_path}")
    result_dir = f"results_2d_low_res_noise_level_{str(cfg.lowres_noise_level).replace('.', '_')}"
    result_dir += f"_{checkpopint_num}_min{cfg.min_noise_step}_max{cfg.max_noise_step}"
    if cfg.use_gaussian_sr:
        result_dir += f"_gaussian_sr"
        current_run_args.append(f"--use-gaussian-sr")
        if cfg.sds_loss_type != "none":
            if cfg.sds_lambda > 0.0:
                result_dir += f"_lambda_{str(cfg.sds_lambda).replace('.', '_')}"
                current_run_args.append(f"--sds-lambda {cfg.sds_lambda}")
            result_dir += f"_{cfg.noise_scheduler_type}"
            current_run_args.append(
                f"--noise-scheduler-type {cfg.noise_scheduler_type}"
            )
            result_dir += f"_{cfg.sds_loss_type}"
            current_run_args.append(f"--sds-loss-type {cfg.sds_loss_type}")
        if cfg.noise_step_anealing > 0 and cfg.noise_scheduler_type == "annealing":
            result_dir += f"_anealing_{cfg.noise_step_anealing}"
            current_run_args.append(f"--noise-step-anealing {cfg.noise_step_anealing}")

    if cfg.grad_clipping > 0.0:
        result_dir += f"_grad_clip_{str(cfg.grad_clipping).replace('.', '_')}"
        current_run_args.append(f"--grad-clipping {cfg.grad_clipping}")
    if cfg.use_strategy:
        result_dir += "_strategy"
        current_run_args.append(f"--use-strategy")
        if cfg.densification_dropout > 0:
            result_dir += (
                f"_dens_dropout_{str(cfg.densification_dropout).replace('.', '_')}"
            )
            current_run_args.append(
                f"--densification-dropout {cfg.densification_dropout}"
            )

    result_dir += f"_{cfg.classic_loss_type}"
    current_run_args.append(f"--classic-loss-type {cfg.classic_loss_type}")

    result_dir += f"_num_points_{cfg.num_points}"
    current_run_args.append(f"--num-points {cfg.num_points}")
    current_run_args.append(f"--lowres-noise-level {cfg.lowres_noise_level}")
    current_run_args.append(f"--results-dir {result_dir}")
    current_run_args.append(f"--min-noise-step {cfg.min_noise_step}")
    current_run_args.append(f"--max-noise-step {cfg.max_noise_step}")
    current_run_args.append(f"--scale-factor {cfg.scale_factor}")
    if (cfg.width, cfg.height) != (0, 0):
        current_run_args.append(f"--width {cfg.width}")
        current_run_args.append(f"--height {cfg.height}")

    file_content = (
        SBATCH_TEMPLATE + "\n" + f"echo '{result_dir}'\n" + " ".join(current_run_args)
    )
    if DEBUG:
        print(file_content)
    with open(SBATCH_FILENAME, "w") as file:
        file.write(file_content)
    if not DEBUG and not GET_PLOTS and not TOP_PSNRS:
        os.system(f"sbatch {SBATCH_FILENAME}")
    return [result_dir]


def main(
    cfg: Config2D,
) -> None:

    do_sds_experiments = True
    do_classic_experiments = False

    default_run_args = [
        "PYTHONPATH=$PYTHONPATH:./StableSR python3 -u trainer_2d.py",
    ]
    result_dirs = []
    if do_sds_experiments:
        result_dirs += sds_experiments_2d(cfg, default_run_args)
    if do_classic_experiments:
        result_dirs += classic_splats_with_validation_2d(cfg, default_run_args)

    result_dirs = glob.glob("./results_2d/results_2d_low*")
    if TOP_PSNRS:
        print("Getting psnrs...")
        psnrs_to_dirs = []
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
    cfg = tyro.cli(Config2D)
    main(cfg)
