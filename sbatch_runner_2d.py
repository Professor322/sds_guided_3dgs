from configs import Config2D
import os
import json
import glob
from typing import List
import argparse

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


def classic_splats_with_validation_2d(cfg: Config2D, default_run_args: List[str], opt):
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
    current_run_args.append(f"--iterations {cfg.iterations}")
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
    if opt.debug:
        print(file_content)
    with open(SBATCH_FILENAME, "w") as file:
        file.write(file_content)
    if not opt.debug and not opt.top_psnr:
        os.system(f"sbatch {SBATCH_FILENAME}")

    return [result_dir]


def sds_experiments_2d(cfg: Config2D, default_run_args: List[str], opt):
    checkpopint_num = 29999
    cfg.use_gaussian_sr = True
    cfg.scale_factor = 4
    cfg.iterations = 5_000
    cfg.refine_stop_iter = 2_000
    cfg.sds_loss_type = "stable_sr_sds"
    cfg.classic_loss_type = "l1loss"
    cfg.noise_scheduler_type = "annealing"
    cfg.noise_step_anealing = 100
    cfg.sds_lambda = 0.001
    cfg.use_strategy = True
    cfg.densification_dropout = 0.7
    training_scale = 16
    validation_scale = training_scale // cfg.scale_factor
    cfg.validation_image_path = (
        f"data/360_v2/bicycle/images_{validation_scale}/_DSC8679.JPG"
    )
    cfg.training_image_path = (
        f"data/360_v2/bicycle/images_{training_scale}/_DSC8679.JPG"
    )
    # disable debugging
    cfg.encoder_checkpoint_path = ""
    cfg.encoder_config_path = ""

    current_run_args = default_run_args.copy()

    current_run_args.append(f"--refine-stop-iter {cfg.refine_stop_iter}")
    current_run_args.append(f"--encoder-checkpoint-path {cfg.encoder_checkpoint_path}")
    current_run_args.append(f"--encoder-configh-path {cfg.encoder_config_path}")
    current_run_args.append(f"--validation-image-path {cfg.validation_image_path}")
    current_run_args.append(f"--training-image-path {cfg.training_image_path}")
    current_run_args.append(f"--iterations {cfg.iterations}")

    checkpoint_path = (
        "results_2d_classic_num_points_10000_l1loss_original_strategy_scale_16"
    )
    current_run_args.append(f"--ckpt-path {checkpoint_path}")
    result_dir = f"results_2d"
    result_dir += f"_ckpt{checkpopint_num}"
    if cfg.use_gaussian_sr:
        result_dir += f"_gaussian_sr"
        current_run_args.append(f"--use-gaussian-sr")
        if cfg.sds_loss_type != "none":
            result_dir += f"_{cfg.noise_scheduler_type}"
            current_run_args.append(
                f"--noise-scheduler-type {cfg.noise_scheduler_type}"
            )
            if cfg.noise_step_anealing > 0 and cfg.noise_scheduler_type == "annealing":
                result_dir += f"{cfg.noise_step_anealing}"
                current_run_args.append(
                    f"--noise-step-anealing {cfg.noise_step_anealing}"
                )
            result_dir += f"_{cfg.sds_loss_type}"
            current_run_args.append(f"--sds-loss-type {cfg.sds_loss_type}")
            if cfg.sds_lambda > 0.0:
                result_dir += f"_lambda{str(cfg.sds_lambda).replace('.', '_')}"
                current_run_args.append(f"--sds-lambda {cfg.sds_lambda}")
            result_dir += f"_min{cfg.min_noise_step}_max{cfg.max_noise_step}"
            current_run_args.append(f"--min-noise-step {cfg.min_noise_step}")
            current_run_args.append(f"--max-noise-step {cfg.max_noise_step}")

    if cfg.grad_clipping > 0.0:
        result_dir += f"_grad_clip_{str(cfg.grad_clipping).replace('.', '_')}"
        current_run_args.append(f"--grad-clipping {cfg.grad_clipping}")
    if cfg.use_strategy:
        result_dir += "_strategy"
        current_run_args.append(f"--use-strategy")
        if cfg.densification_dropout > 0:
            result_dir += f"_dropout{str(cfg.densification_dropout).replace('.', '_')}"
            current_run_args.append(
                f"--densification-dropout {cfg.densification_dropout}"
            )

    result_dir += f"_{cfg.classic_loss_type}"
    current_run_args.append(f"--classic-loss-type {cfg.classic_loss_type}")
    # ssim only used in l1loss
    if cfg.classic_loss_type == "l1loss":
        result_dir += f"_ssim_lambda{str(cfg.ssim_lambda).replace('.', '_')}"
        current_run_args.append(f"--ssim-lambda {cfg.ssim_lambda}")
    result_dir += f"_inter_{cfg.interpolation_type}"
    current_run_args.append(f"--interpolation-type {cfg.interpolation_type}")
    current_run_args.append(f"--lowres-noise-level {cfg.lowres_noise_level}")
    if cfg.use_gaussian_sr:
        current_run_args.append(f"--scale-factor {cfg.scale_factor}")
        result_dir += f"_scale{cfg.scale_factor}"
    current_run_args.append(f"--results-dir {result_dir}")
    if (cfg.width, cfg.height) != (0, 0):
        current_run_args.append(f"--width {cfg.width}")
        current_run_args.append(f"--height {cfg.height}")

    file_content = (
        SBATCH_TEMPLATE + "\n" + f"echo '{result_dir}'\n" + " ".join(current_run_args)
    )
    if opt.debug:
        print(file_content)
    with open(SBATCH_FILENAME, "w") as file:
        file.write(file_content)
    if not opt.debug and not opt.top_psnr:
        os.system(f"sbatch {SBATCH_FILENAME}")
    return [result_dir]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", help="does a dry run of sbatch command"
    )
    parser.add_argument(
        "--top-psnr",
        action="store_true",
        help="scans directory for results_2d* folders and checks psnr in checkpoints",
    )
    parser.add_argument("--dir", type=str, default=".", help="dir to check for psnr")
    parser.add_argument(
        "--sds-experiments", action="store_true", help="will perform sds experiments"
    )
    parser.add_argument(
        "--classic-experiments",
        action="store_true",
        help="will perform classic experiments",
    )
    opt = parser.parse_args()

    default_run_args = [
        "PYTHONPATH=$PYTHONPATH:./StableSR python3 -u trainer_2d.py",
    ]
    result_dirs = []
    if opt.sds_experiments:
        result_dirs += sds_experiments_2d(Config2D(), default_run_args, opt)
    if opt.classic_experiments:
        result_dirs += classic_splats_with_validation_2d(
            Config2D(), default_run_args, opt
        )

    if opt.top_psnr:
        result_dirs = glob.glob(f"{opt.dir}/results_2d_low*")
        print("Getting psnrs...")
        save_steps = Config2D().save_steps
        psnrs_to_dirs = []
        for result_dir in result_dirs:
            for save_step in save_steps:
                filename = f"{result_dir}/stats/step{save_step - 1}.json"
                if not os.path.exists(filename):
                    continue
                with open(
                    filename,
                    "r",
                ) as file:
                    data = file.read()
                    results = json.loads(data)
                    psnrs_to_dirs.append((results["psnr"], save_step, result_dir))

        psnrs_to_dirs = sorted(psnrs_to_dirs, reverse=True)
        for psnr_to_dir in psnrs_to_dirs:
            print(psnr_to_dir)


if __name__ == "__main__":
    main()
