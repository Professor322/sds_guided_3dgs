from typing import List
from configs import Config3D, DefaultStrategy, MCMCStrategy
import tyro
import os
import json
import glob

DEBUG = False
GET_PLOTS = False
TOP_PSNRS = False

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=train_3d
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1

module purge
module load Python gnu10

source deactivate
source activate gsplat_sds

nvidia-smi
echo "starting 3d training"
"""

SBATCH_FILENAME = "3d_training_generated.sbatch"


def classic_splats_with_validation_3d(cfg: Config3D, default_run_args: List[str]):
    # upscale from 64 by the factor of 4 -> 16_bicubic
    cfg.upscale_suffix = "bicubic"
    cfg.data_factor = 16
    current_run_args = default_run_args.copy()
    scene = cfg.data_dir.split(sep="/")[-1]
    result_dir = f"results_3d_classic_data_factor_{cfg.data_factor}_{scene}_max_steps_{cfg.max_steps}"

    if cfg.upscale_suffix != "":
        current_run_args.append(f"--upscale-suffix {cfg.upscale_suffix}")
        result_dir += f"_{cfg.upscale_suffix}"

    current_run_args.append(f"--data-factor {cfg.data_factor}")
    current_run_args.append(f"--result-dir {result_dir}")

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


def sds_experiments_3d(cfg: Config3D, default_run_args: List[str]):
    cfg.gaussian_sr = True
    cfg.sds_loss_type = "none"
    cfg.noise_scheduler_type = "annealing"
    cfg.scale_factor = 4
    cfg.loss_type = "l2loss"
    cfg.data_factor = 64

    current_run_args = default_run_args.copy()
    scene = cfg.data_dir.split(sep="/")[-1]
    checkpoint_path = f"results_3d_classic_data_factor_{cfg.data_factor}_{scene}_max_steps_{cfg.max_steps}"
    checkpoint_path += "/ckpts/ckpt_29999_rank0.pt"
    result_dir = f"results_3d_classic_data_factor_{cfg.data_factor}_{scene}_max_steps_{cfg.max_steps}"

    if cfg.gaussian_sr:
        current_run_args.append("--gaussian-sr")
        result_dir += "_gaussian_sr"
        if cfg.sds_loss_type != "none":
            current_run_args.append(f"--sds-loss-type {cfg.sds_loss_type}")
            result_dir += f"_{cfg.sds_loss_type}"
            # noise scheduling only makes sense when sds loss is enabled
            if cfg.noise_scheduler_type != "none":
                current_run_args.append(
                    f"--noise-scheduler-type {cfg.noise_scheduler_type}"
                )
                result_dir += f"_noise_scheduler_{cfg.noise_scheduler_type}"
        if cfg.scale_factor > 0.0:
            current_run_args.append(f"--scale-factor {cfg.scale_factor}")
            result_dir += f"_scale_factor_{cfg.scale_factor}"
        if checkpoint_path != "":
            current_run_args.append(f"--ckpt {checkpoint_path}")
        if cfg.densification_dropout > 0.0:
            current_run_args.append(
                f"--densification-dropout {cfg.densification_dropout}"
            )
            result_dir += (
                f"_dens_dropout_{str(cfg.densification_dropout).replace('.', '_')}"
            )
        if cfg.loss_type == "l2loss":
            current_run_args.append(f"--loss-type {cfg.loss_type}")
            result_dir += f"_{cfg.loss_type}"

    current_run_args.append(f"--data-factor {cfg.data_factor}")
    current_run_args.append(f"--result-dir {result_dir}")
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
    cfg: Config3D,
) -> None:

    cfg.data_factor = 64
    cfg.data_dir = "data/360_v2/bicycle"
    cfg.disable_viewer = True

    do_classic_experiments = True
    do_gaussian_sr_experiments = False
    default_run_args = [
        "python3 -u trainer_3d.py",
        "default" if isinstance(cfg.strategy, DefaultStrategy) else "mcmc",
        f"--data-dir {cfg.data_dir}",
        f"--disable-viewer" if cfg.disable_viewer else "",
    ]
    result_dirs = []
    if do_gaussian_sr_experiments:
        result_dirs += sds_experiments_3d(cfg, default_run_args)
    if do_classic_experiments:
        result_dirs += classic_splats_with_validation_3d(cfg, default_run_args)

    result_dirs = glob.glob("/home/nskochetkov/sds_guided_3dgs/results_3d_low*")
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
        result_dirs = glob.glob("/home/nskochetkov/sds_guided_3dgs/results_3d_low*")
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
    cfg = tyro.cli(Config3D)
    main(cfg)
