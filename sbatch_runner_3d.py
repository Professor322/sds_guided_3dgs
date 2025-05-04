from typing import List
from configs import Config3D, MCMCStrategy
import os
import json
import glob
import argparse


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


def run_classic_configuration_with_validation_3d(
    cfg: Config3D, default_run_args: List[str], opt
):
    current_run_args = default_run_args.copy()
    scene = cfg.data_dir.split(sep="/")[-1]
    cfg.result_dir = f"results_3d_classic_data_factor{cfg.data_factor}_{scene}_max_steps{cfg.max_steps}"

    if isinstance(cfg.strategy, MCMCStrategy):
        cfg.result_dir += "_mcmc"
        current_run_args.append("mcmc")
    else:
        cfg.result_dir += "_default"
        current_run_args.append("default")

    if cfg.disable_viewer:
        current_run_args.append(f"--disable-viewer")

    if cfg.upscale_suffix != "":
        current_run_args.append(f"--upscale-suffix {cfg.upscale_suffix}")
        cfg.result_dir += f"_{cfg.upscale_suffix}"

    current_run_args.append(f"--data-factor {cfg.data_factor}")
    current_run_args.append(f"--data-dir {cfg.data_dir}")
    if cfg.densification_dropout > 0.0:
        current_run_args.append(f"--densification-dropout {cfg.densification_dropout}")
        cfg.result_dir += (
            f"_dens_dropout_{str(cfg.densification_dropout).replace('.', '_')}"
        )
    # disable debugging
    cfg.encoder_checkpoint_path = ""
    cfg.encoder_config_path = ""
    current_run_args.append(
        f'--encoder-checkpoint-path "{cfg.encoder_checkpoint_path}"'
    )
    current_run_args.append(f'--encoder-config-path "{cfg.encoder_config_path}"')
    current_run_args.append(f"--result-dir {cfg.result_dir}")

    file_content = (
        SBATCH_TEMPLATE
        + "\n"
        + f"echo '{cfg.result_dir}'\n"
        + "PYTHONPATH=$PYTHONPATH:./StableSR srun "
        + " ".join(current_run_args)
    )
    if cfg.upscale_suffix != "":
        # means we need to do a validation of resulting checkpoint on the real "data factor" images
        for checkpoint_num in cfg.save_steps:
            checkpoint_path = (
                f"{cfg.result_dir}/ckpts/ckpt_{checkpoint_num - 1}_rank0.pt"
            )
            current_run_args = default_run_args.copy()
            current_run_args.append(f"--data-factor {cfg.data_factor}")
            current_run_args.append(f"--result-dir {cfg.result_dir}")
            current_run_args.append(f"--ckpt {checkpoint_path}")
            current_run_args.append(
                f"--densification-dropout {cfg.densification_dropout}"
            )
            file_content += (
                "\n"
                + "PYTHONPATH=$PYTHONPATH:./StableSR srun "
                + " ".join(current_run_args)
            )

    if opt.debug:
        print(file_content)
    with open(SBATCH_FILENAME, "w") as file:
        file.write(file_content)
    if not opt.debug and not opt.top_psnr:
        os.system(f"sbatch {SBATCH_FILENAME}")

    return [cfg.result_dir]


def run_gaussian_sr_configuration(cfg: Config3D, default_run_args: List[str], opt):
    current_run_args = default_run_args.copy()
    current_run_args.insert(0, "PYTHONPATH=$PYTHONPATH:./StableSR")
    if isinstance(cfg.strategy, MCMCStrategy):
        current_run_args.append("mcmc")
    else:
        current_run_args.append("default")
    if cfg.disable_viewer:
        current_run_args.append(f"--disable-viewer")
    # disable debugging
    cfg.encoder_checkpoint_path = ""
    cfg.encoder_config_path = ""
    current_run_args.append(
        f'--encoder-checkpoint-path "{cfg.encoder_checkpoint_path}"'
    )
    current_run_args.append(f'--encoder-config-path "{cfg.encoder_config_path}"')
    current_run_args.append(f"--data-dir {cfg.data_dir}")
    scene = cfg.data_dir.split(sep="/")[-1]
    cfg.result_dir = (
        f"results_3d_data_factor{cfg.data_factor}_{scene}_max_steps{cfg.max_steps}"
    )

    if isinstance(cfg.strategy, MCMCStrategy):
        cfg.result_dir += "_mcmc"
        cfg.result_dir += f"_max_splats{cfg.max_splats}"
        current_run_args.append(f"--max-splats {cfg.max_splats}")
    else:
        cfg.result_dir += "_default"
        if cfg.densification_dropout > 0.0:
            current_run_args.append(
                f"--densification-dropout {cfg.densification_dropout}"
            )
            cfg.result_dir += (
                f"_dens_drop{str(cfg.densification_dropout).replace('.', '_')}"
            )

    current_run_args.append(f"--max-steps {cfg.max_steps}")
    if cfg.gaussian_sr:
        current_run_args.append("--gaussian-sr")
        current_run_args.append(f"--ckpt {cfg.ckpt}")
        cfg.result_dir += "_gaussian_sr"
        if cfg.sds_loss_type != "none":
            current_run_args.append(f"--sds-loss-type {cfg.sds_loss_type}")
            cfg.result_dir += f"_{cfg.sds_loss_type}"
            current_run_args.append(f"--interpolation-type {cfg.interpolation_type}")
            # noise scheduling only makes sense when sds loss is enabled
            if cfg.noise_scheduler_type != "none":
                current_run_args.append(
                    f"--noise-scheduler-type {cfg.noise_scheduler_type}"
                )
                cfg.result_dir += f"_noise_scheduler_{cfg.noise_scheduler_type}"

            current_run_args.append(f"--min-noise-step {cfg.min_noise_step}")
            current_run_args.append(f"--max-noise-step {cfg.max_noise_step}")
            cfg.result_dir += (
                f"_min_step{cfg.min_noise_step}_max_step{cfg.max_noise_step}"
            )

        if cfg.scale_factor > 0.0:
            current_run_args.append(f"--scale-factor {cfg.scale_factor}")
            cfg.result_dir += f"_scale_factor{cfg.scale_factor}"

    current_run_args.append(f"--loss-type {cfg.loss_type}")
    cfg.result_dir += f"_{cfg.loss_type}"
    current_run_args.append(f"--data-factor {cfg.data_factor}")
    current_run_args.append(f"--result-dir {cfg.result_dir}")
    file_content = (
        SBATCH_TEMPLATE
        + "\n"
        + f"echo '{cfg.result_dir}'\n"
        + " ".join(current_run_args)
    )
    if opt.debug:
        print(file_content)
    with open(SBATCH_FILENAME, "w") as file:
        file.write(file_content)
    if not opt.debug and not opt.top_psnr:
        os.system(f"sbatch {SBATCH_FILENAME}")
    return [cfg.result_dir]


def do_gaussian_sr_experiments(default_run_args: List[str], opt):
    return run_gaussian_sr_configuration(
        Config3D(
            data_factor=16,
            disable_viewer=True,
            data_dir="data/360_v2/bicycle",
            ckpt="results_3d_classic_data_factor16_bicycle_max_steps30000_default/ckpts/ckpt_29999_rank0.pt",
            scale_factor=4,
            gaussian_sr=True,
            densification_dropout=0.7,
            noise_scheduler_type="annealing",
            noise_step_annealing=100,
            sds_loss_type="stablesr",
            interpolation_type="bicubic",
        ),
        default_run_args,
        opt,
    )


def do_classic_experiments_with_validation(default_run_args: List[str], opt):
    return run_classic_configuration_with_validation_3d(
        Config3D(data_factor=16, disable_viewer=True, data_dir="data/360_v2/bicycle"),
        default_run_args,
        opt,
    )


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
        "python3 trainer_3d.py",
    ]
    result_dirs = []
    if opt.sds_experiments:
        result_dirs.extend(do_gaussian_sr_experiments(default_run_args, opt))
    if opt.classic_experiments:
        result_dirs.extend(
            do_classic_experiments_with_validation(default_run_args, opt)
        )

    if opt.top_psnr:
        print("Getting psnrs...")
        psnrs_to_dirs = []
        result_dirs = glob.glob(f"{opt.dir}/results_3d*")
        eval_steps = Config3D().eval_steps
        for result_dir in result_dirs:
            for checkpoint in eval_steps:
                filename = f"{result_dir}/stats/val_step{checkpoint - 1}.json"
                if not os.path.exists(filename):
                    continue
                with open(
                    filename,
                    "r",
                ) as file:
                    data = file.read()
                    results = json.loads(data)
                    psnrs_to_dirs.append((results["psnr"], checkpoint, result_dir))

        psnrs_to_dirs = sorted(psnrs_to_dirs, reverse=True)
        for psnr_to_dir in psnrs_to_dirs:
            print(psnr_to_dir)


if __name__ == "__main__":
    main()
