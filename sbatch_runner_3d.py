from typing import List
from configs import Config3D, MCMCStrategy
import os
import json
import glob
import argparse
import pandas as pd
import shutil


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
    cfg: Config3D, default_run_args: List[str], opt, prefix=""
):
    if opt.collect_validation_data:
        return []
    current_run_args = default_run_args.copy()
    scene = cfg.data_dir.split(sep="/")[-1]
    cfg.result_dir = f"{prefix}results_3d_classic_data_factor{cfg.data_factor}_{scene}_max_steps{cfg.max_steps}"

    if isinstance(cfg.strategy, MCMCStrategy):
        cfg.result_dir += "_mcmc"
        current_run_args.append("mcmc")
        cfg.result_dir += f"_max_splats{cfg.max_splats}"
        current_run_args.append(f"--max-splats {cfg.max_splats}")
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

    if opt.debug:
        print(file_content)
    with open(SBATCH_FILENAME, "w") as file:
        file.write(file_content)
    if not opt.debug:
        os.system(f"sbatch {SBATCH_FILENAME}")

    return [cfg.result_dir]


def run_gaussian_sr_configuration(
    cfg: Config3D, default_run_args: List[str], opt, prefix=""
):
    if opt.collect_validation_data:
        return []
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
    cfg.result_dir = f"{prefix}results_3d_data_factor{cfg.data_factor}_{scene}_max_steps{cfg.max_steps}"

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
            current_run_args.append(f"--sds-lambda {cfg.sds_lambda}")
            cfg.result_dir += f"{str(cfg.sds_lambda).replace('.', '_')}"
            current_run_args.append(f"--interpolation-type {cfg.interpolation_type}")
            # noise scheduling only makes sense when sds loss is enabled
            if cfg.noise_scheduler_type != "none":
                current_run_args.append(
                    f"--noise-scheduler-type {cfg.noise_scheduler_type}"
                )
                cfg.result_dir += f"_noise_scheduler_{cfg.noise_scheduler_type}"
                if cfg.noise_scheduler_type == "annealing":
                    current_run_args.append(
                        f"--noise-step-annealing {cfg.noise_step_annealing}"
                    )
                    cfg.result_dir += f"{cfg.noise_step_annealing}"

            current_run_args.append(f"--min-noise-step {cfg.min_noise_step}")
            current_run_args.append(f"--max-noise-step {cfg.max_noise_step}")
            cfg.result_dir += (
                f"_min_step{cfg.min_noise_step}_max_step{cfg.max_noise_step}"
            )
            if cfg.densification_skip_sds_grad:
                cfg.result_dir += "_skip_grad"
                current_run_args.append(f"--densification-skip-sds-grad")

        if cfg.scale_factor > 0.0:
            current_run_args.append(f"--scale-factor {cfg.scale_factor}")
            cfg.result_dir += f"_scale_factor{cfg.scale_factor}"

    current_run_args.append(f"--loss-type {cfg.loss_type}")
    cfg.result_dir += f"_{cfg.loss_type}"
    if cfg.loss_type == "l1loss":
        current_run_args.append(f"--ssim-lambda {cfg.ssim_lambda}")
        cfg.result_dir += f"_ssim{str(cfg.ssim_lambda).replace('.', '_')}"

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
    if not opt.debug:
        os.system(f"sbatch {SBATCH_FILENAME}")
    return [cfg.result_dir]


def run_srgs_configuration(cfg: Config3D, default_run_args: List[str], opt, prefix=""):
    if opt.collect_validation_data:
        return []
    current_run_args = default_run_args.copy()
    scene = cfg.data_dir.split(sep="/")[-1]
    cfg.result_dir = f"{prefix}results_3d_data_factor{cfg.data_factor}_{scene}_max_steps{cfg.max_steps}"

    if isinstance(cfg.strategy, MCMCStrategy):
        cfg.result_dir += "_mcmc"
        current_run_args.append("mcmc")
        cfg.result_dir += f"_max_splats{cfg.max_splats}"
        current_run_args.append(f"--max-splats {cfg.max_splats}")
    else:
        cfg.result_dir += "_default"
        current_run_args.append("default")
    if cfg.srgs:
        cfg.result_dir += "_srgs"
        current_run_args.append("--srgs")
        cfg.result_dir += f"{cfg.scale_factor}"
        current_run_args.append(f"--scale-factor {cfg.scale_factor}")

    if cfg.disable_viewer:
        current_run_args.append(f"--disable-viewer")

    if cfg.upscale_suffix != "":
        current_run_args.append(f"--upscale-suffix {cfg.upscale_suffix}")
        cfg.result_dir += f"_{cfg.upscale_suffix}"

    if cfg.ckpt is not None:
        cfg.result_dir += "_with_ckpt"
        current_run_args.append(f"--ckpt {cfg.ckpt}")

    current_run_args.append(f"--data-factor {cfg.data_factor}")
    current_run_args.append(f"--data-dir {cfg.data_dir}")

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
        + "PYTHONPATH=$PYTHONPATH:./StableSR "
        + " ".join(current_run_args)
    )
    if opt.debug:
        print(file_content)
    with open(SBATCH_FILENAME, "w") as file:
        file.write(file_content)
    if not opt.debug:
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
            sds_lambda=0.001,
            max_steps=30_000,
            loss_type="l2loss",
            densification_skip_sds_grad=True,
        ),
        default_run_args,
        opt,
    )


def do_classic_experiments_with_validation(default_run_args: List[str], opt):
    return run_classic_configuration_with_validation_3d(
        Config3D(
            data_factor=4,
            disable_viewer=True,
            data_dir="data/360_v2/bicycle",
            upscale_suffix="stablesr",
        ),
        default_run_args,
        opt,
    )


def do_srgs_experiments(default_run_args: List[str], opt):
    return run_srgs_configuration(
        Config3D(
            data_factor=4,
            disable_viewer=True,
            data_dir="data/360_v2/bicycle",
            upscale_suffix="stablesr",
            scale_factor=4,
            srgs=True,
            ckpt="results_3d_classic_data_factor16_bicycle_max_steps30000_default/ckpts/ckpt_29999_rank0.pt",
        ),
        default_run_args,
        opt,
    )


def do_thesis_experiments(default_run_args: List[str], opt):
    # this function performs whole set of validations on the scene
    # needed for the thesis, namely:
    # 1) low resolution training (baseline)
    # 2) low resolution training MCMC [3_000_000], used only for MCMC [3_000_000] methods
    # 3) original high resolution training (upper bound)
    # 4) bicubic upscale training
    # 5) super resolution upscale training
    # 6) SRGS with MCMC [3_000_000, 6_000_000]
    # 7) SRGS with default
    # 8) gaussianSR default
    # 9) gaussianSR MCMC [3_000_000, 6_000_000] + l1loss
    # 10) gaussianSR MCMC [3_000_000, 6_000_000] + l2loss
    # 11) just subpixel optimization (bicubic downsample)
    # 12) bicubic upscale with MCMC
    # 13) stablesr upscale with MCMC
    result_dirs = []
    scene = "bicycle"
    data_dir = f"data/360_v2/{scene}"
    scene_dir = f"thesis_{scene}"

    # create a dir for validation
    os.makedirs(scene_dir, exist_ok=True)

    # train low resolution splats
    result_dirs.extend(
        run_classic_configuration_with_validation_3d(
            Config3D(
                data_factor=16,
                disable_viewer=True,
                data_dir=data_dir,
            ),
            default_run_args,
            opt,
            scene_dir + "/",
        )
    )
    result_dirs.extend(
        run_classic_configuration_with_validation_3d(
            Config3D(
                data_factor=16,
                disable_viewer=True,
                data_dir=data_dir,
                strategy=MCMCStrategy(),
                max_splats=3_000_000,
            ),
            default_run_args,
            opt,
            scene_dir + "/",
        )
    )
    # train highres splat
    result_dirs.extend(
        run_classic_configuration_with_validation_3d(
            Config3D(
                data_factor=4,
                disable_viewer=True,
                data_dir=data_dir,
            ),
            default_run_args,
            opt,
            scene_dir + "/",
        )
    )
    # train bicubic upscale splat
    result_dirs.extend(
        run_classic_configuration_with_validation_3d(
            Config3D(
                data_factor=4,
                disable_viewer=True,
                data_dir=data_dir,
                upscale_suffix="bicubic",
            ),
            default_run_args,
            opt,
            scene_dir + "/",
        )
    )

    # train bicubic upscale splat with MCMC
    result_dirs.extend(
        run_classic_configuration_with_validation_3d(
            Config3D(
                data_factor=4,
                disable_viewer=True,
                data_dir=data_dir,
                upscale_suffix="bicubic",
                strategy=MCMCStrategy(),
                max_splats=3_000_000,
            ),
            default_run_args,
            opt,
            scene_dir + "/",
        )
    )

    # train super resolution upscale splat
    result_dirs.extend(
        run_classic_configuration_with_validation_3d(
            Config3D(
                data_factor=4,
                disable_viewer=True,
                data_dir=data_dir,
                upscale_suffix="stablesr",
            ),
            default_run_args,
            opt,
            scene_dir + "/",
        )
    )

    # train super resolution upscale splat with MCMC
    result_dirs.extend(
        run_classic_configuration_with_validation_3d(
            Config3D(
                data_factor=4,
                disable_viewer=True,
                data_dir=data_dir,
                upscale_suffix="stablesr",
                strategy=MCMCStrategy(),
                max_splats=3_000_000,
            ),
            default_run_args,
            opt,
            scene_dir + "/",
        )
    )
    # do srgs
    # default
    result_dirs.extend(
        run_srgs_configuration(
            Config3D(
                data_factor=4,
                disable_viewer=True,
                data_dir=data_dir,
                upscale_suffix="stablesr",
                scale_factor=4,
                srgs=True,
            ),
            default_run_args,
            opt,
            scene_dir + "/",
        )
    )

    # MCMC
    for max_splats in [3_000_000, 6_000_000]:
        result_dirs.extend(
            run_srgs_configuration(
                Config3D(
                    data_factor=4,
                    disable_viewer=True,
                    data_dir=data_dir,
                    upscale_suffix="stablesr",
                    scale_factor=4,
                    srgs=True,
                    strategy=MCMCStrategy(),
                    max_splats=max_splats,
                ),
                default_run_args,
                opt,
                scene_dir + "/",
            )
        )

    # do gaussianSR
    # default
    result_dirs.extend(
        run_gaussian_sr_configuration(
            Config3D(
                data_factor=16,
                disable_viewer=True,
                data_dir=data_dir,
                ckpt=f"thesis_{scene}/results_3d_classic_data_factor16_{scene}_max_steps30000_default/ckpts/ckpt_29999_rank0.pt",
                scale_factor=4,
                gaussian_sr=True,
                densification_dropout=0.7,
                noise_scheduler_type="annealing",
                noise_step_annealing=100,
                sds_loss_type="stablesr",
                interpolation_type="bicubic",
                sds_lambda=0.001,
                max_steps=30_000,
                loss_type="l2loss",
                # otherwise we are always running out of memory
                densification_skip_sds_grad=True,
            ),
            default_run_args,
            opt,
            scene_dir + "/",
        )
    )
    for max_splats in [3_000_000, 6_000_000]:
        # MCMC + l1loss
        result_dirs.extend(
            run_gaussian_sr_configuration(
                Config3D(
                    data_factor=16,
                    disable_viewer=True,
                    data_dir=data_dir,
                    ckpt=f"thesis_{scene}/results_3d_classic_data_factor16_{scene}_max_steps30000_mcmc_max_splats3000000/ckpts/ckpt_29999_rank0.pt",
                    scale_factor=4,
                    gaussian_sr=True,
                    noise_scheduler_type="annealing",
                    noise_step_annealing=100,
                    sds_loss_type="stablesr",
                    interpolation_type="bicubic",
                    sds_lambda=0.001,
                    max_steps=30_000,
                    strategy=MCMCStrategy(),
                    max_splats=max_splats,
                ),
                default_run_args,
                opt,
                scene_dir + "/",
            )
        )
        # MCMC
        result_dirs.extend(
            run_gaussian_sr_configuration(
                Config3D(
                    data_factor=16,
                    disable_viewer=True,
                    data_dir=data_dir,
                    ckpt=f"thesis_{scene}/results_3d_classic_data_factor16_{scene}_max_steps30000_mcmc_max_splats3000000/ckpts/ckpt_29999_rank0.pt",
                    scale_factor=4,
                    gaussian_sr=True,
                    noise_scheduler_type="annealing",
                    noise_step_annealing=100,
                    sds_loss_type="stablesr",
                    interpolation_type="bicubic",
                    sds_lambda=0.001,
                    max_steps=30_000,
                    strategy=MCMCStrategy(),
                    max_splats=max_splats,
                    loss_type="l2loss",
                ),
                default_run_args,
                opt,
                scene_dir + "/",
            )
        )

    # just subpixel optimisation
    result_dirs.extend(
        run_gaussian_sr_configuration(
            Config3D(
                data_factor=16,
                disable_viewer=True,
                data_dir=data_dir,
                ckpt=f"thesis_{scene}/results_3d_classic_data_factor16_{scene}_max_steps30000_default/ckpts/ckpt_29999_rank0.pt",
                scale_factor=4,
                gaussian_sr=True,
                noise_scheduler_type="annealing",
                noise_step_annealing=100,
                sds_loss_type="none",
                interpolation_type="bicubic",
                sds_lambda=0.001,
                max_steps=30_000,
                loss_type="l1loss",
            ),
            default_run_args,
            opt,
            scene_dir + "/",
        )
    )
    print(f"Started {len(result_dirs)} experiments")
    return result_dirs


def create_lookup_dict(scene: str) -> dict[str]:
    return {
        # classic methods
        f"results_3d_classic_data_factor4_{scene}_max_steps30000_default": "original",
        f"results_3d_classic_data_factor4_{scene}_max_steps30000_default_bicubic": "bicubic",
        # HRNVS methods
        f"results_3d_classic_data_factor4_{scene}_max_steps30000_default_stablesr": "stablesr",
        f"results_3d_data_factor16_{scene}_max_steps30000_default_dens_drop0_7_gaussian_sr_stablesr0_001_noise_scheduler_annealing100_min_step20_max_step980_skip_grad_scale_factor4_l2loss": "gaussiansr_paper",
        f"results_3d_data_factor16_{scene}_max_steps30000_default_gaussian_sr_scale_factor4_l1loss_ssim0_2": "subpixel",
        f"results_3d_data_factor16_{scene}_max_steps30000_mcmc_max_splats3000000_gaussian_sr_stablesr0_001_noise_scheduler_annealing100_min_step20_max_step980_scale_factor4_l1loss_ssim0_2": "gaussiansr_mcmc_l1loss_ours",
        f"results_3d_data_factor16_{scene}_max_steps30000_mcmc_max_splats3000000_gaussian_sr_stablesr0_001_noise_scheduler_annealing100_min_step20_max_step980_scale_factor4_l2loss": "gaussiansr_mcmc_ours",
        f"results_3d_data_factor4_{scene}_max_steps30000_default_srgs4_stablesr": "srgs_paper",
        f"results_3d_data_factor4_{scene}_max_steps30000_mcmc_max_splats3000000_srgs4_stablesr": "srgs_ours",
        # for testing purposes
        f"results_3d": "test",
    }


def main() -> None:  #
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", help="does a dry run of sbatch command"
    )
    parser.add_argument(
        "--collect-validation-data",
        action="store_true",
        help="scans directories and creates a table with best validation checkpoint",
    )
    parser.add_argument("--dir", type=str, default=".", help="dir to check for psnr")
    parser.add_argument(
        "--sds", action="store_true", help="will perform sds experiments"
    )
    parser.add_argument(
        "--classic",
        action="store_true",
        help="will perform classic experiments",
    )
    parser.add_argument(
        "--srgs", action="store_true", help="will perform srgs experiments"
    )
    parser.add_argument(
        "--thesis",
        action="store_true",
        help="run all expriemnents needed for the thesis for particular scene",
    )
    parser.add_argument("--scene", type=str, default="bicycle", help="scene name")
    opt = parser.parse_args()

    default_run_args = [
        "python3 trainer_3d.py",
    ]
    result_dirs = []
    if opt.sds:
        result_dirs.extend(do_gaussian_sr_experiments(default_run_args, opt))
    if opt.classic:
        result_dirs.extend(
            do_classic_experiments_with_validation(default_run_args, opt)
        )
    if opt.srgs:
        result_dirs.extend(do_srgs_experiments(default_run_args, opt))

    if opt.thesis:
        result_dirs.extend(do_thesis_experiments(default_run_args, opt))

    if opt.collect_validation_data:
        print("Getting metrics...")
        result_dirs = glob.glob(f"{opt.dir}/results_3d*")
        eval_steps = Config3D().eval_steps
        last_training_step = Config3D().save_steps[-1]
        results = {
            "psnr": [],
            "ssim": [],
            "lpips": [],
            "num_GS": [],
            "training_time_min": [],
            "dirs": [],
            "eval_step": [],
        }
        dir_to_method = create_lookup_dict(opt.scene)
        for result_dir in filter(
            lambda dir: os.path.basename(dir) in dir_to_method, result_dirs
        ):
            for eval_step in eval_steps:
                val_filename = f"{result_dir}/stats/val_step{eval_step - 1}.json"
                if not os.path.exists(val_filename):
                    continue
                with open(
                    val_filename,
                    "r",
                ) as val_file:
                    data = val_file.read()
                    val = json.loads(data)
                    results["psnr"].append(val["psnr"])
                    results["ssim"].append(val["ssim"])
                    results["lpips"].append(val["lpips"])
                    results["num_GS"].append(val["num_GS"])
                    results["dirs"].append(result_dir)
                    results["eval_step"].append(eval_step)
                    training_filename = f"{result_dir}/stats/train_step{last_training_step - 1}_rank0.json"
                    if not os.path.exists(training_filename):
                        results["training_time_min"].append(-1)
                    else:
                        with open(training_filename, "r") as train_file:
                            data = train_file.read()
                            train = json.loads(data)
                            results["training_time_min"].append(
                                train["ellipse_time"] / 60
                            )
        df = pd.DataFrame(results)
        pd.set_option("display.max_colwidth", None)
        df.sort_values(by=["dirs", "psnr"], inplace=True, ascending=False)
        df.drop_duplicates(subset=["dirs"], keep="first", inplace=True)
        df.sort_values(by=["psnr"], inplace=True, ascending=False)
        df.set_index(["dirs"], inplace=True)
        method_column = [
            dir_to_method[os.path.basename(dir)] for dir in df.index.values
        ]
        df["method"] = method_column
        print("Selecting best validation renders...")
        overall_val_path = f"{opt.dir}/{opt.scene}_validation_renders"
        os.makedirs(overall_val_path, exist_ok=True)
        for dir in df.index.values:
            eval_step = int(df.loc[dir]["eval_step"])
            method_name = df.loc[dir]["method"]
            val_image_path_src = f"{dir}/renders/val_step{eval_step - 1}_0000.png"
            val_image_path_dst = f"{overall_val_path}/{method_name}.png"
            shutil.copyfile(val_image_path_src, val_image_path_dst)
        print(df.to_string())
        df.to_csv(f"{opt.dir}/{opt.scene}_metrics.csv")


if __name__ == "__main__":
    main()
