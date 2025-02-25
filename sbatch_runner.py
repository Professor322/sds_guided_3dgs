from config import Config
import tyro
import os

CHECKPOINT = 2999
IMG_PATH = "data/360_v2/bicycle/images_8/_DSC8679.JPG"
CHECKPOINT_PATH = (
    f"/home/nskochetkov/sds_guided_3dgs/results_2d/ckpts/ckpt_${CHECKPOINT}.pt"
)
MAX_STEP = 980
MIN_STEP = 20
ITERATIONS = 1000
BATCH_SIZE = 32
DEBUG = False

SBATCH_TEMPLATE = """#!/bin/bash
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


def main(
    cfg: Config,
) -> None:
    # modify parameters for testing
    noise_levels = [0.25, 0.5, 0.75]

    cfg.base_render_as_cond = True

    default_run_args = [
        "CUDA_VISIBLE_DEVICES=0 python3 simple_trainer_2d.py",
        f"--img-path {IMG_PATH}",
        f"--iterations {ITERATIONS}",
        f" --num-points {cfg.num_points}" f"--max-noise-step {MAX_STEP}",
        f"--min-noise-step {MIN_STEP}",
        f"--ckpt-path {CHECKPOINT_PATH}",
        "--use-sds-loss",
    ]

    # adjust result dir name
    for noise_level in noise_levels:
        current_run_args = default_run_args.copy()
        result_dir = f"results_2d_low_res_noise_level_{str(noise_level).replace('.', '_')}_{CHECKPOINT}"
        current_run_args.append(f" --lowres-noise-level {noise_level}")
        if cfg.base_render_as_cond:
            current_run_args.append(" --base-render-as-cond")
            result_dir += "_base_render_as_cond"
        current_run_args.append(f"--results-dir {result_dir}")
        if DEBUG:
            print(SBATCH_TEMPLATE + "\n" + " ".join(current_run_args))

        with open(SBATCH_FILENAME, "w") as file:
            file.write(SBATCH_TEMPLATE + "\n" + " ".join(current_run_args))
            os.system(f"sbatch {SBATCH_FILENAME}")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
