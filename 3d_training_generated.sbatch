#!/bin/bash
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

echo 'result_3d_classic_data_factor_64_bicycle_max_steps_30000'
python3 trainer_3d.py --data-factor 64 --data-dir data/360_v2/bicycle --disable-viewer --result-dir result_3d_classic_data_factor_64_bicycle_max_steps_30000
