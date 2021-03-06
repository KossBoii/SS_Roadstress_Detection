#!/bin/bash
#SBATCH --job-name=road_stress
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o ./slurm_log/output_%j.txt
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=lntruong@cpp.edu

eval "$(conda shell.bash hook)"
conda activate py3

echo "=========================================="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NODELIST: $SLURM_JOB_GPUS"
echo "MODEL_FOLDER_ID: $1"
echo "DATASET_PATH: $2"
echo "EXCLUDE_PATH: $3"
echo "=========================================="

srun python3 inference.py --config-file=./output/$1/config.yaml \
	--dataset=$2 \
	--exclude=$3 \
	--weight=./output/$1/model_final.pth \
	--output ./prediction \
	--confidence-threshold 0.40