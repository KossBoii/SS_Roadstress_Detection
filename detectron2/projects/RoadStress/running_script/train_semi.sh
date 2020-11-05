#!/bin/bash
#SBATCH --job-name=road_stress
#SBATCH --nodes=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH -o ./slurm_log/output_%j.txt
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=lntruong@cpp.edu

eval "$(conda shell.bash hook)"
conda activate py3
dir=`pwd`
echo "$dir/dataset/roadstress_new"

if [ ! -d "$dir/dataset" ]; then
	mkdir dataset
fi
if [ ! -d "$dir/dataset/roadstress_new" ]; then
	cd dataset
	svn export https://github.com/KossBoii/RoadDamageDetection.git/trunk/roadstress_new
	cd ..
else
	echo "Dataset roadstress_new exists"
fi

if [ ! -d "$dir/dataset/roadstress_old" ]; then
	cd dataset
	svn export https://github.com/KossBoii/RoadDamageDetection.git/trunk/roadstress_old
	cd ..
else
	echo "Dataset roadstress_old exists"
fi

echo "=========================================="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_GPULIST: $SLURM_JOB_GPUS"
echo "TRAINING DATASET: $1"
echo "BACKBONE MODEL: $2"
echo "=========================================="

<<MULTILINE-COMMENTS
    Backbone model:
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

        - "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"--> new only
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"--> both (not yet)
            
        - "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"--> both
        - "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml" 
MULTILINE-COMMENTS

srun python3 train_semi.py --num-gpus=2 --training-dataset=$1 --backbone=$2
