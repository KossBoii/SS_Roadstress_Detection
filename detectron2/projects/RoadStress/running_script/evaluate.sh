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
echo "=========================================="

declare -a folderNames=()

if [ $# == 0 ]
then
	folderNames+=(`ls ./output/ -I ^06*$`)
else
	for folder in "$@"
	do
		folderNames+=($folder)
	done
fi

for folder in "${folderNames[@]}"
do
	if [[ ! -f "./output/$folder/model_final.pth" ]]
	then
		echo "model_final.pth file does not exist! Skip inferencing for this folder $folder"
	else
		echo "=========================================="
		echo $folder
		if [[ ! -d "./prediction/$folder" ]]
		then
			cd prediction
			mkdir $folder
			cd ..
		else
			echo "$folder existed!"
		fi
		
		srun python3 evaluate.py --config-file "./output/$folder/config.yaml" \
			--dataset "./dataset/" \
		 	--weight "./output/$folder/model_final.pth" \
		 	--output "./prediction/$folder"
		echo "=========================================="	
	fi
done