#!/usr/bin/env bash

#SBATCH --job-name=exp1_array
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cluster=genius
#SBATCH --mem=40G
#SBATCH --time=70:00:00
#SBATCH --partition=gpu_v100
#SBATCH --gpus-per-node=1
#SBATCH --account=intro_vsc38338
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=andreas.goethals@kuleuven.be
#SBATCH --output=results/experiment1/logs/%x.o%A_%a
#SBATCH --error=results/experiment1/logs/%x.e%A_%a
#SBATCH --array=0-18

# Setup
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit
cd $VSC_DATA/TabPFNCredit
mkdir -p results/experiment1/logs

echo "=== Array task $SLURM_ARRAY_TASK_ID started at $(date) ==="
python -u scripts/Experiment1.py --dataset_idx=$SLURM_ARRAY_TASK_ID
echo "=== Array task $SLURM_ARRAY_TASK_ID finished at $(date) ==="