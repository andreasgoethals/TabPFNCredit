#!/usr/bin/env bash

#SBATCH --job-name=exp1_slow
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=1
#SBATCH --cluster=genius
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu_p100
#SBATCH --gpus-per-node=1
#SBATCH --account=intro_vsc38338
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=andreas.goethals@kuleuven.be
#SBATCH --output=/data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/%x.o%A_%a
#SBATCH --error=/data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/%x.e%A_%a
#SBATCH --array=2,6,12,14

# Large/slow datasets (4 total):
# 2:  0003.vehicle_loan   (~230K rows)
# 6:  0007.hackerearth    (~250K rows)
# 12: 0013.home_credit    (~300K rows)
# 14: 0015.algorithmwatch (~159K rows, 2987 features -> PCA)

# Force unbuffered I/O
export PYTHONUNBUFFERED=1

# Setup environment
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

# Navigate to project
cd $VSC_DATA/TabPFNCredit

# Log start
echo "=== Array task $SLURM_ARRAY_TASK_ID started at $(date) ==="
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Run experiment
python -u scripts/Experiment1/Experiment1.py --dataset_idx=$SLURM_ARRAY_TASK_ID

# Log end
echo "=== Array task $SLURM_ARRAY_TASK_ID finished at $(date) ==="