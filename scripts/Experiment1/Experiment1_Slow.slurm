#!/usr/bin/env bash

#SBATCH --job-name=exp1_slow
#SBATCH --cluster=wice             
#SBATCH --partition=gpu_a100       
#SBATCH --nodes=1
#SBATCH --ntasks=1                 
#SBATCH --cpus-per-task=18         
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G                 
#SBATCH --time=60:00:00            
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=andreas.goethals@kuleuven.be
#SBATCH --output=/data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/%x.o%A_%a
#SBATCH --error=/data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/%x.e%A_%a
#SBATCH --array=14

# Large/slow datasets (4 total):
# 2:  0003.vehicle_loan
# 6:  0007.hackerearth
# 12: 0013.home_credit
# 14: 0015.algorithmwatch

# Force unbuffered I/O
export PYTHONUNBUFFERED=1

# Setup environment
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

# Navigate to project
cd $VSC_DATA/TabPFNCredit

# Log start
echo "=== Slow Array task $SLURM_ARRAY_TASK_ID started at $(date) ==="
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Run experiment
python -u scripts/Experiment1/Experiment1.py --dataset_idx=$SLURM_ARRAY_TASK_ID

echo "=== Slow Array task $SLURM_ARRAY_TASK_ID finished at $(date) ==="