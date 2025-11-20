#!/usr/bin/env bash

#SBATCH --job-name=exp1_single
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --cluster=genius
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_p100
#SBATCH --gpus-per-node=1
#SBATCH --account=intro_vsc38338
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=andreas.goethals@kuleuven.be
#SBATCH --output=/data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/%x.o%j
#SBATCH --error=/data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/%x.e%j

# Force unbuffered output
export PYTHONUNBUFFERED=1

# Setup environment
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

# Navigate to project
cd $VSC_DATA/TabPFNCredit
mkdir -p results/experiment1/logs

# Log start
echo "=== Single debug job started at $(date) ==="
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Run single dataset (change index as needed)
# Dataset indices: 0-13 = PD datasets, 14-18 = LGD datasets
python -u scripts/Experiment1.py --dataset_idx=0

echo "=== Single debug job finished at $(date) ==="