#!/usr/bin/env bash

#SBATCH --job-name=exp1_fast
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=1
#SBATCH --cluster=genius
#SBATCH --mem=15G
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_p100
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=andreas.goethals@kuleuven.be
#SBATCH --output=/data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/%x.o%A_%a
#SBATCH --error=/data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/%x.e%A_%a
#SBATCH --array=0-1,3-5,7-11,13,15-19%8

# Fast datasets (16 total):
# PD: 0001.gmsc(0), 0002.taiwan(1), 0004.lending(3), 0005.case(4), 0006.myhom(5),
#     0008.cobranded(7), 0009.german(8), 0010.bank(9), 0011.thomas(10), 
#     0012.loan_default(11), 0014.hmeq(13)
# LGD: all (15-19)

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