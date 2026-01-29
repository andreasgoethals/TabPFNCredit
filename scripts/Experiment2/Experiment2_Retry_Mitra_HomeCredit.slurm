#!/bin/bash
#SBATCH --job-name=exp2_retry_mitra_hc
#SBATCH --cluster=wice
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --output=results/experiment2/logs/slurm/retry_mitra_hc_%j.out
#SBATCH --error=results/experiment2/logs/slurm/retry_mitra_hc_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --partition=gpu_h100

# ---------------------------------------------------------
# RETRY: mitra on 0013.home_credit (PD)
# Reruns all row_limits for this single combination
# ---------------------------------------------------------

# Force unbuffered I/O
export PYTHONUNBUFFERED=1

# Memory Fragmentation Fix
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Setup conda from $VSC_DATA
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

# USE CONDA'S C++ LIBRARIES
export LD_LIBRARY_PATH="${VSC_DATA}/miniconda3/envs/TabPFNCredit/lib:${LD_LIBRARY_PATH}"

# Navigate to project
cd $VSC_DATA/TabPFNCredit

echo "=========================================="
echo "EXP 2 - RETRY: mitra / 0013.home_credit"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Memory:       100G (Soft Isolation)"
echo "=========================================="

# Run Foundation orchestrator with direct task specification
python -u scripts/Experiment2/Experiment2_GPU_Foundation.py \
    --dataset 0013.home_credit \
    --method mitra \
    --task_type pd \
    --verbose

# Exit status
EXIT_CODE=$?
echo "=========================================="
echo "Task completed with exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
