#!/bin/bash
# =============================================================================
# EXPERIMENT 1 - FINAL RERUN: 0013.home_credit / mitra / NO_HPO
# =============================================================================
# Single task on a dedicated node (no row limit).
# =============================================================================

#SBATCH --job-name=exp1_final_mitra
#SBATCH --cluster=wice
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --output=results/experiment1/logs/slurm/final0_%j.out
#SBATCH --error=results/experiment1/logs/slurm/final0_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --partition=gpu_a100

# ---------------------------------------------------------
# ENVIRONMENT SETUP
# ---------------------------------------------------------

# Force unbuffered I/O for real-time logging
export PYTHONUNBUFFERED=1

# Memory Fragmentation Fix (Crucial for Mitra)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Setup conda from $VSC_DATA
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

# Use conda's C++ libraries
export LD_LIBRARY_PATH="${VSC_DATA}/miniconda3/envs/TabPFNCredit/lib:${LD_LIBRARY_PATH}"

# Navigate to project directory
cd $VSC_DATA/TabPFNCredit

# ---------------------------------------------------------
# JOB INFORMATION
# ---------------------------------------------------------
echo "=========================================="
echo "EXPERIMENT 1 - FINAL RERUN 0: mitra"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Memory:       100G (Soft Isolation)"
echo "Task:         0013.home_credit / mitra / NO_HPO"
echo "=========================================="

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------

python -u scripts/Experiment1/Experiment1_Final.py \
    --dataset 0013.home_credit \
    --method mitra \
    --task_type pd \
    --hpo_mode NO_HPO \
    --row_limit 20000

# ---------------------------------------------------------
# EXIT STATUS
# ---------------------------------------------------------
EXIT_CODE=$?
echo "=========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
