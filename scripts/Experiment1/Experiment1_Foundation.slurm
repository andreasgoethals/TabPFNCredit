#!/bin/bash
# =============================================================================
# FOUNDATION MODEL SEQUENTIAL REPAIR RUN (wICE Cluster)
# =============================================================================
# This script runs the 3 foundation tasks SEQUENTIALLY in a SINGLE job.
# This guarantees they never interfere with each other.
#
# Resources:
# - 100G Memory: Ensures no other large job shares the node (Soft Isolation)
# - 1 GPU: Sufficient for sequential execution
# - 48 Hours: Plenty of time for 3 tasks
# =============================================================================

#SBATCH --job-name=exp1_seq_repair
#SBATCH --cluster=wice
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --output=results/experiment1/logs/slurm/seq_repair_%j.out
#SBATCH --error=results/experiment1/logs/slurm/seq_repair_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --partition=gpu_h100

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
echo "EXPERIMENT 1 - SEQUENTIAL REPAIR RUN"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Memory:       100G (Soft Isolation)"
echo "Mode:         Sequential Execution"
echo "=========================================="

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------

# Run sequential orchestrator
python -u scripts/Experiment1/Experiment1_Foundation.py

# ---------------------------------------------------------
# EXIT STATUS
# ---------------------------------------------------------
EXIT_CODE=$?
echo "=========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE