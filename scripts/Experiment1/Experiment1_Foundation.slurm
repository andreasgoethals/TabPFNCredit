#!/bin/bash
# =============================================================================
# FOUNDATION MODEL RE-RUN: High-Memory GPU Jobs (wICE Cluster)
# =============================================================================
# This script runs foundation models (tabpfn, mitra, tabicl, etc.) that require
# high-memory GPUs due to their in-context learning architecture.
#
# Target Hardware: H100 (80GB) or A100 (80GB) GPUs on wICE cluster
# Memory: 64GB+ system RAM to handle large datasets
#
# To determine the correct array size, run:
#   python scripts/Experiment1/Experiment1_Foundation.py --list-tasks
# =============================================================================

#SBATCH --job-name=exp1_foundation
#SBATCH --cluster=wice
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --output=results/experiment1/logs/slurm/foundation_%A_%a.out
#SBATCH --error=results/experiment1/logs/slurm/foundation_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu_h100

# =============================================================================
# ARRAY CONFIGURATION
# =============================================================================
# Task count calculation:
#   PD: 15 datasets × 5 methods (tabpfn, tabpfn_v2, tabpfn_real, tabicl, mitra) × 2 HPO modes = 150
#   LGD: 7 datasets × 1 method (tabpfn_v2 only) × 2 HPO modes = 14
#   Total: 164 tasks
# =============================================================================
#SBATCH --array=0-163%8

# ---------------------------------------------------------
# STAGGERED START TO PREVENT I/O CONGESTION
# ---------------------------------------------------------
sleep $((RANDOM % 30 + 1))

# ---------------------------------------------------------
# ENVIRONMENT SETUP
# ---------------------------------------------------------

# Force unbuffered I/O for real-time logging
export PYTHONUNBUFFERED=1

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
echo "EXPERIMENT 1 - FOUNDATION MODELS (wICE)"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Cluster:      $SLURM_CLUSTER_NAME"
echo "Partition:    $SLURM_JOB_PARTITION"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Memory:       64GB"
echo "=========================================="
echo "Foundation Methods: tabpfn, tabpfn_v2, tabpfn_real, mitra, tabicl"
echo "=========================================="

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------

# Run foundation model orchestrator
python -u scripts/Experiment1/Experiment1_Foundation.py \
    --array_id=$SLURM_ARRAY_TASK_ID \
    --verbose

# ---------------------------------------------------------
# EXIT STATUS
# ---------------------------------------------------------
EXIT_CODE=$?
echo "=========================================="
echo "Task completed with exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
