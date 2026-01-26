#!/bin/bash
# =============================================================================
# FOUNDATION MODEL REPAIR RUN: 3 Specific Tasks (wICE Cluster)
# =============================================================================
# This script re-runs ONLY the 3 specific foundation model tasks.
#
# Tasks being re-run:
#   [0] 0012.loan_default / mitra / PD / NO_HPO  (Standard)
#   [1] 0013.home_credit  / mitra / PD / NO_HPO  (Standard)
#   [2] 0013.home_credit  / tabicl/ PD / NO_HPO  (Row Limit: 50,000)
#
# Target Hardware: H100 (80GB) GPUs on wICE cluster
# =============================================================================

#SBATCH --job-name=exp1_repair_3
#SBATCH --cluster=wice
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --output=results/experiment1/logs/slurm/repair3_%A_%a.out
#SBATCH --error=results/experiment1/logs/slurm/repair3_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=85G
#SBATCH --partition=gpu_h100
#SBATCH --exclusive

# =============================================================================
# ARRAY CONFIGURATION - REPAIR RUN
# =============================================================================
# Exactly 3 tasks to re-run (indices 0-2)
# =============================================================================
#SBATCH --array=0-2

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
echo "EXPERIMENT 1 - REPAIR RUN (3 Tasks)"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Cluster:      $SLURM_CLUSTER_NAME"
echo "Partition:    $SLURM_JOB_PARTITION"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Memory:       85G"
echo "=========================================="
echo "Special Config: Row Limit 50k for Task [2]"
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