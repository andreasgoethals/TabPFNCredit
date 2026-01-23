#!/bin/bash
# =============================================================================
# FOUNDATION MODEL REPAIR RUN: 6 Failed Tasks (wICE Cluster)
# =============================================================================
# This script re-runs ONLY the 6 specific foundation model tasks that failed
# due to OOM errors during inference.
#
# Fix applied: METHOD_TEST_VAL_LIMITS in method_config.py caps test/val sets
#
# Failed tasks being re-run:
#   [0] 0007.hackerearth / tabpfn / PD / NO_HPO
#   [1] 0007.hackerearth / tabpfn_v2 / PD / NO_HPO
#   [2] 0007.hackerearth / tabpfn_real / PD / NO_HPO
#   [3] 0012.loan_default / mitra / PD / NO_HPO
#   [4] 0013.home_credit / mitra / PD / NO_HPO
#   [5] 0013.home_credit / tabicl / PD / NO_HPO
#
# Target Hardware: H100 (80GB) GPUs on wICE cluster
# =============================================================================

#SBATCH --job-name=exp1_repair
#SBATCH --cluster=wice
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --output=results/experiment1/logs/slurm/repair_%A_%a.out
#SBATCH --error=results/experiment1/logs/slurm/repair_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu_h100

# =============================================================================
# ARRAY CONFIGURATION - REPAIR RUN
# =============================================================================
# Exactly 6 failed tasks to re-run (indices 0-5)
# --exclusive ensures full node access to prevent GPU memory conflicts
# --mem=0 requests all available memory on the node
# =============================================================================
#SBATCH --array=0-5
#SBATCH --exclusive
#SBATCH --mem=0

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
echo "EXPERIMENT 1 - REPAIR RUN (6 Failed Tasks)"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Node:         $SLURMD_NODENAME (EXCLUSIVE)"
echo "Cluster:      $SLURM_CLUSTER_NAME"
echo "Partition:    $SLURM_JOB_PARTITION"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Memory:       ALL (--mem=0)"
echo "=========================================="
echo "Fix: METHOD_TEST_VAL_LIMITS caps test/val sets"
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
