#!/bin/bash
#SBATCH --job-name=exp3_foundation0
#SBATCH --cluster=wice
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --output=results/experiment3/logs/slurm/foundation0_%A_%a.out
#SBATCH --error=results/experiment3/logs/slurm/foundation0_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --partition=gpu_a100
#SBATCH --array=0-39%8


# ---------------------------------------------------------
# EXPERIMENT 3: CLASS IMBALANCE ANALYSIS - FOUNDATION GPU
# BATCH 0: Tasks 0-39
# ---------------------------------------------------------
# Memory Strategy: #SBATCH --mem=100G
# Fragmentation Fix: YES
# Each task runs an imbalance curve for ONE method + ONE dataset
# Loop over minority_proportion happens inside Python
# ---------------------------------------------------------

# Force unbuffered I/O
export PYTHONUNBUFFERED=1

# Memory Fragmentation Fix (Crucial for Foundation Models)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Setup conda from $VSC_DATA
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

# USE CONDA'S C++ LIBRARIES
export LD_LIBRARY_PATH="${VSC_DATA}/miniconda3/envs/TabPFNCredit/lib:${LD_LIBRARY_PATH}"

# Navigate to project
cd $VSC_DATA/TabPFNCredit

echo "=========================================="
echo "EXP 3 - FOUNDATION - BATCH 0"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Batch:        0"
echo "Task offset:  0"
echo "Node:         $SLURMD_NODENAME"
echo "Memory:       100G (Soft Isolation)"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Calculate global task ID from batch offset
GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + 0))

# Run GPU orchestrator with global task ID
python -u scripts/Experiment3/Experiment3_GPU_Foundation.py --array_id=$GLOBAL_TASK_ID --verbose

# Exit status
EXIT_CODE=$?
echo "=========================================="
echo "Task completed with exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
