#!/bin/bash
#SBATCH --job-name=exp3_standard0
#SBATCH --cluster=genius
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --output=results/experiment3/logs/slurm/standard0_%A_%a.out
#SBATCH --error=results/experiment3/logs/slurm/standard0_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=45G
#SBATCH --partition=gpu_p100
#SBATCH --array=0-31%12


# ---------------------------------------------------------
# EXPERIMENT 3: CLASS IMBALANCE ANALYSIS - STANDARD GPU
# BATCH 0: Tasks 0-31
# ---------------------------------------------------------
# Memory Strategy: #SBATCH --mem=45G
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
echo "EXP 3 - STANDARD - BATCH 0"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Batch:        0"
echo "Task offset:  0"
echo "Node:         $SLURMD_NODENAME"
echo "Memory:       45G"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Calculate global task ID from batch offset
GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + 0))

# Run GPU orchestrator with global task ID
python -u scripts/Experiment3/Experiment3_GPU_Standard.py --array_id=$GLOBAL_TASK_ID --verbose

# Exit status
EXIT_CODE=$?
echo "=========================================="
echo "Task completed with exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
