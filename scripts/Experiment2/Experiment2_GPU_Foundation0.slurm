#!/bin/bash
#SBATCH --job-name=exp2_foundation0
#SBATCH --cluster=wice
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --output=results/experiment2/logs/slurm/foundation0_%A_%a.out
#SBATCH --error=results/experiment2/logs/slurm/foundation0_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --partition=gpu_h100
#SBATCH --array=0-56%8
#SBATCH --exclusive

# ---------------------------------------------------------
# EXPERIMENT 2: LEARNING CURVE ANALYSIS - FOUNDATION GPU
# BATCH 0: Tasks 0-56
# ---------------------------------------------------------
# Cluster: wice | Partition: gpu_h100 | Exclusive Mode
# Each task runs a learning curve for ONE method + ONE dataset
# ---------------------------------------------------------

# Force unbuffered I/O
export PYTHONUNBUFFERED=1

# CRITICAL MEMORY FIX FOR FOUNDATION MODELS (Mitra/TabICL)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Setup conda from $VSC_DATA
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

# USE CONDA'S C++ LIBRARIES
export LD_LIBRARY_PATH="${VSC_DATA}/miniconda3/envs/TabPFNCredit/lib:${LD_LIBRARY_PATH}"

# Navigate to project
cd $VSC_DATA/TabPFNCredit

echo "=========================================="
echo "EXPERIMENT 2 - LEARNING CURVE - FOUNDATION GPU - BATCH 0"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Batch:        0"
echo "Task offset:  0"
echo "Cluster:      wice"
echo "Mode:         EXCLUSIVE (Full Node Access)"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Calculate global task ID from batch offset
GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + 0))

# Run GPU orchestrator with global task ID
python -u scripts/Experiment2/Experiment2_GPU_Foundation.py --array_id=$GLOBAL_TASK_ID --verbose

# Exit status
EXIT_CODE=$?
echo "=========================================="
echo "Task completed with exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
