#!/bin/bash
#SBATCH --job-name=exp1_retry_a100
#SBATCH --cluster="wice"
#SBATCH --account="lp_verbekelab" 
#SBATCH --nodes="1" 
#SBATCH --output=results/experiment1/logs/slurm/retry_a100_%A_%a.out
#SBATCH --error=results/experiment1/logs/slurm/retry_a100_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=gpu_a100
#SBATCH --array=0-6%4

# ============================================================
# RETRY SCRIPT FOR A100 GPU (CUDA OOM + Timeouts)
# ============================================================
# This script targets TabR (CUDA OOM) and tangos (timeout)
# 
# Tasks:
#   0: 0007.hackerearth/tabr/pd/NO_HPO
#   1: 0007.hackerearth/tabr/pd/HPO
#   2: 0012.loan_default/tabr/pd/NO_HPO
#   3: 0013.home_credit/tabr/pd/NO_HPO
#   4: 0013.home_credit/tabr/pd/HPO
#   5: 0015.algorithmwatch/tabr/pd/NO_HPO
#   6: 0007.hackerearth/tangos/pd/HPO
# ============================================================

# Load conda
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate TabPFNCredit

echo "========================================================================"
echo "EXP1 - A100 RETRY (40GB VRAM)"
echo "========================================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Cluster:      wICE"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          A100 (40GB VRAM)"
echo "CPUs:         16"
echo "Memory:       64GB"
echo "Time limit:   72h"
echo "Python:       $(which python)"
echo "========================================================================"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Define task list (dataset, method, task_type, hpo_mode)
declare -a TASKS=(
    "0007.hackerearth tabr pd NO_HPO"
    "0007.hackerearth tabr pd HPO"
    "0012.loan_default tabr pd NO_HPO"
    "0013.home_credit tabr pd NO_HPO"
    "0013.home_credit tabr pd HPO"
    "0015.algorithmwatch tabr pd NO_HPO"
    "0007.hackerearth tangos pd HPO"
)

# Get task for this array ID
TASK_INFO=(${TASKS[$SLURM_ARRAY_TASK_ID]})
DATASET=${TASK_INFO[0]}
METHOD=${TASK_INFO[1]}
TASK_TYPE=${TASK_INFO[2]}
HPO_MODE=${TASK_INFO[3]}

echo ""
echo "Running task $SLURM_ARRAY_TASK_ID:"
echo "  Dataset:  $DATASET"
echo "  Method:   $METHOD"
echo "  Task:     $TASK_TYPE"
echo "  HPO Mode: $HPO_MODE"
echo ""

# Add random delay to stagger starts
DELAY=$((RANDOM % 60 + 1))
echo "Sleeping ${DELAY}s to stagger job starts..."
sleep $DELAY

# Run the task
python scripts/Experiment1/Experiment1_GPU.py \
    --dataset="$DATASET" \
    --method="$METHOD" \
    --task_type="$TASK_TYPE" \
    --hpo_mode="$HPO_MODE" \
    --verbose

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Task completed successfully"
else
    echo "Task failed with exit code: $EXIT_CODE"
fi
echo "========================================================================"

exit $EXIT_CODE