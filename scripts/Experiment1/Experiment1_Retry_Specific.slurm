#!/bin/bash
#SBATCH --job-name=retry_specific
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=69:00:00
#SBATCH --partition=gpu_p100
#SBATCH --array=0-8

# --- LOG PATHS (Absolute) ---
#SBATCH --output=/data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/slurm/retry_specific_%A_%a.out
#SBATCH --error=/data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/slurm/retry_specific_%A_%a.err

# =============================================================================
# RETRY SPECIFIC FAILED METHOD-DATASET-TASK COMBINATIONS
# =============================================================================
#
# This script reruns specific combinations that failed or need to be rerun.
# It uses direct task specification mode (not array-based task list lookup).
#
# Combinations to rerun:
#   - 0013.home_credit + amformer (PD, HPO)
#   - 0005.base_modelisation + catboost (LGD, HPO & NO_HPO)
#   - 0007.hackerearth + tabicl (PD, HPO & NO_HPO)
#   - 0013.home_credit + tabicl (PD, HPO & NO_HPO)
#   - 0007.hackerearth + tangos (PD, HPO)
#   - 0013.home_credit + tangos (PD, HPO)
#
# =============================================================================

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
set -e

# Stagger starts slightly to reduce I/O contention
sleep $((RANDOM % 30 + 1))

export PYTHONUNBUFFERED=1

echo "=========================================="
echo "RETRY SPECIFIC COMBINATIONS"
echo "Job ID:   $SLURM_JOB_ID"
echo "Array ID: $SLURM_ARRAY_TASK_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Setup conda from $VSC_DATA
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

# USE CONDA'S C++ LIBRARIES to avoid driver/library mismatches
export LD_LIBRARY_PATH="${VSC_DATA}/miniconda3/envs/TabPFNCredit/lib:${LD_LIBRARY_PATH}"

# Navigate to project root
cd $VSC_DATA/TabPFNCredit

# ---------------------------------------------------------
# DEFINE TASKS TO RERUN
# ---------------------------------------------------------
# Format: "dataset method task hpo_mode"
TASKS=(
    "0013.home_credit amformer pd HPO"
    "0005.base_modelisation catboost lgd HPO"
    "0005.base_modelisation catboost lgd NO_HPO"
    "0007.hackerearth tabicl pd HPO"
    "0013.home_credit tabicl pd HPO"
    "0007.hackerearth tabicl pd NO_HPO"
    "0013.home_credit tabicl pd NO_HPO"
    "0007.hackerearth tangos pd HPO"
    "0013.home_credit tangos pd HPO"
)

# Validate array ID
if [ $SLURM_ARRAY_TASK_ID -ge ${#TASKS[@]} ]; then
    echo "ERROR: Array ID $SLURM_ARRAY_TASK_ID out of range [0-$((${#TASKS[@]}-1))]"
    exit 1
fi

# Extract task info based on Array ID
TASK_INFO=(${TASKS[$SLURM_ARRAY_TASK_ID]})
DATASET=${TASK_INFO[0]}
METHOD=${TASK_INFO[1]}
TASK_TYPE=${TASK_INFO[2]}
HPO_MODE=${TASK_INFO[3]}

echo ""
echo "[CONFIG] Dataset:   ${DATASET}"
echo "[CONFIG] Method:    ${METHOD}"
echo "[CONFIG] Task:      ${TASK_TYPE}"
echo "[CONFIG] HPO Mode:  ${HPO_MODE}"
echo ""

# ---------------------------------------------------------
# CATBOOST DRIVER FIX
# ---------------------------------------------------------
# CatBoost on GPU nodes sometimes encounters driver mismatch errors.
# Hide GPU to force CPU execution as a workaround.
if [ "$METHOD" == "catboost" ]; then
    echo "[FIX] CatBoost detected: Hiding GPU to prevent driver mismatch errors."
    echo "[FIX] CatBoost will run on CPU instead."
    export CUDA_VISIBLE_DEVICES=""
fi

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
echo "=========================================="
echo "Starting execution..."
echo "=========================================="

python -u scripts/Experiment1/Experiment1_GPU.py \
    --dataset "${DATASET}" \
    --method "${METHOD}" \
    --task_type "${TASK_TYPE}" \
    --hpo_mode "${HPO_MODE}" \
    --verbose

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ COMPLETED SUCCESSFULLY"
else
    echo "✗ FAILED with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
