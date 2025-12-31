#!/bin/bash
#SBATCH --job-name=exp1_retry_tabicl
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab" 
#SBATCH --nodes="1" 
#SBATCH --output=results/experiment1/logs/slurm/retry_tabicl_%A_%a.out
#SBATCH --error=results/experiment1/logs/slurm/retry_tabicl_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=gpu_p100
#SBATCH --array=0-15%8

# ============================================================
# RETRY SCRIPT FOR TABICL HPO (Copy NO_HPO to HPO)
# ============================================================
# TabICL is in NO_HPO_METHODS, so this will:
# 1. Check if NO_HPO results exist
# 2. Copy NO_HPO results to HPO key
# 3. Skip actual HPO (as intended)
#
# We're rerunning because the previous copy didn't happen
# due to a bug.
# ============================================================

# Load conda - use DATA directory path
if [ -f "$VSC_DATA/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda.sh"
    echo "Tried: $VSC_DATA/miniconda3/etc/profile.d/conda.sh"
    echo "Tried: $HOME/miniconda3/etc/profile.d/conda.sh"
    exit 1
fi

# Activate environment
conda activate TabPFNCredit

echo "========================================================================"
echo "EXP1 - TABICL HPO RETRY (Copy NO_HPO to HPO)"
echo "========================================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Cluster:      Genius"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          P100 (16GB VRAM)"
echo "CPUs:         8"
echo "Memory:       32GB"
echo "Time limit:   6h"
echo "Python:       $(which python)"
echo "Python ver:   $(python --version)"
echo "Conda env:    $CONDA_DEFAULT_ENV"
echo "========================================================================"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Define task list - ALL PD datasets with tabicl/HPO
declare -a TASKS=(
    "0001.gmsc tabicl pd HPO"
    "0002.taiwan_creditcard tabicl pd HPO"
    "0003.vehicle_loan tabicl pd HPO"
    "0004.lendingclub tabicl pd HPO"
    "0005.case_study tabicl pd HPO"
    "0006.myhom tabicl pd HPO"
    "0007.hackerearth tabicl pd HPO"
    "0008.cobranded tabicl pd HPO"
    "0009.german tabicl pd HPO"
    "0010.bank_status tabicl pd HPO"
    "0011.thomas tabicl pd HPO"
    "0012.loan_default tabicl pd HPO"
    "0013.home_credit tabicl pd HPO"
    "0014.hmeq tabicl pd HPO"
    "0015.algorithmwatch tabicl pd HPO"
    "0016.fico tabicl pd HPO"
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
echo "  Method:   $METHOD (NO_HPO method)"
echo "  Task:     $TASK_TYPE"
echo "  HPO Mode: $HPO_MODE (will copy from NO_HPO)"
echo ""

# Add random delay to stagger starts
DELAY=$((RANDOM % 60 + 1))
echo "Sleeping ${DELAY}s to stagger job starts..."
sleep $DELAY

# Run the task - will trigger NO_HPO -> HPO copy
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
    echo "Task completed successfully (NO_HPO results copied to HPO)"
else
    echo "Task failed with exit code: $EXIT_CODE"
fi
echo "========================================================================"

exit $EXIT_CODE