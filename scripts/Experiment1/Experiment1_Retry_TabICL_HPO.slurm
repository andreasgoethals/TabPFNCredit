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
# Staggered start to prevent I/O congestion
sleep $((RANDOM % 60 + 1))

# Force unbuffered I/O
export PYTHONUNBUFFERED=1

# Setup conda from $VSC_DATA (SAME AS WORKING SCRIPT)
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

# USE CONDA'S C++ LIBRARIES
export LD_LIBRARY_PATH="${VSC_DATA}/miniconda3/envs/TabPFNCredit/lib:${LD_LIBRARY_PATH}"

# Navigate to project
cd $VSC_DATA/TabPFNCredit

echo "========================================================================"
echo "EXP1 - TABICL HPO RETRY (Copy NO_HPO to HPO)"
echo "========================================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Cluster:      Genius"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
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

# Run the task with unbuffered output - will trigger NO_HPO -> HPO copy
python -u scripts/Experiment1/Experiment1_GPU.py \
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