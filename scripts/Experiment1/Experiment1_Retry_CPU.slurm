#!/bin/bash
#SBATCH --job-name=exp1_retry_cpu
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab" 
#SBATCH --nodes="1" 
#SBATCH --output=results/experiment1/logs/slurm/retry_cpu_%A_%a.out
#SBATCH --error=results/experiment1/logs/slurm/retry_cpu_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --partition=batch
#SBATCH --array=0-1%2

# ============================================================
# RETRY SCRIPT FOR CPU TIMEOUTS
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
echo "EXP1 - CPU RETRY (Maxed Resources)"
echo "========================================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Node:         $SLURMD_NODENAME"
echo "CPUs:         16"
echo "Memory:       60GB"
echo "Time limit:   72h"
echo "Python:       $(which python)"
echo "Python ver:   $(python --version)"
echo "Conda env:    $CONDA_DEFAULT_ENV"
echo "========================================================================"

# Define task list (dataset, method, task_type, hpo_mode)
declare -a TASKS=(
    "0007.hackerearth knn pd HPO"
    "0013.home_credit knn pd HPO"
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

# Run the task with unbuffered output
python -u scripts/Experiment1/Experiment1_CPU.py \
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