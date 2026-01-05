#!/bin/bash
#SBATCH --job-name=Exp1_Miss_CPU
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=5000M
#SBATCH --cluster=genius
#SBATCH --partition=batch
#SBATCH --array=0-7%4

# ============================================================================
# Experiment1 - Retry Missing CPU Tasks
# ============================================================================
# Resources (VERIFIED from scontrol):
# - 36 CPUs (full node)
# - 5GB per CPU × 36 = 180GB total RAM
# - 72h walltime (3 days - max for batch partition)
# - Cluster: genius
# - Partition: batch
# ============================================================================

set -e

# Stagger starts
RANDOM_DELAY=$((RANDOM % 60 + 1))
echo "Staggering start: sleeping ${RANDOM_DELAY}s..."
sleep ${RANDOM_DELAY}

echo "========================================================================"
echo "SLURM Job Configuration"
echo "========================================================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Job Name:      ${SLURM_JOB_NAME}"
echo "Node:          ${SLURMD_NODENAME}"
echo "Cluster:       ${SLURM_CLUSTER_NAME}"
echo "Partition:     ${SLURM_JOB_PARTITION}"
echo "CPUs:          ${SLURM_CPUS_PER_TASK}"
echo "Mem per CPU:   5000MB"
echo "Total Memory:  180GB"
echo "Time Limit:    72h"
echo "Start Time:    $(date)"
echo "========================================================================"

# Define missing CPU tasks
TASKS=(
    "0005.base_modelisation catboost lgd HPO"
    "0005.base_modelisation catboost lgd NO_HPO"
    "0005.base_modelisation lightgbm lgd HPO"
    "0005.base_modelisation lightgbm lgd NO_HPO"
    "0005.base_modelisation tabpfn_v2 lgd HPO"
    "0005.base_modelisation tabpfn_v2 lgd NO_HPO"
    "0005.base_modelisation xgboost lgd HPO"
    "0005.base_modelisation xgboost lgd NO_HPO"
)

TASK_INFO=(${TASKS[$SLURM_ARRAY_TASK_ID]})
DATASET=${TASK_INFO[0]}
METHOD=${TASK_INFO[1]}
TASK_TYPE=${TASK_INFO[2]}
HPO_MODE=${TASK_INFO[3]}

echo ""
echo "========================================================================"
echo "Task Configuration"
echo "========================================================================"
echo "Dataset:    ${DATASET}"
echo "Method:     ${METHOD}"
echo "Task Type:  ${TASK_TYPE}"
echo "HPO Mode:   ${HPO_MODE}"
echo "========================================================================"
echo ""

# Setup environment
echo "Setting up environment..."
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit
export LD_LIBRARY_PATH="${VSC_DATA}/miniconda3/envs/TabPFNCredit/lib:${LD_LIBRARY_PATH}"

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

cd ${VSC_DATA}/TabPFNCredit

# Run task
echo "Starting execution at $(date)..."
echo "========================================================================"

python -u scripts/Experiment1/Experiment1_CPU.py \
    --dataset "${DATASET}" \
    --method "${METHOD}" \
    --task_type "${TASK_TYPE}" \
    --hpo_mode "${HPO_MODE}"

EXIT_CODE=$?

echo "========================================================================"
echo "Execution completed at $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "========================================================================"

exit ${EXIT_CODE}