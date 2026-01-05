#!/bin/bash
#SBATCH --job-name=Exp1_Missing_CPU
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=180G
#SBATCH --cluster=genius
#SBATCH --partition=batch
#SBATCH --array=0-7%4

# ============================================================================
# Experiment1 - Retry Missing CPU Tasks (MAXIMUM RESOURCES)
# ============================================================================
# Resources: FULL NODE on Genius
# - 36 CPUs (full node)
# - 180GB RAM (leaving 12GB for OS, out of 192GB total)
# - 168h walltime (7 days - maximum typical limit)
# 
# Total tasks: 8 (4 methods × 2 modes)
# Cluster: Genius (CPU partition)
# ============================================================================

set -e  # Exit on error

# Stagger job starts to prevent I/O congestion
RANDOM_DELAY=$((RANDOM % 60 + 1))
echo "Staggering start: sleeping ${RANDOM_DELAY}s..."
sleep ${RANDOM_DELAY}

# Configuration
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
echo "Memory:        ${SLURM_MEM_PER_NODE}MB"
echo "Time Limit:    ${SLURM_TIMELIMIT}"
echo "Start Time:    $(date)"
echo "========================================================================"

# Define all missing CPU tasks
# Format: "dataset method task_type hpo_mode"
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

# Get task for this array index
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

# Setup conda environment
echo "Setting up environment..."
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit
export LD_LIBRARY_PATH="${VSC_DATA}/miniconda3/envs/TabPFNCredit/lib:${LD_LIBRARY_PATH}"

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Change to project directory
cd ${VSC_DATA}/TabPFNCredit

# Run the task
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