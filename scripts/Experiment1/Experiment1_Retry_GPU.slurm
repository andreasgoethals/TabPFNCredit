#!/bin/bash
#SBATCH --job-name=Exp1_Missing_GPU
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=240G
#SBATCH --cluster=wice
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --array=0-33%8

# ============================================================================
# Experiment1 - Retry Missing GPU Tasks (MAXIMUM RESOURCES)
# ============================================================================
# Resources: MAXIMUM on wICE GPU nodes
# - 72 CPUs (requesting maximum, will get what's available per node)
# - 240GB RAM (conservative maximum for GPU nodes)
# - 1 full GPU (any available: H100, A100, V100, or P100)
# - 168h walltime (7 days - maximum typical limit)
# 
# Total tasks: 34
# Cluster: wICE (GPU partition)
# 
# NOTE: If 72 CPUs not available, SLURM will allocate maximum available
#       on the GPU node type. Typical GPU nodes have 36-72 cores.
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
echo "GPUs:          ${SLURM_GPUS_PER_NODE}"
echo "Time Limit:    ${SLURM_TIMELIMIT}"
echo "Start Time:    $(date)"
echo "========================================================================"

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
echo ""

# Define all missing GPU tasks
# Format: "dataset method task_type hpo_mode"
TASKS=(
    # PD tasks - TabICL
    "0007.hackerearth tabicl pd HPO"
    "0007.hackerearth tabicl pd NO_HPO"
    "0013.home_credit tabicl pd HPO"
    "0013.home_credit tabicl pd NO_HPO"
    # PD tasks - Other methods
    "0011.thomas danets pd NO_HPO"
    "0011.thomas tabm pd NO_HPO"
    # LGD tasks - base_modelisation dataset
    "0005.base_modelisation autoint lgd HPO"
    "0005.base_modelisation autoint lgd NO_HPO"
    "0005.base_modelisation dcn2 lgd HPO"
    "0005.base_modelisation dcn2 lgd NO_HPO"
    "0005.base_modelisation excelformer lgd NO_HPO"
    "0005.base_modelisation ftt lgd HPO"
    "0005.base_modelisation ftt lgd NO_HPO"
    "0005.base_modelisation mlp lgd HPO"
    "0005.base_modelisation mlp lgd NO_HPO"
    "0005.base_modelisation mlp_plr lgd HPO"
    "0005.base_modelisation mlp_plr lgd NO_HPO"
    "0005.base_modelisation modernNCA lgd HPO"
    "0005.base_modelisation modernNCA lgd NO_HPO"
    "0005.base_modelisation ptarl lgd HPO"
    "0005.base_modelisation ptarl lgd NO_HPO"
    "0005.base_modelisation realmlp lgd HPO"
    "0005.base_modelisation realmlp lgd NO_HPO"
    "0005.base_modelisation resnet lgd HPO"
    "0005.base_modelisation resnet lgd NO_HPO"
    "0005.base_modelisation tabnet lgd HPO"
    "0005.base_modelisation tabnet lgd NO_HPO"
    "0005.base_modelisation tangos lgd HPO"
    "0005.base_modelisation tangos lgd NO_HPO"
    # LGD tasks - base_model dataset
    "0004.base_model t2gformer lgd HPO"
    "0004.base_model t2gformer lgd NO_HPO"
    "0004.base_model tabm lgd HPO"
    "0004.base_model tabm lgd NO_HPO"
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

python -u scripts/Experiment1/Experiment1_GPU.py \
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