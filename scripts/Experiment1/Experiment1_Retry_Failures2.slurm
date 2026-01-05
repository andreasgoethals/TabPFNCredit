#!/bin/bash
#SBATCH --job-name=retry_failures
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=69:00:00
#SBATCH --partition=gpu_p100
#SBATCH --array=0

# --- LOG PATHS (Absolute) ---
#SBATCH --output=/data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/slurm/retry_%A_%a.out
#SBATCH --error=/data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/slurm/retry_%A_%a.err

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
set -e

# Stagger starts slightly to reduce IO contention
sleep $((RANDOM % 30 + 1))

export PYTHONUNBUFFERED=1

# Setup conda
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

# USE CONDA'S C++ LIBRARIES
export LD_LIBRARY_PATH="${VSC_DATA}/miniconda3/envs/TabPFNCredit/lib:${LD_LIBRARY_PATH}"

# Navigate to project root
cd $VSC_DATA/TabPFNCredit

echo "=========================================="
echo "RETRYING FAILURES"
echo "Job ID:   $SLURM_JOB_ID"
echo "Array ID: $SLURM_ARRAY_TASK_ID"
echo "Node:     $SLURMD_NODENAME"
echo "=========================================="

# ---------------------------------------------------------
# DEFINE TASKS
# ---------------------------------------------------------
# Format: "dataset method task hpo_mode"
TASKS=(
    "0013.home_credit tabicl pd NO_HPO"
)

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
# If method is catboost, hide GPU to force CPU execution and avoid driver errors
if [ "$METHOD" == "catboost" ]; then
    echo "[FIX] Detected CatBoost: Hiding GPU to prevent driver mismatch errors."
    export CUDA_VISIBLE_DEVICES=""
fi

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
echo "Starting execution..."

python -u scripts/Experiment1/Experiment1_GPU.py \
    --dataset "${DATASET}" \
    --method "${METHOD}" \
    --task_type "${TASK_TYPE}" \
    --hpo_mode "${HPO_MODE}" 

echo "Done."