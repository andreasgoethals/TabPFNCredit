#!/bin/bash
#SBATCH --job-name=exp2_foundation0
#SBATCH --cluster=wice
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --partition=gpu_h100
#SBATCH --time=48:00:00
#SBATCH --output=${VSC_DATA}/TabPFNCredit/results/experiment2/logs/slurm/foundation0_%A_%a.out
#SBATCH --error=${VSC_DATA}/TabPFNCredit/results/experiment2/logs/slurm/foundation0_%A_%a.err
#SBATCH --requeue
#SBATCH --array=0-46%8

# ---------------------------------------------------------
# EXPERIMENT 2 - FOUNDATION GPU - BATCH 0 - Tasks 0-46
# Memory: 100G (soft isolation on wICE H100)
# ---------------------------------------------------------

set -euo pipefail
sleep $((SLURM_ARRAY_TASK_ID % 30))  # deterministic stagger, avoids thundering-herd I/O

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit
export LD_LIBRARY_PATH="${VSC_DATA}/miniconda3/envs/TabPFNCredit/lib:${LD_LIBRARY_PATH:-}"

cd "${VSC_DATA}/TabPFNCredit"
mkdir -p "${VSC_DATA}/TabPFNCredit/results/experiment2/logs/slurm"

echo "EXP 2 - FOUNDATION - BATCH 0"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node:     ${SLURMD_NODENAME}"
echo "GPU:      ${CUDA_VISIBLE_DEVICES:-N/A}"

GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + 0))

python -u scripts/Experiment2/Experiment2_GPU_Foundation.py --array_id="${GLOBAL_TASK_ID}" --verbose

EXIT_CODE=$?
echo "Task completed with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
