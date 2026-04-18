#!/bin/bash
#SBATCH --job-name=exp0_gpu1
#SBATCH --cluster=genius
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=20G
#SBATCH --partition=gpu_p100
#SBATCH --time=00:59:00
#SBATCH --output=${VSC_DATA}/TabPFNCredit/results/experiment0/logs/slurm/gpu1_%A_%a.out
#SBATCH --error=${VSC_DATA}/TabPFNCredit/results/experiment0/logs/slurm/gpu1_%A_%a.err
#SBATCH --requeue
#SBATCH --array=0-290%16

# ---------------------------------------------------------
# EXPERIMENT 0 - GPU - BATCH 1 - Tasks 400-690
# ---------------------------------------------------------

set -euo pipefail
sleep $((RANDOM % 60 + 1))

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit
export LD_LIBRARY_PATH="${VSC_DATA}/miniconda3/envs/TabPFNCredit/lib:${LD_LIBRARY_PATH:-}"

cd "${VSC_DATA}/TabPFNCredit"
mkdir -p "${VSC_DATA}/TabPFNCredit/results/experiment0/logs/slurm"

echo "EXP 0 - GPU - BATCH 1"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node:     ${SLURMD_NODENAME}"

GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + 400))

python -u scripts/Experiment0/Experiment0_GPU.py --array_id="${GLOBAL_TASK_ID}" --verbose

EXIT_CODE=$?
echo "Task completed with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
