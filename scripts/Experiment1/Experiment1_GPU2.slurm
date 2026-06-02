#!/bin/bash
#SBATCH --job-name=exp1_gpu2
#SBATCH --cluster=genius
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=45G
#SBATCH --partition=gpu_p100
#SBATCH --time=70:00:00
#SBATCH --output=${VSC_DATA}/TabPFNCredit/results/experiment1/logs/slurm/gpu2_%A_%a.out
#SBATCH --error=${VSC_DATA}/TabPFNCredit/results/experiment1/logs/slurm/gpu2_%A_%a.err
#SBATCH --requeue
#SBATCH --array=0-59%16

# ---------------------------------------------------------
# EXPERIMENT 1 - GPU - BATCH 2 - Tasks 800-859
# ---------------------------------------------------------

set -euo pipefail
sleep $((SLURM_ARRAY_TASK_ID % 30))  # deterministic stagger, avoids thundering-herd I/O

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit
export LD_LIBRARY_PATH="${VSC_DATA}/miniconda3/envs/TabPFNCredit/lib:${LD_LIBRARY_PATH:-}"

cd "${VSC_DATA}/TabPFNCredit"
mkdir -p "${VSC_DATA}/TabPFNCredit/results/experiment1/logs/slurm"

echo "EXPERIMENT 1 - GPU - BATCH 2"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node:     ${SLURMD_NODENAME}"
echo "GPU:      ${CUDA_VISIBLE_DEVICES:-N/A}"

GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + 800))

python -u scripts/Experiment1/Experiment1_GPU.py --array_id="${GLOBAL_TASK_ID}" --verbose

EXIT_CODE=$?
echo "Task completed with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
