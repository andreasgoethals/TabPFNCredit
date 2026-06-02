#!/bin/bash
#SBATCH --job-name=exp0_cpu0
#SBATCH --cluster=genius
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --output=${VSC_DATA}/TabPFNCredit/results/experiment0/logs/slurm/cpu0_%A_%a.out
#SBATCH --error=${VSC_DATA}/TabPFNCredit/results/experiment0/logs/slurm/cpu0_%A_%a.err
#SBATCH --requeue
#SBATCH --array=0-132%32

# ---------------------------------------------------------
# EXPERIMENT 0 - CPU VALIDATION - BATCH 0 - Tasks 0-132
# ---------------------------------------------------------

set -euo pipefail
sleep $((SLURM_ARRAY_TASK_ID % 30))  # deterministic stagger, avoids thundering-herd I/O
export PYTHONUNBUFFERED=1

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

cd "${VSC_DATA}/TabPFNCredit"
mkdir -p "${VSC_DATA}/TabPFNCredit/results/experiment0/logs/slurm"

echo "EXP 0 - CPU - BATCH 0"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Array ID: ${SLURM_ARRAY_TASK_ID}"

GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + 0))

python -u scripts/Experiment0/Experiment0_CPU.py --array_id="${GLOBAL_TASK_ID}" --verbose

EXIT_CODE=$?
echo "Task completed with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
