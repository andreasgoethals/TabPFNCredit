#!/bin/bash
#SBATCH --job-name=exp2_cpu0
#SBATCH --cluster=genius
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=batch
#SBATCH --time=72:00:00
#SBATCH --output=${VSC_DATA}/TabPFNCredit/results/experiment2/logs/slurm/cpu0_%A_%a.out
#SBATCH --error=${VSC_DATA}/TabPFNCredit/results/experiment2/logs/slurm/cpu0_%A_%a.err
#SBATCH --requeue
#SBATCH --array=0-10%11

# ---------------------------------------------------------
# EXPERIMENT 2 - CPU - BATCH 0 - Tasks 0-10
# ---------------------------------------------------------

set -euo pipefail
sleep $((SLURM_ARRAY_TASK_ID % 30))  # deterministic stagger, avoids thundering-herd I/O
export PYTHONUNBUFFERED=1

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

cd "${VSC_DATA}/TabPFNCredit"
mkdir -p "${VSC_DATA}/TabPFNCredit/results/experiment2/logs/slurm"

echo "EXP 2 - CPU - BATCH 0"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Array ID: ${SLURM_ARRAY_TASK_ID}"

GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + 0))

python -u scripts/Experiment2/Experiment2_CPU.py --array_id="${GLOBAL_TASK_ID}" --verbose

EXIT_CODE=$?
echo "Task completed with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
