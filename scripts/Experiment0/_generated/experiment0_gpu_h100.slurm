#!/bin/bash -l
#SBATCH --job-name=experiment0_gpu_h100
#SBATCH --clusters=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu_cmode=shared
#SBATCH --mem=186000M
#SBATCH --time=00:40:00
#SBATCH --output=${VSC_DATA}/TabPFNCredit/results/experiment0/logs/experiment0_gpu_h100_%A_%a.out
#SBATCH --error=${VSC_DATA}/TabPFNCredit/results/experiment0/logs/experiment0_gpu_h100_%A_%a.err
#SBATCH --requeue
#SBATCH --array=0-167%16

set -euo pipefail

# Deterministic stagger (0-29 s) based on array index -- avoids the
# thundering-herd I/O storm that RANDOM%60 would create.
sleep $((${SLURM_ARRAY_TASK_ID:-0} % 30))

# Unbuffered Python output so SLURM streams stdout in real time.
export PYTHONUNBUFFERED=1

# Memory-fragmentation mitigation for long-running PyTorch jobs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# VSC: never load modules in ~/.bashrc; do it here.
module --force purge

# Activate the conda env (the docs prefer Miniforge but the repo's
# legacy env is fine -- we just keep it working).
if [ -d "${VSC_DATA}/miniforge3" ]; then
    source "${VSC_DATA}/miniforge3/etc/profile.d/conda.sh"
else
    source "${VSC_DATA}/miniconda3/etc/profile.d/conda.sh"
fi
conda activate TabPFNCredit

# Conda's C++ runtime (avoids glibc++ mismatches on VSC nodes).
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${VSC_DATA}/TabPFNCredit/results/experiment0/logs"
echo "============================================="
echo "Experiment0 on wice/gpu_h100"
echo "JobID:   ${SLURM_JOB_ID}"
echo "ArrayID: ${SLURM_ARRAY_TASK_ID}"
echo "Node:    ${SLURMD_NODENAME}"
echo "GPU:     ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "Plan:    experiment0_gpu_h100_plan.json (slot ${SLURM_ARRAY_TASK_ID})"
echo "============================================="

cd "${VSC_DATA}/TabPFNCredit"

export TABPFN_RESULTS_ROOT="${VSC_DATA}/TabPFNCredit/results"
mkdir -p "${TABPFN_RESULTS_ROOT}"

tabpfncredit slurm-task \
    --experiment Experiment0 \
    --partition gpu_h100 \
    --array-id "${SLURM_ARRAY_TASK_ID}"

EXIT_CODE=$?
echo "Task done, exit=${EXIT_CODE}"
exit ${EXIT_CODE}
