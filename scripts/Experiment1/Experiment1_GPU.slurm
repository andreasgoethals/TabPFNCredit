#!/usr/bin/env bash
#SBATCH --job-name=exp1_gpu
#SBATCH --partition=gpu_a100
#SBATCH --cluster=wice          
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=40:00:00
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=andreas.goethals@kuleuven.be
#SBATCH --output=/data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/slurm/gpu_%A_%a.out
#SBATCH --error=/data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/slurm/gpu_%A_%a.err
#SBATCH --array=PLACEHOLDER

export PYTHONUNBUFFERED=1
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

cd $VSC_DATA/TabPFNCredit

mkdir -p results/experiment1/logs/slurm

TASK_LIST="results/experiment1/task_list_gpu.json"

if [ ! -f "$TASK_LIST" ]; then
    echo "ERROR: GPU task list not found"
    exit 1
fi

echo "=== GPU Task $SLURM_ARRAY_TASK_ID started at $(date) ==="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv

python -u scripts/Experiment1/Experiment1.py \
    --task_idx=$SLURM_ARRAY_TASK_ID \
    --verbose

EXIT_CODE=$?
echo "=== Task completed at $(date) with exit code $EXIT_CODE ==="
exit $EXIT_CODE