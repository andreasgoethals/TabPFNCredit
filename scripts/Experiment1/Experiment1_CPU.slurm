#!/usr/bin/env bash
#SBATCH --job-name=exp1_cpu
#SBATCH --cluster=genius             
#SBATCH --partition=Skylake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=andreas.goethals@kuleuven.be
#SBATCH --output=/data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/slurm/cpu_%A_%a.out
#SBATCH --error=/data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/slurm/cpu_%A_%a.err
#SBATCH --array=PLACEHOLDER

export PYTHONUNBUFFERED=1
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

cd $VSC_DATA/TabPFNCredit

mkdir -p results/experiment1/logs/slurm

TASK_LIST="results/experiment1/task_list_cpu.json"

if [ ! -f "$TASK_LIST" ]; then
    echo "ERROR: CPU task list not found"
    exit 1
fi

echo "=== CPU Task $SLURM_ARRAY_TASK_ID started at $(date) ==="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"

python -u scripts/Experiment1/Experiment1.py \
    --task_idx=$SLURM_ARRAY_TASK_ID \
    --verbose

EXIT_CODE=$?
echo "=== Task completed at $(date) with exit code $EXIT_CODE ==="
exit $EXIT_CODE