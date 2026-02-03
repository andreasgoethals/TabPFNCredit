#!/bin/bash
#SBATCH --job-name=xgboost_parallel
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab" 
#SBATCH --output=results/experiment1/logs/slurm/rerun_d12_%A_%a.out
#SBATCH --error=results/experiment1/logs/slurm/rerun_d12_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=batch
#SBATCH --array=0-1

# Force unbuffered I/O
export PYTHONUNBUFFERED=1

# Setup conda environment
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

# Navigate to project root
cd $VSC_DATA/TabPFNCredit

# Map Array ID to HPO Mode
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    MODE="NO_HPO"
else
    MODE="HPO"
fi

echo "=========================================="
echo "PARALLEL RERUN: XGBOOST - DATASET 12"
echo "Job ID:    $SLURM_JOB_ID"
echo "Array ID:  $SLURM_ARRAY_TASK_ID"
echo "HPO Mode:  $MODE"
echo "Node:      $SLURMD_NODENAME"
echo "=========================================="

# Run the specific task
python -u scripts/Experiment1/Experiment1_CPU.py \
    --dataset "0012.loan_default" \
    --method "xgboost" \
    --task_type "pd" \
    --hpo_mode "$MODE" \
    --verbose

echo "=========================================="
echo "TASK $MODE COMPLETED"
echo "=========================================="