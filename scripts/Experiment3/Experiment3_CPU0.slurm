#!/bin/bash
#SBATCH --job-name=exp3_cpu0
#SBATCH --cluster=genius
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --output=results/experiment3/logs/slurm/cpu0_%A_%a.out
#SBATCH --error=results/experiment3/logs/slurm/cpu0_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=batch
#SBATCH --array=0-7%8

# ---------------------------------------------------------
# EXPERIMENT 3: CLASS IMBALANCE ANALYSIS - CPU
# BATCH 0: Tasks 0-7
# ---------------------------------------------------------
# Cluster: genius | Partition: batch | Memory: 40G
# Each task runs an imbalance curve for ONE method + ONE dataset
# Loop over minority_proportion happens inside Python
# ---------------------------------------------------------
# STAGGERED START TO PREVENT I/O CONGESTION
# ---------------------------------------------------------
sleep $((RANDOM % 60 + 1))
# ---------------------------------------------------------

# Force unbuffered I/O
export PYTHONUNBUFFERED=1

# Setup conda from $VSC_DATA
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

# Navigate to project
cd $VSC_DATA/TabPFNCredit

echo "=========================================="
echo "EXPERIMENT 3 - IMBALANCE - CPU - BATCH 0"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Batch:        0"
echo "Task offset:  0"
echo "Node:         $SLURMD_NODENAME"
echo "=========================================="

# Calculate global task ID from batch offset
GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + 0))

# Run CPU orchestrator with global task ID
python -u scripts/Experiment3/Experiment3_CPU.py --array_id=$GLOBAL_TASK_ID --verbose

# Exit status
EXIT_CODE=$?
echo "=========================================="
echo "Task completed with exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
