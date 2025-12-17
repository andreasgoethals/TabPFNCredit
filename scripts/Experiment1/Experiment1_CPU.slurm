#!/bin/bash
#SBATCH --job-name=exp1_cpu
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab" 
#SBATCH --output=results/experiment1/logs/slurm/cpu_%A_%a.out
#SBATCH --error=results/experiment1/logs/slurm/cpu_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --partition=batch
#SBATCH --array=0-17%18

# Load conda from your installation
source $HOME/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate TabPFNCredit

echo "=========================================="
echo "CPU JOB"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Python:       $(which python)"
echo "Python ver:   $(python --version)"
echo "=========================================="

# Run CPU orchestrator
python scripts/Experiment1/Experiment1_CPU.py --array_id=$SLURM_ARRAY_TASK_ID --verbose
