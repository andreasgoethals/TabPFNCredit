#!/bin/bash
#SBATCH --job-name=exp1_gpu
#SBATCH --cluster="wice"
#SBATCH --account="lp_verbekelab" 
#SBATCH --output=results/experiment1/logs/slurm/gpu_%A_%a.out
#SBATCH --error=results/experiment1/logs/slurm/gpu_%A_%a.err
#SBATCH --time=71:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=45G
#SBATCH --partition=gpu_a100
#SBATCH --array=0-32%16

# Load conda from your installation
source $HOME/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate TabPFNCredit

echo "=========================================="
echo "GPU JOB"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Python:       $(which python)"
echo "Python ver:   $(python --version)"
echo "=========================================="

# Run GPU orchestrator
python scripts/Experiment1/Experiment1_GPU.py --array_id=$SLURM_ARRAY_TASK_ID --verbose
