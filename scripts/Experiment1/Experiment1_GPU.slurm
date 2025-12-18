#!/bin/bash
#SBATCH --job-name=exp1_gpu
#SBATCH --cluster="wice"
#SBATCH --account="lp_verbekelab" 
#SBATCH --nodes="1" 
#SBATCH --output=results/experiment1/logs/slurm/gpu_%A_%a.out
#SBATCH --error=results/experiment1/logs/slurm/gpu_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=45G
#SBATCH --partition=gpu_a100
#SBATCH --array=0-32%16

# Force unbuffered I/O
export PYTHONUNBUFFERED=1

# Setup conda from $VSC_DATA (correct path)
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

# Navigate to project
cd $VSC_DATA/TabPFNCredit

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
python -u scripts/Experiment1/Experiment1_GPU.py --array_id=$SLURM_ARRAY_TASK_ID --verbose
