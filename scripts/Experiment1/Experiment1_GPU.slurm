#!/bin/bash
#SBATCH --job-name=exp1_gpu
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab" 
#SBATCH --nodes="1" 
#SBATCH --output=${VSC_DATA}/TabPFNCredit/results/experiment1/logs/slurm/gpu_%A_%a.out
#SBATCH --error=${VSC_DATA}/TabPFNCredit/results/experiment1/logs/slurm/gpu_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=gpu_p100
#SBATCH --array=0-32%16

# Force unbuffered I/O
export PYTHONUNBUFFERED=1

# Load required modules
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a
module load CUDA/11.7.0

# Activate venv
source ${VSC_DATA}/TabPFNCredit_venv/bin/activate

# Navigate to project
cd ${VSC_DATA}/TabPFNCredit

echo "=========================================="
echo "EXPERIMENT 1 - GPU JOB"
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
