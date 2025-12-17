#!/bin/bash
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
#SBATCH --array=0-20%16

# Load modules
module purge
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/11.7.0

# Activate environment
source activate TabPFNCredit

echo "=========================================="
echo "GPU JOB"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Run GPU orchestrator
python scripts/Experiment1/Experiment1_GPU.py --array_id=$SLURM_ARRAY_TASK_ID --verbose
