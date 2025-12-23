#!/usr/bin/bash
#SBATCH --job-name=exp0_gpu
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab" 
#SBATCH --nodes="1" 
#SBATCH --output=results/experiment0/logs/slurm/gpu_%A_%a.out
#SBATCH --error=results/experiment0/logs/slurm/gpu_%A_%a.err
#SBATCH --time=00:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --partition=gpu_p100
#SBATCH --array=0-653%16

# ---------------------------------------------------------
# STAGGERED START TO PREVENT I/O CONGESTION
# ---------------------------------------------------------
# Sleep for a random duration between 1 and 60 seconds.
# This ensures jobs don't all hit the file system at the exact same millisecond.
sleep $((RANDOM % 60 + 1))
# ---------------------------------------------------------

# Force unbuffered I/O
export PYTHONUNBUFFERED=1

# Setup conda from $VSC_DATA
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

# USE CONDA'S C++ LIBRARIES
export LD_LIBRARY_PATH="${VSC_DATA}/miniconda3/envs/TabPFNCredit/lib:${LD_LIBRARY_PATH}"

# Navigate to project
cd $VSC_DATA/TabPFNCredit

echo "=========================================="
echo "EXPERIMENT 0 - GPU VALIDATION"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Run GPU orchestrator
python -u scripts/Experiment0/Experiment0_GPU.py --array_id=$SLURM_ARRAY_TASK_ID --verbose
