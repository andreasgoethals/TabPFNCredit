#!/bin/bash
#SBATCH --job-name=exp2_gpu0
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab"
#SBATCH --nodes="1"
#SBATCH --output=results/experiment2/logs/slurm/gpu0_%A_%a.out
#SBATCH --error=results/experiment2/logs/slurm/gpu0_%A_%a.err
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=45G
#SBATCH --partition=gpu_p100
#SBATCH --array=0-110%12

# ---------------------------------------------------------
# EXPERIMENT 2: LEARNING CURVE ANALYSIS
# BATCH 0: Tasks 0-110
# ---------------------------------------------------------
# Each task runs a learning curve for ONE method + ONE dataset
# Loop over row_limits happens inside Python
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

# USE CONDA'S C++ LIBRARIES
export LD_LIBRARY_PATH="${VSC_DATA}/miniconda3/envs/TabPFNCredit/lib:${LD_LIBRARY_PATH}"

# Navigate to project
cd $VSC_DATA/TabPFNCredit

echo "=========================================="
echo "EXPERIMENT 2 - LEARNING CURVE - GPU - BATCH 0"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Batch:        0"
echo "Task offset:  0"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Calculate global task ID from batch offset
GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + 0))

# Run GPU orchestrator with global task ID
python -u scripts/Experiment2/Experiment2_GPU.py --array_id=$GLOBAL_TASK_ID --verbose
