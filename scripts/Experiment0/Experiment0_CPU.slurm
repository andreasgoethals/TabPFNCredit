#!/usr/bin/bash
#SBATCH --job-name=exp0_cpu
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab" 
#SBATCH --output=../../results/experiment0/logs/slurm/cpu_%A_%a.out
#SBATCH --error=../../results/experiment0/logs/slurm/cpu_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --partition=batch
#SBATCH --array=0-32%32

# Force unbuffered I/O
export PYTHONUNBUFFERED=1

# Setup conda from $VSC_DATA
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

# Navigate to project root
cd ${VSC_DATA}/TabPFNCredit

echo "=========================================="
echo "EXPERIMENT 0 - CPU VALIDATION"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Python:       $(which python)"
echo "Conda env:    $CONDA_DEFAULT_ENV"
echo "=========================================="

# Run CPU orchestrator
python -u scripts/Experiment0/Experiment0_CPU.py --array_id=$SLURM_ARRAY_TASK_ID --verbose
