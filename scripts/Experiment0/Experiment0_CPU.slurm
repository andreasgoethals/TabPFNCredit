#!/usr/bin/bash
#SBATCH --job-name=exp0_cpu
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab" 
#SBATCH --output=${VSC_DATA}/TabPFNCredit/results/experiment0/logs/slurm/cpu_%A_%a.out
#SBATCH --error=${VSC_DATA}/TabPFNCredit/results/experiment0/logs/slurm/cpu_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --partition=batch
#SBATCH --array=0-32%32

# Force unbuffered I/O
export PYTHONUNBUFFERED=1

# Load required modules
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a

# Activate venv
source ${VSC_DATA}/TabPFNCredit_venv/bin/activate

# Navigate to project
cd ${VSC_DATA}/TabPFNCredit

echo "=========================================="
echo "EXPERIMENT 0 - CPU VALIDATION"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Python:       $(which python)"
echo "=========================================="

# Run CPU orchestrator
python -u scripts/Experiment0/Experiment0_CPU.py --array_id=$SLURM_ARRAY_TASK_ID --verbose
