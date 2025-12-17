#!/bin/bash
#SBATCH --job-name=exp1_cpu
#SBATCH --cluster=genius             
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=andreas.goethals@kuleuven.be
#SBATCH --output=/data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/slurm/cpu_%A_%a.out
#SBATCH --error=/data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/slurm/cpu_%A_%a.err
#SBATCH --array=0-29%30

# Load modules
module purge
module load Python/3.10.8-GCCcore-12.2.0

# Activate environment
source activate TabPFNCredit

echo "=========================================="
echo "CPU JOB"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Node:         $SLURMD_NODENAME"
echo "=========================================="

# Run CPU orchestrator
python scripts/Experiment1/Experiment1_CPU.py --array_id=$SLURM_ARRAY_TASK_ID --verbose
