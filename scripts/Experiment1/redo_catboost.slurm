#!/bin/bash
#SBATCH --job-name=redo_catboost
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --output=results/logs/redo_catboost_%j.out
#SBATCH --error=results/logs/redo_catboost_%j.err

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
# Force unbuffered I/O
export PYTHONUNBUFFERED=1

# Setup conda
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

# Navigate to project root
cd $VSC_DATA/TabPFNCredit

echo "=========================================="
echo "REDOING CATBOOST FOR 0005.base_modelisation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "=========================================="

# ---------------------------------------------------------
# EMBEDDED PYTHON SCRIPT
# ---------------------------------------------------------
python -u - <<EOF
import sys
import os
import logging

# Ensure project root is in path
sys.path.insert(0, os.getcwd())

from src.methods.method_runner import run_talent_method

# Common configuration matching CONFIG_DATA.yaml
# Dataset: 0005.base_modelisation (LGD task)
PARAMS = {
    'task': 'lgd',
    'dataset': '0005.base_modelisation',
    'method': 'catboost',
    'test_size': 0.2,
    'val_size': 0.2,
    'cv_splits': 5,
    'seed': 42,
    'row_limit': None,
    'clean_temp_dir': True,
    'verbose': True,
    # Standard settings from HPO_runner.py defaults
    'max_epoch': 200,
    'batch_size': 1024,
    'early_stopping': True,
    'early_stopping_patience': 10
}

print(f"\n[{'='*40}]")
print(f"STARTING: CatBoost NO_HPO")
print(f"[{'='*40}]")

try:
    run_talent_method(
        tune=False,
        **PARAMS
    )
    print("SUCCESS: CatBoost NO_HPO")
except Exception as e:
    print(f"FAILURE: CatBoost NO_HPO failed with: {e}")

print(f"\n[{'='*40}]")
print(f"STARTING: CatBoost HPO (100 trials)")
print(f"[{'='*40}]")

try:
    run_talent_method(
        tune=True,
        n_trials=100,  # Matching HPO_runner default
        **PARAMS
    )
    print("SUCCESS: CatBoost HPO")
except Exception as e:
    print(f"FAILURE: CatBoost HPO failed with: {e}")

EOF