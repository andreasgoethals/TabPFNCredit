#!/bin/bash
#SBATCH --job-name=retry_failures
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=69:00:00
#SBATCH --partition=gpu_p100
#SBATCH --array=0-8

# --- UPDATED LOG PATHS (Absolute path to project root) ---
#SBATCH --output=/data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/slurm/retry_%A_%a.out
#SBATCH --error=/data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/slurm/retry_%A_%a.err

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
export PYTHONUNBUFFERED=1

# Setup conda
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate TabPFNCredit

# USE CONDA'S C++ LIBRARIES
export LD_LIBRARY_PATH="${VSC_DATA}/miniconda3/envs/TabPFNCredit/lib:${LD_LIBRARY_PATH}"

# Navigate to project root
cd $VSC_DATA/TabPFNCredit

echo "=========================================="
echo "RETRYING FAILURES WITH ROW_LIMIT=100k"
echo "Job ID:   $SLURM_JOB_ID"
echo "Array ID: $SLURM_ARRAY_TASK_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Log Path: /data/leuven/383/vsc38338/TabPFNCredit/results/experiment1/logs/slurm/"
echo "=========================================="

# ---------------------------------------------------------
# PYTHON ORCHESTRATOR
# ---------------------------------------------------------
python -u - <<EOF
import sys
import os
import logging

# Ensure project root is in path
sys.path.insert(0, os.getcwd())

from src.methods.method_runner import run_talent_method

# ---------------------------------------------------------
# DEFINE FAILURES TO RETRY
# ---------------------------------------------------------
# Map array indices to specific configurations
JOBS = [
    # 0: amformer / HPO / 0013.home_credit
    {"method": "amformer", "dataset": "0013.home_credit", "task": "pd", "tune": True},
    
    # 1: catboost / HPO / 0005.base_modelisation
    {"method": "catboost", "dataset": "0005.base_modelisation", "task": "lgd", "tune": True},
    
    # 2: catboost / NO_HPO / 0005.base_modelisation
    {"method": "catboost", "dataset": "0005.base_modelisation", "task": "lgd", "tune": False},
    
    # 3: tabicl / HPO / 0007.hackerearth
    {"method": "tabicl", "dataset": "0007.hackerearth", "task": "pd", "tune": True},
    
    # 4: tabicl / HPO / 0013.home_credit
    {"method": "tabicl", "dataset": "0013.home_credit", "task": "pd", "tune": True},
    
    # 5: tabicl / NO_HPO / 0007.hackerearth
    {"method": "tabicl", "dataset": "0007.hackerearth", "task": "pd", "tune": False},
    
    # 6: tabicl / NO_HPO / 0013.home_credit
    {"method": "tabicl", "dataset": "0013.home_credit", "task": "pd", "tune": False},
    
    # 7: tangos / HPO / 0007.hackerearth
    {"method": "tangos", "dataset": "0007.hackerearth", "task": "pd", "tune": True},
    
    # 8: tangos / HPO / 0013.home_credit
    {"method": "tangos", "dataset": "0013.home_credit", "task": "pd", "tune": True},
]

# Get current job config
task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

# Safety check
if task_id >= len(JOBS):
    print(f"Error: Task ID {task_id} is out of range (Max: {len(JOBS)-1})")
    sys.exit(1)

config = JOBS[task_id]

print(f"\n[CONFIG] Method:  {config['method']}")
print(f"[CONFIG] Dataset: {config['dataset']}")
print(f"[CONFIG] Task:    {config['task']}")
print(f"[CONFIG] HPO:     {'Enabled' if config['tune'] else 'Disabled'}")
print(f"[CONFIG] Limit:   100,000 rows")

# ---------------------------------------------------------
# APPLY CATBOOST FIX (FORCE CPU)
# ---------------------------------------------------------
if config['method'] == 'catboost':
    print("\n[FIX] Detected CatBoost: Hiding GPU to prevent driver errors.")
    os.environ['CUDA_VISIBLE_DEVICES'] = "" # Hide GPU from this process
    # Verify
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print(f"[FIX] CUDA_VISIBLE_DEVICES set to: '{os.environ['CUDA_VISIBLE_DEVICES']}'")

# ---------------------------------------------------------
# RUN METHOD
# ---------------------------------------------------------
try:
    run_talent_method(
        task=config['task'],
        dataset=config['dataset'],
        method=config['method'],
        tune=config['tune'],
        
        # Standard Experiment Params
        test_size=0.2,
        val_size=0.2,
        cv_splits=5,
        seed=42,
        
        # RESOURCE FIX: Limit rows to 100k
        row_limit=100000,
        
        # Logging
        verbose=True,
        clean_temp_dir=True,
        
        # HPO/Training Params (Standard)
        max_epoch=200,
        batch_size=1024,
        n_trials=100,
        early_stopping=True,
        early_stopping_patience=10
    )
    print("\n[SUCCESS] Method completed successfully.")
except Exception as e:
    print(f"\n[FAILURE] Method failed with error: {e}")
    sys.exit(1)

EOF