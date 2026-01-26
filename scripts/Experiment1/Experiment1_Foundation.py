#!/usr/bin/env python3
"""
Foundation Model Orchestrator: SEQUENTIAL REPAIR RUN

This script re-runs the 3 specific foundation model tasks SEQUENTIALLY.
It guarantees that only one model runs at a time, ensuring full GPU/Memory availability.

TASKS:
    1. 0012.loan_default / mitra / PD / NO_HPO (No limit)
    2. 0013.home_credit  / mitra / PD / NO_HPO (No limit)
    3. 0013.home_credit  / tabicl/ PD / NO_HPO (Limit 50k)
"""

import sys
import os
import gc
import torch
import logging
from pathlib import Path
from time import time

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from scripts.Experiment1.Experiment1 import run_single_method

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# TASK LIST
# =============================================================================
TASKS = [
    # (dataset, method, task_type, hpo_mode, row_limit)
    ('0012.loan_default', 'mitra',  'pd', 'NO_HPO', None),
    ('0013.home_credit',  'mitra',  'pd', 'NO_HPO', None),
    ('0013.home_credit',  'tabicl', 'pd', 'NO_HPO', 50000),
]

def cleanup_gpu():
    """Aggressive GPU cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("  -> GPU Memory Cleared")

def main():
    print(f"\n{'='*70}")
    print(f"EXPERIMENT 1 - SEQUENTIAL REPAIR RUN")
    print(f"Tasks to run: {len(TASKS)}")
    print(f"{'='*70}\n")

    # Load base config
    base_config = load_config("Experiment1")

    for i, (dataset, method, task_type, hpo_mode, row_limit) in enumerate(TASKS):
        print(f"\n[{i+1}/{len(TASKS)}] STARTING: {dataset} / {method}")
        print("-" * 50)
        
        # 1. Cleanup before start
        cleanup_gpu()

        # 2. Prepare Config
        # Deep copy to avoid pollution
        config = base_config.copy()
        
        # Apply row limit if needed
        if row_limit is not None:
            if 'split' not in config:
                config['split'] = {}
            config['split']['row_limit'] = row_limit
            print(f"  -> Applied Row Limit: {row_limit}")

        # 3. Run Method
        try:
            start_time = time()
            run_single_method(
                dataset=dataset,
                method=method,
                task_type=task_type,
                hpo_mode=hpo_mode,
                config=config,
                experiment_name='experiment1',
                verbose=True
            )
            duration = time() - start_time
            print(f"  -> SUCCESS ({duration:.1f}s)")
            
        except Exception as e:
            print(f"  -> FAILED: {e}")
            # We continue to the next task even if one fails
        
        # 4. Final Cleanup
        cleanup_gpu()
        print("-" * 50)

    print(f"\n{'='*70}")
    print("ALL TASKS COMPLETED")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()