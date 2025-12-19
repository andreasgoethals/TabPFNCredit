#!/usr/bin/env python3
"""
GPU Orchestrator: Manages GPU method execution on GPU nodes

This script:
1. Reads enabled GPU methods and datasets from config
2. Builds list of all GPU tasks
3. Picks one task based on SLURM_ARRAY_TASK_ID
4. Executes that task using Experiment1.run_single_method()
"""

import sys
import argparse
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from src.methods.method_config import NO_HPO_METHODS

# Import core executor
from Experiment1 import run_single_method

# Import GPU methods from setup (defined at module level in Experiment1_Setup.py)
from Experiment1_Setup import GPU_METHODS


def build_gpu_task_list(config):
    """
    Build list of all GPU tasks from config.
    
    Returns list of tuples: (dataset, method, task_type, hpo_mode)
    """
    tasks = []
    
    # Get enabled methods (only GPU methods)
    pd_methods = [m for m in config['methods']['pd'].keys() if m in GPU_METHODS]
    lgd_methods = [m for m in config['methods']['lgd'].keys() if m in GPU_METHODS]
    
    # Get enabled datasets
    pd_datasets = list(config['datasets']['pd'].keys())
    lgd_datasets = list(config['datasets']['lgd'].keys())
    
    # Build PD tasks
    for dataset in pd_datasets:
        for method in pd_methods:
            # Check if method needs HPO
            if method in NO_HPO_METHODS:
                hpo_modes = ['NO_HPO']
            else:
                hpo_modes = ['NO_HPO', 'HPO']
            
            for hpo_mode in hpo_modes:
                tasks.append((dataset, method, 'pd', hpo_mode))
    
    # Build LGD tasks
    for dataset in lgd_datasets:
        for method in lgd_methods:
            if method in NO_HPO_METHODS:
                hpo_modes = ['NO_HPO']
            else:
                hpo_modes = ['NO_HPO', 'HPO']
            
            for hpo_mode in hpo_modes:
                tasks.append((dataset, method, 'lgd', hpo_mode))
    
    return tasks


def main():
    parser = argparse.ArgumentParser(description="GPU Orchestrator for Experiment 1")
    parser.add_argument('--array_id', type=int, required=True,
                        help='SLURM array task ID')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable detailed logging')
    parser.add_argument('--experiment', type=str, default='experiment1',
                        help='Experiment name')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config("Experiment1")
    
    # Build GPU task list
    gpu_tasks = build_gpu_task_list(config)
    
    print(f"\n{'='*70}")
    print(f"GPU ORCHESTRATOR")
    print(f"{'='*70}")
    print(f"Total GPU tasks: {len(gpu_tasks)}")
    print(f"Array ID: {args.array_id}")
    print(f"{'='*70}\n")
    
    # Validate array ID
    if args.array_id < 0 or args.array_id >= len(gpu_tasks):
        print(f"ERROR: Array ID {args.array_id} out of range [0, {len(gpu_tasks)-1}]")
        sys.exit(1)
    
    # Get task for this array ID
    dataset, method, task_type, hpo_mode = gpu_tasks[args.array_id]
    
    print(f"Running task {args.array_id}: {dataset}/{method}/{task_type}/{hpo_mode}\n")
    
    # Execute task
    run_single_method(
        dataset=dataset,
        method=method,
        task_type=task_type,
        hpo_mode=hpo_mode,
        config=config,
        experiment_name=args.experiment,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()