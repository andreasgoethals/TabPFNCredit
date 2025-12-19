#!/usr/bin/env python3
"""
GPU Orchestrator for Experiment0: Method Validation

Tests ALL GPU methods on 2 datasets with NO_HPO only.
"""

import sys
import argparse
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config

# Import core executor
from Experiment0 import run_single_method

# Import GPU methods
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "Experiment1"))
from Experiment1_Setup import GPU_METHODS


def build_gpu_task_list(config):
    """Build list of all GPU tasks (NO_HPO only)."""
    tasks = []
    
    # Get enabled methods (only GPU methods)
    pd_methods = [m for m in config['methods']['pd'].keys() if m in GPU_METHODS]
    lgd_methods = [m for m in config['methods']['lgd'].keys() if m in GPU_METHODS]
    
    # Get enabled datasets
    pd_datasets = list(config['datasets']['pd'].keys())
    lgd_datasets = list(config['datasets']['lgd'].keys())
    
    # Build PD tasks (NO_HPO only)
    for dataset in pd_datasets:
        for method in pd_methods:
            tasks.append((dataset, method, 'pd'))
    
    # Build LGD tasks (NO_HPO only)
    for dataset in lgd_datasets:
        for method in lgd_methods:
            tasks.append((dataset, method, 'lgd'))
    
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Experiment0 GPU Orchestrator")
    parser.add_argument('--array_id', type=int, required=True)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config("Experiment0")
    
    # Build GPU task list
    gpu_tasks = build_gpu_task_list(config)
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT 0 - GPU ORCHESTRATOR")
    print(f"{'='*70}")
    print(f"Total GPU tasks: {len(gpu_tasks)}")
    print(f"Array ID: {args.array_id}")
    print(f"{'='*70}\n")
    
    # Validate array ID
    if args.array_id < 0 or args.array_id >= len(gpu_tasks):
        print(f"ERROR: Array ID {args.array_id} out of range [0, {len(gpu_tasks)-1}]")
        sys.exit(1)
    
    # Get task for this array ID
    dataset, method, task_type = gpu_tasks[args.array_id]
    
    print(f"Running: {dataset}/{method}/{task_type}\n")
    
    # Execute task
    run_single_method(
        dataset=dataset,
        method=method,
        task_type=task_type,
        config=config,
        experiment_name='experiment0',
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()