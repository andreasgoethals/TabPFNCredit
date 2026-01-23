#!/usr/bin/env python3
"""
Standard GPU Orchestrator for Experiment 3: Class Imbalance Analysis

This script handles STANDARD GPU methods (non-foundation models) that run on
the genius cluster with gpu_p100 partition.

Standard GPU methods include: xgboost, catboost, lightgbm, mlp, resnet, ftt,
tabnet, node, etc. (everything in GPU_METHODS except FOUNDATION_METHODS)

This script:
1. Reads enabled GPU methods and datasets from config
2. Filters to ONLY standard (non-foundation) GPU methods
3. Builds list of tasks (method x dataset combinations) - PD only
4. Picks one task based on SLURM_ARRAY_TASK_ID or direct parameters
5. Executes class imbalance analysis using Experiment3.py
"""

import sys
import os
import argparse
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from src.methods.method_config import GPU_METHODS, FOUNDATION_METHODS
from scripts.Experiment3.Experiment3 import run_imbalance_analysis


def build_standard_gpu_task_list(config):
    """
    Build list of STANDARD GPU tasks for class imbalance analysis.

    Filters to GPU methods that are NOT in FOUNDATION_METHODS.
    Only PD (classification) tasks - LGD not applicable for imbalance analysis.

    Each task = (dataset, method, task_type)
    No HPO mode variation - all use NO_HPO (default parameters).

    Returns:
        List of tuples: (dataset, method, task_type)
    """
    tasks = []

    # Standard GPU = GPU methods minus Foundation methods
    standard_gpu_methods = GPU_METHODS - FOUNDATION_METHODS

    # PD tasks only (class imbalance analysis doesn't apply to regression)
    pd_datasets = list(config['datasets']['pd'].keys())
    pd_methods = [m for m in config['methods']['pd'].keys() if m in standard_gpu_methods]

    for dataset in pd_datasets:
        for method in pd_methods:
            tasks.append((dataset, method, 'pd'))

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description='Run STANDARD GPU methods for Class Imbalance Analysis (Experiment 3)'
    )
    parser.add_argument('--array_id', type=int,
                       help='SLURM array task ID (for array jobs)')
    parser.add_argument('--dataset', type=str,
                       help='Specific dataset to run')
    parser.add_argument('--method', type=str,
                       help='Specific method to run')
    parser.add_argument('--task_type', type=str, default='pd',
                       help='Task type (pd only for Experiment 3)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable detailed logging')
    parser.add_argument('--experiment', type=str, default='experiment3',
                       help='Experiment name')
    parser.add_argument('--list-tasks', action='store_true',
                       help='List all tasks and exit')

    args = parser.parse_args()

    # Load config
    config = load_config("Experiment3")

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 3: STANDARD GPU ORCHESTRATOR - CLASS IMBALANCE ANALYSIS")
    print(f"{'='*70}")
    print(f"Cluster: genius | Partition: gpu_p100 | Memory: 45G")

    # Build task list
    gpu_tasks = build_standard_gpu_task_list(config)

    # List tasks mode
    if args.list_tasks:
        print(f"\nTotal standard GPU tasks: {len(gpu_tasks)}")
        print(f"\nTask list:")
        for i, (ds, method, tt) in enumerate(gpu_tasks):
            print(f"  [{i:3d}] {ds} / {method} / {tt}")
        print(f"\n{'='*70}")
        return

    # Check if direct parameters provided (for retry scripts)
    if args.dataset and args.method:
        # Direct task specification
        dataset = args.dataset
        method = args.method
        task_type = args.task_type

        print(f"Mode:         Direct task specification")
        print(f"Dataset:      {dataset}")
        print(f"Method:       {method}")
        print(f"Task type:    {task_type}")
        print(f"{'='*70}\n")

    elif args.array_id is not None:
        # Array job - use task list
        print(f"Mode:         Array-based execution")
        print(f"Total tasks:  {len(gpu_tasks)}")
        print(f"Array ID:     {args.array_id}")
        print(f"{'='*70}\n")

        # Validate array ID
        if args.array_id < 0 or args.array_id >= len(gpu_tasks):
            print(f"ERROR: Array ID {args.array_id} out of range [0, {len(gpu_tasks)-1}]")
            sys.exit(1)

        # Get task for this array ID
        dataset, method, task_type = gpu_tasks[args.array_id]
        print(f"Running task {args.array_id}: {dataset}/{method}/{task_type}\n")

    else:
        print("\nERROR: Must provide either --array_id or (--dataset, --method)")
        print("       Use --list-tasks to see all available tasks.")
        print(f"{'='*70}\n")
        sys.exit(1)

    # Execute class imbalance analysis
    run_imbalance_analysis(
        dataset=dataset,
        method=method,
        task_type=task_type,
        config=config,
        experiment_name=args.experiment,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
