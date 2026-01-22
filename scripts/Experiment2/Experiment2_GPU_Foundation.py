#!/usr/bin/env python3
"""
Foundation GPU Orchestrator for Experiment 2: Learning Curve Analysis

This script handles FOUNDATION models that require high-memory GPUs on the
wICE cluster with gpu_h100 partition.

Foundation methods: tabpfn, tabpfn_v2, tabpfn_real, mitra, tabicl, tabptm

These models use in-context learning and need to load entire datasets into
GPU memory, requiring 64GB+ VRAM for larger datasets.

This script:
1. Reads enabled GPU methods and datasets from config
2. Filters to ONLY foundation GPU methods
3. Builds list of tasks (method × dataset combinations)
4. Picks one task based on SLURM_ARRAY_TASK_ID or direct parameters
5. Executes learning curve analysis using Experiment2.py
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
from scripts.Experiment2.Experiment2 import run_learning_curve


def build_foundation_gpu_task_list(config):
    """
    Build list of FOUNDATION GPU tasks for learning curve analysis.

    Filters to GPU methods that ARE in FOUNDATION_METHODS.

    Each task = (dataset, method, task_type)
    No HPO mode variation - all use NO_HPO (default parameters).

    Returns:
        List of tuples: (dataset, method, task_type)
    """
    tasks = []

    # Foundation GPU = GPU methods that are also foundation methods
    foundation_gpu_methods = GPU_METHODS & FOUNDATION_METHODS

    # PD tasks
    pd_datasets = list(config['datasets']['pd'].keys())
    pd_methods = [m for m in config['methods']['pd'].keys() if m in foundation_gpu_methods]

    for dataset in pd_datasets:
        for method in pd_methods:
            tasks.append((dataset, method, 'pd'))

    # LGD tasks
    lgd_datasets = list(config['datasets']['lgd'].keys())
    lgd_methods = [m for m in config['methods']['lgd'].keys() if m in foundation_gpu_methods]

    for dataset in lgd_datasets:
        for method in lgd_methods:
            tasks.append((dataset, method, 'lgd'))

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description='Run FOUNDATION GPU methods for Learning Curve Analysis (Experiment 2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Foundation Methods (require high-memory GPUs):
    tabpfn, tabpfn_v2, tabpfn_real, mitra, tabicl, tabptm

These models use in-context learning and need 64GB+ GPU memory.
        """
    )
    parser.add_argument('--array_id', type=int,
                       help='SLURM array task ID (for array jobs)')
    parser.add_argument('--dataset', type=str,
                       help='Specific dataset to run')
    parser.add_argument('--method', type=str,
                       help='Specific method to run')
    parser.add_argument('--task_type', type=str,
                       help='Task type (pd or lgd)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable detailed logging')
    parser.add_argument('--experiment', type=str, default='experiment2',
                       help='Experiment name')
    parser.add_argument('--list-tasks', action='store_true',
                       help='List all tasks and exit')

    args = parser.parse_args()

    # Load config
    config = load_config("Experiment2")

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 2: FOUNDATION GPU ORCHESTRATOR - LEARNING CURVE ANALYSIS")
    print(f"{'='*70}")
    print(f"Cluster: wICE | Partition: gpu_h100 | Memory: 64G")
    print(f"Target methods: {', '.join(sorted(FOUNDATION_METHODS))}")

    # Build task list
    gpu_tasks = build_foundation_gpu_task_list(config)

    # List tasks mode
    if args.list_tasks:
        print(f"\nTotal foundation GPU tasks: {len(gpu_tasks)}")
        print(f"\nTask list:")
        for i, (ds, method, tt) in enumerate(gpu_tasks):
            print(f"  [{i:3d}] {ds} / {method} / {tt}")
        print(f"\n{'='*70}")
        return

    # Check if direct parameters provided (for retry scripts)
    if args.dataset and args.method and args.task_type:
        # Validate method is a foundation method
        if args.method not in FOUNDATION_METHODS:
            print(f"\nERROR: Method '{args.method}' is not a foundation method.")
            print(f"Valid foundation methods: {', '.join(sorted(FOUNDATION_METHODS))}")
            print(f"{'='*70}\n")
            sys.exit(1)

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
        print("\nERROR: Must provide either --array_id or (--dataset, --method, --task_type)")
        print("       Use --list-tasks to see all available tasks.")
        print(f"{'='*70}\n")
        sys.exit(1)

    # Execute learning curve analysis
    run_learning_curve(
        dataset=dataset,
        method=method,
        task_type=task_type,
        config=config,
        experiment_name=args.experiment,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
