#!/usr/bin/env python3
"""
CPU Orchestrator for Experiment 2: Learning Curve Analysis

This script:
1. Reads enabled CPU methods and datasets from config
2. Builds list of all CPU tasks (method × dataset combinations)
3. Picks one task based on SLURM_ARRAY_TASK_ID or direct parameters
4. Executes learning curve analysis using Experiment2.py

Key difference from Experiment1:
- Each SLURM task = ONE method + ONE dataset
- Loop over row_limits happens INSIDE Python (not as separate SLURM jobs)
- No HPO mode variation (always NO_HPO)
"""

import sys
import os
import argparse
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from src.methods.method_config import CPU_METHODS
from scripts.Experiment2.Experiment2 import run_learning_curve


def build_cpu_task_list(config):
    """
    Build list of all CPU tasks for learning curve analysis.

    Each task = (dataset, method, task_type)
    No HPO mode variation - all use NO_HPO (default parameters).

    Returns:
        List of tuples: (dataset, method, task_type)
    """
    tasks = []

    # PD tasks
    pd_datasets = list(config['datasets']['pd'].keys())
    pd_methods = [m for m in config['methods']['pd'].keys() if m in CPU_METHODS]

    for dataset in pd_datasets:
        for method in pd_methods:
            tasks.append((dataset, method, 'pd'))

    # LGD tasks
    lgd_datasets = list(config['datasets']['lgd'].keys())
    lgd_methods = [m for m in config['methods']['lgd'].keys() if m in CPU_METHODS]

    for dataset in lgd_datasets:
        for method in lgd_methods:
            tasks.append((dataset, method, 'lgd'))

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description='Run CPU methods for Learning Curve Analysis (Experiment 2)'
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

    args = parser.parse_args()

    # Load config
    config = load_config("Experiment2")

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 2: CPU ORCHESTRATOR - LEARNING CURVE ANALYSIS")
    print(f"{'='*70}")

    # Check if direct parameters provided (for retry scripts)
    if args.dataset and args.method and args.task_type:
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
        cpu_tasks = build_cpu_task_list(config)

        print(f"Mode:         Array-based execution")
        print(f"Total tasks:  {len(cpu_tasks)}")
        print(f"Array ID:     {args.array_id}")
        print(f"{'='*70}\n")

        # Validate array ID
        if args.array_id < 0 or args.array_id >= len(cpu_tasks):
            print(f"ERROR: Array ID {args.array_id} out of range [0, {len(cpu_tasks)-1}]")
            sys.exit(1)

        # Get task for this array ID
        dataset, method, task_type = cpu_tasks[args.array_id]
        print(f"Running task {args.array_id}: {dataset}/{method}/{task_type}\n")

    else:
        print("ERROR: Must provide either --array_id or (--dataset, --method, --task_type)")
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
