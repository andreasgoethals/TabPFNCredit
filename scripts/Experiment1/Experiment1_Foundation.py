#!/usr/bin/env python3
"""
Foundation Model Orchestrator: REPAIR RUN for specific failed tasks

This script re-runs ONLY the 3 specific foundation model tasks requested.
It applies a row limit of 50,000 to the Home Credit / TabICL task.

REPAIR RUN: 3 specific tasks
    - 0012.loan_default / mitra / PD / NO_HPO (No limit)
    - 0013.home_credit / mitra / PD / NO_HPO (No limit)
    - 0013.home_credit / tabicl / PD / NO_HPO (Row Limit: 50,000)

Usage:
    # Array-based execution (for SLURM array jobs)
    python Experiment1_Foundation.py --array_id 0
"""

import sys
import os
import argparse
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from scripts.Experiment1.Experiment1 import run_single_method

# =============================================================================
# FOUNDATION MODELS - Methods requiring high-memory GPUs
# =============================================================================
FOUNDATION_METHODS = {
    'tabpfn',        # Original TabPFN (v1)
    'tabpfn_v2',     # TabPFN v2 (standard)
    'tabpfn_real',   # TabPFN v2.5 "Real" mode
    'mitra',         # Mitra foundation model
    'tabicl',        # TabICL (In-Context Learning)
}

# =============================================================================
# HARDCODED REPAIR RUN LIST - 3 Specific Tasks
# =============================================================================
# Format: (dataset, method, task_type, hpo_mode, row_limit)
# row_limit: int or None
# =============================================================================

FAILED_TASKS = [
    # Task 0: Loan Default / Mitra (No Limit)
    ('0012.loan_default', 'mitra',  'pd', 'NO_HPO', None),
    
    # Task 1: Home Credit / Mitra (No Limit)
    ('0013.home_credit',  'mitra',  'pd', 'NO_HPO', None),
    
    # Task 2: Home Credit / TabICL (Limit 50k)
    ('0013.home_credit',  'tabicl', 'pd', 'NO_HPO', 50000),
]


def build_foundation_task_list(config):
    """
    Return the hardcoded list of tasks for this repair run.
    Returns:
        List of tuples: (dataset, method, task_type, hpo_mode, row_limit)
    """
    return FAILED_TASKS.copy()


def main():
    parser = argparse.ArgumentParser(
        description='Run foundation models on high-memory GPUs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--array_id', type=int,
                        help='SLURM array task ID (for array jobs)')
    parser.add_argument('--dataset', type=str, help='Specific dataset')
    parser.add_argument('--method', type=str, help='Specific method')
    parser.add_argument('--task_type', type=str, help='Task type')
    parser.add_argument('--hpo_mode', type=str, help='HPO mode')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable detailed logging')
    parser.add_argument('--experiment', type=str, default='experiment1',
                        help='Experiment name (default: experiment1)')
    parser.add_argument('--list-tasks', action='store_true',
                        help='List all tasks and exit')

    args = parser.parse_args()

    # Load config
    config = load_config("Experiment1")

    print(f"\n{'='*70}")
    print(f"FOUNDATION MODEL REPAIR RUN - 3 Specific Tasks")
    print(f"{'='*70}")
    print(f"Target methods: {', '.join(sorted(FOUNDATION_METHODS))}")

    # Build task list
    foundation_tasks = build_foundation_task_list(config)

    # List tasks mode
    if args.list_tasks:
        print(f"\nTotal foundation tasks: {len(foundation_tasks)}")
        print(f"\nTask list:")
        for i, (ds, method, tt, hpo, limit) in enumerate(foundation_tasks):
            limit_str = f"Limit: {limit}" if limit else "No Limit"
            print(f"  [{i:3d}] {ds} / {method} / {tt} / {hpo} ({limit_str})")
        print(f"\n{'='*70}")
        return

    # Check if direct parameters provided (Manual Override)
    if args.dataset and args.method and args.task_type and args.hpo_mode:
        dataset = args.dataset
        method = args.method
        task_type = args.task_type
        hpo_mode = args.hpo_mode
        row_limit = None # Default to None for manual runs unless logic added
        
        # Check if this manual run matches our special case
        if dataset == '0013.home_credit' and method == 'tabicl':
             row_limit = 50000
             print("Applying implicit row limit of 50,000 for Home Credit/TabICL")

        print(f"Mode:         Direct task specification")
        print(f"Dataset:      {dataset}")
        print(f"Method:       {method}")
        print(f"Task type:    {task_type}")
        print(f"HPO mode:     {hpo_mode}")
        print(f"Row Limit:    {row_limit}")
        print(f"{'='*70}\n")

    elif args.array_id is not None:
        # Array job - use task list
        print(f"Mode:         Array-based execution")
        print(f"Total tasks:  {len(foundation_tasks)}")
        print(f"Array ID:     {args.array_id}")
        print(f"{'='*70}\n")

        # Validate array ID
        if args.array_id < 0 or args.array_id >= len(foundation_tasks):
            print(f"ERROR: Array ID {args.array_id} out of range [0, {len(foundation_tasks)-1}]")
            sys.exit(1)

        # Get task for this array ID
        dataset, method, task_type, hpo_mode, row_limit = foundation_tasks[args.array_id]
        print(f"Running task {args.array_id}: {dataset}/{method}/{task_type}/{hpo_mode}")
        if row_limit:
            print(f"Apply Config Row Limit: {row_limit}")
        print("\n")

    else:
        print("\nERROR: Must provide either --array_id or (--dataset, --method, --task_type, --hpo_mode)")
        sys.exit(1)

    # Apply row limit to configuration if specified
    if row_limit is not None:
        if 'split' not in config:
            config['split'] = {}
        config['split']['row_limit'] = row_limit
        print(f"Configuration updated: split.row_limit = {row_limit}")

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