#!/usr/bin/env python3
"""
Foundation Model Orchestrator: REPAIR RUN for failed foundation model tasks

This script re-runs ONLY the specific foundation model tasks that failed due to
OOM errors during inference. The underlying issue (test/val set size limits)
has been fixed in method_runner.py with METHOD_TEST_VAL_LIMITS.

REPAIR RUN: 6 specific failed tasks (hardcoded)
    - 0007.hackerearth / tabpfn / PD / NO_HPO
    - 0007.hackerearth / tabpfn_v2 / PD / NO_HPO
    - 0007.hackerearth / tabpfn_real / PD / NO_HPO
    - 0012.loan_default / mitra / PD / NO_HPO
    - 0013.home_credit / mitra / PD / NO_HPO
    - 0013.home_credit / tabicl / PD / NO_HPO

Usage:
    # Array-based execution (for SLURM array jobs)
    python Experiment1_Foundation.py --array_id 0

    # Direct task specification (for retries)
    python Experiment1_Foundation.py --dataset 0007.hackerearth --method tabpfn --task_type pd --hpo_mode NO_HPO

Supports both array-based execution and direct task specification.
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
# These methods use in-context learning and need to load entire datasets
# into GPU memory, requiring 64GB+ VRAM for larger datasets.

FOUNDATION_METHODS = {
    'tabpfn',        # Original TabPFN (v1)
    'tabpfn_v2',     # TabPFN v2 (standard)
    'tabpfn_real',   # TabPFN v2.5 "Real" mode
    'mitra',         # Mitra foundation model
    'tabicl',        # TabICL (In-Context Learning)
}


# =============================================================================
# HARDCODED REPAIR RUN LIST - 6 Failed Tasks
# =============================================================================
# These specific tasks failed with OOM errors during inference.
# The fix: METHOD_TEST_VAL_LIMITS in method_config.py caps test/val sets.
#
# Format: (dataset, method, task_type, hpo_mode)
# =============================================================================

FAILED_TASKS = [
    ('0007.hackerearth', 'tabpfn',      'pd', 'NO_HPO'),
    ('0007.hackerearth', 'tabpfn_v2',   'pd', 'NO_HPO'),
    ('0007.hackerearth', 'tabpfn_real', 'pd', 'NO_HPO'),
    ('0012.loan_default', 'mitra',      'pd', 'NO_HPO'),
    ('0013.home_credit',  'mitra',      'pd', 'NO_HPO'),
    ('0013.home_credit',  'tabicl',     'pd', 'NO_HPO'),
]


def build_foundation_task_list(config):
    """
    Return the hardcoded list of failed tasks for this repair run.

    NOTE: This function has been modified for a REPAIR RUN.
    It returns only the 6 specific tasks that failed, NOT all foundation tasks.

    To restore full task generation, replace this with dynamic generation from config.

    Returns:
        List of tuples: (dataset, method, task_type, hpo_mode)
    """
    # REPAIR RUN: Return hardcoded failed tasks only
    return FAILED_TASKS.copy()


def main():
    parser = argparse.ArgumentParser(
        description='Run foundation models on high-memory GPUs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Array-based execution
    python Experiment1_Foundation.py --array_id 0

    # Direct task specification
    python Experiment1_Foundation.py --dataset 0001.gmsc --method tabpfn_v2 --task_type pd --hpo_mode NO_HPO

Foundation Methods:
    tabpfn, tabpfn_v2, tabpfn_real, mitra, tabicl
        """
    )
    parser.add_argument('--array_id', type=int,
                       help='SLURM array task ID (for array jobs)')
    parser.add_argument('--dataset', type=str,
                       help='Specific dataset to run')
    parser.add_argument('--method', type=str,
                       help='Specific method to run (must be a foundation method)')
    parser.add_argument('--task_type', type=str,
                       help='Task type (pd or lgd)')
    parser.add_argument('--hpo_mode', type=str,
                       help='HPO mode (NO_HPO or HPO)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable detailed logging')
    parser.add_argument('--experiment', type=str, default='experiment1',
                       help='Experiment name (default: experiment1)')
    parser.add_argument('--list-tasks', action='store_true',
                       help='List all tasks and exit (useful for debugging)')

    args = parser.parse_args()

    # Load config
    config = load_config("Experiment1")

    print(f"\n{'='*70}")
    print(f"FOUNDATION MODEL REPAIR RUN - 6 Failed Tasks")
    print(f"{'='*70}")
    print(f"Fix applied: METHOD_TEST_VAL_LIMITS caps test/val sets")
    print(f"Target methods: {', '.join(sorted(FOUNDATION_METHODS))}")

    # Build task list
    foundation_tasks = build_foundation_task_list(config)

    # List tasks mode
    if args.list_tasks:
        print(f"\nTotal foundation tasks: {len(foundation_tasks)}")
        print(f"\nTask list:")
        for i, (ds, method, tt, hpo) in enumerate(foundation_tasks):
            print(f"  [{i:3d}] {ds} / {method} / {tt} / {hpo}")
        print(f"\n{'='*70}")
        return

    # Check if direct parameters provided (for retry scripts)
    if args.dataset and args.method and args.task_type and args.hpo_mode:
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
        hpo_mode = args.hpo_mode

        print(f"Mode:         Direct task specification")
        print(f"Dataset:      {dataset}")
        print(f"Method:       {method}")
        print(f"Task type:    {task_type}")
        print(f"HPO mode:     {hpo_mode}")
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
        dataset, method, task_type, hpo_mode = foundation_tasks[args.array_id]
        print(f"Running task {args.array_id}: {dataset}/{method}/{task_type}/{hpo_mode}\n")

    else:
        print("\nERROR: Must provide either --array_id or (--dataset, --method, --task_type, --hpo_mode)")
        print("       Use --list-tasks to see all available tasks.")
        print(f"{'='*70}\n")
        sys.exit(1)

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
