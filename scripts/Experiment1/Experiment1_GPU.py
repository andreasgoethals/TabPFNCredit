#!/usr/bin/env python3
"""
GPU Orchestrator: Manages GPU method execution on GPU nodes

This script:
1. Reads enabled GPU methods and datasets from config
2. Builds list of all GPU tasks
3. Picks one task based on SLURM_ARRAY_TASK_ID or direct parameters
4. Executes task using shared logic from Experiment1.py

Supports both array-based execution (backwards compatible) and direct task specification (for retry scripts).
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

# GPU methods (all non-CPU methods)
GPU_METHODS = {
    'xgboost', 'catboost', 'lightgbm',
    'mlp', 'resnet', 'tabnet',
    'node', 'autoint', 'danets', 'dcn2',
    'ftt', 'saint', 'tabtransformer',
    'tabpfn', 'tabpfn_v2', 'tabpfn_real',
    'tabicl', 'tabptm', 'trompt',
    'modernNCA', 'realmlp', 'bishop',
    'tabr', 'grownet', 'snn', 'tabcaps',
    'tangos', 'ptarl', 'switchtab', 'dnnr',
    'hyperfast', 'protogate', 'mlp_plr',
    'excelformer', 'grande', 'amformer',
    'tabm', 't2gformer', 'tabautopnpnet',
    'limix', 'mitra',
}


def build_gpu_task_list(config):
    """
    Build list of all GPU tasks.
    
    Returns:
        List of tuples: (dataset, method, task_type, hpo_mode)
    """
    tasks = []
    
    # PD tasks
    pd_datasets = list(config['datasets']['pd'].keys())
    pd_methods = [m for m in config['methods']['pd'].keys() if m in GPU_METHODS]
    
    for dataset in pd_datasets:
        for method in pd_methods:
            tasks.append((dataset, method, 'pd', 'NO_HPO'))
            tasks.append((dataset, method, 'pd', 'HPO'))
    
    # LGD tasks
    lgd_datasets = list(config['datasets']['lgd'].keys())
    lgd_methods = [m for m in config['methods']['lgd'].keys() if m in GPU_METHODS]
    
    for dataset in lgd_datasets:
        for method in lgd_methods:
            tasks.append((dataset, method, 'lgd', 'NO_HPO'))
            tasks.append((dataset, method, 'lgd', 'HPO'))
    
    return tasks


def main():
    parser = argparse.ArgumentParser(description='Run GPU methods')
    parser.add_argument('--array_id', type=int,
                       help='SLURM array task ID (for array jobs)')
    parser.add_argument('--dataset', type=str,
                       help='Specific dataset to run')
    parser.add_argument('--method', type=str,
                       help='Specific method to run')
    parser.add_argument('--task_type', type=str,
                       help='Task type (pd or lgd)')
    parser.add_argument('--hpo_mode', type=str,
                       help='HPO mode (NO_HPO or HPO)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable detailed logging')
    parser.add_argument('--experiment', type=str, default='experiment1',
                       help='Experiment name')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    print(f"\n{'='*70}")
    print(f"GPU ORCHESTRATOR")
    print(f"{'='*70}")
    
    # Check if direct parameters provided (for retry scripts)
    if args.dataset and args.method and args.task_type and args.hpo_mode:
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
        # Array job - use task list (backwards compatible)
        gpu_tasks = build_gpu_task_list(config)
        
        print(f"Mode:         Array-based execution")
        print(f"Total tasks:  {len(gpu_tasks)}")
        print(f"Array ID:     {args.array_id}")
        print(f"{'='*70}\n")
        
        # Validate array ID
        if args.array_id < 0 or args.array_id >= len(gpu_tasks):
            print(f"ERROR: Array ID {args.array_id} out of range [0, {len(gpu_tasks)-1}]")
            sys.exit(1)
        
        # Get task for this array ID
        dataset, method, task_type, hpo_mode = gpu_tasks[args.array_id]
        print(f"Running task {args.array_id}: {dataset}/{method}/{task_type}/{hpo_mode}\n")
        
    else:
        print("ERROR: Must provide either --array_id or (--dataset, --method, --task_type, --hpo_mode)")
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