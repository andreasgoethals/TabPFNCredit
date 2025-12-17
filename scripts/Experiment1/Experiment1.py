#!/usr/bin/env python3
"""
Experiment 1: Method-level parallelization with file locking
"""

import sys
import argparse
import json
import pickle
import fcntl  # Unix file locking
import time
from pathlib import Path
from datetime import datetime
import logging

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from src.utils.storage_handler import StorageHandler
from src.methods.method_runner import run_talent_method  
from src.methods.method_config import DEEP_METHODS, CLASSICAL_METHODS


def generate_task_list(experiment_name="experiment1"):
    """Generate all (dataset, method, hpo_mode) combinations."""
    
    config = load_config()
    
    # Get enabled datasets and methods
    pd_datasets = list(config['datasets']['pd'].keys())
    lgd_datasets = list(config['datasets']['lgd'].keys())
    pd_methods = list(config['methods']['pd'].keys())
    lgd_methods = list(config['methods']['lgd'].keys())
    
    all_tasks = []
    task_idx = 0
    
    # PD tasks
    for dataset in pd_datasets:
        for method in pd_methods:
            for hpo_mode in ['NO_HPO', 'HPO']:
                task = {
                    'task_idx': task_idx,
                    'dataset': dataset,
                    'method': method,
                    'task': 'pd',
                    'hpo_mode': hpo_mode,
                    'is_gpu_method': method in DEEP_METHODS,
                }
                all_tasks.append(task)
                task_idx += 1
    
    # LGD tasks
    for dataset in lgd_datasets:
        for method in lgd_methods:
            for hpo_mode in ['NO_HPO', 'HPO']:
                task = {
                    'task_idx': task_idx,
                    'dataset': dataset,
                    'method': method,
                    'task': 'lgd',
                    'hpo_mode': hpo_mode,
                    'is_gpu_method': method in DEEP_METHODS,
                }
                all_tasks.append(task)
                task_idx += 1
    
    # Separate by type
    gpu_tasks = [t for t in all_tasks if t['is_gpu_method']]
    cpu_tasks = [t for t in all_tasks if not t['is_gpu_method']]
    
    return all_tasks, {'gpu': gpu_tasks, 'cpu': cpu_tasks, 'all': all_tasks}


def save_task_lists(experiment_name, tasks_by_type):
    """Save task lists as JSON for SLURM to reference."""
    
    storage = StorageHandler(experiment_name)
    experiment_path = storage.get_experiment_path()
    
    # Save all task lists
    for task_type, tasks in tasks_by_type.items():
        task_file = experiment_path / f"task_list_{task_type}.json"
        with open(task_file, 'w') as f:
            json.dump(tasks, f, indent=2)
    
    # Summary
    summary = {
        'total_tasks': len(tasks_by_type['all']),
        'gpu_tasks': len(tasks_by_type['gpu']),
        'cpu_tasks': len(tasks_by_type['cpu']),
        'generated_at': datetime.now().isoformat(),
    }
    
    summary_file = experiment_path / "task_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"TASK LISTS GENERATED")
    print(f"{'='*70}")
    print(f"Total tasks:  {summary['total_tasks']}")
    print(f"GPU tasks:    {summary['gpu_tasks']}")
    print(f"CPU tasks:    {summary['cpu_tasks']}")
    print(f"Saved to:     {experiment_path}")
    print(f"{'='*70}\n")


def run_single_task(task, experiment_name, config, verbose=False):
    """
    Run ONE method on ONE dataset with ONE HPO mode.
    Uses file locking for safe concurrent writes.
    """
    
    # Extract task parameters FIRST
    task_idx = task['task_idx']
    dataset = task['dataset']
    method = task['method']
    task_type = task['task']
    hpo_mode = task['hpo_mode']
    
    # Print banner at start
    print(f"\n{'='*70}")
    print(f"TASK {task_idx}: {dataset} | {method} | {hpo_mode}")
    print(f"{'='*70}")
    
    # Result file for this dataset
    result_file = experiment_path / task_type / f"{dataset}.pkl"
    
    # ==========================================
    # IMPROVED LOGGING: One file per dataset
    # ==========================================
    log_dir = experiment_path / "logs" / task_type
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{dataset}.log"  # All methods for this dataset
    
    # Setup logger with method/hpo identification
    logger_name = f"{experiment_name}.{dataset}.{method}.{hpo_mode}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.handlers.clear()  # Remove any existing handlers
    
    # Format includes method and hpo_mode for identification
    log_format = f'%(asctime)s - [{method}/{hpo_mode}] - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # File handler (append mode - multiple methods write to same file)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (if verbose)
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"Starting task")
    
    # ==========================================
    # Check if already completed (with locking)
    # ==========================================
    if result_file.exists():
        try:
            with open(result_file, 'rb') as f:
                # Shared lock for reading (multiple readers OK)
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    existing_results = pickle.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            if hpo_mode in existing_results and method in existing_results[hpo_mode]:
                logger.info(f"Already completed, skipping")
                print(f"[SKIP] {dataset}/{method}/{hpo_mode} - already done")
                return
        except Exception as e:
            logger.warning(f"Could not check existing results: {e}")
            # Continue with task anyway
    
    # Build parameters for method_runner
    experiment_params = {
        'task': task_type,
        'dataset': dataset,
        'test_size': config['split']['test_size'],
        'val_size': config['split']['val_size'],
        'cv_splits': config['split']['cv_splits'],
        'seed': config['split']['seed'],
        'row_limit': config['split'].get('row_limit', None),
        'sampling': config['split'].get('sampling', None),
        'method': method,
        'max_epoch': config['training']['max_epochs'],
        'batch_size': config['training']['batch_size'],
        'early_stopping': config['training']['early_stopping'],
        'early_stopping_patience': config['training']['early_stopping_patience'],
        'n_trials': config['tuning']['n_trials'],
        'tune': tune,
        'config_base_dir': experiment_path,
        'verbose': verbose,
    }
    
    try:
        # ==========================================
        # TRAINING (no file lock, fully parallel)
        # ==========================================
        logger.info("Training started")
        method_results = run_talent_method(**experiment_params)
        logger.info("Training completed")
        
        # ==========================================
        # SAVE RESULTS (with file locking)
        # ==========================================
        logger.info("Saving results (acquiring file lock)")
        
        max_retries = 10
        retry_delay = 0.5  # seconds
        
        for attempt in range(max_retries):
            try:
                # Ensure directory exists
                result_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Determine file mode
                if result_file.exists():
                    mode = 'r+b'  # Read and write existing
                else:
                    mode = 'w+b'  # Create new
                
                with open(result_file, mode) as f:
                    # ========================================
                    # EXCLUSIVE LOCK - blocks other processes
                    # ========================================
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    
                    try:
                        # Load existing results (or create new structure)
                        if result_file.exists() and f.seek(0, 2) > 0:  # Check file size
                            f.seek(0)
                            results = pickle.load(f)
                        else:
                            results = {'NO_HPO': {}, 'HPO': {}}
                        
                        # Double-check not already written (race condition safety)
                        if hpo_mode in results and method in results[hpo_mode]:
                            logger.info("Results already written by another process, skipping save")
                            return
                        
                        # Add our results
                        results[hpo_mode][method] = method_results
                        
                        # Write back to file
                        f.seek(0)
                        f.truncate()
                        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
                        f.flush()
                        
                        logger.info("Results saved successfully")
                        
                    finally:
                        # UNLOCK
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
                # Success - break retry loop
                break
                
            except (IOError, OSError) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"File lock failed (attempt {attempt+1}/{max_retries}), "
                                 f"waiting {wait_time:.1f}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to acquire file lock after {max_retries} attempts")
                    raise RuntimeError(
                        f"Could not save results after {max_retries} attempts. "
                        f"File may be corrupted: {result_file}"
                    ) from e
        
        logger.info(f"Task completed successfully")
        print(f"[DONE] {dataset}/{method}/{hpo_mode}")
        
    except Exception as e:
        logger.error(f"Task failed with error: {e}", exc_info=True)
        print(f"[FAIL] {dataset}/{method}/{hpo_mode}: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Experiment 1 with file locking")
    parser.add_argument('--task_idx', type=int, default=None,
                        help='Task index for SLURM array')
    parser.add_argument('--generate_tasks', action='store_true',
                        help='Generate task lists only')
    parser.add_argument('--verbose', action='store_true',
                        help='Detailed logging')
    parser.add_argument('--experiment', type=str, default='experiment1',
                        help='Experiment name')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    # Generate task lists
    all_tasks, tasks_by_type = generate_task_list(args.experiment)
    save_task_lists(args.experiment, tasks_by_type)
    
    if args.generate_tasks:
        print("Task lists generated. Submit SLURM jobs to run.")
        return
    
    # Determine which tasks to run
    if args.task_idx is not None:
        # SLURM array mode: run ONE task
        if args.task_idx < 0 or args.task_idx >= len(all_tasks):
            print(f"ERROR: task_idx {args.task_idx} out of range")
            sys.exit(1)
        
        tasks_to_run = [all_tasks[args.task_idx]]
        print(f"\n[SLURM] Running task {args.task_idx}/{len(all_tasks)}")
    else:
        # Sequential mode: run all tasks (for testing)
        tasks_to_run = all_tasks
        print(f"\n[SEQUENTIAL] Running all {len(all_tasks)} tasks")
    
    # Execute tasks
    for task in tasks_to_run:
        run_single_task(task, args.experiment, config, args.verbose)


if __name__ == "__main__":
    main()