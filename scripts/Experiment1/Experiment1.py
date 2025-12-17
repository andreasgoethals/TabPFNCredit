#!/usr/bin/env python3
"""
Experiment 1: Benchmarking TALENT methods with method-level parallelization

This script implements parallel execution of tabular ML methods using SLURM array jobs
with file locking for safe concurrent writes. Each task represents one method on one
dataset with one HPO mode.

Key Features:
- Method-level parallelization (GPU vs CPU separation)
- File locking for safe concurrent result storage
- Automatic cache cleanup to prevent corruption
- Force CPU mode for tree boosting on CPU nodes
- Skip HPO for methods that don't benefit from tuning
"""

import sys
import os
import argparse
import json
import pickle
import fcntl  # Unix file locking
import time
import shutil
from pathlib import Path
from datetime import datetime
import logging

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from src.utils.storage_handler import StorageHandler
from src.methods.method_runner import run_talent_method  
from src.methods.method_config import (
    GPU_METHODS, CPU_METHODS, NO_HPO_METHODS,
    DEEP_METHODS, CLASSICAL_METHODS  # For backwards compatibility
)


def cleanup_corrupted_caches():
    """
    Clean up potentially corrupted cache files that can cause issues.
    
    This addresses the pickle truncation errors we encountered when multiple
    parallel jobs tried to access shared cache files (especially for KNN).
    """
    cache_patterns = [
        '/tmp/talent_*',
        Path.home() / '.cache' / 'talent*',
    ]
    
    cleaned = []
    for pattern in cache_patterns:
        if isinstance(pattern, str):
            # Glob pattern in /tmp
            import glob
            for cache_dir in glob.glob(pattern):
                try:
                    if os.path.isdir(cache_dir):
                        shutil.rmtree(cache_dir)
                    else:
                        os.remove(cache_dir)
                    cleaned.append(cache_dir)
                except Exception:
                    pass
        else:
            # Path object
            for cache_dir in pattern.parent.glob(pattern.name):
                try:
                    if cache_dir.is_dir():
                        shutil.rmtree(cache_dir)
                    else:
                        cache_dir.unlink()
                    cleaned.append(str(cache_dir))
                except Exception:
                    pass
    
    if cleaned:
        print(f"[CLEANUP] Removed {len(cleaned)} corrupted cache(s)")
    
    return len(cleaned)


def generate_task_list(experiment_name="experiment1"):
    """
    Generate all (dataset, method, hpo_mode) combinations.
    
    Tasks are separated into GPU and CPU tasks based on method requirements.
    Methods in NO_HPO_METHODS will only have NO_HPO tasks generated.
    
    Returns:
        tuple: (all_tasks, tasks_by_type) where tasks_by_type contains
               'gpu', 'cpu', and 'all' task lists
    """
    
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
            # Determine which HPO modes to run
            if method in NO_HPO_METHODS:
                hpo_modes = ['NO_HPO']  # Only run without HPO
            else:
                hpo_modes = ['NO_HPO', 'HPO']  # Run both
            
            for hpo_mode in hpo_modes:
                task = {
                    'task_idx': task_idx,
                    'dataset': dataset,
                    'method': method,
                    'task': 'pd',
                    'hpo_mode': hpo_mode,
                    'is_gpu_method': method in GPU_METHODS,
                }
                all_tasks.append(task)
                task_idx += 1
    
    # LGD tasks
    for dataset in lgd_datasets:
        for method in lgd_methods:
            # Determine which HPO modes to run
            if method in NO_HPO_METHODS:
                hpo_modes = ['NO_HPO']  # Only run without HPO
            else:
                hpo_modes = ['NO_HPO', 'HPO']  # Run both
            
            for hpo_mode in hpo_modes:
                task = {
                    'task_idx': task_idx,
                    'dataset': dataset,
                    'method': method,
                    'task': 'lgd',
                    'hpo_mode': hpo_mode,
                    'is_gpu_method': method in GPU_METHODS,
                }
                all_tasks.append(task)
                task_idx += 1
    
    # Separate by execution type
    gpu_tasks = [t for t in all_tasks if t['is_gpu_method']]
    cpu_tasks = [t for t in all_tasks if not t['is_gpu_method']]
    
    return all_tasks, {'gpu': gpu_tasks, 'cpu': cpu_tasks, 'all': all_tasks}


def save_task_lists(experiment_name, tasks_by_type):
    """
    Save task lists as JSON files for SLURM to reference.
    
    Creates separate task lists for GPU and CPU jobs, plus a summary file.
    """
    
    storage = StorageHandler(experiment_name)
    experiment_path = storage.get_experiment_path()
    
    # Save all task lists
    for task_type, tasks in tasks_by_type.items():
        task_file = experiment_path / f"task_list_{task_type}.json"
        with open(task_file, 'w') as f:
            json.dump(tasks, f, indent=2)
    
    # Generate summary
    summary = {
        'total_tasks': len(tasks_by_type['all']),
        'gpu_tasks': len(tasks_by_type['gpu']),
        'cpu_tasks': len(tasks_by_type['cpu']),
        'generated_at': datetime.now().isoformat(),
        'method_breakdown': _generate_method_breakdown(tasks_by_type['all']),
    }
    
    summary_file = experiment_path / "task_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"TASK LISTS GENERATED")
    print(f"{'='*70}")
    print(f"Total tasks:  {summary['total_tasks']}")
    print(f"  GPU tasks:  {summary['gpu_tasks']}")
    print(f"  CPU tasks:  {summary['cpu_tasks']}")
    print(f"Saved to:     {experiment_path}")
    print(f"{'='*70}\n")


def _generate_method_breakdown(all_tasks):
    """Generate summary of tasks by method type."""
    breakdown = {}
    for task in all_tasks:
        method = task['method']
        if method not in breakdown:
            breakdown[method] = {'NO_HPO': 0, 'HPO': 0}
        breakdown[method][task['hpo_mode']] += 1
    return breakdown


def force_cpu_mode_for_method(method, logger):
    """
    Force CPU mode for methods that might try to use GPU on CPU nodes.
    
    This prevents the CatBoost CUDA errors we encountered when tree boosting
    methods try to auto-detect and use GPU on CPU nodes with old CUDA drivers.
    
    Args:
        method: Method name
        logger: Logger instance
        
    Returns:
        dict: Environment variables to set for CPU-only execution
    """
    env_updates = {}
    
    if method == 'catboost':
        # CatBoost: Force CPU mode
        env_updates['CATBOOST_TASK_TYPE'] = 'CPU'
        logger.info("Environment: CATBOOST_TASK_TYPE=CPU")
    
    elif method == 'lightgbm':
        # LightGBM: Force CPU device
        env_updates['LIGHTGBM_DEVICE'] = 'cpu'
        logger.info("Environment: LIGHTGBM_DEVICE=cpu")
    
    elif method == 'xgboost':
        # XGBoost: Force CPU tree method
        env_updates['XGBOOST_TREE_METHOD'] = 'hist'  # CPU histogram method
        logger.info("Environment: XGBOOST_TREE_METHOD=hist")
    
    # Apply environment updates
    for key, value in env_updates.items():
        os.environ[key] = value
    
    return env_updates


def run_single_task(task, experiment_name, config, verbose=False):
    """
    Run ONE method on ONE dataset with ONE HPO mode.
    
    Uses file locking for safe concurrent writes. Implements cache cleanup
    and CPU forcing to prevent the errors encountered in production.
    
    Args:
        task: Task dictionary with dataset, method, task_type, hpo_mode, etc.
        experiment_name: Name of experiment (for storage paths)
        config: Loaded configuration dictionary
        verbose: Enable detailed logging
    """
    
    # Extract task parameters
    task_idx = task['task_idx']
    dataset = task['dataset']
    method = task['method']
    task_type = task['task']
    hpo_mode = task['hpo_mode']
    tune = (hpo_mode == 'HPO')
    is_gpu_method = task['is_gpu_method']
    
    # Initialize storage handler and get experiment path
    storage = StorageHandler(experiment_name)
    experiment_path = storage.get_experiment_path()
    
    # ==========================================
    # SETUP LOGGING (one file per dataset)
    # ==========================================
    log_dir = experiment_path / "logs" / task_type
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{dataset}.log"
    
    # Create logger with unique name to avoid conflicts
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
    
    # ==========================================
    # PRINT TASK BANNER
    # ==========================================
    print(f"\n{'='*70}")
    print(f"TASK {task_idx}: {dataset} | {method} | {hpo_mode}")
    print(f"{'='*70}")
    print(f"Execution:   {'GPU' if is_gpu_method else 'CPU'}")
    print(f"Job ID:      {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    print(f"Array ID:    {os.environ.get('SLURM_ARRAY_TASK_ID', 'N/A')}")
    print(f"Node:        {os.environ.get('SLURMD_NODENAME', 'N/A')}")
    print(f"{'='*70}\n")
    
    logger.info(f"Starting task on node {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    
    # ==========================================
    # CACHE CLEANUP (prevent pickle corruption)
    # ==========================================
    if not is_gpu_method:
        # CPU methods more likely to have cache issues
        cleanup_corrupted_caches()
    
    # ==========================================
    # FORCE CPU MODE (prevent CUDA errors)
    # ==========================================
    if not is_gpu_method and method in {'catboost', 'lightgbm', 'xgboost'}:
        env_updates = force_cpu_mode_for_method(method, logger)
        if env_updates:
            logger.info(f"Forced CPU mode: {env_updates}")
    
    # ==========================================
    # CHECK IF ALREADY COMPLETED
    # ==========================================
    result_file = experiment_path / task_type / f"{dataset}.pkl"
    
    if result_file.exists():
        try:
            with open(result_file, 'rb') as f:
                # Shared lock for reading (multiple readers OK)
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    existing_results = pickle.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Check if this specific combination is done
            if hpo_mode in existing_results and method in existing_results[hpo_mode]:
                logger.info(f"Already completed, skipping")
                print(f"[SKIP] {dataset}/{method}/{hpo_mode} - already done")
                return
        except Exception as e:
            logger.warning(f"Could not check existing results: {e}")
            # Continue with task anyway - better to duplicate than skip
    
    # ==========================================
    # BUILD EXPERIMENT PARAMETERS
    # ==========================================
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
        print(f"[START] {dataset}/{method}/{hpo_mode}")
        
        method_results = run_talent_method(**experiment_params)
        
        logger.info("Training completed successfully")
        
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
                        if mode == 'r+b':
                            try:
                                f.seek(0)
                                results = pickle.load(f)
                            except (EOFError, pickle.UnpicklingError):
                                # Corrupted file - start fresh
                                logger.warning("Corrupted result file, creating new")
                                results = {'NO_HPO': {}, 'HPO': {}}
                        else:
                            results = {'NO_HPO': {}, 'HPO': {}}
                        
                        # Double-check not already written (race condition safety)
                        if hpo_mode in results and method in results[hpo_mode]:
                            logger.info("Results already written by another process, skipping save")
                            return
                        
                        # Add our results
                        if hpo_mode not in results:
                            results[hpo_mode] = {}
                        results[hpo_mode][method] = method_results
                        
                        # Write back to file
                        f.seek(0)
                        f.truncate()
                        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
                        f.flush()
                        os.fsync(f.fileno())  # Force write to disk
                        
                        logger.info("Results saved successfully")
                        
                    finally:
                        # UNLOCK
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
                # Success - break retry loop
                break
                
            except (IOError, OSError, BlockingIOError) as e:
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
        
        # Log to separate error file for easy debugging
        error_log = experiment_path / "logs" / "errors.log"
        with open(error_log, 'a') as ef:
            ef.write(f"\n{'='*70}\n")
            ef.write(f"FAILED: {dataset}/{method}/{hpo_mode}\n")
            ef.write(f"Time: {datetime.now().isoformat()}\n")
            ef.write(f"Error: {str(e)}\n")
            ef.write(f"Node: {os.environ.get('SLURMD_NODENAME', 'N/A')}\n")
            ef.write(f"{'='*70}\n")
        
        raise


def main():
    """Main entry point for Experiment 1."""
    
    parser = argparse.ArgumentParser(
        description="Experiment 1: Benchmark TALENT methods with parallel execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate task lists only
  python Experiment1.py --generate_tasks

  # Run specific task (SLURM array mode)
  python Experiment1.py --task_idx=5 --verbose

  # Run all tasks sequentially (testing)
  python Experiment1.py --verbose
        """
    )
    
    parser.add_argument('--task_idx', type=int, default=None,
                        help='Task index for SLURM array (0-based)')
    parser.add_argument('--generate_tasks', action='store_true',
                        help='Generate task lists only, do not run')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable detailed logging output')
    parser.add_argument('--experiment', type=str, default='experiment1',
                        help='Experiment name (default: experiment1)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Generate task lists
    print(f"\n{'='*70}")
    print("EXPERIMENT 1: TALENT METHOD BENCHMARKING")
    print(f"{'='*70}\n")
    
    all_tasks, tasks_by_type = generate_task_list(args.experiment)
    save_task_lists(args.experiment, tasks_by_type)
    
    if args.generate_tasks:
        print("\n✓ Task lists generated. Submit SLURM jobs to execute:")
        print("  sbatch scripts/Experiment1/Experiment1_GPU.slurm")
        print("  sbatch scripts/Experiment1/Experiment1_CPU.slurm\n")
        return
    
    # Determine which tasks to run
    if args.task_idx is not None:
        # SLURM array mode: run ONE task
        if args.task_idx < 0 or args.task_idx >= len(all_tasks):
            print(f"ERROR: task_idx {args.task_idx} out of range [0, {len(all_tasks)-1}]")
            sys.exit(1)
        
        tasks_to_run = [all_tasks[args.task_idx]]
        print(f"\n[SLURM MODE] Running task {args.task_idx}/{len(all_tasks)-1}")
    else:
        # Sequential mode: run all tasks (for testing only!)
        tasks_to_run = all_tasks
        print(f"\n[SEQUENTIAL MODE] Running all {len(all_tasks)} tasks")
        print("WARNING: This will take a VERY long time. Use SLURM for production!\n")
        
        response = input("Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return
    
    # Execute tasks
    start_time = time.time()
    
    for i, task in enumerate(tasks_to_run):
        try:
            run_single_task(task, args.experiment, config, args.verbose)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            sys.exit(1)
        except Exception as e:
            print(f"\nTask {task['task_idx']} failed: {e}")
            if len(tasks_to_run) == 1:
                # Single task mode - re-raise error
                raise
            else:
                # Multi-task mode - continue with next task
                print("Continuing with next task...\n")
                continue
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"COMPLETED {len(tasks_to_run)} task(s) in {elapsed/60:.1f} minutes")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()