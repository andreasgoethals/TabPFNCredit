#!/usr/bin/env python3
"""
Experiment 2: Learning Curve Analysis

Core execution logic for evaluating how method performance degrades
as the number of training rows decreases.

Key differences from Experiment1:
- Iterates through multiple row limits for a single method/dataset
- Uses NO HPO (default parameters only)
- Row limit sequence generated dynamically based on dataset size

Result Structure (mirrors Experiment1):
    Experiment1: {hpo_mode: {method: {fold_id: {...}}}}
    Experiment2: {row_limit: {method: {fold_id: {...}}}}

    The row_limit replaces hpo_mode as the first-level key, allowing
    consistent handling across experiments while enabling learning curve analysis.
"""

import os
import sys
import pickle
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from src.utils.storage_handler import StorageHandler
from src.utils.file_lock import FileLock, acquire_lock, release_lock, atomic_pickle_write
from src.methods.method_runner import run_talent_method
from src.data.preprocessing import preprocess_dataset
import gc

# Try to import torch for GPU cleanup between iterations
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# Backwards-compatible aliases
_acquire_lock = acquire_lock
_release_lock = release_lock


def _cleanup_gpu():
    """Free GPU memory between iterations to prevent OOM from fragmentation."""
    gc.collect()
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_dataset_size(task_type: str, dataset: str) -> int:
    """
    Get the number of rows in a dataset.

    Args:
        task_type: 'pd' or 'lgd'
        dataset: Dataset name (e.g., '0008.german')

    Returns:
        Number of rows in the dataset
    """
    # Load dataset to get size (minimal processing)
    N, C, y, info = preprocess_dataset(task_type, dataset)
    return len(y)


def generate_row_limit_sequence(
    dataset_size: int,
    row_max: int,
    row_min: int,
    row_step: int
) -> List[int]:
    """
    Generate the sequence of row limits for learning curve analysis.

    Algorithm:
    1. Start: current = min(dataset_size, row_max)
    2. First step: If current is not a multiple of row_step,
       include current, then align to nearest lower multiple of row_step
    3. Continue: Decrease by row_step until row_min is reached

    Example:
        dataset_size=8500, row_max=50000, row_min=1000, row_step=1000
        -> [8500, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000]

    Args:
        dataset_size: Actual number of rows in the dataset
        row_max: Upper bound for row limit
        row_min: Lower bound for row limit
        row_step: Step size for decreasing rows

    Returns:
        List of row limits in decreasing order
    """
    sequence = []

    # Start with the smaller of dataset_size and row_max
    current = min(dataset_size, row_max)

    # If dataset is smaller than row_min, just return the dataset size
    if current < row_min:
        return [current]

    # Add the starting point
    sequence.append(current)

    # Align to nearest lower multiple of row_step
    aligned = (current // row_step) * row_step
    if aligned < current and aligned >= row_min:
        current = aligned
        sequence.append(current)

    # Decrease by row_step until row_min
    while current - row_step >= row_min:
        current -= row_step
        sequence.append(current)

    return sequence


def run_learning_curve(
    dataset: str,
    method: str,
    task_type: str,
    config: dict,
    experiment_name: str = 'experiment2',
    verbose: bool = False
):
    """
    Execute learning curve analysis for ONE method on ONE dataset.

    Iterates through all row limits and stores results incrementally.

    Args:
        dataset: Dataset name (e.g., '0008.german')
        method: Method name (e.g., 'xgboost', 'LogReg')
        task_type: Task type ('pd' or 'lgd')
        config: Configuration dictionary (must include 'learning_curve' section)
        experiment_name: Experiment name for storage
        verbose: Enable detailed logging
    """

    # Initialize storage
    storage = StorageHandler(experiment_name)
    experiment_path = storage.get_experiment_path()

    # ==========================================
    # ENSURE ALL DIRECTORIES EXIST
    # ==========================================
    experiment_path.mkdir(parents=True, exist_ok=True)
    (experiment_path / "pd").mkdir(exist_ok=True)
    (experiment_path / "lgd").mkdir(exist_ok=True)
    (experiment_path / "logs").mkdir(exist_ok=True)

    # Create config_hpo directories
    config_hpo_base = experiment_path / "config_hpo"
    config_hpo_base.mkdir(exist_ok=True)
    (config_hpo_base / "pd").mkdir(exist_ok=True)
    (config_hpo_base / "lgd").mkdir(exist_ok=True)

    # ==========================================
    # SETUP LOGGING
    # ==========================================
    log_dir = experiment_path / "logs" / task_type
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{dataset}_{method}_learning_curve.log"

    logger = logging.getLogger(f"{experiment_name}.{dataset}.{method}")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.handlers.clear()

    formatter = logging.Formatter(
        f'%(asctime)s - [{method}] - %(levelname)s - %(message)s'
    )

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # ==========================================
    # PRINT BANNER
    # ==========================================
    print(f"\n{'='*70}")
    print(f"LEARNING CURVE: {dataset} | {method} | {task_type.upper()}")
    print(f"{'='*70}")
    print(f"Node:       {os.environ.get('SLURMD_NODENAME', 'LOCAL')}")
    print(f"Job ID:     {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    print(f"Array ID:   {os.environ.get('SLURM_ARRAY_TASK_ID', 'N/A')}")
    print(f"Log file:   {log_file}")
    print(f"{'='*70}\n")

    logger.info(f"Starting learning curve on node {os.environ.get('SLURMD_NODENAME', 'unknown')}")

    # ==========================================
    # GET DATASET SIZE & GENERATE ROW LIMITS
    # ==========================================
    lc_config = config['learning_curve']
    row_max = lc_config['row_max']
    row_min = lc_config['row_min']
    row_step = lc_config['row_step']

    print(f"[CONFIG] row_max={row_max}, row_min={row_min}, row_step={row_step}")
    logger.info(f"Learning curve config: max={row_max}, min={row_min}, step={row_step}")

    # Get actual dataset size
    print(f"[INFO] Loading dataset to determine size...")
    dataset_size = get_dataset_size(task_type, dataset)
    print(f"[INFO] Dataset size: {dataset_size:,} rows")
    logger.info(f"Dataset {dataset} has {dataset_size} rows")

    # Generate row limit sequence
    row_limits = generate_row_limit_sequence(dataset_size, row_max, row_min, row_step)
    print(f"[INFO] Row limit sequence ({len(row_limits)} points): {row_limits}")
    logger.info(f"Row limit sequence: {row_limits}")

    # ==========================================
    # LOAD EXISTING RESULTS (if any)
    # ==========================================
    # Result structure: {row_limit: {method: {fold_id: {...}}}}
    # This mirrors Experiment1's {hpo_mode: {method: {fold_id: {...}}}}
    result_file = experiment_path / task_type / f"{dataset}.pkl"

    existing_results = {}
    if result_file.exists() and result_file.stat().st_size > 0:
        try:
            with FileLock(result_file, exclusive=False) as f:
                f.seek(0)
                existing_results = pickle.load(f)

            # Check which row_limits already have this method completed
            completed_row_limits = [
                rl for rl, methods in existing_results.items()
                if isinstance(methods, dict) and method in methods
            ]
            if completed_row_limits:
                logger.info(f"Found existing results for {method} at row_limits: {completed_row_limits}")
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")

    # ==========================================
    # ITERATE THROUGH ROW LIMITS
    # ==========================================
    completed_limits = []
    failed_limits = []

    for i, row_limit in enumerate(row_limits):
        print(f"\n{'─'*50}")
        print(f"[{i+1}/{len(row_limits)}] Row limit: {row_limit:,}")
        print(f"{'─'*50}")

        # Check if already completed for this method at this row_limit
        if row_limit in existing_results and method in existing_results.get(row_limit, {}):
            print(f"  [SKIP] Already completed")
            logger.info(f"Skipping row_limit={row_limit} (already completed)")
            completed_limits.append(row_limit)
            continue

        # Build parameters for this row limit
        experiment_params = {
            'task': task_type,
            'dataset': dataset,
            'test_size': config['split']['test_size'],
            'val_size': config['split']['val_size'],
            'cv_splits': config['split']['cv_splits'],
            'seed': config['split']['seed'],
            'row_limit': row_limit,  # Key difference: explicit row limit
            'sampling': config['split'].get('sampling', None),
            'method': method,
            'max_epoch': config['training']['max_epochs'],
            'batch_size': config['training']['batch_size'],
            'early_stopping': config['training']['early_stopping'],
            'early_stopping_patience': config['training']['early_stopping_patience'],
            'n_trials': 1,  # No HPO
            'tune': False,  # No HPO
            'config_base_dir': experiment_path,
            'verbose': verbose,
            'clean_temp_dir': True,
        }

        try:
            # ==========================================
            # TRAINING & PREDICTION
            # ==========================================
            logger.info(f"Training with row_limit={row_limit}")
            print(f"  [TRAIN] Starting...")

            t_start = time.time()
            fold_results = run_talent_method(**experiment_params)
            t_total = time.time() - t_start

            logger.info(f"Training completed in {t_total:.1f}s")
            print(f"  [DONE] Completed in {t_total:.1f}s")

            # Validate results
            if not fold_results or len(fold_results) == 0:
                raise RuntimeError("run_talent_method returned empty results")

            # ==========================================
            # SAVE RESULTS INCREMENTALLY
            # ==========================================
            _save_results_incremental(
                result_file=result_file,
                method=method,
                row_limit=row_limit,
                fold_results=fold_results,
                logger=logger
            )

            completed_limits.append(row_limit)

        except Exception as e:
            logger.error(f"Failed at row_limit={row_limit}: {e}", exc_info=True)
            print(f"  [FAIL] {str(e)}")
            failed_limits.append((row_limit, str(e)))

            # Log to consolidated error file
            error_log = experiment_path / "logs" / "errors.log"
            with open(error_log, 'a') as ef:
                ef.write(f"\n{'='*70}\n")
                ef.write(f"FAILED: {dataset}/{method}/row_limit={row_limit}\n")
                ef.write(f"Time: {datetime.now().isoformat()}\n")
                ef.write(f"Error: {str(e)}\n")
                ef.write(f"Node: {os.environ.get('SLURMD_NODENAME', 'N/A')}\n")
                ef.write(f"{'='*70}\n")

            # Continue with next row limit (don't abort entire learning curve)
            continue

        finally:
            # Free GPU memory between iterations to prevent OOM from fragmentation
            _cleanup_gpu()

    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print(f"\n{'='*70}")
    print(f"LEARNING CURVE COMPLETE: {dataset} | {method}")
    print(f"{'='*70}")
    print(f"Completed: {len(completed_limits)}/{len(row_limits)}")
    if failed_limits:
        print(f"Failed:    {len(failed_limits)}")
        for rl, err in failed_limits:
            print(f"  - row_limit={rl}: {err[:50]}...")
    print(f"{'='*70}\n")

    logger.info(f"Learning curve completed: {len(completed_limits)}/{len(row_limits)} successful")


def _save_results_incremental(
    result_file: Path,
    method: str,
    row_limit: int,
    fold_results: dict,
    logger: logging.Logger,
    max_retries: int = 10,
    retry_delay: float = 0.5
):
    """
    Save results incrementally with file locking.

    Result structure: results[row_limit][method] = fold_results

    This mirrors Experiment1's structure where the first level was hpo_mode:
        Experiment1: results[hpo_mode][method] = fold_results
        Experiment2: results[row_limit][method] = fold_results

    Args:
        result_file: Path to pickle file
        method: Method name
        row_limit: Row limit used
        fold_results: Results from run_talent_method
        logger: Logger instance
        max_retries: Maximum retry attempts
        retry_delay: Initial retry delay (exponential backoff)
    """
    for attempt in range(max_retries):
        try:
            result_file.parent.mkdir(parents=True, exist_ok=True)

            with FileLock(result_file, exclusive=True) as f:
                if result_file.stat().st_size > 0:
                    try:
                        f.seek(0)
                        results = pickle.load(f)
                    except (EOFError, pickle.UnpicklingError):
                        logger.warning("Corrupted file, creating new")
                        results = {}
                else:
                    results = {}

                # First-level key is the row_limit (mirrors Experiment1's hpo_mode).
                results.setdefault(row_limit, {})[method] = fold_results

                # Atomic write: writes {file}.tmp then os.replace().
                atomic_pickle_write(result_file, results)
                logger.info(f"Saved results for row_limit={row_limit}")

            return  # Success!

        except (IOError, OSError, BlockingIOError) as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(
                    f"Lock failed (attempt {attempt+1}/{max_retries}), "
                    f"waiting {wait_time:.1f}s"
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to save after {max_retries} attempts")
                raise RuntimeError(f"Could not save results: {result_file}") from e


if __name__ == "__main__":
    print("This module provides shared execution logic for Experiment2.")
    print("Use Experiment2_GPU.py or Experiment2_CPU.py instead.")
