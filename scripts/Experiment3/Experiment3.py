#!/usr/bin/env python3
"""
Experiment 3: Class Imbalance Analysis

Core execution logic for evaluating how method performance varies with
different minority class proportions (class imbalance levels).

Key differences from Experiment1/2:
- Iterates through multiple minority_proportion values for a single method/dataset
- Uses NO HPO (default parameters only)
- Only applies to PD (classification) tasks
- Uses DataFeeder's sampling parameter for resampling

Result Structure (mirrors Experiment1/2):
    Experiment1: {hpo_mode: {method: {fold_id: {...}}}}
    Experiment2: {row_limit: {method: {fold_id: {...}}}}
    Experiment3: {minority_proportion: {method: {fold_id: {...}}}}

    The minority_proportion (float) replaces row_limit as the first-level key,
    allowing consistent handling across experiments while enabling imbalance analysis.
"""

import os
import sys
import pickle
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# File locking imports
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
    try:
        import portalocker
        HAS_PORTALOCKER = True
    except ImportError:
        HAS_PORTALOCKER = False

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from src.utils.storage_handler import StorageHandler
from src.methods.method_runner import run_talent_method
from src.data.preprocessing import preprocess_dataset
import numpy as np
import gc

# Try to import torch for GPU cleanup between iterations
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _acquire_lock(file_handle, exclusive: bool = True) -> bool:
    """Acquire file lock (cross-platform)."""
    if HAS_FCNTL:
        try:
            lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(file_handle.fileno(), lock_type)
            return True
        except IOError:
            return False
    elif HAS_PORTALOCKER:
        try:
            lock_type = portalocker.LOCK_EX if exclusive else portalocker.LOCK_SH
            portalocker.lock(file_handle, lock_type)
            return True
        except Exception:
            return False
    return True  # Proceed without locking if unavailable


def _release_lock(file_handle) -> None:
    """Release file lock (cross-platform)."""
    if HAS_FCNTL:
        try:
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass
    elif HAS_PORTALOCKER:
        try:
            portalocker.unlock(file_handle)
        except Exception:
            pass


def _cleanup_gpu():
    """Free GPU memory between iterations to prevent OOM from fragmentation."""
    gc.collect()
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_dataset_class_info(task_type: str, dataset: str) -> dict:
    """
    Get class distribution info for a dataset.

    Args:
        task_type: 'pd' (classification only for this experiment)
        dataset: Dataset name (e.g., '0009.german')

    Returns:
        Dict with keys: 'minority_proportion', 'n_minority', 'n_majority', 'n_total'
    """
    # Load dataset to get class distribution
    N, C, y, info = preprocess_dataset(task_type, dataset)

    unique, counts = np.unique(y, return_counts=True)
    if len(unique) != 2:
        raise ValueError(f"Dataset {dataset} is not binary classification (found {len(unique)} classes)")

    n_minority = int(counts.min())
    n_majority = int(counts.max())
    n_total = int(counts.sum())

    return {
        'minority_proportion': n_minority / n_total,
        'n_minority': n_minority,
        'n_majority': n_majority,
        'n_total': n_total,
    }


def generate_minority_proportion_sequence(
    original_proportion: float,
    proportion_max: float,
    proportion_min: float,
    proportion_step: float
) -> List[float]:
    """
    Generate the sequence of minority proportions for class imbalance analysis.

    Algorithm:
    1. Start: current = min(original_proportion, proportion_max)
    2. Include the starting point
    3. Decrease by proportion_step until proportion_min is reached
    4. Uses 10 decimal precision internally to avoid floating-point accumulation errors
       while preserving full precision in output (critical for Experiment 3 analysis)

    Example:
        original=0.35, max=0.50, min=0.01, step=0.05
        -> [0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.01]

    Args:
        original_proportion: Original minority proportion in the dataset
        proportion_max: Upper bound for minority proportion
        proportion_min: Lower bound for minority proportion
        proportion_step: Step size for decreasing proportion

    Returns:
        List of minority proportions in decreasing order (full precision preserved)
    """
    sequence = []

    # Start with the smaller of original_proportion and proportion_max
    # We can only decrease the minority proportion, not increase it beyond original
    # Use 10 decimal precision to avoid floating-point errors while preserving accuracy
    current = min(original_proportion, proportion_max)
    current = round(current, 10)

    # If original proportion is already below min, just return original
    if current < proportion_min:
        return [current]

    # Add the starting point (full precision preserved)
    sequence.append(current)

    # Align to nearest lower multiple of proportion_step
    # Use high precision (10 decimals) to avoid floating-point accumulation errors
    aligned = round((current // proportion_step) * proportion_step, 10)
    if aligned < current and aligned >= proportion_min:
        current = aligned
        sequence.append(current)

    # Decrease by proportion_step until proportion_min
    # CRITICAL: Use epsilon comparison to avoid infinite loops from floating-point errors
    epsilon = 1e-12
    while round(current - proportion_step, 10) >= proportion_min - epsilon:
        current = round(current - proportion_step, 10)  # High precision arithmetic
        # Safety check: if we've gone below min due to floating-point, stop
        if current < proportion_min - epsilon:
            break
        sequence.append(current)

    # Ensure proportion_min is included if we haven't reached it exactly
    if abs(sequence[-1] - proportion_min) > epsilon and sequence[-1] > proportion_min:
        sequence.append(proportion_min)

    return sequence


def run_imbalance_analysis(
    dataset: str,
    method: str,
    task_type: str,
    config: dict,
    experiment_name: str = 'experiment3',
    verbose: bool = False
):
    """
    Execute class imbalance analysis for ONE method on ONE dataset.

    Iterates through all minority proportions and stores results incrementally.

    Args:
        dataset: Dataset name (e.g., '0009.german')
        method: Method name (e.g., 'xgboost', 'LogReg')
        task_type: Task type ('pd' only - LGD not supported for imbalance analysis)
        config: Configuration dictionary (must include 'imbalance' section)
        experiment_name: Experiment name for storage
        verbose: Enable detailed logging
    """
    # Validate task type
    if task_type.lower() != 'pd':
        raise ValueError(
            f"Experiment 3 (Class Imbalance Analysis) only supports PD (classification) tasks. "
            f"Got task_type='{task_type}'"
        )

    # Initialize storage
    storage = StorageHandler(experiment_name)
    experiment_path = storage.get_experiment_path()

    # ==========================================
    # ENSURE ALL DIRECTORIES EXIST
    # ==========================================
    experiment_path.mkdir(parents=True, exist_ok=True)
    (experiment_path / "pd").mkdir(exist_ok=True)
    (experiment_path / "logs").mkdir(exist_ok=True)

    # Create config_hpo directories
    config_hpo_base = experiment_path / "config_hpo"
    config_hpo_base.mkdir(exist_ok=True)
    (config_hpo_base / "pd").mkdir(exist_ok=True)

    # ==========================================
    # SETUP LOGGING
    # ==========================================
    log_dir = experiment_path / "logs" / task_type
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{dataset}_{method}_imbalance_curve.log"

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
    print(f"CLASS IMBALANCE ANALYSIS: {dataset} | {method} | {task_type.upper()}")
    print(f"{'='*70}")
    print(f"Node:       {os.environ.get('SLURMD_NODENAME', 'LOCAL')}")
    print(f"Job ID:     {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    print(f"Array ID:   {os.environ.get('SLURM_ARRAY_TASK_ID', 'N/A')}")
    print(f"Log file:   {log_file}")
    print(f"{'='*70}\n")

    logger.info(f"Starting imbalance analysis on node {os.environ.get('SLURMD_NODENAME', 'unknown')}")

    # ==========================================
    # GET DATASET INFO & GENERATE PROPORTION SEQUENCE
    # ==========================================
    imbalance_config = config['imbalance']
    proportion_max = imbalance_config['minority_proportion_max']
    proportion_min = imbalance_config['minority_proportion_min']
    proportion_step = imbalance_config['minority_proportion_step']

    print(f"[CONFIG] proportion_max={proportion_max}, proportion_min={proportion_min}, proportion_step={proportion_step}")
    logger.info(f"Imbalance config: max={proportion_max}, min={proportion_min}, step={proportion_step}")

    # Get original class distribution info
    print(f"[INFO] Loading dataset to determine class distribution...")
    class_info = get_dataset_class_info(task_type, dataset)
    original_proportion = class_info['minority_proportion']
    n_minority_original = class_info['n_minority']
    n_majority_original = class_info['n_majority']
    print(f"[INFO] Original minority proportion: {original_proportion:.4f} ({original_proportion*100:.2f}%)")
    print(f"[INFO] Class counts: minority={n_minority_original}, majority={n_majority_original}")
    logger.info(f"Dataset {dataset} has original minority proportion {original_proportion:.4f} "
                f"(minority={n_minority_original}, majority={n_majority_original})")

    # Generate minority proportion sequence
    proportions = generate_minority_proportion_sequence(
        original_proportion, proportion_max, proportion_min, proportion_step
    )
    print(f"[INFO] Minority proportion sequence ({len(proportions)} points): {proportions}")
    logger.info(f"Proportion sequence: {proportions}")

    # ==========================================
    # LOAD EXISTING RESULTS (if any)
    # ==========================================
    # Result structure: {minority_proportion: {method: {fold_id: {...}}}}
    result_file = experiment_path / task_type / f"{dataset}.pkl"

    existing_results = {}
    if result_file.exists():
        try:
            with open(result_file, 'rb') as f:
                _acquire_lock(f, exclusive=False)
                try:
                    existing_results = pickle.load(f)
                finally:
                    _release_lock(f)

            # Check which proportions already have this method completed
            completed_proportions = [
                p for p, methods in existing_results.items()
                if isinstance(methods, dict) and method in methods
            ]
            if completed_proportions:
                logger.info(f"Found existing results for {method} at proportions: {completed_proportions}")
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")

    # ==========================================
    # ITERATE THROUGH MINORITY PROPORTIONS
    # ==========================================
    completed_proportions = []
    failed_proportions = []

    for i, minority_proportion in enumerate(proportions):
        print(f"\n{'-'*50}")
        print(f"[{i+1}/{len(proportions)}] Minority proportion: {minority_proportion:.2f} ({minority_proportion*100:.1f}%)")
        print(f"{'-'*50}")

        # Check if already completed for this method at this proportion
        if minority_proportion in existing_results and method in existing_results.get(minority_proportion, {}):
            print(f"  [SKIP] Already completed")
            logger.info(f"Skipping minority_proportion={minority_proportion} (already completed)")
            completed_proportions.append(minority_proportion)
            continue

        # Check if this proportion would leave enough minority samples for CV
        # After sampling: if current > target, minority gets downsampled to:
        #   desired_min = int(target / (1 - target) * n_majority)
        # We need enough minority samples so that:
        #   - StratifiedKFold can create cv_splits folds (>= cv_splits per class)
        #   - Each fold's validation set has at least 1 minority sample
        # Conservative threshold: cv_splits * 3 ensures enough for train/val/test in each fold
        cv_splits = config['split']['cv_splits']
        min_required = cv_splits * 3
        if original_proportion > minority_proportion:
            # Minority will be downsampled
            expected_minority = int(minority_proportion / (1 - minority_proportion) * n_majority_original)
        else:
            # Majority will be downsampled (minority stays the same)
            expected_minority = n_minority_original

        if expected_minority < min_required:
            print(f"  [SKIP] Proportion {minority_proportion:.4f} would yield only {expected_minority} "
                  f"minority samples (need >= {min_required} for reliable {cv_splits}-fold CV)")
            logger.info(f"Skipping minority_proportion={minority_proportion}: "
                        f"only {expected_minority} minority samples (need >= {min_required})")
            completed_proportions.append(minority_proportion)
            continue

        # Build parameters for this minority proportion
        # The key is passing 'sampling' parameter to run_talent_method
        # DataFeeder will handle the actual resampling
        experiment_params = {
            'task': task_type,
            'dataset': dataset,
            'test_size': config['split']['test_size'],
            'val_size': config['split']['val_size'],
            'cv_splits': config['split']['cv_splits'],
            'seed': config['split']['seed'],
            'row_limit': config['split'].get('row_limit', None),
            'sampling': minority_proportion,  # KEY: Pass minority proportion as sampling parameter
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
            logger.info(f"Training with minority_proportion={minority_proportion}")
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
                minority_proportion=minority_proportion,
                fold_results=fold_results,
                logger=logger
            )

            completed_proportions.append(minority_proportion)

        except Exception as e:
            error_str = str(e)

            # Check if this is a class-imbalance-related error that should be skipped
            # rather than treated as a real failure
            skip_patterns = [
                "only one label",
                "too few. The minimum number of groups",
                "least populated class",
                "Cannot optimize threshold without validation predictions",
            ]
            is_imbalance_error = any(p in error_str for p in skip_patterns)

            if is_imbalance_error:
                # This proportion is too extreme for this method/dataset — skip gracefully
                print(f"  [SKIP] Proportion too extreme for reliable CV: {error_str[:80]}...")
                logger.info(f"Skipping minority_proportion={minority_proportion} "
                            f"(class imbalance too extreme): {error_str[:120]}")
                completed_proportions.append(minority_proportion)
            else:
                # Real failure — log to error file
                logger.error(f"Failed at minority_proportion={minority_proportion}: {e}", exc_info=True)
                print(f"  [FAIL] {error_str}")
                failed_proportions.append((minority_proportion, error_str))

                error_log = experiment_path / "logs" / "errors.log"
                with open(error_log, 'a') as ef:
                    ef.write(f"\n{'='*70}\n")
                    ef.write(f"FAILED: {dataset}/{method}/minority_proportion={minority_proportion}\n")
                    ef.write(f"Time: {datetime.now().isoformat()}\n")
                    ef.write(f"Error: {error_str}\n")
                    ef.write(f"Node: {os.environ.get('SLURMD_NODENAME', 'N/A')}\n")
                    ef.write(f"{'='*70}\n")

            # Continue with next proportion (don't abort entire analysis)
            continue

        finally:
            # Free GPU memory between iterations to prevent OOM from fragmentation
            _cleanup_gpu()

    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print(f"\n{'='*70}")
    print(f"CLASS IMBALANCE ANALYSIS COMPLETE: {dataset} | {method}")
    print(f"{'='*70}")
    print(f"Completed: {len(completed_proportions)}/{len(proportions)}")
    if failed_proportions:
        print(f"Failed:    {len(failed_proportions)}")
        for prop, err in failed_proportions:
            print(f"  - minority_proportion={prop}: {err[:50]}...")
    print(f"{'='*70}\n")

    logger.info(f"Imbalance analysis completed: {len(completed_proportions)}/{len(proportions)} successful")


def _save_results_incremental(
    result_file: Path,
    method: str,
    minority_proportion: float,
    fold_results: dict,
    logger: logging.Logger,
    max_retries: int = 10,
    retry_delay: float = 0.5
):
    """
    Save results incrementally with file locking.

    Result structure: results[minority_proportion][method] = fold_results

    This mirrors Experiment1/2's structure where the first level was hpo_mode/row_limit:
        Experiment1: results[hpo_mode][method] = fold_results
        Experiment2: results[row_limit][method] = fold_results
        Experiment3: results[minority_proportion][method] = fold_results

    Args:
        result_file: Path to pickle file
        method: Method name
        minority_proportion: Minority class proportion used
        fold_results: Results from run_talent_method
        logger: Logger instance
        max_retries: Maximum retry attempts
        retry_delay: Initial retry delay (exponential backoff)
    """
    for attempt in range(max_retries):
        try:
            result_file.parent.mkdir(parents=True, exist_ok=True)

            mode = 'r+b' if result_file.exists() else 'w+b'

            with open(result_file, mode) as f:
                _acquire_lock(f, exclusive=True)

                try:
                    if mode == 'r+b':
                        try:
                            f.seek(0)
                            results = pickle.load(f)
                        except (EOFError, pickle.UnpicklingError):
                            logger.warning("Corrupted file, creating new")
                            results = {}
                    else:
                        results = {}

                    # Ensure minority_proportion key exists (first level)
                    if minority_proportion not in results:
                        results[minority_proportion] = {}

                    # Store results for this method (second level)
                    results[minority_proportion][method] = fold_results

                    # Write back
                    f.seek(0)
                    f.truncate()
                    pickle.dump(results, protocol=pickle.HIGHEST_PROTOCOL, file=f)
                    f.flush()
                    os.fsync(f.fileno())

                    logger.info(f"Saved results for minority_proportion={minority_proportion}")

                finally:
                    _release_lock(f)

            return  # Success!

        except (IOError, OSError, BlockingIOError) as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"Lock failed (attempt {attempt+1}/{max_retries}), waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to save after {max_retries} attempts")
                raise RuntimeError(f"Could not save results: {result_file}") from e


if __name__ == "__main__":
    print("This module provides shared execution logic for Experiment3.")
    print("Use Experiment3_GPU_Standard.py, Experiment3_GPU_Foundation.py, or Experiment3_CPU.py instead.")
