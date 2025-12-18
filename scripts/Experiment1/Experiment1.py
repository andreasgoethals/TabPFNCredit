#!/usr/bin/env python3
"""
Experiment 1: Core execution logic for TALENT benchmarking

Orchestrates method execution with proper error handling and validation.
Supports both GPU and CPU method execution with clear separation.
"""

import os
import sys
import pickle
import fcntl
import time
import logging
from pathlib import Path
from datetime import datetime

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from src.utils.storage_handler import StorageHandler
from src.methods.method_runner import run_talent_method
from src.methods.method_config import NO_HPO_METHODS


def run_single_method(
    dataset: str,
    method: str, 
    task_type: str,
    hpo_mode: str,
    config: dict,
    experiment_name: str = 'experiment1',
    verbose: bool = False
):
    """
    Execute ONE method on ONE dataset with ONE HPO mode.
    
    Args:
        dataset: Dataset name (e.g., '0009.german')
        method: Method name (e.g., 'xgboost', 'LogReg')
        task_type: Task type ('pd' or 'lgd')
        hpo_mode: HPO mode ('NO_HPO' or 'HPO')
        config: Configuration dictionary
        experiment_name: Experiment name for storage
        verbose: Enable detailed logging
    """
    
    tune = (hpo_mode == 'HPO')
    
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
    
    # Create config_hpo directories with proper structure
    config_hpo_base = experiment_path / "config_hpo"
    config_hpo_base.mkdir(exist_ok=True)
    (config_hpo_base / "pd").mkdir(exist_ok=True)
    (config_hpo_base / "lgd").mkdir(exist_ok=True)
    

    
    # Set proper permissions (VSC-specific)
    try:
        dataset_config_dir.chmod(0o755)
    except Exception:
        pass
    
    # ==========================================
    # SETUP LOGGING
    # ==========================================
    log_dir = experiment_path / "logs" / task_type
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{dataset}_{method}_{hpo_mode}.log"
    
    logger = logging.getLogger(f"{experiment_name}.{dataset}.{method}.{hpo_mode}")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        f'%(asctime)s - [{method}/{hpo_mode}] - %(levelname)s - %(message)s'
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
    print(f"{dataset} | {method} | {hpo_mode}")
    print(f"{'='*70}")
    print(f"Node:       {os.environ.get('SLURMD_NODENAME', 'LOCAL')}")
    print(f"Job ID:     {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    print(f"Array ID:   {os.environ.get('SLURM_ARRAY_TASK_ID', 'N/A')}")
    print(f"Log file:   {log_file}")
    print(f"{'='*70}\n")
    
    logger.info(f"Starting on node {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    
    # ==========================================
    # CHECK IF ALREADY COMPLETED
    # ==========================================
    result_file = experiment_path / task_type / f"{dataset}.pkl"
    
    if result_file.exists():
        try:
            with open(result_file, 'rb') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    existing_results = pickle.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            if hpo_mode in existing_results and method in existing_results[hpo_mode]:
                logger.info("Already completed, skipping")
                print(f"[SKIP] {dataset}/{method}/{hpo_mode} - already done")
                return
        except Exception as e:
            logger.warning(f"Could not check existing results: {e}")
    
    # ==========================================
    # BUILD PARAMETERS
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
        'clean_temp_dir': True,  # Clean up temp dirs after completion
    }
    
    try:
        # ==========================================
        # TRAINING & PREDICTION
        # ==========================================
        logger.info("Training started")
        print(f"[TRAIN] {dataset}/{method}/{hpo_mode}")
        
        t_start = time.time()
        method_results = run_talent_method(**experiment_params)
        t_total = time.time() - t_start
        
        logger.info(f"Training completed in {t_total:.1f}s")
        
        # ==========================================
        # VALIDATE RESULTS
        # ==========================================
        if not method_results or len(method_results) == 0:
            raise RuntimeError("run_talent_method returned empty results")
        
        # Check that we got results for all folds
        expected_folds = config['split']['cv_splits']
        if len(method_results) != expected_folds:
            logger.warning(
                f"Expected {expected_folds} folds, got {len(method_results)}"
            )
        
        # Validate each fold has required fields
        for fold_id, fold_results in method_results.items():
            required_fields = ['y_true', 'y_pred', 'metrics', 'train_time']
            missing = [f for f in required_fields if f not in fold_results]
            if missing:
                raise RuntimeError(
                    f"Fold {fold_id} missing required fields: {missing}"
                )
        
        logger.info(f"Results validated: {len(method_results)} folds")
        
        # ==========================================
        # SAVE RESULTS (with file locking)
        # ==========================================
        logger.info("Saving results")
        
        max_retries = 10
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                result_file.parent.mkdir(parents=True, exist_ok=True)
                
                mode = 'r+b' if result_file.exists() else 'w+b'
                
                with open(result_file, mode) as f:
                    # Acquire exclusive lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    
                    try:
                        # Load existing results
                        if mode == 'r+b':
                            try:
                                f.seek(0)
                                results = pickle.load(f)
                            except (EOFError, pickle.UnpicklingError):
                                logger.warning("Corrupted file, creating new")
                                results = {'NO_HPO': {}, 'HPO': {}}
                        else:
                            results = {'NO_HPO': {}, 'HPO': {}}
                        
                        # Ensure structure exists
                        if hpo_mode not in results:
                            results[hpo_mode] = {}
                        
                        # Store results
                        results[hpo_mode][method] = method_results
                        
                        # For NO_HPO methods, duplicate to HPO key for consistency
                        if method in NO_HPO_METHODS and hpo_mode == 'NO_HPO':
                            if 'HPO' not in results:
                                results['HPO'] = {}
                            results['HPO'][method] = method_results
                            logger.info(f"Duplicated NO_HPO results to HPO key for {method}")
                        
                        # Write back
                        f.seek(0)
                        f.truncate()
                        pickle.dump(results, protocol=pickle.HIGHEST_PROTOCOL, file=f)
                        f.flush()
                        os.fsync(f.fileno())
                        
                        logger.info("Results saved successfully")
                        
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
                break  # Success!
                
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
        
        logger.info("Task completed successfully")
        print(f"[DONE] {dataset}/{method}/{hpo_mode}")
        
    except Exception as e:
        logger.error(f"Task failed: {e}", exc_info=True)
        print(f"[FAIL] {dataset}/{method}/{hpo_mode}: {str(e)}")
        
        # Log to consolidated error file
        error_log = experiment_path / "logs" / "errors.log"
        error_log.parent.mkdir(exist_ok=True)
        
        with open(error_log, 'a') as ef:
            ef.write(f"\n{'='*70}\n")
            ef.write(f"FAILED: {dataset}/{method}/{hpo_mode}\n")
            ef.write(f"Time: {datetime.now().isoformat()}\n")
            ef.write(f"Error: {str(e)}\n")
            ef.write(f"Node: {os.environ.get('SLURMD_NODENAME', 'N/A')}\n")
            ef.write(f"{'='*70}\n")
        
        raise


if __name__ == "__main__":
    print("This module provides shared execution logic.")
    print("Use Experiment1_GPU.py or Experiment1_CPU.py instead.")