#!/usr/bin/env python3
"""
Experiment 1: Core execution logic for TALENT benchmarking

This module provides shared execution logic used by both GPU and CPU orchestrators.
Handles training, file locking, result storage, and error handling.

Key Features:
- Safe concurrent writes with file locking
- Automatic cache cleanup
- CPU mode forcing for tree boosting
- NO_HPO result duplication to HPO key for consistency
"""

import os
import sys
import pickle
import fcntl
import time
import shutil
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


def cleanup_corrupted_caches():
    """
    Clean up potentially corrupted cache files.
    Prevents pickle truncation errors from concurrent access.
    """
    cache_patterns = [
        '/tmp/talent_*',
        Path.home() / '.cache' / 'talent*',
    ]
    
    cleaned = []
    for pattern in cache_patterns:
        if isinstance(pattern, str):
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


def force_cpu_mode_for_method(method, logger):
    """
    Force CPU mode for tree boosting methods on CPU nodes.
    Prevents CUDA errors when methods try to auto-detect GPU.
    """
    env_updates = {}
    
    if method == 'catboost':
        env_updates['CATBOOST_TASK_TYPE'] = 'CPU'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide all GPUs
        logger.info("Environment: CATBOOST_TASK_TYPE=CPU, CUDA_VISIBLE_DEVICES=''")
    
    elif method == 'lightgbm':
        env_updates['LIGHTGBM_DEVICE'] = 'cpu'
        logger.info("Environment: LIGHTGBM_DEVICE=cpu")
    
    elif method == 'xgboost':
        env_updates['XGBOOST_TREE_METHOD'] = 'hist'
        logger.info("Environment: XGBOOST_TREE_METHOD=hist")
    
    for key, value in env_updates.items():
        os.environ[key] = value
    
    return env_updates


def run_single_method(dataset, method, task_type, hpo_mode, 
                      config, experiment_name='experiment1', verbose=False):
    """
    Execute ONE method on ONE dataset with ONE HPO mode.
    
    This is the core execution function called by both GPU and CPU orchestrators.
    Uses file locking to safely write results to shared files.
    
    For methods in NO_HPO_METHODS:
    - Results are saved to both NO_HPO and HPO keys
    - This ensures downstream analysis always finds results in both places
    
    Args:
        dataset: Dataset name (e.g., '0009.german')
        method: Method name (e.g., 'xgboost', 'tabpfn')
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
    # SETUP LOGGING
    # ==========================================
    log_dir = experiment_path / "logs" / task_type
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{dataset}.log"
    
    logger_name = f"{experiment_name}.{dataset}.{method}.{hpo_mode}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.handlers.clear()
    
    log_format = f'%(asctime)s - [{method}/{hpo_mode}] - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
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
    print(f"Node:     {os.environ.get('SLURMD_NODENAME', 'N/A')}")
    print(f"Job ID:   {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    print(f"Array ID: {os.environ.get('SLURM_ARRAY_TASK_ID', 'N/A')}")
    print(f"{'='*70}\n")
    
    logger.info(f"Starting on node {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    
    # ==========================================
    # CACHE CLEANUP
    # ==========================================
    cleanup_corrupted_caches()
    
    # ==========================================
    # FORCE CPU MODE (for tree boosting on CPU)
    # ==========================================
    if method in {'catboost', 'lightgbm', 'xgboost'}:
        # Check if we're on CPU node (no GPU visible)
        is_cpu_node = (
            'CUDA_VISIBLE_DEVICES' not in os.environ or 
            os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        )
        
        if is_cpu_node:
            env_updates = force_cpu_mode_for_method(method, logger)
            logger.info(f"Forced CPU mode: {env_updates}")
    
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
    }
    
    try:
        # ==========================================
        # TRAINING
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
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                result_file.parent.mkdir(parents=True, exist_ok=True)
                
                mode = 'r+b' if result_file.exists() else 'w+b'
                
                with open(result_file, mode) as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    
                    try:
                        if mode == 'r+b':
                            try:
                                f.seek(0)
                                results = pickle.load(f)
                            except (EOFError, pickle.UnpicklingError):
                                logger.warning("Corrupted file, creating new")
                                results = {'NO_HPO': {}, 'HPO': {}}
                        else:
                            results = {'NO_HPO': {}, 'HPO': {}}
                        
                        if hpo_mode in results and method in results[hpo_mode]:
                            logger.info("Already written by another process")
                            return
                        
                        # Ensure both keys exist
                        if 'NO_HPO' not in results:
                            results['NO_HPO'] = {}
                        if 'HPO' not in results:
                            results['HPO'] = {}
                        
                        # ==========================================
                        # SPECIAL HANDLING FOR NO_HPO METHODS
                        # ==========================================
                        if method in NO_HPO_METHODS:
                            # For NO_HPO methods, save to BOTH keys
                            results['NO_HPO'][method] = method_results
                            results['HPO'][method] = method_results
                            logger.info(f"NO_HPO method: duplicated results to both NO_HPO and HPO")
                        else:
                            # Normal methods: save to specified key
                            results[hpo_mode][method] = method_results
                        
                        # Write back to file
                        f.seek(0)
                        f.truncate()
                        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
                        f.flush()
                        os.fsync(f.fileno())
                        
                        logger.info("Results saved successfully")
                        
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
                break
                
            except (IOError, OSError, BlockingIOError) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Lock failed (attempt {attempt+1}/{max_retries}), "
                                 f"waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {max_retries} attempts")
                    raise RuntimeError(f"Could not save results: {result_file}") from e
        
        logger.info("Task completed successfully")
        print(f"[DONE] {dataset}/{method}/{hpo_mode}")
        
    except Exception as e:
        logger.error(f"Task failed: {e}", exc_info=True)
        print(f"[FAIL] {dataset}/{method}/{hpo_mode}: {str(e)}")
        
        error_log = experiment_path / "logs" / "errors.log"
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