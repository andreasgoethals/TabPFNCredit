#!/usr/bin/env python3
"""
Experiment 0: Core execution logic for method validation

Tests ALL methods on 2 datasets with minimal settings to identify working methods.
NO HPO - just quick validation runs.
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
from src.utils.logging_setup import configure_task_logging, task_timer
from src.utils.result_io import save_method
from src.methods.method_runner import run_talent_method
from src.methods.method_config import NO_HPO_METHODS


def run_single_method(
    dataset: str,
    method: str, 
    task_type: str,
    config: dict,
    experiment_name: str = 'experiment0',
    verbose: bool = False
):
    """
    Execute ONE method on ONE dataset (NO HPO mode only).
    
    Note: Experiment0 only runs NO_HPO mode for all methods.
    
    Args:
        dataset: Dataset name (e.g., '0001.gmsc')
        method: Method name (e.g., 'xgboost', 'LogReg')
        task_type: Task type ('pd' or 'lgd')
        config: Configuration dictionary
        experiment_name: Experiment name for storage
        verbose: Enable detailed logging
    """
    
    # Experiment0 always uses NO_HPO
    hpo_mode = 'NO_HPO'
    tune = False

    # Initialize storage
    storage = StorageHandler(experiment_name)
    experiment_path = storage.get_experiment_path()
    experiment_path.mkdir(parents=True, exist_ok=True)
    (experiment_path / "pd").mkdir(exist_ok=True)
    (experiment_path / "lgd").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Hybrid + minimal logging (per-task detail + summary + errors)
    # ------------------------------------------------------------------
    log_file = configure_task_logging(
        experiment=experiment_name,
        dataset=dataset,
        method=method,
        task=task_type,
        results_root=experiment_path.parent,
        verbose=verbose,
    )
    logger = logging.getLogger(__name__)
    logger.info(
        "node=%s job=%s array=%s log=%s",
        os.environ.get('SLURMD_NODENAME', 'LOCAL'),
        os.environ.get('SLURM_JOB_ID', '-'),
        os.environ.get('SLURM_ARRAY_TASK_ID', '-'),
        log_file,
    )
    
    # ==========================================
    # CHECK IF ALREADY COMPLETED
    # ==========================================
    result_file = experiment_path / task_type / f"{dataset}.pkl"
    
    # Skip if a per-(dataset, method) JSON already exists with all folds.
    method_json = experiment_path / task_type / dataset / f"{method}.json"
    if method_json.exists():
        logger.info("Already completed (%s); skipping", method_json)
        return
    
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
        'n_trials': 1,  # No tuning in Experiment0
        'tune': False,  # Always NO_HPO
        'config_base_dir': experiment_path,
        'verbose': verbose,
        'clean_temp_dir': True,
    }
    
    try:
        with task_timer(f"{dataset}/{method}/{task_type}", logger):
            method_results = run_talent_method(**experiment_params)

        if not method_results:
            raise RuntimeError("run_talent_method returned empty results")

        # Per-(dataset, method) JSON + npz -- no locks needed.
        save_method(
            method_results,
            base=experiment_path.parent,
            experiment=experiment_name,
            task=task_type,
            dataset=dataset,
            method=method,
        )
        # Headline metric for the summary log
        first = next(iter(method_results.values()))
        headline = first["metrics"].get("AUC" if task_type == "pd" else "RMSE", "n/a")
        logger.info("metric=%s value=%s folds=%d",
                    "AUC" if task_type == "pd" else "RMSE",
                    f"{headline:.4f}" if isinstance(headline, (int, float)) else headline,
                    len(method_results))
        
    except Exception:
        # task_timer already logged the traceback at ERROR level; that lands
        # in both the per-task .log and errors.log via configure_task_logging.
        raise


if __name__ == "__main__":
    print("This module provides shared execution logic for Experiment0.")
    print("Use Experiment0_GPU.py or Experiment0_CPU.py instead.")