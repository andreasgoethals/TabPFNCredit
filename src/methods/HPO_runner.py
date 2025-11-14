# src/methods/HPO_runner.py
"""
Run all enabled methods on a single dataset with and without HPO.

This module orchestrates running the complete benchmark pipeline, executing
all enabled methods twice: once with default hyperparameters and once with
hyperparameter optimization.

The underlying all_methods_runner handles edge cases automatically - methods
that don't benefit from HPO (TabPFN, dummy, etc.) run only once with their
results shared between both configurations to avoid redundant computation.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path

from src.methods.all_methods_runner import run_dataset


def run_hpo_comparison(
    *,
    task: str,
    dataset: str,
    test_size: float,
    val_size: float,
    cv_splits: int,
    seed: int,
    row_limit: Optional[int] = None,
    sampling: Optional[float] = None,
    categorical_encoding: Optional[str] = None,
    numerical_encoding: Optional[str] = None,
    normalization: Optional[str] = None,
    num_nan_policy: Optional[str] = None,
    cat_nan_policy: Optional[str] = None,
    max_epoch: int = 200,
    batch_size: int = 1024,
    n_trials: int = 100,
    early_stopping: bool = True,
    early_stopping_patience: int = 10,
    evaluate_option: str = "best-val",
    model_config: Optional[dict] = None,
    fit_config: Optional[dict] = None,
    config_dir: Optional[Path] = None,
    verbose: bool = False,
    clean_temp_dir: bool = True,
) -> Dict[str, Dict[str, Dict[int, Dict[str, Any]]]]:
    """
    Run all enabled methods on a single dataset with both HPO and default configs.
    
    Executes the complete benchmarking pipeline by running all methods twice:
    1. NO_HPO: Methods run with default hyperparameters from TALENT
    2. HPO: Methods run with Optuna-optimized hyperparameters
    
    Smart handling for pre-trained/simple methods:
    - Methods like TabPFN (pre-trained) and dummy classifiers don't benefit from HPO
    - These methods run only once and share results between NO_HPO and HPO keys
    - This avoids redundant computation and TALENT assertion errors
    
    Args:
        task: Task type ('pd' for classification, 'lgd' for regression)
        dataset: Dataset name
        test_size: Test set fraction
        val_size: Validation set fraction
        cv_splits: Number of cross-validation folds
        seed: Random seed for reproducibility
        row_limit: Optional limit on dataset rows (useful for testing)
        sampling: Optional sampling fraction for downsampling
        categorical_encoding: Categorical encoding policy
        numerical_encoding: Numerical encoding policy
        normalization: Normalization method
        num_nan_policy: Numerical NaN handling strategy
        cat_nan_policy: Categorical NaN handling strategy
        max_epoch: Maximum training epochs for deep methods
        batch_size: Batch size for deep methods
        n_trials: Number of Optuna HPO trials
        early_stopping: Whether to use early stopping
        early_stopping_patience: Patience epochs for early stopping
        evaluate_option: Which model to evaluate ('best-val' or 'last')
        model_config: Custom model hyperparameters
        fit_config: Custom fit configuration
        config_dir: Custom directory for storing HPO configs
        verbose: Whether to print detailed progress information
        clean_temp_dir: Whether to clean up temporary directories after run
        
    Returns:
        Nested dictionary with structure:
        {
            'NO_HPO': {
                method_name: {fold_id: {y_true, y_pred, y_prob, metrics, ...}, ...},
                ...
            },
            'HPO': {
                method_name: {fold_id: {y_true, y_pred, y_prob, metrics, ...}, ...},
                ...
            }
        }
        
    Example:
        >>> results = run_hpo_comparison(
        ...     task='pd',
        ...     dataset='0001.gmsc',
        ...     test_size=0.2,
        ...     val_size=0.2,
        ...     cv_splits=5,
        ...     seed=42,
        ...     verbose=True
        ... )
        >>> # Compare performance with/without HPO
        >>> xgb_default = results['NO_HPO']['xgboost'][1]['metrics']
        >>> xgb_tuned = results['HPO']['xgboost'][1]['metrics']
        >>> # TabPFN results are identical (pre-trained model)
        >>> assert results['NO_HPO']['tabpfn'] == results['HPO']['tabpfn']
    """
    
    # Common parameters shared across both HPO and NO_HPO runs
    common_params = {
        'task': task,
        'dataset': dataset,
        'test_size': test_size,
        'val_size': val_size,
        'cv_splits': cv_splits,
        'seed': seed,
        'row_limit': row_limit,
        'sampling': sampling,
        'categorical_encoding': categorical_encoding,
        'numerical_encoding': numerical_encoding,
        'normalization': normalization,
        'num_nan_policy': num_nan_policy,
        'cat_nan_policy': cat_nan_policy,
        'max_epoch': max_epoch,
        'batch_size': batch_size,
        'n_trials': n_trials,
        'early_stopping': early_stopping,
        'early_stopping_patience': early_stopping_patience,
        'evaluate_option': evaluate_option,
        'model_config': model_config,
        'fit_config': fit_config,
        'config_dir': config_dir,
        'verbose': verbose,
        'clean_temp_dir': clean_temp_dir,
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running HPO comparison for dataset: {dataset} ({task.upper()})")
        print(f"{'='*80}")
    
    # Run all methods with default hyperparameters (NO_HPO)
    if verbose:
        print(f"\n[PHASE 1/2] Running all methods with default configurations...")
    
    no_hpo_results = run_dataset(**common_params, tune=False)
    
    # Run all methods with hyperparameter optimization (HPO)
    # Note: Methods that don't support HPO (TabPFN, dummy, etc.) will automatically
    # run with defaults instead, and their results will be identical to NO_HPO
    if verbose:
        print(f"\n[PHASE 2/2] Running all methods with hyperparameter optimization...")
    
    hpo_results = run_dataset(**common_params, tune=True)
    
    # Combine results into final structure
    results = {
        'NO_HPO': no_hpo_results,
        'HPO': hpo_results
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"HPO comparison complete for {dataset}")
        print(f"Methods evaluated: {list(no_hpo_results.keys())}")
        print(f"{'='*80}\n")
    
    return results