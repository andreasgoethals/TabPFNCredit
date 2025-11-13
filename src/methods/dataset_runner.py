# src/methods/dataset_runner.py
"""
Run all enabled methods on a single dataset.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path

from src.utils.config_reader import load_config
from src.methods.method_runner import run_talent_method


def run_dataset(
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
    tune: bool = False,
    n_trials: int = 50,
    early_stopping: bool = True,
    early_stopping_patience: int = 10,
    evaluate_option: str = "best-val",
    model_config: Optional[dict] = None,
    fit_config: Optional[dict] = None,
    config_dir: Optional[Path] = None,
    verbose: bool = False,
    clean_temp_dir: bool = True,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Run all enabled methods on a single dataset.
    
    Args:
        task: Task type ('pd' for classification, 'lgd' for regression)
        dataset: Dataset name
        test_size: Test set fraction
        val_size: Validation set fraction
        cv_splits: Number of cross-validation folds
        seed: Random seed
        row_limit: Optional limit on dataset rows
        sampling: Optional sampling fraction
        categorical_encoding: Categorical encoding policy
        numerical_encoding: Numerical encoding policy
        normalization: Normalization method
        num_nan_policy: Numerical NaN handling
        cat_nan_policy: Categorical NaN handling
        max_epoch: Maximum training epochs
        batch_size: Batch size
        tune: Whether to perform HPO
        n_trials: Number of HPO trials
        early_stopping: Whether to use early stopping
        early_stopping_patience: Patience for early stopping
        evaluate_option: Which model to use for evaluation
        model_config: Custom model hyperparameters
        fit_config: Custom fit configuration
        config_dir: Custom directory for storing HPO configs
        verbose: Whether to print detailed progress
        clean_temp_dir: Whether to clean up temporary directories
        
    Returns:
        Dictionary: {method_name: results_dict, ...}
    """
    
    config = load_config()
    
    enabled_methods = list(config['methods'][task.lower()].keys())
    
    results = {}
    
    for method in enabled_methods:
        method_results = run_talent_method(
            task=task,
            dataset=dataset,
            test_size=test_size,
            val_size=val_size,
            cv_splits=cv_splits,
            seed=seed,
            row_limit=row_limit,
            sampling=sampling,
            method=method,
            categorical_encoding=categorical_encoding,
            numerical_encoding=numerical_encoding,
            normalization=normalization,
            num_nan_policy=num_nan_policy,
            cat_nan_policy=cat_nan_policy,
            max_epoch=max_epoch,
            batch_size=batch_size,
            tune=tune,
            n_trials=n_trials,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            evaluate_option=evaluate_option,
            model_config=model_config,
            fit_config=fit_config,
            config_dir=config_dir,
            verbose=verbose,
            clean_temp_dir=clean_temp_dir,
        )
        
        results[method] = method_results
    
    return results