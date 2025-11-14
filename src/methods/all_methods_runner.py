# src/methods/all_methods_runner.py
"""
Run all enabled methods on a single dataset.

This module provides a unified interface for running multiple TALENT methods
on a dataset. Methods are configured in CONFIG_METHOD.yaml - only enabled 
methods will be executed.

Intelligent HPO handling: If tune=True is requested but a method doesn't 
benefit from hyperparameter optimization (e.g., TabPFN, dummy classifiers),
the method automatically runs with default parameters instead.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path

from src.utils.config_reader import load_config
from src.methods.method_runner import run_talent_method, supports_hpo


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
    n_trials: int = 100,
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
    
    Automatically handles methods that don't support HPO: if tune=True is requested
    for a method like TabPFN (pre-trained) or dummy (too simple), the method runs 
    with default parameters instead to avoid errors.
    
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
        tune: Whether to perform HPO (auto-disabled for incompatible methods)
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
        Dictionary mapping method names to their results:
        {method_name: {fold_id: {y_true, y_pred, y_prob, metrics, ...}, ...}, ...}
    """
    
    # Load configuration to get enabled methods
    config = load_config()
    enabled_methods = list(config['methods'][task.lower()].keys())
    
    # Initialize results dictionary
    results = {}
    
    # Run each enabled method
    for method in enabled_methods:
        # Determine if HPO should be used for this specific method
        # Methods like TabPFN (pre-trained) or dummy (too simple) don't benefit from HPO
        # and will cause assertion errors in TALENT if tune=True is passed
        method_tune = tune and supports_hpo(method)
        
        # Run the method with appropriate HPO setting
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
            tune=method_tune,  # Automatically set based on method capabilities
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
        
        # Store results for this method
        results[method] = method_results
    
    return results