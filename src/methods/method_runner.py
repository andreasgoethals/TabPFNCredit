"""
TALENT-compatible method runner for TabPFNCredit benchmarking.

This module provides a unified interface for running both classical and deep learning
methods through TALENT's API, properly handling argument parsing, configuration,
and data preparation.

Key features:
- Supports all TALENT methods (classical ML and deep learning)
- Handles cross-validation with proper fold isolation
- HPO (hyperparameter optimization) support with config persistence across folds
- Method-specific preprocessing policy enforcement
- Proper cleanup of temporary directories
- Persistent config storage for HPO reuse
- Organized config storage by task type (pd/lgd)

Architecture notes:
- TALENT was designed as CLI scripts, not a library, so we manipulate sys.argv
- Each method has strict preprocessing requirements that must be satisfied
- HPO configs are saved as {method}-tuned.json in persistent config directory
- Configs are stored in project's 'config_hpo' folder organized by task type
- DataFeeder returns fold IDs starting at 1 (not 0), so we handle this correctly
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import sys
import time
import inspect
import shutil
import contextlib
import warnings
from io import StringIO
from pathlib import Path
import tempfile
import numpy as np
import copy
import json
import os

# TALENT core utilities
from TALENT.model.utils import (
    get_deep_args,
    get_classical_args,
    get_method,
    set_seeds,
    tune_hyper_parameters,  
)

# Local data loader
from src.data.data_feeder import DataFeeder


# ======================================================================================
#                               CONFIGURATION
# ======================================================================================
# Deep learning methods - EXACT NAMES AS TALENT EXPECTS THEM
DEEP_METHODS = {
    "mlp", "tabnet", "tabpfn", "PFN-v2",
    "resnet", "node", "ftt", "tabptm", "tabr",
    "saint", "tabtransformer", "grownet", "autoint",
    "snn", "danets", "tabcaps", "dcn2", "tangos",
    "ptarl", "switchtab", "dnnr", "modernNCA",
    "hyperfast", "bishop", "realmlp", "protogate",
    "mlp_plr", "excelformer", "grande", "amformer",
    "trompt", "tabm", "t2gformer", "tabautopnpnet", "tabicl"
}

# Classical methods - EXACT NAMES AS TALENT EXPECTS THEM
CLASSICAL_METHODS = {
    "xgboost", "catboost", "lightgbm", "RandomForest",
    "LogReg", "LinearRegression", "knn", "svm",
    "NaiveBayes", "NCM", "dummy"
}

# Methods that don't benefit from HPO (pre-trained or too simple)
NO_HPO_METHODS = {
    'tabpfn', 'PFN-v2', 'dummy', 'NCM', 
    'NaiveBayes', 'LinearRegression'
}


# ======================================================================================
#                          PROJECT ROOT DETECTION
# ======================================================================================

def _find_project_root() -> Path:
    """
    Find the project root directory dynamically.
    
    Searches upward from the current file location until it finds a directory
    containing markers that indicate the project root (like 'src' folder,
    'setup.py', 'pyproject.toml', etc.).
    
    Returns:
        Path to project root directory
        
    Raises:
        RuntimeError: If project root cannot be found
    """
    # Start from this file's directory
    current = Path(__file__).resolve().parent
    
    # Root markers to look for
    root_markers = {
        'src',           # Source folder
        'setup.py',      # Setup file
        'pyproject.toml',# Modern Python project config
        'README.md',     # README
        '.git',          # Git repository
    }
    
    # Search upward (max 10 levels)
    for _ in range(10):
        # Check if any marker exists in current directory
        if any((current / marker).exists() for marker in root_markers):
            return current
        
        # Move up one level
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent
    
    # Fallback: use current working directory
    return Path.cwd()


# Cache the project root to avoid repeated searches
_PROJECT_ROOT = _find_project_root()


def get_default_config_dir() -> Path:
    """
    Get the default directory for storing HPO configs.
    
    Returns:
        Path to {project_root}/config_hpo directory
    """
    config_dir = _PROJECT_ROOT / "config_hpo"
    config_dir.mkdir(exist_ok=True)
    return config_dir


# ======================================================================================
#                               INTERNAL UTILITIES
# ======================================================================================

@contextlib.contextmanager
def _silence(enabled: bool = True):
    """
    Context manager to suppress stdout/stderr and warnings.
    
    Used to reduce noise from TALENT's internal logging, especially during
    argument parsing and model initialization.
    
    Args:
        enabled: If True, suppress output. If False, leave output as is.
    """
    if not enabled:
        yield
        return
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


@contextlib.contextmanager
def _suppress_all_output(enabled: bool = True):
    """
    Completely suppress ALL output including print statements, progress bars, and warnings.
    
    More aggressive than _silence(). Redirects to devnull to catch everything.
    Used during model training and HPO when verbose=False.
    
    Args:
        enabled: If True, suppress output. If False, leave output as is.
    """
    if not enabled:
        yield
        return
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    devnull = open(os.devnull, 'w')
    
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        devnull.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def _inject_configs(
    args: Any,
    model_config: Optional[dict],
    fit_config: Optional[dict],
    verbose: bool
) -> None:
    """
    Inject user-provided model and fit configurations into args.
    
    This allows users to override default hyperparameters without running full HPO.
    For example, specifying a specific learning rate or number of estimators.
    
    The injected configs are merged with TALENT's default configs, with user
    values taking precedence.
    
    Args:
        args: Argument namespace from TALENT (modified in-place)
        model_config: Model-specific hyperparameters (e.g., hidden sizes, dropout)
        fit_config: Training configuration (e.g., learning rate, weight decay)
        verbose: Whether to suppress verbose output during training
    """
    # Initialize config dict if needed
    if not hasattr(args, 'config') or args.config is None:
        args.config = {}
    
    # Inject model config
    if model_config:
        if 'model' not in args.config:
            args.config['model'] = {}
        args.config['model'].update(model_config)
    
    # Inject fit config
    if fit_config:
        if 'fit' not in args.config:
            args.config['fit'] = {}
        args.config['fit'].update(fit_config)
    
    # Ensure verbose setting for fit
    if 'fit' not in args.config:
        args.config['fit'] = {}
    if not verbose:
        args.config['fit']['verbose'] = False


def _sanitize_sklearn_params(estimator_class, params: dict) -> dict:
    """
    Remove parameters that are not valid for sklearn estimator.
    
    This prevents errors when users provide extra parameters that sklearn doesn't recognize.
    We inspect the estimator's __init__ signature and filter out invalid parameters.
    
    Args:
        estimator_class: Sklearn estimator class
        params: Parameter dictionary
        
    Returns:
        Filtered parameter dictionary with only valid parameters
    """
    if not params or estimator_class is None:
        return params
    
    try:
        # Get valid parameters from __init__ signature
        sig = inspect.signature(estimator_class.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        
        # Filter params
        filtered = {k: v for k, v in params.items() if k in valid_params}
        
        # Warn about dropped params
        dropped = set(params.keys()) - set(filtered.keys())
        if dropped:
            warnings.warn(
                f"Dropped invalid parameters for {estimator_class.__name__}: {dropped}"
            )
        
        return filtered
        
    except Exception as e:
        warnings.warn(f"Could not sanitize parameters: {e}")
        return params


def _sanitize_classical_params(args: Any, method: str) -> None:
    """
    Sanitize model parameters for classical (sklearn-based) methods.
    
    Different sklearn models accept different parameters. This function ensures
    only valid parameters are passed to each specific model.
    
    For example, RandomForest accepts 'n_estimators' but not 'learning_rate',
    while XGBoost accepts both.
    
    Modifies args.config['model'] in-place.
    
    Args:
        args: Argument namespace (modified in-place)
        method: Method name (e.g., 'RandomForest', 'xgboost')
    """
    if not hasattr(args, 'config') or not isinstance(args.config, dict):
        return
    
    if 'model' not in args.config:
        return
    
    params = args.config['model']
    is_regression = getattr(args, 'is_regression', False)
    
    # Map method to sklearn estimator class
    estimator_class = None
    
    try:
        if method == "RandomForest":
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            estimator_class = RandomForestRegressor if is_regression else RandomForestClassifier
            
        elif method == "knn":
            from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
            estimator_class = KNeighborsRegressor if is_regression else KNeighborsClassifier
            
        elif method == "svm":
            from sklearn.svm import SVR, SVC
            estimator_class = SVR if is_regression else SVC
            
        elif method == "LogReg":
            from sklearn.linear_model import LogisticRegression
            estimator_class = LogisticRegression
            
        elif method == "LinearRegression":
            from sklearn.linear_model import LinearRegression
            estimator_class = LinearRegression
            
        elif method == "NaiveBayes":
            from sklearn.naive_bayes import GaussianNB
            estimator_class = GaussianNB
        
        # Sanitize if we found the class
        if estimator_class is not None:
            args.config['model'] = _sanitize_sklearn_params(estimator_class, params)
            
    except ImportError:
        pass  # sklearn not available or module import failed


def _cleanup_temp_directories(tmp_dir: Path, clean: bool = True) -> None:
    """
    Clean up temporary directories and any stray results folders.
    
    TALENT sometimes creates output directories in the current working directory.
    This function ensures everything is cleaned up properly.
    
    Args:
        tmp_dir: Temporary directory to remove
        clean: If True, remove directories. If False, leave as-is (useful for debugging).
    """
    if not clean:
        return
    
    # Remove temp directory
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    
    # Remove common stray directories in current working directory
    cwd = Path.cwd()
    stray_dirs = [
        cwd / "results_model",
        cwd / "results",
        cwd / "checkpoints",
        cwd / "models",  
    ]
    
    for d in stray_dirs:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


def _cleanup_checkpoints_from_config_dir(config_dir: Path) -> None:
    """
    Remove model checkpoint files and training logs from config directory.
    
    TALENT saves both configs AND checkpoints/logs to args.save_path.
    We only want to keep the JSON config files.
    
    Files removed:
    - *.pth: PyTorch model checkpoints
    - *.pkl: Pickle model files
    - *.npy: NumPy arrays (e.g., cluster centers)
    - trlog: Training log files
    
    Args:
        config_dir: Directory to clean
    """
    if not config_dir.exists():
        return
    
    # Patterns for files to remove
    unwanted_patterns = [
        '*.pth',       # PyTorch model files
        '*.pkl',       # Pickle model files  
        '*.npy',       # NumPy array files
        '*-*.pth',     # Files like "best-val-42.pth"
        '*-*.pkl',     # Files like "model-42.pkl"
    ]
    
    for pattern in unwanted_patterns:
        for file in config_dir.rglob(pattern):
            try:
                file.unlink()
            except Exception:
                pass  # Ignore errors during cleanup
    
    # Also remove 'trlog' files (no extension)
    for file in config_dir.rglob('*'):
        if file.is_file() and file.name == 'trlog':
            try:
                file.unlink()
            except Exception:
                pass


def _parse_prediction_output(output: Any) -> Tuple[Optional[float], Any, Optional[list], Any]:
    """
    Parse TALENT's prediction output.
    
    TALENT methods return predictions in various formats:
    - Just predictions: output
    - (metrics, predictions): tuple of 2
    - (metrics, metric_names, predictions): tuple of 3
    - (val_loss, metrics, metric_names, predictions): tuple of 4+
    
    Returns:
        Tuple of (val_loss, metrics, metric_names, predictions)
        where metric_names is always a list of strings
    """
    val_loss = None
    metrics = None
    metric_names = None
    predictions = None
    
    if isinstance(output, tuple):
        if len(output) >= 4:
            val_loss, metrics, metric_names, predictions = output[:4]
        elif len(output) == 3:
            metrics, metric_names, predictions = output
        elif len(output) == 2:
            metrics, predictions = output
        elif len(output) == 1:
            predictions = output[0]
    else:
        predictions = output
    
    # Ensure metric_names is a list of strings
    if metric_names is None:
        metric_names = []
    elif isinstance(metric_names, str):
        metric_names = [metric_names]
    elif not isinstance(metric_names, list):
        metric_names = list(metric_names)
    
    return val_loss, metrics, metric_names, predictions


# Sentinel values that indicate "missing" or "not specified"
_MISSING_SENTINELS = {None, "", "nothing", "Nothing", "NONE", "None"}


def _is_missing(x) -> bool:
    """
    Check if a value represents "missing" or "not specified".
    
    Used to determine whether a user explicitly provided a preprocessing option
    or if we should use defaults.
    
    Args:
        x: Value to check
        
    Returns:
        True if value is considered missing
    """
    try:
        return x in _MISSING_SENTINELS
    except TypeError:
        return False


def _apply_preprocessing_policies(args, method: str, user_specified: dict[str, bool]) -> None:
    """
    Apply preprocessing policy defaults and method-specific requirements.
    Uses EXACT method names as TALENT expects them - no case conversion!
    """
    
    # =============================================================================
    # STEP 1: Fill project defaults for missing values
    # =============================================================================
    if _is_missing(getattr(args, 'cat_policy', None)):
        args.cat_policy = 'ordinal'
    if _is_missing(getattr(args, 'num_policy', None)):       
        args.num_policy = 'none'
    if _is_missing(getattr(args, 'normalization', None)):    
        args.normalization = 'standard'
    if _is_missing(getattr(args, 'num_nan_policy', None)):   
        args.num_nan_policy = 'mean'
    if _is_missing(getattr(args, 'cat_nan_policy', None)):   
        args.cat_nan_policy = 'new'

    # =============================================================================
    # STEP 2: Apply method-specific preprocessing requirements - EXACT NAMES
    # =============================================================================
    
    # Method groups with specific requirements - EXACT NAMES
    requires_indices = {
        'amformer', 'autoint', 'bishop', 'dcn2', 'ftt', 'grande', 'grownet',
        'hyperfast', 'ptarl', 'realmlp', 'saint', 'snn',
        't2gformer', 'tabm', 'tabtransformer', 'trompt'
    }
    requires_tabr_ohe = {'modernNCA', 'tabr', 'mlp_plr', 'tabautopnpnet'}
    forbids_indices = {
        'mlp', 'resnet', 'switchtab', 'danets', 'dnnr', 'excelformer',
        'node', 'protogate', 'tabcaps', 'tabnet', 'tangos'
    }
    requires_no_normalization = {'hyperfast', 'tabicl'}
    requires_no_num_encoding = {
        'hyperfast', 'modernNCA', 'tabicl', 'tabptm', 'tabr'
    }

    # TabPFN and PFN-v2 - EXACT NAMES
    if method in {'tabpfn', 'PFN-v2'}:
        if user_specified.get('cat_policy', False):
            if args.cat_policy != 'indices':
                raise ValueError(f"{method} requires cat_policy='indices' but got '{args.cat_policy}'")
        else:
            args.cat_policy = 'indices'

        if user_specified.get('normalization', False):
            if args.normalization != 'none':
                raise ValueError(f"{method} requires normalization='none' but got '{args.normalization}'")
        else:
            args.normalization = 'none'

        if user_specified.get('num_policy', False):
            if args.num_policy != 'none':
                raise ValueError(f"{method} requires num_policy='none' but got '{args.num_policy}'")
        else:
            args.num_policy = 'none'

    # TabPTM - EXACT NAME
    elif method == 'tabptm':
        if user_specified.get('cat_policy', False):
            if args.cat_policy != 'ohe':
                raise ValueError(f"{method} requires cat_policy='ohe' but got '{args.cat_policy}'")
        else:
            args.cat_policy = 'ohe'

        if user_specified.get('normalization', False):
            if args.normalization != 'standard':
                raise ValueError(f"{method} requires normalization='standard' but got '{args.normalization}'")
        else:
            args.normalization = 'standard'

        if user_specified.get('num_policy', False):
            if args.num_policy != 'none':
                raise ValueError(f"{method} requires num_policy='none' but got '{args.num_policy}'")
        else:
            args.num_policy = 'none'

    # Methods requiring 'indices'
    elif method in requires_indices:
        if user_specified.get('cat_policy', False):
            if args.cat_policy != 'indices':
                raise ValueError(f"{method} requires cat_policy='indices' but got '{args.cat_policy}'")
        else:
            args.cat_policy = 'indices'

    # Methods requiring 'tabr_ohe'
    elif method in requires_tabr_ohe:
        if user_specified.get('cat_policy', False):
            if args.cat_policy != 'tabr_ohe':
                raise ValueError(f"{method} requires cat_policy='tabr_ohe' but got '{args.cat_policy}'")
        else:
            args.cat_policy = 'tabr_ohe'

        if user_specified.get('num_policy', False):
            if args.num_policy != 'none':
                raise ValueError(f"{method} requires num_policy='none' but got '{args.num_policy}'")
        else:
            args.num_policy = 'none'

    # Methods forbidding 'indices'
    elif method in forbids_indices:
        if user_specified.get('cat_policy', False):
            if args.cat_policy == 'indices':
                raise ValueError(f"{method} does not support cat_policy='indices'")
        else:
            if args.cat_policy == 'indices':
                args.cat_policy = 'ordinal'

    # Methods requiring no normalization
    if method in requires_no_normalization:
        if user_specified.get('normalization', False):
            if args.normalization != 'none':
                raise ValueError(f"{method} requires normalization='none' but got '{args.normalization}'")
        else:
            args.normalization = 'none'

    # Methods requiring no numerical encoding
    if method in requires_no_num_encoding:
        if user_specified.get('num_policy', False):
            if args.num_policy != 'none':
                raise ValueError(f"{method} requires num_policy='none' but got '{args.num_policy}'")
        else:
            args.num_policy = 'none'

# ======================================================================================
#                               MAIN RUN FUNCTION
# ======================================================================================
def run_talent_method(
    *,
    task: str,
    dataset: str,
    test_size: float,
    val_size: float,
    cv_splits: int,
    seed: int,
    row_limit: Optional[int] = None,
    sampling: Optional[float] = None,
    method: str,
    categorical_encoding: Optional[str] = None,
    numerical_encoding: Optional[str] = None,
    normalization: Optional[str] = None,
    num_nan_policy: Optional[str] = None,
    cat_nan_policy: Optional[str] = None,
    max_epoch: int = 100,
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
) -> Dict[int, Dict[str, Any]]:
    """
    Run a TALENT method on credit risk data with proper preprocessing and configuration.
    
    This is the main entry point for running any TALENT method. It handles:
    - Data loading and splitting into cross-validation folds
    - Preprocessing policy application (with method-specific requirements)
    - Method instantiation and training
    - Cross-validation loop with proper fold isolation
    - Optional hyperparameter optimization (HPO) on first fold, reusing config for others
    - Persistent config storage for future reuse across runs
    - Proper cleanup of temporary directories
    
    HPO Configuration Behavior:
    ┌─────────────┬──────────────────────────────────────────────────────────┐
    │   Setting   │                       Behavior                           │
    ├─────────────┼──────────────────────────────────────────────────────────┤
    │ tune=True   │ Fold 1: Run HPO, save config to config_hpo/{task}/      │
    │             │ Fold 2+: Load saved config from fold 1                  │
    │             │                                                          │
    │ tune=False  │ All folds: Use TALENT's default hyperparameters         │
    │             │ (Never loads saved configs, always uses defaults)       │
    └─────────────┴──────────────────────────────────────────────────────────┘
    
    Config Storage:
    - Configs are stored at: {project_root}/config_hpo/{task}/{dataset}/{method}-tuned.json
    - Task is 'pd' for classification or 'lgd' for regression
    - Each dataset has its own folder for organization
    - Configs persist across runs for reproducibility
    - tune=False always ignores saved configs
    
    Architecture notes:
    - CV is implemented outside TALENT (in DataFeeder), so we manually handle fold iteration
    - DataFeeder returns fold IDs starting at 1 (not 0), so we detect the first fold dynamically
    - When tune=True, HPO runs only on the first fold and saves config
    - Subsequent folds automatically load and reuse the optimized config (ONLY when tune=True)
    
    Args:
        task: Task type ('pd' for classification, 'lgd' for regression)
        dataset: Dataset name (must exist in data directory)
        test_size: Test set fraction (0.0 to 1.0)
        val_size: Validation set fraction (0.0 to 1.0)
        cv_splits: Number of cross-validation folds
        seed: Random seed for reproducibility
        row_limit: Optional limit on dataset rows (useful for quick testing)
        sampling: Optional sampling fraction (downsampling for large datasets)
        method: TALENT method name (must be canonical name, e.g., 'xgboost' not 'XGBoost')
        categorical_encoding: Categorical encoding policy (None = use method default)
        numerical_encoding: Numerical encoding policy (None = use method default)
        normalization: Normalization method (None = use method default)
        num_nan_policy: Numerical NaN handling (None = use method default)
        cat_nan_policy: Categorical NaN handling (None = use method default; 'new' is required by TALENT)
        max_epoch: Maximum training epochs (deep methods only)
        batch_size: Batch size (deep methods only)
        tune: Whether to perform hyperparameter optimization on first fold
        n_trials: Number of HPO trials (if tune=True)
        early_stopping: Whether to use early stopping (deep methods only)
        early_stopping_patience: Patience for early stopping
        evaluate_option: Which model to use for evaluation ('best-val', 'last')
        model_config: Custom model hyperparameters (overrides defaults, not HPO)
        fit_config: Custom fit configuration (overrides defaults, not HPO)
        config_dir: Custom directory for storing HPO configs (default: {project_root}/config_hpo)
        verbose: Whether to print detailed progress information
        clean_temp_dir: Whether to clean up temporary directories after run
        
    Returns:
        Dictionary mapping fold_id to results dict containing:
            - y_true: Ground truth labels/values (np.array)
            - y_pred: Model predictions (np.array) 
            - metrics: Performance metrics (list or dict)
            - metric_names: List of metric names (list[str])
            - primary_metric: Name of primary metric (str)
            - val_loss: Validation loss (float or None)
            - train_time: Training time in seconds (float)
            - info: Dataset information (dict)
            - method, dataset, task, fold_id: Metadata
            - used_hpo: Whether HPO was used for this fold (bool)
            
    Raises:
        ValueError: If method requirements conflict with user-specified options
        RuntimeError: If training or prediction fails
        
    Example:
        >>> # Run with HPO - optimized config saved for this dataset+method
        >>> results_hpo = run_talent_method(
        ...     task='pd',
        ...     dataset='0014.hmeq',
        ...     test_size=0.2,
        ...     val_size=0.2,
        ...     cv_splits=5,
        ...     seed=42,
        ...     method='xgboost',
        ...     tune=True,  # Fold 1 runs HPO, fold 2-5 reuse config
        ...     n_trials=100,
        ... )
        >>> 
        >>> # Run without HPO - always uses defaults (ignores saved config)
        >>> results_default = run_talent_method(
        ...     task='pd',
        ...     dataset='0014.hmeq',
        ...     test_size=0.2,
        ...     val_size=0.2,
        ...     cv_splits=5,
        ...     seed=42,
        ...     method='xgboost',
        ...     tune=False,  # Uses TALENT defaults, ignores saved config
        ... )
    """
    
    # Determine task type and method category
    is_regression = (task.lower() == "lgd")
    is_deep = method in DEEP_METHODS
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Running {method} ({'deep' if is_deep else 'classical'}) on {dataset} ({task.upper()})")
        print(f"{'='*70}")
    
    # -------------------------------------------------------------------------
    # Track which preprocessing options user explicitly provided
    # -------------------------------------------------------------------------
    user_specified = {
        'cat_policy': not _is_missing(categorical_encoding),
        'num_policy': not _is_missing(numerical_encoding),
        'normalization': not _is_missing(normalization),
        'num_nan_policy': not _is_missing(num_nan_policy),
        'cat_nan_policy': not _is_missing(cat_nan_policy),
    }
    
    # -------------------------------------------------------------------------
    # Prepare data with cross-validation folds
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\nPreparing data with {cv_splits} CV splits...")
    
    feeder = DataFeeder(
        task=task,
        dataset=dataset,
        test_size=test_size,
        val_size=val_size,
        cv_splits=cv_splits,
        seed=seed,
        row_limit=row_limit,
        sampling=sampling,
    )
    folds = feeder.prepare()
    
    # CRITICAL: Determine the first fold ID dynamically
    first_fold_id = min(folds.keys())
    
    if verbose:
        print(f"Fold IDs: {sorted(folds.keys())}")
        print(f"First fold ID: {first_fold_id}")
    
    # -------------------------------------------------------------------------
    # Initialize results storage
    # -------------------------------------------------------------------------
    results: Dict[int, Dict[str, Any]] = {}
    
    # -------------------------------------------------------------------------
    # Setup directories: Organized by task (pd/lgd), then dataset, then method files
    # 
    # Structure: {project_root}/config_hpo/{task}/{dataset}/{method}-tuned.json
    # Example:   ~/TabPFNCredit/config_hpo/pd/0014.hmeq/xgboost-tuned.json
    #                                                     /lightgbm-tuned.json
    #                                                     /catboost-tuned.json
    #            ~/TabPFNCredit/config_hpo/lgd/0001.heloc/xgboost-tuned.json
    # -------------------------------------------------------------------------
    if config_dir is None:
        # Default: Use project's config_hpo directory, organized by task then dataset
        base_config_dir = get_default_config_dir()
        dataset_config_dir = base_config_dir / task.lower() / dataset
    else:
        dataset_config_dir = Path(config_dir) / task.lower() / dataset
    
    # Create the dataset config directory
    dataset_config_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temp directory for model checkpoints (will be cleaned up)
    checkpoint_tmp_dir = Path(tempfile.mkdtemp(prefix=f"talent_ckpt_{dataset}_{method}_"))
    
    if verbose:
        print(f"\nDirectory setup:")
        print(f"  Config directory (persistent): {dataset_config_dir}")
        print(f"  Checkpoint directory (temp):   {checkpoint_tmp_dir}")
        
        # Show HPO behavior
        if tune:
            print(f"\n[HPO] Mode: ENABLED")
            print(f"[HPO] Fold 1: Will run HPO and save config")
            print(f"[HPO] Fold 2+: Will load config from fold 1")
        else:
            print(f"\n[HPO] Mode: DISABLED")
            print(f"[HPO] All folds: Will use TALENT's default hyperparameters")
    
    try:
        # Process each fold
        for fold_id, ((N, C, y), info) in folds.items():
            if verbose:
                print(f"\n{'='*70}")
                print(f"Fold {fold_id}/{len(folds)}")
                print(f"{'='*70}")
            
            # ---------------------------------------------------------------------
            # Get base arguments from TALENT
            # CRITICAL: Always silence to prevent pprint(vars(args)) output
            # ---------------------------------------------------------------------
            orig_argv = sys.argv.copy()
            
            try:
                sys.argv = [
                    "train.py",
                    "--model_type", method,
                    "--dataset", dataset,
                    "--dataset_path", "./data",
                    "--model_path", str(checkpoint_tmp_dir),
                ]
                
                # CRITICAL: Always silence get_*_args() because it calls pprint()
                with _suppress_all_output(True):
                    if is_deep:
                        args, default_para, opt_space = get_deep_args()
                    else:
                        args, default_para, opt_space = get_classical_args()
            finally:
                sys.argv = orig_argv
            
            # ---------------------------------------------------------------------
            # Override TALENT's computed paths
            # ---------------------------------------------------------------------
            args.save_path = str(dataset_config_dir)  # Configs: {project}/config_hpo/{task}/{dataset}/
            args.model_path = str(checkpoint_tmp_dir)  # Checkpoints: /tmp/.../
            
            # ---------------------------------------------------------------------
            # Set random seed
            # ---------------------------------------------------------------------
            args.seed = seed
            set_seeds(seed)
            
            # ---------------------------------------------------------------------
            # Set preprocessing options
            # ---------------------------------------------------------------------
            args.is_regression = is_regression
            args.normalization = normalization
            args.num_nan_policy = num_nan_policy
            args.cat_nan_policy = cat_nan_policy
            args.cat_policy = categorical_encoding
            args.num_policy = numerical_encoding
            _apply_preprocessing_policies(args, method, user_specified)
            
            if verbose and fold_id == first_fold_id:
                print(f"\nPreprocessing configuration:")
                print(f"  - cat_policy: {args.cat_policy}")
                print(f"  - num_policy: {args.num_policy}")
                print(f"  - normalization: {args.normalization}")
                print(f"  - num_nan_policy: {args.num_nan_policy}")
                print(f"  - cat_nan_policy: {args.cat_nan_policy}")

            # ---------------------------------------------------------------------
            # Set HPO flags
            # ---------------------------------------------------------------------
            args.tune = tune  # Will only be used if we call tune_hyper_parameters()
            args.retune = False  # Never retune
            args.n_trials = n_trials
            args.evaluate_option = evaluate_option
            
            # ---------------------------------------------------------------------
            # Set method-specific parameters
            # ---------------------------------------------------------------------
            if is_deep:
                args.max_epoch = max_epoch
                args.batch_size = batch_size
                args.early_stopping = early_stopping
                args.early_stopping_patience = early_stopping_patience
                
                if not hasattr(args, 'config') or args.config is None:
                    args.config = {}
                if 'training' not in args.config:
                    args.config['training'] = {}
                args.config['training']['max_epoch'] = max_epoch
            
            # Inject custom configs
            if model_config or fit_config:
                _inject_configs(args, model_config, fit_config, verbose)
            
            # Set n_bins
            if not is_deep:
                if not hasattr(args, 'config') or args.config is None:
                    args.config = {}
                if 'fit' not in args.config:
                    args.config['fit'] = {}
                args.config['fit']['n_bins'] = getattr(args, 'n_bins', 2)
                _sanitize_classical_params(args, method)
            else:
                if 'training' not in args.config:
                    args.config['training'] = {}
                args.config['training']['n_bins'] = getattr(args, 'n_bins', 2)
            
            # ===================================================================
            # CRITICAL HPO LOGIC
            # 
            # tune=True:  Call tune_hyper_parameters() for ALL folds
            #             - Fold 1: Runs HPO, saves config
            #             - Fold 2+: Loads saved config
            #
            # tune=False: NEVER call tune_hyper_parameters()
            #             - All folds: Use default config from get_*_args()
            #             - Defaults are from: .venv/.../TALENT/configs/default/
            # ===================================================================
            if tune:
                tuned_config_path = dataset_config_dir / f"{method}-tuned.json"
                
                if fold_id == first_fold_id:
                    # First fold with tune=True
                    if tuned_config_path.exists() and not args.retune:
                        if verbose:
                            print(f"\n[HPO] Existing config found: {tuned_config_path.name}")
                            print(f"[HPO] Will load saved config (use retune=True to re-optimize)")
                    else:
                        if verbose:
                            print(f"\n[HPO] Running hyperparameter optimization...")
                            print(f"[HPO] Trials: {n_trials}")
                            print(f"[HPO] Will save to: {tuned_config_path}")
                else:
                    # Subsequent folds with tune=True
                    if verbose:
                        print(f"\n[HPO] Loading config from fold {first_fold_id}...")
                
                try:
                    train_val_data = (N, C, y)
                    
                    # Call TALENT's HPO function
                    # It handles both running HPO and loading existing configs
                    # Use aggressive suppression to prevent Optuna progress bars
                    with _suppress_all_output(not verbose):
                        args = tune_hyper_parameters(args, opt_space, train_val_data, info)
                    
                    # Verify
                    if fold_id == first_fold_id:
                        if tuned_config_path.exists():
                            if verbose:
                                print(f"[HPO] ✓ Config ready: {tuned_config_path.name}")
                        else:
                            raise RuntimeError(f"HPO failed to save config to {tuned_config_path}")
                    else:
                        if verbose:
                            print(f"[HPO] ✓ Config loaded successfully")
                            
                except Exception as e:
                    if verbose:
                        print(f"[HPO] ✗ Error: {e}")
                    raise
            
            else:
                # tune=False: Use defaults, never load saved configs
                if verbose and fold_id == first_fold_id:
                    print(f"\n[DEFAULT] Using TALENT's default hyperparameters")
                    print(f"[DEFAULT] Location: .venv/.../TALENT/configs/default/{method}.json")
                    print(f"[DEFAULT] Saved configs in config_hpo/ are ignored when tune=False")
            
            # ---------------------------------------------------------------------
            # Train model with aggressive output suppression
            # ---------------------------------------------------------------------
            try:
                with _suppress_all_output(not verbose):
                    MethodClass = get_method(args.model_type)
                    model = MethodClass(args, is_regression=is_regression)
                    
                    t0 = time.time()
                    train_time = model.fit((N, C, y), info, train=True)
                    if train_time is None:
                        train_time = time.time() - t0
                    
                    output = model.predict((N, C, y), info, model_name=args.evaluate_option)
                
                # Parse output
                val_loss, metrics, metric_names, predictions = _parse_prediction_output(output)
                
                # Store results
                results[fold_id] = {
                    "y_true": np.asarray(y["test"]),
                    "y_pred": np.asarray(predictions),
                    "metrics": metrics if isinstance(metrics, (dict, list, tuple)) else {},
                    "metric_names": metric_names,
                    "primary_metric": metric_names[0] if metric_names else None,
                    "val_loss": float(val_loss) if val_loss is not None else None,
                    "train_time": float(train_time),
                    "info": info,
                    "method": method,
                    "dataset": dataset,
                    "task": task,
                    "fold_id": fold_id,
                    "used_hpo": fold_id == first_fold_id and tune,
                }
                
                if verbose and metrics:
                    print(f"\nFold {fold_id} results:")
                    if isinstance(metrics, (list, tuple)):
                        for name, res in zip(metric_names, metrics):
                            if name:
                                print(f"  {name}: {res:.4f}")
                    else:
                        print(f"  {metrics}")
        
            except Exception as e:
                if verbose:
                    print(f"\nError during training: {e}")
                raise
    
    finally:
        # Clean up checkpoints from temp directory
        _cleanup_temp_directories(checkpoint_tmp_dir, clean=clean_temp_dir)
        
        # CRITICAL: Remove checkpoint files and training logs from config directory
        # TALENT saves both configs AND checkpoints to args.save_path
        # We only want to keep the JSON files
        _cleanup_checkpoints_from_config_dir(dataset_config_dir)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Completed {len(results)} folds for {method}")
        if tune:
            tuned_config_path = dataset_config_dir / f"{method}-tuned.json"
            print(f"\n[HPO] Config saved: {tuned_config_path}")
            print(f"[HPO] This config is ONLY used when tune=True")
            print(f"[HPO] When tune=False, defaults are always used")
        print(f"{'='*70}\n")
    
    return results


# ======================================================================================
#                               CONVENIENCE FUNCTIONS
# ======================================================================================

def get_available_methods() -> Dict[str, list]:
    """
    Get all available methods in TALENT.
    
    Returns:
        Dictionary with 'classical' and 'deep' lists of method names
        
    Example:
        >>> methods = get_available_methods()
        >>> print(f"Classical: {methods['classical']}")
        >>> print(f"Deep: {methods['deep']}")
    """
    return {
        "classical": sorted(CLASSICAL_METHODS),
        "deep": sorted(DEEP_METHODS),
    }


def validate_method(method: str) -> Tuple[str, bool]:
    """
    Validate that a method name is supported.
    
    Args:
        method: Method name to validate (must be TALENT canonical name)
        
    Returns:
        Tuple of (method_name, is_deep)
        
    Raises:
        ValueError: If method is not in TALENT's supported methods
        
    Example:
        >>> method, is_deep = validate_method("xgboost")
        >>> print(f"{method} is {'deep' if is_deep else 'classical'}")
    """
    if method in DEEP_METHODS:
        return method, True
    elif method in CLASSICAL_METHODS:
        return method, False
    else:
        all_methods = sorted(DEEP_METHODS | CLASSICAL_METHODS)
        raise ValueError(
            f"Unknown method: '{method}'. Must use TALENT canonical name.\n"
            f"Supported methods: {all_methods}"
        )


def supports_hpo(method: str) -> bool:
    """
    Check if a method supports meaningful hyperparameter optimization.
    
    Some methods are pre-trained (TabPFN) or too simple (dummy) to benefit from HPO.
    
    Args:
        method: TALENT method name
        
    Returns:
        True if method has meaningful HPO space, False otherwise
        
    Example:
        >>> if not supports_hpo('tabpfn'):
        ...     print("TabPFN is pre-trained, no HPO needed")
    """
    return method not in NO_HPO_METHODS