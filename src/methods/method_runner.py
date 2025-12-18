"""
TALENT-compatible method runner for TabPFNCredit benchmarking 

This module provides a unified interface for running both classical and deep learning
methods through TALENT's API, properly handling argument parsing, configuration,
and data preparation.

Key features:
- Supports all TALENT methods (classical ML and deep learning)
- Handles cross-validation with proper fold isolation
- HPO (hyperparameter optimization) support with config persistence across folds
- Method-specific preprocessing policy enforcement
- Proper cleanup of temporary directories
- Persistent config storage for accessing the HPO results later 
- Organized config storage by task type (pd/lgd)
- Handles CUDA tensor to numpy conversion
- Extracts probabilities from logits for classification tasks
- Automatic row limit capping for TabPFN (10k) and PFN-v2 (50k)
- Calculates comprehensive metrics internally (not relying on TALENT)
- Clips LGD predictions to [0, 1] range

Architecture notes:
- TALENT was designed as CLI scripts, not a library, so we manipulate sys.argv
- Each method has strict preprocessing requirements that must be satisfied
- HPO configs are saved as {method}-tuned.json in persistent config directory
- Configs are stored in project's 'config_hpo' folder organized by task type (pd/lgd)
- Deep learning methods return logits; classical methods return probabilities
- TabPFN and PFN-v2 have inherent dataset size limitations
- All metrics are calculated by us for consistency and control
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import sys
import time
import inspect
import shutil
import contextlib
import warnings
import logging
from io import StringIO
from pathlib import Path
import tempfile
import numpy as np
import os

# Suppress LightGBM warnings globally
warnings.filterwarnings('ignore', message='No further splits with positive gain')
warnings.filterwarnings('ignore', message='categorical_column')
os.environ['LIGHTGBM_VERBOSITY'] = '-1'

# Try to import torch for tensor detection
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None  # type: ignore

# TALENT core utilities
from TALENT.model.utils import (
    get_deep_args,
    get_classical_args,
    get_method,
    set_seeds,
    tune_hyper_parameters,  
)

# Local imports
from src.data.data_feeder import DataFeeder
from src.methods.method_config import (
    DEEP_METHODS,
    CLASSICAL_METHODS,
    NO_HPO_METHODS,
    LOGIT_METHODS,
    PROBABILITY_METHODS,
    METHOD_ROW_LIMITS,
    REQUIRES_CAT_INDICES,
    REQUIRES_CAT_TABR_OHE,
    REQUIRES_CAT_OHE,
    FORBIDS_CAT_INDICES,
    TABPFN_VARIANTS,
    REQUIRES_NO_NORMALIZATION,
    REQUIRES_NO_NUM_ENCODING,
    REQUIRES_STANDARD_NORMALIZATION,
    apply_preprocessing_policies,
    apply_method_row_limit,
    _is_missing,
)
from src.methods.method_metrics import (
    calculate_pd_metrics,
    calculate_lgd_metrics,
)

# Setup logger
logger = logging.getLogger(__name__)


# ======================================================================================
#                          TENSOR TO NUMPY CONVERSION
# ======================================================================================

def _ensure_numpy_array(arr) -> np.ndarray:
    """
    Convert array-like object to NumPy array, handling PyTorch CUDA tensors.
    
    Addresses "can't convert cuda:0 device type tensor to numpy" error by
    moving GPU tensors to CPU before conversion.
    
    Args:
        arr: Array-like object (numpy array, torch tensor, list, etc.)
        
    Returns:
        NumPy array on CPU memory
    """
    # Handle PyTorch tensors (CPU or CUDA)
    if _HAS_TORCH and isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    
    # Handle NumPy arrays (pass through)
    if isinstance(arr, np.ndarray):
        return arr
    
    # Handle nested structures (lists of tensors)
    if isinstance(arr, (list, tuple)):
        if len(arr) > 0 and _HAS_TORCH and isinstance(arr[0], torch.Tensor):
            return np.array([x.detach().cpu().numpy() for x in arr])
        return np.asarray(arr)
    
    # Fallback
    try:
        return np.asarray(arr)
    except Exception as e:
        raise TypeError(
            f"Cannot convert {type(arr).__name__} to NumPy array. "
            f"Expected: numpy.ndarray, torch.Tensor, list, or tuple. "
            f"Error: {e}"
        )


# ======================================================================================
#                    PROBABILITY EXTRACTION FROM LOGITS/PREDICTIONS
# ======================================================================================

def _extract_class_probabilities(
    predictions: np.ndarray,
    method: str,
    is_regression: bool
) -> Optional[np.ndarray]:
    """
    Extract class probabilities from model predictions.
    
    Deep learning methods typically return logits (unbounded values).
    Classical methods typically return probabilities (0-1 range).
    
    For binary classification:
    - If predictions are 2D with 2 columns: extract probability of positive class (column 1)
    - If predictions are 1D: interpret as probability of positive class
    
    Args:
        predictions: Model predictions (may be logits or probabilities)
        method: Method name to determine if logits or probabilities expected
        is_regression: Whether this is a regression task
        
    Returns:
        1D array of probabilities for positive class (classification only)
        None for regression tasks
    """
    if is_regression:
        return None
    
    returns_logits = method in LOGIT_METHODS
    
    # Handle 2D predictions: (n_samples, n_classes)
    if len(predictions.shape) == 2 and predictions.shape[1] >= 2:
        if returns_logits:
            # Apply softmax: p_i = exp(logit_i) / sum(exp(logit_j))
            exp_logits = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
            probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        else:
            probabilities = predictions
        
        return probabilities[:, 1]
    
    # Handle 1D predictions: (n_samples,) or (n_samples, 1)
    elif len(predictions.shape) == 1 or (len(predictions.shape) == 2 and predictions.shape[1] == 1):
        if len(predictions.shape) == 2:
            predictions = predictions.ravel()
        
        if returns_logits:
            # Apply sigmoid: p = 1 / (1 + exp(-logit))
            probabilities = 1.0 / (1.0 + np.exp(-predictions))
        else:
            # Check if values are in [0,1] range
            if np.all((predictions >= 0) & (predictions <= 1)):
                probabilities = predictions
            else:
                # Values outside [0,1] - treat as logits
                probabilities = 1.0 / (1.0 + np.exp(-predictions))
        
        return probabilities
    
    else:
        warnings.warn(
            f"Unexpected prediction shape {predictions.shape} for method {method}. "
            f"Cannot extract class probabilities."
        )
        return None


def _extract_binary_predictions(
    predictions: np.ndarray,
    y_prob: Optional[np.ndarray],
    threshold: float = 0.5
) -> np.ndarray:
    """
    Extract binary class predictions from model output.
    
    For classification, converts probabilities or raw predictions to binary 0/1.
    
    Args:
        predictions: Raw model predictions (may be 2D with class scores)
        y_prob: Class probabilities if already computed
        threshold: Classification threshold (default 0.5)
        
    Returns:
        1D array of binary predictions (0 or 1)
    """
    # If we have probabilities, threshold them
    if y_prob is not None:
        return (y_prob >= threshold).astype(np.int32)
    
    # Handle 2D predictions: argmax across classes
    if len(predictions.shape) == 2 and predictions.shape[1] >= 2:
        return np.argmax(predictions, axis=1).astype(np.int32)
    
    # Handle 1D predictions
    predictions = predictions.ravel()
    
    # Check if already binary
    unique_vals = np.unique(predictions)
    if len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0, 1])):
        return predictions.astype(np.int32)
    
    # Threshold continuous values
    return (predictions >= threshold).astype(np.int32)


def _clip_regression_predictions(
    predictions: np.ndarray,
    lower: float = 0.0,
    upper: float = 1.0
) -> Tuple[np.ndarray, int, int]:
    """
    Clip regression predictions to valid range for LGD.
    
    LGD (Loss Given Default) must be in [0, 1] range. This function clips
    predictions and reports how many were outside the valid range.
    
    Args:
        predictions: Raw regression predictions
        lower: Lower bound (default 0.0)
        upper: Upper bound (default 1.0)
        
    Returns:
        Tuple of:
        - clipped predictions (np.ndarray)
        - count of predictions clipped from below
        - count of predictions clipped from above
    """
    predictions = predictions.ravel()
    
    n_below = int(np.sum(predictions < lower))
    n_above = int(np.sum(predictions > upper))
    
    clipped = np.clip(predictions, lower, upper)
    
    return clipped, n_below, n_above


# ======================================================================================
#                          MONKEY PATCHING FOR SILENCE
# ======================================================================================

def _noop_pprint(x):
    """No-op replacement for pprint."""
    pass

def _patch_talent_pprint(enable_silence: bool = True):
    """
    Monkey-patch TALENT's pprint function to suppress output.
    
    TALENT calls pprint(vars(args)) in get_deep_args() and get_classical_args(),
    which prints the entire argument namespace during initialization.
    
    Args:
        enable_silence: If True, replace pprint with no-op. If False, restore original.
    """
    try:
        import TALENT.model.utils as utils
        if enable_silence:
            if not hasattr(_patch_talent_pprint, '_original_pprint'):
                _patch_talent_pprint._original_pprint = utils.pprint
            utils.pprint = _noop_pprint
        else:
            if hasattr(_patch_talent_pprint, '_original_pprint'):
                utils.pprint = _patch_talent_pprint._original_pprint
    except Exception:
        pass


def _fix_windows_compatibility():
    """
    Fix Windows compatibility issues for methods that use os.sysconf.
    
    PFN-v2 and other methods use os.sysconf which is Unix-only.
    Mock it with reasonable default values.
    """
    if not hasattr(os, 'sysconf'):
        def _mock_sysconf(name):
            if isinstance(name, str):
                if 'PAGE_SIZE' in name:
                    return 4096
                elif 'NPROCESSORS' in name:
                    return os.cpu_count() or 4
            return 0
        os.sysconf = _mock_sysconf
        os.sysconf_names = {'SC_PAGE_SIZE': 30, 'SC_NPROCESSORS_ONLN': 84}


# Apply fixes on module import
_fix_windows_compatibility()


# ======================================================================================
#                          PROJECT ROOT DETECTION
# ======================================================================================

def _find_project_root() -> Path:
    """
    Find the project root directory dynamically.
    
    Searches upward from the current file location for directories containing
    standard project markers (src folder, setup.py, etc.).
    
    Returns:
        Path to project root directory
        
    Raises:
        RuntimeError: If project root cannot be found
    """
    current = Path(__file__).resolve().parent
    
    root_markers = {
        'src',
        'setup.py',
        'pyproject.toml',
        'README.md',
        '.git',
    }
    
    # Search upward (max 10 levels)
    for _ in range(10):
        if any((current / marker).exists() for marker in root_markers):
            return current
        
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent
    
    # Fallback: use current working directory
    return Path.cwd()


# Cache the project root
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
    Completely suppress all output including print statements, progress bars, and warnings.
    
    More aggressive than _silence(). Redirects to devnull.
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
    
    Allows users to override default hyperparameters without running full HPO.
    
    Args:
        args: Argument namespace from TALENT (modified in-place)
        model_config: Model-specific hyperparameters
        fit_config: Training configuration
        verbose: Whether to suppress verbose output during training
    """
    if not hasattr(args, 'config') or args.config is None:
        args.config = {}
    
    if model_config:
        if 'model' not in args.config:
            args.config['model'] = {}
        args.config['model'].update(model_config)
    
    if fit_config:
        if 'fit' not in args.config:
            args.config['fit'] = {}
        args.config['fit'].update(fit_config)
    
    if 'fit' not in args.config:
        args.config['fit'] = {}
    if not verbose:
        args.config['fit']['verbose'] = False


def _sanitize_sklearn_params(estimator_class, params: dict) -> dict:
    """
    Remove parameters that are not valid for sklearn estimator.
    
    Inspects the estimator's __init__ signature and filters out invalid parameters.
    
    Args:
        estimator_class: Sklearn estimator class
        params: Parameter dictionary
        
    Returns:
        Filtered parameter dictionary with only valid parameters
    """
    if not params or estimator_class is None:
        return params
    
    try:
        sig = inspect.signature(estimator_class.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        
        filtered = {k: v for k, v in params.items() if k in valid_params}
        
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
    
    Different sklearn models accept different parameters. Ensures only valid
    parameters are passed to each specific model.
    
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
        
        if estimator_class is not None:
            args.config['model'] = _sanitize_sklearn_params(estimator_class, params)
            
    except ImportError:
        pass


def _cleanup_temp_directories(tmp_dir: Path, clean: bool = True) -> None:
    """
    Clean up temporary directories and any stray results folders.
    
    TALENT sometimes creates output directories in the current working directory.
    
    Args:
        tmp_dir: Temporary directory to remove
        clean: If True, remove directories. If False, leave as-is for debugging.
    """
    if not clean:
        return
    
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    
    # Remove common stray directories in current working directory
    cwd = Path.cwd()
    stray_dirs = [
        cwd / "results_model",
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
    
    Args:
        config_dir: Directory to clean
    """
    if not config_dir.exists():
        return
    
    unwanted_patterns = [
        '*.pth',
        '*.pkl',
        '*.npy',
        '*-*.pth',
        '*-*.pkl',
    ]
    
    for pattern in unwanted_patterns:
        for file in config_dir.rglob(pattern):
            try:
                file.unlink()
            except Exception:
                pass
    
    # Remove 'trlog' files
    for file in config_dir.rglob('*'):
        if file.is_file() and file.name == 'trlog':
            try:
                file.unlink()
            except Exception:
                pass


def _parse_prediction_output(output: Any) -> Tuple[Any]:
    """
    Parse TALENT's prediction output to extract just predictions.
    
    TALENT methods return predictions in various formats:
    - Just predictions: output
    - (metrics, predictions): tuple of 2
    - (metrics, metric_names, predictions): tuple of 3
    - (val_loss, metrics, metric_names, predictions): tuple of 4+
    
    We only care about predictions since we calculate metrics ourselves.
    
    Returns:
        Raw predictions array
    """
    if isinstance(output, tuple):
        # Predictions are always the last element
        return output[-1]
    else:
        return output


# ======================================================================================
#                               METHOD-SPECIFIC FIXES
# ======================================================================================

def _fix_catboost_config(args, is_regression: bool) -> None:
    """
    Fix CatBoost configuration to ensure compatibility.
    
    CatBoost only accepts ONE of ['verbose', 'logging_level', 'verbose_eval', 'silent'].
    
    Args:
        args: Argument namespace (modified in-place)
        is_regression: Whether this is a regression task
    """
    if not hasattr(args, 'config') or args.config is None:
        args.config = {}
    
    if 'fit' not in args.config:
        args.config['fit'] = {}
    
    if 'model' not in args.config:
        args.config['model'] = {}
    
    # Use logging_level and remove all others
    args.config['fit']['logging_level'] = 'Silent'
    
    for key in ['verbose', 'verbose_eval', 'silent']:
        args.config['fit'].pop(key, None)
    
    if 'early_stopping_rounds' not in args.config['model']:
        args.config['model']['early_stopping_rounds'] = 50


def _fix_lightgbm_config(args) -> None:
    """
    Suppress ALL LightGBM verbosity completely.
    
    LightGBM verbosity is controlled through MODEL parameters only,
    not fit parameters.
    
    Args:
        args: Argument namespace (modified in-place)
    """
    if not hasattr(args, 'config') or args.config is None:
        args.config = {}
    
    if 'model' not in args.config:
        args.config['model'] = {}
    
    # Suppress all verbosity parameters (MODEL parameters only)
    args.config['model']['verbosity'] = -1
    args.config['model']['verbose'] = -1
    args.config['model']['silent'] = True
    
    # Environment variable for extra safety
    os.environ['LIGHTGBM_VERBOSITY'] = '-1'


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
    config_base_dir: Optional[Path] = None,
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
    - Handles CUDA tensor to numpy conversion
    - Extracts probabilities from logits for classification tasks
    - Automatic row limit enforcement for TabPFN (10k) and PFN-v2 (50k)
    - Clips LGD predictions to [0, 1] range
    - Calculates comprehensive metrics internally (not relying on TALENT)
    
    Returns dictionary structure:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Key              │ Type              │ Description                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ y_true           │ np.ndarray        │ Ground truth values              │
    │ y_prob           │ np.ndarray/None   │ Class probabilities (PD only)    │
    │ y_pred           │ np.ndarray        │ Predictions (binary PD/clipped LGD)│
    │ y_pred_raw       │ np.ndarray        │ Raw unprocessed predictions      │
    │ metrics          │ Dict[str, float]  │ Comprehensive metrics dict       │
    │ train_time       │ float             │ Training time in seconds         │
    │ info             │ dict              │ Dataset information              │
    │ method           │ str               │ Method name                      │
    │ dataset          │ str               │ Dataset name                     │
    │ task             │ str               │ Task type (pd/lgd)               │
    │ fold_id          │ int               │ Fold identifier                  │
    │ used_hpo         │ bool              │ Whether HPO was used             │
    │ n_clipped_below  │ int               │ Predictions < 0 (LGD only)       │
    │ n_clipped_above  │ int               │ Predictions > 1 (LGD only)       │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Args:
        task: Task type ('pd' for classification, 'lgd' for regression)
        dataset: Dataset name (must exist in data directory)
        test_size: Test set fraction (0.0 to 1.0)
        val_size: Validation set fraction (0.0 to 1.0)
        cv_splits: Number of cross-validation folds
        seed: Random seed for reproducibility
        row_limit: Optional limit on dataset rows (auto-capped for TabPFN/PFN-v2)
        sampling: Optional sampling fraction
        method: TALENT method name (canonical name, e.g., 'xgboost' not 'XGBoost')
        categorical_encoding: Categorical encoding policy
        numerical_encoding: Numerical encoding policy
        normalization: Normalization method
        num_nan_policy: Numerical NaN handling
        cat_nan_policy: Categorical NaN handling
        max_epoch: Maximum training epochs (deep methods only)
        batch_size: Batch size (deep methods only)
        tune: Whether to perform hyperparameter optimization on first fold
        n_trials: Number of HPO trials (if tune=True)
        early_stopping: Whether to use early stopping (deep methods only)
        early_stopping_patience: Patience for early stopping
        evaluate_option: Which model to use for evaluation ('best-val', 'last')
        model_config: Custom model hyperparameters
        fit_config: Custom fit configuration
        config_base_dir: Base directory where config_hpo folder will be created
        verbose: Whether to print detailed progress
        clean_temp_dir: Whether to clean up temporary directories after run
        
    Returns:
        Dictionary mapping fold_id to results dict
        
    Raises:
        ValueError: If method requirements conflict with user-specified options
        RuntimeError: If training or prediction fails
    """
    
    # Log method start 
    logger.info(f"Starting {method} on {dataset}")
    
    _patch_talent_pprint(enable_silence=not verbose)
    
    try:
        is_regression = (task.lower() == "lgd")
        is_deep = method in DEEP_METHODS
        
        # Apply method-specific row limits (TabPFN: 10k, PFN-v2: 50k)
        original_row_limit = row_limit
        row_limit = apply_method_row_limit(method, row_limit)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Running {method} ({'deep' if is_deep else 'classical'}) on {dataset} ({task.upper()})")
            print(f"{'='*70}")
            
            if row_limit != original_row_limit:
                if original_row_limit is None:
                    print(f"\nRow limit: {row_limit:,} (method maximum for {method})")
                else:
                    print(f"\nRow limit: {row_limit:,} (capped from {original_row_limit:,} due to {method} constraint)")
        
        # Track which preprocessing options user explicitly provided
        user_specified = {
            'cat_policy': not _is_missing(categorical_encoding),
            'num_policy': not _is_missing(numerical_encoding),
            'normalization': not _is_missing(normalization),
            'num_nan_policy': not _is_missing(num_nan_policy),
            'cat_nan_policy': not _is_missing(cat_nan_policy),
        }
        
        # Prepare data with cross-validation folds
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
        
        first_fold_id = min(folds.keys())
        
        if verbose:
            print(f"Fold IDs: {sorted(folds.keys())}")
            print(f"First fold ID: {first_fold_id}")
        
        results: Dict[int, Dict[str, Any]] = {}
        
        # Setup directories
        if config_base_dir is None:
            base_config_dir = _PROJECT_ROOT / "config_hpo"
        else:
            base_config_dir = Path(config_base_dir) / "config_hpo"
        
        # Each method gets its own subdirectory, separated by HPO mode to prevent race conditions
        hpo_subdir = "HPO" if tune else "NO_HPO"
        dataset_config_dir = base_config_dir / task.lower() / dataset / method / hpo_subdir
        dataset_config_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_tmp_dir = Path(tempfile.mkdtemp(prefix=f"talent_ckpt_{dataset}_{method}_"))
        
        if verbose:
            print(f"\nDirectory setup:")
            print(f"  Config directory (persistent): {dataset_config_dir}")
            print(f"  Checkpoint directory (temp):   {checkpoint_tmp_dir}")
            
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
                
                # Get base arguments from TALENT
                orig_argv = sys.argv.copy()
                
                try:
                    sys.argv = [
                        "train.py",
                        "--model_type", method,
                        "--dataset", dataset,
                        "--dataset_path", "./data",
                        "--model_path", str(checkpoint_tmp_dir),
                    ]
                    
                    with _suppress_all_output(not verbose):
                        if is_deep:
                            args, default_para, opt_space = get_deep_args()
                        else:
                            args, default_para, opt_space = get_classical_args()
                finally:
                    sys.argv = orig_argv
                
                # Override TALENT's computed paths
                args.save_path = str(dataset_config_dir)
                args.model_path = str(checkpoint_tmp_dir)
                
                # Set random seed
                args.seed = seed
                set_seeds(seed)
                
                # Set preprocessing options
                args.is_regression = is_regression
                args.normalization = normalization
                args.num_nan_policy = num_nan_policy
                args.cat_nan_policy = cat_nan_policy
                args.cat_policy = categorical_encoding
                args.num_policy = numerical_encoding
                apply_preprocessing_policies(args, method, user_specified)
                
                if verbose and fold_id == first_fold_id:
                    print(f"\nPreprocessing configuration:")
                    print(f"  - cat_policy: {args.cat_policy}")
                    print(f"  - num_policy: {args.num_policy}")
                    print(f"  - normalization: {args.normalization}")
                    print(f"  - num_nan_policy: {args.num_nan_policy}")
                    print(f"  - cat_nan_policy: {args.cat_nan_policy}")

                # Set HPO flags
                args.tune = tune
                args.retune = False
                args.n_trials = n_trials
                args.evaluate_option = evaluate_option
                
                # Set method-specific parameters
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
                
                # Apply method-specific fixes
                if method == 'catboost':
                    _fix_catboost_config(args, is_regression)
                
                if method == 'lightgbm':
                    _fix_lightgbm_config(args)
                
                # HPO logic
                if tune:
                    tuned_config_path = dataset_config_dir / f"{method}-tuned.json"
                    
                    if fold_id == first_fold_id:
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
                        if verbose:
                            print(f"\n[HPO] Loading config from fold {first_fold_id}...")
                    
                    try:
                        train_val_data = (N, C, y)
                        
                        with _suppress_all_output(not verbose):
                            args = tune_hyper_parameters(args, opt_space, train_val_data, info)
                        
                        if fold_id == first_fold_id:
                            if tuned_config_path.exists():
                                if verbose:
                                    print(f"[HPO] Config ready: {tuned_config_path.name}")
                            else:
                                raise RuntimeError(f"HPO failed to save config to {tuned_config_path}")
                        else:
                            if verbose:
                                print(f"[HPO] Config loaded successfully")
                                
                    except Exception as e:
                        if verbose:
                            print(f"[HPO] Error: {e}")
                        raise
                
                else:
                    if verbose and fold_id == first_fold_id:
                        print(f"\n[DEFAULT] Using TALENT's default hyperparameters")
                        print(f"[DEFAULT] Location: .venv/.../TALENT/configs/default/{method}.json")
                        print(f"[DEFAULT] Saved configs in config_hpo/ are ignored when tune=False")
                
                # Train model
                try:
                    with _suppress_all_output(not verbose):
                        MethodClass = get_method(args.model_type)
                        model = MethodClass(args, is_regression=is_regression)
                        
                        t0 = time.time()
                        train_time = model.fit((N, C, y), info, train=True)
                        if train_time is None:
                            train_time = time.time() - t0
                        
                        output = model.predict((N, C, y), info, model_name=args.evaluate_option)
                    
                    # Parse output - we only need predictions
                    predictions_raw = _parse_prediction_output(output)
                    
                    # Convert to numpy arrays
                    y_true_np = _ensure_numpy_array(y["test"])
                    y_pred_raw_np = _ensure_numpy_array(predictions_raw)
                    
                    # Initialize clipping counters
                    n_clipped_below = 0
                    n_clipped_above = 0
                    
                    if is_regression:
                        # LGD: Clip predictions to [0, 1]
                        y_pred_np, n_clipped_below, n_clipped_above = _clip_regression_predictions(
                            y_pred_raw_np
                        )
                        y_prob_np = None  # No probabilities for regression
                        
                        if verbose and (n_clipped_below > 0 or n_clipped_above > 0):
                            total = len(y_pred_np)
                            pct_clipped = 100 * (n_clipped_below + n_clipped_above) / total
                            print(f"\n[CLIPPING] {n_clipped_below} predictions < 0, {n_clipped_above} > 1 ({pct_clipped:.1f}% total)")
                            if pct_clipped > 5:
                                print(f"[WARNING] High clipping rate may indicate model issues")
                        
                        # Calculate our own metrics
                        metrics = calculate_lgd_metrics(y_true_np, y_pred_np)
                        
                    else:
                        # PD: Extract probabilities and binary predictions
                        y_prob_np = _extract_class_probabilities(y_pred_raw_np, method, is_regression)
                        y_pred_np = _extract_binary_predictions(y_pred_raw_np, y_prob_np)
                        
                        # Calculate our own metrics
                        metrics = calculate_pd_metrics(y_true_np, y_prob_np, y_pred_np)
                    
                    # Store results in new structure
                    results[fold_id] = {
                        # Core predictions (in requested order)
                        "y_true": y_true_np,
                        "y_prob": y_prob_np,
                        "y_pred": y_pred_np,
                        
                        # Raw predictions for debugging
                        "y_pred_raw": y_pred_raw_np,
                        
                        # Our calculated metrics
                        "metrics": metrics,
                        
                        # Clipping diagnostics (LGD only, 0 for PD)
                        "n_clipped_below": n_clipped_below,
                        "n_clipped_above": n_clipped_above,
                        
                        # Training info
                        "train_time": float(train_time),
                        "info": info,
                        
                        # Metadata
                        "method": method,
                        "dataset": dataset,
                        "task": task,
                        "fold_id": fold_id,
                        "used_hpo": fold_id == first_fold_id and tune,
                    }
                    
                    if verbose:
                        print(f"\nFold {fold_id} metrics:")
                        for metric_name, metric_value in metrics.items():
                            if not np.isnan(metric_value):
                                print(f"  {metric_name}: {metric_value:.4f}")
            
                except Exception as e:
                    if verbose:
                        print(f"\nError during training: {e}")
                    raise
        
        finally:
            _cleanup_temp_directories(checkpoint_tmp_dir, clean=clean_temp_dir)
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
        
        # Log method completion 
        logger.info(f"Finished {method} on {dataset}")
        
        return results
    
    finally:
        _patch_talent_pprint(enable_silence=False)


# ======================================================================================
#                               CONVENIENCE FUNCTIONS
# ======================================================================================

def get_available_methods() -> Dict[str, list]:
    """
    Get all available methods in TALENT.
    
    Returns:
        Dictionary with 'classical' and 'deep' lists of method names
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
        True if method has meaningful HPO space
    """
    return method not in NO_HPO_METHODS