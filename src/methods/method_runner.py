"""
TALENT-compatible method runner for TabPFNCredit benchmarking.

This module provides a unified interface for running both classical and deep learning
methods through TALENT's API, properly handling argument parsing, configuration,
and data preparation.
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

# TALENT core utilities
from TALENT.model.utils import (
    get_deep_args,
    get_classical_args,
    get_method,
    set_seeds,
)

# Local data loader
from src.data.data_feeder import DataFeeder


# ======================================================================================
#                               CONFIGURATION
# ======================================================================================
# Deep learning methods (TALENT's canonical names only)
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

# Classical methods (TALENT's canonical names only)
CLASSICAL_METHODS = {
    "xgboost", "catboost", "lightgbm", "RandomForest",
    "LogReg", "LinearRegression", "knn", "svm",
    "NaiveBayes", "NCM", "dummy"
}



# ======================================================================================
#                               INTERNAL UTILITIES
# ======================================================================================

@contextlib.contextmanager
def _silence(enabled: bool = True):
    """
    Context manager to suppress stdout/stderr.
    
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


def _inject_configs(
    args: Any,
    model_config: Optional[dict],
    fit_config: Optional[dict],
    verbose: bool
) -> None:
    """
    Inject user-provided model and fit configurations into args.
    
    This allows users to override default hyperparameters without full HPO.
    Modifies args.config in-place.
    
    Args:
        args: Argument namespace from TALENT
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
    
    Modifies args.config['model'] in-place.
    
    Args:
        args: Argument namespace
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


def _setup_temp_directories(args: Any, dataset: str, method: str) -> Path:
    """
    Set up temporary directories for model checkpoints and outputs.
    
    Args:
        args: Arguments namespace
        dataset: Dataset name
        method: Method name
        
    Returns:
        Path to temporary directory
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"talent_{dataset}_{method}_"))
    args.model_path = str(tmp_dir)
    args.save_path = str(tmp_dir)
    return tmp_dir


def _cleanup_temp_directories(tmp_dir: Path, clean: bool = True) -> None:
    """
    Clean up temporary directories and any stray results folders.
    
    Args:
        tmp_dir: Temporary directory to remove
        clean: If True, remove directories. If False, leave as-is.
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


def _parse_prediction_output(output: Any) -> Tuple[Optional[float], Any, Optional[list], Any]:
    """
    Parse TALENT's prediction output.
    
    Returns:
        Tuple of (val_loss, metrics, metric_names, predictions)
        metric_names is always a list of strings
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
    
    This function implements the preprocessing requirements discovered from TALENT's source code.
    Each method has specific requirements for how data should be preprocessed.
    
    Process:
    1. Fill in project defaults for any unspecified preprocessing options
    2. Apply method-specific requirements (some methods MUST use specific policies)
    3. Raise error if user explicitly specified conflicting requirements
    
    Args:
        args: Argument namespace to modify
        method: Method name (lowercase for comparison)
        user_specified: Dict indicating which options user explicitly provided
        
    Raises:
        ValueError: If user-specified options conflict with method requirements
    """
    m = method.lower()

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
        args.cat_nan_policy = 'most_frequent'

    # =============================================================================
    # STEP 2: Apply method-specific preprocessing requirements
    # =============================================================================
    requires_indices = {
        'amformer', 'autoint', 'bishop', 'dcn2', 'ftt', 'grande', 'grownet',
        'hyperfast', 'mitra', 'ptarl', 'realmlp', 'saint', 'snn',
        't2gformer', 'tabm', 'tabtransformer', 'trompt'
    }
    requires_tabr_ohe = {'modernnca', 'tabr', 'mlp_plr', 'tabautopnpnet'}
    forbids_indices = {
        'mlp', 'resnet', 'switchtab', 'danets', 'dnnr', 'excelformer',
        'node', 'protogate', 'tabcaps', 'tabnet', 'tangos'
    }
    requires_no_normalization = {'hyperfast', 'mitra', 'tabicl'}
    requires_no_num_encoding = {
        'hyperfast', 'mitra', 'modernNCA', 'tabicl', 'tabptm', 'tabr'
    }

    # -------------------------------------------------------------------------
    # TabPFN and PFN-v2: Special zero-shot models
    # -------------------------------------------------------------------------
    if m in {'tabpfn', 'pfn-v2', 'pfn_v2', 'pfnv2'}:
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

    # -------------------------------------------------------------------------
    # TabPTM: One-hot encoding + standard normalization
    # -------------------------------------------------------------------------
    elif m == 'tabptm':
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

    # -------------------------------------------------------------------------
    # Methods requiring 'indices'
    # -------------------------------------------------------------------------
    elif m in requires_indices:
        if user_specified.get('cat_policy', False):
            if args.cat_policy != 'indices':
                raise ValueError(f"{method} requires cat_policy='indices' but got '{args.cat_policy}'")
        else:
            args.cat_policy = 'indices'

    # -------------------------------------------------------------------------
    # Methods requiring 'tabr_ohe'
    # -------------------------------------------------------------------------
    elif m in requires_tabr_ohe:
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

    # -------------------------------------------------------------------------
    # Methods forbidding 'indices'
    # -------------------------------------------------------------------------
    elif m in forbids_indices:
        if user_specified.get('cat_policy', False):
            if args.cat_policy == 'indices':
                raise ValueError(f"{method} does not support cat_policy='indices'")
        else:
            if args.cat_policy == 'indices':
                args.cat_policy = 'ordinal'

    # -------------------------------------------------------------------------
    # Methods requiring no normalization
    # -------------------------------------------------------------------------
    if m in requires_no_normalization:
        if user_specified.get('normalization', False):
            if args.normalization != 'none':
                raise ValueError(f"{method} requires normalization='none' but got '{args.normalization}'")
        else:
            args.normalization = 'none'

    # -------------------------------------------------------------------------
    # Methods requiring no numerical encoding
    # -------------------------------------------------------------------------
    if m in requires_no_num_encoding:
        if user_specified.get('num_policy', False):
            if args.num_policy != 'none':
                raise ValueError(f"{method} requires num_policy='none' but got '{args.num_policy}'")
        else:
            args.num_policy = 'none'

    # -------------------------------------------------------------------------
    # NEW RULE: MLP-based models must use cat_nan_policy='new'
    # -------------------------------------------------------------------------
    if m in {'mlp', 'realmlp', 'mlp_plr'}:
        if user_specified.get('cat_nan_policy', False):
            if args.cat_nan_policy != 'new':
                raise ValueError(
                    f"Method {method} requires cat_nan_policy='new' but got '{args.cat_nan_policy}'"
                )
        else:
            args.cat_nan_policy = 'new'


def _load_hpo_config(args: Any, method: str, fold_id: int) -> Optional[Dict[str, Any]]:
    """
    Load HPO configuration if available.
    
    Returns:
        Config dict if found, None otherwise
    """
    tuned_config_path = Path(args.save_path) / f'{method}-tuned-fold{fold_id}.json'
    
    if tuned_config_path.exists():
        try:
            with open(tuned_config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            warnings.warn(f"Failed to load HPO config from {tuned_config_path}: {e}")
    
    return None


def _save_hpo_config(config: Dict[str, Any], args: Any, method: str, fold_id: int) -> None:
    """Save HPO configuration for fold."""
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    tuned_config_path = Path(args.save_path) / f'{method}-tuned-fold{fold_id}.json'
    
    try:
        with open(tuned_config_path, 'w') as f:
            json.dump(config, f, sort_keys=True, indent=2)
    except Exception as e:
        warnings.warn(f"Failed to save HPO config to {tuned_config_path}: {e}")

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
    verbose: bool = False,
    clean_temp_dir: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Run a TALENT method on credit risk data with proper preprocessing and configuration.
    
    This is the main entry point for running any TALENT method. It handles:
    - Data loading and splitting
    - Preprocessing policy application
    - Method instantiation and training
    - Cross-validation loop
    - Optional hyperparameter optimization
    
    Args:
        task: Task type ('pd' or 'lgd')
        dataset: Dataset name
        test_size: Test set fraction
        val_size: Validation set fraction  
        cv_splits: Number of CV folds
        seed: Random seed for reproducibility
        row_limit: Optional limit on dataset rows
        sampling: Optional sampling fraction
        method: TALENT method name (canonical)
        categorical_encoding: Categorical encoding policy (None uses method default)
        numerical_encoding: Numerical encoding policy (None uses method default)
        normalization: Normalization method (None uses method default)
        num_nan_policy: Numerical NaN handling (None uses method default)
        cat_nan_policy: Categorical NaN handling (None uses method default)
        max_epoch: Maximum training epochs (deep methods only)
        batch_size: Batch size (deep methods only)
        tune: Whether to perform hyperparameter optimization
        n_trials: Number of HPO trials (if tune=True)
        early_stopping: Whether to use early stopping
        early_stopping_patience: Patience for early stopping
        evaluate_option: Which model to use for evaluation ('best-val', 'last')
        model_config: Custom model hyperparameters (overrides defaults)
        fit_config: Custom fit configuration (overrides defaults)
        verbose: Whether to print detailed progress
        clean_temp_dir: Whether to clean up temporary directories after run
        
    Returns:
        Dictionary mapping fold_id to results dict containing:
            - y_true: Ground truth labels
            - y_pred: Predictions
            - metrics: Performance metrics
            - train_time: Training time in seconds
            - method, dataset, task, fold_id: Metadata
            
    Raises:
        ValueError: If method requirements conflict with user-specified options
        RuntimeError: If training or prediction fails
    """
    
    # Determine task type and method category
    is_regression = (task.lower() == "lgd")
    is_deep = method.lower() in DEEP_METHODS
    
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
    # Get base arguments from TALENT
    # -------------------------------------------------------------------------
    orig_argv = sys.argv.copy()
    tmp_dir = None
    
    try:
        # Build command-line arguments for TALENT's parser
        sys.argv = [
            "train.py",
            "--model_type", method,
            "--dataset", dataset,
            "--seed", str(seed),
            "--dataset_path", "./data",
            "--model_path", "./models",
        ]
        
        # Add tune flag if HPO is requested
        if tune:
            sys.argv.append("--tune")  # action='store_true' - no value
            sys.argv.extend(["--n_trials", str(n_trials)])
        
        with _silence(not verbose):
            if is_deep:
                args, default_para, opt_space = get_deep_args()
            else:
                args, default_para, opt_space = get_classical_args()
    finally:
        sys.argv = orig_argv
    
    # -------------------------------------------------------------------------
    # Set preprocessing options from user input
    # -------------------------------------------------------------------------
    args.is_regression = is_regression
    args.normalization = normalization
    args.num_nan_policy = num_nan_policy
    args.cat_nan_policy = cat_nan_policy
    args.cat_policy = categorical_encoding
    args.num_policy = numerical_encoding

    # Apply defaults + method-specific requirements
    _apply_preprocessing_policies(args, method, user_specified)
    
    if verbose:
        print(f"\nPreprocessing configuration:")
        print(f"  - cat_policy: {args.cat_policy}")
        print(f"  - num_policy: {args.num_policy}")
        print(f"  - normalization: {args.normalization}")
        print(f"  - num_nan_policy: {args.num_nan_policy}")
        print(f"  - cat_nan_policy: {args.cat_nan_policy}")

    # -------------------------------------------------------------------------
    # Set hyperparameter optimization flags
    # -------------------------------------------------------------------------
    args.tune = tune
    args.n_trials = n_trials
    args.evaluate_option = evaluate_option
    
    # -------------------------------------------------------------------------
    # Set deep learning specific parameters BEFORE method creation
    # -------------------------------------------------------------------------
    if is_deep:
        args.max_epoch = max_epoch
        args.batch_size = batch_size
        args.early_stopping = early_stopping
        args.early_stopping_patience = early_stopping_patience
        
        # FIX: Also set in config dict so method reads it there
        if not hasattr(args, 'config') or args.config is None:
            args.config = {}
        if 'fit' not in args.config:
            args.config['fit'] = {}
        args.config['fit']['max_epoch'] = max_epoch
    
    # -------------------------------------------------------------------------
    # Inject custom model/fit configurations (before method creation)
    # -------------------------------------------------------------------------
    if model_config or fit_config:
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
    
    # -------------------------------------------------------------------------
    # Ensure n_bins is set for classical methods (before method creation)
    # -------------------------------------------------------------------------
    if not is_deep:
        if not hasattr(args, 'config') or args.config is None:
            args.config = {}
        if 'fit' not in args.config:
            args.config['fit'] = {}
        args.config['fit']['n_bins'] = getattr(args, 'n_bins', 2)
        _sanitize_classical_params(args, method)
    
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
    
    # -------------------------------------------------------------------------
    # Initialize method and run cross-validation
    # -------------------------------------------------------------------------
    set_seeds(seed)
    MethodClass = get_method(args.model_type)
    
    results: Dict[int, Dict[str, Any]] = {}
    optimal_hpo_config = None  # Store HPO config from first fold
    
    try:
        for fold_id, ((N, C, y), info) in folds.items():
            if verbose:
                print(f"\n{'='*70}")
                print(f"Fold {fold_id}/{len(folds)}")
                print(f"{'='*70}")
            
            # FIX: Create unique temporary directory per fold (avoids XGBoost config caching)
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"talent_{dataset}_{method}_fold{fold_id}_"))
            args.model_path = str(tmp_dir)
            args.save_path = str(tmp_dir)
            
            try:
                # FIX: For fold > 0, disable HPO and reuse config from fold 0
                if fold_id > 0:
                    args.tune = False
                    if optimal_hpo_config is not None:
                        args.config = copy.deepcopy(optimal_hpo_config)
                
                with _silence(not verbose):
                    try:
                        # Create fresh model instance for this fold (after all config is set)
                        model = MethodClass(args, is_regression=is_regression)
                        
                        # Train the model
                        t0 = time.time()
                        train_time = model.fit((N, C, y), info)
                        if train_time is None:
                            train_time = time.time() - t0
                        
                        # FIX: Capture HPO config after first fold's training
                        if fold_id == 0 and tune and hasattr(model.args, 'config'):
                            optimal_hpo_config = copy.deepcopy(model.args.config)
                            _save_hpo_config(optimal_hpo_config, args, method, fold_id)
                        
                        # Get predictions
                        output = model.predict((N, C, y), info, model_name=args.evaluate_option)
                        
                    except Exception as e:
                        raise RuntimeError(
                            f"Training/prediction failed for fold {fold_id}: {e}"
                        ) from e
                
                # Parse prediction output
                # Note: TALENT already computes metrics via its metric() function
                # predictions are returned as-is from the model
                val_loss, metrics, metric_names, predictions = _parse_prediction_output(output)
                
                # Store results for this fold
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
                    "optimal_hpo_config": optimal_hpo_config if tune else None,
                }
                
                if verbose and metrics:
                    print(f"\nFold {fold_id} results:")
                    for name, res in zip(metric_names, metrics) if isinstance(metrics, (list, tuple)) else [(None, metrics)]:
                        if name:
                            print(f"  {name}: {res:.4f}")
                        else:
                            print(f"  {res}")
        
            finally:
                # Always clean up temporary directory for this fold
                _cleanup_temp_directories(tmp_dir, clean=clean_temp_dir)
    
    except Exception as e:
        if verbose:
            print(f"\nError during training: {e}")
        raise
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Completed {len(results)} folds for {method}")
        if tune and optimal_hpo_config:
            print(f"Optimal HPO config captured and reused across folds")
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