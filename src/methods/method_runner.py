# src/methods/method_runner.py
from __future__ import annotations
from typing import Dict, Any, Optional
import os
import sys
import time
import inspect
import shutil
import contextlib
from io import StringIO
from pathlib import Path
import tempfile
import numpy as np

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
#                               INTERNAL UTILITIES
# ======================================================================================

@contextlib.contextmanager
def _silence(enabled: bool = True):
    """Suppress stdout/stderr when enabled=True."""
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
        yield


def _fake_cli_args(method: str, dataset: str, task: str, seed: int, is_deep: bool, silence: bool):
    """
    Build TALENT's argparse.Namespace by mimicking a CLI call.
    This is the only reliable way to make TALENT respect dataset/method names.
    """
    orig_argv = sys.argv
    sys.argv = [
        "train.py",
        "--model_type", method,
        "--dataset", dataset,
        "--task", task,
        "--seed", str(seed),
        "--tune", "False",
    ]
    try:
        with _silence(True):
            args = get_classical_args()
    except SystemExit:
        sys.argv = ["train.py", "--model_type", "xgboost", "--dataset", "dummy"]
        args = get_classical_args()

    return args


def _maybe_inject_configs(args, model_config: Optional[dict], fit_config: Optional[dict], verbose: bool):
    """Attach model/fit configs and enforce silent training."""
    cfg = getattr(args, "config", {}) or {}
    if model_config:
        cfg["model"] = dict(model_config)
    fit_cfg = dict(fit_config or {})
    if not verbose:
        fit_cfg.setdefault("verbose", False)
    cfg["fit"] = fit_cfg
    args.config = cfg


def _sanitize_sklearn_model_kwargs(estimator_ctor, params: dict) -> dict:
    """Remove invalid kwargs for sklearn constructors."""
    if not params or estimator_ctor is None:
        return params
    try:
        allowed = set(inspect.signature(estimator_ctor).parameters.keys())
        return {k: v for k, v in params.items() if k in allowed}
    except Exception:
        return params


def _maybe_sanitize_for_classical(args, method: str):
    """Filter out invalid sklearn params for classical models."""
    cfg = getattr(args, "config", None)
    if not isinstance(cfg, dict) or "model" not in cfg:
        return
    params = cfg["model"]
    est_ctor = None
    try:
        if method == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            est_ctor = RandomForestRegressor.__init__ if getattr(args, "is_regression", False) else RandomForestClassifier.__init__
        elif method == "knn":
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            est_ctor = KNeighborsRegressor.__init__ if getattr(args, "is_regression", False) else KNeighborsClassifier.__init__
        elif method == "svm":
            from sklearn.svm import SVC, SVR
            est_ctor = SVR.__init__ if getattr(args, "is_regression", False) else SVC.__init__
        elif method == "LogReg":
            from sklearn.linear_model import LogisticRegression
            est_ctor = LogisticRegression.__init__
        elif method == "LinearRegression":
            from sklearn.linear_model import LinearRegression
            est_ctor = LinearRegression.__init__
    except Exception:
        pass
    cfg["model"] = _sanitize_sklearn_model_kwargs(est_ctor, params)


_DEEP_MODELS = {
    "mlp", "tabnet", "tabpfn", "PFN-v2",
    "resnet", "node", "ftt", "tabptm", "tabr",
    "saint", "tabtransformer", "grownet", "autoint",
}


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
    categorical_encoding: str,
    numerical_encoding: str,
    normalization: str,
    num_nan_policy: str,
    cat_nan_policy: str,
    max_epochs: int = 100,
    batch_size: int = 1024,
    tune: bool = False,
    n_trials: int = 50,
    early_stopping: bool = True,
    early_stopping_patience: int = 10,
    evaluate_option: str = "best",
    model_config: Optional[dict] = None,
    fit_config: Optional[dict] = None,
    verbose: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Run one TALENT model (no saving, no printing, no leftovers).
    Returns per-fold dict of results.
    """

    is_regression = (task == "lgd")
    is_deep = method in _DEEP_MODELS

    # --- Create fake args via CLI mimic
    args = _fake_cli_args(method, dataset, task, seed, is_deep, silence=not verbose)

    # --- Inject preprocessing and training options
    args.is_regression = is_regression
    args.cat_policy = categorical_encoding
    args.num_policy = numerical_encoding
    args.normalization = normalization
    args.num_nan_policy = num_nan_policy
    args.cat_nan_policy = cat_nan_policy
    args.max_epochs = max_epochs
    args.batch_size = batch_size
    args.tune = tune
    args.n_trials = n_trials
    args.early_stopping = early_stopping
    args.early_stopping_patience = early_stopping_patience
    args.evaluate_option = evaluate_option

    # --- Redirect saving paths to a temporary dir
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset}-{method}-"))
    args.model_path = str(tmp_dir)
    args.save_path = str(tmp_dir)

    # --- Merge configs and sanitize
    _maybe_inject_configs(args, model_config, fit_config, verbose)
    if not is_deep:
        _maybe_sanitize_for_classical(args, method)

    # --- Prepare data folds
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

    set_seeds(seed)
    MethodClass = get_method(args.model_type)

    results: Dict[int, Dict[str, Any]] = {}

    try:
        for fold_id, ((N, C, y), info) in folds.items():
            with _silence(not verbose):
                model = MethodClass(args, is_regression=is_regression)
                t0 = time.time()
                train_time = model.fit((N, C, y), info)
                if train_time is None:
                    train_time = time.time() - t0
                out = model.predict((N, C, y), info, model_name=args.evaluate_option)

            val_loss, metrics, metric_name, y_pred = None, None, None, None
            if isinstance(out, tuple):
                if len(out) >= 4:
                    val_loss, metrics, metric_name, y_pred = out[:4]
                elif len(out) == 3:
                    metrics, metric_name, y_pred = out
                elif len(out) == 2:
                    metrics, y_pred = out
                elif len(out) == 1:
                    y_pred = out[0]
            else:
                y_pred = out

            results[fold_id] = {
                "y_true": y["test"],
                "y_pred": np.asarray(y_pred),
                "metrics": metrics if isinstance(metrics, (dict, list, tuple)) else {},
                "primary_metric": metric_name,
                "val_loss": float(val_loss) if val_loss is not None else None,
                "train_time": float(train_time),
                "info": info,
                "method": method,
                "dataset": dataset,
                "task": task,
            }

    finally:
        # --- Clean up: delete both temp dir and any stray results_model folder
        shutil.rmtree(tmp_dir, ignore_errors=True)
        for maybe in (Path.cwd() / "results_model", Path(__file__).resolve().parents[2] / "results_model"):
            shutil.rmtree(maybe, ignore_errors=True)

    return results
