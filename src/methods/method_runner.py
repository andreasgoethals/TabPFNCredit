# src/methods/method_runner.py
from __future__ import annotations
from typing import Dict, Any, Optional

import numpy as np

from TALENT.model.utils import (
    get_deep_args,
    get_classical_args,
    get_method,
    set_seeds,
)

from src.data.data_feeder import DataFeeder


# Which methods are "deep" in TALENT (so we should build deep args and honor epochs/batch, etc.)
_DEEP_METHODS = {"mlp", "tabnet", "tabpfn"}

# Allowed methods per task (as per your CONFIG_METHOD.yaml)
_ALLOWED_PD = {
    "cb", "knn", "lgbm", "logreg", "nb", "rf", "svm", "xgb", "ncm", "dummy",
    "mlp", "tabnet", "tabpfn"
}
_ALLOWED_LGD = {
    "cb", "knn", "lgbm", "lr", "rf", "xgb",
    "mlp", "tabnet", "tabpfn"
}


def _build_args_for_talent(
    *,
    method: str,
    seed: int,
    # preprocessing policies from CONFIG_EXPERIMENT.yaml
    categorical_encoding: str,
    numerical_encoding: str,
    normalization: str,
    num_nan_policy: str,
    cat_nan_policy: str,
    # training knobs from CONFIG_EXPERIMENT.yaml
    max_epochs: int,
    batch_size: int,
    tune: bool,
    n_trials: int,
    early_stopping: bool,
    early_stopping_patience: int,
    evaluate_option: str = "best",
):
    """
    Construct TALENT's args Namespace exactly as its entry points expect.
    Deep models -> get_deep_args; classical models -> get_classical_args.
    We set only fields that TALENT's methods actually read.
    """
    # 1) Choose args builder
    if method in _DEEP_METHODS:
        args_pack = get_deep_args()     # usually returns (args, default_para, opt_space)
    else:
        args_pack = get_classical_args()

    # Handle both (args, ...) and args-only returns defensively
    args = args_pack[0] if isinstance(args_pack, (tuple, list)) and len(args_pack) >= 1 else args_pack

    # 2) Core identifiers
    args.seed = seed
    args.model_type = method
    args.evaluate_option = evaluate_option

    # 3) Preprocessing / encoding policies
    # (TALENT reads these in Method.data_format()/preprocessors)
    args.cat_policy = categorical_encoding             # "ordinal" | "onehot" | "embedding" | "indices" | "none"
    args.num_policy = numerical_encoding               # "quantile" | "standard" | "power" | "discretize" | "Q_bins" | "none"
    args.normalization = normalization                 # "standard" | "minmax" | "robust" | "log" | "none"
    args.num_nan_policy = num_nan_policy               # "mean" | "median" | "zero" | "none"
    args.cat_nan_policy = cat_nan_policy               # "most_frequent" | "constant" | "none"

    # 4) Training knobs (deep methods will use them; classical methods safely ignore)
    args.max_epochs = int(max_epochs)
    args.batch_size = int(batch_size)
    args.tune = bool(tune)
    args.n_trials = int(n_trials)
    args.early_stopping = bool(early_stopping)
    args.early_stopping_patience = int(early_stopping_patience)

    return args


def run_talent_method(
    *,
    # ---------------------------
    # DataFeeder inputs
    # ---------------------------
    task: str,                     # "pd" or "lgd"
    dataset: str,                  # e.g., "0014.hmeq" or "0001.heloc"
    test_size: float,
    val_size: float,
    cv_splits: int,
    seed: int,
    row_limit: Optional[int] = None,
    sampling: Optional[float] = None,   # PD only (desired minority proportion)

    # ---------------------------
    # Method + experiment configs
    # ---------------------------
    method: str,                   # one of the allowed methods for the given task
    # Preprocessing settings (from CONFIG_EXPERIMENT.yaml)
    categorical_encoding: str,
    numerical_encoding: str,
    normalization: str,
    num_nan_policy: str,
    cat_nan_policy: str,
    # Training & optimization (from CONFIG_EXPERIMENT.yaml)
    max_epochs: int,
    batch_size: int,
    tune: bool,
    n_trials: int,
    early_stopping: bool,
    early_stopping_patience: int,
    # Optional TALENT evaluation tag
    evaluate_option: str = "best",
) -> Dict[int, Dict[str, Any]]:
    """
    Train and evaluate a single TALENT method across all folds for (task, dataset).

    Returns
    -------
    results_by_fold : dict
        fold_id -> {
            "y_true": np.ndarray,
            "y_pred": np.ndarray,
            "metrics": dict,            # TALENT's val_results (on test split)
            "primary_metric": str,      # TALENT's metric_name
            "val_loss": float | None,   # loss reported by method.predict
            "train_time": float,        # seconds returned by method.fit
            "info": dict,               # fold info (task_type, n_num_features, n_cat_features)
            "method": str,
            "dataset": str,
            "task": str,
        }
    """
    # 0) Validate method availability for the requested task
    if task == "pd":
        if method not in _ALLOWED_PD:
            raise ValueError(f"Method '{method}' is not allowed for PD. Allowed: {sorted(_ALLOWED_PD)}")
        is_regression = False
    elif task == "lgd":
        if method not in _ALLOWED_LGD:
            raise ValueError(f"Method '{method}' is not allowed for LGD. Allowed: {sorted(_ALLOWED_LGD)}")
        is_regression = True
    else:
        raise ValueError("task must be 'pd' or 'lgd'")

    # 1) Build args exactly as TALENT expects
    args = _build_args_for_talent(
        method=method,
        seed=seed,
        categorical_encoding=categorical_encoding,
        numerical_encoding=numerical_encoding,
        normalization=normalization,
        num_nan_policy=num_nan_policy,
        cat_nan_policy=cat_nan_policy,
        max_epochs=max_epochs,
        batch_size=batch_size,
        tune=tune,
        n_trials=n_trials,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        evaluate_option=evaluate_option,
    )

    # 2) Prepare folds with your DataFeeder (already returns TALENT format)
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
    folds = feeder.prepare()  # fold_id -> ((N, C, y), info)

    # 3) Reproducibility across folds
    set_seeds(args.seed)

    # 4) Instantiate TALENT method class once per fold (fresh model each fold)
    MethodClass = get_method(args.model_type)

    results_by_fold: Dict[int, Dict[str, Any]] = {}

    for fold_id, ((N, C, y), info) in folds.items():
        # Recreate method for each fold (fresh weights / params)
        method_impl = MethodClass(args, is_regression=is_regression)

        # Fit on train/val; TALENT's Method.fit returns training time in seconds
        train_time = method_impl.fit((N, C, y), info)

        # Predict/evaluate on test; TALENT usually returns a 4-tuple:
        # (val_loss, val_results, metric_name, predictions)
        pred_out = method_impl.predict((N, C, y), info, model_name=args.evaluate_option)

        # Be slightly defensive: some classical methods might return fewer items
        if isinstance(pred_out, tuple):
            if len(pred_out) >= 4:
                val_loss, val_results, metric_name, y_pred = pred_out[:4]
            elif len(pred_out) == 3:
                val_loss, val_results, y_pred = pred_out
                metric_name = None
            elif len(pred_out) == 2:
                val_results, y_pred = pred_out
                val_loss, metric_name = None, None
            else:
                y_pred = pred_out[0]
                val_loss, val_results, metric_name = None, None, None
        else:
            y_pred = pred_out
            val_loss, val_results, metric_name = None, None, None

        # Ground truth (from the same (N, C, y) dict the model sees)
        y_true = y["test"]

        # For PD we assume binary classification only (per project scope)
        # TALENT already computes all standard metrics inside val_results.
        results_by_fold[fold_id] = {
            "y_true": y_true,
            "y_pred": y_pred if isinstance(y_pred, np.ndarray) else np.asarray(y_pred),
            "metrics": val_results if isinstance(val_results, dict) else {},
            "primary_metric": metric_name,
            "val_loss": None if val_loss is None else float(val_loss),
            "train_time": float(train_time) if train_time is not None else float("nan"),
            "info": info,
            "method": method,
            "dataset": dataset,
            "task": task,
        }

    return results_by_fold
