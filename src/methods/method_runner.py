"""TALENT method runner for TabPFNCredit -- now powered by ``TALENT.run()``.

History
-------
Previously this module carried ~1700 lines of glue code (``sys.argv``
mutation, manual softmax / sigmoid conversion, tuple-unpacking
heuristics, classical-param sanitization, CLI artifact cleanup,
monkey-patches against TALENT's pprint, ...). TALENT now exposes a
typed :class:`~TALENT.api.RunResult` and a :func:`~TALENT.api.run`
function that subsume all of that, so this module is now ~500 lines and
does only what is **wrapper-specific**:

* Cross-validation fold assembly via :class:`~src.data.data_feeder.DataFeeder`.
* Foundation-model val/test downsampling (TALENT's row-limit caps train only).
* Per-fold HPO orchestration with SLURM-safe merged-config JSON via
  :class:`~src.utils.file_lock.FileLock`.
* Resumable checkpoint paths keyed by a stable hash (so re-runs skip).
* Credit-risk metric enrichment (Gini / KS / MAPE-with-zeros) via
  :mod:`src.methods.method_metrics`.
* A persistent folds cache via :mod:`joblib.Memory` so SLURM workers
  share the prepared fold dict across processes.

Logging
-------
Uses the stdlib ``logging`` module throughout (no module-level ``print``,
no ``os.environ['LIGHTGBM_VERBOSITY']`` side effects at import time).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Optional torch -- only used for the dtype check below
try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False
    torch = None  # type: ignore

# TALENT public surface (the rewrite)
import TALENT
from TALENT.model.method_registry import (
    METHOD_REGISTRY,
    MethodSpec,
    get_method_spec,
)
from TALENT.model.utils import set_seeds

# Wrapper-side modules
from sklearn.model_selection import StratifiedShuffleSplit

from src.data.data_feeder import DataFeeder
from src.methods.method_config import (
    DEEP_METHODS,
    CLASSICAL_METHODS,
    METHOD_ROW_LIMITS,
    METHOD_TEST_VAL_LIMITS,
    apply_preprocessing_policies,
    _is_missing,
)
from src.methods.method_metrics import enrich_pd_metrics, enrich_lgd_metrics
from src.methods.cost_metrics import cost_sensitive_summary
from src.utils.file_lock import FileLock


# ============================================================================
#  Module setup
# ============================================================================

logger = logging.getLogger(__name__)


def _setup_lightgbm_verbosity() -> None:
    """Opt-in LightGBM silencer. Called from public entry points, never at import."""
    os.environ.setdefault("LIGHTGBM_VERBOSITY", "-1")


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
#  Folds cache (joblib.Memory keyed on dataset hash + split params)
# ============================================================================
#
# Persists across processes -- SLURM workers share the prepared fold dict.
# Invalidates automatically when input arguments change (joblib hashes
# them) or when DataFeeder.prepare's source changes (joblib hashes the
# function bytecode too).

_FOLDS_CACHE_DIR = _PROJECT_ROOT / ".cache" / "folds"
_FOLDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_folds_uncached(
    task: str,
    dataset: str,
    test_size: float,
    val_size: float,
    cv_splits: int,
    seed: int,
    row_limit: Optional[int],
    train_row_limit: Optional[int],
    sampling: Optional[float],
) -> Dict[int, Tuple[Tuple[Any, Any, Any], Dict[str, Any]]]:
    """Build folds via DataFeeder. Wrapped by joblib.Memory below."""
    feeder = DataFeeder(
        task=task,
        dataset=dataset,
        test_size=test_size,
        val_size=val_size,
        cv_splits=cv_splits,
        seed=seed,
        row_limit=row_limit,
        train_row_limit=train_row_limit,
        sampling=sampling,
    )
    return feeder.prepare()


def _get_folds_cache():
    """Lazily build the joblib.Memory cache (importing joblib is cheap but optional)."""
    try:
        from joblib import Memory
    except ImportError:
        return None
    return Memory(_FOLDS_CACHE_DIR, verbose=0)


_MEMORY = _get_folds_cache()
if _MEMORY is not None:
    _prepare_folds_cached = _MEMORY.cache(_prepare_folds_uncached)
else:
    _prepare_folds_cached = _prepare_folds_uncached  # type: ignore


def clear_folds_cache() -> None:
    """Drop all cached folds from disk + memory."""
    if _MEMORY is not None:
        _MEMORY.clear(warn=False)


# ============================================================================
#  HPO config persistence (per-fold merged JSON, SLURM-safe)
# ============================================================================

def save_fold_config_safely(
    config_path: Path,
    fold_id: int,
    hyperparameters: dict,
    n_trials: int,
) -> None:
    """Merge this fold's HPO config into the shared JSON file, under exclusive lock."""
    with FileLock(config_path, exclusive=True) as f:
        f.seek(0)
        content = f.read()
        try:
            all_configs = json.loads(content) if content.strip() else {}
        except json.JSONDecodeError:
            all_configs = {}

        all_configs[f"fold_{fold_id}"] = {
            "hyperparameters": hyperparameters,
            "n_trials": n_trials,
            "timestamp": datetime.now().isoformat(),
        }

        f.seek(0)
        f.truncate()
        f.write(json.dumps(all_configs, indent=2))
        f.flush()
        try:
            os.fsync(f.fileno())
        except (AttributeError, OSError):
            pass


# ============================================================================
#  Resumable checkpoint dirs (B5)
# ============================================================================
#
# Hash the (dataset, method, fold, seed, config) tuple so that a re-run of
# the same configuration picks up the cached checkpoint instead of
# retraining. Crucial for SLURM job recovery.

def _stable_checkpoint_dir(
    base: Path,
    task: str,
    dataset: str,
    method: str,
    fold_id: int,
    seed: int,
    config: Optional[dict] = None,
) -> Path:
    payload = {
        "task": task, "dataset": dataset, "method": method,
        "fold_id": fold_id, "seed": seed,
        "config": config or {},
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    out = base / f"{dataset}_{method}_fold{fold_id}_seed{seed}_{digest}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ============================================================================
#  Foundation-model val/test downsampling
# ============================================================================
#
# TALENT's `MethodSpec.train_row_limit` caps the *training* context for
# in-context learners. Inference on a huge validation/test set can still
# OOM (cross-attention is O(N_train * N_test)). This helper applies
# stratified (PD) / random (LGD) downsampling to val + test only.

def _downsample_split(
    N: Optional[np.ndarray],
    C: Optional[np.ndarray],
    y: np.ndarray,
    limit: int,
    is_classification: bool,
    seed: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    n = len(y)
    if n <= limit:
        return N, C, y
    if is_classification:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=limit, random_state=seed)
        idx, _ = next(sss.split(np.zeros(n), y))
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=limit, replace=False))
    return (
        N[idx] if N is not None else None,
        C[idx] if C is not None else None,
        y[idx],
    )


def _apply_val_test_caps(
    N: Optional[Dict[str, Any]],
    C: Optional[Dict[str, Any]],
    y: Dict[str, Any],
    method: str,
    is_classification: bool,
    seed: int,
) -> Tuple[Optional[Dict], Optional[Dict], Dict, Dict[str, int]]:
    """Cap val and test splits if the method has a context-size limit."""
    if method not in METHOD_TEST_VAL_LIMITS:
        return N, C, y, {}

    limit = METHOD_TEST_VAL_LIMITS[method]
    stats: Dict[str, int] = {}
    N_new = dict(N) if N is not None else None
    C_new = dict(C) if C is not None else None
    y_new = dict(y)

    for split, seed_offset in (("val", 0), ("test", 1)):
        original = len(y[split])
        if original <= limit:
            continue
        N_split = N[split] if N is not None else None
        C_split = C[split] if C is not None else None
        N_d, C_d, y_d = _downsample_split(
            N_split, C_split, y[split], limit, is_classification, seed + seed_offset
        )
        if N_new is not None:
            N_new[split] = N_d
        if C_new is not None:
            C_new[split] = C_d
        y_new[split] = y_d
        stats[f"{split}_original"] = original
        stats[f"{split}_downsampled"] = len(y_d)
        logger.info(
            "[%s LIMIT] %s: downsampled %s split from %d to %d (%s)",
            split.upper(), method, split, original, limit,
            "stratified" if is_classification else "random",
        )

    return N_new, C_new, y_new, stats


# ============================================================================
#  Args builder -- pure programmatic, no sys.argv
# ============================================================================

def _build_talent_args(
    *,
    method: str,
    seed: int,
    is_regression: bool,
    save_path: Path,
    tune: bool,
    n_trials: int,
    max_epoch: int,
    batch_size: int,
    early_stopping: bool,
    early_stopping_patience: int,
    evaluate_option: str,
    user_overrides: Dict[str, Any],
):
    """Replacement for the old ``sys.argv`` hack -- uses ``TALENT.build_args``."""
    spec = get_method_spec(method)

    # Honour the user's preprocessing choices if they specified them; else
    # let TALENT's spec-driven defaults win.
    overrides = dict(user_overrides)
    overrides.setdefault("seed", seed)
    overrides.setdefault("tune", tune)
    overrides.setdefault("n_trials", n_trials)
    overrides.setdefault("evaluate_option", evaluate_option)
    if spec.architecture.value == "deep":
        overrides.setdefault("max_epoch", max_epoch)
        overrides.setdefault("batch_size", batch_size)

    args = TALENT.build_args(
        method,
        save_path=str(save_path),
        **overrides,
    )

    # Provide credit-risk-style defaults (these only fire if not already set)
    apply_preprocessing_policies(args, method, user_specified={})

    # Re-validate after our default fill-in
    spec.validate_args(args)

    # Surface early-stopping config to the method (TALENT doesn't standardize this)
    args.is_regression = is_regression
    args.early_stopping = early_stopping
    args.early_stopping_patience = early_stopping_patience

    return args


# ============================================================================
#  Per-fold execution
# ============================================================================

@dataclass
class _FoldResult:
    """Single-fold output, ready to be serialized."""
    fold_id: int
    metrics: Dict[str, float]
    y_true: np.ndarray
    y_prob: Optional[np.ndarray]
    y_pred: np.ndarray
    val_y_true: Optional[np.ndarray]
    val_y_prob: Optional[np.ndarray]
    train_time: float
    predict_time: float
    threshold: Optional[float]
    used_hpo: bool
    hpo_config: Optional[Dict[str, Any]]
    hpo_n_trials: Optional[int]
    info: Dict[str, Any]
    method: str
    dataset: str
    task: str
    n_clipped_below: int = 0
    n_clipped_above: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "metrics": self.metrics,
            "y_true": self.y_true,
            "y_prob": self.y_prob,
            "y_pred": self.y_pred,
            "val_y_true": self.val_y_true,
            "val_y_prob": self.val_y_prob,
            "train_time": self.train_time,
            "predict_time": self.predict_time,
            "threshold": self.threshold,
            "used_hpo": self.used_hpo,
            "hpo_config": self.hpo_config,
            "hpo_n_trials": self.hpo_n_trials,
            "info": self.info,
            "method": self.method,
            "dataset": self.dataset,
            "task": self.task,
            "n_clipped_below": self.n_clipped_below,
            "n_clipped_above": self.n_clipped_above,
        }


def _run_one_fold(
    fold_id: int,
    train_val_data: Tuple[Any, Any, Any],
    test_data: Tuple[Any, Any, Any],
    info: Dict[str, Any],
    *,
    method: str,
    dataset: str,
    task: str,
    is_regression: bool,
    seed: int,
    tune: bool,
    n_trials: int,
    max_epoch: int,
    batch_size: int,
    early_stopping: bool,
    early_stopping_patience: int,
    evaluate_option: str,
    user_overrides: Dict[str, Any],
    checkpoint_dir: Path,
    merged_hpo_config_path: Optional[Path],
) -> _FoldResult:
    spec = get_method_spec(method)
    args = _build_talent_args(
        method=method,
        seed=seed,
        is_regression=is_regression,
        save_path=checkpoint_dir,
        tune=tune,
        n_trials=n_trials,
        max_epoch=max_epoch,
        batch_size=batch_size,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        evaluate_option=evaluate_option,
        user_overrides=user_overrides,
    )

    set_seeds(seed)

    # Hand off to TALENT
    run_result = TALENT.run(
        method,
        train_val_data,
        test_data,
        info,
        args=args,
        seed=seed,
        tune=tune,
        n_trials=n_trials,
        save_path=str(checkpoint_dir),
    )

    # Persist HPO config (per-fold merged JSON)
    if tune and merged_hpo_config_path is not None:
        save_fold_config_safely(
            config_path=merged_hpo_config_path,
            fold_id=fold_id,
            hyperparameters=dict(run_result.config.get("model", {})),
            n_trials=n_trials,
        )

    # Build credit-risk metrics
    y_test = _to_numpy(test_data[2]["test"])
    if is_regression:
        metrics = enrich_lgd_metrics(run_result, y_test)
        # Clipping diagnostics (LGD targets live in [0, 1])
        raw = np.asarray(run_result.predictions).ravel()
        n_below = int(np.sum(raw < 0))
        n_above = int(np.sum(raw > 1))
        y_pred = np.clip(raw, 0.0, 1.0)
        y_prob = None
        val_y_prob = None
    else:
        metrics = enrich_pd_metrics(run_result, y_test)
        y_pred = (
            run_result.predict_labels
            if run_result.predict_labels is not None
            else np.zeros_like(y_test)
        )
        y_prob = run_result.predict_proba
        # Validation-set probabilities aren't returned by TALENT.run() by
        # default; we expose None and let downstream callers decide
        # whether they need them (Experiment{1,2,3} historically did not).
        val_y_prob = None
        n_below = n_above = 0

        # Cost-sensitive metrics (Expected_Loss_Normalized + profit) -- B9
        try:
            cost_metrics = cost_sensitive_summary(y_test, y_prob) if y_prob is not None else {}
        except Exception:
            cost_metrics = {}
        metrics.update(cost_metrics)

    return _FoldResult(
        fold_id=fold_id,
        metrics=metrics,
        y_true=y_test,
        y_prob=y_prob,
        y_pred=y_pred,
        val_y_true=None,
        val_y_prob=val_y_prob,
        train_time=float(run_result.fit_time or 0.0),
        predict_time=float(run_result.predict_time or 0.0),
        threshold=run_result.threshold,
        used_hpo=tune,
        hpo_config=dict(run_result.config.get("model", {})) if tune else None,
        hpo_n_trials=n_trials if tune else None,
        info=info,
        method=method,
        dataset=dataset,
        task=task,
        n_clipped_below=n_below,
        n_clipped_above=n_above,
    )


def _to_numpy(arr: Any) -> np.ndarray:
    """Best-effort tensor / list -> numpy."""
    if arr is None:
        return None  # type: ignore[return-value]
    if _HAS_TORCH and isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


# ============================================================================
#  Public entry point
# ============================================================================

def run_talent_method(
    *,
    task: str,
    dataset: str,
    method: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    cv_splits: int = 5,
    seed: int = 42,
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
    early_stopping_patience: int = 16,
    evaluate_option: str = "best-val",
    model_config: Optional[dict] = None,
    fit_config: Optional[dict] = None,
    config_base_dir: Optional[Path] = None,
    verbose: bool = False,
    clean_temp_dir: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """Run a TALENT method across CV folds with credit-risk metric enrichment.

    Returns
    -------
    dict
        ``{fold_id: fold_result_dict}`` where ``fold_result_dict`` has the
        keys documented on :class:`_FoldResult.to_dict`.
    """
    _setup_lightgbm_verbosity()
    if method not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method {method!r}. Available: {sorted(METHOD_REGISTRY)}"
        )

    is_regression = task.lower() == "lgd"
    spec = get_method_spec(method)

    # Method-side training cap (e.g. TabPFN v1 ~ 10k rows)
    method_train_cap = METHOD_ROW_LIMITS.get(method)
    train_row_limit = method_train_cap
    if (
        row_limit is not None
        and method_train_cap is not None
        and row_limit * (1 - test_size - val_size) <= method_train_cap
    ):
        # User cap is already below method cap on the train split, so we
        # don't need to additionally clip the train set.
        train_row_limit = None

    logger.info(
        "[%s] %s on %s (task=%s, fold_count=%d, tune=%s, n_trials=%d)",
        method, "tuning" if tune else "training", dataset, task, cv_splits,
        tune, n_trials if tune else 0,
    )

    folds = _prepare_folds_cached(
        task=task, dataset=dataset,
        test_size=test_size, val_size=val_size, cv_splits=cv_splits,
        seed=seed,
        row_limit=row_limit,
        train_row_limit=train_row_limit,
        sampling=sampling,
    )

    # Checkpoint root: persistent so SLURM job retries resume. Falls
    # back to a tempdir if the project root is read-only.
    config_base_dir = (
        Path(config_base_dir) if config_base_dir is not None else _PROJECT_ROOT
    )
    base_config_dir = config_base_dir / "config_hpo"
    try:
        base_config_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        base_config_dir = Path(tempfile.mkdtemp(prefix=f"talent_{dataset}_{method}_"))

    mode_subdir = "HPO_PER_FOLD" if tune else "NO_HPO"
    dataset_config_dir = base_config_dir / task.lower() / dataset / method / mode_subdir
    dataset_config_dir.mkdir(parents=True, exist_ok=True)
    merged_hpo_config_path = (
        dataset_config_dir / f"{method}-all-folds.json" if tune else None
    )

    user_overrides: Dict[str, Any] = {}
    for attr, value in (
        ("cat_policy", categorical_encoding),
        ("num_policy", numerical_encoding),
        ("normalization", normalization),
        ("num_nan_policy", num_nan_policy),
        ("cat_nan_policy", cat_nan_policy),
    ):
        if not _is_missing(value):
            user_overrides[attr] = value

    results: Dict[int, Dict[str, Any]] = {}

    for fold_id, ((N, C, y), info) in folds.items():
        logger.debug("[%s] fold %d/%d on %s", method, fold_id, len(folds), dataset)

        # Apply per-method val/test caps (TALENT only caps train)
        N, C, y, _stats = _apply_val_test_caps(
            N, C, y, method=method, is_classification=not is_regression, seed=seed,
        )

        # Sanitize all-None C dict to the bare None TALENT expects
        if isinstance(C, dict) and C.get("train") is None:
            C = None

        # Reshape into TALENT's (train_val_data, test_data) split
        N_train_val = {"train": N["train"], "val": N["val"]} if N is not None else None
        C_train_val = {"train": C["train"], "val": C["val"]} if C is not None else None
        y_train_val = {"train": y["train"], "val": y["val"]}
        N_test = {"test": N["test"]} if N is not None else None
        C_test = {"test": C["test"]} if C is not None else None
        y_test = {"test": y["test"]}

        train_val_data = (N_train_val, C_train_val, y_train_val)
        test_data = (N_test, C_test, y_test)

        checkpoint_dir = _stable_checkpoint_dir(
            base=dataset_config_dir,
            task=task, dataset=dataset, method=method,
            fold_id=fold_id, seed=seed,
            config=model_config or fit_config,
        )

        try:
            fr = _run_one_fold(
                fold_id=fold_id,
                train_val_data=train_val_data,
                test_data=test_data,
                info=info,
                method=method,
                dataset=dataset,
                task=task,
                is_regression=is_regression,
                seed=seed,
                tune=tune,
                n_trials=n_trials,
                max_epoch=max_epoch,
                batch_size=batch_size,
                early_stopping=early_stopping,
                early_stopping_patience=early_stopping_patience,
                evaluate_option=evaluate_option,
                user_overrides=user_overrides,
                checkpoint_dir=checkpoint_dir,
                merged_hpo_config_path=merged_hpo_config_path,
            )
            results[fold_id] = fr.to_dict()
        except Exception:
            logger.exception("[%s] fold %d failed on %s", method, fold_id, dataset)
            raise

    logger.info("[%s] completed %d folds on %s", method, len(results), dataset)
    return results


# ============================================================================
#  Convenience accessors (kept for backwards-compatible imports)
# ============================================================================

def get_available_methods() -> Dict[str, list]:
    return {
        "classical": sorted(CLASSICAL_METHODS),
        "deep": sorted(DEEP_METHODS),
    }


def validate_method(method: str) -> Tuple[str, bool]:
    if method in DEEP_METHODS:
        return method, True
    if method in CLASSICAL_METHODS:
        return method, False
    raise ValueError(
        f"Unknown method {method!r}. Available: "
        f"{sorted(DEEP_METHODS | CLASSICAL_METHODS)}"
    )


__all__ = [
    "run_talent_method",
    "get_available_methods",
    "validate_method",
    "save_fold_config_safely",
    "clear_folds_cache",
]
