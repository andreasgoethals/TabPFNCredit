"""TALENT method runner for TabPFNCredit -- now powered by ``TALENT.run()``.

Built on TALENT's typed :class:`~TALENT.api.RunResult` and
:func:`~TALENT.api.run`, this module does only what is **wrapper-specific**:

* Cross-validation fold assembly via :class:`~src.data.data_feeder.DataFeeder`.
* Foundation-model val/test downsampling (TALENT's row-limit caps train only).
* Per-fold HPO orchestration via TALENT's Optuna tuner (in-process; no
  persistent per-fold state is written -- see the temp-directory bullet).
* A throwaway temp directory per fold for TALENT's internal scratch
  (model snapshots, logits, …). Nothing persistent: the only "save" is
  the final per-(dataset, method) JSON + npz that ``save_method``
  writes once the whole sweep finishes; skip-if-done logic checks
  *that* file, not any checkpoint directory.
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

import logging
import tempfile
from dataclasses import dataclass
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
from src.methods.tabfm_chunked import install as install_tabfm_chunked_inference
from src.utils.paths import cache_root
from src.utils.runtime_quiet import configure_quiet_runtime


# ============================================================================
#  Module setup
# ============================================================================

logger = logging.getLogger(__name__)


def _setup_lightgbm_verbosity() -> None:
    """Silence high-volume third-party log noise (LightGBM / Optuna / sklearn).

    Called from public entry points, never at import. Delegates to the shared
    :func:`~src.utils.runtime_quiet.configure_quiet_runtime` (idempotent).
    """
    configure_quiet_runtime()


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
#  Folds cache (joblib.Memory keyed on dataset hash + split params)
# ============================================================================
#
# Persists across processes -- SLURM workers share the prepared fold dict.
# Invalidates automatically when input arguments change (joblib hashes
# them) or when DataFeeder.prepare's source changes (joblib hashes the
# function bytecode too).

_FOLDS_CACHE_DIR = cache_root() / "folds"
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
#  Per-fold scratch directory
# ============================================================================
#
# TALENT needs a ``save_path`` to write its internal scratch (model
# snapshots, intermediate logits, …). We use a throwaway ``tempfile``
# directory per fold and let it be garbage-collected at process exit;
# nothing about the run is reconstructed from it on a re-invocation.
# The only resume mechanism is the existence of the final per-(dataset,
# method) JSON + npz on disk, which the CLI checks before launching.


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
    """Cap val/test splits ONLY for methods explicitly listed in
    METHOD_TEST_VAL_LIMITS as inference-OOM-prone.

    The cap exists solely to avoid inference OOM, not to normalise evaluation.
    By default METHOD_TEST_VAL_LIMITS is empty, so every method is scored on the
    full val/test fold (identical test set across methods -- required for
    cross-method comparability).
    """
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

# ----------------------------------------------------------------------------
#  HPO objective metric
# ----------------------------------------------------------------------------
# Which validation metric TALENT's hyper-parameter search optimizes. PD
# (classification) tunes on AUC and LGD (regression) tunes on R2 -- i.e. the
# headline metric we report for each task. A few methods optimize an internal
# training loss and do not accept a configurable objective (TALENT raises for
# them), so they fall back to that default via ``None``. Keep this set in sync
# with TALENT's ``model.lib.tuning_metric.METHODS_WITHOUT_TUNE_METRIC``.
HPO_METRIC_CLASSIFICATION = "AUC"
HPO_METRIC_REGRESSION = "R2"
_HPO_METRIC_UNSUPPORTED = frozenset({"tabnet", "ptarl", "tabcaps"})


def _resolve_hpo_metric(method: str, is_regression: bool) -> Optional[str]:
    """HPO objective metric for ``method`` (``None`` = TALENT's legacy default)."""
    if method in _HPO_METRIC_UNSUPPORTED:
        return None
    return HPO_METRIC_REGRESSION if is_regression else HPO_METRIC_CLASSIFICATION


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
    seed_num: int = 1,
):
    """Build the TALENT args object for a method via ``TALENT.build_args``."""
    spec = get_method_spec(method)

    # Honour the user's preprocessing choices if they specified them; else
    # let TALENT's spec-driven defaults win.
    overrides = dict(user_overrides)
    overrides.setdefault("seed", seed)
    overrides.setdefault("tune", tune)
    overrides.setdefault("n_trials", n_trials)
    # Model-seed repeats per fold. MUST be set explicitly: TALENT's
    # ``build_args`` layers its packaged ``deep_configs.json`` /
    # ``classical_configs.json`` over its own baked-in defaults, and those
    # packaged files carry ``seed_num: 15``. Left alone, every method is
    # therefore refit 15x per fold while ``RunResult.predictions`` /
    # ``.predict_proba`` / ``.metrics`` only ever carry the LAST seed -- and
    # those are exactly the fields this repo consumes (see
    # ``enrich_pd_metrics`` / ``enrich_lgd_metrics``). Nothing reads
    # ``metrics_mean`` / ``per_seed``, so 14 of the 15 fits were pure waste.
    overrides.setdefault("seed_num", max(1, int(seed_num)))
    overrides.setdefault("evaluate_option", evaluate_option)
    # Tune PD on AUC and LGD on R2 (see _resolve_hpo_metric). Forwarded to
    # TALENT as ``args.tune_metric``; older TALENT builds simply ignore it.
    overrides.setdefault("tune_metric", _resolve_hpo_metric(method, is_regression))

    # Pin the compute device to the hardware that is actually present.
    # TALENT decides CatBoost's ``task_type`` purely from ``args.gpu``
    # (whose default is ``"0"``), so on a CPU-only node it would request GPU
    # training and abort with a CUDA driver error. Passing ``gpu="cpu"`` when
    # no CUDA device is visible keeps every classical method on the CPU;
    # ``gpu="0"`` restores the normal GPU path when a device is available.
    _cuda_available = _HAS_TORCH and torch.cuda.is_available()
    overrides.setdefault("gpu", "0" if _cuda_available else "cpu")

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

    # Silence LightGBM's per-tree chatter at the source. verbose=-1 maps to the
    # LightGBM core 'verbosity' and suppresses the "[Info] / [Warning] No
    # further splits with positive gain" spam (tens of thousands of lines per
    # sweep). The global logger is muted too in configure_quiet_runtime().
    if method == "lightgbm" and isinstance(getattr(args, "config", None), dict):
        args.config.setdefault("model", {}).setdefault("verbose", -1)

    # TabFM context cap, calibrated from measured H100-80GB runs.
    #
    # HOW TABFM BATCHES (from tabfm 1.0.0 `_batch_forward`): one "batch element"
    # is one ENSEMBLE MEMBER's whole sequence = its in-context train rows + the
    # ENTIRE test split (`Xs: (n_members, train+test, features)`);
    # ``general.batch_size`` is how many members go through one forward pass.
    # Memory therefore scales ~ batch_size x (context + test) rows. The stock
    # ``batch_size=1`` is the right setting here and is left untouched: a single
    # member's sequence of ~138k rows fits in 80 GB, whereas raising batch_size
    # multiplies whole sequences per forward and asks for 81-134 GiB.
    #
    # ``max_num_rows`` (TabFM's OWN per-ensemble-member row subsampler) caps the
    # CONTEXT half of that sequence. 10k per member keeps the context cost flat
    # across datasets, and each of the 32 members draws its OWN seeded 10k
    # sample, so the ensemble collectively covers up to ~320k distinct rows.
    # Same cap size as the Mitra / TabPFN-v2 registry caps -- disclose alongside
    # them in the paper.
    #
    # It does NOT cap the test half, and that half is what decides whether a run
    # survives: at a 10k context, splits of 106k x 35, 61k x 120 and 32k x 500
    # features each ran 17-27 GiB short, while every split of <= 30k rows
    # completed. It is handled by ``src.methods.tabfm_chunked``, which scores the
    # split in one pass by default and halves the chunk on CUDA OOM. Set
    # ``general.predict_chunk_size`` to force a chunk size up front and skip the
    # probe. Do NOT raise ``batch_size`` on the big datasets.
    if method == "tabfm" and isinstance(getattr(args, "config", None), dict):
        gen = args.config.setdefault("general", {})
        gen.setdefault("max_num_rows", 10_000)
        install_tabfm_chunked_inference()

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
    seed_num: int = 1,
) -> _FoldResult:
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
        seed_num=seed_num,
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

        # Cost-sensitive metrics (Expected_Loss_Normalized + profit).
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
    seed_num: int = 1,
    model_config: Optional[dict] = None,
    fit_config: Optional[dict] = None,
    config_base_dir: Optional[Path] = None,
    verbose: bool = False,
    clean_temp_dir: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """Run a TALENT method across CV folds with credit-risk metric enrichment.

    Parameters
    ----------
    seed_num : int, default 1
        Model-seed repeats per fold. Only the last repeat's predictions are
        reported, so anything above 1 multiplies the cost of a fold without
        changing what is recorded -- see ``_build_talent_args``.

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
        "[%s] %s on %s (task=%s, fold_count=%d, tune=%s, n_trials=%d, seed_num=%d)",
        method, "tuning" if tune else "training", dataset, task, cv_splits,
        tune, n_trials if tune else 0, seed_num,
    )

    folds = _prepare_folds_cached(
        task=task, dataset=dataset,
        test_size=test_size, val_size=val_size, cv_splits=cv_splits,
        seed=seed,
        row_limit=row_limit,
        train_row_limit=train_row_limit,
        sampling=sampling,
    )

    # ``config_base_dir`` is kept on the signature for backwards
    # compatibility (older callers passed it in) but is unused now -- we
    # no longer write any persistent state on a per-fold basis. The only
    # save is the final ``<method>.json`` from ``save_method``.
    config_base_dir = (
        Path(config_base_dir) if config_base_dir is not None else _PROJECT_ROOT
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

        # Class-balance trace for binary classification -- crucial for
        # Experiment 3 (imbalance sweep), but useful in every PD run.
        if not is_regression:
            try:
                train_y = np.asarray(y["train"]).astype(int).ravel()
                val_y = np.asarray(y["val"]).astype(int).ravel()
                test_y_arr = np.asarray(y["test"]).astype(int).ravel()
                logger.info(
                    "[%s] fold %d class balance "
                    "(train pos/total %d/%d = %.4f) "
                    "(val pos/total %d/%d = %.4f) "
                    "(test pos/total %d/%d = %.4f)",
                    method, fold_id,
                    int((train_y == 1).sum()), len(train_y),
                    float(train_y.mean()) if len(train_y) else float("nan"),
                    int((val_y == 1).sum()), len(val_y),
                    float(val_y.mean()) if len(val_y) else float("nan"),
                    int((test_y_arr == 1).sum()), len(test_y_arr),
                    float(test_y_arr.mean()) if len(test_y_arr) else float("nan"),
                )
            except Exception:
                # Defensive -- don't let a malformed y kill the run.
                pass

        # Per-fold throwaway scratch -- TALENT.run needs *some* save_path
        # but nothing in this repo reads it back. Cleaned up via
        # ``tempfile`` semantics (process exit + best-effort rmtree).
        scratch_root = Path(tempfile.mkdtemp(
            prefix=f"talent_{dataset}_{method}_fold{fold_id}_"
        ))

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
                checkpoint_dir=scratch_root,
                seed_num=seed_num,
            )
            results[fold_id] = fr.to_dict()
        except Exception:
            logger.exception("[%s] fold %d failed on %s", method, fold_id, dataset)
            raise
        finally:
            # Best-effort cleanup of the per-fold scratch directory; failure
            # is harmless (process exit will also reap it).
            import shutil
            shutil.rmtree(scratch_root, ignore_errors=True)

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
    "clear_folds_cache",
]
