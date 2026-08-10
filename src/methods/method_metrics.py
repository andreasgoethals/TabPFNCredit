"""Credit-risk metrics for TabPFNCredit (wrapper-specific layer over TALENT).

What this module does
---------------------
TALENT now computes the bread-and-butter classification metrics
(Accuracy, F1, Precision, Recall, balanced accuracy, LogLoss, AUC) **plus
calibration metrics (Brier, ECE)** for every classifier, and tunes the
decision threshold on the validation split automatically. So this module
is reduced to credit-risk-specific KPIs that TALENT does not carry:

* **Gini coefficient** (`2 * AUC - 1`) -- the standard credit-risk
  discrimination metric.
* **KS statistic** -- maximum vertical distance between the cumulative
  distributions of positives and negatives.
* **MAPE with zero-exclusion bookkeeping** -- LGD targets can be exactly
  zero (no loss), so MAPE excludes them and reports the count for
  transparency.
* **Spearman / Pearson correlations** -- LGD rank-correlation reporting.

TALENT's tuned-threshold metrics are surfaced via :func:`enrich_pd_metrics`
and :func:`enrich_lgd_metrics`, which take a TALENT ``RunResult`` plus the
ground-truth labels and return the full credit-risk metric dictionary.

Historical helpers
------------------
``find_optimal_threshold_f1`` and the local Brier block were deleted --
TALENT's ``model/lib/threshold.tune_threshold`` and
``model/lib/calibration.{brier_score, expected_calibration_error}``
supersede them.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    max_error,
    explained_variance_score,
    r2_score,
)
from scipy import stats


# ============================================================================
#  Constants
# ============================================================================

# sklearn's log_loss requires probabilities in (0, 1); pin both ends.
_LOG_EPS = 1e-15
# Threshold below which a correlation denominator is treated as 0.
_CORR_EPS = 1e-10


# ============================================================================
#  Credit-risk specific helpers
# ============================================================================

def _safe(fn, *args, **kwargs) -> float:
    """Call ``fn(*args, **kwargs)`` and return NaN on exception."""
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


def gini_from_auc(auc: float) -> float:
    """Gini coefficient: ``2 * AUC - 1``. Returns NaN if AUC is NaN."""
    if auc is None or np.isnan(auc):
        return float("nan")
    return float(2.0 * auc - 1.0)


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic for binary classification.

    Computes the maximum vertical distance between the empirical CDFs of
    the positive-class probability for positives vs. negatives. Standard
    credit-scoring KPI; ranges in ``[0, 1]``, larger is better.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return _safe(lambda: stats.ks_2samp(pos, neg).statistic)


# ============================================================================
#  PD (binary classification) metrics
# ============================================================================

def calculate_pd_metrics(
    y_true: np.ndarray,
    y_prob: Optional[np.ndarray],
    y_pred: np.ndarray,
    *,
    threshold: Optional[float] = None,
    talent_metrics: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Compute the full credit-risk PD metric set.

    Parameters
    ----------
    y_true : (N,) ndarray
        Ground-truth binary labels (0 = non-default, 1 = default).
    y_prob : (N, 2) or (N,) ndarray, optional
        Predicted probabilities of the positive class. Required for
        threshold-independent metrics (AUC / LogLoss / Brier / KS / Gini).
    y_pred : (N,) ndarray
        Hard predictions, already produced with the **tuned threshold**
        (TALENT does this; see ``RunResult.predict_labels``).
    threshold : float, optional
        The decision threshold used to produce ``y_pred``. Stored as a
        metric for transparency.
    talent_metrics : mapping, optional
        Metrics dict from ``RunResult.metrics`` (keyed by
        ``RunResult.metric_names``). If supplied, we forward TALENT's
        Brier/ECE/AUC/LogLoss directly rather than recomputing.

    Returns
    -------
    dict
        Keys: ``Accuracy``, ``Balanced_Accuracy``, ``Precision``,
        ``Recall``, ``F1``, ``MCC``, ``AUC``, ``LogLoss``, ``Brier``,
        ``ECE``, ``Gini``, ``KS``, ``Optimal_Threshold``.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    pos_proba: Optional[np.ndarray] = None
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            pos_proba = y_prob[:, 1]
        elif y_prob.ndim == 1:
            pos_proba = y_prob

    metrics: Dict[str, float] = {}

    # Hard-prediction metrics (already use the tuned threshold from TALENT)
    metrics["Accuracy"] = _safe(accuracy_score, y_true, y_pred)
    metrics["Balanced_Accuracy"] = _safe(balanced_accuracy_score, y_true, y_pred)
    metrics["F1"] = _safe(lambda: f1_score(y_true, y_pred, zero_division=0))
    metrics["Precision"] = _safe(lambda: precision_score(y_true, y_pred, zero_division=0))
    metrics["Recall"] = _safe(lambda: recall_score(y_true, y_pred, zero_division=0))
    metrics["MCC"] = _safe(matthews_corrcoef, y_true, y_pred)

    # Threshold-independent probability metrics.
    if talent_metrics is not None:
        # Forward TALENT's calibration / AUC / LogLoss directly when available.
        for key in ("AUC", "LogLoss", "Brier", "ECE"):
            if key in talent_metrics:
                metrics[key] = float(talent_metrics[key])
    if "AUC" not in metrics:
        if pos_proba is not None and len(np.unique(y_true)) == 2:
            metrics["AUC"] = _safe(roc_auc_score, y_true, pos_proba)
        else:
            metrics["AUC"] = float("nan")
    if "LogLoss" not in metrics:
        if pos_proba is not None:
            clipped = np.clip(pos_proba, _LOG_EPS, 1.0 - _LOG_EPS)
            # Pass full two-column probabilities with explicit labels. sklearn's
            # log_loss reads a bare 1-D array as multiclass scores and warns
            # that the rows "do not sum to one" (a hard error from sklearn
            # 1.5+); stacking [1-p, p] and pinning labels keeps it a well-formed
            # binary distribution.
            proba_2d = np.column_stack([1.0 - clipped, clipped])
            metrics["LogLoss"] = _safe(log_loss, y_true, proba_2d, labels=[0, 1])
        else:
            metrics["LogLoss"] = float("nan")
    metrics.setdefault("Brier", float("nan"))
    metrics.setdefault("ECE", float("nan"))

    # Credit-risk specific
    metrics["Gini"] = gini_from_auc(metrics["AUC"])
    metrics["KS"] = ks_statistic(y_true, pos_proba) if pos_proba is not None else float("nan")
    metrics["Optimal_Threshold"] = float(threshold) if threshold is not None else float("nan")

    # ---- Average Precision (area under PR curve) + adjusted AP ----
    #
    # Ordinary AP has a random-ranking reference value equal to the
    # positive-class prevalence pi (scikit-learn: "with random predictions, the
    # AP is the fraction of positive samples"), and its maximum is 1. Since pi
    # differs per dataset, those two reference points sit at different places on
    # every dataset. The adjusted form
    #     AP_adjusted = (AP - pi) / (1 - pi)
    # rescales the range so that random ranking maps to 0 and perfect ranking to
    # 1 on every dataset. Aligning the reference points is all it does: it does
    # NOT remove all dependence on prevalence, and adjusted values from datasets
    # with very different pi should still be compared with that in mind.
    # We store, for transparency:
    #   AP             -- raw average precision
    #   AP_baseline    -- pi, the positive-class prevalence (AP's random-ranking
    #                     reference value)
    #   AP_minus_baseline -- AP - pi (absolute lift above random ranking)
    #   AP_adjusted    -- (AP - pi) / (1 - pi)  <-- reported cross-dataset metric
    metrics.update(average_precision_deviation(y_true, pos_proba))

    return metrics


def average_precision_deviation(
    y_true: np.ndarray, pos_proba: Optional[np.ndarray]
) -> Dict[str, float]:
    """Average precision, its random-ranking reference, and the adjusted form.

    Returns a dict with ``AP``, ``AP_baseline`` (the positive-class prevalence
    pi, which is AP's random-ranking reference value), ``AP_minus_baseline``
    (``AP - pi``), and ``AP_adjusted`` = ``(AP - pi) / (1 - pi)``.

    The adjustment places random ranking at 0 and perfect ranking at 1 on every
    dataset, which makes the reported values easier to read side by side. It
    aligns those reference points only; it does not make the metric independent
    of prevalence.
    """
    out: Dict[str, float] = {
        "AP": float("nan"),
        "AP_baseline": float("nan"),
        "AP_minus_baseline": float("nan"),
        "AP_adjusted": float("nan"),
    }
    y_true = np.asarray(y_true).ravel()
    if pos_proba is None or len(y_true) == 0 or len(np.unique(y_true)) < 2:
        # AP undefined without both classes; still record the baseline pi.
        if len(y_true):
            out["AP_baseline"] = float(np.mean(y_true.astype(float)))
        return out

    pi = float(np.mean(y_true.astype(float)))  # positive-class prevalence
    ap = _safe(average_precision_score, y_true, pos_proba)
    out["AP"] = ap
    out["AP_baseline"] = pi
    if ap == ap:  # not NaN
        out["AP_minus_baseline"] = ap - pi
        # (1 - pi) == 0 only if everything is positive (excluded above).
        out["AP_adjusted"] = (ap - pi) / (1.0 - pi) if pi < 1.0 else float("nan")
    return out


# ============================================================================
#  LGD (regression) metrics
# ============================================================================

def calculate_lgd_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    talent_metrics: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Compute the full credit-risk LGD metric set.

    Predictions are expected to be already clipped to ``[0, 1]``. The
    expected-loss-style metrics (MAPE with zero exclusion, Pearson and
    Spearman correlation) are wrapper-specific; the rest mirror standard
    sklearn regression metrics and are computed locally for parity with
    the PD report shape.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    metrics: Dict[str, float] = {}

    # Standard regression metrics
    metrics["R2"] = _safe(r2_score, y_true, y_pred)
    mse = _safe(mean_squared_error, y_true, y_pred)
    metrics["MSE"] = mse
    metrics["RMSE"] = float(np.sqrt(mse)) if not np.isnan(mse) else float("nan")
    metrics["MAE"] = _safe(mean_absolute_error, y_true, y_pred)
    metrics["MedAE"] = _safe(median_absolute_error, y_true, y_pred)
    metrics["MaxError"] = _safe(max_error, y_true, y_pred)
    metrics["Explained_Variance"] = _safe(explained_variance_score, y_true, y_pred)

    # MAPE with zero-exclusion bookkeeping
    mask = y_true != 0
    n_zeros_excluded = int(len(y_true) - int(mask.sum()))
    if int(mask.sum()) > 0:
        metrics["MAPE"] = float(
            np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
        )
    else:
        metrics["MAPE"] = float("nan")
    metrics["MAPE_n_zeros_excluded"] = n_zeros_excluded
    if n_zeros_excluded > 0:
        metrics["MAPE_pct_zeros_excluded"] = float(100.0 * n_zeros_excluded / len(y_true))

    # Correlations
    if np.std(y_pred) > _CORR_EPS and np.std(y_true) > _CORR_EPS:
        metrics["Pearson_Corr"] = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        metrics["Pearson_Corr"] = float("nan")
    try:
        spearman, _ = stats.spearmanr(y_true, y_pred)
        metrics["Spearman_Corr"] = float(spearman)
    except Exception:
        metrics["Spearman_Corr"] = float("nan")

    return metrics


# ============================================================================
#  RunResult bridge -- consume TALENT.run() output directly
# ============================================================================

def enrich_pd_metrics(run_result: Any, y_true: np.ndarray) -> Dict[str, float]:
    """Build a full PD metric dict from a TALENT ``RunResult`` + ground truth.

    Uses ``RunResult.predict_proba`` (already standardized to (N, 2)),
    ``RunResult.predict_labels`` (already produced with the tuned
    threshold), and forwards TALENT's calibration metrics directly.
    """
    talent_metrics = dict(zip(run_result.metric_names, run_result.metrics))
    return calculate_pd_metrics(
        y_true=y_true,
        y_prob=run_result.predict_proba,
        y_pred=run_result.predict_labels,
        threshold=run_result.threshold,
        talent_metrics=talent_metrics,
    )


def enrich_lgd_metrics(run_result: Any, y_true: np.ndarray) -> Dict[str, float]:
    """Build a full LGD metric dict from a TALENT ``RunResult`` + ground truth.

    Predictions are clipped to ``[0, 1]`` first (LGD targets live in that
    range), then standard regression metrics + credit-risk specific
    extras are computed.
    """
    raw = np.asarray(run_result.predictions).ravel()
    clipped = np.clip(raw, 0.0, 1.0)
    return calculate_lgd_metrics(y_true=y_true, y_pred=clipped)


__all__ = [
    "calculate_pd_metrics",
    "calculate_lgd_metrics",
    "enrich_pd_metrics",
    "enrich_lgd_metrics",
    "gini_from_auc",
    "ks_statistic",
]
