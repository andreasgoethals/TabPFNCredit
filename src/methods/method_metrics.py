# src/methods/method_metrics.py
"""
Metric calculation functions for credit risk benchmarking.

This module provides comprehensive metric calculations for:
- PD (Probability of Default): Binary classification metrics
- LGD (Loss Given Default): Regression metrics

These functions calculate all metrics internally rather than relying on
TALENT's metric computation, ensuring consistency and control over the
evaluation process.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import numpy as np


def find_optimal_threshold_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_thresholds: int = 100,
) -> Tuple[float, float]:
    """
    Find the optimal probability threshold that maximises the F1 score.

    The search grid is a concatenation of three linearly-spaced segments:

    * ``[1e-4, 1e-2]`` -- fine resolution for highly imbalanced regimes
      (Experiment 3's ``minority_proportion`` as low as 0.01 requires
      thresholds < 0.01 to peak F1).
    * ``[0.01, 0.99]``  -- the classical midrange, resolution controlled by
      ``n_thresholds``.
    * ``[0.99, 1 - 1e-4]`` -- symmetric fine resolution at the upper tail.

    Args:
        y_true: Ground-truth binary labels, shape ``(n_samples,)``.
        y_prob: Predicted probabilities for the positive class, shape
            ``(n_samples,)``.
        n_thresholds: Number of thresholds to evaluate in the midrange grid.
            The lower and upper tails always get 20 points each.

    Returns:
        Tuple ``(optimal_threshold, best_f1_score)``.
    """
    from sklearn.metrics import f1_score

    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(1e-4, 1e-2, 20),
                np.linspace(0.01, 0.99, n_thresholds),
                np.linspace(0.99, 1.0 - 1e-4, 20),
            ]
        )
    )

    f1_scores = np.empty_like(thresholds)
    for i, threshold in enumerate(thresholds):
        y_pred_temp = (y_prob >= threshold).astype(int)
        f1_scores[i] = f1_score(y_true, y_pred_temp, zero_division=0)

    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def calculate_pd_metrics(
    y_true: np.ndarray, 
    y_prob: Optional[np.ndarray], 
    y_pred: np.ndarray,
    val_y_true: Optional[np.ndarray] = None,
    val_y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics for PD (Probability of Default) task.
    
    IMPROVED THRESHOLD LOGIC:
    This function prevents data leakage by allowing the optimal threshold to be
    calculated on a separate validation set, then applied to the current (test) set.
    
    Logic:
    1. If val_y_true and val_y_prob are provided:
       - Find optimal threshold using Validation data.
       - Apply that threshold to Current (Test) y_prob to get predictions.
    2. If validation data is NOT provided:
       - Find optimal threshold using Current data (standard behavior for val evaluation).
       - Apply that threshold to Current y_prob.
    
    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_prob: Predicted probabilities for positive class (0.0 to 1.0)
        y_pred: Predicted binary labels (ignored if y_prob is present)
        val_y_true: (Optional) Ground truth labels of the VALIDATION set
        val_y_prob: (Optional) Predicted probabilities of the VALIDATION set
        
    Returns:
        Dictionary mapping metric names to float values.
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, accuracy_score,
        balanced_accuracy_score, f1_score, precision_score, recall_score,
        brier_score_loss, log_loss, matthews_corrcoef
    )
    from scipy import stats
    
    metrics: Dict[str, float] = {}
    
    # Ensure arrays are 1D
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_prob is not None:
        y_prob = np.asarray(y_prob).ravel()
    
    # Check if we have both classes in the target
    has_both_classes = len(np.unique(y_true)) > 1
    
    # ==========================================================================
    # THRESHOLD OPTIMIZATION
    # ==========================================================================
    
    optimal_threshold = 0.5 # Default
    threshold_source = "default"

    if y_prob is not None:
        # Check if validation data is available for optimization
        if val_y_true is not None and val_y_prob is not None:
            # OPTIMIZE ON VALIDATION SET (Correct for Test Evaluation)
            val_y_true = np.asarray(val_y_true).ravel()
            val_y_prob = np.asarray(val_y_prob).ravel()
            
            if len(np.unique(val_y_true)) > 1:
                optimal_threshold, _ = find_optimal_threshold_f1(val_y_true, val_y_prob)
                threshold_source = "validation"
            else:
                # Fallback if validation set is single-class (rare)
                optimal_threshold = 0.5
        elif has_both_classes:
            # OPTIMIZE ON CURRENT SET (Correct for Val Evaluation, or fallback)
            optimal_threshold, _ = find_optimal_threshold_f1(y_true, y_prob)
            threshold_source = "self"
        
        # Apply the determined threshold to the CURRENT probabilities
        y_pred = (y_prob >= optimal_threshold).astype(int)
        
        metrics['Optimal_Threshold'] = optimal_threshold
        # metrics['Threshold_Source'] = 1.0 if threshold_source == "validation" else 0.0 # Optional debug
    else:
        metrics['Optimal_Threshold'] = np.nan
    
    # ==========================================================================
    # Probability-based metrics (require y_prob)
    # ==========================================================================
    
    # AUC-ROC
    try:
        if y_prob is not None and has_both_classes:
            metrics['AUC'] = float(roc_auc_score(y_true, y_prob))
            metrics['Gini'] = 2 * metrics['AUC'] - 1
        else:
            metrics['AUC'] = np.nan
            metrics['Gini'] = np.nan
    except Exception:
        metrics['AUC'] = np.nan
        metrics['Gini'] = np.nan
    
    # Average Precision (PR-AUC)
    try:
        if y_prob is not None and has_both_classes:
            metrics['Avg_Precision'] = float(average_precision_score(y_true, y_prob))
        else:
            metrics['Avg_Precision'] = np.nan
    except Exception:
        metrics['Avg_Precision'] = np.nan
    
    # KS Statistic
    try:
        if y_prob is not None and has_both_classes:
            pos_probs = y_prob[y_true == 1]
            neg_probs = y_prob[y_true == 0]
            if len(pos_probs) > 0 and len(neg_probs) > 0:
                ks_stat, _ = stats.ks_2samp(pos_probs, neg_probs)
                metrics['KS'] = float(ks_stat)
            else:
                metrics['KS'] = np.nan
        else:
            metrics['KS'] = np.nan
    except Exception:
        metrics['KS'] = np.nan
    
    # Brier Score
    try:
        if y_prob is not None:
            metrics['Brier'] = float(brier_score_loss(y_true, y_prob))
        else:
            metrics['Brier'] = np.nan
    except Exception:
        metrics['Brier'] = np.nan
    
    # Log Loss
    try:
        if y_prob is not None:
            # Clip probabilities to avoid log(0)
            y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
            metrics['LogLoss'] = float(log_loss(y_true, y_prob_clipped))
        else:
            metrics['LogLoss'] = np.nan
    except Exception:
        metrics['LogLoss'] = np.nan
    
    # ==========================================================================
    # Prediction-based metrics (calculated using Optimized Threshold)
    # ==========================================================================
    
    # Accuracy
    metrics['Accuracy'] = float(accuracy_score(y_true, y_pred))
    
    # Balanced Accuracy
    metrics['Balanced_Accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
    
    # F1 Score
    metrics['F1'] = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Precision
    metrics['Precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    
    # Recall
    metrics['Recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    
    # MCC
    metrics['MCC'] = float(matthews_corrcoef(y_true, y_pred))
    
    return metrics


def calculate_lgd_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics for LGD (Loss Given Default) task.
    Predictions should be clipped to [0, 1] before calling this function.
    """
    from sklearn.metrics import (
        r2_score, mean_squared_error, mean_absolute_error,
        explained_variance_score, median_absolute_error, max_error
    )
    from scipy import stats
    
    metrics: Dict[str, float] = {}
    
    # Ensure arrays are 1D
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    # Error metrics
    try:
        metrics['R2'] = float(r2_score(y_true, y_pred))
    except Exception:
        metrics['R2'] = np.nan
    
    metrics['MSE'] = float(mean_squared_error(y_true, y_pred))
    metrics['RMSE'] = float(np.sqrt(metrics['MSE']))
    metrics['MAE'] = float(mean_absolute_error(y_true, y_pred))
    metrics['MedAE'] = float(median_absolute_error(y_true, y_pred))
    metrics['MaxError'] = float(max_error(y_true, y_pred))
    metrics['Explained_Variance'] = float(explained_variance_score(y_true, y_pred))
    
    # MAPE (Mean Absolute Percentage Error) - exclude zeros to avoid division by zero
    # Note: Excluding y_true=0 samples is standard practice for MAPE but affects comparability
    mask = y_true != 0
    n_zeros_excluded = len(y_true) - mask.sum()
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        metrics['MAPE'] = float(mape)
    else:
        metrics['MAPE'] = np.nan
    # Store count of excluded zeros for transparency in results
    metrics['MAPE_n_zeros_excluded'] = int(n_zeros_excluded)
    if n_zeros_excluded > 0:
        pct_excluded = 100 * n_zeros_excluded / len(y_true)
        metrics['MAPE_pct_zeros_excluded'] = float(pct_excluded)
    
    # Correlation metrics
    if np.std(y_pred) > 1e-10 and np.std(y_true) > 1e-10:
        metrics['Pearson_Corr'] = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        metrics['Pearson_Corr'] = np.nan
    
    try:
        spearman_corr, _ = stats.spearmanr(y_true, y_pred)
        metrics['Spearman_Corr'] = float(spearman_corr)
    except Exception:
        metrics['Spearman_Corr'] = np.nan
    
    return metrics