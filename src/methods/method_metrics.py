"""
Metric calculation functions for credit risk benchmarking.

This module provides comprehensive metric calculations for:
- PD (Probability of Default): Binary classification metrics
- LGD (Loss Given Default): Regression metrics

These functions calculate all metrics internally rather than relying on
TALENT's metric computation, ensuring consistency and control over the
evaluation process.

Can be imported independently for use in analysis notebooks.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np


def find_optimal_threshold_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_thresholds: int = 100
) -> Tuple[float, float]:
    """
    Find the optimal probability threshold that maximizes F1 score.
    
    Args:
        y_true: Ground truth binary labels (0 or 1), shape (n_samples,)
        y_prob: Predicted probabilities for positive class, shape (n_samples,)
        n_thresholds: Number of thresholds to test (default: 100)
        
    Returns:
        Tuple of (optimal_threshold, best_f1_score)
    """
    from sklearn.metrics import f1_score
    
    # Test thresholds from 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred_temp = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred_temp, zero_division=0)
        f1_scores.append(f1)
    
    # Find threshold with maximum F1
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    return float(optimal_threshold), float(best_f1)


def calculate_pd_metrics(
    y_true: np.ndarray, 
    y_prob: Optional[np.ndarray], 
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics for PD (Probability of Default) task.
    
    This function computes all relevant metrics for binary classification in credit risk.
    Metrics are divided into probability-based (require y_prob) and prediction-based
    (require y_pred) categories.
    
    IMPORTANT: If y_prob is provided, the function will:
    1. Find the optimal threshold that maximizes F1 score
    2. Use that threshold to generate optimized binary predictions
    3. Calculate Accuracy, Precision, Recall, and F1 using the optimized predictions
    4. Return the optimal threshold in the metrics dictionary
    
    Args:
        y_true: Ground truth binary labels (0 or 1), shape (n_samples,)
        y_prob: Predicted probabilities for positive class (0.0 to 1.0), shape (n_samples,)
                Can be None if method doesn't produce probabilities
        y_pred: Predicted binary labels (0 or 1), shape (n_samples,)
                NOTE: This is ignored if y_prob is provided (optimal threshold is used instead)
        
    Returns:
        Dictionary mapping metric names to float values. NaN is used for metrics
        that cannot be computed (e.g., AUC when y_prob is None).
        
    Metrics computed:
        Probability-based (require y_prob):
        - AUC: Area Under ROC Curve (higher is better, 0.5 = random)
        - Gini: Gini coefficient = 2*AUC - 1 (higher is better, 0 = random)
        - Avg_Precision: Average Precision / PR-AUC (higher is better)
        - KS: Kolmogorov-Smirnov statistic (higher is better)
        - Brier: Brier score (lower is better, measures calibration)
        - LogLoss: Log loss / cross-entropy (lower is better)
        
        Prediction-based (with F1-optimized threshold if y_prob provided):
        - Optimal_Threshold: Threshold that maximizes F1 (only if y_prob provided)
        - Accuracy: Overall accuracy (higher is better)
        - Balanced_Accuracy: Balanced accuracy (higher is better, handles imbalance)
        - F1: F1 score (higher is better) - OPTIMIZED
        - Precision: Precision (higher is better) - using optimal threshold
        - Recall: Recall / Sensitivity / TPR (higher is better) - using optimal threshold
        - MCC: Matthews Correlation Coefficient (higher is better, -1 to 1)
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
    
    # Check if we have both classes
    has_both_classes = len(np.unique(y_true)) > 1
    
    # ==========================================================================
    # THRESHOLD OPTIMIZATION (if y_prob is provided)
    # ==========================================================================
    
    if y_prob is not None and has_both_classes:
        # Find optimal threshold that maximizes F1
        optimal_threshold, _ = find_optimal_threshold_f1(y_true, y_prob)
        
        # Generate optimized predictions using optimal threshold
        y_pred_optimized = (y_prob >= optimal_threshold).astype(int)
        
        # Store optimal threshold in metrics
        metrics['Optimal_Threshold'] = optimal_threshold
        
        # Use optimized predictions for all prediction-based metrics
        y_pred = y_pred_optimized
    else:
        # No optimization possible, use provided y_pred
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
    
    # KS Statistic (Kolmogorov-Smirnov)
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
    
    # Brier Score (lower is better, measures calibration)
    try:
        if y_prob is not None:
            metrics['Brier'] = float(brier_score_loss(y_true, y_prob))
        else:
            metrics['Brier'] = np.nan
    except Exception:
        metrics['Brier'] = np.nan
    
    # Log Loss (lower is better)
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
    # Prediction-based metrics (using F1-optimized threshold if available)
    # ==========================================================================
    
    # Accuracy
    metrics['Accuracy'] = float(accuracy_score(y_true, y_pred))
    
    # Balanced Accuracy (handles class imbalance)
    metrics['Balanced_Accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
    
    # F1 Score (optimized if y_prob was provided)
    metrics['F1'] = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Precision (using optimal threshold if y_prob was provided)
    metrics['Precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    
    # Recall (Sensitivity, True Positive Rate) (using optimal threshold if y_prob was provided)
    metrics['Recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    
    # MCC (Matthews Correlation Coefficient) - robust to class imbalance
    metrics['MCC'] = float(matthews_corrcoef(y_true, y_pred))
    
    return metrics


def calculate_lgd_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics for LGD (Loss Given Default) task.
    
    This function computes all relevant metrics for regression in credit risk.
    Predictions should already be clipped to [0, 1] before calling this function.
    
    Args:
        y_true: Ground truth LGD values, shape (n_samples,)
        y_pred: Predicted LGD values (should be clipped to [0, 1]), shape (n_samples,)
        
    Returns:
        Dictionary mapping metric names to float values. NaN is used for metrics
        that cannot be computed.
        
    Metrics computed:
        Error metrics:
        - R2: R-squared / Coefficient of Determination (higher is better, can be negative)
        - MSE: Mean Squared Error (lower is better)
        - RMSE: Root Mean Squared Error (lower is better)
        - MAE: Mean Absolute Error (lower is better)
        - MedAE: Median Absolute Error (lower is better, robust to outliers)
        - MaxError: Maximum absolute error (lower is better)
        - Explained_Variance: Explained variance score (higher is better)
        
        Percentage metrics:
        - MAPE: Mean Absolute Percentage Error (lower is better, excludes y_true=0)
        
        Correlation metrics:
        - Pearson_Corr: Pearson correlation coefficient (higher is better)
        - Spearman_Corr: Spearman rank correlation (higher is better, robust to outliers)
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
    
    # ==========================================================================
    # Error metrics
    # ==========================================================================
    
    # R² (Coefficient of Determination) - can be negative if model is worse than mean
    try:
        metrics['R2'] = float(r2_score(y_true, y_pred))
    except Exception:
        metrics['R2'] = np.nan
    
    # MSE (Mean Squared Error)
    metrics['MSE'] = float(mean_squared_error(y_true, y_pred))
    
    # RMSE (Root Mean Squared Error)
    metrics['RMSE'] = float(np.sqrt(metrics['MSE']))
    
    # MAE (Mean Absolute Error)
    metrics['MAE'] = float(mean_absolute_error(y_true, y_pred))
    
    # Median Absolute Error (more robust to outliers than MAE)
    metrics['MedAE'] = float(median_absolute_error(y_true, y_pred))
    
    # Max Error (worst case)
    metrics['MaxError'] = float(max_error(y_true, y_pred))
    
    # Explained Variance
    metrics['Explained_Variance'] = float(explained_variance_score(y_true, y_pred))
    
    # ==========================================================================
    # Percentage error metrics
    # ==========================================================================
    
    # MAPE (Mean Absolute Percentage Error) - exclude zeros to avoid division by zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        metrics['MAPE'] = float(mape)
    else:
        metrics['MAPE'] = np.nan
    
    # ==========================================================================
    # Correlation metrics
    # ==========================================================================
    
    # Pearson Correlation
    if np.std(y_pred) > 1e-10 and np.std(y_true) > 1e-10:
        metrics['Pearson_Corr'] = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        metrics['Pearson_Corr'] = np.nan
    
    # Spearman Correlation (rank-based, robust to outliers and non-linear relationships)
    try:
        spearman_corr, _ = stats.spearmanr(y_true, y_pred)
        metrics['Spearman_Corr'] = float(spearman_corr)
    except Exception:
        metrics['Spearman_Corr'] = np.nan
    
    return metrics