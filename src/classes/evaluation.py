# tooling:
import warnings
from typing import Dict

import numpy as np

# metrics:
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, precision_score,
                             brier_score_loss, average_precision_score, mean_squared_error,
                             mean_absolute_error, r2_score, root_mean_squared_error, recall_score)
from hmeasure import h_score

# Proprietary imports
from src.classes.models.models import ModelConfiguration

# hide warnings
warnings.filterwarnings("ignore")

class ModelEvaluator:
    """
    Handles the evaluation of machine learning models by calculating metrics for
    both classification and regression tasks. This class is configured using a
    ModelConfiguration instance, which dictates metric applicability, thresholds,
    and rounding preferences.

    :ivar config: The configuration containing evaluation parameters, metric
                  applicability, threshold values, and rounding digits.
    :type config: ModelConfiguration
    """
    def __init__(self, config: ModelConfiguration):
        self.config = config

    def evaluate_classification(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Evaluates classification performance by calculating various metrics such as
        accuracy, F1 score, precision, recall, and others. The threshold for
        binarization of predictions is determined by the configuration.

        :param y_true: Ground truth binary labels (0s and 1s).
                       Expected as a numpy array.
        :type y_true: np.ndarray
        :param y_pred_proba: Predicted probabilities for the positive class.
                             Expected as a numpy array.
        :type y_pred_proba: np.ndarray
        :return: A dictionary containing the computed classification metrics as keys
                 and their corresponding rounded values as floats.
        :rtype: Dict[str, float]
        """
        results = {}
        t = float(self.config.binary_threshold)
        y_pred = (y_pred_proba > t).astype(int)

        metrics = {
            'accuracy': lambda: accuracy_score(y_true, y_pred),
            'brier': lambda: brier_score_loss(y_true, y_pred_proba),
            'f1': lambda: f1_score(y_true, y_pred),
            'precision': lambda: precision_score(y_true, y_pred, zero_division=0.0),
            'recall': lambda: recall_score(y_true, y_pred, zero_division=0.0),
            'h_measure': lambda: h_score(y_true, y_pred_proba) if len(np.unique(y_true)) == 2 else np.nan,
            'aucroc': lambda: roc_auc_score(y_true, y_pred_proba),
            'aucpr': lambda: average_precision_score(y_true, y_pred_proba)
        }

        for metric_name, metric_func in metrics.items():
            if self.config.metrics['pd'].get(metric_name, False):
                results[metric_name] = round(metric_func(), self.config.round_digits)

        return results

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluates regression model predictions against true values based on configured metrics.

        This function computes various regression metrics, such as Mean Squared Error (MSE),
        Mean Absolute Error (MAE), R-squared (R2), and Root Mean Squared Error (RMSE), depending
        on the configuration provided in the object's `self.config`. It processes these
        metrics for the given true values and predicted values, rounding results to the
        specified number of digits.

        :param y_true: True values for the regression problem.
        :type y_true: np.ndarray
        :param y_pred: Predicted values from the model.
        :type y_pred: np.ndarray
        :return: A dictionary containing scores for the selected metrics, rounded to the
            configured number of decimal places.
        :rtype: Dict[str, float]
        """
        results = {}
        metrics = {
            'mse': lambda: mean_squared_error(y_true, y_pred),
            'mae': lambda: mean_absolute_error(y_true, y_pred),
            'r2': lambda: r2_score(y_true, y_pred),
            'rmse': lambda: root_mean_squared_error(y_true, y_pred)
        }

        for metric_name, metric_func in metrics.items():
            if self.config.metrics['lgd'].get(metric_name, False):
                results[metric_name] = round(metric_func(), self.config.round_digits)

        return results