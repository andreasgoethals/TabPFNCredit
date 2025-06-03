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
    def __init__(self, config: ModelConfiguration):
        self.config = config

    def evaluate_classification(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        results = {}
        t = float(self.config.binary_threshold)
        y_pred = (y_pred_proba > t).astype(int)

        metrics = {
            'accuracy': lambda: accuracy_score(y_true, y_pred),
            'brier': lambda: brier_score_loss(y_true, y_pred_proba),
            'f1': lambda: f1_score(y_true, y_pred),
            'precision': lambda: precision_score(y_true, y_pred, zero_division=0.0),
            'recall': lambda: recall_score(y_true, y_pred, zero_division=0.0),
            'h_measure': lambda: h_score(y_true, y_pred_proba),
            'aucroc': lambda: roc_auc_score(y_true, y_pred_proba),
            'aucpr': lambda: average_precision_score(y_true, y_pred_proba)
        }

        for metric_name, metric_func in metrics.items():
            if self.config.metrics['pd'].get(metric_name, False):
                results[metric_name] = round(metric_func(), self.config.round_digits)

        return results

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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