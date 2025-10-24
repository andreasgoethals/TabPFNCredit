import os
import yaml
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
from hmeasure import h_score

# sklearn metrics
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    brier_score_loss, average_precision_score, mean_squared_error,
    mean_absolute_error, r2_score
)

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Evaluates model predictions for PD (classification) and LGD (regression) tasks.
    Dynamically loads all metrics and settings from CONFIG_EVALUATION.yaml.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize evaluator from YAML config file.
        """
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        self.config_path = config_path or os.path.join(self.project_root, "config", "CONFIG_EVALUATION.yaml")

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"CONFIG_EVALUATION.yaml not found at {self.config_path}")

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.binary_threshold = float(self.config.get("binary_threshold", 0.5))
        self.round_digits = int(self.config.get("round_digits", 5))
        self.metrics = self.config.get("metrics", {"pd": {}, "lgd": {}})

        logger.info(f"[Evaluator] Loaded configuration from {self.config_path}")

    # ----------------------------------------------------------------------
    # CLASSIFICATION (PD)
    # ----------------------------------------------------------------------
    def evaluate_classification(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Compute all enabled classification metrics for PD task.
        """
        results = {}
        t = self.binary_threshold
        y_pred = (y_pred_proba > t).astype(int)

        available_metrics = {
            "accuracy": lambda: accuracy_score(y_true, y_pred),
            "brier": lambda: brier_score_loss(y_true, y_pred_proba),
            "f1": lambda: f1_score(y_true, y_pred),
            "precision": lambda: precision_score(y_true, y_pred, zero_division=0.0),
            "recall": lambda: recall_score(y_true, y_pred, zero_division=0.0),
            "aucroc": lambda: roc_auc_score(y_true, y_pred_proba),
            "aucpr": lambda: average_precision_score(y_true, y_pred_proba),
            "h_measure": lambda: (
                h_score(y_true, y_pred_proba)
                if HAS_HMEASURE and len(np.unique(y_true)) == 2 else np.nan
            ),
        }

        for metric_name, metric_func in available_metrics.items():
            if self.metrics["pd"].get(metric_name, False):
                try:
                    value = metric_func()
                    results[metric_name] = round(float(value), self.round_digits)
                except Exception as e:
                    logger.warning(f"Metric '{metric_name}' failed: {e}")
                    results[metric_name] = np.nan

        return results

    # ----------------------------------------------------------------------
    # REGRESSION (LGD)
    # ----------------------------------------------------------------------
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute all enabled regression metrics for LGD task.
        """
        results = {}
        available_metrics = {
            "mse": lambda: mean_squared_error(y_true, y_pred),
            "mae": lambda: mean_absolute_error(y_true, y_pred),
            "r2": lambda: r2_score(y_true, y_pred),
            "rmse": lambda: np.sqrt(mean_squared_error(y_true, y_pred)),
        }

        for metric_name, metric_func in available_metrics.items():
            if self.metrics["lgd"].get(metric_name, False):
                try:
                    value = metric_func()
                    results[metric_name] = round(float(value), self.round_digits)
                except Exception as e:
                    logger.warning(f"Metric '{metric_name}' failed: {e}")
                    results[metric_name] = np.nan

        return results

    # ----------------------------------------------------------------------
    # AGGREGATION (PER-DATASET, PER-METHOD)
    # ----------------------------------------------------------------------
    def aggregate_folds(self, fold_results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate results across folds (compute mean of each metric).
        """
        if not fold_results:
            return {}

        df = pd.DataFrame(fold_results)
        mean_results = df.mean(numeric_only=True).to_dict()
        return {k: round(v, self.round_digits) for k, v in mean_results.items()}

    # ----------------------------------------------------------------------
    # FULL EVALUATION PIPELINE
    # ----------------------------------------------------------------------
    def evaluate_dataset_method(self, task_type: str, dataset_name: str, method_name: str, method_dir: str):
        """
        Evaluate all folds for a dataset × method combination and save aggregated results.
        - Reads fold prediction CSVs (y_true, y_pred)
        - Computes metrics
        - Aggregates results and includes average training time
        - Saves aggregated_metrics.csv to method_dir
        """
        fold_files = sorted(
            [f for f in os.listdir(method_dir) if f.startswith("fold_") and f.endswith("_predictions.csv")]
        )
        if not fold_files:
            logger.warning(f"No fold prediction files found in {method_dir}")
            return None

        rows = []
        for f in fold_files:
            df = pd.read_csv(os.path.join(method_dir, f))
            y_true, y_pred = df["y_true"].to_numpy(), df["y_pred"].to_numpy()

            if task_type == "pd":
                metrics = self.evaluate_classification(y_true, y_pred)
            else:
                metrics = self.evaluate_regression(y_true, y_pred)

            metrics["fold_file"] = f
            metrics["dataset"] = dataset_name
            metrics["method"] = method_name
            rows.append(metrics)

        # Aggregate means of all metrics
        mean_row = self.aggregate_folds(rows)
        mean_row.update({"fold_file": "mean", "dataset": dataset_name, "method": method_name})

        # --------------------------------------------------------------
        # ADD TRAINING TIME INFORMATION (mean from configdata.csv)
        # --------------------------------------------------------------
        config_path = os.path.join(method_dir, "configdata.csv")
        if os.path.exists(config_path):
            try:
                config_df = pd.read_csv(config_path)
                if "train_time_sec" in config_df.columns:
                    mean_train_time = config_df["train_time_sec"].mean()
                    mean_row["train_time_sec"] = round(mean_train_time, self.round_digits)
            except Exception as e:
                logger.warning(f"Failed to load training time from configdata.csv: {e}")

        rows.append(mean_row)

        # Save aggregated metrics to CSV
        df_out = pd.DataFrame(rows)
        out_path = os.path.join(method_dir, "aggregated_metrics.csv")
        df_out.to_csv(out_path, index=False)
        logger.info(f"Saved aggregated metrics (with avg training time) to {out_path}")

        return df_out
