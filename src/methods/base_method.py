import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class BaseMethod(ABC):
    """
    Abstract base class for all TALENT-integrated models.
    It provides a consistent interface for model training and inference.
    Cross-validation, tuning, and output saving are handled externally.
    """

    def __init__(self, name: str, dataset_name: str, task_type: str = "pd"):
        """
        Args:
            name (str): Method name (e.g., 'rf', 'tabpfn', 'xgboost').
            dataset_name (str): Name of dataset being processed.
            task_type (str): 'pd' for probability of default, 'lgd' for regression.
        """
        self.name = name
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.model = None
        self.train_time_sec = None

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model on provided data. Implemented by subclass."""
        pass

    @abstractmethod
    def predict(self, X):
        """Return model predictions."""
        pass

    def fit_with_timing(self, X_train, y_train, X_val=None, y_val=None):
        """
        Wrapper for timing model training duration.
        Returns the time in seconds.
        """
        logger.info(f"[{self.name}] Training on dataset '{self.dataset_name}'...")
        start_time = time.perf_counter()
        self.fit(X_train, y_train, X_val, y_val)
        self.train_time_sec = time.perf_counter() - start_time
        logger.info(f"[{self.name}] Training completed in {self.train_time_sec:.2f}s")
        return self.train_time_sec

    def run_inference(self, X_test, y_test):
        """
        Run model inference and return predictions + minimal metadata.
        Returns:
            dict: {
                "y_true": np.ndarray,
                "y_pred": np.ndarray,
                "train_time_sec": float,
                "timestamp": str,
                "model": str,
                "dataset": str,
                "n_test": int
            }
        """
        y_pred = self.predict(X_test)
        return {
            "y_true": np.array(y_test),
            "y_pred": np.array(y_pred),
            "train_time_sec": self.train_time_sec,
            "timestamp": datetime.now().isoformat(),
            "model": self.name,
            "dataset": self.dataset_name,
            "n_test": len(y_test),
            "n_features": X_test.shape[1],
        }
