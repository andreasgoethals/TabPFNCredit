# src/methods/talent_method.py
from .base_method import BaseMethod
import numpy as np

class TalentMethod(BaseMethod):
    """
    Thin adapter that wraps a TALENT estimator and exposes the BaseMethod API.
    You pass a TALENT class (from talent_interface) and optional kwargs
    (possibly filled by your tuner) and it handles fit/predict.
    """

    def __init__(self, model_class, name: str, dataset_name: str, task_type: str = "pd", **model_kwargs):
        super().__init__(name=name, dataset_name=dataset_name, task_type=task_type)
        self._model_class = model_class
        self._model_kwargs = model_kwargs
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.model is None:
            self.model = self._model_class(**self._model_kwargs)
        # Many TALENT models accept (X, y) directly; some can take eval_set for early stopping
        if X_val is not None and y_val is not None and hasattr(self.model, "fit"):
            try:
                self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            except TypeError:
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X):
        # For PD, prefer probabilities if available (improves AUC/AP)
        if self.task_type == "pd":
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)
                # assume binary; take positive-class prob
                if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
                    return proba[:, 1]
                # some wrappers might return 1D already
                return proba
            elif hasattr(self.model, "decision_function"):
                return self.model.decision_function(X)
        # Fallback to labels/values
        return self.model.predict(X)
