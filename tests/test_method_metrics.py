"""Tests for src.methods.method_metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.methods.method_metrics import (
    calculate_lgd_metrics,
    calculate_pd_metrics,
    gini_from_auc,
    ks_statistic,
)


class TestPDMetrics:

    def test_perfect_predictions(self, synthetic_probas):
        y_true, y_proba = synthetic_probas
        # Perfect: y_pred matches y_true
        y_pred = y_true.copy()
        m = calculate_pd_metrics(y_true, y_proba, y_pred=y_pred)
        assert m["Accuracy"] == 1.0
        assert m["F1"] >= 0.99
        assert m["MCC"] == pytest.approx(1.0, abs=1e-6)
        # Threshold-independent metrics should still be sane
        assert 0.0 <= m["AUC"] <= 1.0
        assert m["KS"] >= 0.0

    def test_keys_present(self, synthetic_probas):
        y_true, y_proba = synthetic_probas
        y_pred = (y_proba[:, 1] >= 0.5).astype(int)
        m = calculate_pd_metrics(y_true, y_proba, y_pred=y_pred, threshold=0.5)
        expected = {"Accuracy", "Balanced_Accuracy", "Precision", "Recall",
                    "F1", "MCC", "AUC", "LogLoss", "Brier", "ECE",
                    "Gini", "KS", "Optimal_Threshold"}
        assert expected <= set(m.keys())
        assert m["Optimal_Threshold"] == pytest.approx(0.5)

    def test_gini_from_auc_perfect(self):
        assert gini_from_auc(1.0) == pytest.approx(1.0)
        assert gini_from_auc(0.5) == pytest.approx(0.0)
        assert np.isnan(gini_from_auc(float("nan")))

    def test_ks_statistic(self, synthetic_probas):
        y_true, y_proba = synthetic_probas
        ks = ks_statistic(y_true, y_proba[:, 1])
        assert 0.0 <= ks <= 1.0


class TestLGDMetrics:

    def test_perfect_predictions(self):
        y_true = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        y_pred = y_true.copy()
        m = calculate_lgd_metrics(y_true, y_pred)
        assert m["R2"] == pytest.approx(1.0)
        assert m["RMSE"] == pytest.approx(0.0, abs=1e-9)
        assert m["MAE"] == pytest.approx(0.0, abs=1e-9)
        assert m["Pearson_Corr"] == pytest.approx(1.0)
        assert m["Spearman_Corr"] == pytest.approx(1.0)

    def test_mape_zero_exclusion_bookkeeping(self):
        y_true = np.array([0.0, 0.0, 0.5, 1.0])
        y_pred = np.array([0.1, 0.1, 0.6, 0.9])
        m = calculate_lgd_metrics(y_true, y_pred)
        assert m["MAPE_n_zeros_excluded"] == 2
        assert m["MAPE_pct_zeros_excluded"] == pytest.approx(50.0)
        # MAPE computed on the two non-zero rows only
        # |(0.5 - 0.6)/0.5| = 0.2; |(1.0 - 0.9)/1.0| = 0.1; mean = 0.15 -> 15%
        assert m["MAPE"] == pytest.approx(15.0)

    def test_all_zero_targets(self):
        y_true = np.zeros(5)
        y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        m = calculate_lgd_metrics(y_true, y_pred)
        assert np.isnan(m["MAPE"])
        assert m["MAPE_n_zeros_excluded"] == 5
