"""Tests for src.methods.method_metrics."""

from __future__ import annotations

from pathlib import Path

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


# ---------------------------------------------------------------------------
#  Adjusted Average Precision
# ---------------------------------------------------------------------------
#
# The reported key is AP_adjusted (paper label "Adjusted AP"). It was previously
# called AP_normalized and described as prevalence-invariant and attributed to
# Flach & Kull; the quantity is unchanged but that framing overstated it, so both
# the name and the wording are pinned here.

class TestAdjustedAveragePrecision:

    @staticmethod
    def _dev(y_true, scores):
        from src.methods.method_metrics import average_precision_deviation
        return average_precision_deviation(np.asarray(y_true), np.asarray(scores))

    def test_exactly_these_keys(self):
        out = self._dev([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert set(out) == {"AP", "AP_baseline", "AP_minus_baseline", "AP_adjusted"}
        assert "AP_normalized" not in out, "the old key must be gone"

    def test_formula_is_exact(self):
        out = self._dev([0, 0, 0, 1, 1, 1, 1], [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9])
        ap, pi = out["AP"], out["AP_baseline"]
        assert out["AP_minus_baseline"] == pytest.approx(ap - pi)
        assert out["AP_adjusted"] == pytest.approx((ap - pi) / (1.0 - pi))

    def test_baseline_is_the_positive_prevalence(self):
        out = self._dev([0, 0, 0, 1], [0.4, 0.3, 0.2, 0.1])
        assert out["AP_baseline"] == pytest.approx(0.25)

    def test_perfect_ranking_maps_to_one(self):
        out = self._dev([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.95])
        assert out["AP"] == pytest.approx(1.0)
        assert out["AP_adjusted"] == pytest.approx(1.0)

    def test_random_ranking_sits_near_zero(self):
        """The reference point the adjustment maps to 0."""
        rng = np.random.default_rng(0)
        y = np.repeat([0, 1], [900, 100])
        vals = [self._dev(y, rng.random(len(y)))["AP_adjusted"] for _ in range(30)]
        assert abs(float(np.mean(vals))) < 0.05

    def test_single_class_yields_nan_but_records_the_baseline(self):
        out = self._dev([1, 1, 1], [0.2, 0.5, 0.9])
        assert np.isnan(out["AP"]) and np.isnan(out["AP_adjusted"])
        assert out["AP_baseline"] == pytest.approx(1.0)

    def test_pd_metric_bundle_exposes_the_adjusted_key(self):
        """The full PD bundle is what gets written to every result file."""
        from src.methods.method_metrics import calculate_pd_metrics
        y = np.array([0, 0, 1, 1, 0, 1])
        p = np.array([0.1, 0.2, 0.7, 0.9, 0.3, 0.6])
        m = calculate_pd_metrics(y, p, (p >= 0.5).astype(int), threshold=0.5)
        assert "AP_adjusted" in m and "AP_normalized" not in m
        assert m["AP_adjusted"] == pytest.approx(
            (m["AP"] - m["AP_baseline"]) / (1.0 - m["AP_baseline"]))

    def test_documentation_does_not_overclaim(self):
        """No prevalence-invariance claim and no Flach & Kull attribution."""
        from src.methods import method_metrics

        source = Path(method_metrics.__file__).read_text(encoding="utf-8").lower()
        for banned in ("prevalence-invariant", "prevalence invariant", "flach",
                       "ap_normalized", "normalized ap"):
            assert banned not in source, f"{banned!r} still present"
        assert "does not remove all dependence on prevalence" in source or \
               "not make the metric independent" in source, \
               "the limitation must be stated explicitly"
