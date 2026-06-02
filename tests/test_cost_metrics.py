"""Tests for src.methods.cost_metrics (B9)."""

from __future__ import annotations

import numpy as np
import pytest

from src.methods.cost_metrics import (
    CostMatrix,
    cost_sensitive_summary,
    expected_loss,
    profit_curve,
)


class TestExpectedLoss:

    def test_perfect_predictor_beats_baseline(self, synthetic_probas):
        y_true, y_proba = synthetic_probas
        # A perfect classifier (proba = y_true): loss should be << 1
        proba = y_true.astype(float)
        proba[proba == 0] = 0.01
        proba[proba == 1] = 0.99
        loss = expected_loss(y_true, proba, cost_ratio=5.0)
        assert loss < 1.0

    def test_random_predictor_around_baseline(self):
        rng = np.random.default_rng(0)
        y_true = (rng.random(500) < 0.3).astype(int)
        proba = rng.random(500)  # pure noise
        loss = expected_loss(y_true, proba, cost_ratio=5.0)
        # Random should be within 0.5x to 2x of the baseline
        assert 0.5 <= loss <= 2.0


class TestProfitCurve:

    def test_optimum_exists(self, synthetic_probas):
        y_true, y_proba = synthetic_probas
        thresholds, profits, opt_t, opt_p = profit_curve(y_true, y_proba[:, 1])
        assert thresholds.shape == profits.shape
        assert 0.0 <= opt_t <= 1.0
        # Optimum must dominate the all-positive and all-negative extremes
        assert opt_p >= profits[0] - 1e-9
        assert opt_p >= profits[-1] - 1e-9

    def test_zero_cost_matrix_is_constant(self):
        rng = np.random.default_rng(1)
        y_true = (rng.random(50) < 0.5).astype(int)
        proba = rng.random(50)
        costs = CostMatrix(cost_fn=0, cost_fp=0, benefit_tn=0, benefit_tp=0)
        _, profits, _, _ = profit_curve(y_true, proba, costs=costs)
        assert np.allclose(profits, 0.0)


class TestSummary:

    def test_keys_present(self, synthetic_probas):
        y_true, y_proba = synthetic_probas
        s = cost_sensitive_summary(y_true, y_proba)
        expected = {"Expected_Loss_Normalized", "Optimal_Profit_Threshold",
                    "Optimal_Profit", "H_Measure"}
        assert expected <= set(s.keys())
