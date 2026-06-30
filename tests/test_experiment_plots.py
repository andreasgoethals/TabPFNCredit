import numpy as np
import pytest

from src.visualizations.experiment_plots import _relative_metric_gain


def test_r2_relative_gain_uses_baseline_value_like_auc():
    gain = _relative_metric_gain(np.array([0.84]), np.array([0.80]), "R2")
    assert gain[0] == pytest.approx(5.0)


def test_auc_relative_gain_keeps_baseline_value_denominator():
    gain = _relative_metric_gain(np.array([0.84]), np.array([0.80]), "AUC")
    assert gain[0] == pytest.approx(5.0)


def test_lower_is_better_relative_gain_is_positive_when_foundation_improves():
    gain = _relative_metric_gain(
        np.array([0.18]), np.array([0.20]), "Brier", higher_is_better=False
    )
    assert gain[0] == pytest.approx(10.0)
