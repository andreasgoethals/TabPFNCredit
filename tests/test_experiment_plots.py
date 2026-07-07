import numpy as np
import pandas as pd
import pytest

from src.visualizations import experiment_plots as ep
from src.visualizations.experiment_plots import _relative_metric_gain, _sweep_window_size


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


def test_sweep_window_variants_keep_standard_rule_and_bracket_it():
    assert _sweep_window_size(120, "standard") == 10
    assert _sweep_window_size(120, "less") == 6
    assert _sweep_window_size(120, "more") == 15


def test_learning_curve_title_stays_general(monkeypatch):
    captured = {}

    def fake_save(fig, out_dir, stem):
        captured["title"] = fig.axes[0].get_title()
        captured["inset_titles"] = [ax.get_title() for ax in fig.axes[0].child_axes]
        ep.plt.close(fig)
        return None

    monkeypatch.setattr(ep, "_save", fake_save)
    df = pd.DataFrame({
        "method": ["m1", "m1", "m1", "m2", "m2", "m2"],
        "sweep_axis": ["row_limit"] * 6,
        "sweep_value": [100, 800, 1200, 100, 800, 1200],
        "metric.AUC": [0.70, 0.78, 0.80, 0.68, 0.76, 0.82],
    })

    ep.learning_curve(df, "AUC", task_name="PD", zoom=True)

    assert captured["title"] == "PD learning curve"
    assert captured["inset_titles"] == ["Low-data range"]
    assert "AUC" not in captured["title"]
    assert "1000" not in captured["title"]


def test_combined_r2_learning_curve_title_has_no_rendering_note(monkeypatch):
    captured = {}

    def fake_save(fig, out_dir, stem):
        captured["title"] = fig.axes[0].get_title()
        ep.plt.close(fig)
        return None

    monkeypatch.setattr(ep, "_save", fake_save)
    df = pd.DataFrame({
        "method": ["m1", "m1", "m1"],
        "sweep_axis": ["row_limit"] * 3,
        "sweep_value": [100, 800, 1200],
        "metric.R2": [-0.2, 0.1, 0.2],
    })

    ep.learning_curve_moving_average_with_dots(df, "R2", task_name="LGD")

    assert captured["title"] == "LGD learning curve"
    assert "below 0" not in captured["title"]


def test_smooth_window_variant_changes_filename(monkeypatch):
    stems = []

    def fake_save(fig, out_dir, stem):
        stems.append(stem)
        ep.plt.close(fig)
        return None

    monkeypatch.setattr(ep, "_save", fake_save)
    df = pd.DataFrame({
        "method": ["m1", "m1", "m1", "m1"],
        "sweep_axis": ["row_limit"] * 4,
        "sweep_value": [100, 200, 300, 400],
        "metric.AUC": [0.70, 0.72, 0.73, 0.74],
    })

    ep.learning_curve(df, "AUC", task_name="PD", smooth=True, smooth_window="less")
    ep.learning_curve(df, "AUC", task_name="PD", smooth=True, smooth_window="standard")
    ep.learning_curve(df, "AUC", task_name="PD", smooth=True, smooth_window="more")
    ep.learning_curve_moving_average_with_dots(df, "AUC", task_name="PD", smooth_window="more")

    assert stems == [
        "pd_learning_curve_auc_smooth_less",
        "pd_learning_curve_auc_smooth",
        "pd_learning_curve_auc_smooth_more",
        "pd_learning_curve_auc_combined_more",
    ]
