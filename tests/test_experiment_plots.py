import numpy as np
import pandas as pd
import pytest

from src.visualizations import experiment_plots as ep
from src.visualizations.experiment_plots import _relative_metric_gain, _sweep_window_size


def _write_pd_oof(path, y_true, y_prob):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fold_1_y_true=np.asarray(y_true[: len(y_true) // 2]),
        fold_1_y_prob=np.asarray(y_prob[: len(y_prob) // 2]),
        fold_2_y_true=np.asarray(y_true[len(y_true) // 2 :]),
        fold_2_y_prob=np.asarray(y_prob[len(y_prob) // 2 :]),
    )


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


def test_calibration_bias_table_pools_out_of_fold_probabilities(tmp_path):
    result = tmp_path / "experiment1" / "pd" / "dataset_a" / "method_a__HPO.npz"
    y_true = np.array([0, 1, 0, 0])
    y_prob = np.column_stack([1 - np.array([0.2, 0.6, 0.1, 0.3]),
                              np.array([0.2, 0.6, 0.1, 0.3])])
    _write_pd_oof(result, y_true, y_prob)
    df = pd.DataFrame({"dataset": ["dataset_a"], "method": ["method_a"]})

    table = ep.calibration_bias_table(df, results_root=tmp_path, task="pd")

    assert len(table) == 1
    assert table.loc[0, "observed_mean"] == pytest.approx(0.25)
    assert table.loc[0, "predicted_mean"] == pytest.approx(0.30)
    assert table.loc[0, "calibration_bias"] == pytest.approx(-0.05)
    assert table.loc[0, "n_folds"] == 2


def test_selected_calibration_summary_has_four_panels_and_requested_order(monkeypatch):
    methods = ["tabpfn_v3", "tabicl_v2", "catboost", "LogReg"]
    table = pd.DataFrame({
        "dataset": ["d1", "d2"] * 4,
        "method": [method for method in methods for _ in range(2)],
        "observed_mean": [0.10, 0.20] * 4,
        "predicted_mean": [0.11, 0.19, 0.12, 0.18, 0.09, 0.21, 0.10, 0.20],
    })
    table["calibration_bias"] = table["observed_mean"] - table["predicted_mean"]
    captured = {}

    def fake_save(fig, out_dir, stem):
        captured["stem"] = stem
        captured["axes"] = len(fig.axes)
        captured["labels"] = [tick.get_text() for tick in fig.axes[0].get_xticklabels()]
        ep.plt.close(fig)
        return None

    monkeypatch.setattr(ep, "_save", fake_save)

    ep.selected_method_calibration_summary(table, task="pd")

    assert captured["stem"] == "pd_selected_calibration_summary"
    assert captured["axes"] == 4
    assert captured["labels"] == ["TabPFN-3", "TabICLv2", "CatBoost", "log. reg"]


def test_calibration_decile_curve_bins_by_rank_and_averages(tmp_path, monkeypatch):
    # Known preds/labels, n_bins=5 -> per-bin means are exact and checkable.
    result = tmp_path / "experiment1" / "pd" / "d1" / "m1__HPO.npz"
    preds = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    _write_pd_oof(result, y_true, np.column_stack([1 - preds, preds]))
    df = pd.DataFrame({"dataset": ["d1"], "method": ["m1"]})
    captured = {}

    def fake_save(fig, out_dir, stem):
        ax = fig.axes[0]
        captured["stem"] = stem
        # line 0 is the y = x diagonal; line 1 the method's decile curve.
        captured["x"] = ax.lines[1].get_xdata()
        captured["y"] = ax.lines[1].get_ydata()
        ep.plt.close(fig)
        return None

    monkeypatch.setattr(ep, "_save", fake_save)
    ep.calibration_decile_curve(df, results_root=tmp_path, task="pd",
                                methods=("m1",), n_bins=5)

    assert captured["stem"] == "pd_calibration_deciles"
    assert captured["x"] == pytest.approx([10.0, 30.0, 50.0, 70.0, 90.0])
    assert captured["y"] == pytest.approx([0.0, 0.0, 50.0, 100.0, 100.0])


def test_calibration_bias_vs_default_rate_uses_log_x(monkeypatch):
    table = pd.DataFrame({
        "dataset": ["d1", "d2", "d1", "d2"],
        "method": ["m1", "m1", "m2", "m2"],
        "observed_mean": [0.03, 0.20, 0.03, 0.20],
        "predicted_mean": [0.04, 0.18, 0.02, 0.22],
    })
    table["calibration_bias"] = table["observed_mean"] - table["predicted_mean"]
    captured = {}

    def fake_save(fig, out_dir, stem):
        captured["stem"] = stem
        captured["xscale"] = fig.axes[0].get_xscale()
        ep.plt.close(fig)
        return None

    monkeypatch.setattr(ep, "_save", fake_save)
    ep.calibration_bias_vs_default_rate(table, task="pd", methods=("m1", "m2"))

    assert captured["stem"] == "pd_calibration_bias_vs_default_rate"
    assert captured["xscale"] == "log"


def test_imbalance_trend_uses_processed_minority_proportion(monkeypatch):
    from src.data import dataset_inventory

    proportions = {"d1": 0.05, "d2": 0.20}
    monkeypatch.setattr(
        dataset_inventory, "minority_proportion", lambda task, dataset: proportions[dataset]
    )
    captured = {}

    def fake_save(fig, out_dir, stem):
        captured["stem"] = stem
        captured["x"] = sorted(
            float(value)
            for collection in fig.axes[0].collections
            for value in collection.get_offsets()[:, 0]
        )
        ep.plt.close(fig)
        return None

    monkeypatch.setattr(ep, "_save", fake_save)
    df = pd.DataFrame({
        "dataset": ["d1", "d1", "d2", "d2"],
        "method": ["tabpfn_v3", "catboost", "tabpfn_v3", "catboost"],
        "metric.AUC": [0.80, 0.75, 0.76, 0.77],
    })

    ep.foundation_vs_baseline_imbalance_trend(
        df, metric="AUC", task_name="PD"
    )

    assert captured["stem"] == "pd_tabpfn_v3_vs_catboost_imbalancetrend_auc"
    assert captured["x"] == pytest.approx([5.0, 20.0])


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


def test_smooth_r2_learning_curve_y_axis_uses_plotted_line(monkeypatch):
    captured = {}

    def fake_save(fig, out_dir, stem):
        ax = fig.axes[0]
        captured["ylim"] = ax.get_ylim()
        captured["ydata"] = ax.lines[0].get_ydata()
        ep.plt.close(fig)
        return None

    monkeypatch.setattr(ep, "_save", fake_save)
    df = pd.DataFrame({
        "method": ["m1", "m1", "m1", "m1"],
        "sweep_axis": ["row_limit"] * 4,
        "sweep_value": [100, 800, 1200, 2000],
        "metric.R2": [0.42, 0.48, 0.54, 0.60],
    })

    ep.learning_curve(df, "R2", task_name="LGD", smooth=True)

    assert min(captured["ydata"]) > 0.3
    assert captured["ylim"][0] > 0.3


def test_combined_r2_learning_curve_y_axis_uses_plotted_points(monkeypatch):
    captured = {}

    def fake_save(fig, out_dir, stem):
        captured["ylim"] = fig.axes[0].get_ylim()
        ep.plt.close(fig)
        return None

    monkeypatch.setattr(ep, "_save", fake_save)
    df = pd.DataFrame({
        "method": ["m1", "m1", "m1", "m1"],
        "sweep_axis": ["row_limit"] * 4,
        "sweep_value": [100, 800, 1200, 2000],
        "metric.R2": [0.42, 0.48, 0.54, 0.60],
    })

    ep.learning_curve_moving_average_with_dots(df, "R2", task_name="LGD")

    assert captured["ylim"][0] > 0.3


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
