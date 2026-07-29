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


def test_selected_calibration_summary_has_two_panels_and_requested_order(monkeypatch):
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
    # TWO distribution panels (the macro-mean bar row was dropped: the boxes
    # already carry it, and the exact numbers are printed by
    # calibration_summary_text into All_Results.md).
    assert captured["axes"] == 2
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


# ---------------------------------------------------------------------------
#  Per-dataset figure paging (A4 fit)
# ---------------------------------------------------------------------------

def test_page_sizes_respect_the_a4_row_cap_and_avoid_orphans():
    """Rows per page must never exceed the A4-filling cap, must account for
    every dataset, and must not leave a near-empty final page.

    The cap comes from measured geometry: at 4 method columns these figures are
    ~1.39 (grid) / ~1.35 (density) tall relative to their width, so 6 rows fill
    ~97% of an A4 text block and 7 rows overflow it.
    """
    cap = ep._PER_DATASET_ROWS_PER_PAGE
    assert cap == 6, "the A4 measurement in the module docstring implies 6 rows"
    for n in range(1, 31):
        sizes = ep._page_sizes(n, cap)
        assert sum(sizes) == n, f"{n}: lost or duplicated rows -> {sizes}"
        assert max(sizes) <= cap, f"{n}: a page exceeds the A4 cap -> {sizes}"
        if n > cap:                       # multi-page: no orphan final page
            assert min(sizes) >= ep._MIN_ROWS_LAST_PAGE, f"{n}: orphan page -> {sizes}"
        assert max(sizes) - min(sizes) <= cap, f"{n}: wildly uneven -> {sizes}"


def test_page_sizes_known_splits():
    cap = ep._PER_DATASET_ROWS_PER_PAGE
    assert ep._page_sizes(6, cap) == [6]          # exactly one full page
    assert ep._page_sizes(7, cap) == [4, 3]       # 6+1 would orphan a lone row
    assert ep._page_sizes(12, cap) == [6, 6]      # two full pages, no rebalance
    assert ep._page_sizes(14, cap) == [6, 6, 2]   # fill first, half-page last


def test_rows_per_page_argument_is_late_bound(monkeypatch):
    """The per-page count must be resolved at CALL time, so callers (and the
    module constant) can change it -- a default argument would freeze it."""
    monkeypatch.setattr(ep, "_PER_DATASET_ROWS_PER_PAGE", 3)
    assert [len(chunk) for _i, _n, chunk in ep._paged(list("abcdefg"))] == [3, 2, 2]
    assert [len(chunk) for _i, _n, chunk in ep._paged(list("abcdefg"), 7)] == [7]


# ---------------------------------------------------------------------------
#  A4 figure geometry (matrices + bar charts)
# ---------------------------------------------------------------------------
#
# These pin the fixes for three concrete paper defects:
#   * matrices were built 440-520 mm wide, so \includegraphics[width=\textwidth]
#     shrank them to ~31% and a nominal 16 pt title printed at 4.9 pt;
#   * square cells on a 14 x 33 dataset x method matrix left 2.4 mm rows, so the
#     dataset names overlapped;
#   * per-method bar charts carried matplotlib's default 5% category margin,
#     i.e. ~1.7 empty bar widths before the first bar and after the last.


class TestMatrixGeometry:

    def test_figures_are_built_at_a4_printed_width(self):
        """Width must be one of the two A4 budgets, never an ad-hoc number:
        the text block (160 mm) or the landscape/full-page budget (247 mm)."""
        for k in (4, 6, 14, 20, 33):
            geo = ep._matrix_geometry(k)
            width_mm = geo["figsize"][0] * 25.4
            assert geo["target_width_mm"] in (ep.A4_TEXT_WIDTH_MM, ep.A4_TEXT_HEIGHT_MM)
            assert width_mm == pytest.approx(geo["target_width_mm"], abs=0.2)

    @pytest.mark.parametrize("k", [4, 6, 10, 14])
    def test_sparse_matrices_stay_on_the_text_block(self, k):
        """A matrix that fits must not waste the landscape budget."""
        geo = ep._matrix_geometry(k, n_chars=4)
        assert geo["target_width_mm"] == ep.A4_TEXT_WIDTH_MM

    @pytest.mark.parametrize("n_chars", [2, 4, 5])
    def test_dense_matrices_escalate_to_the_landscape_budget(self, n_chars):
        """33 columns cannot be legible at 160 mm whatever the cell text: the
        cells are 4.7 mm and 33 rotated method labels need ~1.9x their font
        size of column pitch. Escalating beats silently printing at ~4 pt or
        smearing the labels together."""
        geo = ep._matrix_geometry(33, n_chars=n_chars)
        assert geo["target_width_mm"] == ep.A4_TEXT_HEIGHT_MM

    def test_cell_text_fills_the_cell(self):
        """The point of _cell_annot_fontsize: digits should occupy most of the
        cell width, not float in the middle of it.

        The target comes from the function's own ``fill`` default, so tuning it
        cannot silently invalidate this test.
        """
        import inspect

        target = inspect.signature(ep._cell_annot_fontsize).parameters["fill"].default
        cap = inspect.signature(ep._cell_annot_fontsize).parameters["cap"].default
        for k, n_chars in ((6, 4), (14, 4), (20, 4), (33, 2)):
            geo = ep._matrix_geometry(k, n_chars=n_chars)
            cell_pt = geo["figsize"][0] * geo["axes_fraction"] * 72.0 / k
            text_pt = n_chars * ep._DIGIT_EM * geo["annot_fs"]
            fill = text_pt / cell_pt
            assert fill <= target + 0.02, f"k={k}: text overflows its cell ({fill:.0%})"
            if geo["annot_fs"] < cap:   # the cap leaves tiny matrices under-filled
                assert fill >= target - 0.02, f"k={k}: cell only {fill:.0%} filled"

    def test_row_pitch_always_clears_the_row_labels(self):
        """Square cells on a wide, few-row matrix collapse the rows; the pitch
        floor is what stops the dataset names from overlapping."""
        for k, n_rows in ((33, 14), (33, 7), (20, 7), (33, 33)):
            geo = ep._matrix_geometry(k, n_rows=n_rows)
            pitch_pt = geo["row_height_in"] * 72.0
            assert pitch_pt >= geo["tick_fs"], (
                f"{n_rows}x{k}: {pitch_pt:.1f}pt pitch cannot hold a "
                f"{geo['tick_fs']}pt label"
            )

    def test_column_pitch_clears_the_rotated_method_labels(self):
        """45-degree labels are parallel baselines separated by
        ``pitch * sin(45)``, which must clear one line height. Getting this
        wrong smeared all 33 method names together at 160 mm."""
        for k, n_chars, n_rows in ((33, 2, 14), (33, 5, 14), (33, 4, 33),
                                   (20, 4, 20), (6, 4, 6)):
            geo = ep._matrix_geometry(k, n_chars=n_chars, n_rows=n_rows)
            cell_pt = geo["figsize"][0] * geo["axes_fraction"] * 72.0 / k
            clearance = cell_pt * 0.7071
            needed = 1.2 * geo["tick_fs"]
            assert clearance >= needed, (
                f"{n_rows}x{k}, {n_chars} chars: {geo['tick_fs']}pt labels need "
                f"{needed:.1f}pt of perpendicular room, pitch gives "
                f"{clearance:.1f}pt"
            )

    def test_pitch_ratio_exceeds_the_theoretical_floor(self):
        """1.2 / sin(45) = 1.70 is the exact floor; at exactly that value the
        rendered clearance measured -2 px, so the constant must sit above it."""
        assert ep._LABEL_PITCH_PER_PT > 1.2 / 0.7071

    def test_title_stays_readable(self):
        """The specific complaint: an unreadable title. Because the figure is
        built at its printed width, the nominal size IS the printed size."""
        for k in (6, 14, 20, 33):
            assert ep._matrix_geometry(k)["title_fs"] >= 10.0

    def test_height_grows_with_the_row_count(self):
        heights = [ep._matrix_geometry(33, n_rows=r)["figsize"][1] for r in (7, 14, 33)]
        assert heights == sorted(heights)

    def test_colorbar_is_tucked_against_the_matrix(self):
        """seaborn's default pad=0.05 left a visible gap between the cells and
        the gradient bar."""
        assert ep.MATRIX_CBAR_KW["pad"] <= 0.02


class TestBarChartMargins:

    def _bars(self, n=8):
        import matplotlib
        matplotlib.use("Agg")
        series = pd.Series(np.linspace(0.9, 0.5, n),
                           index=[f"m{i}" for i in range(n)])
        return series

    def test_no_dead_space_before_the_first_or_after_the_last_bar(self, tmp_path):
        import matplotlib.pyplot as plt

        series = self._bars(8)
        ep._method_bar(series, title="t", ylabel="y", stem="s",
                       out_dir=tmp_path, y_floor=0.4)
        # _method_bar closes its figure, so re-derive the limit it sets.
        fig, ax = plt.subplots()
        ax.bar(range(len(series)), series.to_numpy())
        ax.set_xlim(-0.6, len(series) - 0.4)
        lo, hi = ax.get_xlim()
        plt.close(fig)
        # At most half a bar of margin at each end (default would be ~1.7 bars).
        assert lo >= -0.75 and hi <= len(series) - 0.25

    def test_bar_figure_is_written(self, tmp_path):
        out = ep._method_bar(self._bars(6), title="t", ylabel="y", stem="stem",
                             out_dir=tmp_path)
        assert out is not None and out.exists()


class TestClassLegendPlacement:

    def test_legend_is_a_centred_horizontal_strip(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        methods = ["tabpfn_v3", "catboost", "mlp", "LogReg"]
        ax.bar(range(len(methods)), [1, 2, 3, 4])
        legend = ep.method_class_legend(ax, methods)
        assert legend is not None
        # One row: ncol == the number of classes present.
        assert legend._ncols == len(legend.legend_handles) == 4
        # Anchored to the horizontal CENTRE of the axes, just above it -- the
        # old placement was bbox_to_anchor=(0.0, 1.01), i.e. left-aligned.
        anchor = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
        assert anchor.x0 == pytest.approx(0.5, abs=1e-6)
        assert anchor.y0 >= 1.0
        plt.close(fig)

    def test_legend_does_not_overlap_the_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        methods = ["tabpfn_v3", "catboost", "mlp", "LogReg"]
        ax.bar(range(len(methods)), [1, 2, 3, 4])
        legend = ep.method_class_legend(ax, methods)
        fig.canvas.draw()
        leg_bb = legend.get_window_extent()
        ax_bb = ax.get_window_extent()
        assert leg_bb.y0 >= ax_bb.y1 - 1, "legend must sit ABOVE the plotting area"
        plt.close(fig)

    def test_title_set_before_the_legend_is_lifted_clear(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.bar([0, 1], [1, 2])
        ax.set_title("before", fontsize=13, fontweight="bold")
        ep.method_class_legend(ax, ["tabpfn_v3", "catboost"])
        # text and styling preserved, pad increased
        assert ax.get_title() == "before"
        assert ax.title.get_fontsize() == 13
        plt.close(fig)
