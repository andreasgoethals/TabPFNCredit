"""Caption generator: every known figure stem gets a fitting, non-fallback caption."""
from __future__ import annotations

from src.utils.generate_captions import caption_for, generate_captions, S_FALLBACK

# Representative stems from every figure family actually produced.
KNOWN = [
    "pd_heatmap_auc", "pd_bar_mean_brier", "pd_box_f1", "pd_rank_matrix_auc",
    "pd_ranking_auc", "pd_rank_box_auc", "pd_hpo_effect_auc", "pd_bar_compute_time",
    "pd_box_compute_time", "pd_cost_quality_auc",
    "pd_tabpfn_v3_vs_catboost_sizetrend_auc", "pd_tabpfn_v3_vs_catboost_scatter_auc",
    "pd_learning_curve_auc", "pd_learning_curve_auc_relative_smooth",
    "pd_learning_curve_auc_logx", "pd_learning_curve_auc_smooth_logx",
    "pd_imbalance_curve_ap_normalized", "pd_imbalance_curve_ap_normalized_smooth",
    "pd_imbalance_curve_auc_logx",
    "pd_row_limit_0003_vehicle_loan_auc", "pd_minority_proportion_0002_taiwan_creditcard_auc",
    "pd_pama", "pd_pama_min2wins", "pd_cd_auc", "pd_winloss", "pd_significance",
    "lgd_heatmap_r2", "lgd_bar_mean_pearson_corr", "lgd_cd_rmse",
    "lgd_tabpfn_v3_vs_catboost_scatter_r2", "pd_dataset_sizes", "lgd_target_hists",
    "pd_tabpfn_v3_vs_LogReg_scatter_auc", "lgd_tabpfn_v3_vs_LinearRegression_sizetrend_r2",
]


def test_known_stems_are_recognised():
    bad = [s for s in KNOWN if caption_for(s)[0][0] == S_FALLBACK]
    assert not bad, f"unrecognised stems: {bad}"


def test_display_names_and_metrics_render():
    _, cap = caption_for("pd_tabpfn_v3_vs_catboost_scatter_auc")
    assert "TabPFN-3" in cap and "CatBoost" in cap and "AUC" in cap
    _, cap = caption_for("lgd_heatmap_r2")
    assert "R²" in cap
    _, cap = caption_for("pd_imbalance_curve_ap_normalized")
    assert "average precision" in cap.lower()          # multi-token metric parsed


def test_no_capitalize_mangling():
    # "AUC"/"the Brier score" must not be mangled by sentence-casing.
    assert "Auc" not in caption_for("pd_heatmap_auc")[1]
    assert "Mean the Brier" not in caption_for("pd_bar_mean_brier")[1]


def test_unknown_stem_falls_back_gracefully():
    key, cap = caption_for("pd_some_new_plot_xyz")
    assert key[0] == S_FALLBACK and "pd_some_new_plot_xyz" in cap


def test_matrix_views_order_by_metric_then_view():
    # AUC block (all six views) sorts before the Brier block.
    auc = [caption_for(f"pd_{v}_auc")[0] for v in
           ("heatmap", "bar_mean", "box", "rank_matrix", "ranking", "rank_box")]
    assert auc == sorted(auc)                          # already in view order
    assert caption_for("pd_rank_box_auc")[0] < caption_for("pd_heatmap_brier")[0]


def test_pama_min2wins_sorts_after_pama():
    # The "at least two wins" PAMA chart is a distinct figure, listed right after
    # the all-winners PAMA in each statistical chapter.
    assert caption_for("pd_pama")[0] < caption_for("pd_pama_min2wins")[0]
    assert "two" in caption_for("pd_pama_min2wins")[1].lower()


def test_logx_curves_sort_next_to_their_linear_sibling():
    # base < base_logx < smooth < smooth_logx < relative (so a log twin sits
    # right after its linear counterpart), and the caption notes the log axis.
    order = [caption_for(f"pd_learning_curve_auc{sfx}")[0] for sfx in
             ("", "_logx", "_smooth", "_smooth_logx", "_relative")]
    assert order == sorted(order)
    assert "log-scaled x-axis" in caption_for("pd_imbalance_curve_auc_logx")[1]


def test_regression_baselines_render_abbreviated():
    # The head-to-head captions must use the abbreviated baseline labels.
    assert "log. reg" in caption_for("pd_tabpfn_v3_vs_LogReg_scatter_auc")[1]
    assert "lin. reg" in caption_for("lgd_tabpfn_v3_vs_LinearRegression_sizetrend_r2")[1]


def test_generate_writes_one_consolidated_file_in_notebook_order(tmp_path):
    # Two chapters' figures, created out of order; output must be ONE file at the
    # figures root, chapters in notebook order, figures in generation order.
    for sub, stems in (("experiment1/pd", ["pd_bar_mean_auc", "pd_heatmap_auc"]),
                       ("experiment0", ["pd_heatmap_auc"])):
        (tmp_path / sub).mkdir(parents=True)
        for stem in stems:
            (tmp_path / sub / f"{stem}.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "empty").mkdir()
    written = generate_captions(tmp_path)
    assert written == [tmp_path / "CAPTIONS.md"]                 # single file, at the root
    txt = written[0].read_text(encoding="utf-8")
    # Experiment 0 chapter precedes Experiment 1.1, and within a chapter the
    # heatmap is listed before the bar (generation order), not alphabetically.
    assert txt.index("Experiment 0") < txt.index("Experiment 1.1")
    assert txt.index("`pd_heatmap_auc.pdf`") < txt.index("`pd_bar_mean_auc.pdf`")


def test_generate_removes_stale_per_directory_files(tmp_path):
    d = tmp_path / "experiment0"
    d.mkdir(parents=True)
    (d / "pd_pama.pdf").write_bytes(b"%PDF-1.4")
    (d / "CAPTIONS.md").write_text("old per-dir file", encoding="utf-8")   # stale
    generate_captions(tmp_path)
    assert not (d / "CAPTIONS.md").exists()                     # pruned
    assert (tmp_path / "CAPTIONS.md").exists()                  # single file written
