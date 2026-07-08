"""Caption generator: every known figure stem gets a fitting caption."""
from __future__ import annotations

from src.utils import generate_captions as gc
from src.utils.generate_captions import (
    caption_for,
    generate_captions,
    latex_caption_block,
    S_FALLBACK,
)

# Representative stems from every figure family actually produced.
KNOWN = [
    "pd_heatmap_auc", "pd_bar_mean_brier", "pd_box_f1", "pd_rank_matrix_auc",
    "pd_ranking_auc", "pd_rank_box_auc", "pd_hpo_effect_auc", "pd_bar_compute_time",
    "pd_box_compute_time", "pd_cost_quality_auc",
    "pd_tabpfn_v3_vs_catboost_sizetrend_auc", "pd_tabpfn_v3_vs_catboost_scatter_auc",
    "pd_learning_curve_auc", "pd_learning_curve_auc_zoom",
    "pd_learning_curve_auc_smooth_less", "pd_learning_curve_auc_smooth_more",
    "pd_learning_curve_auc_relative_smooth", "lgd_learning_curve_r2_zoom",
    "pd_learning_curve_auc_combined", "pd_learning_curve_auc_combined_less",
    "pd_learning_curve_auc_combined_more", "lgd_learning_curve_r2_combined",
    "pd_imbalance_curve_ap_normalized", "pd_imbalance_curve_ap_normalized_smooth",
    "pd_imbalance_curve_auc_zoom", "pd_imbalance_curve_ap_normalized_zoom",
    "pd_imbalance_curve_auc_combined",
    "pd_row_limit_0003_vehicle_loan_auc", "pd_minority_proportion_0002_taiwan_creditcard_auc",
    "pd_pama", "pd_pama_min2wins", "pd_cd_auc", "pd_winloss", "pd_significance",
    "lgd_heatmap_r2", "lgd_bar_mean_pearson_corr", "lgd_cd_rmse",
    "lgd_tabpfn_v3_vs_catboost_scatter_r2", "pd_dataset_sizes", "lgd_target_hists",
    "pd_tabpfn_v3_vs_LogReg_scatter_auc", "lgd_tabpfn_v3_vs_LinearRegression_sizetrend_r2",
]


def test_known_stems_are_recognised():
    bad = [s for s in KNOWN if caption_for(s)[0][0] == S_FALLBACK]
    assert not bad, f"unrecognised stems: {bad}"


def test_display_names_and_multitoken_metrics_parse():
    _, cap = caption_for("pd_tabpfn_v3_vs_catboost_scatter_auc")
    assert "TabPFN-3" in cap and "CatBoost" in cap
    _, cap = caption_for("pd_imbalance_curve_ap_normalized")
    assert "Curves average scores" in cap


def test_no_capitalize_mangling():
    # Captions should not sentence-case metric names into odd prose.
    assert "Auc" not in caption_for("pd_heatmap_auc")[1]
    assert "Mean the Brier" not in caption_for("pd_bar_mean_brier")[1]


def test_unknown_stem_falls_back_gracefully():
    key, cap = caption_for("pd_some_new_plot_xyz")
    assert key[0] == S_FALLBACK and r"pd\_some\_new\_plot\_xyz" in cap


def test_matrix_views_order_by_metric_then_view():
    # AUC block (all six views) sorts before the Brier block.
    auc = [caption_for(f"pd_{v}_auc")[0] for v in
           ("heatmap", "bar_mean", "box", "rank_matrix", "ranking", "rank_box")]
    assert auc == sorted(auc)
    assert caption_for("pd_rank_box_auc")[0] < caption_for("pd_heatmap_brier")[0]


def test_pama_min2wins_sorts_after_pama_without_expanding_acronym():
    # The at-least-two-wins PAMA chart is a distinct figure, listed right after
    # the all-winners PAMA in each statistical chapter.
    assert caption_for("pd_pama")[0] < caption_for("pd_pama_min2wins")[0]
    assert "two" in caption_for("pd_pama_min2wins")[1].lower()
    pama_cap = caption_for("pd_pama")[1]
    assert "Probability of achieving" not in pama_cap
    assert "maximal accuracy" not in pama_cap.lower()


def test_curve_variants_sort_in_notebook_order():
    # base < zoomed base < smooth variants < combined variants < relative views,
    # matching the analysis notebooks' pooled-curve order.
    order = [caption_for(f"pd_learning_curve_auc{sfx}")[0] for sfx in
             ("", "_zoom", "_smooth_less", "_smooth", "_smooth_more",
              "_combined_less", "_combined", "_combined_more", "_relative",
              "_relative_smooth")]
    assert order == sorted(order)
    assert "shorter window" in caption_for("pd_learning_curve_auc_smooth_less")[1]
    assert "standard window" in caption_for("pd_learning_curve_auc_smooth")[1]
    assert "longer window" in caption_for("pd_learning_curve_auc_smooth_more")[1]
    assert "shorter window" in caption_for("pd_learning_curve_auc_combined_less")[1]
    cap = caption_for("pd_imbalance_curve_auc_zoom")[1]
    assert "inset repeats" in cap
    assert "minority proportion <= 0.025" not in cap
    assert "y-axis spanning all shown points" not in cap
    cap = caption_for("pd_imbalance_curve_auc_combined")[1]
    assert "pooled score estimates" in cap
    assert "moving averages" in cap and "inset" not in cap


def test_learning_curve_captions_avoid_split_details():
    # Experiment 2's row_limit caps the dataset before the CV split, so captions
    # must never describe it as a training-set split.
    for stem in ("pd_learning_curve_auc", "pd_learning_curve_auc_zoom",
                 "lgd_learning_curve_r2_combined",
                 "pd_row_limit_0003_vehicle_loan_auc"):
        cap = caption_for(stem)[1]
        assert "training-set" not in cap, stem
        assert "cross-validation split" not in cap, stem


def test_regression_baselines_render_abbreviated():
    # The head-to-head captions must use the abbreviated baseline labels.
    assert "log. reg" in caption_for("pd_tabpfn_v3_vs_LogReg_scatter_auc")[1]
    assert "lin. reg" in caption_for("lgd_tabpfn_v3_vs_LinearRegression_sizetrend_r2")[1]


def test_size_trend_captions_explain_color_and_trend_encoding():
    _, auc_cap = caption_for("pd_tabpfn_v3_vs_catboost_sizetrend_auc")
    assert "green points favour TabPFN-3" in auc_cap
    assert "red points favour CatBoost" in auc_cap
    assert "log dataset size" in auc_cap
    assert "remaining unexplained variance" not in auc_cap

    _, r2_cap = caption_for("lgd_tabpfn_v3_vs_catboost_sizetrend_r2")
    assert "green points favour TabPFN-3" in r2_cap
    assert "red points favour CatBoost" in r2_cap
    assert "remaining unexplained variance" not in r2_cap

    _, lin_cap = caption_for("lgd_tabpfn_v3_vs_LinearRegression_sizetrend_r2")
    assert "red points favour lin. reg" in lin_cap
    assert "remaining unexplained variance" not in lin_cap


def test_latex_caption_blocks_are_copy_ready():
    block = latex_caption_block("pd_heatmap_auc")
    assert block.startswith(r"\caption{")
    assert r"\label{fig:pd-heatmap-auc}" in block
    assert "```" not in block


def test_curve_captions_avoid_rendering_notes():
    for stem in (
        "lgd_learning_curve_r2",
        "lgd_learning_curve_r2_zoom",
        "lgd_learning_curve_r2_combined",
        "pd_learning_curve_auc_zoom",
        "pd_imbalance_curve_auc_zoom",
    ):
        cap = caption_for(stem)[1]
        assert "displayed at the zero baseline" not in cap
        assert "rows <= 1000" not in cap
        assert "minority proportion <= 0.025" not in cap


def test_generate_writes_one_consolidated_file_in_notebook_order(tmp_path):
    # Two chapters' figures, created out of order; output must be one file at
    # the figures root, with chapters in notebook order and figures in
    # generation order.
    for sub, stems in (("experiment1/pd", ["pd_bar_mean_auc", "pd_heatmap_auc"]),
                       ("experiment0", ["pd_heatmap_auc"])):
        (tmp_path / sub).mkdir(parents=True)
        for stem in stems:
            (tmp_path / sub / f"{stem}.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "empty").mkdir()
    written = generate_captions(tmp_path)
    assert written == [tmp_path / "CAPTIONS.md"]
    txt = written[0].read_text(encoding="utf-8")
    assert txt.index("Experiment 0") < txt.index("Experiment 1.1")
    assert txt.index("`pd_heatmap_auc.pdf`") < txt.index("`pd_bar_mean_auc.pdf`")
    assert "```latex" in txt
    assert r"\caption{" in txt
    assert r"\label{fig:experiment0-pd-heatmap-auc}" in txt
    assert r"\label{fig:experiment1-pd-pd-heatmap-auc}" in txt


def test_generate_removes_stale_per_directory_files(tmp_path):
    d = tmp_path / "experiment0"
    d.mkdir(parents=True)
    (d / "pd_pama.pdf").write_bytes(b"%PDF-1.4")
    (d / "CAPTIONS.md").write_text("old per-dir file", encoding="utf-8")
    generate_captions(tmp_path)
    assert not (d / "CAPTIONS.md").exists()
    assert (tmp_path / "CAPTIONS.md").exists()


def test_saved_project_figure_refreshes_consolidated_captions(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "PROJECT_ROOT", tmp_path)
    figure = tmp_path / "figures" / "experiment0" / "pd_heatmap_auc.pdf"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"%PDF-1.4")

    gc.refresh_captions_for_saved_figure(figure)

    captions = tmp_path / "figures" / "CAPTIONS.md"
    assert captions.exists()
    assert "`pd_heatmap_auc.pdf`" in captions.read_text(encoding="utf-8")


def test_saved_figure_refresh_respects_disable_env(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("TABPFNCREDIT_AUTO_CAPTIONS", "0")
    figure = tmp_path / "figures" / "experiment0" / "pd_heatmap_auc.pdf"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"%PDF-1.4")

    gc.refresh_captions_for_saved_figure(figure)

    assert not (tmp_path / "figures" / "CAPTIONS.md").exists()
