import os
import streamlit as st
import pandas as pd
import pathlib
from plots import (
    plot_training_time_bar,
    plot_training_time_vs_metric,
    plot_model_metric_heatmap,
    plot_model_boxplot,
    plot_metric_bar_avg_by_model,
    plot_combined_metric_bar,
    plot_combined_boxplot,
    plot_combined_metric_heatmap,
    # plot_combined_radar,
    plot_combined_radar_interactive,
    plot_pairwise_comparison_table,
    plot_relative_rank,
    plot_ranking_bar,
)

from app_helpers import generate_meta_results_for_task

OUTPUTS_DIR = "outputs"
TASK_SUBFOLDERS = [
    f for f in os.listdir(OUTPUTS_DIR)
    if os.path.isdir(os.path.join(OUTPUTS_DIR, f))
    and not f.startswith("analysis")
    and not f.startswith("logs")
]

task = st.sidebar.selectbox("Select Task", TASK_SUBFOLDERS, index=0)
task_dir = os.path.join(OUTPUTS_DIR, task)

# --- Generate meta results on app load ---
results_files_by_dataset, combined_meta_path = generate_meta_results_for_task(task_dir)

# ---- DATASET SELECTION ----
dataset_options = ["All Datasets (combined)" if d == "total_combined" else d
                   for d in sorted(results_files_by_dataset.keys())]
selected_dataset = st.sidebar.selectbox("Select Dataset", dataset_options, index=0)

if selected_dataset == "All Datasets (combined)":
    dataset_path = os.path.join(task_dir, "total_combined")
    results_files = ["total_combined_meta_results.csv"]
else:
    dataset_path = os.path.join(task_dir, selected_dataset)
    results_files = results_files_by_dataset[selected_dataset]

# Pick result file for display (usually meta or a run)
selected_file = st.sidebar.selectbox(
    "Select results file", results_files, index=0
)

# --- MAIN AREA ---
st.title("TABPFN Credit Benchmark Results")
all_dfs = {}
csv_path = os.path.join(dataset_path, selected_file)
df = pd.read_csv(csv_path)

if 'tuning' in df.columns:
    df['tuning'] = df['tuning'].astype(str)
st.subheader(f"{selected_dataset}: {selected_file}")
st.dataframe(df)

all_dfs[f"{selected_dataset}/{selected_file}"] = df


# # ---- PLOTS ----
MIN_TRAIN_TIME = 20  # in seconds

if not selected_file.endswith("_meta_results.csv"):
    st.markdown("### Bar Plot (average over splits)")
    # Metric selection
    plot_metrics = ["accuracy", "brier", "f1", "precision", "recall", "aucroc", "aucpr", "h_measure", "training_time"]
    available_metrics = [col for col in df.columns if col in plot_metrics and df[col].dtype in ['float64', 'int64']]
    metric = st.selectbox("Select the evaluation metric", available_metrics, index=0)

    plot_metric_bar_avg_by_model(df, metric, title=f"{metric} by Model (mean over folds)")

    # Training time plot
    st.markdown(f"### Training Time (only models ≥ {MIN_TRAIN_TIME}s)")
    plot_training_time_bar(df, min_time=MIN_TRAIN_TIME)


# Check if this is a meta file (endswith "_meta_results.csv") and only one selected file
if selected_file.endswith("_meta_results.csv"):
    st.header("Meta Results Plots")

    plot_metrics = ["accuracy", "brier", "f1", "precision", "recall", "aucroc", "aucpr", "h_measure", "training_time"]
    available_metrics = [col for col in df.columns if col in plot_metrics and df[col].dtype in ['float64', 'int64']]
    metric = st.selectbox("Select metric for bar plot", available_metrics, index=0)

    if selected_dataset == "All Datasets (combined)":
        # ---- COMBINED ALL DATASETS: Show all comparative plots ----
        st.markdown("### [A] Bar Plot (All Datasets, Average per Model)")
        plot_combined_metric_bar(df, metric, title=f"{metric} by Model (All Datasets)")

        st.markdown("### [B] Boxplot of Metric by Model (All Datasets)")
        plot_combined_boxplot(df, metric)

        st.markdown("### [C] Model × Metric Heatmap (All Datasets)")
        plot_combined_metric_heatmap(df, [m for m in available_metrics if m != "training_time"])

        # Training Time vs. Metric
        st.markdown("### [D] Training Time vs. Metric")
        min_train_time = st.slider("Min training time for scatter (seconds)", min_value=0, max_value=600, value=60,
                                   step=10)

        # Add min_metric slider for currently selected metric (dynamically set range)
        metric_min = float(df[metric].min())
        metric_max = float(df[metric].max())
        metric_default = float(df[metric].quantile(0.1))  # Or set to metric_min if you prefer
        min_metric = st.slider(f"Min {metric} for scatter", min_value=metric_min, max_value=metric_max,
                               value=metric_default, step=0.01)

        plot_training_time_vs_metric(df, metric, min_time=min_train_time, min_metric=min_metric)

        st.markdown("### [E] Radar/Spider Plot: Model Comparison")
        radar_metrics = st.multiselect("Select metrics for radar plot", [m for m in available_metrics if m != "training_time"], default=["accuracy", "aucroc", "f1"])
        if len(radar_metrics) >= 3:
            plot_combined_radar_interactive(df, radar_metrics)
        else:
            st.info("Select at least 3 metrics for radar plot.")

        st.markdown(f"### [F] Pairwise Comparison Table for {metric}")
        plot_pairwise_comparison_table(df, metric)

        st.markdown("### [G] Relative Rank Plot")
        plot_relative_rank(df, metric)

    else:
        # ---- SINGLE DATASET META PLOTS (as before) ----
        st.markdown("#### Bar Plot (Mean over Folds)")
        plot_metric_bar_avg_by_model(df, metric, title=f"{metric} by Model (mean over folds)")

        st.markdown("#### Boxplot of Metric by Model")
        plot_model_boxplot({selected_dataset: df}, metric)

        st.markdown("#### Model × Metric Heatmap")
        plot_model_metric_heatmap(df, [m for m in available_metrics if m != "training_time"])

        st.markdown("#### Training Time vs. Metric (for slower models)")
        min_train_time = st.slider("Min training time for scatter (seconds)", min_value=0, max_value=600, value=60, step=10)
        plot_training_time_vs_metric(df, metric, min_time=min_train_time)

# --- RANKING CSV (aggregated ranks from outputs/analysis) ---
ranking_file = pathlib.Path("outputs/analysis/pd_ranking.csv")
if ranking_file.exists():
    st.sidebar.markdown("---")
    if st.sidebar.checkbox("Show Aggregated Ranking Table & Plot", value=False):
        st.header("Aggregated Rankings (Wide Table)")
        rank_df = pd.read_csv(ranking_file)
        st.dataframe(rank_df)

        # choose a metric_dataset column
        metric_cols = [c for c in rank_df.columns if c != "model"]
        selected_rank_col = st.selectbox("Select ranking column to plot", metric_cols, index=0)

        plot_ranking_bar(rank_df, selected_rank_col)
