import os
import streamlit as st
import pandas as pd
from plots import plot_metric_by_model, plot_grouped_bar_by_model

OUTPUTS_DIR = "outputs"
TASK_SUBFOLDERS = [
    f for f in os.listdir(OUTPUTS_DIR)
    if os.path.isdir(os.path.join(OUTPUTS_DIR, f))
    and not f.startswith("analysis")
    and not f.startswith("logs")
]

task = st.sidebar.selectbox("Select Task", TASK_SUBFOLDERS, index=0)
task_dir = os.path.join(OUTPUTS_DIR, task)

# Dataset selection
all_dataset_folders = [
    f for f in os.listdir(task_dir)
    if os.path.isdir(os.path.join(task_dir, f)) and not f.startswith("config")
]
selected_datasets = st.sidebar.multiselect(
    "Select Dataset(s)", sorted(all_dataset_folders), default=sorted(all_dataset_folders)[:1]
)

results_files_by_dataset = {}
for dataset in selected_datasets:
    dataset_path = os.path.join(task_dir, dataset)
    csv_files = [
        f for f in os.listdir(dataset_path)
        if f.endswith(".csv") and os.path.isfile(os.path.join(dataset_path, f))
    ]
    # --- Meta results creation (always create on app load) ---
    meta_name = f"{dataset}_meta_results.csv"
    meta_path = os.path.join(dataset_path, meta_name)
    if len(csv_files) > 1:
        dfs = [pd.read_csv(os.path.join(dataset_path, f)) for f in csv_files]
        meta_df = pd.concat(dfs, ignore_index=True)
        meta_df.to_csv(meta_path, index=False)
        if meta_name not in csv_files:
            csv_files.append(meta_name)
    # Always include meta file if exists
    if os.path.exists(meta_path) and meta_name not in csv_files:
        csv_files.append(meta_name)
    results_files_by_dataset[dataset] = sorted(csv_files, reverse=True)

# Results selection
selected_files = {}
for dataset, files in results_files_by_dataset.items():
    if files:
        chosen_files = st.sidebar.multiselect(
            f"Select results file(s) for {dataset}",
            files,
            default=files[:1],
            key=f"{dataset}_files"
        )
        selected_files[dataset] = chosen_files

# --- Main area ---
st.title("TABPFN Credit Benchmark Results")
all_dfs = {}
for dataset, files in selected_files.items():
    for result_file in files:
        csv_path = os.path.join(task_dir, dataset, result_file)
        df = pd.read_csv(csv_path)
        # For display, cast bool to str (fixes Streamlit blank checkbox issue)
        if 'tuning' in df.columns:
            df['tuning'] = df['tuning'].astype(str)
        st.subheader(f"{dataset}: {result_file}")
        st.dataframe(df)
        all_dfs[f"{dataset}/{result_file}"] = df

# --- Plots ---
if all_dfs:
    st.header("Visualizations")
    # Metric selection (main metrics, rest can be added)
    plot_metrics = ["accuracy", "f1", "aucroc", "training_time"]
    # Use available metrics in the current data
    all_metrics = [col for df in all_dfs.values() for col in df.columns if df[col].dtype in ['float64', 'int64']]
    metrics_avail = [m for m in plot_metrics if m in all_metrics]
    if not metrics_avail:
        metrics_avail = all_metrics  # fallback: show all

    metric = st.selectbox("Select metric to plot", metrics_avail, index=0)

    for label, df in all_dfs.items():
        st.markdown(f"#### {label}")
        plot_metric_by_model(df, metric, title=f"{metric} by Model")

    if len(all_dfs) > 1:
        st.subheader("Comparison Across Selected Results")
        plot_grouped_bar_by_model(all_dfs, metric)
