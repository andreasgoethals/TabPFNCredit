import os
import streamlit as st
import pandas as pd

OUTPUTS_DIR = "outputs"

st.sidebar.title("Filters")

# 1. Multi-select Datasets (skip 'analysis')
all_dataset_folders = [
    f for f in os.listdir(OUTPUTS_DIR)
    if os.path.isdir(os.path.join(OUTPUTS_DIR, f)) and not f.startswith("analysis")
]

selected_datasets = st.sidebar.multiselect(
    "Select Dataset(s)", sorted(all_dataset_folders), default=sorted(all_dataset_folders)[:1]
)

results_files_by_dataset = {}
for dataset in selected_datasets:
    dataset_path = os.path.join(OUTPUTS_DIR, dataset)
    csv_files = [
        f for f in os.listdir(dataset_path)
        if f.endswith(".csv") and f.startswith(dataset) and os.path.isfile(os.path.join(dataset_path, f))
    ]
    results_files_by_dataset[dataset] = csv_files

# 2. Multi-select result files for each selected dataset
selected_files = {}
for dataset, files in results_files_by_dataset.items():
    if files:
        chosen_files = st.sidebar.multiselect(
            f"Select results file(s) for {dataset}",
            sorted(files, reverse=True),
            default=sorted(files, reverse=True)[:1],
            key=f"{dataset}_files"
        )
        selected_files[dataset] = chosen_files

# --- Main Area ---
st.title("TABPFN Credit Benchmark Results")

for dataset, files in selected_files.items():
    for result_file in files:
        st.subheader(f"{dataset}: {result_file}")
        csv_path = os.path.join(OUTPUTS_DIR, dataset, result_file)
        df = pd.read_csv(csv_path)
        st.dataframe(df)
