import os
import pandas as pd

def generate_meta_results_for_task(task_dir):
    """
    For each dataset in task_dir, generates a meta_results.csv (all runs for that dataset).
    Also generates a total_combined_meta_results.csv across all datasets.
    Returns dict: {dataset_name: [csv_files]}, and path to combined meta file.
    """
    all_dataset_folders = [
        f for f in os.listdir(task_dir)
        if os.path.isdir(os.path.join(task_dir, f)) and not f.startswith("config")
    ]
    results_files_by_dataset = {}
    all_meta_dfs = []
    for dataset in all_dataset_folders:
        dataset_path = os.path.join(task_dir, dataset)
        meta_name = f"{dataset}_meta_results.csv"
        meta_path = os.path.join(dataset_path, meta_name)
        # Only use real result files, exclude meta results when building meta
        csv_files = [
            f for f in os.listdir(dataset_path)
            if (
                f.endswith(".csv")
                and os.path.isfile(os.path.join(dataset_path, f))
                and not f.endswith("_meta_results.csv")
            )
        ]
        # Always re-create meta (robust for new runs)
        if csv_files:
            dfs = [pd.read_csv(os.path.join(dataset_path, f)) for f in csv_files]
            meta_df = pd.concat(dfs, ignore_index=True)
            meta_df["dataset"] = dataset  # ensure column exists
            meta_df.to_csv(meta_path, index=False)
            csv_files.append(meta_name)
            all_meta_dfs.append(meta_df)
        # Always include meta if it exists
        elif os.path.exists(meta_path):
            csv_files.append(meta_name)
        # Add meta to selection if not already
        if os.path.exists(meta_path) and meta_name not in csv_files:
            csv_files.append(meta_name)
        results_files_by_dataset[dataset] = sorted(csv_files, reverse=True)
    # ---- TOTAL COMBINED ----
    combined_meta_path = None
    if all_meta_dfs:
        combined = pd.concat(all_meta_dfs, ignore_index=True)
        combined_dir = os.path.join(task_dir, "total_combined")
        os.makedirs(combined_dir, exist_ok=True)
        combined_meta_path = os.path.join(combined_dir, "total_combined_meta_results.csv")
        combined.to_csv(combined_meta_path, index=False)
    return results_files_by_dataset, combined_meta_path
