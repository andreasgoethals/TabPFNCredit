import pandas as pd
from pathlib import Path
import os
# Define paths and metric preferences
os.chdir("/home/linux_vmedina/projects/CreditScoring")
output_dir = Path("outputs")
tasks = {
    "pd": {
        "prefix": "pd_",
        "metrics": ["accuracy", "brier", "f1", "precision", "recall", "h_measure", "aucroc", "aucpr"],
        "maximize": {"accuracy", "f1", "precision", "recall", "h_measure", "aucroc", "aucpr"},
    },
    "lgd": {
        "prefix": "lgd_",
        "metrics": ["mse", "mae", "r2", "rmse"],
        "maximize": {"r2"},
    },
}

ranking_outputs = {}

for task, config in tasks.items():
    prefix = config["prefix"] # pd or lgd
    metrics = config["metrics"] 
    maximize = config["maximize"]

    # Find task-specific folders
    task_folders = [f for f in output_dir.iterdir() if f.is_dir() and f.name.startswith(prefix)]
    all_rankings = {}

    for folder in task_folders:
        dataset = folder.name[len(prefix):]

        # Find latest result file in the folder
        csv_files = list(folder.glob(f"{prefix}{dataset}_*.csv"))
        if not csv_files:
            continue
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        
        # Load and reduce results
        df = pd.read_csv(latest_file)
        df_mean = df.groupby("model")[metrics].mean()
        
        # Compute ranking per metric
        ranks = {}
        for metric in metrics:
            ascending = metric not in maximize
            ranks[metric] = df_mean[metric].rank(ascending=ascending, method="min")
        
        df_ranks = pd.DataFrame(ranks)
        df_ranks.columns = [f"{metric}_{dataset}" for metric in df_ranks.columns]
        all_rankings[dataset] = df_ranks

    # Merge all rankings by model
    if all_rankings:
        merged = pd.concat(all_rankings.values(), axis=1)
        merged.index.name = "model"
        ranking_outputs[task] = merged.reset_index()

# Save rankings to CSV in output/analysis directory
analysis_dir = output_dir / "analysis"
analysis_dir.mkdir(parents=True, exist_ok=True)
for task, df in ranking_outputs.items():
    output_file = analysis_dir / f"{task}_ranking.csv"
    df.to_csv(output_file, index=False)
    print(f"Ranking saved to {output_file}")