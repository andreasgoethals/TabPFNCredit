# src/postprocessing/Summarize_Results.py
"""
Summarize Results: Aggregate all experiment results into summary CSV files.

This script reads all pickle result files from an experiment folder and creates
clean summary CSV files containing only the metrics for each fold.

Structure:
    Input:  results/{experiment}/pd/*.pkl and results/{experiment}/lgd/*.pkl
    Output: results/{experiment}/summary/

Output files:
    - summary_pd_raw.csv:      All PD metrics per method, dataset, fold, HPO mode
    - summary_lgd_raw.csv:     All LGD metrics per method, dataset, fold, HPO mode
    - summary_pd_aggregated.csv:   Mean ± std across folds for PD
    - summary_lgd_aggregated.csv:  Mean ± std across folds for LGD
    - pivot_pd_AUC.csv:        Pivot table of AUC scores (methods × datasets)
    - pivot_lgd_R2.csv:        Pivot table of R2 scores (methods × datasets)

Usage:
    python src/postprocessing/Summarize_Results.py
    python src/postprocessing/Summarize_Results.py --experiment experiment1
    python src/postprocessing/Summarize_Results.py --experiment experiment2
"""

import sys
import argparse
import pickle
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Setup paths (src/postprocessing/ -> src/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# CONFIGURATION
# =============================================================================

# Metrics to extract for PD (classification) tasks
PD_METRICS = [
    'AUC', 'Gini', 'KS', 'Brier', 'LogLoss',
    'Accuracy', 'Balanced_Accuracy', 'F1', 'Precision', 'Recall', 'MCC',
    'Avg_Precision', 'Avg_Recall'
]

# Metrics to extract for LGD (regression) tasks
LGD_METRICS = [
    'R2', 'RMSE', 'MAE', 'MSE', 'MAPE', 'Correlation', 'Spearman'
]

# Additional fields to extract
EXTRA_FIELDS = ['train_time', 'n_clipped_below', 'n_clipped_above']


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_pickle_file(pkl_path: Path) -> Dict[str, Any]:
    """Load a single pickle file."""
    try:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  ERROR loading {pkl_path.name}: {e}")
        return {}


def extract_fold_data(
    fold_results: Dict[str, Any],
    method: str,
    dataset: str,
    task: str,
    hpo_mode: str,
    fold_id: int
) -> Dict[str, Any]:
    """
    Extract relevant data from a single fold result.
    
    Returns a flat dictionary with all metrics and metadata.
    """
    row = {
        'method': method,
        'dataset': dataset,
        'task': task,
        'hpo_mode': hpo_mode,
        'fold_id': fold_id,
    }
    
    # Extract metrics
    metrics = fold_results.get('metrics', {})
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            row[key] = value
    
    # Extract extra fields
    for field in EXTRA_FIELDS:
        if field in fold_results:
            row[field] = fold_results[field]
    
    # Extract info fields
    info = fold_results.get('info', {})
    if isinstance(info, dict):
        row['n_num_features'] = info.get('n_num_features', np.nan)
        row['n_cat_features'] = info.get('n_cat_features', np.nan)
    
    return row


def process_dataset_results(
    results: Dict[str, Any],
    dataset: str,
    task: str
) -> List[Dict[str, Any]]:
    """
    Process results for a single dataset.
    
    Returns a list of dictionaries, one per (method, fold, hpo_mode) combination.
    """
    rows = []
    
    for hpo_mode in ['NO_HPO', 'HPO']:
        if hpo_mode not in results:
            continue
        
        hpo_results = results[hpo_mode]
        
        for method_name, method_results in hpo_results.items():
            if not isinstance(method_results, dict):
                continue
            
            for fold_id, fold_data in method_results.items():
                if not isinstance(fold_data, dict):
                    continue
                
                row = extract_fold_data(
                    fold_data, method_name, dataset, task, hpo_mode, fold_id
                )
                rows.append(row)
    
    return rows


def load_task_results(experiment_dir: Path, task: str) -> pd.DataFrame:
    """
    Load all results for a task (pd or lgd) into a DataFrame.
    
    Args:
        experiment_dir: Path to experiment directory
        task: 'pd' or 'lgd'
        
    Returns:
        DataFrame with all fold results
    """
    task_dir = experiment_dir / task.lower()
    
    if not task_dir.exists():
        print(f"  Directory not found: {task_dir}")
        return pd.DataFrame()
    
    pkl_files = list(task_dir.glob("*.pkl"))
    
    if not pkl_files:
        print(f"  No pickle files found in {task_dir}")
        return pd.DataFrame()
    
    print(f"  Found {len(pkl_files)} dataset files")
    
    all_rows = []
    
    for pkl_file in sorted(pkl_files):
        dataset_name = pkl_file.stem
        results = load_pickle_file(pkl_file)
        
        if not results:
            continue
        
        rows = process_dataset_results(results, dataset_name, task)
        all_rows.extend(rows)
        print(f"    {dataset_name}: {len(rows)} fold results")
    
    if not all_rows:
        return pd.DataFrame()
    
    return pd.DataFrame(all_rows)


def aggregate_results(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Aggregate fold results: compute mean ± std for each (method, dataset, hpo_mode).
    
    Args:
        df: Raw DataFrame with per-fold results
        task: 'pd' or 'lgd'
        
    Returns:
        Aggregated DataFrame
    """
    if df.empty:
        return pd.DataFrame()
    
    # Define which columns to aggregate
    metrics = PD_METRICS if task == 'pd' else LGD_METRICS
    numeric_cols = [c for c in df.columns if c in metrics or c in EXTRA_FIELDS]
    
    # Group by method, dataset, hpo_mode
    group_cols = ['method', 'dataset', 'hpo_mode']
    
    agg_rows = []
    
    for (method, dataset, hpo_mode), group in df.groupby(group_cols):
        row = {
            'method': method,
            'dataset': dataset,
            'hpo_mode': hpo_mode,
            'n_folds': len(group),
        }
        
        for col in numeric_cols:
            if col in group.columns:
                values = group[col].dropna()
                if len(values) > 0:
                    row[f'{col}_mean'] = values.mean()
                    row[f'{col}_std'] = values.std()
                else:
                    row[f'{col}_mean'] = np.nan
                    row[f'{col}_std'] = np.nan
        
        agg_rows.append(row)
    
    return pd.DataFrame(agg_rows)


def create_pivot_table(
    agg_df: pd.DataFrame,
    metric: str,
    hpo_mode: str = 'NO_HPO'
) -> pd.DataFrame:
    """
    Create a pivot table: methods (rows) × datasets (columns).
    
    Args:
        agg_df: Aggregated DataFrame
        metric: Metric to pivot (e.g., 'AUC', 'R2')
        hpo_mode: 'NO_HPO' or 'HPO'
        
    Returns:
        Pivot DataFrame
    """
    if agg_df.empty:
        return pd.DataFrame()
    
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_std'
    
    if mean_col not in agg_df.columns:
        return pd.DataFrame()
    
    # Filter by HPO mode
    df = agg_df[agg_df['hpo_mode'] == hpo_mode].copy()
    
    if df.empty:
        return pd.DataFrame()
    
    # Create formatted string: mean ± std
    df['value'] = df.apply(
        lambda row: f"{row[mean_col]:.4f} ± {row[std_col]:.4f}"
        if pd.notna(row[mean_col]) and pd.notna(row[std_col])
        else "",
        axis=1
    )
    
    # Pivot
    pivot = df.pivot(index='method', columns='dataset', values='value')
    
    # Add mean column (average across datasets)
    mean_values = df.groupby('method')[mean_col].mean()
    pivot['AVERAGE'] = pivot.index.map(lambda m: f"{mean_values.get(m, np.nan):.4f}")
    
    return pivot


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def summarize_results(experiment: str = "experiment1") -> None:
    """
    Main function to summarize all experiment results.
    
    Args:
        experiment: Name of experiment folder (e.g., 'experiment1')
    """
    
    # Determine paths
    results_dir = PROJECT_ROOT / "results"
    experiment_dir = results_dir / experiment
    summary_dir = experiment_dir / "summary"
    
    print("=" * 70)
    print(f" SUMMARIZING RESULTS: {experiment}")
    print("=" * 70)
    
    if not experiment_dir.exists():
        print(f"\nERROR: Experiment directory not found: {experiment_dir}")
        sys.exit(1)
    
    print(f"\nExperiment directory: {experiment_dir}")
    
    # Create summary directory
    summary_dir.mkdir(parents=True, exist_ok=True)
    print(f"Summary output: {summary_dir}")
    
    # Process each task
    for task in ['pd', 'lgd']:
        print(f"\n{'-' * 50}")
        print(f" {task.upper()} RESULTS")
        print(f"{'-' * 50}")
        
        # Load raw results
        raw_df = load_task_results(experiment_dir, task)
        
        if raw_df.empty:
            print(f"  No results found for {task.upper()}")
            continue
        
        # Save raw results (all folds)
        raw_file = summary_dir / f"summary_{task}_raw.csv"
        raw_df.to_csv(raw_file, index=False, float_format='%.6f')
        print(f"\n  Saved: {raw_file.name} ({len(raw_df)} rows)")
        
        # Aggregate results
        agg_df = aggregate_results(raw_df, task)
        
        if not agg_df.empty:
            agg_file = summary_dir / f"summary_{task}_aggregated.csv"
            agg_df.to_csv(agg_file, index=False, float_format='%.6f')
            print(f"  Saved: {agg_file.name} ({len(agg_df)} rows)")
        
        # Create pivot tables
        primary_metric = 'AUC' if task == 'pd' else 'R2'
        
        for hpo_mode in ['NO_HPO', 'HPO']:
            pivot_df = create_pivot_table(agg_df, primary_metric, hpo_mode)
            
            if not pivot_df.empty:
                pivot_file = summary_dir / f"pivot_{task}_{primary_metric}_{hpo_mode.lower()}.csv"
                pivot_df.to_csv(pivot_file)
                print(f"  Saved: {pivot_file.name}")
    
    print(f"\n{'=' * 70}")
    print(" DONE!")
    print(f"{'=' * 70}")
    print(f"\nSummary files saved to: {summary_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize experiment results into CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/postprocessing/Summarize_Results.py
    python src/postprocessing/Summarize_Results.py --experiment experiment1
    python src/postprocessing/Summarize_Results.py -e experiment2
        """
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        default='experiment1',
        help='Name of experiment folder (default: experiment1)'
    )
    
    args = parser.parse_args()
    summarize_results(experiment=args.experiment)


if __name__ == "__main__":
    main()
