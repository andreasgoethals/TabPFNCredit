# scripts/Summarize_Results.py
"""
Summarize Results: Aggregate all experiment results into summary files.

This script reads all pickle result files from an experiment folder and creates
summary CSV files for PD (classification) and LGD (regression) tasks.

Output files are saved to: results/<experiment_name>/summary/

For each method, it calculates:
- Average performance (mean ± std) over all folds for each dataset
- Average performance (mean ± std) over all datasets

Usage:
    python scripts/Summarize_Results.py
    python scripts/Summarize_Results.py --experiment experiment1
    python scripts/Summarize_Results.py --experiment experiment2 --format xlsx
"""

import sys
import argparse
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_all_results(experiment_dir: Path, task: str) -> Dict[str, Any]:
    """
    Load all pickle result files for a given task.
    
    Args:
        experiment_dir: Path to experiment directory (e.g., results/experiment1)
        task: 'pd' or 'lgd'
        
    Returns:
        Dictionary mapping dataset names to their results
    """
    task_dir = experiment_dir / task.lower()
    
    if not task_dir.exists():
        print(f"WARNING: Task directory not found: {task_dir}")
        return {}
    
    results = {}
    pkl_files = list(task_dir.glob("*.pkl"))
    
    if not pkl_files:
        print(f"WARNING: No pickle files found in {task_dir}")
        return {}
    
    for pkl_file in pkl_files:
        dataset_name = pkl_file.stem
        try:
            with open(pkl_file, 'rb') as f:
                results[dataset_name] = pickle.load(f)
            print(f"  Loaded: {dataset_name}")
        except Exception as e:
            print(f"  ERROR loading {dataset_name}: {e}")
    
    return results


def extract_fold_metrics(
    fold_results: Dict[int, Dict[str, Any]],
    task: str
) -> Tuple[Dict[str, List[float]], List[str]]:
    """
    Extract metrics from all folds for a single method.
    
    Args:
        fold_results: Dictionary mapping fold_id to fold results
        task: 'pd' or 'lgd'
        
    Returns:
        Tuple of (metrics_dict, metric_names)
        - metrics_dict: {metric_name: [values per fold]}
        - metric_names: list of metric names
    """
    metrics_dict = {}
    metric_names = None
    
    for fold_id, fold_data in fold_results.items():
        # Get metric names from first fold
        if metric_names is None:
            if 'metric_names' in fold_data:
                metric_names = fold_data['metric_names']
            else:
                # Fallback metric names
                if task == 'pd':
                    metric_names = ['Accuracy', 'Avg_Recall', 'Avg_Precision', 'F1', 'LogLoss', 'AUC']
                else:
                    metric_names = ['RMSE', 'R2', 'MAE']
        
        # Extract metrics
        if 'metrics' in fold_data:
            metrics = fold_data['metrics']
            if isinstance(metrics, (tuple, list)):
                for i, metric_name in enumerate(metric_names):
                    if i < len(metrics):
                        if metric_name not in metrics_dict:
                            metrics_dict[metric_name] = []
                        metrics_dict[metric_name].append(metrics[i])
            elif isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if metric_name not in metrics_dict:
                        metrics_dict[metric_name] = []
                    metrics_dict[metric_name].append(value)
    
    return metrics_dict, metric_names or []


def compute_summary_stats(values: List[float]) -> Tuple[float, float]:
    """
    Compute mean and std for a list of values.
    
    Args:
        values: List of metric values
        
    Returns:
        Tuple of (mean, std)
    """
    if not values:
        return np.nan, np.nan
    
    values = [v for v in values if v is not None and not np.isnan(v)]
    if not values:
        return np.nan, np.nan
    
    return np.mean(values), np.std(values)


def summarize_task_results(
    all_results: Dict[str, Any],
    task: str,
    hpo_mode: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create summary DataFrames for a specific task and HPO mode.
    
    Args:
        all_results: Dictionary mapping dataset names to results
        task: 'pd' or 'lgd'
        hpo_mode: 'NO_HPO' or 'HPO'
        
    Returns:
        Tuple of (per_dataset_df, overall_df)
        - per_dataset_df: Detailed results per dataset
        - overall_df: Aggregated results across all datasets
    """
    
    # Collect all metrics across all datasets and methods
    # Structure: {method: {dataset: {metric: [fold_values]}}}
    method_dataset_metrics = {}
    
    # Discover all metric names
    all_metric_names = set()
    
    for dataset_name, dataset_results in all_results.items():
        if hpo_mode not in dataset_results:
            continue
        
        hpo_results = dataset_results[hpo_mode]
        
        for method_name, method_results in hpo_results.items():
            if method_name not in method_dataset_metrics:
                method_dataset_metrics[method_name] = {}
            
            # Extract metrics for this method on this dataset
            metrics_dict, metric_names = extract_fold_metrics(method_results, task)
            all_metric_names.update(metric_names)
            
            method_dataset_metrics[method_name][dataset_name] = metrics_dict
    
    # Sort metric names for consistent ordering
    # Prioritize important metrics first
    if task == 'pd':
        priority_metrics = ['AUC', 'Accuracy', 'F1', 'Avg_Recall', 'Avg_Precision', 'LogLoss']
    else:
        priority_metrics = ['R2', 'RMSE', 'MAE', 'MSE']
    
    sorted_metric_names = []
    for m in priority_metrics:
        if m in all_metric_names:
            sorted_metric_names.append(m)
            all_metric_names.discard(m)
    sorted_metric_names.extend(sorted(all_metric_names))
    
    # Build per-dataset summary
    per_dataset_rows = []
    
    for method_name in sorted(method_dataset_metrics.keys()):
        for dataset_name in sorted(method_dataset_metrics[method_name].keys()):
            row = {
                'Method': method_name,
                'Dataset': dataset_name,
            }
            
            metrics_dict = method_dataset_metrics[method_name][dataset_name]
            
            for metric_name in sorted_metric_names:
                if metric_name in metrics_dict:
                    values = metrics_dict[metric_name]
                    mean, std = compute_summary_stats(values)
                    row[f'{metric_name}_mean'] = mean
                    row[f'{metric_name}_std'] = std
                else:
                    row[f'{metric_name}_mean'] = np.nan
                    row[f'{metric_name}_std'] = np.nan
            
            per_dataset_rows.append(row)
    
    per_dataset_df = pd.DataFrame(per_dataset_rows)
    
    # Build overall summary (average across all datasets for each method)
    overall_rows = []
    
    for method_name in sorted(method_dataset_metrics.keys()):
        row = {
            'Method': method_name,
            'N_Datasets': len(method_dataset_metrics[method_name]),
        }
        
        # Collect all fold values across all datasets for each metric
        all_method_metrics = {}
        
        for dataset_name, metrics_dict in method_dataset_metrics[method_name].items():
            for metric_name, values in metrics_dict.items():
                if metric_name not in all_method_metrics:
                    all_method_metrics[metric_name] = []
                all_method_metrics[metric_name].extend(values)
        
        for metric_name in sorted_metric_names:
            if metric_name in all_method_metrics:
                values = all_method_metrics[metric_name]
                mean, std = compute_summary_stats(values)
                row[f'{metric_name}_mean'] = mean
                row[f'{metric_name}_std'] = std
            else:
                row[f'{metric_name}_mean'] = np.nan
                row[f'{metric_name}_std'] = np.nan
        
        overall_rows.append(row)
    
    overall_df = pd.DataFrame(overall_rows)
    
    return per_dataset_df, overall_df


def create_pivot_summary(per_dataset_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Create a pivot table showing method performance across datasets.
    
    Args:
        per_dataset_df: Per-dataset summary DataFrame
        metric: Metric name to pivot (e.g., 'AUC')
        
    Returns:
        Pivot DataFrame with methods as rows and datasets as columns
    """
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_std'
    
    if mean_col not in per_dataset_df.columns:
        return pd.DataFrame()
    
    # Create combined mean±std string
    per_dataset_df = per_dataset_df.copy()
    per_dataset_df['value'] = per_dataset_df.apply(
        lambda row: f"{row[mean_col]:.4f}±{row[std_col]:.4f}" 
        if pd.notna(row[mean_col]) and pd.notna(row[std_col])
        else "",
        axis=1
    )
    
    pivot = per_dataset_df.pivot(
        index='Method',
        columns='Dataset',
        values='value'
    )
    
    return pivot


def summarize_results(
    experiment_name: str = "experiment1",
    results_base_dir: str = "results",
    output_format: str = "csv"
) -> None:
    """
    Main function to summarize all experiment results.
    
    Args:
        experiment_name: Name of experiment folder
        results_base_dir: Base directory for results (default: "results")
        output_format: Output format ('csv' or 'xlsx')
    """
    
    # Determine paths
    # Try relative to script location first, then absolute
    script_dir = Path(__file__).resolve().parent
    
    # Check if we're in scripts/ directory
    if script_dir.name == 'scripts':
        project_root = script_dir.parent
    else:
        project_root = script_dir
    
    # Check multiple possible locations for results
    possible_paths = [
        project_root / results_base_dir / experiment_name,
        Path(results_base_dir) / experiment_name,
        Path.cwd() / results_base_dir / experiment_name,
    ]
    
    experiment_dir = None
    for path in possible_paths:
        if path.exists():
            experiment_dir = path
            break
    
    if experiment_dir is None:
        print(f"ERROR: Experiment directory not found. Tried:")
        for path in possible_paths:
            print(f"  - {path}")
        sys.exit(1)
    
    print("=" * 80)
    print(f" SUMMARIZING RESULTS: {experiment_name}")
    print("=" * 80)
    print(f"\nExperiment directory: {experiment_dir}")
    print(f"Output format: {output_format}")
    
    # Create summary output directory
    summary_dir = experiment_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    print(f"Summary output directory: {summary_dir}")
    
    # Process each task
    for task in ['pd', 'lgd']:
        print(f"\n{'-' * 40}")
        print(f" Processing {task.upper()} results...")
        print(f"{'-' * 40}")
        
        # Load all results for this task
        all_results = load_all_results(experiment_dir, task)
        
        if not all_results:
            print(f"  No results found for {task.upper()}")
            continue
        
        print(f"\n  Loaded {len(all_results)} datasets")
        
        # Process both HPO modes
        for hpo_mode in ['NO_HPO', 'HPO']:
            print(f"\n  Processing {hpo_mode}...")
            
            # Check if any results have this HPO mode
            has_mode = any(hpo_mode in results for results in all_results.values())
            if not has_mode:
                print(f"    No {hpo_mode} results found")
                continue
            
            # Generate summaries
            per_dataset_df, overall_df = summarize_task_results(all_results, task, hpo_mode)
            
            if per_dataset_df.empty:
                print(f"    No data to summarize for {hpo_mode}")
                continue
            
            # Save per-dataset summary
            per_dataset_file = summary_dir / f"summary_{task}_{hpo_mode.lower()}_per_dataset.{output_format}"
            if output_format == 'csv':
                per_dataset_df.to_csv(per_dataset_file, index=False, float_format='%.6f')
            else:
                per_dataset_df.to_excel(per_dataset_file, index=False)
            print(f"    Saved: {per_dataset_file.name}")
            
            # Save overall summary
            overall_file = summary_dir / f"summary_{task}_{hpo_mode.lower()}_overall.{output_format}"
            if output_format == 'csv':
                overall_df.to_csv(overall_file, index=False, float_format='%.6f')
            else:
                overall_df.to_excel(overall_file, index=False)
            print(f"    Saved: {overall_file.name}")
            
            # Create pivot tables for key metrics
            primary_metric = 'AUC' if task == 'pd' else 'R2'
            pivot_df = create_pivot_summary(per_dataset_df, primary_metric)
            
            if not pivot_df.empty:
                pivot_file = summary_dir / f"summary_{task}_{hpo_mode.lower()}_pivot_{primary_metric}.{output_format}"
                if output_format == 'csv':
                    pivot_df.to_csv(pivot_file, float_format='%.6f')
                else:
                    pivot_df.to_excel(pivot_file)
                print(f"    Saved: {pivot_file.name}")
    
    print(f"\n{'=' * 80}")
    print(" SUMMARY COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nAll summary files saved to: {summary_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize experiment results into CSV/Excel files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python Summarize_Results.py
    python Summarize_Results.py --experiment experiment1
    python Summarize_Results.py --experiment experiment2 --format xlsx
        """
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        default='experiment1',
        help='Name of experiment folder (default: experiment1)'
    )
    
    parser.add_argument(
        '--results_dir', '-r',
        type=str,
        default='results',
        help='Base directory for results (default: results)'
    )
    
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['csv', 'xlsx'],
        default='csv',
        help='Output format (default: csv)'
    )
    
    args = parser.parse_args()
    
    summarize_results(
        experiment_name=args.experiment,
        results_base_dir=args.results_dir,
        output_format=args.format
    )


if __name__ == "__main__":
    main()