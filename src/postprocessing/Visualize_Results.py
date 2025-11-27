# scripts/Visualize_Results.py
"""
Visualize Results: Generate publication-quality figures from experiment results.

This script reads summary CSV files and raw pickle results to create various
visualizations comparing model performance across datasets.

Output files are saved to: results/<experiment_name>/images/

Visualizations include:
- Bar plots (mean performance by method)
- Boxplots (performance distribution across folds/datasets)
- Heatmaps (method × metric)
- Radar/Spider plots (multi-metric comparison)
- Ranking plots (relative performance)
- Critical difference diagrams
- Training time comparisons

Usage:
    python scripts/Visualize_Results.py
    python scripts/Visualize_Results.py --experiment experiment1
    python scripts/Visualize_Results.py --experiment experiment2
"""

import sys
import argparse
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CONFIGURATION
# =============================================================================

# Color scheme - standard colors
DEFAULT_COLOR = "#2980b9"   # Blue
COLORS_PALETTE = sns.color_palette("husl", 20)

# Figure settings
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_method_color(method: str) -> str:
    """Get color for a method."""
    return DEFAULT_COLOR


def get_method_colors(methods: List[str]) -> List[str]:
    """Get colors for a list of methods."""
    return [DEFAULT_COLOR for m in methods]


def rotate_xticks(ax, deg: int = 30):
    """Rotate x-axis tick labels."""
    for label in ax.get_xticklabels():
        label.set_rotation(deg)
        label.set_horizontalalignment('right')


def save_figure(fig, images_dir: Path, filename: str, formats: List[str] = ['pdf']):
    """Save figure in PDF format."""
    for fmt in formats:
        filepath = images_dir / f"{filename}.{fmt}"
        fig.savefig(filepath, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_summary_data(experiment_dir: Path, task: str, hpo_mode: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load per-dataset and overall summary CSVs.
    
    Returns:
        Tuple of (per_dataset_df, overall_df)
    """
    summary_dir = experiment_dir / "summary"
    
    per_dataset_file = summary_dir / f"summary_{task}_{hpo_mode.lower()}_per_dataset.csv"
    overall_file = summary_dir / f"summary_{task}_{hpo_mode.lower()}_overall.csv"
    
    per_dataset_df = pd.DataFrame()
    overall_df = pd.DataFrame()
    
    if per_dataset_file.exists():
        per_dataset_df = pd.read_csv(per_dataset_file)
    
    if overall_file.exists():
        overall_df = pd.read_csv(overall_file)
    
    return per_dataset_df, overall_df


def load_raw_results(experiment_dir: Path, task: str) -> Dict[str, Any]:
    """Load all raw pickle results for a task."""
    task_dir = experiment_dir / task.lower()
    results = {}
    
    if not task_dir.exists():
        return results
    
    for pkl_file in task_dir.glob("*.pkl"):
        dataset_name = pkl_file.stem
        try:
            with open(pkl_file, 'rb') as f:
                results[dataset_name] = pickle.load(f)
        except Exception as e:
            print(f"  Warning: Could not load {dataset_name}: {e}")
    
    return results


def extract_fold_data(raw_results: Dict[str, Any], hpo_mode: str) -> pd.DataFrame:
    """
    Extract per-fold data from raw results into a DataFrame.
    
    Returns DataFrame with columns: Dataset, Method, Fold, and all metrics
    """
    rows = []
    
    for dataset_name, dataset_results in raw_results.items():
        if hpo_mode not in dataset_results:
            continue
        
        hpo_results = dataset_results[hpo_mode]
        
        for method_name, method_results in hpo_results.items():
            for fold_id, fold_data in method_results.items():
                row = {
                    'Dataset': dataset_name,
                    'Method': method_name,
                    'Fold': fold_id,
                }
                
                # Extract metrics
                if 'metrics' in fold_data and 'metric_names' in fold_data:
                    metrics = fold_data['metrics']
                    metric_names = fold_data['metric_names']
                    if isinstance(metrics, (tuple, list)):
                        for i, name in enumerate(metric_names):
                            if i < len(metrics):
                                row[name] = metrics[i]
                
                # Extract training time
                if 'train_time' in fold_data:
                    row['TrainingTime'] = fold_data['train_time']
                
                rows.append(row)
    
    return pd.DataFrame(rows)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_metric_bar(
    df: pd.DataFrame,
    metric: str,
    title: str,
    images_dir: Path,
    filename: str,
    ascending: bool = False
):
    """
    Bar plot showing mean metric by method.
    """
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_std'
    
    if mean_col not in df.columns:
        print(f"    Skipping {filename}: {mean_col} not found")
        return
    
    # Sort by metric
    df_sorted = df.sort_values(mean_col, ascending=ascending).copy()
    
    methods = df_sorted['Method'].tolist()
    means = df_sorted[mean_col].tolist()
    stds = df_sorted[std_col].tolist() if std_col in df_sorted.columns else [0] * len(means)
    colors = get_method_colors(methods)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(methods, means, yerr=stds, color=colors, capsize=3, alpha=0.85)
    
    # Adjust y-axis to show differences better
    if means:
        valid_means = [m for m in means if not np.isnan(m)]
        if valid_means:
            min_val = min(valid_means)
            max_val = max(valid_means)
            padding = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
            ax.set_ylim(max(0, min_val - padding), max_val + padding)
    
    ax.set_ylabel(metric)
    ax.set_title(title)
    
    rotate_xticks(ax, deg=35)
    
    plt.tight_layout()
    save_figure(fig, images_dir, filename)
    print(f"    Saved: {filename}")


def plot_boxplot(
    fold_df: pd.DataFrame,
    metric: str,
    title: str,
    images_dir: Path,
    filename: str
):
    """
    Boxplot showing metric distribution across folds/datasets.
    """
    if metric not in fold_df.columns:
        print(f"    Skipping {filename}: {metric} not found")
        return
    
    # Filter out zero/nan values
    plot_df = fold_df[fold_df[metric].notna() & (fold_df[metric] != 0)].copy()
    
    if plot_df.empty:
        print(f"    Skipping {filename}: No valid data")
        return
    
    # Order methods by median performance
    method_order = plot_df.groupby('Method')[metric].median().sort_values(ascending=False).index.tolist()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create palette
    palette = {m: get_method_color(m) for m in method_order}
    
    sns.boxplot(
        data=plot_df,
        x='Method',
        y=metric,
        order=method_order,
        palette=palette,
        ax=ax
    )
    
    ax.set_title(title)
    rotate_xticks(ax, deg=35)
    
    plt.tight_layout()
    save_figure(fig, images_dir, filename)
    print(f"    Saved: {filename}")


def plot_heatmap(
    df: pd.DataFrame,
    metrics: List[str],
    title: str,
    images_dir: Path,
    filename: str
):
    """
    Heatmap showing method × metric performance.
    """
    # Get mean columns for specified metrics
    mean_cols = [f'{m}_mean' for m in metrics if f'{m}_mean' in df.columns]
    
    if not mean_cols:
        print(f"    Skipping {filename}: No metric columns found")
        return
    
    # Create pivot table
    pivot_data = df.set_index('Method')[mean_cols].copy()
    pivot_data.columns = [c.replace('_mean', '') for c in pivot_data.columns]
    
    # Sort methods by first metric
    first_metric = pivot_data.columns[0]
    pivot_data = pivot_data.sort_values(first_metric, ascending=False)
    
    fig, ax = plt.subplots(figsize=(max(10, len(mean_cols) * 1.5), max(8, len(pivot_data) * 0.4)))
    
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        ax=ax,
        cbar_kws={'label': 'Score'}
    )
    
    ax.set_title(title)
    
    # Highlight TabPFN rows
    
    plt.tight_layout()
    save_figure(fig, images_dir, filename)
    print(f"    Saved: {filename}")


def plot_dataset_comparison(
    per_dataset_df: pd.DataFrame,
    metric: str,
    title: str,
    images_dir: Path,
    filename: str
):
    """
    Heatmap showing method performance across datasets.
    """
    mean_col = f'{metric}_mean'
    
    if mean_col not in per_dataset_df.columns:
        print(f"    Skipping {filename}: {mean_col} not found")
        return
    
    # Pivot: methods as rows, datasets as columns
    pivot = per_dataset_df.pivot(index='Method', columns='Dataset', values=mean_col)
    
    if pivot.empty:
        print(f"    Skipping {filename}: No data to plot")
        return
    
    # Sort by mean across datasets
    pivot['_mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('_mean', ascending=False)
    pivot = pivot.drop('_mean', axis=1)
    
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.8), max(8, len(pivot) * 0.5)))
    
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        ax=ax,
        cbar_kws={'label': metric}
    )
    
    ax.set_title(title)
    rotate_xticks(ax, deg=45)
    
    plt.tight_layout()
    save_figure(fig, images_dir, filename)
    print(f"    Saved: {filename}")


def plot_ranking(
    per_dataset_df: pd.DataFrame,
    metric: str,
    title: str,
    images_dir: Path,
    filename: str,
    ascending: bool = False
):
    """
    Bar plot showing average rank across datasets.
    """
    mean_col = f'{metric}_mean'
    
    if mean_col not in per_dataset_df.columns:
        print(f"    Skipping {filename}: {mean_col} not found")
        return
    
    # Calculate rank for each dataset
    def rank_within_dataset(group):
        group = group.copy()
        group['Rank'] = group[mean_col].rank(ascending=ascending, method='min')
        return group
    
    ranked_df = per_dataset_df.groupby('Dataset', group_keys=False).apply(rank_within_dataset)
    
    # Average rank per method
    avg_rank = ranked_df.groupby('Method')['Rank'].mean().sort_values()
    
    methods = avg_rank.index.tolist()
    ranks = avg_rank.values.tolist()
    colors = get_method_colors(methods)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(methods, ranks, color=colors, alpha=0.85)
    
    ax.set_ylabel('Average Rank (lower is better)')
    ax.set_title(title)
    ax.axhline(y=np.mean(ranks), color='gray', linestyle='--', alpha=0.5, label='Mean Rank')
    
    rotate_xticks(ax, deg=35)
    
    plt.tight_layout()
    save_figure(fig, images_dir, filename)
    print(f"    Saved: {filename}")


def plot_pairwise_wins(
    fold_df: pd.DataFrame,
    metric: str,
    title: str,
    images_dir: Path,
    filename: str
):
    """
    Heatmap showing pairwise win counts between methods.
    """
    if metric not in fold_df.columns:
        print(f"    Skipping {filename}: {metric} not found")
        return
    
    methods = fold_df['Method'].unique().tolist()
    
    # Create pairwise comparison matrix
    wins = pd.DataFrame(index=methods, columns=methods, data=0.0)
    
    # Group by dataset and fold to compare methods
    for (dataset, fold), group in fold_df.groupby(['Dataset', 'Fold']):
        for m1 in methods:
            for m2 in methods:
                if m1 == m2:
                    continue
                
                score1 = group[group['Method'] == m1][metric].values
                score2 = group[group['Method'] == m2][metric].values
                
                if len(score1) > 0 and len(score2) > 0:
                    if score1[0] > score2[0]:
                        wins.loc[m1, m2] += 1
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        wins.astype(float),
        annot=True,
        fmt='.0f',
        cmap='Blues',
        ax=ax,
        cbar_kws={'label': 'Win Count'}
    )
    
    ax.set_title(title)
    ax.set_xlabel('Opponent')
    ax.set_ylabel('Method')
    
    
    plt.tight_layout()
    save_figure(fig, images_dir, filename)
    print(f"    Saved: {filename}")


def plot_method_comparison_summary(
    overall_df: pd.DataFrame,
    primary_metric: str,
    title: str,
    images_dir: Path,
    filename: str
):
    """
    Summary comparison plot with error bars and significance indicators.
    """
    mean_col = f'{primary_metric}_mean'
    std_col = f'{primary_metric}_std'
    
    if mean_col not in overall_df.columns:
        print(f"    Skipping {filename}: {mean_col} not found")
        return
    
    df_sorted = overall_df.sort_values(mean_col, ascending=False).copy()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(df_sorted))
    methods = df_sorted['Method'].tolist()
    means = df_sorted[mean_col].values
    stds = df_sorted[std_col].values if std_col in df_sorted.columns else np.zeros_like(means)
    colors = get_method_colors(methods)
    
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.annotate(
            f'{mean:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.005),
            ha='center',
            va='bottom',
            fontsize=8
        )
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel(primary_metric)
    ax.set_title(title)
    
    # Adjust y limits
    max_val = max(means + stds)
    min_val = min(means - stds)
    padding = (max_val - min_val) * 0.15
    ax.set_ylim(max(0, min_val - padding), max_val + padding)
    
    rotate_xticks(ax, deg=35)
    
    plt.tight_layout()
    save_figure(fig, images_dir, filename)
    print(f"    Saved: {filename}")


# =============================================================================
# MAIN VISUALIZATION PIPELINE
# =============================================================================

def generate_visualizations(
    experiment_name: str = "experiment1",
    results_base_dir: str = "results"
) -> None:
    """
    Main function to generate all visualizations.
    """
    
    # Determine experiment directory
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == 'scripts':
        project_root = script_dir.parent
    else:
        project_root = script_dir
    
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
        print(f"ERROR: Experiment directory not found")
        sys.exit(1)
    
    # Create images directory
    images_dir = experiment_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f" GENERATING VISUALIZATIONS: {experiment_name}")
    print("=" * 80)
    print(f"\nExperiment directory: {experiment_dir}")
    print(f"Images output directory: {images_dir}")
    
    # Process each task
    for task in ['pd', 'lgd']:
        task_label = "PD (Classification)" if task == 'pd' else "LGD (Regression)"
        primary_metric = 'AUC' if task == 'pd' else 'R2'
        secondary_metrics = ['Accuracy', 'F1', 'LogLoss'] if task == 'pd' else ['RMSE', 'MAE']
        
        print(f"\n{'=' * 60}")
        print(f" {task_label}")
        print(f"{'=' * 60}")
        
        # Load raw results for fold-level data
        raw_results = load_raw_results(experiment_dir, task)
        
        if not raw_results:
            print(f"  No raw results found for {task.upper()}")
            continue
        
        # Process both HPO modes
        for hpo_mode in ['NO_HPO', 'HPO']:
            print(f"\n  --- {hpo_mode} ---")
            
            # Load summary data
            per_dataset_df, overall_df = load_summary_data(experiment_dir, task, hpo_mode)
            
            if overall_df.empty:
                print(f"    No summary data found")
                continue
            
            # Extract fold-level data
            fold_df = extract_fold_data(raw_results, hpo_mode)
            
            prefix = f"{task}_{hpo_mode.lower()}"
            
            # 1. Main performance bar chart
            plot_method_comparison_summary(
                overall_df,
                primary_metric,
                f"{task_label} - {hpo_mode.replace('_', ' ')} - {primary_metric} by Method",
                images_dir,
                f"{prefix}_bar_{primary_metric.lower()}"
            )
            
            # 2. Boxplot of primary metric
            if not fold_df.empty:
                plot_boxplot(
                    fold_df,
                    primary_metric,
                    f"{task_label} - {hpo_mode.replace('_', ' ')} - {primary_metric} Distribution",
                    images_dir,
                    f"{prefix}_boxplot_{primary_metric.lower()}"
                )
            
            # 3. Heatmap of all metrics
            all_metrics = [primary_metric] + secondary_metrics
            plot_heatmap(
                overall_df,
                all_metrics,
                f"{task_label} - {hpo_mode.replace('_', ' ')} - Method × Metric",
                images_dir,
                f"{prefix}_heatmap_metrics"
            )
            
            # 4. Dataset comparison heatmap
            if not per_dataset_df.empty:
                plot_dataset_comparison(
                    per_dataset_df,
                    primary_metric,
                    f"{task_label} - {hpo_mode.replace('_', ' ')} - {primary_metric} by Dataset",
                    images_dir,
                    f"{prefix}_heatmap_datasets"
                )
            
            # 5. Ranking plot
            if not per_dataset_df.empty:
                plot_ranking(
                    per_dataset_df,
                    primary_metric,
                    f"{task_label} - {hpo_mode.replace('_', ' ')} - Average Rank",
                    images_dir,
                    f"{prefix}_ranking"
                )
            
            # 6. Pairwise wins heatmap
            if not fold_df.empty and primary_metric in fold_df.columns:
                plot_pairwise_wins(
                    fold_df,
                    primary_metric,
                    f"{task_label} - {hpo_mode.replace('_', ' ')} - Pairwise Wins ({primary_metric})",
                    images_dir,
                    f"{prefix}_pairwise_wins"
                )
        
        # Compare HPO vs NO_HPO
        print(f"\n  --- HPO Comparison ---")
        create_hpo_comparison_plots(experiment_dir, task, images_dir, primary_metric, task_label)
    
    print(f"\n{'=' * 80}")
    print(" VISUALIZATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nAll images saved to: {images_dir}")


def create_hpo_comparison_plots(
    experiment_dir: Path,
    task: str,
    images_dir: Path,
    primary_metric: str,
    task_label: str
):
    """
    Create plots comparing HPO vs NO_HPO performance.
    """
    _, no_hpo_df = load_summary_data(experiment_dir, task, 'NO_HPO')
    _, hpo_df = load_summary_data(experiment_dir, task, 'HPO')
    
    if no_hpo_df.empty or hpo_df.empty:
        print("    Skipping HPO comparison: Missing data")
        return
    
    mean_col = f'{primary_metric}_mean'
    
    if mean_col not in no_hpo_df.columns or mean_col not in hpo_df.columns:
        print(f"    Skipping HPO comparison: {mean_col} not found")
        return
    
    # Merge dataframes
    merged = no_hpo_df[['Method', mean_col]].merge(
        hpo_df[['Method', mean_col]],
        on='Method',
        suffixes=('_NO_HPO', '_HPO')
    )
    
    if merged.empty:
        print("    Skipping HPO comparison: No matching methods")
        return
    
    # Sort by HPO performance (best first)
    merged = merged.sort_values(f'{mean_col}_HPO', ascending=False)
    
    # ==========================================================================
    # MAIN COMPARISON CHART (like the screenshot - overlapping bars style)
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(14, 7))
    
    methods = merged['Method'].tolist()
    no_hpo_vals = merged[f'{mean_col}_NO_HPO'].values
    hpo_vals = merged[f'{mean_col}_HPO'].values
    
    x = np.arange(len(methods))
    width = 0.7
    
    # Plot "No Tuning" bars first (with red edge, slightly transparent fill)
    bars_no_hpo = ax.bar(
        x, no_hpo_vals, width,
        label='No Tuning',
        color='#AEC7E8',  # Light blue fill
        edgecolor='#E74C3C',  # Red edge
        linewidth=2,
        alpha=0.6
    )
    
    # Plot "Optuna" bars on top (solid blue, slightly narrower to show overlap effect)
    bars_hpo = ax.bar(
        x, hpo_vals, width * 0.85,
        label='Optuna',
        color='#1F77B4',  # Solid blue
        alpha=0.85
    )
    
    ax.set_ylabel(f'Average {primary_metric.lower()}', fontsize=12)
    ax.set_xlabel('Learning Algorithms', fontsize=12)
    ax.set_title(f'{task_label} - Average {primary_metric} Before and After Optuna HPO', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    
    # Y-axis from 0
    ax.set_ylim(0, max(max(no_hpo_vals), max(hpo_vals)) * 1.05)
    
    # Legend in upper right
    ax.legend(loc='upper right', fontsize=10)
    
    rotate_xticks(ax, deg=45)
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    save_figure(fig, images_dir, f"{task}_tuning_comparison")
    print(f"    Saved: {task}_tuning_comparison")
    
    # ==========================================================================
    # GROUPED BAR CHART (side by side version)
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(merged))
    width = 0.35
    
    bars1 = ax.bar(
        x - width/2, no_hpo_vals, width,
        label='No Tuning',
        color='#E74C3C',  # Red
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    bars2 = ax.bar(
        x + width/2, hpo_vals, width,
        label='Optuna',
        color='#3498DB',  # Blue
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_ylabel(f'Average {primary_metric}', fontsize=12)
    ax.set_xlabel('Learning Algorithms', fontsize=12)
    ax.set_title(f'{task_label} - {primary_metric} Comparison: No Tuning vs Optuna HPO', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='upper right', fontsize=10)
    
    rotate_xticks(ax, deg=45)
    
    # Adjust y limits - start from reasonable minimum to show differences
    all_vals = np.concatenate([no_hpo_vals, hpo_vals])
    valid_vals = all_vals[~np.isnan(all_vals)]
    if len(valid_vals) > 0:
        min_val = valid_vals.min()
        max_val = valid_vals.max()
        range_val = max_val - min_val
        # Start from 0 or slightly below minimum
        y_min = 0 if min_val > 0.5 * max_val else max(0, min_val - range_val * 0.1)
        ax.set_ylim(y_min, max_val * 1.05)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    save_figure(fig, images_dir, f"{task}_hpo_comparison")
    print(f"    Saved: {task}_hpo_comparison")
    
    # ==========================================================================
    # IMPROVEMENT PLOT
    # ==========================================================================
    merged['Improvement'] = merged[f'{mean_col}_HPO'] - merged[f'{mean_col}_NO_HPO']
    merged_sorted = merged.sort_values('Improvement', ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    methods_sorted = merged_sorted['Method'].tolist()
    improvements = merged_sorted['Improvement'].values
    colors = ['#27AE60' if imp >= 0 else '#C0392B' for imp in improvements]
    
    bars = ax.bar(methods_sorted, improvements, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel(f'{primary_metric} Improvement (Optuna - No Tuning)', fontsize=12)
    ax.set_xlabel('Learning Algorithms', fontsize=12)
    ax.set_title(f'{task_label} - {primary_metric} Improvement from Optuna HPO', fontsize=14)
    
    rotate_xticks(ax, deg=45)
    
    # Add value labels on bars
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 0.005 if height >= 0 else -0.005
        ax.annotate(
            f'{imp:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height + offset),
            ha='center', va=va,
            fontsize=8,
            fontweight='bold'
        )
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    save_figure(fig, images_dir, f"{task}_hpo_improvement")
    print(f"    Saved: {task}_hpo_improvement")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations from experiment results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python Visualize_Results.py
    python Visualize_Results.py --experiment experiment1
    python Visualize_Results.py --experiment experiment2
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
    
    args = parser.parse_args()
    
    generate_visualizations(
        experiment_name=args.experiment,
        results_base_dir=args.results_dir
    )


if __name__ == "__main__":
    main()