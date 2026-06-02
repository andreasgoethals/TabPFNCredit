"""Plot functions for the Experiment notebooks (extracted from notebooks/).

History
-------
Each ``notebooks/Experiment*.ipynb`` used to carry near-duplicate
``create_performance_heatmap`` / ``plot_method_ranking`` /
``plot_per_dataset_bars`` functions inline (~700 LOC repeated across
6 notebooks). They are now consolidated here, with the notebooks reduced
to ``df = load_summary(...)``-then-``plot_xxx(df, ...)`` calls.

All functions return ``(Figure, list[Path])`` -- the figure for inline
display and the list of saved file paths (PDF + PNG by default). Saving
defaults to ``figures/<task>/<plot_name>.pdf`` so the figure tree mirrors
the result tree.

Public surface
--------------
* :func:`performance_heatmap` -- methods x datasets heatmap of one metric.
* :func:`method_ranking_bars` -- mean-rank bar chart, one bar per method.
* :func:`per_dataset_bars` -- grouped bars: dataset on x, methods stacked.
* :func:`learning_curve` -- Experiment2 metric-vs-rows curve.
* :func:`imbalance_curve` -- Experiment3 metric-vs-minority-rate curve.
* :func:`load_summary` -- read the per-fold / per-method CSVs produced by
  :func:`src.utils.summarize_results_polars.summarize_to_csv`.

All functions are dependency-light (numpy / pandas / seaborn / matplotlib)
so notebooks remain runnable offline once the CSV summaries are saved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg", force=False)  # keep GUI backends usable locally
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Style
# ---------------------------------------------------------------------------

DEFAULT_RC = {
    "figure.figsize": (16, 9),
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def apply_style(rc: Optional[dict] = None, sns_style: str = "whitegrid") -> None:
    """Apply project-default rcParams + seaborn style. Idempotent."""
    plt.rcParams.update(DEFAULT_RC)
    if rc:
        plt.rcParams.update(rc)
    sns.set_style(sns_style)


# ---------------------------------------------------------------------------
#  Summary loaders
# ---------------------------------------------------------------------------

def load_summary(summary_dir: Path, *, task: str, aggregated: bool = True) -> pd.DataFrame:
    """Load the polars-written summary CSV produced by ``summarize_to_csv``.

    ``task`` is "pd" or "lgd"; ``aggregated`` picks per-method aggregates
    over per-fold rows.
    """
    summary_dir = Path(summary_dir)
    suffix = "per_method.csv" if aggregated else "per_fold.csv"
    # Try the canonical pattern first; fall back to legacy filenames.
    for pattern in (
        f"*_per_method.csv" if aggregated else "*_per_fold.csv",
        f"summary_{task}_aggregated.csv" if aggregated else f"summary_{task}_raw.csv",
    ):
        for path in summary_dir.glob(pattern):
            df = pd.read_csv(path)
            if "task" not in df.columns or (df["task"] == task).any():
                return df[df.get("task", task) == task] if "task" in df.columns else df
    raise FileNotFoundError(f"No {task} {suffix} found under {summary_dir}")


# ---------------------------------------------------------------------------
#  Heatmap (methods x datasets) of a single metric
# ---------------------------------------------------------------------------

def performance_heatmap(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    cmap: str = "RdYlGn",
    figsize: Tuple[int, int] = (24, 8),
    out_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, List[Path]]:
    """Datasets x methods heatmap of ``metric`` (mean across folds).

    The ``df`` must expose either ``{metric}_mean`` (polars summary) or
    ``metric.{metric}_mean`` (legacy summary). For R2 (regression) the
    colormap is diverging and centred on zero.
    """
    mean_col = _resolve_mean_column(df, metric)
    pivot = df.pivot(index="dataset", columns="method", values=mean_col).sort_index()
    # Sort methods by descending mean across datasets
    method_order = pivot.mean(axis=0).sort_values(ascending=False).index
    pivot = pivot[method_order]

    fig, ax = plt.subplots(figsize=figsize)
    is_diverging = metric.upper() in {"R2"}
    if is_diverging:
        abs_max = float(np.nanmax(np.abs(pivot.values)))
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap=cmap, center=0,
            vmin=-abs_max, vmax=abs_max,
            cbar_kws={"label": metric}, linewidths=0.5, ax=ax,
        )
    else:
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap=cmap,
            vmin=float(np.nanmin(pivot.values)), vmax=float(np.nanmax(pivot.values)),
            cbar_kws={"label": metric}, linewidths=0.5, ax=ax,
        )
    ax.set_title(f"{task_name} performance: {metric} (datasets x methods)",
                 fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Method", fontsize=12, fontweight="bold")
    ax.set_ylabel("Dataset", fontsize=12, fontweight="bold")
    plt.tight_layout()
    paths = _save(fig, out_dir, f"{task_name.lower()}_heatmap_{metric.lower()}")
    return fig, paths


# ---------------------------------------------------------------------------
#  Method ranking bar chart
# ---------------------------------------------------------------------------

def method_ranking_bars(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    higher_is_better: bool = True,
    figsize: Tuple[int, int] = (14, 8),
    out_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, List[Path]]:
    """Bar chart: each method's mean rank across datasets for ``metric``."""
    mean_col = _resolve_mean_column(df, metric)
    pivot = df.pivot(index="dataset", columns="method", values=mean_col)
    rank = pivot.rank(axis=1, ascending=not higher_is_better)
    method_rank = rank.mean(axis=0).sort_values(ascending=higher_is_better)

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("viridis", n_colors=len(method_rank))
    ax.barh(method_rank.index, method_rank.values, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel(f"Mean rank ({metric}; lower is better)", fontsize=12, fontweight="bold")
    ax.set_title(f"{task_name} method ranking by {metric}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    paths = _save(fig, out_dir, f"{task_name.lower()}_ranking_{metric.lower()}")
    return fig, paths


# ---------------------------------------------------------------------------
#  Per-dataset bars
# ---------------------------------------------------------------------------

def per_dataset_bars(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    figsize: Tuple[int, int] = (24, 10),
    methods: Optional[Sequence[str]] = None,
    out_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, List[Path]]:
    """Grouped bar chart with one bar per (dataset, method)."""
    mean_col = _resolve_mean_column(df, metric)
    sub = df[["dataset", "method", mean_col]].copy()
    if methods:
        sub = sub[sub["method"].isin(methods)]
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=sub, x="dataset", y=mean_col, hue="method", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel(metric, fontsize=12, fontweight="bold")
    ax.set_title(f"{task_name} per-dataset {metric}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.0), fontsize=9)
    plt.tight_layout()
    paths = _save(fig, out_dir, f"{task_name.lower()}_per_dataset_{metric.lower()}")
    return fig, paths


# ---------------------------------------------------------------------------
#  Experiment 2 (learning curve) + Experiment 3 (imbalance curve)
# ---------------------------------------------------------------------------

def _curve(
    df: pd.DataFrame,
    *,
    x_col: str,
    metric: str,
    task_name: str,
    title: str,
    xlabel: str,
    figsize: Tuple[int, int],
    out_dir: Optional[Path],
    plot_name: str,
) -> Tuple[plt.Figure, List[Path]]:
    mean_col = _resolve_mean_column(df, metric)
    pivot = df.pivot_table(index=x_col, columns="method", values=mean_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=figsize)
    for method in pivot.columns:
        ax.plot(pivot.index, pivot[method], marker="o", label=method, linewidth=1.6)
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(metric, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    paths = _save(fig, out_dir, plot_name)
    return fig, paths


def learning_curve(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    figsize: Tuple[int, int] = (14, 8),
    out_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, List[Path]]:
    """Experiment 2: ``metric`` vs training rows."""
    return _curve(
        df, x_col="row_limit", metric=metric, task_name=task_name,
        title=f"{task_name} learning curve: {metric}",
        xlabel="Training rows", figsize=figsize, out_dir=out_dir,
        plot_name=f"{task_name.lower()}_learning_curve_{metric.lower()}",
    )


def imbalance_curve(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    figsize: Tuple[int, int] = (14, 8),
    out_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, List[Path]]:
    """Experiment 3: ``metric`` vs minority-class proportion."""
    return _curve(
        df, x_col="minority_proportion", metric=metric, task_name=task_name,
        title=f"{task_name} imbalance robustness: {metric}",
        xlabel="Minority-class proportion", figsize=figsize, out_dir=out_dir,
        plot_name=f"{task_name.lower()}_imbalance_curve_{metric.lower()}",
    )


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _resolve_mean_column(df: pd.DataFrame, metric: str) -> str:
    for cand in (f"{metric}_mean", f"metric.{metric}_mean", metric):
        if cand in df.columns:
            return cand
    raise KeyError(
        f"Could not find a mean column for {metric!r} in df with columns "
        f"{list(df.columns)[:20]}"
    )


def _save(fig: plt.Figure, out_dir: Optional[Path], stem: str) -> List[Path]:
    """Persist ``fig`` to ``{out_dir}/{stem}.{pdf,png}`` and return both paths."""
    if out_dir is None:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        paths.append(path)
    return paths


__all__ = [
    "apply_style",
    "load_summary",
    "performance_heatmap",
    "method_ranking_bars",
    "per_dataset_bars",
    "learning_curve",
    "imbalance_curve",
]
