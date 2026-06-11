"""Plot functions for the Experiment notebooks (extracted from notebooks/).

History
-------
Each ``notebooks/Experiment*.ipynb`` used to carry near-duplicate
``create_performance_heatmap`` / ``plot_method_ranking`` /
``plot_per_dataset_bars`` functions inline (~700 LOC repeated across
6 notebooks). They are now consolidated here, with the notebooks reduced
to ``df = load_summary(...)``-then-``plot_xxx(df, ...)`` calls.

All functions return the saved PDF path (or ``None`` when no output
directory is supplied). The figure is also rendered inline via
``IPython.display.display(fig)`` so notebook cells show it right at
the call site, and then closed to keep memory in check across loops.
Saving defaults to ``figures/<task>/<plot_name>.pdf`` so the figure
tree mirrors the result tree.

Public surface
--------------
* :func:`performance_heatmap` -- methods x datasets heatmap of one metric.
* :func:`method_ranking_bars` -- mean-rank bar chart, one bar per method.
* :func:`per_dataset_bars` -- grouped bars: dataset on x, methods stacked.
* :func:`learning_curve` -- Experiment2 metric-vs-rows curve.
* :func:`imbalance_curve` -- Experiment3 metric-vs-minority-rate curve.
* :func:`load_summary` -- read the per-fold / per-method CSVs produced by
  :func:`src.utils.result_summary.summarize_to_csv`.

All functions are dependency-light (numpy / pandas / seaborn / matplotlib)
so notebooks remain runnable offline once the CSV summaries are saved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib
# Pin the non-interactive Agg backend so the module imports cleanly
# under SLURM (no ``$DISPLAY``), CI (no Tk), and Jupyter alike. Figures
# are rendered inline by encoding a PNG into memory and pushing it
# through ``IPython.display.Image`` -- see ``_save`` below.
matplotlib.use("Agg", force=False)
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


def reset_figure_dir(figures_dir: Path) -> Path:
    """Delete all PNG / PDF / SVG files under ``figures_dir`` and recreate it.

    Used at the top of every Experiment notebook so re-running the
    notebook gives a clean output set rather than mixing old + new
    figures. Subdirectories are recursively cleared too.
    """
    import shutil
    figures_dir = Path(figures_dir)
    if figures_dir.exists():
        for path in figures_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".pdf", ".svg", ".jpg", ".jpeg"}:
                path.unlink()
        # Remove now-empty subdirectories
        for sub in sorted(figures_dir.rglob("*"), reverse=True):
            if sub.is_dir() and not any(sub.iterdir()):
                sub.rmdir()
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


# ---------------------------------------------------------------------------
#  Summary loaders
# ---------------------------------------------------------------------------

def load_summary(
    summary_dir: Path,
    *,
    experiment: str,
    task: str,
    aggregated: bool = True,
    hpo_mode: Optional[str] = "NO_HPO",
) -> pd.DataFrame:
    """Load ONE experiment's summary CSV produced by ``summarize_to_csv``.

    ``experiment`` is e.g. ``"experiment1"`` -- the CSVs are named
    ``<experiment>_per_method.csv`` / ``<experiment>_per_fold.csv``, and
    selecting by experiment here is what keeps Experiment 2's notebook from
    silently plotting Experiment 0's numbers. ``task`` is "pd" or "lgd".
    ``hpo_mode`` filters to "NO_HPO" (default) or "HPO"; pass ``None`` to
    keep both (e.g. for HPO-vs-NO_HPO comparisons).
    """
    summary_dir = Path(summary_dir)
    suffix = "per_method.csv" if aggregated else "per_fold.csv"
    path = summary_dir / f"{experiment.lower()}_{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `tabpfncredit summarize` first")
    df = pd.read_csv(path)
    if "task" in df.columns:
        df = df[df["task"] == task]
    if hpo_mode is not None and "hpo_mode" in df.columns:
        df = df[df["hpo_mode"] == hpo_mode]
    if df.empty:
        raise FileNotFoundError(f"No {task}/{hpo_mode} rows in {path}")
    return df.reset_index(drop=True)


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
) -> Optional[Path]:
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
    return _save(fig, out_dir, f"{task_name.lower()}_heatmap_{metric.lower()}")


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
) -> Optional[Path]:
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
    return _save(fig, out_dir, f"{task_name.lower()}_ranking_{metric.lower()}")


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
) -> Optional[Path]:
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
    return _save(fig, out_dir, f"{task_name.lower()}_per_dataset_{metric.lower()}")


# ---------------------------------------------------------------------------
#  Experiment 2 (learning curve) + Experiment 3 (imbalance curve)
# ---------------------------------------------------------------------------

def _sweep_curve(
    df: pd.DataFrame,
    *,
    sweep_axis: str,
    metric: str,
    title: str,
    xlabel: str,
    figsize: Tuple[int, int],
    out_dir: Optional[Path],
    plot_name: str,
    logx: bool = False,
) -> Optional[Path]:
    """ONE line per METHOD: the metric averaged over all datasets at each
    sweep value, with a +-1 std band across datasets. The summary CSV encodes
    the sweep in the ``sweep_axis`` / ``sweep_value`` columns."""
    mean_col = _resolve_mean_column(df, metric)
    sub = df[df["sweep_axis"] == sweep_axis].dropna(subset=["sweep_value"])
    if sub.empty:
        logger.warning("No rows with sweep_axis=%s -- is this the right experiment?",
                       sweep_axis)
        return None
    grp = sub.groupby(["method", "sweep_value"])[mean_col].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("tab10", n_colors=grp["method"].nunique())
    for color, (method, g) in zip(palette, grp.groupby("method")):
        g = g.sort_values("sweep_value")
        ax.plot(g["sweep_value"], g["mean"], marker="o", ms=4, lw=2,
                label=method, color=color)
        ax.fill_between(g["sweep_value"], g["mean"] - g["std"], g["mean"] + g["std"],
                        alpha=0.12, color=color)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(metric, fontsize=12, fontweight="bold")
    ax.set_title(f"{title}\n(mean over datasets; band = +-1 std across datasets)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    return _save(fig, out_dir, plot_name)


def learning_curve(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    figsize: Tuple[int, int] = (14, 8),
    logx: bool = True,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Experiment 2: ``metric`` vs training rows -- one line per method,
    averaged over every included dataset."""
    return _sweep_curve(
        df, sweep_axis="row_limit", metric=metric,
        title=f"{task_name} learning curve: {metric}",
        xlabel="Training rows", figsize=figsize, out_dir=out_dir, logx=logx,
        plot_name=f"{task_name.lower()}_learning_curve_{metric.lower()}",
    )


def imbalance_curve(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    figsize: Tuple[int, int] = (14, 8),
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Experiment 3: ``metric`` vs minority proportion -- one line per method,
    averaged over every included dataset."""
    return _sweep_curve(
        df, sweep_axis="minority_proportion", metric=metric,
        title=f"{task_name} imbalance robustness: {metric}",
        xlabel="Minority-class proportion", figsize=figsize, out_dir=out_dir,
        plot_name=f"{task_name.lower()}_imbalance_curve_{metric.lower()}",
    )


# ---------------------------------------------------------------------------
#  Distribution, HPO-effect and cost plots
# ---------------------------------------------------------------------------

def metric_boxplots(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    higher_is_better: bool = True,
    figsize: Tuple[int, int] = (16, 7),
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Box + strip plot of the metric's distribution across datasets per method."""
    mean_col = _resolve_mean_column(df, metric)
    order = (df.groupby("method")[mean_col].median()
             .sort_values(ascending=not higher_is_better).index)
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x="method", y=mean_col, order=order, color="#cfe8ff", ax=ax)
    sns.stripplot(data=df, x="method", y=mean_col, order=order,
                  color="#1f4e79", size=4, alpha=0.6, ax=ax)
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    ax.set_ylabel(metric, fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_title(f"{task_name} {metric} distribution across datasets",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, out_dir, f"{task_name.lower()}_box_{metric.lower()}")


def hpo_improvement_bars(
    df_both: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    higher_is_better: bool = True,
    figsize: Tuple[int, int] = (14, 7),
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Mean HPO-minus-NO_HPO improvement per method (Experiment 1).

    ``df_both`` must be loaded with ``hpo_mode=None``. Methods whose HPO run
    is a copy of NO_HPO (foundation models) show a 0 bar by construction.
    """
    mean_col = _resolve_mean_column(df_both, metric)
    piv = df_both.pivot_table(index=["dataset", "method"], columns="hpo_mode",
                              values=mean_col, aggfunc="mean").dropna()
    if not {"HPO", "NO_HPO"} <= set(piv.columns):
        logger.warning("hpo_improvement_bars: need both HPO and NO_HPO rows")
        return None
    delta = (piv["HPO"] - piv["NO_HPO"]) * (1 if higher_is_better else -1)
    per_method = delta.groupby("method").agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in per_method["mean"]]
    ax.barh(per_method.index, per_method["mean"], xerr=per_method["std"].fillna(0),
            color=colors, error_kw={"alpha": 0.4})
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel(f"HPO improvement in {metric} (positive = tuning helps)",
                  fontsize=12, fontweight="bold")
    ax.set_title(f"{task_name}: effect of hyper-parameter tuning on {metric}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, out_dir, f"{task_name.lower()}_hpo_effect_{metric.lower()}")


def runtime_performance_scatter(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    figsize: Tuple[int, int] = (12, 8),
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Mean train time (log x) vs mean metric -- the cost/quality frontier."""
    mean_col = _resolve_mean_column(df, metric)
    if "train_time_mean" not in df.columns:
        logger.warning("runtime_performance_scatter: no train_time_mean column")
        return None
    agg = df.groupby("method").agg(
        perf=(mean_col, "mean"), time=("train_time_mean", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(agg["time"], agg["perf"], s=70, c=sns.color_palette("tab10", len(agg)))
    for _i, r in agg.iterrows():
        ax.annotate(r["method"], (r["time"], r["perf"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Mean train time per fold (s, log scale)", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Mean {metric}", fontsize=12, fontweight="bold")
    ax.set_title(f"{task_name}: performance vs training cost",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, out_dir, f"{task_name.lower()}_cost_quality_{metric.lower()}")


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


def _save(fig: plt.Figure, out_dir: Optional[Path], stem: str) -> Optional[Path]:
    """Save ``fig`` to ``{out_dir}/{stem}.pdf`` AND display inline in Jupyter.

    Behaviour
    ---------
    * Saves a single PDF (no PNG).
    * Encodes a PNG into memory and ``IPython.display.Image``-s it so
      the figure renders inline at the call site inside a notebook. The
      Agg backend is sufficient for this (no GUI required), which keeps
      the helper safe to call from SLURM jobs and CI runs too.
    * Closes the figure to free memory (important when notebooks loop
      over many datasets calling these helpers in sequence).

    Returns the saved path, or ``None`` if ``out_dir`` is ``None``.
    """
    saved: Optional[Path] = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = out_dir / f"{stem}.pdf"
        fig.savefig(saved, bbox_inches="tight")
    # Inline display inside Jupyter; no-op elsewhere.
    from src.visualizations.data_exploration import _display_inline
    _display_inline(fig)
    plt.close(fig)
    return saved


__all__ = [
    "apply_style",
    "reset_figure_dir",
    "load_summary",
    "performance_heatmap",
    "method_ranking_bars",
    "per_dataset_bars",
    "learning_curve",
    "imbalance_curve",
    "metric_boxplots",
    "hpo_improvement_bars",
    "runtime_performance_scatter",
]
