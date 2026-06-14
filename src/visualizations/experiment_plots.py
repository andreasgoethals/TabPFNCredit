"""Shared plotting helpers for the Experiment notebooks.

Each ``notebooks/Experiment*.ipynb`` reduces to ``df = load_summary(...)``
followed by ``plot_xxx(df, ...)`` calls -- all the figure logic lives here.
:func:`load_summary` reads the per-fold / per-method CSVs produced by
:func:`src.utils.result_summary.summarize_to_csv`; the plot helpers cover
metric heatmaps, rankings, box plots, rank matrices, learning / imbalance
curves, the HPO-effect bars and the cost/quality scatter (see ``__all__``).

A plot helper saves a PDF only when the caller passes ``out_dir`` (the
notebooks pass ``figures/<experiment>/...``), returning the saved path or
``None``. Any extension is normalised to ``.pdf``. The figure is also
rendered inline -- a PNG pushed through ``IPython.display.Image`` via
``data_exploration._display_inline`` (no-op outside Jupyter) -- then closed
to keep memory in check across long notebook loops. Dependencies are kept
light (numpy / pandas / seaborn / matplotlib) so the notebooks run offline
once the CSV summaries exist.
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
    higher_is_better: bool = True,
    cmap: Optional[str] = None,
    fmt: str = ".3f",
    figsize: Tuple[int, int] = (24, 8),
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Datasets x methods heatmap of ``metric`` (mean across folds).

    Columns are ALWAYS sorted best -> worst from left to right, where "best"
    follows ``higher_is_better`` (e.g. highest AUC left; lowest Brier left).
    The colormap likewise maps green to good: pass ``higher_is_better=False``
    for lower-is-better metrics and the default cmap flips to ``RdYlGn_r``.
    """
    mean_col = _resolve_mean_column(df, metric)
    pivot = df.pivot_table(index="dataset", columns="method",
                           values=mean_col, aggfunc="mean").sort_index()
    # Best performance on the LEFT, worst on the RIGHT.
    method_order = pivot.mean(axis=0).sort_values(ascending=not higher_is_better).index
    pivot = pivot[method_order]
    if cmap is None:
        cmap = "RdYlGn" if higher_is_better else "RdYlGn_r"

    fig, ax = plt.subplots(figsize=figsize)
    is_diverging = metric.upper() in {"R2"}
    if is_diverging:
        abs_max = float(np.nanmax(np.abs(pivot.values)))
        sns.heatmap(
            pivot, annot=True, fmt=fmt, cmap=cmap, center=0,
            vmin=-abs_max, vmax=abs_max,
            cbar_kws={"label": metric}, linewidths=0.5, ax=ax,
        )
    else:
        sns.heatmap(
            pivot, annot=True, fmt=fmt, cmap=cmap,
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
    relative: bool = False,
) -> Optional[Path]:
    """ONE line per METHOD: the metric averaged over all datasets at each
    sweep value. With ``relative=True`` each method's curve is divided by its
    OWN maximum, showing performance relative to that method's top (1.0 =
    the method's best point). Linear x axis with dense ticks; small markers."""
    from matplotlib.ticker import MaxNLocator

    mean_col = _resolve_mean_column(df, metric)
    sub = df[df["sweep_axis"] == sweep_axis].dropna(subset=["sweep_value"])
    if sub.empty:
        logger.warning("No rows with sweep_axis=%s -- is this the right experiment?",
                       sweep_axis)
        return None
    grp = sub.groupby(["method", "sweep_value"])[mean_col].mean().reset_index()

    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("tab10", n_colors=grp["method"].nunique())
    for color, (method, g) in zip(palette, grp.groupby("method")):
        g = g.sort_values("sweep_value")
        y = g[mean_col]
        if relative:
            top = y.max()
            y = y / top if top else y
        ax.plot(g["sweep_value"], y, marker="o", ms=2.5, lw=1.7,
                label=method, color=color)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=14))
    if relative:
        ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{metric} / own max" if relative else metric,
                  fontsize=12, fontweight="bold")
    ax.set_title(f"{title} (mean over datasets)", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    suffix = "_relative" if relative else ""
    return _save(fig, out_dir, plot_name + suffix)


def learning_curve(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    figsize: Tuple[int, int] = (14, 8),
    relative: bool = False,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Experiment 2: ``metric`` vs training rows -- one line per method,
    averaged over every included dataset. ``relative=True`` divides each
    method's curve by its own best value."""
    return _sweep_curve(
        df, sweep_axis="row_limit", metric=metric,
        title=f"{task_name} learning curve: {metric}",
        xlabel="Training rows", figsize=figsize, out_dir=out_dir,
        relative=relative,
        plot_name=f"{task_name.lower()}_learning_curve_{metric.lower()}",
    )


def imbalance_curve(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    figsize: Tuple[int, int] = (14, 8),
    relative: bool = False,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Experiment 3: ``metric`` vs minority proportion -- one line per method,
    averaged over every included dataset. ``relative=True`` divides each
    method's curve by its own best value."""
    return _sweep_curve(
        df, sweep_axis="minority_proportion", metric=metric,
        title=f"{task_name} imbalance robustness: {metric}",
        xlabel="Minority-class proportion", figsize=figsize, out_dir=out_dir,
        relative=relative,
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


def metric_bars(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    higher_is_better: bool = True,
    agg: str = "mean",
    figsize: Tuple[int, int] = (16, 6),
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Bar chart of the ``agg`` (mean/median) of ``metric`` per method across
    datasets, sorted best -> worst from left to right."""
    mean_col = _resolve_mean_column(df, metric)
    vals = getattr(df.groupby("method")[mean_col], agg)()
    vals = vals.sort_values(ascending=not higher_is_better)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(vals.index, vals.values, color="#4878CF")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    ax.set_ylabel(f"{agg} {metric}", fontsize=12, fontweight="bold")
    ax.set_title(f"{task_name}: {agg} {metric} per method (best left)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, out_dir, f"{task_name.lower()}_bar_{agg}_{metric.lower()}")


def median_time_bars(
    df: pd.DataFrame,
    *,
    task_name: str = "PD",
    figsize: Tuple[int, int] = (16, 6),
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Median training time per fold per method (log y; fastest left)."""
    if "train_time_mean" not in df.columns:
        logger.warning("median_time_bars: no train_time_mean column")
        return None
    vals = df.groupby("method")["train_time_mean"].median().sort_values()
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(vals.index, vals.values, color="#E1812C")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    ax.set_ylabel("median train time per fold (s, log)", fontsize=12, fontweight="bold")
    ax.set_title(f"{task_name}: median training time per method (fastest left)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, out_dir, f"{task_name.lower()}_bar_median_time")


def _rank_pivot(df: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    mean_col = _resolve_mean_column(df, metric)
    pivot = df.pivot_table(index="dataset", columns="method",
                           values=mean_col, aggfunc="mean")
    ranks = pivot.rank(axis=1, ascending=not higher_is_better)
    return ranks[ranks.mean(axis=0).sort_values().index]  # best (lowest rank) left


def rank_heatmap(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    higher_is_better: bool = True,
    figsize: Tuple[int, int] = (24, 8),
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Datasets x methods matrix of the method's RANK on that dataset by
    ``metric`` (1 = best; ties share average ranks). Best mean rank left."""
    ranks = _rank_pivot(df, metric, higher_is_better)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(ranks, annot=True, fmt=".0f", cmap="RdYlGn_r",
                vmin=1, vmax=ranks.shape[1],
                cbar_kws={"label": f"rank by {metric} (1 = best)"},
                linewidths=0.5, ax=ax)
    ax.set_title(f"{task_name} rank matrix by {metric} (1 = best; best mean rank left)",
                 fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Method", fontsize=12, fontweight="bold")
    ax.set_ylabel("Dataset", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return _save(fig, out_dir, f"{task_name.lower()}_rank_matrix_{metric.lower()}")


def rank_boxplots(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    higher_is_better: bool = True,
    figsize: Tuple[int, int] = (16, 7),
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Box + strip of each method's per-dataset ranks by ``metric``."""
    ranks = _rank_pivot(df, metric, higher_is_better)
    long = ranks.melt(var_name="method", value_name="rank")
    order = list(ranks.columns)
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=long, x="method", y="rank", order=order, color="#ffe2b8", ax=ax)
    sns.stripplot(data=long, x="method", y="rank", order=order,
                  color="#a05a00", size=4, alpha=0.6, ax=ax)
    ax.invert_yaxis()  # rank 1 (best) on top
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    ax.set_ylabel(f"rank by {metric} (1 = best)", fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_title(f"{task_name} rank distribution across datasets ({metric})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, out_dir, f"{task_name.lower()}_rank_box_{metric.lower()}")


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
    agg = (df.groupby("method")
           .agg(perf=(mean_col, "mean"), time=("train_time_mean", "mean"))
           .reset_index().sort_values("time"))
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(agg["time"], agg["perf"], s=55, color="#1f77b4",
               edgecolor="white", linewidth=0.6, zorder=3)
    # Collision-aware labels: alternate above/below along the x-order, and
    # push successive near-identical x positions further out.
    last_x = None
    bump = 0
    for i, r in enumerate(agg.itertuples()):
        if last_x is not None and r.time > 0 and last_x > 0 \
                and abs(np.log10(r.time) - np.log10(last_x)) < 0.06:
            bump += 1
        else:
            bump = 0
        last_x = r.time
        above = (i % 2 == 0)
        dy = (8 + 9 * bump) * (1 if above else -1)
        ax.annotate(r.method, (r.time, r.perf), xytext=(0, dy),
                    textcoords="offset points", fontsize=8.5,
                    ha="center", va="bottom" if above else "top",
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6)
                    if abs(dy) > 10 else None)
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
    "metric_bars",
    "median_time_bars",
    "rank_heatmap",
    "rank_boxplots",
    "hpo_improvement_bars",
    "runtime_performance_scatter",
]
