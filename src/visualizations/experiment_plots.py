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
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

# Shared sizing so every figure in this module reads the same way.
TICK_FS = 12     # method / dataset tick labels -- must stay legible
LABEL_FS = 13    # axis labels
TITLE_FS = 15    # panel titles
ANNOT_FS = 8     # heatmap cell numbers -- small enough to fit a "0.xxx" cell


def apply_style(rc: Optional[dict] = None, sns_style: str = "whitegrid") -> None:
    """Apply project-default rcParams + seaborn style. Idempotent."""
    plt.rcParams.update(DEFAULT_RC)
    if rc:
        plt.rcParams.update(rc)
    sns.set_style(sns_style)


# ---------------------------------------------------------------------------
#  Shared styling helpers -- one colour code + one bar style everywhere
# ---------------------------------------------------------------------------

def _best_to_worst_colors(n: int):
    """Green (best) -> red (worst) gradient. The caller passes data ALREADY
    sorted best-first, so position 0 is greenest and the last bar is reddest."""
    return plt.get_cmap("RdYlGn")(np.linspace(0.92, 0.08, max(n, 1)))


def _heatmap_figsize(n_rows: int, n_cols: int) -> Tuple[float, float]:
    """Figure size that keeps each matrix cell large enough for a 0.xxx
    annotation and a legible tick label."""
    return (max(14.0, 0.62 * n_cols + 4), max(6.0, 0.55 * n_rows + 3))


def _style_method_axis(ax) -> None:
    """Uniform per-method x axis: 45-deg right-aligned labels at TICK_FS."""
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_horizontalalignment("right")


def _method_bar(
    series: pd.Series,
    *,
    title: str,
    ylabel: str,
    stem: str,
    out_dir: Optional[Path],
    errs: Optional[pd.Series] = None,
    logy: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
) -> Optional[Path]:
    """Consistent vertical per-method bar chart used by every bar plot here:
    a green(best)->red(worst) gradient, a black contour per bar, optional
    ``± std`` error bars, and the shared label styling. ``series`` must be
    sorted best-first so the gradient lines up with performance."""
    names = list(series.index)
    n = len(names)
    fig, ax = plt.subplots(figsize=figsize or (max(12.0, 0.5 * n + 4), 6))
    ax.bar(
        range(n), series.to_numpy(),
        color=_best_to_worst_colors(n), edgecolor="black", linewidth=0.8,
        yerr=(errs.reindex(series.index).to_numpy() if errs is not None else None),
        capsize=3 if errs is not None else 0,
        error_kw={"ecolor": "0.35", "lw": 1.1},
    )
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(range(n))
    ax.set_xticklabels(names)
    _style_method_axis(ax)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS, fontweight="bold")
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold")
    plt.tight_layout()
    return _save(fig, out_dir, stem)


def reset_figure_dir(figures_dir: Path) -> Path:
    """Delete all image files under ``figures_dir`` and (re)create it.

    Called at the top of every Experiment notebook so a rerun produces a
    clean figure set instead of mixing old and new outputs.

    Deletion is BEST-EFFORT: a file or directory that is momentarily locked
    (e.g. a PDF still open in a viewer, or a Windows handle not yet released
    after the files inside were removed) is skipped rather than raising. A
    stale figure left behind is harmless; a crashed setup cell is not.
    """
    figures_dir = Path(figures_dir)
    if figures_dir.exists():
        for path in figures_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".pdf", ".svg", ".jpg", ".jpeg"}:
                try:
                    path.unlink()
                except OSError as exc:
                    logger.warning("Could not delete stale figure %s (%s); leaving it.", path, exc)
        # Remove now-empty subdirectories (best-effort -- Windows may hold a
        # transient lock on a directory right after its files are deleted).
        for sub in sorted(figures_dir.rglob("*"), reverse=True):
            if sub.is_dir() and not any(sub.iterdir()):
                try:
                    sub.rmdir()
                except OSError:
                    pass
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
    auto_summarize: bool = True,
) -> pd.DataFrame:
    """Load ONE experiment's summary CSV produced by ``summarize_to_csv``.

    ``experiment`` is e.g. ``"experiment1"`` -- the CSVs are named
    ``<experiment>_per_method.csv`` / ``<experiment>_per_fold.csv``, and
    selecting by experiment here is what keeps Experiment 2's notebook from
    silently plotting Experiment 0's numbers. ``task`` is "pd" or "lgd".
    ``hpo_mode`` filters to "NO_HPO" (default) or "HPO"; pass ``None`` to
    keep both (e.g. for HPO-vs-NO_HPO comparisons).

    If the summary CSV is absent and ``auto_summarize`` is set (the default),
    the per-fold + per-method CSVs are built on the fly from the result files
    under ``summary_dir.parent`` (the results root) -- so a freshly downloaded
    results folder works in the notebooks with no separate
    ``tabpfncredit summarize`` step.
    """
    summary_dir = Path(summary_dir)
    suffix = "per_method.csv" if aggregated else "per_fold.csv"
    path = summary_dir / f"{experiment.lower()}_{suffix}"
    if not path.exists() and auto_summarize:
        from src.utils.result_summary import summarize_to_csv
        results_root = summary_dir.parent
        logger.info("Summary %s missing -- building it from %s ...", path.name, results_root)
        summarize_to_csv(base=results_root, experiment=experiment.lower(), out_dir=summary_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found and could not be built -- are there result files "
            f"under {summary_dir.parent / experiment.lower()}/ ?"
        )
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
    figsize: Optional[Tuple[float, float]] = None,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Datasets x methods heatmap of ``metric`` (mean across folds).

    Columns are ALWAYS sorted best -> worst from left to right, where "best"
    follows ``higher_is_better`` (e.g. highest AUC left; lowest Brier left).
    The colormap likewise maps green to good: pass ``higher_is_better=False``
    for lower-is-better metrics and the default cmap flips to ``RdYlGn_r``.
    The figure auto-sizes to the matrix shape so each ``0.xxx`` annotation
    fits its cell and the method / dataset labels stay legible.
    """
    mean_col = _resolve_mean_column(df, metric)
    pivot = df.pivot_table(index="dataset", columns="method",
                           values=mean_col, aggfunc="mean").sort_index()
    # Best performance on the LEFT, worst on the RIGHT.
    method_order = pivot.mean(axis=0).sort_values(ascending=not higher_is_better).index
    pivot = pivot[method_order]
    if cmap is None:
        cmap = "RdYlGn" if higher_is_better else "RdYlGn_r"

    n_rows, n_cols = pivot.shape
    fig, ax = plt.subplots(figsize=figsize or _heatmap_figsize(n_rows, n_cols))
    common = dict(annot=True, fmt=fmt, cmap=cmap, linewidths=0.5, ax=ax,
                  annot_kws={"size": ANNOT_FS}, cbar_kws={"label": metric})
    if metric.upper() in {"R2"}:  # diverging, centred on zero
        abs_max = float(np.nanmax(np.abs(pivot.values)))
        sns.heatmap(pivot, center=0, vmin=-abs_max, vmax=abs_max, **common)
    else:
        sns.heatmap(pivot, vmin=float(np.nanmin(pivot.values)),
                    vmax=float(np.nanmax(pivot.values)), **common)
    ax.set_title(f"{task_name} performance: {metric} (datasets x methods)",
                 fontsize=TITLE_FS + 1, fontweight="bold", pad=20)
    ax.set_xlabel("Method", fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel("Dataset", fontsize=LABEL_FS, fontweight="bold")
    _style_method_axis(ax)
    ax.tick_params(axis="y", rotation=0)
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
    """Bar chart of each method's mean rank across datasets for ``metric``
    (1 = best), with ± std error bars. Same vertical, gradient-coloured style
    as the other bar charts."""
    mean_col = _resolve_mean_column(df, metric)
    pivot = df.pivot(index="dataset", columns="method", values=mean_col)
    rank = pivot.rank(axis=1, ascending=not higher_is_better)   # 1 = best
    mean_rank = rank.mean(axis=0).sort_values()                 # lowest (best) first
    return _method_bar(
        mean_rank,
        errs=rank.std(axis=0),
        title=f"{task_name} method ranking by {metric} (mean rank ± std; lower = better)",
        ylabel=f"mean rank ({metric}; lower is better)",
        stem=f"{task_name.lower()}_ranking_{metric.lower()}",
        out_dir=out_dir, figsize=figsize,
    )


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
    _style_method_axis(ax)
    ax.set_ylabel(metric, fontsize=LABEL_FS, fontweight="bold")
    ax.set_xlabel("")
    ax.set_title(f"{task_name} {metric} distribution across datasets",
                 fontsize=TITLE_FS, fontweight="bold")
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
    datasets, sorted best -> worst left to right. The ``mean`` variant adds
    ± std error bars (std across datasets)."""
    mean_col = _resolve_mean_column(df, metric)
    grp = df.groupby("method")[mean_col]
    vals = grp.agg(agg).sort_values(ascending=not higher_is_better)
    errs = grp.std() if agg == "mean" else None
    suffix = " ± std across datasets" if errs is not None else ""
    return _method_bar(
        vals, errs=errs,
        title=f"{task_name}: {agg} {metric} per method (best left){suffix}",
        ylabel=f"{agg} {metric}",
        stem=f"{task_name.lower()}_bar_{agg}_{metric.lower()}",
        out_dir=out_dir, figsize=figsize,
    )


def median_time_bars(
    df: pd.DataFrame,
    *,
    task_name: str = "PD",
    figsize: Tuple[int, int] = (16, 6),
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Median training time per fold per method (log y; fastest=greenest, left)."""
    if "train_time_mean" not in df.columns:
        logger.warning("median_time_bars: no train_time_mean column")
        return None
    vals = df.groupby("method")["train_time_mean"].median().sort_values()  # fastest first
    return _method_bar(
        vals, logy=True,
        title=f"{task_name}: median training time per method (fastest left)",
        ylabel="median train time per fold (s, log)",
        stem=f"{task_name.lower()}_bar_median_time",
        out_dir=out_dir, figsize=figsize,
    )


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
    figsize: Optional[Tuple[float, float]] = None,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Datasets x methods matrix of the method's RANK on that dataset by
    ``metric`` (1 = best; ties share average ranks). Best mean rank left.
    Auto-sizes to the matrix shape for legible cells and labels."""
    ranks = _rank_pivot(df, metric, higher_is_better)
    n_rows, n_cols = ranks.shape
    fig, ax = plt.subplots(figsize=figsize or _heatmap_figsize(n_rows, n_cols))
    sns.heatmap(ranks, annot=True, fmt=".0f", cmap="RdYlGn_r",
                vmin=1, vmax=n_cols, annot_kws={"size": ANNOT_FS + 1},
                cbar_kws={"label": f"rank by {metric} (1 = best)"},
                linewidths=0.5, ax=ax)
    ax.set_title(f"{task_name} rank matrix by {metric} (1 = best; best mean rank left)",
                 fontsize=TITLE_FS + 1, fontweight="bold", pad=20)
    ax.set_xlabel("Method", fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel("Dataset", fontsize=LABEL_FS, fontweight="bold")
    _style_method_axis(ax)
    ax.tick_params(axis="y", rotation=0)
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
    _style_method_axis(ax)
    ax.set_ylabel(f"rank by {metric} (1 = best)", fontsize=LABEL_FS, fontweight="bold")
    ax.set_xlabel("")
    ax.set_title(f"{task_name} rank distribution across datasets ({metric})",
                 fontsize=TITLE_FS, fontweight="bold")
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
