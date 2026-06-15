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
    """Paper-friendly matrix size: scales with the shape but caps the width so
    a wide pilot matrix doesn't become a giant image that the notebook then
    shrinks to an unreadable thumbnail."""
    return (min(0.55 * n_cols + 3, 19.0), min(0.5 * n_rows + 2.5, 12.0))


def _annot_fontsize(n_cols: int) -> int:
    """Largest annotation font that still fits a cell at the capped width."""
    return 11 if n_cols <= 14 else 10 if n_cols <= 20 else 9 if n_cols <= 30 else 7


def _foundation_methods() -> set:
    """Foundation-model names (for highlighting); empty set if unavailable."""
    try:
        from src.methods.method_config import FOUNDATION_METHODS
        return set(FOUNDATION_METHODS)
    except Exception:  # pragma: no cover -- keep plotting usable standalone
        return set()


def _color_foundation_ticks(ax) -> None:
    """Render tabular-foundation-model names in red+bold on the x axis, so
    they stand out in every chart."""
    fnd = _foundation_methods()
    if not fnd:
        return
    for lbl in ax.get_xticklabels():
        if lbl.get_text() in fnd:
            lbl.set_color("crimson")
            lbl.set_fontweight("bold")


def _style_method_axis(ax) -> None:
    """Uniform per-method x axis: 45-deg right-aligned labels at TICK_FS, with
    foundation-model names highlighted in red."""
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_horizontalalignment("right")
    _color_foundation_ticks(ax)


def _method_bar(
    series: pd.Series,
    *,
    title: str,
    ylabel: str,
    stem: str,
    out_dir: Optional[Path],
    errs: Optional[pd.Series] = None,
    logy: bool = False,
    value_fmt: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Optional[Path]:
    """Consistent vertical per-method bar chart used by every bar plot here:
    a green(best)->red(worst) gradient, a black contour per bar, optional
    ``± std`` error bars, the metric's average printed above each bar, and the
    shared label styling (foundation models in red). ``series`` must be sorted
    best-first so the gradient lines up with performance. Width is capped for
    paper figures."""
    names = list(series.index)
    n = len(names)
    vals = series.to_numpy(dtype=float)
    # errs may be a Series (symmetric) or a ready (2, N) array (asymmetric).
    if errs is None:
        yerr = None
    elif isinstance(errs, pd.Series):
        yerr = errs.reindex(series.index).to_numpy()
    else:
        yerr = np.asarray(errs)
    fig, ax = plt.subplots(figsize=figsize or (min(0.42 * n + 3, 18.0), 5.2))
    bars = ax.bar(
        range(n), vals,
        color=_best_to_worst_colors(n), edgecolor="black", linewidth=0.8,
        yerr=yerr, capsize=4 if yerr is not None else 0,
        # Navy error bars read clearly over the green→red bar gradient.
        error_kw={"ecolor": "#0b1f4d", "lw": 2.0, "capthick": 2.0, "zorder": 5},
    )
    if logy:
        ax.set_yscale("log")
    if value_fmt is None:
        vmax = float(np.nanmax(np.abs(vals))) if n else 0.0
        value_fmt = "{:.3f}" if vmax < 10 else "{:.1f}"
    ax.bar_label(bars, labels=[value_fmt.format(v) for v in vals],
                 padding=8 if errs is not None else 4, fontsize=8.5, fontweight="bold")
    ax.margins(y=0.16)  # headroom so the value labels aren't clipped
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
    fmt: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Datasets x methods heatmap of ``metric`` (mean across folds).

    Columns are ALWAYS sorted best -> worst from left to right, where "best"
    follows ``higher_is_better`` (e.g. highest AUC left; lowest Brier left).
    The colormap likewise maps green to good: pass ``higher_is_better=False``
    for lower-is-better metrics and the default cmap flips to ``RdYlGn_r``.
    The annotation font auto-scales to the matrix width (and drops to 2
    decimals for very wide pilot matrices) so the numbers stay readable, and
    foundation-model column names are shown in red.
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
    if fmt is None:  # fewer decimals when the matrix is wide, so digits fit
        fmt = ".3f" if n_cols <= 30 else ".2f"
    fig, ax = plt.subplots(figsize=figsize or _heatmap_figsize(n_rows, n_cols))
    pm = _pretty_metric(metric)
    common = dict(annot=True, fmt=fmt, cmap=cmap, linewidths=0.5, ax=ax,
                  annot_kws={"size": _annot_fontsize(n_cols)},
                  cbar_kws={"label": pm})
    if metric.upper() in {"R2"}:  # diverging, centred on zero
        abs_max = float(np.nanmax(np.abs(pivot.values)))
        sns.heatmap(pivot, center=0, vmin=-abs_max, vmax=abs_max, **common)
    else:
        sns.heatmap(pivot, vmin=float(np.nanmin(pivot.values)),
                    vmax=float(np.nanmax(pivot.values)), **common)
    ax.set_title(f"{task_name} performance: {pm} (datasets x methods)",
                 fontsize=TITLE_FS + 4, fontweight="bold", pad=20)
    ax.set_xlabel("Method", fontsize=LABEL_FS + 1, fontweight="bold")
    ax.set_ylabel("Dataset", fontsize=LABEL_FS + 1, fontweight="bold")
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
    """Bar chart of each method's mean rank for ``metric`` (1 = best), with ±
    std error bars. The rank is taken WITHIN each (dataset, fold), so the std
    is fold-level; the lower whisker is clipped so it can never dip below the
    best-possible rank of 1 (no impossible below-zero bars)."""
    rk = _fold_ranks(df, metric, higher_is_better)
    mean_rank = rk["mean"].sort_values()                        # lowest (best) first
    std = rk["std"].reindex(mean_rank.index).fillna(0.0).to_numpy()
    mean = mean_rank.to_numpy()
    lower = np.minimum(std, np.maximum(mean - 1.0, 0.0))        # never cross rank 1
    yerr = np.vstack([lower, std])
    pm = _pretty_metric(metric)
    return _method_bar(
        mean_rank, errs=yerr, value_fmt="{:.2f}",
        title=f"{task_name} method ranking by {pm} (mean rank ± fold std; lower = better)",
        ylabel=f"mean rank ({pm}; lower is better)",
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
    smooth: bool = False,
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
        if smooth:
            # Moving average across sweep points (trend, not the raw wiggle).
            win = max(3, len(y) // 6)
            y_ma = pd.Series(y.to_numpy()).rolling(win, min_periods=1, center=True).mean()
            ax.plot(g["sweep_value"], y_ma.to_numpy(), lw=2.2, label=method, color=color)
        else:
            ax.plot(g["sweep_value"], y, marker="o", ms=2.0, lw=1.6,
                    label=method, color=color)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=14))
    if relative:
        ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
    pm = _pretty_metric(metric)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel(f"{pm} (% of each method's own best)" if relative else pm,
                  fontsize=LABEL_FS, fontweight="bold")
    # The line is averaged over folds; per equal fold counts that equals the
    # mean over datasets, so we keep the familiar "mean over datasets" label.
    note = []
    if relative:
        note.append("relative to each method's own best")
    if smooth:
        note.append("moving average")
    note.append("mean over datasets")
    ax.set_title(f"{title} ({'; '.join(note)})", fontsize=TITLE_FS, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    suffix = ("_relative" if relative else "") + ("_smooth" if smooth else "")
    return _save(fig, out_dir, plot_name + suffix)


def learning_curve(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    figsize: Tuple[int, int] = (14, 8),
    relative: bool = False,
    smooth: bool = False,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Experiment 2: ``metric`` vs training rows -- one line per method,
    averaged over every included dataset. ``relative=True`` divides each
    method's curve by its own best value; ``smooth=True`` plots the moving
    average instead of the raw points."""
    return _sweep_curve(
        df, sweep_axis="row_limit", metric=metric,
        title=f"{task_name} learning curve: {_pretty_metric(metric)}",
        xlabel="Training rows", figsize=figsize, out_dir=out_dir,
        relative=relative, smooth=smooth,
        plot_name=f"{task_name.lower()}_learning_curve_{metric.lower()}",
    )


def imbalance_curve(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    figsize: Tuple[int, int] = (14, 8),
    relative: bool = False,
    smooth: bool = False,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Experiment 3: ``metric`` vs minority proportion -- one line per method,
    averaged over every included dataset. ``relative=True`` divides each
    method's curve by its own best value; ``smooth=True`` plots the moving
    average instead of the raw points."""
    return _sweep_curve(
        df, sweep_axis="minority_proportion", metric=metric,
        title=f"{task_name} imbalance robustness: {_pretty_metric(metric)}",
        xlabel="Minority-class proportion", figsize=figsize, out_dir=out_dir,
        relative=relative, smooth=smooth,
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
    """Box + strip of the metric's distribution across datasets (one point per
    dataset = its fold-mean), one box per method."""
    col = _resolve_mean_column(df, metric)
    # One value per (dataset, method) so the box shows the cross-DATASET spread
    # regardless of whether ``df`` is per-fold or per-method.
    per_ds = df.groupby(["dataset", "method"], as_index=False)[col].mean()
    order = (per_ds.groupby("method")[col].median()
             .sort_values(ascending=not higher_is_better).index)
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=per_ds, x="method", y=col, order=order, color="#cfe8ff", ax=ax)
    sns.stripplot(data=per_ds, x="method", y=col, order=order,
                  color="#1f4e79", size=4, alpha=0.6, ax=ax)
    _style_method_axis(ax)
    pm = _pretty_metric(metric)
    ax.set_ylabel(pm, fontsize=LABEL_FS, fontweight="bold")
    ax.set_xlabel("")
    ax.set_title(f"{task_name} {pm} distribution across datasets",
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
    """Bar chart of the ``agg`` (mean/median) of ``metric`` per method, sorted
    best -> worst. The ``mean`` variant carries ± std error bars; when ``df``
    is the per-fold frame that std is the **fold-level** std (pooled over all
    dataset×fold observations of the method), not the across-dataset spread."""
    col = _resolve_mean_column(df, metric)
    grp = df.groupby("method")[col]
    vals = grp.agg(agg).sort_values(ascending=not higher_is_better)
    errs = grp.std() if agg == "mean" else None
    pm = _pretty_metric(metric)
    ylabel = (f"{pm} (mean ± std)" if agg == "mean" else f"{pm} (median)")
    return _method_bar(
        vals, errs=errs,
        title=f"{task_name}: {agg} {pm} per method",
        ylabel=ylabel,
        stem=f"{task_name.lower()}_bar_{agg}_{metric.lower()}",
        out_dir=out_dir, figsize=figsize,
    )


def compute_time_bars(
    df: pd.DataFrame,
    *,
    task_name: str = "PD",
    figsize: Optional[Tuple[float, float]] = None,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """MEDIAN TOTAL compute time per method = fit + predict, in seconds
    (log y; fastest = greenest, left). Uses fit + predict because in-context
    models (TabPFN v1/v2/Real, Mitra) report ~0 fit time -- their cost is at
    predict -- so a fit-only bar would drop them off a log axis. No error bars
    (a ± band reads poorly on a log scale); the median is printed above each
    bar."""
    vals = _total_time(df, agg="median")
    if vals is None:
        logger.warning("compute_time_bars: no train_time/predict_time columns")
        return None
    return _method_bar(
        vals, errs=None, logy=True, value_fmt="{:.1f}",
        title=f"{task_name}: median compute time per method (fit + predict, seconds)",
        ylabel="median compute time per fold (s)",
        stem=f"{task_name.lower()}_bar_compute_time",
        out_dir=out_dir, figsize=figsize,
    )


def compute_time_boxplot(
    df: pd.DataFrame,
    *,
    task_name: str = "PD",
    figsize: Tuple[int, int] = (16, 7),
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Box + strip of TOTAL compute time (fit + predict) per method, **log y**,
    one point per (dataset, fold). Ordered fastest-median first; foundation
    names in red. The log axis is essential -- times span several orders of
    magnitude across methods."""
    if "train_time" in df.columns:
        pt = df["predict_time"] if "predict_time" in df.columns else 0.0
        total = df["train_time"].fillna(0.0) + (pt.fillna(0.0) if hasattr(pt, "fillna") else pt)
    elif "train_time_mean" in df.columns:
        pt = df["predict_time_mean"] if "predict_time_mean" in df.columns else 0.0
        total = df["train_time_mean"].fillna(0.0) + (pt.fillna(0.0) if hasattr(pt, "fillna") else pt)
    else:
        logger.warning("compute_time_boxplot: no time columns")
        return None
    work = pd.DataFrame({"method": df["method"].to_numpy(), "t": total.to_numpy()})
    work = work[work["t"] > 0]  # log axis can't show zeros
    order = work.groupby("method")["t"].median().sort_values().index
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=work, x="method", y="t", order=order, color="#cfe8ff", ax=ax)
    sns.stripplot(data=work, x="method", y="t", order=order,
                  color="#1f4e79", size=4, alpha=0.6, ax=ax)
    ax.set_yscale("log")
    _style_method_axis(ax)
    ax.set_ylabel("compute time per fold (s, log)", fontsize=LABEL_FS, fontweight="bold")
    ax.set_xlabel("")
    ax.set_title(f"{task_name} compute-time distribution (fit + predict, fastest left)",
                 fontsize=TITLE_FS, fontweight="bold")
    plt.tight_layout()
    return _save(fig, out_dir, f"{task_name.lower()}_box_compute_time")


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
    pm = _pretty_metric(metric)
    sns.heatmap(ranks, annot=True, fmt=".0f", cmap="RdYlGn_r",
                vmin=1, vmax=n_cols, annot_kws={"size": _annot_fontsize(n_cols) + 1},
                cbar_kws={"label": f"rank by {pm} (1 = best)"},
                linewidths=0.5, ax=ax)
    ax.set_title(f"{task_name} rank matrix by {pm} (1 = best; best mean rank left)",
                 fontsize=TITLE_FS + 4, fontweight="bold", pad=20)
    ax.set_xlabel("Method", fontsize=LABEL_FS + 1, fontweight="bold")
    ax.set_ylabel("Dataset", fontsize=LABEL_FS + 1, fontweight="bold")
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
    # Same blue palette as metric_boxplots for one consistent layout.
    sns.boxplot(data=long, x="method", y="rank", order=order, color="#cfe8ff", ax=ax)
    sns.stripplot(data=long, x="method", y="rank", order=order,
                  color="#1f4e79", size=4, alpha=0.6, ax=ax)
    ax.invert_yaxis()  # rank 1 (best) on top
    _style_method_axis(ax)
    pm = _pretty_metric(metric)
    ax.set_ylabel(f"rank by {pm} (1 = best)", fontsize=LABEL_FS, fontweight="bold")
    ax.set_xlabel("")
    ax.set_title(f"{task_name} rank distribution across datasets ({pm})",
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
    pm = _pretty_metric(metric)
    ax.set_xlabel(f"HPO improvement in {pm} (positive = tuning helps)",
                  fontsize=LABEL_FS, fontweight="bold")
    ax.set_title(f"{task_name}: effect of hyper-parameter tuning on {pm}",
                 fontsize=TITLE_FS, fontweight="bold")
    plt.tight_layout()
    return _save(fig, out_dir, f"{task_name.lower()}_hpo_effect_{metric.lower()}")


def runtime_performance_scatter(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    higher_is_better: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Compute cost (fit + predict, log x) vs mean metric -- the cost/quality
    frontier. Foundation models are drawn as red stars, everything else as
    blue circles; every point gets a leader line to its label so the mapping
    is unambiguous, and the Pareto-best methods (top-left: cheap + accurate)
    sit toward the upper left."""
    from matplotlib.lines import Line2D

    mean_col = _resolve_mean_column(df, metric)
    total = _total_time(df)
    if total is None:
        logger.warning("runtime_performance_scatter: no time columns")
        return None
    perf = df.groupby("method")[mean_col].mean()
    agg = (pd.DataFrame({"time": total, "perf": perf}).dropna()
           .reset_index().rename(columns={"index": "method"}))
    agg = agg.sort_values("perf", ascending=not higher_is_better).reset_index(drop=True)
    fnd = _foundation_methods()

    fig, ax = plt.subplots(figsize=figsize or (12, 8))
    ax.set_xscale("log")
    for r in agg.itertuples():
        is_f = r.method in fnd
        ax.scatter(r.time, r.perf, s=150 if is_f else 90,
                   marker="*" if is_f else "o",
                   color="crimson" if is_f else "#1f77b4",
                   edgecolor="black", linewidth=0.7, zorder=3)
    # Labels in a RESERVED RIGHT BAND, evenly spaced in y (so no two labels can
    # overlap and none sits on a dot), each joined to its dot by a leader line.
    # Both dots and labels are ordered by performance, so the lines stay
    # roughly parallel and rarely cross. The x-limits are widened on the log
    # axis to free ~40% of the width on the right for the label column.
    xlo, xhi = agg["time"].min(), agg["time"].max()
    ratio = (xhi / xlo) if xlo > 0 else 10.0
    ax.set_xlim(xlo / ratio ** 0.06, xlo * ratio ** (1.0 / 0.60))
    ax.margins(y=0.08)
    order = agg.sort_values("perf", ascending=False).reset_index(drop=True)
    n = len(order)
    y_fracs = np.linspace(0.975, 0.025, n) if n > 1 else [0.5]
    for yf, r in zip(y_fracs, order.itertuples()):
        is_f = r.method in fnd
        ax.annotate(
            r.method, xy=(r.time, r.perf), xycoords="data",
            xytext=(0.66, yf), textcoords=ax.transAxes,
            ha="left", va="center", fontsize=9,
            color="crimson" if is_f else "black",
            fontweight="bold" if is_f else "normal",
            arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6, shrinkA=0, shrinkB=4),
        )
    ax.grid(True, which="both", alpha=0.25)
    pm = _pretty_metric(metric)
    ax.set_xlabel("compute time per fold — fit + predict (s, log scale)",
                  fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel(f"mean {pm}", fontsize=LABEL_FS, fontweight="bold")
    ax.set_title(f"{task_name}: performance vs compute cost", fontsize=TITLE_FS, fontweight="bold")
    ax.legend(handles=[
        Line2D([0], [0], marker="*", color="w", markerfacecolor="crimson",
               markeredgecolor="black", markersize=14, label="foundation model"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
               markeredgecolor="black", markersize=10, label="other"),
    ], loc="best", fontsize=10)
    plt.tight_layout()
    return _save(fig, out_dir, f"{task_name.lower()}_cost_quality_{metric.lower()}")


# ---------------------------------------------------------------------------
#  Copy-pasteable text summary (printed at the end of a results notebook)
# ---------------------------------------------------------------------------

# Preferred column order per task (only those actually present are shown).
_PD_METRIC_ORDER = ["AUC", "Gini", "KS", "AP_normalized", "F1", "Accuracy",
                    "Balanced_Accuracy", "MCC", "Brier", "ECE", "LogLoss"]
_LGD_METRIC_ORDER = ["R2", "RMSE", "MAE", "Spearman_Corr", "Pearson_Corr"]


def _summary_text(df: pd.DataFrame, *, task_name: str, metric_order, sort_by, higher_is_better) -> str:
    """Shared core: one row per method, one column per metric (mean across all
    fold×dataset observations), plus the MEDIAN compute time. All-NaN columns
    (a metric the task never emits) are dropped, so PD tables carry no orphan
    regression columns and vice versa."""
    present: Dict[str, str] = {}
    for c in df.columns:
        if not c.startswith("metric.") or c.endswith("_std"):
            continue
        name = c[len("metric."):]
        name = name[:-5] if name.endswith("_mean") else name
        present.setdefault(name, c)

    # Order: the task's preferred metrics first, then any extras.
    names = [m for m in metric_order if m in present] + \
            [m for m in present if m not in metric_order]
    table = pd.DataFrame({_pretty_metric(n): df.groupby("method")[present[n]].mean()
                          for n in names})
    table = table.dropna(axis=1, how="all")   # drop metrics this task never emits
    total = _total_time(df, agg="median")
    if total is not None:
        table["time_s (median)"] = total

    if sort_by and _pretty_metric(sort_by) in table.columns:
        table = table.sort_values(_pretty_metric(sort_by), ascending=not higher_is_better)
    else:
        table = table.sort_index()            # alphabetical -- neutral, not a ranking

    header = (f"{task_name} — mean of every metric per method   "
              f"(datasets: {df['dataset'].nunique()}, methods: {df['method'].nunique()})")
    text = header + "\n" + "=" * max(len(header), 64) + "\n" + table.round(4).to_string()
    print(text)
    return text


def pd_summary_text(df: pd.DataFrame, *, task_name: str = "PD",
                    sort_by: Optional[str] = "AUC", higher_is_better: bool = True) -> str:
    """Copy-pasteable table of the mean of every **PD (classification)** metric
    per method, plus median compute time. Not a leaderboard (full list); pass
    ``sort_by=None`` for alphabetical order."""
    return _summary_text(df, task_name=task_name, metric_order=_PD_METRIC_ORDER,
                         sort_by=sort_by, higher_is_better=higher_is_better)


def lgd_summary_text(df: pd.DataFrame, *, task_name: str = "LGD",
                     sort_by: Optional[str] = "R2", higher_is_better: bool = True) -> str:
    """Copy-pasteable table of the mean of every **LGD (regression)** metric
    per method, plus median compute time. Not a leaderboard (full list); pass
    ``sort_by=None`` for alphabetical order."""
    return _summary_text(df, task_name=task_name, metric_order=_LGD_METRIC_ORDER,
                         sort_by=sort_by, higher_is_better=higher_is_better)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _resolve_mean_column(df: pd.DataFrame, metric: str) -> str:
    """Best metric column. Prefer the RAW per-fold ``metric.<X>`` so bars can
    compute fold-level std (and everything else aggregates it by mean); fall
    back to the pre-aggregated ``metric.<X>_mean`` of the per-method summary."""
    for cand in (f"metric.{metric}", f"metric.{metric}_mean", f"{metric}_mean", metric):
        if cand in df.columns:
            return cand
    raise KeyError(
        f"Could not find a column for {metric!r} in df with columns "
        f"{list(df.columns)[:20]}"
    )


def _pretty_metric(metric: str) -> str:
    """Display name for a metric (e.g. ``R2`` -> ``R²``). Column lookups and
    file names keep the raw key; only what the reader sees is prettified."""
    return "R²" if str(metric).upper() == "R2" else str(metric)


def _total_time(df: pd.DataFrame, agg: str = "median") -> Optional[pd.Series]:
    """TOTAL compute time per method = fit + predict, in seconds, aggregated by
    ``agg`` (default MEDIAN -- robust to a slow outlier fold/dataset).

    In-context models (TabPFN v1/v2/Real, Mitra) report ``fit_time = 0`` by
    design -- their cost lives in ``predict`` -- so plotting fit-time alone
    drops them from a log axis. Fit+predict is the fair, always-positive cost.
    """
    if "train_time" in df.columns:                 # per-fold frame
        pt = df["predict_time"] if "predict_time" in df.columns else 0.0
        total = df["train_time"].fillna(0.0) + (pt.fillna(0.0) if hasattr(pt, "fillna") else pt)
    elif "train_time_mean" in df.columns:           # per-method frame
        pt = df["predict_time_mean"] if "predict_time_mean" in df.columns else 0.0
        total = df["train_time_mean"].fillna(0.0) + (pt.fillna(0.0) if hasattr(pt, "fillna") else pt)
    else:
        return None
    return total.groupby(df["method"]).agg(agg).sort_values()


def _fold_ranks(df: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    """Per-method mean & std of the method's rank, ranked WITHIN each
    (dataset, fold) when fold ids are present (so the spread is fold-level),
    else within each dataset. Returns columns ``mean`` / ``std``."""
    col = _resolve_mean_column(df, metric)
    if "fold_id" in df.columns:
        sub = df[["dataset", "fold_id", "method", col]].copy()
        sub["rank"] = sub.groupby(["dataset", "fold_id"])[col].rank(
            ascending=not higher_is_better)
        g = sub.groupby("method")["rank"]
        return pd.DataFrame({"mean": g.mean(), "std": g.std()})
    ranks = _rank_pivot(df, metric, higher_is_better)
    return pd.DataFrame({"mean": ranks.mean(axis=0), "std": ranks.std(axis=0)})


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
    "compute_time_bars",
    "compute_time_boxplot",
    "rank_heatmap",
    "rank_boxplots",
    "hpo_improvement_bars",
    "runtime_performance_scatter",
    "pd_summary_text",
    "lgd_summary_text",
]
