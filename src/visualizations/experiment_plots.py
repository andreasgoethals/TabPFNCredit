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

# Standard, consistent figure labels (TALENT-free import: safe everywhere).
from src.methods.method_names import display_name as _display_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Style
# ---------------------------------------------------------------------------

# ============================================================================
#  Canonical paper-wide style -- ONE size per element, shared by EVERY figure
#  here AND in ``statistical_testing.py`` (which imports these names). The goal:
#  every generated PDF uses the same title / axis-label / tick / legend /
#  value-label / call-out sizes and the same marker, line and bar styling, so the
#  figures read as one coherent set when dropped into the paper. Change a size in
#  exactly one place -- here.
# ============================================================================
TITLE_FS  = 16   # panel title (every figure)
LABEL_FS  = 14   # x / y axis titles
TICK_FS   = 12   # tick labels: method names, dataset names, numeric ticks
LEGEND_FS = 12   # legend / colour-index entries
VALUE_FS  = 11   # numbers printed on/above bars and inside heatmap cells
NOTE_FS   = 11   # call-out boxes + per-point (dataset) labels
ANNOT_FS  = VALUE_FS  # backwards-compatible alias (heatmap cell numbers)

# Element geometry -- identical across figures.
PT_NORMAL = 70.0    # scatter marker area (pt^2) for an ordinary method
PT_FND    = 130.0   # scatter marker area for a foundation-model marker
EDGE_LW   = 0.8     # black contour width on bars and scatter markers
CURVE_LW  = 2.0     # line width for the pooled sweep curves
RUN_MS    = 1.5     # marker size for the individual-run dots on every sweep curve
                    # (Experiment 2 & 3, PD & LGD) -- one size everywhere

# Canonical figure geometry. Two charts of the same shape class get the SAME
# width, so when included at one width in LaTeX their fonts scale identically.
FIG_SQUARE = (7.5, 7.5)   # head-to-head scatter (equal x/y aspect)
FIG_WIDE   = (9.5, 6.0)   # single-panel landscape: curves, cost/quality, trend


def _vbar_figsize(n: int) -> Tuple[float, float]:
    """Size for every VERTICAL methods-on-x chart (bar / box / rank box /
    compute bar / compute box). Width grows with the method count so the slanted
    labels never collide; height is fixed, so all such charts of the same method
    set are byte-for-byte identical and scale the same way in the paper."""
    return (round(min(0.50 * n + 4.0, 19.0), 2), 6.0)


def _hbar_figsize(n: int) -> Tuple[float, float]:
    """Size for every HORIZONTAL methods-on-y chart (HPO effect / PAMA /
    %-of-max). Fixed width; height grows with the method count."""
    return (10.0, round(max(4.5, 0.40 * n + 2.0), 2))


DEFAULT_RC = {
    "figure.figsize": FIG_WIDE,
    "savefig.dpi": 200,
    "font.size": TICK_FS,
    "axes.titlesize": TITLE_FS,
    "axes.titleweight": "bold",
    "axes.labelsize": LABEL_FS,
    "axes.labelweight": "bold",
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "legend.fontsize": LEGEND_FS,
    "legend.framealpha": 0.92,
    "lines.linewidth": CURVE_LW,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

# Well-known baselines worth labelling in an otherwise-crowded scatter (the
# foundation models are always added on top of this at call time). Listed in a
# few likely spellings so the match is robust to registry naming.
NOTABLE_METHODS = {
    "LogReg", "logreg", "LinearRegression", "xgboost", "XGBoost",
    "catboost", "CatBoost", "lightgbm", "LightGBM", "randomforest",
    "RandomForest", "mlp", "MLP", "knn", "KNN", "svm", "SVM",
    "NaiveBayes", "ftt", "FTTransformer", "resnet", "tabnet", "node",
}

# Fixed method -> colour map for the SWEEP curves (Experiments 2 & 3), matching
# the tab10 order used by the existing per-dataset sweep plots. Keeping this
# map explicit means adding CatBoost later will not shift the colours of the
# four methods that are already in the notebooks.
SWEEP_METHOD_COLORS = {
    "LogReg":           "#1f77b4",   # tab10 blue
    "LinearRegression": "#1f77b4",   # tab10 blue
    "tabicl_v2":        "#ff7f0e",   # tab10 orange
    "tabpfn_v3":        "#2ca02c",   # tab10 green
    "xgboost":          "#d62728",   # tab10 red
    "catboost":         "#9467bd",   # tab10 purple
}
# Remaining tab10 colours cycled for any method not pinned above.
_SWEEP_FALLBACK_COLORS = [
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#000000"
]


def _sweep_colors(methods) -> dict:
    """Colour per sweep method: the fixed map first, then the fallback cycle
    for anything unmapped (deterministic in ``methods`` order)."""
    out, i = {}, 0
    for m in methods:
        if m in SWEEP_METHOD_COLORS:
            out[m] = SWEEP_METHOD_COLORS[m]
        else:
            out[m] = _SWEEP_FALLBACK_COLORS[i % len(_SWEEP_FALLBACK_COLORS)]
            i += 1
    return out


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


def _strip_size(n_per_method: int) -> float:
    """Marker size for the strip-plot dots overlaid on box plots, tied to how
    many points each method contributes: bigger when there are FEWER (so the
    sparser LGD plots read clearly), smaller when dense (so PD doesn't blob).
    Clamped to a sane range so every box plot's dots stay within one size band."""
    return float(np.clip(18.0 / max(n_per_method, 1) ** 0.5, 4.0, 8.0))


# A metric's natural lower bound for a TRUNCATED y axis: AUC at the random-
# classifier 0.5, R2 at 0 (predicting the mean). Other metrics get no floor.
_METRIC_BASELINE = {"AUC": 0.5, "R2": 0.0}


def _metric_floor(metric: str, values) -> Optional[float]:
    """Lower y-limit for a bar/box of ``metric``, or ``None`` if the metric has
    no meaningful floor. AUC starts at ``min(0.5, lowest value shown)`` -- 0.5
    (random) unless something dips below it, in which case the lowest value is
    shown. R2 is floored hard at 0 ("only show from 0")."""
    m = str(metric).upper()
    base = _METRIC_BASELINE.get(m)
    if base is None:
        return None
    if m == "R2":
        return 0.0
    arr = np.asarray(values, dtype=float)
    lo = float(np.nanmin(arr)) if arr.size else base
    return min(base, lo)


def _relative_metric_gain(
    fnd,
    base,
    metric: str,
    *,
    higher_is_better: bool = True,
):
    """Relative head-to-head gain used by the size-trend plots.

    This intentionally mirrors the PD AUC plot: percentage improvement over
    the baseline metric value, i.e. 100 * (model - baseline) / |baseline| for
    higher-is-better metrics.
    """
    fnd = np.asarray(fnd, dtype=float)
    base = np.asarray(base, dtype=float)
    diff = (fnd - base) if higher_is_better else (base - fnd)
    denom = np.abs(base)
    denom = np.where(np.abs(denom) < 1e-9, np.nan, denom)
    return 100.0 * diff / denom


def _heatmap_figsize(n_rows: int, n_cols: int) -> Tuple[float, float]:
    """Paper-friendly matrix size: scales with the shape but caps the width so
    a wide pilot matrix doesn't become a giant image that the notebook then
    shrinks to an unreadable thumbnail."""
    return (min(0.55 * n_cols + 3, 19.0), min(0.5 * n_rows + 2.5, 12.0))


def _annot_fontsize(n_cols: int) -> int:
    """Largest annotation font that still fits a cell at the capped width.
    Sized up for print legibility."""
    return 14 if n_cols <= 14 else 12 if n_cols <= 20 else 11 if n_cols <= 30 else 9


def _foundation_methods() -> set:
    """Foundation-model names (for highlighting); empty set if unavailable."""
    try:
        from src.methods.method_config import FOUNDATION_METHODS
        return set(FOUNDATION_METHODS)
    except Exception:  # pragma: no cover -- keep plotting usable standalone
        return set()


def _color_foundation_ticks(ax, axis: str = "x") -> None:
    """Relabel a per-method tick ``axis`` ("x"/"y") with the standard display
    names AND render foundation-model names in crimson + bold, so naming is
    identical in every chart (this module AND the stats module, which reuses it).

    Matching is done on the RAW registry key *before* relabelling, so it is
    correct whether the tick text is still a raw key or already a display name.
    Non-method ticks (e.g. dataset names) pass through ``display_name``
    unchanged, so calling this on a non-method axis is a harmless no-op.
    Existing rotation / alignment is preserved across the relabel."""
    from matplotlib.ticker import FixedLocator
    get = ax.get_xticklabels if axis == "x" else ax.get_yticklabels
    setlab = ax.set_xticklabels if axis == "x" else ax.set_yticklabels
    cur = get()
    raws = [t.get_text() for t in cur]
    if not any(raws):                      # ticks not drawn yet -> nothing to do
        return
    rot = cur[0].get_rotation()
    ha = cur[0].get_horizontalalignment()
    # Pin a FixedLocator at the current positions so relabelling is legitimate
    # (no "FixedFormatter without FixedLocator" warning) and survives a redraw.
    locs = ax.get_xticks() if axis == "x" else ax.get_yticks()
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_locator(FixedLocator(list(locs)))
    fnd = _foundation_methods()
    fnd_disp = {_display_name(m) for m in fnd}
    new = setlab([_display_name(r) for r in raws], rotation=rot, ha=ha)
    for lbl, raw in zip(new, raws):
        if raw in fnd or raw in fnd_disp:
            lbl.set_color("crimson")
            lbl.set_fontweight("bold")


def _style_method_axis(ax, *, connect: bool = False) -> None:
    """Uniform per-method x axis: standard display labels, 45-deg right-aligned
    at TICK_FS, with foundation-model names highlighted in red. ``connect=True``
    draws a thin dotted vertical line at each tick (from the x-axis up to the
    box/point) so it is unambiguous which method a box belongs to."""
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    _color_foundation_ticks(ax)            # relabel -> display names + foundation red
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_horizontalalignment("right")
    if connect:
        for x in ax.get_xticks():
            ax.axvline(x, color="0.6", lw=0.6, ls=":", alpha=0.7, zorder=0)


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
    y_floor: Optional[float] = None,
) -> Optional[Path]:
    """Consistent vertical per-method bar chart used by every bar plot here:
    a green(best)->red(worst) gradient, a black contour per bar, optional
    ``± std`` error bars, the metric's average printed above each bar, and the
    shared label styling (foundation models in red). ``series`` must be sorted
    best-first so the gradient lines up with performance. ``y_floor`` truncates
    the y axis at a given value (e.g. AUC at 0.5) instead of starting at 0; width
    follows the shared :func:`_vbar_figsize` rule for a uniform paper look."""
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
    fig, ax = plt.subplots(figsize=figsize or _vbar_figsize(n))
    bars = ax.bar(
        range(n), vals,
        color=_best_to_worst_colors(n), edgecolor="black", linewidth=EDGE_LW,
        yerr=yerr, capsize=4 if yerr is not None else 0,
        # Navy error bars read clearly over the green→red bar gradient.
        error_kw={"ecolor": "#0b1f4d", "lw": 2.0, "capthick": 2.0, "zorder": 5},
    )
    if logy:
        ax.set_yscale("log")
    if value_fmt is None:
        vmax = float(np.nanmax(np.abs(vals))) if n else 0.0
        value_fmt = "{:.3f}" if vmax < 10 else "{:.1f}"
    # Value labels rotated 45° and placed ABOVE each bar. Using matplotlib's
    # bar_label (default rotation_mode) aligns the whole rotated text box just
    # above the bar edge, so a number can never dip INSIDE the bar -- including on
    # a log axis. The 45° slant clears the taller neighbour.
    ax.bar_label(bars, labels=[value_fmt.format(v) for v in vals],
                 padding=9 if errs is not None else 5, fontsize=VALUE_FS,
                 fontweight="bold", rotation=45)
    head = 0.34 if logy else 0.22
    if y_floor is not None and not logy:
        # Truncated axis: start at y_floor; compute the top from the bars (and
        # their upper whisker, if any) and add headroom -- relative to the
        # truncated span -- for the slanted value labels.
        tops = vals.astype(float)
        if yerr is not None:
            up = yerr[1] if (isinstance(yerr, np.ndarray) and yerr.ndim == 2) else np.asarray(yerr)
            tops = tops + np.nan_to_num(up)
        top = float(np.nanmax(tops)) if n else y_floor + 1.0
        span = max(top - y_floor, 1e-9)
        ax.set_ylim(y_floor, top + (head + 0.06) * span)
    else:
        ax.margins(y=head)
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

# Experiments whose summary CSVs were already rebuilt in this kernel session, so
# opening a notebook refreshes them ONCE (not on every load_summary call).
_SUMMARIZED_THIS_SESSION: set = set()


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
    exp = experiment.lower()
    suffix = "per_method.csv" if aggregated else "per_fold.csv"
    path = summary_dir / f"{exp}_{suffix}"
    if auto_summarize:
        results_root = summary_dir.parent
        exp_dir = results_root / exp
        results_present = exp_dir.is_dir() and next(exp_dir.rglob("*.json"), None) is not None
        # Refresh the CSVs ONCE per experiment per session whenever the result
        # files are present -- so simply RUNNING a notebook rebuilds the summaries
        # from the latest results (old CSVs are deleted first in summarize_to_csv).
        # A CSV-only download (no result files locally) just reads the CSV as-is.
        if results_present and exp not in _SUMMARIZED_THIS_SESSION:
            from src.utils.result_summary import summarize_to_csv
            logger.info("Refreshing %s summaries from %s ...", exp, results_root)
            summarize_to_csv(base=results_root, experiment=exp, out_dir=summary_dir)
            _SUMMARIZED_THIS_SESSION.add(exp)
        elif not path.exists():
            from src.utils.result_summary import summarize_to_csv
            logger.info("Summary %s missing -- building it from %s ...", path.name, results_root)
            summarize_to_csv(base=results_root, experiment=exp, out_dir=summary_dir)
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
                 fontsize=TITLE_FS, fontweight="bold", pad=20)
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
    figsize: Optional[Tuple[int, int]] = None,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Bar chart of each method's mean rank for ``metric`` (1 = best), with ±
    std error bars. Uses the SAME per-dataset ranks as :func:`rank_heatmap`
    (one rank per dataset, via ``_rank_pivot``), so the matrix and this bar are
    ordered identically. The lower whisker is clipped so it can never dip below
    the best-possible rank of 1 (no impossible below-zero bars)."""
    ranks = _rank_pivot(df, metric, higher_is_better)           # per-dataset; matches the matrix
    mean_rank = ranks.mean(axis=0).sort_values()               # lowest (best) first
    std = ranks.std(axis=0).reindex(mean_rank.index).fillna(0.0).to_numpy()
    mean = mean_rank.to_numpy()
    lower = np.minimum(std, np.maximum(mean - 1.0, 0.0))        # never cross rank 1
    yerr = np.vstack([lower, std])
    pm = _pretty_metric(metric)
    return _method_bar(
        mean_rank, errs=yerr, value_fmt="{:.2f}",
        title=f"{task_name} method ranking by {pm} (mean rank ± std across datasets; lower = better)",
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
    ax.tick_params(labelsize=TICK_FS)
    ax.set_ylabel(_pretty_metric(metric), fontsize=LABEL_FS, fontweight="bold")
    ax.set_title(f"{task_name} per-dataset {_pretty_metric(metric)}",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.0), fontsize=LEGEND_FS)
    leg = ax.get_legend()                                  # standardise method names
    if leg is not None:
        for t in leg.get_texts():
            t.set_text(_display_name(t.get_text()))
    plt.tight_layout()
    return _save(fig, out_dir, f"{task_name.lower()}_per_dataset_{metric.lower()}")


# ---------------------------------------------------------------------------
#  Experiment 2 (learning curve) + Experiment 3 (imbalance curve)
# ---------------------------------------------------------------------------

_SMOOTH_WINDOW_VARIANTS = {
    "less": 0.60,
    "standard": 1.00,
    "more": 1.50,
}


def _normalise_smooth_window(window: str) -> str:
    window = str(window).lower()
    if window not in _SMOOTH_WINDOW_VARIANTS:
        raise ValueError(
            f"Unknown smooth_window={window!r}; expected one of "
            f"{sorted(_SMOOTH_WINDOW_VARIANTS)}"
        )
    return window


def _sweep_window_size(n: int, smooth_window: str = "standard") -> int:
    """Rolling-window size for a sweep curve.

    ``standard`` is the historical ``len // 12`` rule. ``less`` averages fewer
    neighbouring points; ``more`` uses a wider window.
    """
    smooth_window = _normalise_smooth_window(smooth_window)
    base = max(3, int(n) // 12)
    return max(3, int(round(base * _SMOOTH_WINDOW_VARIANTS[smooth_window])))


def _smooth_suffix(prefix: str, smooth_window: str) -> str:
    smooth_window = _normalise_smooth_window(smooth_window)
    return prefix if smooth_window == "standard" else f"{prefix}_{smooth_window}"


def _sweep_moving_average(y, *, smooth_window: str = "standard") -> np.ndarray:
    """Centered rolling mean used by every pooled sweep moving-average view."""
    ser = pd.Series(np.asarray(y, dtype=float))
    if ser.empty:
        return np.asarray([], dtype=float)
    win = _sweep_window_size(len(ser), smooth_window)
    return ser.rolling(win, min_periods=1, center=True).mean().to_numpy()


def _add_raw_sweep_zoom(
    ax,
    series: Sequence[dict],
    *,
    zoom_limit: float,
    title: str,
) -> None:
    """Add a lower-right inset for raw pooled sweep curves up to ``zoom_limit``.

    The inset is opaque and its y-axis spans every point shown. Its bounds are
    chosen from lower-right candidates and expanded to the largest area that
    stays below the main curves in that part of the axes.
    """
    from matplotlib.ticker import MaxNLocator

    zoom_series = []
    for s in series:
        order = np.argsort(s["x"])
        x = np.asarray(s["x"], dtype=float)[order]
        y = np.asarray(s["y"], dtype=float)[order]
        in_window = x <= zoom_limit
        x = x[in_window]
        y = y[in_window]
        keep = np.isfinite(x) & np.isfinite(y)
        if np.any(keep):
            zoom_series.append({**s, "x": x[keep], "y": y[keep]})
    if not zoom_series:
        return

    x_all = np.concatenate([s["x"] for s in zoom_series])
    y_all = np.concatenate([s["y"] for s in zoom_series])
    x_lo, x_hi = float(np.nanmin(x_all)), float(np.nanmax(x_all))
    y_lo, y_hi = float(np.nanmin(y_all)), float(np.nanmax(y_all))
    if not np.isfinite(x_lo + x_hi + y_lo + y_hi):
        return

    ax.axvspan(x_lo, x_hi, color="0.45", alpha=0.10, zorder=0, lw=0)
    ax.axvline(x_hi, color="0.45", lw=1.0, ls=(0, (4, 3)), zorder=1)

    # Lower-right, but not so tall or wide that it hides saturated curves. Try
    # several lower-right boxes and keep the largest one whose top stays below
    # the lowest visible curve in its horizontal span.
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_span_ax = max(xlim[1] - xlim[0], 1e-9)
    y_span_ax = max(ylim[1] - ylim[0], 1e-9)

    def _lowest_curve_fraction(left_frac: float, right_frac: float) -> float:
        left = xlim[0] + left_frac * x_span_ax
        right = xlim[0] + right_frac * x_span_ax
        lowest = 1.0
        for s in series:
            x = np.asarray(s["x"], dtype=float)
            y = np.asarray(s["y"], dtype=float)
            keep = np.isfinite(x) & np.isfinite(y)
            x, y = x[keep], y[keep]
            if x.size == 0:
                continue
            order = np.argsort(x)
            x, y = x[order], y[order]
            samples = y[(x >= left) & (x <= right)]
            if x[0] <= left <= x[-1]:
                samples = np.append(samples, np.interp(left, x, y))
            if x[0] <= right <= x[-1]:
                samples = np.append(samples, np.interp(right, x, y))
            if samples.size:
                lowest = min(lowest, float((np.nanmin(samples) - ylim[0]) / y_span_ax))
        return lowest

    best_bounds = None
    best_area = -np.inf
    right_edge = 0.985
    for x0_candidate in np.linspace(0.50, 0.68, 13):
        width_candidate = right_edge - float(x0_candidate)
        if width_candidate < 0.28:
            continue
        curve_floor = _lowest_curve_fraction(max(0.0, x0_candidate - 0.015), right_edge)
        for y0_candidate in (0.075, 0.085, 0.095, 0.105, 0.115):
            top = min(0.92, curve_floor - 0.045, y0_candidate + 0.47)
            height_candidate = top - y0_candidate
            if height_candidate < 0.22:
                continue
            area = width_candidate * height_candidate
            if area > best_area:
                best_area = area
                best_bounds = (
                    float(x0_candidate),
                    float(y0_candidate),
                    float(width_candidate),
                    float(height_candidate),
                )

    x0, y0, width, height = best_bounds or (0.625, 0.115, 0.335, 0.24)
    zoom_ax = ax.inset_axes([x0, y0, width, height])

    for s in zoom_series:
        zoom_ax.plot(
            s["x"],
            s["y"],
            marker="o",
            ms=max(RUN_MS + 0.8, 2.1),
            lw=1.0,
            color=s["color"],
            solid_capstyle="round",
        )

    x_span = max(x_hi - x_lo, 1e-6)
    y_span = max(y_hi - y_lo, 1e-6)
    zoom_ax.set_xlim(x_lo - 0.03 * x_span, x_hi + 0.03 * x_span)
    # Use the full y-range of every shown point. No percentile trimming here:
    # if a point is in the zoom inset, it must be visible on the y-axis.
    zoom_ax.set_ylim(y_lo - 0.06 * y_span, y_hi + 0.06 * y_span)
    zoom_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    zoom_ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    zoom_ax.tick_params(axis="both", labelsize=TICK_FS - 1, length=2.5, pad=2)
    zoom_ax.grid(True, alpha=0.22)
    zoom_ax.set_facecolor("white")
    zoom_ax.patch.set_alpha(1.0)
    zoom_ax.set_title(title, fontsize=NOTE_FS, fontweight="bold", pad=4)
    for sp in zoom_ax.spines.values():
        sp.set_color("0.45")
        sp.set_linewidth(1.1)
    zoom_ax.set_zorder(6)


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
    zoom: bool = False,
    zoom_limit: Optional[float] = None,
    zoom_title: Optional[str] = None,
    logx: bool = False,
    smooth_window: str = "standard",
) -> Optional[Path]:
    """ONE line per METHOD: the metric averaged over all datasets at each
    sweep value. With ``relative=True`` each method's curve is divided by its
    OWN maximum, showing performance relative to that method's top (1.0 =
    the method's best point). ``logx=True`` puts the sweep axis on a log scale
    (so the small-data / extreme-imbalance end is readable)."""
    from matplotlib.ticker import MaxNLocator

    mean_col = _resolve_mean_column(df, metric)
    sub = df[df["sweep_axis"] == sweep_axis].dropna(subset=["sweep_value"])
    if sub.empty:
        logger.warning("No rows with sweep_axis=%s -- is this the right experiment?",
                       sweep_axis)
        return None
    grp = sub.groupby(["method", "sweep_value"])[mean_col].mean().reset_index()
    if logx:
        grp = grp[grp["sweep_value"] > 0]               # a log axis can't show <= 0
    # R² has a meaningful floor at 0 (predicting the mean); a sub-0 point (e.g.
    # lin. reg on a tiny training set) is shown AT 0 rather than dragging the
    # whole axis negative.
    clamp0 = str(metric).upper() == "R2"

    zoom_active = bool(zoom and not smooth and not relative and not logx and zoom_limit is not None)
    fig, ax = plt.subplots(figsize=figsize)
    colors = _sweep_colors(sorted(grp["method"].unique()))
    zoom_series = []
    for method, g in grp.groupby("method"):
        g = g.sort_values("sweep_value")
        y = g[mean_col]
        if clamp0:
            y = y.clip(lower=0.0)
        if relative:
            top = y.max()
            y = y / top if top else y
        x = g["sweep_value"].to_numpy(dtype=float)
        if smooth:
            # Moving average across sweep points (trend, not the raw wiggle).
            y_ma = _sweep_moving_average(y, smooth_window=smooth_window)
            ax.plot(x, y_ma, lw=CURVE_LW,
                    label=_display_name(method), color=colors[method])
        else:
            y_arr = y.to_numpy(dtype=float)
            ax.plot(x, y_arr, marker="o", ms=RUN_MS, lw=1.2,
                    label=_display_name(method), color=colors[method])
            zoom_series.append({"method": method, "color": colors[method], "x": x, "y": y_arr})
    if logx:
        ax.set_xscale("log")
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=14))
    if relative:
        ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
    if clamp0:
        ax.set_ylim(bottom=0.0)                          # R2 axis starts at 0
    if zoom_active and zoom_series:
        _add_raw_sweep_zoom(
            ax,
            zoom_series,
            zoom_limit=float(zoom_limit),
            title=zoom_title or f"x <= {zoom_limit:g}",
        )
    pm = _pretty_metric(metric)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel(f"{pm} (% of each method's own best)" if relative else pm,
                  fontsize=LABEL_FS, fontweight="bold")
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold",
                 pad=34 if zoom_active else None)
    if zoom_active:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.005),
            ncol=max(1, len(labels)),
            fontsize=LEGEND_FS,
            frameon=False,
            handlelength=1.8,
            borderaxespad=0.0,
            columnspacing=1.1,
        )
    else:
        # Method legend bottom-right (the data-rich corner is usually empty here).
        ax.legend(loc="lower right", fontsize=LEGEND_FS, framealpha=0.92)
    plt.tight_layout()
    smooth_suffix = _smooth_suffix("_smooth", smooth_window) if smooth else ""
    suffix = (("_relative" if relative else "") + smooth_suffix
              + ("_zoom" if zoom_active else "") + ("_logx" if logx else ""))
    return _save(fig, out_dir, plot_name + suffix)


def _sweep_curve_combined(
    df: pd.DataFrame,
    *,
    sweep_axis: str,
    metric: str,
    title: str,
    xlabel: str,
    figsize: Tuple[int, int],
    out_dir: Optional[Path],
    plot_name: str,
    smooth_window: str = "standard",
) -> Optional[Path]:
    """Pooled sweep curve overlaying EVERY raw point with its moving average.

    One colour per method: small transparent dots are the raw pooled sweep
    points (their transparency scales with the sweep density, so a 1 200-point
    sweep reads as a soft ribbon while a 300-point sweep keeps visible dots)
    and a solid, slightly thinner line is the centred moving average. This view
    intentionally has no zoom inset; the inset belongs to the raw curve.
    """
    from matplotlib.ticker import MaxNLocator

    mean_col = _resolve_mean_column(df, metric)
    sub = df[df["sweep_axis"] == sweep_axis].dropna(subset=["sweep_value"])
    if sub.empty:
        logger.warning("No rows with sweep_axis=%s -- is this the right experiment?",
                       sweep_axis)
        return None
    grp = sub.groupby(["method", "sweep_value"])[mean_col].mean().reset_index()
    if grp.empty:
        logger.warning("No grouped rows for sweep_axis=%s and metric=%s", sweep_axis, metric)
        return None

    clamp0 = str(metric).upper() == "R2"
    xvals = np.sort(grp["sweep_value"].dropna().astype(float).unique())
    if xvals.size == 0:
        return None

    # ---- pass 1: compute every method's series ------------------------------
    # The fixed map keeps a method's colour identical in every sweep view
    # (base / smooth / relative / per-dataset / combined).
    colors = _sweep_colors(sorted(grp["method"].unique()))
    series = []
    for method, g in grp.groupby("method"):
        g = g.sort_values("sweep_value")
        x = g["sweep_value"].to_numpy(dtype=float)
        y = g[mean_col].astype(float)
        if clamp0:
            y = y.clip(lower=0.0)
        y_raw = y.to_numpy(dtype=float)
        y_ma = _sweep_moving_average(y_raw, smooth_window=smooth_window)
        series.append({"method": method, "color": colors[method], "x": x,
                       "y_raw": y_raw, "y_ma": y_ma,
                       "final": float(y_ma[-1]) if y_ma.size else -np.inf})
    # Draw worst-first so the strongest trend line ends up on top; the legend
    # is reordered best-first below (the project-wide convention).
    series.sort(key=lambda s: s["final"])

    fig, ax = plt.subplots(figsize=figsize)

    handles = {}
    ma_lw = max(1.2, CURVE_LW - 0.35)
    for s in series:
        # Ink-balanced transparency: denser sweeps get fainter dots so every
        # figure carries roughly the same visual weight of raw data.
        s["alpha"] = float(np.clip(10.0 / np.sqrt(max(s["x"].size, 1)), 0.18, 0.40))
        ax.scatter(s["x"], s["y_raw"], s=7.0, alpha=s["alpha"], color=s["color"],
                   edgecolors="none", zorder=2)
        (ln,) = ax.plot(s["x"], s["y_ma"], lw=ma_lw, color=s["color"],
                        zorder=5, label=_display_name(s["method"]),
                        solid_capstyle="round")
        handles[s["method"]] = ln

    pm = _pretty_metric(metric)
    # Fewer, rounder ticks than the sibling curves: this figure also carries
    # the dot clouds and the inset, so the axis must stay quiet.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
    ax.set_xlabel(xlabel, fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel(pm, fontsize=LABEL_FS, fontweight="bold")
    # Extra title pad leaves room for the legend row between title and axes.
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=34)
    if clamp0:
        ax.set_ylim(bottom=0.0)
    # One frameless legend row above the axes -- structurally outside the data
    # -- ordered best-first (by each method's final moving-average value).
    best_first = sorted(series, key=lambda s: -s["final"])
    ax.legend(handles=[handles[s["method"]] for s in best_first],
              loc="lower left", bbox_to_anchor=(0.0, 1.005),
              ncol=max(1, len(series)), fontsize=LEGEND_FS,
              frameon=False, borderaxespad=0.0, handlelength=1.6,
              columnspacing=1.2)

    plt.tight_layout()
    return _save(fig, out_dir, plot_name + _smooth_suffix("_combined", smooth_window))


def learning_curve(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    figsize: Tuple[int, int] = FIG_WIDE,
    relative: bool = False,
    smooth: bool = False,
    zoom: bool = False,
    zoom_limit: float = 1000.0,
    logx: bool = False,
    smooth_window: str = "standard",
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Experiment 2: ``metric`` vs DATASET SIZE -- one line per method,
    averaged over every included dataset. The swept ``row_limit`` caps the
    dataset BEFORE the cross-validation split (train and test shrink
    together), so the x-axis is the total rows retained, not the training-set
    size. ``relative=True`` divides each method's curve by its own best value;
    ``smooth=True`` plots the moving average instead of the raw points;
    ``zoom=True`` adds a lower-right inset for rows ``<= zoom_limit``;
    ``logx=True`` puts the dataset-size axis on a log scale."""
    return _sweep_curve(
        df, sweep_axis="row_limit", metric=metric,
        title=f"{task_name} learning curve",
        xlabel="Dataset size (rows)", figsize=figsize, out_dir=out_dir,
        relative=relative, smooth=smooth, zoom=zoom, zoom_limit=zoom_limit,
        zoom_title="Low-data range", logx=logx, smooth_window=smooth_window,
        plot_name=f"{task_name.lower()}_learning_curve_{metric.lower()}",
    )


def learning_curve_moving_average_with_dots(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    figsize: Tuple[int, int] = FIG_WIDE,
    smooth_window: str = "standard",
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Experiment 2 combined view: transparent raw pooled points plus a
    centred moving-average line. The zoomed inset belongs to
    :func:`learning_curve` with ``zoom=True``."""
    return _sweep_curve_combined(
        df, sweep_axis="row_limit", metric=metric,
        title=f"{task_name} learning curve",
        xlabel="Dataset size (rows)", figsize=figsize, out_dir=out_dir,
        smooth_window=smooth_window,
        plot_name=f"{task_name.lower()}_learning_curve_{metric.lower()}",
    )


def imbalance_curve(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    figsize: Tuple[int, int] = FIG_WIDE,
    relative: bool = False,
    smooth: bool = False,
    zoom: bool = False,
    zoom_limit: float = 0.025,
    logx: bool = False,
    smooth_window: str = "standard",
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Experiment 3: ``metric`` vs minority proportion -- one line per method,
    averaged over every included dataset. ``relative=True`` divides each
    method's curve by its own best value; ``smooth=True`` plots the moving
    average instead of the raw points; ``zoom=True`` adds a lower-right inset
    for minority proportions ``<= zoom_limit``; ``logx=True`` puts the
    minority-proportion axis on a log scale."""
    return _sweep_curve(
        df, sweep_axis="minority_proportion", metric=metric,
        title=f"{task_name} imbalance-robustness curve",
        xlabel="Minority-class proportion", figsize=figsize, out_dir=out_dir,
        relative=relative, smooth=smooth, zoom=zoom, zoom_limit=zoom_limit,
        zoom_title="Severe-imbalance range", logx=logx, smooth_window=smooth_window,
        plot_name=f"{task_name.lower()}_imbalance_curve_{metric.lower()}",
    )


def imbalance_curve_moving_average_with_dots(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    figsize: Tuple[int, int] = FIG_WIDE,
    smooth_window: str = "standard",
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Experiment 3 combined view: transparent raw pooled points plus a
    centred moving-average line. The zoomed inset belongs to
    :func:`imbalance_curve` with ``zoom=True``."""
    return _sweep_curve_combined(
        df, sweep_axis="minority_proportion", metric=metric,
        title=f"{task_name} imbalance-robustness curve",
        xlabel="Minority-class proportion", figsize=figsize, out_dir=out_dir,
        smooth_window=smooth_window,
        plot_name=f"{task_name.lower()}_imbalance_curve_{metric.lower()}",
    )


def per_dataset_sweep_curves(
    df: pd.DataFrame,
    metric: str,
    *,
    sweep_axis: str,
    xlabel: str,
    task_name: str = "PD",
    figsize: Tuple[int, int] = FIG_WIDE,
    ms: float = RUN_MS,
    out_dir: Optional[Path] = None,
) -> List[Path]:
    """One RAW-points plot PER dataset: the metric vs the sweep value, one
    dotted line per method (NO smoothing / no cross-dataset averaging). Lets you
    see how each individual dataset behaves, not only the pooled mean. Returns
    the list of saved paths (one figure per dataset)."""
    from matplotlib.ticker import MaxNLocator

    mean_col = _resolve_mean_column(df, metric)
    sub = df[df["sweep_axis"] == sweep_axis].dropna(subset=["sweep_value"])
    if sub.empty:
        logger.warning("per_dataset_sweep_curves: no rows with sweep_axis=%s", sweep_axis)
        return []
    pm = _pretty_metric(metric)
    colors = _sweep_colors(sorted(sub["method"].unique()))
    paths: List[Path] = []
    curve_kind = {
        "row_limit": "learning curve",
        "minority_proportion": "imbalance-robustness curve",
    }.get(sweep_axis, "sweep curve")
    for dataset in sorted(sub["dataset"].unique()):
        dataset_label = str(dataset).split(".")[-1]
        first, sep, rest = dataset_label.partition("_")
        if sep and first.isdigit():
            dataset_label = rest
        dataset_label = dataset_label.replace("_", " ")
        dsub = sub[sub["dataset"] == dataset]
        grp = dsub.groupby(["method", "sweep_value"])[mean_col].mean().reset_index()
        fig, ax = plt.subplots(figsize=figsize)
        for method, g in grp.groupby("method"):
            g = g.sort_values("sweep_value")
            ax.plot(g["sweep_value"], g[mean_col], marker="o", ms=ms, lw=0.8,
                    label=_display_name(method), color=colors[method])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
        ax.set_xlabel(xlabel, fontsize=LABEL_FS, fontweight="bold")
        ax.set_ylabel(pm, fontsize=LABEL_FS, fontweight="bold")
        ax.set_title(f"{task_name} {curve_kind} - {dataset_label}",
                     fontsize=TITLE_FS, fontweight="bold")
        ax.legend(loc="lower right", fontsize=LEGEND_FS, framealpha=0.92)
        plt.tight_layout()
        stem = f"{task_name.lower()}_{sweep_axis}_{str(dataset).replace('.', '_')}_{metric.lower()}"
        p = _save(fig, out_dir, stem)
        if p:
            paths.append(p)
    return paths


def sweep_evolution_summary(
    df: pd.DataFrame,
    metric: str,
    *,
    sweep_axis: str,
    task_name: str = "PD",
    n_points: int = 16,
    higher_is_better: bool = True,
    log_spacing: bool = False,
) -> str:
    """Print the EVOLUTION of ``metric`` per method across the whole swept range
    (not just the endpoint), in TWO tables, both rows = methods (best average
    first), columns = a spread of ~``n_points`` sweep values:

    1. **absolute** ``metric`` (mean over all included datasets) at each point;
    2. **relative to each method's own best** (% of that method's best point) --
       so you can read each method's trajectory shape independently of its level
       (100% = the method's own peak over the sweep).
    """
    mean_col = _resolve_mean_column(df, metric)
    sub = df[df["sweep_axis"] == sweep_axis].dropna(subset=["sweep_value"])
    if sub.empty:
        print(f"(no {sweep_axis} sweep rows for {metric})")
        return ""
    grp = sub.groupby(["method", "sweep_value"])[mean_col].mean().reset_index()
    n_all = grp["sweep_value"].nunique()
    piv = grp.pivot(index="method", columns="sweep_value", values=mean_col)
    cols_all = sorted(piv.columns)
    if len(cols_all) > n_points:
        if log_spacing:
            # Denser at the LOW end (e.g. Experiment 2: small training-set sizes
            # matter most). Pick the column nearest each log-spaced target value.
            lo = max(float(cols_all[0]), 1e-9)
            targets = np.logspace(np.log10(lo), np.log10(float(cols_all[-1])), n_points)
            cols = sorted({min(cols_all, key=lambda c: abs(float(c) - t)) for t in targets})
        else:
            idx = np.linspace(0, len(cols_all) - 1, n_points).round().astype(int)
            cols = [cols_all[i] for i in sorted(set(idx))]
    else:
        cols = cols_all
    piv = piv[cols]
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=not higher_is_better).index]
    # Relative to each method's OWN best over the FULL sweep (peak = 100%).
    own_best = piv.max(axis=1) if higher_is_better else piv.min(axis=1)
    rel = 100.0 * piv.div(own_best, axis=0)
    pm = _pretty_metric(metric)
    rule = "=" * 118

    h1 = (f"{task_name} — {pm} evolution over {sweep_axis}  "
          f"(mean across {sub['dataset'].nunique()} datasets; "
          f"{len(cols)} of {n_all} sweep points; best average first)")
    h2 = (f"{task_name} — {pm} as % of each method's OWN best over the sweep "
          f"(100% = that method's peak)")
    text = (h1 + "\n" + rule + "\n" + piv.round(4).to_string()
            + "\n\n" + h2 + "\n" + rule + "\n" + rel.round(1).to_string())
    print(text)
    return text


# ---------------------------------------------------------------------------
#  Distribution, HPO-effect and cost plots
# ---------------------------------------------------------------------------

def metric_boxplots(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    higher_is_better: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Per-method box of the metric, one dot per dataset. The **box** spans the
    full set of per-fold scores -- so a method whose individual folds drop below
    the AUC = 0.5 baseline is visible in the whiskers -- while the **dots** are
    the per-dataset fold-means (one point per dataset). AUC starts the y axis at
    ``min(0.5, lowest fold shown)`` and R² at 0; the dot size scales with the
    dataset count so sparser (LGD) plots read clearly."""
    col = _resolve_mean_column(df, metric)
    # Box over EVERY per-fold observation (so sub-baseline folds show in the
    # whiskers); dots are the per-dataset fold-means (one clean point per dataset).
    obs = df[["dataset", "method", col]].dropna(subset=[col])
    per_ds = df.groupby(["dataset", "method"], as_index=False)[col].mean()
    # Order by MEAN (not median) so this box plot is sorted identically to the
    # matrix and bar of the same metric -- one consistent ordering everywhere.
    order = (obs.groupby("method")[col].mean()
             .sort_values(ascending=not higher_is_better).index)
    n_per_method = int(per_ds["method"].value_counts().max() or 1)
    fig, ax = plt.subplots(figsize=figsize or _vbar_figsize(len(order)))
    sns.boxplot(data=obs, x="method", y=col, order=order, color="#cfe8ff", ax=ax)
    sns.stripplot(data=per_ds, x="method", y=col, order=order,
                  color="#1f4e79", size=_strip_size(n_per_method), alpha=0.6, ax=ax)
    _style_method_axis(ax, connect=True)
    pm = _pretty_metric(metric)
    ax.set_ylabel(pm, fontsize=LABEL_FS, fontweight="bold")
    ax.set_xlabel("")
    ax.set_title(f"{task_name} {pm} distribution across datasets",
                 fontsize=TITLE_FS, fontweight="bold")
    # Truncate the y axis at the metric's natural floor (AUC 0.5 / R² 0), but
    # drop below it to min(0.5, lowest fold) when an individual fold falls under
    # the baseline -- so a box whose folds dip below 0.5 is never clipped.
    floor = _metric_floor(metric, obs[col].to_numpy(float))
    if floor is not None:
        base = _METRIC_BASELINE[str(metric).upper()]
        if floor < base:
            top = float(np.nanmax(obs[col].to_numpy(float)))
            ax.set_ylim(bottom=floor - 0.03 * max(top - floor, 1e-9))
        else:
            ax.set_ylim(bottom=floor)
    plt.tight_layout()
    return _save(fig, out_dir, f"{task_name.lower()}_box_{metric.lower()}")


def metric_bars(
    df: pd.DataFrame,
    metric: str,
    *,
    task_name: str = "PD",
    higher_is_better: bool = True,
    agg: str = "mean",
    figsize: Optional[Tuple[int, int]] = None,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Bar chart of the ``agg`` (mean/median) of ``metric`` per method, sorted
    best -> worst. The ``mean`` variant carries ± std error bars; when ``df``
    is the per-fold frame that std is the **fold-level** std (pooled over all
    dataset×fold observations of the method), not the across-dataset spread.
    AUC bars start the y axis at ``min(0.5, lowest bar)`` and R² bars at 0."""
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
        y_floor=_metric_floor(metric, vals.to_numpy(float)),
    )


def compute_time_bars(
    df: pd.DataFrame,
    *,
    task_name: str = "PD",
    figsize: Optional[Tuple[float, float]] = None,
    out_dir: Optional[Path] = None,
    hpo_methods: Optional[Sequence[str]] = None,
    n_trials: int = 1,
) -> Optional[Path]:
    """MEAN TOTAL compute time per method = train + predict, in seconds
    (log y; fastest = greenest, left). Uses train + predict because in-context
    models (TabPFN v1/v2/Real, Mitra) report ~0 train time -- their cost is at
    predict -- so a train-only bar would drop them off a log axis. No error bars
    (a ± band reads poorly on a log scale); the mean is printed above each bar.

    Pass ``hpo_methods`` + ``n_trials`` to fold the hyperparameter-search cost
    into the tunable methods' bars (their train time × ``n_trials``); the title
    then reads "train + predict + HPO"."""
    vals = _total_time(df, agg="mean", hpo_methods=hpo_methods, n_trials=n_trials)
    if vals is None:
        logger.warning("compute_time_bars: no train_time/predict_time columns")
        return None
    with_hpo = bool(hpo_methods) and n_trials > 1
    cost = "train + predict + HPO" if with_hpo else "train + predict"
    return _method_bar(
        vals, errs=None, logy=True, value_fmt="{:.1f}",
        title=f"{task_name}: mean compute time per method ({cost}, seconds)",
        ylabel="mean compute time per fold (s)",
        stem=f"{task_name.lower()}_bar_compute_time",
        out_dir=out_dir, figsize=figsize,
    )


def compute_time_boxplot(
    df: pd.DataFrame,
    *,
    task_name: str = "PD",
    figsize: Optional[Tuple[int, int]] = None,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Box + strip of TOTAL compute time (fit + predict) per method, **log y**,
    one point per (dataset, fold). Ordered fastest-mean first; foundation
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
    order = work.groupby("method")["t"].mean().sort_values().index
    n_per_method = int(len(work) / max(work["method"].nunique(), 1))
    fig, ax = plt.subplots(figsize=figsize or _vbar_figsize(len(order)))
    sns.boxplot(data=work, x="method", y="t", order=order, color="#cfe8ff", ax=ax)
    sns.stripplot(data=work, x="method", y="t", order=order,
                  color="#1f4e79", size=_strip_size(n_per_method), alpha=0.6, ax=ax)
    ax.set_yscale("log")
    _style_method_axis(ax, connect=True)
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
                 fontsize=TITLE_FS, fontweight="bold", pad=20)
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
    figsize: Optional[Tuple[int, int]] = None,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Box + strip of each method's per-dataset ranks by ``metric``."""
    ranks = _rank_pivot(df, metric, higher_is_better)
    long = ranks.melt(var_name="method", value_name="rank")
    order = list(ranks.columns)
    fig, ax = plt.subplots(figsize=figsize or _vbar_figsize(len(order)))
    # Same blue palette as metric_boxplots for one consistent layout.
    sns.boxplot(data=long, x="method", y="rank", order=order, color="#cfe8ff", ax=ax)
    sns.stripplot(data=long, x="method", y="rank", order=order,
                  color="#1f4e79", size=_strip_size(len(ranks)), alpha=0.6, ax=ax)
    ax.invert_yaxis()  # rank 1 (best) on top
    _style_method_axis(ax, connect=True)
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
    figsize: Optional[Tuple[int, int]] = None,
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
    fig, ax = plt.subplots(figsize=figsize or _hbar_figsize(len(per_method)))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in per_method["mean"]]
    ax.barh(per_method.index, per_method["mean"], xerr=per_method["std"].fillna(0),
            color=colors, edgecolor="black", linewidth=EDGE_LW, error_kw={"alpha": 0.4})
    ax.axvline(0, color="black", lw=1)
    ax.tick_params(labelsize=TICK_FS)
    _color_foundation_ticks(ax, axis="y")     # foundation names in crimson, like everywhere
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
    hpo_methods: Optional[Sequence[str]] = None,
    n_trials: int = 1,
) -> Optional[Path]:
    """Compute cost (train + predict, log x) vs mean metric -- the cost/quality
    frontier. Foundation models are red stars, everything else blue circles.
    To keep the figure legible, only the **notable** methods (all foundation
    models + the well-known baselines in ``NOTABLE_METHODS``) are labelled, and
    their names sit right next to their dot; the rest stay as unlabelled dots.

    Pass ``hpo_methods`` + ``n_trials`` to put the full train + predict + HPO
    cost on the x axis for the tunable methods (their train time × ``n_trials``)."""
    from matplotlib.lines import Line2D

    mean_col = _resolve_mean_column(df, metric)
    total = _total_time(df, agg="mean", hpo_methods=hpo_methods, n_trials=n_trials)
    if total is None:
        logger.warning("runtime_performance_scatter: no time columns")
        return None
    perf = df.groupby("method")[mean_col].mean()
    agg = (pd.DataFrame({"time": total, "perf": perf}).dropna()
           .reset_index().rename(columns={"index": "method"}))
    agg = agg.sort_values("perf", ascending=not higher_is_better).reset_index(drop=True)
    fnd = _foundation_methods()
    notable = fnd | NOTABLE_METHODS

    fig, ax = plt.subplots(figsize=figsize or FIG_WIDE)
    ax.set_xscale("log")
    for r in agg.itertuples():
        is_f = r.method in fnd
        ax.scatter(r.time, r.perf, s=PT_FND if is_f else PT_NORMAL,
                   marker="*" if is_f else "o",
                   color="crimson" if is_f else "#1f77b4",
                   edgecolor="black", linewidth=EDGE_LW, zorder=3)
    ax.margins(x=0.22, y=0.18)  # whitespace around the cloud for the labels

    # Label ONLY the notable methods, then de-overlap with a small force pass in
    # PIXEL space: each label is repelled from EVERY point (so a name never sits
    # on top of a dot/star) AND from every other label, then joined to its own
    # dot with a thin leader line. Robust to dense clusters such as tabicl_v2 and
    # tabpfn_v2.5 sitting at near-identical cost/quality.
    fig.canvas.draw()
    lab = agg[agg["method"].isin(notable)]
    lab = lab[lab["time"] > 0]
    if len(lab):
        items = [(r.time, r.perf, _display_name(r.method), r.method in fnd) for r in lab.itertuples()]
        fs = NOTE_FS
        all_px = ax.transData.transform(agg[["time", "perf"]].to_numpy(float))
        pos = ax.transData.transform(np.array([(t, p) for t, p, _, _ in items], float))
        pos[:, 1] += 24.0                       # start each label above its dot
        boxes = np.array([[max(len(t), 3) * 0.62 * fs, 1.45 * fs] for _, _, t, _ in items])
        bb = ax.get_window_extent()
        for _ in range(150):
            moved = False
            for i in range(len(items)):
                wi, hi = boxes[i]
                fx = fy = 0.0
                for jj in range(len(items)):              # repel from other labels
                    if jj == i:
                        continue
                    dx, dy = pos[i, 0] - pos[jj, 0], pos[i, 1] - pos[jj, 1]
                    ox = (wi + boxes[jj, 0]) / 2 + 4 - abs(dx)
                    oy = (hi + boxes[jj, 1]) / 2 + 3 - abs(dy)
                    if ox > 0 and oy > 0:
                        if oy <= ox:
                            fy += np.copysign(oy, dy if dy else 1.0)
                        else:
                            fx += np.copysign(ox, dx if dx else 1.0)
                for px, py in all_px:                     # repel from every point
                    dx, dy = pos[i, 0] - px, pos[i, 1] - py
                    ox = wi / 2 + 10 - abs(dx)
                    oy = hi / 2 + 10 - abs(dy)
                    if ox > 0 and oy > 0:
                        if oy <= ox:
                            fy += np.copysign(oy, dy if dy else 1.0)
                        else:
                            fx += np.copysign(ox, dx if dx else 1.0)
                if fx or fy:
                    pos[i, 0] = min(max(pos[i, 0] + 0.5 * fx, bb.x0 + wi / 2), bb.x1 - wi / 2)
                    pos[i, 1] = min(max(pos[i, 1] + 0.5 * fy, bb.y0 + hi / 2), bb.y1 - hi / 2)
                    moved = True
            if not moved:
                break
        inv = ax.transData.inverted()
        for (x, y, t, is_f), p in zip(items, pos):
            xt, yt = inv.transform(p)
            ax.annotate(
                t, xy=(x, y), xytext=(xt, yt), textcoords="data",
                ha="center", va="center", fontsize=fs,
                color="crimson" if is_f else "black",
                fontweight="bold" if is_f else "normal",
                arrowprops=dict(arrowstyle="-", color="0.55", lw=0.6),
            )
    ax.grid(True, which="both", alpha=0.25)
    pm = _pretty_metric(metric)
    _with_hpo = bool(hpo_methods) and n_trials > 1
    _cost = "train + predict + HPO" if _with_hpo else "train + predict"
    # x = MEAN compute time across folds/datasets; y = MEAN of the metric.
    # State both on the axes so it is unambiguous.
    ax.set_xlabel(f"mean compute time per fold — {_cost} (s, log scale)",
                  fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel(f"mean {pm}", fontsize=LABEL_FS, fontweight="bold")
    ax.set_title(f"{task_name}: mean {pm} vs mean compute cost ({_cost})",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.legend(handles=[
        Line2D([0], [0], marker="*", color="w", markerfacecolor="crimson",
               markeredgecolor="black", markersize=14, label="foundation model"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
               markeredgecolor="black", markersize=10, label="other"),
    ], loc="best", fontsize=LEGEND_FS)
    plt.tight_layout()
    return _save(fig, out_dir, f"{task_name.lower()}_cost_quality_{metric.lower()}")


# ---------------------------------------------------------------------------
#  Foundation-model vs baseline head-to-head (e.g. TabPFN v3 vs CatBoost)
# ---------------------------------------------------------------------------

def _declutter_labels(ax, xy, texts, *, fontsize: int = NOTE_FS, color: str = "0.30") -> None:
    """De-overlap point labels in PIXEL space: each label is repelled from every
    other label AND from every point, then joined to its own point by a thin
    leader line. Keeps crowded dataset names from intertwining. Call AFTER the
    axis limits/scale are final."""
    if not texts:
        return
    fig = ax.figure
    fig.canvas.draw()
    pts = ax.transData.transform(np.asarray(xy, dtype=float))
    pos = pts.astype(float).copy()
    pos[:, 1] += 16.0                                    # start each label above its point
    boxes = np.array([[max(len(t), 2) * 0.60 * fontsize, 1.5 * fontsize] for t in texts])
    bb = ax.get_window_extent()
    for _ in range(120):
        moved = False
        for i in range(len(texts)):
            wi, hi = boxes[i]
            fx = fy = 0.0
            for j in range(len(texts)):                  # repel from other labels
                if j == i:
                    continue
                dx, dy = pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]
                ox = (wi + boxes[j, 0]) / 2 + 3 - abs(dx)
                oy = (hi + boxes[j, 1]) / 2 + 2 - abs(dy)
                if ox > 0 and oy > 0:
                    if oy <= ox:
                        fy += np.copysign(oy, dy if dy else 1.0)
                    else:
                        fx += np.copysign(ox, dx if dx else 1.0)
            for px, py in pts:                           # repel from every point
                dx, dy = pos[i, 0] - px, pos[i, 1] - py
                ox = wi / 2 + 8 - abs(dx)
                oy = hi / 2 + 8 - abs(dy)
                if ox > 0 and oy > 0:
                    if oy <= ox:
                        fy += np.copysign(oy, dy if dy else 1.0)
                    else:
                        fx += np.copysign(ox, dx if dx else 1.0)
            if fx or fy:
                pos[i, 0] = min(max(pos[i, 0] + 0.5 * fx, bb.x0 + wi / 2), bb.x1 - wi / 2)
                pos[i, 1] = min(max(pos[i, 1] + 0.5 * fy, bb.y0 + hi / 2), bb.y1 - hi / 2)
                moved = True
        if not moved:
            break
    inv = ax.transData.inverted()
    for (x, y), p, t in zip(xy, pos, texts):
        xt, yt = inv.transform(p)
        ax.annotate(t, xy=(x, y), xytext=(xt, yt), textcoords="data",
                    ha="center", va="center", fontsize=fontsize, color=color,
                    arrowprops=dict(arrowstyle="-", color="0.7", lw=0.5))


def foundation_vs_baseline_size_trend(
    df: pd.DataFrame,
    *,
    metric: str,
    task: str,
    fnd_method: str = "tabpfn_v3",
    base_method: str = "catboost",
    relative: bool = True,
    higher_is_better: bool = True,
    task_name: str = "PD",
    figsize: Tuple[int, int] = FIG_WIDE,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Per-dataset gain of ``fnd_method`` over ``base_method`` vs **dataset size**,
    with an OLS trend line — does the foundation model's edge grow on SMALL data?

    y = relative gain (%) when ``relative`` else the raw difference. Relative
    gain is the metric improvement divided by the baseline metric value, matching
    the PD AUC plot. x = dataset rows (log). Dots are **green** where the
    foundation model wins and red otherwise; the zero (equal-performance) line is
    dashed and the regression slope (per decade of size) is annotated."""
    from src.data.dataset_inventory import row_counts

    fnd_disp, base_disp = _display_name(fnd_method), _display_name(base_method)
    mean_col = _resolve_mean_column(df, metric)
    piv = (df[df["method"].isin([fnd_method, base_method])]
           .groupby(["dataset", "method"])[mean_col].mean().unstack("method"))
    if fnd_method not in piv.columns or base_method not in piv.columns:
        logger.warning("size_trend: need both %s and %s in the data", fnd_method, base_method)
        return None
    piv = piv.dropna(subset=[fnd_method, base_method])
    sizes = row_counts(task)
    piv = piv[[d in sizes for d in piv.index]]
    if piv.empty:
        logger.warning("size_trend: no datasets with a known row count (need processed data)")
        return None

    x = np.array([sizes[d] for d in piv.index], dtype=float)
    fnd, base = piv[fnd_method].to_numpy(), piv[base_method].to_numpy()
    if relative:
        y = _relative_metric_gain(fnd, base, metric, higher_is_better=higher_is_better)
        ylabel = f"Relative {_pretty_metric(metric)} improvement (%)"
    else:
        y = fnd - base
        ylabel = f"{_pretty_metric(metric)} difference ({fnd_disp} − {base_disp})"
    win = (fnd >= base) if higher_is_better else (fnd <= base)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale("log")
    ax.axhline(0.0, color="0.4", lw=1.3, ls="--", zorder=1)
    ax.scatter(x[win], y[win], s=PT_NORMAL, c="#2ca02c", edgecolor="black", linewidth=EDGE_LW,
               zorder=3, label=f"{fnd_disp} better")
    ax.scatter(x[~win], y[~win], s=PT_NORMAL, c="#d62728", edgecolor="black", linewidth=EDGE_LW,
               zorder=3, label=f"{base_disp} better")
    mask = np.isfinite(y)
    if mask.sum() >= 2:
        lx = np.log10(x[mask])
        coef = np.polyfit(lx, y[mask], 1)
        xs = np.linspace(lx.min(), lx.max(), 100)
        ax.plot(10 ** xs, np.polyval(coef, xs), color="#1f4e79", lw=CURVE_LW, zorder=2,
                label=f"OLS trend ({coef[0]:+.2f}/decade of size)")
    ax.set_xlabel("dataset size (rows, log scale)", fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=LABEL_FS, fontweight="bold")
    ax.set_title(f"{task_name}: does {fnd_disp}'s edge over {base_disp} grow on small data?",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.legend(loc="best", fontsize=LEGEND_FS)
    ax.grid(True, which="both", alpha=0.25)
    _declutter_labels(ax, [(xi, yi) for xi, yi in zip(x, y) if np.isfinite(yi)],
                      [str(d).split(".")[-1] for d, yi in zip(piv.index, y) if np.isfinite(yi)])
    plt.tight_layout()
    return _save(fig, out_dir,
                 f"{task_name.lower()}_{fnd_method}_vs_{base_method}_sizetrend_{metric.lower()}")


def foundation_vs_baseline_scatter(
    df: pd.DataFrame,
    *,
    metric: str,
    fnd_method: str = "tabpfn_v3",
    base_method: str = "catboost",
    higher_is_better: bool = True,
    task_name: str = "PD",
    figsize: Tuple[int, int] = FIG_SQUARE,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Per-dataset head-to-head: y = ``fnd_method``'s metric, x = ``base_method``'s,
    with the **y = x diagonal** (equal performance). One point per dataset, green
    above the line (foundation model wins) and red below; labelled. Shows at a
    glance on which datasets the two methods diverge most."""
    fnd_disp, base_disp = _display_name(fnd_method), _display_name(base_method)
    mean_col = _resolve_mean_column(df, metric)
    piv = (df[df["method"].isin([fnd_method, base_method])]
           .groupby(["dataset", "method"])[mean_col].mean().unstack("method"))
    if fnd_method not in piv.columns or base_method not in piv.columns:
        logger.warning("scatter: need both %s and %s in the data", fnd_method, base_method)
        return None
    piv = piv.dropna(subset=[fnd_method, base_method])
    if piv.empty:
        return None
    xb, yf = piv[base_method].to_numpy(), piv[fnd_method].to_numpy()
    win = (yf >= xb) if higher_is_better else (yf <= xb)
    pm = _pretty_metric(metric)

    fig, ax = plt.subplots(figsize=figsize)
    lo, hi = float(min(xb.min(), yf.min())), float(max(xb.max(), yf.max()))
    pad = 0.04 * ((hi - lo) or 1.0)
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, color="0.4", lw=1.3, ls="--", zorder=1, label="equal performance (y = x)")
    ax.scatter(xb[win], yf[win], s=PT_NORMAL, c="#2ca02c", edgecolor="black", linewidth=EDGE_LW,
               zorder=3, label=f"{fnd_disp} better")
    ax.scatter(xb[~win], yf[~win], s=PT_NORMAL, c="#d62728", edgecolor="black", linewidth=EDGE_LW,
               zorder=3, label=f"{base_disp} better")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{base_disp} {pm}", fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel(f"{fnd_disp} {pm}", fontsize=LABEL_FS, fontweight="bold")
    ax.set_title(f"{task_name}: {fnd_disp} vs {base_disp} per dataset ({pm})",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.legend(loc="best", fontsize=LEGEND_FS)
    ax.grid(True, alpha=0.25)
    _declutter_labels(ax, list(zip(xb, yf)), [str(d).split(".")[-1] for d in piv.index])
    plt.tight_layout()
    return _save(fig, out_dir,
                 f"{task_name.lower()}_{fnd_method}_vs_{base_method}_scatter_{metric.lower()}")


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
    total = _total_time(df, agg="mean")
    if total is not None:
        table["time_s (mean)"] = total

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
    per method, plus mean compute time. Not a leaderboard (full list); pass
    ``sort_by=None`` for alphabetical order."""
    return _summary_text(df, task_name=task_name, metric_order=_PD_METRIC_ORDER,
                         sort_by=sort_by, higher_is_better=higher_is_better)


def lgd_summary_text(df: pd.DataFrame, *, task_name: str = "LGD",
                     sort_by: Optional[str] = "R2", higher_is_better: bool = True) -> str:
    """Copy-pasteable table of the mean of every **LGD (regression)** metric
    per method, plus mean compute time. Not a leaderboard (full list); pass
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


_METRIC_PRETTY = {"R2": "R²", "AP_NORMALIZED": "Normalized AP"}


def _pretty_metric(metric: str) -> str:
    """Display name for a metric (e.g. ``R2`` -> ``R²``, ``AP_normalized`` ->
    ``Normalized AP``). Column lookups and file names keep the raw key; only
    what the reader sees is prettified."""
    return _METRIC_PRETTY.get(str(metric).upper(), str(metric))


def _total_time(
    df: pd.DataFrame,
    agg: str = "mean",
    *,
    hpo_methods: Optional[Sequence[str]] = None,
    n_trials: int = 1,
) -> Optional[pd.Series]:
    """TOTAL compute time per method = train + predict, in seconds, aggregated
    by ``agg`` (default MEAN).

    In-context models (TabPFN v1/v2/Real, Mitra) report ``train_time = 0`` by
    design -- their cost lives in ``predict`` -- so plotting train-time alone
    drops them from a log axis. Train+predict is the fair, always-positive cost.

    When ``hpo_methods`` + ``n_trials`` are given, the train time of those
    methods is multiplied by ``n_trials`` so the bar reflects the full
    hyperparameter-search cost (train + predict + HPO) for the tunable methods;
    untuned / in-context methods are unchanged. This is computed on the fly for
    plotting only -- the stored per-fold ``train_time`` is the final refit and
    is never modified.
    """
    if "train_time" in df.columns:                 # per-fold frame
        train = df["train_time"].fillna(0.0)
        pt = df["predict_time"] if "predict_time" in df.columns else 0.0
    elif "train_time_mean" in df.columns:           # per-method frame
        train = df["train_time_mean"].fillna(0.0)
        pt = df["predict_time_mean"] if "predict_time_mean" in df.columns else 0.0
    else:
        return None
    predict = pt.fillna(0.0) if hasattr(pt, "fillna") else pt
    if hpo_methods and n_trials and n_trials > 1:
        hp = set(hpo_methods)
        factor = df["method"].map(lambda m: n_trials if m in hp else 1)
        train = train * factor
    total = train + predict
    return total.groupby(df["method"]).agg(agg).sort_values()


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
        # Crisp on disk (vector PDF + high-dpi raster elements like heatmaps).
        fig.savefig(saved, bbox_inches="tight", dpi=200)
        from src.utils.generate_captions import refresh_captions_for_saved_figure
        refresh_captions_for_saved_figure(saved)
    # Inline display: a LOW-dpi PNG keeps the committed notebook small enough
    # for GitHub (the on-disk PDF stays crisp).
    from src.visualizations.data_exploration import _display_inline
    _display_inline(fig, dpi=96)
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
    "learning_curve_moving_average_with_dots",
    "imbalance_curve",
    "imbalance_curve_moving_average_with_dots",
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
