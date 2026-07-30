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
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

import matplotlib
# Pin the non-interactive Agg backend so the module imports cleanly
# under SLURM (no ``$DISPLAY``), CI (no Tk), and Jupyter alike. Figures
# are rendered inline by encoding a PNG into memory and pushing it
# through ``IPython.display.Image`` -- see ``_save`` below.
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

# Standard, consistent figure labels (TALENT-free import: safe everywhere).
from src.methods.method_names import (
    display_name as _display_name,
    method_class as _method_class,
    method_class_color as _method_class_color,
    METHOD_CLASS_COLORS,
    METHOD_CLASS_ORDER,
)

# Dataset naming + ordering: the single source of truth (see the module docstring
# of src/data/dataset_registry.py). Proprietary datasets are anonymised here,
# and EVERY dataset axis / legend / header is ordered by ``_ds_sort_key``
# (public first, alphabetical; proprietary last) -- never by slug, which would
# reproduce the OLD paper numbering.
from src.data.dataset_names import (
    display_name as _ds_name,
    is_proprietary as _ds_proprietary,
    sort_datasets as _ds_sorted,
    sort_key as _ds_sort_key,
)

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
    sorted best-first, so position 0 is greenest and the last bar is reddest.

    NOTE: for many-method bar charts prefer :func:`method_class_colors` -- the
    value is already printed on every bar there, so a value gradient is
    redundant ink, while the model class is information the figure otherwise
    does not carry. This helper stays for non-method gradients.
    """
    return plt.get_cmap("RdYlGn")(np.linspace(0.92, 0.08, max(n, 1)))


def method_class_colors(methods) -> list:
    """Per-bar colours keyed on MODEL CLASS (foundation / boosting / deep /
    classical) -- the standard colour code for every many-method bar chart."""
    return [_method_class_color(m) for m in methods]


#: Title pad (points) that clears the class-legend strip sitting above the axes.
CLASS_LEGEND_TITLE_PAD = 26


def method_class_legend(ax, methods, *, loc: Optional[str] = None,
                        ncol: Optional[int] = None, above: bool = True,
                        **legend_kw):
    """Add a model-class legend covering exactly the classes present in
    ``methods``, in the canonical class order. Returns the legend (or None).

    By default the legend is one horizontal row **centred above the axes**: it
    is a colour key for the whole chart, not a data series, so it belongs
    outside the plotting area, and centring keeps it from crowding the tall
    left-hand bars (every chart here is sorted best-first). Callers that set the
    title afterwards should pass ``pad=CLASS_LEGEND_TITLE_PAD``; this helper
    re-pads an already-set title itself.

    Pass ``above=False`` for the old in-axes placement.
    """
    from matplotlib.patches import Patch

    present = {_method_class(m) for m in methods}
    handles = [
        Patch(facecolor=METHOD_CLASS_COLORS[c], edgecolor="black",
              linewidth=EDGE_LW, label=c)
        for c in METHOD_CLASS_ORDER if c in present
    ]
    if not handles:
        return None
    if above:
        loc = loc or "lower center"
        ncol = ncol or len(handles)          # a single horizontal strip
        legend_kw.setdefault("bbox_to_anchor", (0.5, 1.01))
        legend_kw.setdefault("borderaxespad", 0.0)
        legend_kw.setdefault("frameon", False)
    else:
        loc = loc or "best"
        ncol = ncol or 1
    legend = ax.legend(handles=handles, loc=loc, ncol=ncol,
                       fontsize=LEGEND_FS, framealpha=0.92, **legend_kw)
    if above and ax.get_title():
        # Lift a title that was set BEFORE the legend clear of the strip.
        title = ax.title
        ax.set_title(title.get_text(), fontsize=title.get_fontsize(),
                     fontweight=title.get_fontweight(),
                     pad=CLASS_LEGEND_TITLE_PAD)
    return legend


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


# ---------------------------------------------------------------------------
#  A4 geometry
# ---------------------------------------------------------------------------
#
# Every figure here ends up in the paper as
# ``\includegraphics[width=\textwidth]``, so LaTeX rescales it to the text-block
# width and the size a reader actually sees is
#
#     printed_pt = nominal_pt * (A4_TEXT_WIDTH_MM / figure_width_mm)
#
# A 19-inch-wide figure at a 160 mm text block is shrunk to 33%: a nominal 13 pt
# label prints at 4 pt. That is why the wide matrices were unreadable -- not
# because their fonts were small in absolute terms, but because the FIGURE was
# three times too wide for the page. The helpers below work in printed
# millimetres/points so the two can't drift apart again.

#: A4 (210 x 297 mm) with 25 mm margins -> a 160 x 247 mm text block.
A4_TEXT_WIDTH_MM = 160.0
A4_TEXT_HEIGHT_MM = 247.0
_MM_PER_IN = 25.4


#: Mean advance width of a digit as a fraction of the font size, for the
#: DejaVu Sans family matplotlib ships with. Measured, not guessed:
#: ``TextPath`` on "0123456789" gives 0.636 em per glyph.
_DIGIT_EM = 0.636


def _cell_annot_fontsize(cell_width_in: float, n_chars: int, *,
                         fill: float = 0.86, cap: float = 30.0) -> float:
    # ``fill`` is always passed explicitly by _matrix_geometry (reference fill
    # for the layout decision, display fill for the rendered size).
    """Font size at which ``n_chars`` digits fill ``fill`` of a cell's width.

    This is "the numbers should occupy as much of the cell as possible" stated
    geometrically: a run of digits is ``n_chars * _DIGIT_EM * size`` points
    wide, so the largest size leaving a ``1 - fill`` margin is
    ``fill * cell_pt / (n_chars * _DIGIT_EM)``.

    Note this is deliberately NOT corrected for the A4 downscale: the cell
    shrinks by the same factor as the glyphs, so the fill fraction is
    scale-invariant. Legibility on paper is decided by the FIGURE WIDTH, which
    is what :func:`_matrix_geometry` picks.
    """
    cell_pt = cell_width_in * 72.0
    size = fill * cell_pt / max(n_chars * _DIGIT_EM, 1e-9)
    return float(min(size, cap))


#: Fill fraction the WIDTH BUDGET is chosen at. Pinned so a cosmetic change to
#: :data:`_DISPLAY_FILL` cannot re-flow a figure between the 160 mm and 247 mm
#: budgets -- raising the fill from 0.78 to 0.86 did exactly that to the LGD
#: win/loss matrix before these were split apart.
_LAYOUT_REFERENCE_FILL = 0.78

#: Fill fraction the cell text is actually RENDERED at: how much of a cell's
#: width the digits occupy. Raise for chunkier numbers, lower for more air.
_DISPLAY_FILL = 0.86

#: Smallest printed font we accept for per-cell text in the paper.
_MIN_PRINTED_PT = 6.5

#: Fraction of the figure width the heatmap axes actually occupy. MEASURED on
#: rendered 33x33 figures after ``tight_layout``: 0.974 without a colorbar,
#: ~0.86 with one (the bar, its padding and its label take the rest).
_MATRIX_AXES_FRACTION = 0.95
_MATRIX_AXES_FRACTION_CBAR = 0.86

#: Minimum row pitch as a multiple of the tick font size, so horizontal row
#: labels can never collide. 1.0 would have them touching; 1.65 leaves the
#: normal single-spaced gap. Matters whenever rows outnumber what square cells
#: would give room for.
_MIN_ROW_PITCH = 1.65

#: Column pitch a 45-degree rotated tick label needs, per point of font size.
#: Two neighbouring labels are parallel baselines separated by
#: ``pitch * sin(45) = 0.707 * pitch``, and that has to clear one line height
#: (~1.2 * fs), so the theoretical floor is ``1.2 / 0.707 = 1.70``. Ignoring it
#: is what made 33 method names overlap into an unreadable smear at 160 mm.
#: 1.9 rather than 1.70 because at the exact floor the rendered clearance
#: measured -2 px (scratch check over all five matrix shapes); 1.9 leaves a
#: real gap.
_LABEL_PITCH_PER_PT = 1.9

#: Smallest font we accept for a rotated method label. Below this the column
#: pitch is too tight and the figure escalates to the wider page budget.
_MIN_LABEL_PT = 7.5


def _matrix_geometry(k: int, *, n_chars: int = 4, n_rows: Optional[int] = None,
                     cbar: bool = False, label_chars: int = 14) -> dict:
    """Size a PAIRWISE ``k`` x ``k`` method matrix for A4, in TRUE PRINTED units.

    Used by the all-learner win/loss and adjusted-p-value matrices only. The
    dataset x method heatmaps deliberately keep :func:`_heatmap_figsize` /
    :func:`_annot_fontsize` -- they read fine at that size and were never the
    problem.

    The figure is built at the width it will be printed at, so a nominal point
    size *is* the size a reader sees. That is the whole fix: these matrices
    were 520 mm wide, so ``\\includegraphics[width=\\textwidth]`` shrank them to
    31% and a nominal 16 pt title printed at 4.9 pt.

    Width is the A4 text block (160 mm) when the per-cell text still clears
    :data:`_MIN_PRINTED_PT` there, and the landscape/full-page budget (the text
    block's 247 mm HEIGHT, for a ``sidewaysfigure`` or a rotated page)
    otherwise. Note the honest limit: a 33-method pairwise matrix with four
    characters per cell has 4.7 mm cells at 160 mm and cannot carry legible
    text at any font size -- the density, not the layout, is binding. It gets
    the wide budget so a landscape placement reads at ~7 pt.

    Returns ``figsize``, ``annot_fs``, ``tick_fs``, ``title_fs``,
    ``target_width_mm`` and ``axes_fraction``.
    """
    n_rows = k if n_rows is None else n_rows
    frac = _MATRIX_AXES_FRACTION_CBAR if cbar else _MATRIX_AXES_FRACTION
    budgets = (A4_TEXT_WIDTH_MM, A4_TEXT_HEIGHT_MM)

    def _fonts(cell_in: float, fill: float):
        """(cell font, tick font) for a given cell width and fill fraction.

        Ticks carry method names, read left-to-right rather than scanned, so
        they can run a little larger than the in-cell digits -- but never larger
        than the COLUMN PITCH allows, or the 45-degree labels collide.
        """
        annot = _cell_annot_fontsize(cell_in, n_chars, fill=fill)
        tick = min(max(annot * 1.05, _MIN_PRINTED_PT), 11.0,
                   cell_in * 72.0 / _LABEL_PITCH_PER_PT)
        return annot, tick

    # WIDTH BUDGET: decided at a FIXED reference fill, deliberately not at the
    # display fill. Otherwise nudging the cell font by a few percent silently
    # re-flows a figure between the two budgets -- which happened once and moved
    # the LGD win/loss matrix from 247 mm to 160 mm as a side effect of making
    # its digits bigger. Layout and type size are separate decisions.
    for target_mm in budgets:
        width_in = target_mm / _MM_PER_IN
        cell_in = width_in * frac / max(k, 1)
        probe_annot, probe_tick = _fonts(cell_in, _LAYOUT_REFERENCE_FILL)
        if (probe_annot >= _MIN_PRINTED_PT and probe_tick >= _MIN_LABEL_PT) \
                or target_mm == budgets[-1]:
            break
    axes_w_in = width_in * frac
    # TYPE SIZE: at the display fill, still bounded by the column pitch so the
    # bigger font cannot reintroduce label collisions.
    annot_fs, tick_fs = _fonts(cell_in, _DISPLAY_FILL)
    tick_fs = round(tick_fs, 1)
    # Title larger again, and floored so a dense matrix still has a readable
    # title -- the specific complaint being fixed here.
    title_fs = round(min(max(annot_fs * 1.8, 10.0), 15.0), 1)
    # Row height: square cells, but never tighter than the ROW LABELS need.
    # For a wide, few-row matrix square cells collapse the rows and the labels
    # sit on top of each other, so the pitch floor wins. Non-square cells are
    # fine on a labelled matrix; illegible labels are not.
    row_h_in = max(axes_w_in / max(k, 1), _MIN_ROW_PITCH * tick_fs / 72.0)
    # Room for the 45-degree column labels, the axis titles and the title. Being
    # GENEROUS is free: ``_save``/``_finish`` write with ``bbox_inches="tight"``,
    # so surplus margin is cropped -- whereas underestimating makes
    # ``tight_layout`` squeeze the axes and collapse the cells.
    labels_in = label_chars * 0.55 * tick_fs / 72.0 / 1.414 + 0.28
    title_in = 2.6 * title_fs / 72.0
    height_in = row_h_in * n_rows + labels_in + title_in
    return {
        "figsize": (round(width_in, 2), round(height_in, 2)),
        "annot_fs": round(annot_fs, 1),
        "tick_fs": tick_fs,
        "title_fs": title_fs,
        "target_width_mm": target_mm,
        "axes_fraction": frac,
        "row_height_in": round(row_h_in, 4),
    }


#: Colorbar geometry for every matrix figure. seaborn/matplotlib default to
#: ``pad=0.05`` (5% of the axes width), which on a wide matrix is a centimetre
#: of dead space between the cells and the gradient bar. 0.012 tucks the bar
#: against the matrix; ``fraction`` keeps the bar itself slim.
MATRIX_CBAR_KW = {"pad": 0.012, "fraction": 0.028, "aspect": 34}


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


def _style_dataset_axis(ax, axis: str = "y") -> None:
    """Relabel a per-DATASET tick ``axis`` with registry display names.

    The method-axis counterpart (:func:`_color_foundation_ticks`) deliberately
    only touches the method axis, so dataset ticks (heatmap rows, per-dataset
    bar groups) would otherwise render the raw on-disk slug -- which both leaks
    proprietary dataset names and shows the old numbering. Call this on any axis
    whose ticks are dataset slugs. Rotation / alignment are preserved.
    """
    from matplotlib.ticker import FixedLocator

    get = ax.get_xticklabels if axis == "x" else ax.get_yticklabels
    setlab = ax.set_xticklabels if axis == "x" else ax.set_yticklabels
    cur = get()
    raws = [t.get_text() for t in cur]
    if not any(raws):                      # ticks not drawn yet -> nothing to do
        return
    rot = cur[0].get_rotation()
    ha = cur[0].get_horizontalalignment()
    locs = ax.get_xticks() if axis == "x" else ax.get_yticks()
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_locator(FixedLocator(list(locs)))
    setlab([_ds_name(r) for r in raws], rotation=rot, ha=ha)


def _order_datasets_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Reindex ``frame`` so its dataset index follows the paper order."""
    if frame.empty:
        return frame
    return frame.reindex(sorted(frame.index, key=_ds_sort_key))


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
        # Bars are coloured by MODEL CLASS (the value is already printed on
        # each bar, so a value gradient would be redundant ink).
        color=method_class_colors(names), edgecolor="black", linewidth=EDGE_LW,
        yerr=yerr, capsize=4 if yerr is not None else 0,
        # Navy error bars read clearly over every class colour.
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
    # Trim the dead space before the first bar and after the last one.
    # matplotlib's default 5% x-margin is 5% of the CATEGORY range, i.e. ~1.7
    # bar widths at each end for a 33-method chart -- which read as a broken
    # axis. Half a bar plus a hair is all that's needed for the bar edges.
    ax.set_xlim(-0.6, n - 0.4)
    _style_method_axis(ax)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS, fontweight="bold")
    # Model-class legend: a horizontal strip CENTRED above the axes, with the
    # title padded to clear it. Inside the axes it covered the tallest bars and
    # their value labels (both on the left after the best-first sort).
    method_class_legend(ax, names)
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold",
                 pad=CLASS_LEGEND_TITLE_PAD if ax.get_legend() is not None else 6)
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


# ---------------------------------------------------------------------------
#  Notebook-level method filters -- notebooks/CONFIG_NOTEBOOKS.yaml
# ---------------------------------------------------------------------------
# One flat exclusion list per task hides methods from EVERY analysis notebook
# except Experiment 0's pilot (enforced inside load_summary, the single door
# all notebooks load their data through); a champions inclusion list feeds the
# two champion-level statistical notebooks. Display-only: nothing here changes
# what the experiments run or what is stored on disk.

def _notebook_config_path() -> Path:
    from src.utils.paths import PROJECT_ROOT
    return PROJECT_ROOT / "notebooks" / "CONFIG_NOTEBOOKS.yaml"


@lru_cache(maxsize=1)
def _notebook_method_filters() -> dict:
    """Parsed ``notebooks/CONFIG_NOTEBOOKS.yaml`` (``{}`` when absent)."""
    try:
        with open(_notebook_config_path(), encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}


def excluded_methods(task: str) -> List[str]:
    """Methods the analysis notebooks hide for ``task`` (``exclude.<task>``)."""
    block = _notebook_method_filters().get("exclude") or {}
    return [str(m) for m in (block.get(task.lower()) or [])]


def champion_methods(task: str) -> List[str]:
    """The champion-level notebooks' inclusion list (``champions.<task>``),
    minus anything in the task's ``exclude`` list."""
    block = _notebook_method_filters().get("champions") or {}
    hidden = set(excluded_methods(task))
    return [str(m) for m in (block.get(task.lower()) or []) if str(m) not in hidden]


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
        # run_notebooks sets TABPFNCREDIT_SKIP_AUTO_SUMMARIZE after building the
        # CSVs itself ONCE up front: notebooks then skip this refresh, which both
        # removes redundant re-summarizing (six experiment1 notebooks = six
        # rebuilds) and makes parallel notebook execution race-free. The
        # missing-CSV fallback below stays active as a safety net.
        skip_refresh = os.environ.get(
            "TABPFNCREDIT_SKIP_AUTO_SUMMARIZE", ""
        ).lower() in {"1", "true", "yes"}
        if results_present and exp not in _SUMMARIZED_THIS_SESSION and not skip_refresh:
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
    # Notebook-level exclusions (notebooks/CONFIG_NOTEBOOKS.yaml): hide these
    # methods from every analysis notebook EXCEPT Experiment 0's pilot, which
    # must show everything that ran. The print keeps the omission visible in
    # the notebook output (and thus in results/All_Results.md).
    if exp != "experiment0" and "method" in df.columns:
        dropped = sorted(set(df["method"].astype(str)) & set(excluded_methods(task)))
        if dropped:
            df = df[~df["method"].astype(str).isin(dropped)]
            print(f"[CONFIG_NOTEBOOKS.yaml] excluded from this analysis: {', '.join(dropped)}")
    if df.empty:
        raise FileNotFoundError(f"No {task}/{hpo_mode} rows in {path}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
#  Out-of-fold prediction diagnostics (Experiment 1)
# ---------------------------------------------------------------------------

def _result_npz_path(
    results_root: Path,
    *,
    experiment: str,
    task: str,
    dataset: str,
    method: str,
    hpo_mode: str,
) -> Path:
    suffix = "__HPO" if str(hpo_mode).upper() == "HPO" else ""
    return (
        Path(results_root)
        / experiment.lower()
        / task.lower()
        / dataset
        / f"{method}{suffix}.npz"
    )


def _load_oof_prediction_pair(path: Path, task: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return pooled test-fold targets and predictions from one result archive."""
    task = task.lower()
    pred_key = "y_prob" if task == "pd" else "y_pred"
    y_parts: List[np.ndarray] = []
    pred_parts: List[np.ndarray] = []
    with np.load(path, allow_pickle=False) as archive:
        fold_ids = sorted(
            int(match.group(1))
            for key in archive.files
            if (match := re.fullmatch(r"fold_(\d+)_y_true", key))
        )
        for fold_id in fold_ids:
            y_key = f"fold_{fold_id}_y_true"
            p_key = f"fold_{fold_id}_{pred_key}"
            if p_key not in archive.files:
                continue
            y_true = np.asarray(archive[y_key], dtype=float).ravel()
            prediction = np.asarray(archive[p_key], dtype=float)
            if task == "pd" and prediction.ndim == 2:
                if prediction.shape[1] == 2:
                    prediction = prediction[:, 1]
                elif prediction.shape[1] == 1:
                    prediction = prediction[:, 0]
                else:
                    raise ValueError(f"Unsupported PD probability shape {prediction.shape} in {path}")
            prediction = prediction.ravel()
            if len(y_true) != len(prediction):
                raise ValueError(
                    f"Target/prediction length mismatch in {path}, fold {fold_id}: "
                    f"{len(y_true)} != {len(prediction)}"
                )
            finite = np.isfinite(y_true) & np.isfinite(prediction)
            y_parts.append(y_true[finite])
            pred_parts.append(prediction[finite])
    if not y_parts:
        raise ValueError(f"No usable out-of-fold predictions in {path}")
    return np.concatenate(y_parts), np.concatenate(pred_parts), len(y_parts)


def calibration_bias_table(
    df: pd.DataFrame,
    *,
    results_root: Path,
    task: str,
    experiment: str = "experiment1",
    hpo_mode: str = "HPO",
    methods: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Dataset-level OOF observed means, predicted means, and signed differences."""
    available_methods = list(dict.fromkeys(df["method"].astype(str)))
    if methods is not None:
        requested = set(methods)
        available_methods = [method for method in available_methods if method in requested]
    datasets = _ds_sorted(df["dataset"].astype(str).unique())   # paper order
    rows: List[Dict[str, object]] = []
    for method in available_methods:
        for dataset in datasets:
            path = _result_npz_path(
                results_root,
                experiment=experiment,
                task=task,
                dataset=dataset,
                method=method,
                hpo_mode=hpo_mode,
            )
            if not path.exists():
                logger.warning("Calibration summary: missing %s", path)
                continue
            try:
                y_true, prediction, n_folds = _load_oof_prediction_pair(path, task)
            except (OSError, ValueError) as exc:
                logger.warning("Calibration summary: skipping %s (%s)", path, exc)
                continue
            observed_mean = float(np.mean(y_true))
            predicted_mean = float(np.mean(prediction))
            rows.append(
                {
                    "task": task.lower(),
                    "dataset": dataset,
                    "method": method,
                    "observed_mean": observed_mean,
                    "predicted_mean": predicted_mean,
                    "calibration_bias": observed_mean - predicted_mean,
                    "n_observations": len(y_true),
                    "n_folds": n_folds,
                }
            )
    return pd.DataFrame(rows)


def _obs_pred_legend(ax, *, headroom: float = 0.18):
    """Neutral observed-vs-predicted key for the ONE panel it describes.

    Colour encodes the METHOD everywhere in the project, so observed vs
    predicted is carried by the fill (solid vs hatched) and explained in grey.
    The key is attached to the axes it applies to -- NOT to the figure -- because
    only the observed/predicted panel shows both series; a figure-level legend
    wrongly implied it also described the signed-difference panel. Extra top
    headroom is added first so the legend sits above the data, never on it.
    """
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor="0.72", edgecolor="black", linewidth=EDGE_LW,
              label="Observed"),
        Patch(facecolor="white", edgecolor="0.30", linewidth=EDGE_LW,
              hatch="///", label="Predicted"),
    ]
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + headroom * (high - low))
    return ax.legend(handles=handles, loc="upper center", ncol=2,
                     fontsize=LEGEND_FS, framealpha=0.92)


def selected_method_calibration_summary(
    calibration_df: pd.DataFrame,
    *,
    task: str,
    methods: Optional[Sequence[str]] = None,
    task_name: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Two-panel calibration DISTRIBUTION summary for the compared methods.

    Left: the dataset-level observed and predicted means, one point per
    dataset. Right: the distribution of the signed differences
    (observed - predicted), where a positive value means UNDERprediction.

    Only the distributions are shown: their macro means (the former top row of
    bars) carry no information the boxes do not already convey, and the exact
    numbers are printed by :func:`calibration_summary_text` into
    ``results/All_Results.md``.

    Styling follows the project conventions: one colour per METHOD (same map as
    the decile curve and the sweep figures), observed vs predicted by
    solid/hatched fill with a single neutral key at the figure bottom,
    45-degree display names with foundation models in crimson, y grid at 0.25.
    """
    task = task.lower()
    if calibration_df.empty:
        logger.warning("selected_method_calibration_summary: no calibration rows")
        return None
    if methods is None:
        baseline = "LogReg" if task == "pd" else "LinearRegression"
        methods = ("tabpfn_v3", "tabicl_v2", "catboost", baseline)
    present = [method for method in methods if method in set(calibration_df["method"])]
    work = calibration_df[calibration_df["method"].isin(present)].copy()
    if work.empty:
        logger.warning("selected_method_calibration_summary: no selected methods")
        return None
    task_name = task_name or task.upper()
    method_colors = _sweep_colors(present)
    x = np.arange(len(present), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=FIG_WIDE)

    # ---- left: dataset-level observed vs predicted distributions ----------
    ax = axes[0]
    dot_size = _strip_size(int(work.groupby("method")["dataset"].nunique().max() or 1))
    positions, distributions, specs = [], [], []
    for index, method in enumerate(present):
        method_data = work[work["method"].eq(method)]
        positions.extend([index - 0.18, index + 0.18])
        distributions.extend([
            100.0 * method_data["observed_mean"].to_numpy(dtype=float),
            100.0 * method_data["predicted_mean"].to_numpy(dtype=float),
        ])
        specs.extend([(method_colors[method], False),
                      (method_colors[method], True)])
    box = ax.boxplot(
        distributions, positions=positions, widths=0.28, patch_artist=True,
        showfliers=False, medianprops={"color": "black", "linewidth": 1.4},
    )
    for patch, (color, is_pred) in zip(box["boxes"], specs):
        if is_pred:
            # Hatch takes the EDGE colour in matplotlib, so an edge in the
            # method colour keeps "one colour = one method" true here too.
            patch.set_facecolor("white")
            patch.set_edgecolor(color)
            patch.set_linewidth(1.4)
            patch.set_hatch("///")
        else:
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
            patch.set_edgecolor("black")
    for position, values, (color, is_pred) in zip(positions, distributions, specs):
        offsets = (np.linspace(-0.045, 0.045, len(values))
                   if len(values) > 1 else np.zeros(len(values)))
        ax.scatter(
            position + offsets, values, s=dot_size ** 2,
            color="white" if is_pred else color,
            edgecolor=color if is_pred else "black",
            linewidth=1.0 if is_pred else 0.35, alpha=0.85, zorder=3,
        )
    ax.set_title("Distribution of dataset-level means",
                 fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel("Dataset-level mean (%)", fontsize=LABEL_FS, fontweight="bold")
    # Key on THIS panel only -- the signed-difference panel has no
    # observed/predicted split, so a shared figure legend would mislabel it.
    _obs_pred_legend(ax)

    # ---- right: distribution of signed differences ------------------------
    ax = axes[1]
    bias_distributions = [
        100.0 * work.loc[work["method"].eq(method),
                         "calibration_bias"].to_numpy(dtype=float)
        for method in present
    ]
    box = ax.boxplot(
        bias_distributions, positions=x, widths=0.5, patch_artist=True,
        showfliers=False, medianprops={"color": "black", "linewidth": 1.4},
    )
    for patch, method in zip(box["boxes"], present):
        patch.set_facecolor(method_colors[method])
        patch.set_alpha(0.45)
        patch.set_edgecolor("black")
    for position, values, method in zip(x, bias_distributions, present):
        offsets = (np.linspace(-0.09, 0.09, len(values))
                   if len(values) > 1 else np.zeros(len(values)))
        ax.scatter(
            position + offsets, values, s=dot_size ** 2,
            color=method_colors[method], edgecolor="black", linewidth=0.35,
            alpha=0.8, zorder=3,
        )
    ax.axhline(0.0, color="0.35", lw=1.2, ls="--")
    ax.set_title("Distribution of signed differences",
                 fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel("Observed - predicted (percentage points)",
                  fontsize=LABEL_FS, fontweight="bold")

    for ax in axes.ravel():
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        # Project method-axis convention: 45-degree right-aligned display
        # names, foundation models in crimson.
        ax.set_xticklabels(present, rotation=45, ha="right")
        ax.tick_params(labelsize=TICK_FS)
        _color_foundation_ticks(ax, axis="x")
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(
        f"{task_name}: observed versus predicted values across datasets",
        fontsize=TITLE_FS, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out_dir, f"{task}_selected_calibration_summary")

def _decile_curve_points(y_true: np.ndarray, prediction: np.ndarray,
                         n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rank the predictions into ``n_bins`` equal-count bins; return the
    per-bin (mean predicted, mean observed) pair. Rank-based splitting always
    yields exactly ``n_bins`` bins, tie-safe."""
    order = np.argsort(prediction, kind="stable")
    pred_means, obs_means = [], []
    for chunk in np.array_split(order, n_bins):
        pred_means.append(float(prediction[chunk].mean()))
        obs_means.append(float(y_true[chunk].mean()))
    return np.asarray(pred_means), np.asarray(obs_means)


def calibration_decile_curve(
    df: pd.DataFrame,
    *,
    results_root: Path,
    task: str,
    experiment: str = "experiment1",
    hpo_mode: str = "HPO",
    methods: Optional[Sequence[str]] = None,
    n_bins: int = 10,
    task_name: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Decile calibration curve, the credit-risk standard reliability view.

    Within EVERY dataset the pooled out-of-fold predictions are ranked into
    ``n_bins`` equal-count bins; each bin contributes its (mean predicted,
    mean observed) pair, and the k-th bin's pair is averaged across datasets
    (equal weight per dataset, matching the project's mean-over-datasets
    convention). One line per method in the shared sweep colours -- no markers,
    so ten bins per method stay readable; the dashed diagonal marks perfect
    calibration.
    """
    task = task.lower()
    if methods is None:
        baseline = "LogReg" if task == "pd" else "LinearRegression"
        methods = ("tabpfn_v3", "tabicl_v2", "catboost", baseline)
    datasets = _ds_sorted(df["dataset"].astype(str).unique())   # paper order
    present = [m for m in methods if m in set(df["method"].astype(str))]

    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for method in present:
        preds, obs = [], []
        for dataset in datasets:
            path = _result_npz_path(results_root, experiment=experiment,
                                    task=task, dataset=dataset, method=method,
                                    hpo_mode=hpo_mode)
            if not path.exists():
                logger.warning("Calibration deciles: missing %s", path)
                continue
            try:
                y_true, prediction, _ = _load_oof_prediction_pair(path, task)
            except (OSError, ValueError) as exc:
                logger.warning("Calibration deciles: skipping %s (%s)", path, exc)
                continue
            p, o = _decile_curve_points(y_true, prediction, n_bins)
            preds.append(p)
            obs.append(o)
        if preds:
            curves[method] = (np.mean(np.stack(preds), axis=0),
                              np.mean(np.stack(obs), axis=0))
    if not curves:
        logger.warning("calibration_decile_curve: no out-of-fold predictions found")
        return None

    task_name = task_name or task.upper()
    quantity = "default rate" if task == "pd" else "LGD"
    colors = _sweep_colors(sorted(curves))
    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    # Shared limits + the y = x diagonal, exactly like the head-to-head scatter.
    all_vals = np.concatenate([np.concatenate(v) for v in curves.values()]) * 100.0
    lo, hi = float(all_vals.min()), float(all_vals.max())
    pad = 0.04 * ((hi - lo) or 1.0)
    lim = (max(lo - pad, 0.0), hi + pad)
    ax.plot(lim, lim, color="0.4", lw=1.3, ls="--", zorder=1,
            label="perfect calibration (y = x)")
    for method in present:
        if method not in curves:
            continue
        p, o = curves[method]
        # Thin, marker-free lines: with four methods over ten deciles the
        # markers collided and hid where the curves actually separate.
        ax.plot(100.0 * p, 100.0 * o, lw=1.6, solid_capstyle="round",
                color=colors[method], zorder=3, label=_display_name(method))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Mean predicted {quantity} per decile (%)",
                  fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel(f"Mean observed {quantity} per decile (%)",
                  fontsize=LABEL_FS, fontweight="bold")
    ax.tick_params(labelsize=TICK_FS)
    ax.set_title(f"{task_name}: decile calibration curve (mean over datasets)",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.legend(loc="upper left", fontsize=LEGEND_FS, framealpha=0.92)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _save(fig, out_dir, f"{task}_calibration_deciles")


#: Datasets (rows) per per-dataset figure -- chosen to FILL one A4 page.
#:
#: These figures are 4 method columns wide and are included at 	extwidth, so
#: the page height they occupy is fixed by their aspect ratio:
#:     height_on_page = text_width x (fig_height / fig_width)
#: For A4 (210x297 mm) with 25 mm margins the text block is 160 x 247 mm; a
#: 3-line caption leaves ~229 mm of usable figure height, i.e. a target aspect
#: of at most 229/160 = 1.43.
#:
#: Measured aspects of the real figures (4 columns, tight bbox):
#:     rows   binned grid        density
#:        4   0.97 -> 156 mm     0.93 -> 148 mm
#:        5   1.21 -> 193 mm     1.16 -> 186 mm
#:        6   1.42 -> 227 mm     1.38 -> 221 mm   <-- ~97% of the page
#:        7   1.68 -> 268 mm     1.63 -> 261 mm   <-- OVERFLOWS the page
#: So 6 rows fills an A4 page almost exactly while 7 overflows it by ~17%
#: (7 rows had to be scaled down to fit, shrinking the tick labels with it).
#: 14 PD datasets therefore give pages of 6 + 6 + 2, and 7 LGD datasets 6 + 1.
_PER_DATASET_ROWS_PER_PAGE = 6

#: Per-panel size in inches. The height is the lever that sets the aspect above;
#: keep the two in step if you change either.
_PER_DATASET_PANEL_SIZE = (3.1, 2.8)

#: Ridge geometry for the binary (PD) prediction-density panels, in units of the
#: y distance between the two class rows. The ridge height USED to be a fraction
#: of the x range -- a length in prediction units applied to a distance in
#: outcome units -- which left ~half of every panel empty once predictions
#: spanned the full 0-100%. Expressing it against the separation makes the two
#: densities nearly meet: 0.92 leaves a visible 8%-of-separation gap so the grey
#: class baselines still read as separators.
_RIDGE_SEPARATION = 100.0
_RIDGE_HEIGHT_FRAC = 0.92
#: Breathing room below the lower baseline and above the upper ridge.
_RIDGE_MARGIN_FRAC = 0.05


#: A final page with fewer than this many rows is spread away instead of being
#: emitted. Set to 2, i.e. ONLY a single-row last page is rebalanced: that keeps
#: every earlier page at the A4-filling maximum (the point of the exercise),
#: while avoiding the one genuinely degenerate case -- a lone 4-panel strip on
#: its own sheet. A 2-row last page is just a normal half-page figure.
#: Raise this to 3 to trade page fill for uniform page heights
#: (14 datasets would then split 5 + 5 + 4 at ~81% fill instead of 6 + 6 + 2).
_MIN_ROWS_LAST_PAGE = 2


def _page_sizes(n_items: int, per_page: int) -> List[int]:
    """Rows per page: fill to ``per_page``, but never leave an orphan last page.

    ``per_page`` is the A4-filling maximum (see
    :data:`_PER_DATASET_ROWS_PER_PAGE`), so a page never EXCEEDS it. Greedy
    chunking can leave a straggler: 7 datasets would be 6 + 1, where a lone
    4-panel strip gets a sheet of its own. When the remainder is below
    :data:`_MIN_ROWS_LAST_PAGE` the same number of pages is used but the rows are
    spread as evenly as possible instead (7 -> 4 + 3). Otherwise pages are
    filled to ``per_page`` so each one uses as much of the A4 sheet as it can
    (14 -> 6 + 6 + 2: two pages at ~97% fill plus a half-page).
    """
    if n_items <= 0:
        return []
    n_pages = max(1, math.ceil(n_items / per_page))
    remainder = n_items % per_page
    if n_pages > 1 and remainder and remainder < min(_MIN_ROWS_LAST_PAGE, per_page):
        base, extra = divmod(n_items, n_pages)
        return [base + (1 if i < extra else 0) for i in range(n_pages)]
    return [min(per_page, n_items - i * per_page) for i in range(n_pages)]


def _paged(items: Sequence[str], per_page: Optional[int] = None):
    """Yield ``(page_index, n_pages, chunk)`` for a per-dataset figure.

    ``per_page`` defaults to :data:`_PER_DATASET_ROWS_PER_PAGE` at CALL time (a
    default argument would bind the constant at import time, so overriding the
    module constant would silently have no effect).
    """
    per_page = int(per_page or _PER_DATASET_ROWS_PER_PAGE)
    sizes = _page_sizes(len(items), per_page)
    start = 0
    for index, size in enumerate(sizes, start=1):
        yield index, len(sizes), list(items[start:start + size])
        start += size


def _oof_pairs_by_dataset(
    df: pd.DataFrame,
    *,
    results_root: Path,
    task: str,
    experiment: str,
    hpo_mode: str,
    present: Sequence[str],
    datasets: Sequence[str],
) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
    """``{(dataset, method): (prediction, actual)}`` from the stored npz files."""
    out: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    for dataset in datasets:
        for method in present:
            path = _result_npz_path(results_root, experiment=experiment, task=task,
                                    dataset=dataset, method=method, hpo_mode=hpo_mode)
            if not path.exists():
                continue
            try:
                y_true, prediction, _ = _load_oof_prediction_pair(path, task)
            except (OSError, ValueError) as exc:
                logger.warning("per-dataset calibration: skipping %s (%s)", path, exc)
                continue
            out[(dataset, method)] = (np.asarray(prediction, dtype=float),
                                      np.asarray(y_true, dtype=float))
    return out


def per_dataset_calibration_grid(
    df: pd.DataFrame,
    *,
    results_root: Path,
    task: str,
    experiment: str = "experiment1",
    hpo_mode: str = "HPO",
    methods: Optional[Sequence[str]] = None,
    task_name: Optional[str] = None,
    n_bins: int = 20,
    panel_size: Optional[Tuple[float, float]] = None,
    rows_per_page: Optional[int] = None,
    out_dir: Optional[Path] = None,
) -> List[Path]:
    """Binned predicted (x) versus actual (y) per DATASET and per METHOD.

    WHAT EACH PANEL SHOWS. One row = one dataset, one column = one method.
    Every dot is a *bin of loans*, not a loan: the dataset's pooled
    out-of-fold predictions are sorted and cut into ``n_bins`` (default 20)
    equal-count bins, and for each bin we plot

    * **x** = the mean PREDICTED value inside that bin, and
    * **y** = the mean ACTUAL outcome of the same loans (for PD, the observed
      default *rate* in the bin; for LGD, the mean realised LGD),

    with the dot AREA proportional to the bin's share of the dataset. Binning
    is what makes a binary PD outcome plottable on a y axis: individual actuals
    are only 0 or 1, whereas a bin of them has a frequency. So the 20 dots walk
    from the method's lowest-risk to its highest-risk predictions, and their
    vertical distance from the dashed ``y = x`` line is the calibration error
    in that risk band -- above the line the method UNDERpredicts, below it
    overpredicts.

    Axis limits are shared within a row so the diagonal is comparable across
    the methods being compared. Returns one path PER PAGE (7 datasets each).
    For the raw, unbinned distribution see
    :func:`per_dataset_prediction_density`.
    """
    task = task.lower()
    panel_size = panel_size or _PER_DATASET_PANEL_SIZE
    if methods is None:
        baseline = "LogReg" if task == "pd" else "LinearRegression"
        methods = ("tabpfn_v3", "tabicl_v2", "catboost", baseline)
    available = set(df["method"].astype(str))
    present = [m for m in methods if m in available]
    if not present:
        logger.warning("per_dataset_calibration_grid: none of %s present", methods)
        return []
    datasets = _ds_sorted(df["dataset"].astype(str).unique())
    pairs = _oof_pairs_by_dataset(df, results_root=results_root, task=task,
                                  experiment=experiment, hpo_mode=hpo_mode,
                                  present=present, datasets=datasets)
    rows_all = [d for d in datasets if any((d, m) in pairs for m in present)]
    if not rows_all:
        logger.warning("per_dataset_calibration_grid: no out-of-fold predictions found")
        return []

    task_name = task_name or task.upper()
    quantity = "default rate" if task == "pd" else "LGD"
    method_colors = _sweep_colors(present)
    # Bigger axis numbers: these panels are read at page scale, where the
    # shared TICK_FS was too small to make out.
    tick_fs = TICK_FS + 1
    paths: List[Path] = []

    for page, n_pages, rows in _paged(rows_all, rows_per_page):
        n_rows, n_cols = len(rows), len(present)
        fig, axes = plt.subplots(
            n_rows, n_cols, squeeze=False,
            figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows),
        )
        for r, dataset in enumerate(rows):
            row_vals = [v for m in present if (dataset, m) in pairs
                        for v in pairs[(dataset, m)]]
            flat = np.concatenate([np.asarray(v).ravel() for v in row_vals])
            lo, hi = float(np.nanmin(flat)), float(np.nanmax(flat))
            pad = 0.06 * ((hi - lo) or 1.0)
            lim = (100.0 * (lo - pad), 100.0 * (hi + pad))
            for c, method in enumerate(present):
                ax = axes[r][c]
                ax.set_axisbelow(True)
                ax.plot(lim, lim, color="0.4", lw=1.2, ls="--", zorder=1)
                if (dataset, method) in pairs:
                    prediction, actual = pairs[(dataset, method)]
                    order = np.argsort(prediction, kind="stable")
                    bins = [b for b in np.array_split(order, min(n_bins, max(len(order), 1)))
                            if len(b)]
                    px = np.array([prediction[b].mean() for b in bins])
                    py = np.array([actual[b].mean() for b in bins])
                    share = np.array([len(b) for b in bins], dtype=float)
                    share = share / share.sum() if share.sum() else share
                    # Smaller dots than before: at 20 bins the previous sizes
                    # overlapped and hid the shape of the curve.
                    ax.scatter(100.0 * px, 100.0 * py,
                               s=14.0 + 260.0 * share, color=method_colors[method],
                               edgecolor="black", linewidth=0.4, alpha=0.85, zorder=3)
                else:
                    ax.text(0.5, 0.5, "no result", transform=ax.transAxes,
                            ha="center", va="center", fontsize=NOTE_FS, color="0.5")
                ax.set_xlim(lim)
                ax.set_ylim(lim)
                ax.set_aspect("equal", adjustable="box")
                ax.tick_params(labelsize=tick_fs)
                ax.grid(True, alpha=0.25)
                if r == 0:
                    ax.set_title(_display_name(method), fontsize=LABEL_FS,
                                 fontweight="bold",
                                 color="crimson" if method in _foundation_methods() else "black")
                if c == 0:
                    ax.set_ylabel(f"{_ds_name(dataset)}\nobserved {quantity} (%)",
                                  fontsize=tick_fs, fontweight="bold")
                if r == n_rows - 1:
                    ax.set_xlabel(f"predicted {quantity} (%)", fontsize=tick_fs,
                                  fontweight="bold")
        page_note = f" -- page {page} of {n_pages}" if n_pages > 1 else ""
        fig.suptitle(
            f"{task_name}: predicted versus actual per dataset{page_note}\n"
            f"each dot = one of {n_bins} equal-count prediction bins "
            f"(area = share of loans); dashed = perfect calibration",
            fontsize=TITLE_FS, fontweight="bold",
        )
        # Reserve a FIXED 0.62 in for the two-line suptitle. A fraction of the
        # figure height (the old 0.45/n_rows) asked tight_layout for margins it
        # could not satisfy on short pages, which it warned about and ignored.
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.62 / fig.get_figheight()))
        stem = f"{task}_per_dataset_calibration"
        if n_pages > 1:
            stem = f"{stem}_p{page}"
        saved = _save(fig, out_dir, stem)
        if saved:
            paths.append(saved)
    return paths


def per_dataset_prediction_density(
    df: pd.DataFrame,
    *,
    results_root: Path,
    task: str,
    experiment: str = "experiment1",
    hpo_mode: str = "HPO",
    methods: Optional[Sequence[str]] = None,
    task_name: Optional[str] = None,
    panel_size: Optional[Tuple[float, float]] = None,
    rows_per_page: Optional[int] = None,
    out_dir: Optional[Path] = None,
) -> List[Path]:
    """Actual versus predicted as a smooth DENSITY, per dataset and method.

    The companion to :func:`per_dataset_calibration_grid`: instead of 20
    aggregated bins it shows where the individual loans actually sit, so the
    spread of the predictions is visible and not just the bin means.

    * **LGD (continuous actual)** -- a filled 2D kernel-density estimate of the
      raw ``(predicted, actual)`` cloud, with the dashed ``y = x``. Tight,
      diagonal-hugging density = well calibrated *and* sharp; a horizontal
      smear = predictions that barely vary with the realised loss.
    * **PD (binary actual)** -- a 2D density is degenerate when the actual is
      only 0 or 1, so each panel shows the two class-conditional densities of
      the predicted probability: the filled curve rising from the y = 0 row is
      the distribution of predictions for loans that did NOT default, the one
      from the y = 1 row is for loans that DID. The axes keep the same meaning
      (x = predicted, y = actual); separation between the two curves is
      discrimination, and their positions relative to the class base rates are
      calibration.

    Returns one path PER PAGE (7 datasets each).
    """
    task = task.lower()
    if panel_size is None:
        # The LGD panels are square (set_aspect("equal") below), so a slot wider
        # than it is tall can never be filled and the slack shows up as
        # horizontal gaps between columns. Trim the width toward the height.
        panel_size = (_PER_DATASET_PANEL_SIZE if task == "pd"
                      else (round(_PER_DATASET_PANEL_SIZE[1] * 1.05, 2),
                            _PER_DATASET_PANEL_SIZE[1]))
    if methods is None:
        baseline = "LogReg" if task == "pd" else "LinearRegression"
        methods = ("tabpfn_v3", "tabicl_v2", "catboost", baseline)
    available = set(df["method"].astype(str))
    present = [m for m in methods if m in available]
    if not present:
        logger.warning("per_dataset_prediction_density: none of %s present", methods)
        return []
    datasets = _ds_sorted(df["dataset"].astype(str).unique())
    pairs = _oof_pairs_by_dataset(df, results_root=results_root, task=task,
                                  experiment=experiment, hpo_mode=hpo_mode,
                                  present=present, datasets=datasets)
    rows_all = [d for d in datasets if any((d, m) in pairs for m in present)]
    if not rows_all:
        logger.warning("per_dataset_prediction_density: no out-of-fold predictions found")
        return []

    task_name = task_name or task.upper()
    quantity = "default rate" if task == "pd" else "LGD"
    is_binary = task == "pd"
    method_colors = _sweep_colors(present)
    tick_fs = TICK_FS + 1
    paths: List[Path] = []
    rng = np.random.default_rng(0)          # deterministic subsample for the KDE

    def _kde_1d(values: np.ndarray, grid: np.ndarray):
        """Gaussian KDE on ``grid``; None when scipy or the data can't support it."""
        values = values[np.isfinite(values)]
        if len(values) < 5 or np.allclose(values, values[0]):
            return None
        try:
            from scipy.stats import gaussian_kde
            return gaussian_kde(values)(grid)
        except Exception:  # noqa: BLE001 -- fall back to a histogram outline
            hist, edges = np.histogram(values, bins=40, range=(grid[0], grid[-1]),
                                       density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            return np.interp(grid, centers, hist)

    for page, n_pages, rows in _paged(rows_all, rows_per_page):
        n_rows, n_cols = len(rows), len(present)
        fig, axes = plt.subplots(
            n_rows, n_cols, squeeze=False,
            figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows),
        )
        for r, dataset in enumerate(rows):
            row_vals = [v for m in present if (dataset, m) in pairs
                        for v in pairs[(dataset, m)]]
            flat = np.concatenate([np.asarray(v).ravel() for v in row_vals])
            lo, hi = float(np.nanmin(flat)), float(np.nanmax(flat))
            pad = 0.06 * ((hi - lo) or 1.0)
            lim = (100.0 * (lo - pad), 100.0 * (hi + pad))
            for c, method in enumerate(present):
                ax = axes[r][c]
                ax.set_axisbelow(True)
                color = method_colors[method]
                if (dataset, method) not in pairs:
                    ax.text(0.5, 0.5, "no result", transform=ax.transAxes,
                            ha="center", va="center", fontsize=NOTE_FS, color="0.5")
                elif is_binary:
                    prediction, actual = pairs[(dataset, method)]
                    grid = np.linspace(lim[0], lim[1], 256)
                    # Two class-conditional densities, drawn as ridges rising
                    # from the y = 0 and y = 1 rows (y keeps meaning "actual").
                    for cls, ls in ((0.0, "-"), (1.0, "-")):
                        vals = 100.0 * prediction[actual == cls]
                        dens = _kde_1d(vals, grid)
                        if dens is None or not np.isfinite(dens).any():
                            continue
                        dens = dens / dens.max()          # per-class, so both are visible
                        base = _RIDGE_SEPARATION * cls
                        ax.fill_between(
                            grid, base,
                            base + _RIDGE_HEIGHT_FRAC * _RIDGE_SEPARATION * dens,
                            color=color, alpha=0.30 if cls == 0 else 0.55,
                            lw=1.2, edgecolor=color, ls=ls, zorder=2 + int(cls))
                        ax.axhline(base, color="0.6", lw=0.8, zorder=1)
                    ax.set_yticks([0.0, _RIDGE_SEPARATION])
                    # "0 (no default)" / "1 (default)" spelled out cost a wide
                    # left margin on all 24 panels; the axis label says
                    # "actual outcome", so the bare indicator is unambiguous.
                    ax.set_yticklabels(["0", "1"])
                    margin = _RIDGE_MARGIN_FRAC * _RIDGE_SEPARATION
                    ax.set_ylim(
                        -margin,
                        _RIDGE_SEPARATION * (1.0 + _RIDGE_HEIGHT_FRAC) + margin)
                    ax.set_xlim(lim)
                else:
                    prediction, actual = pairs[(dataset, method)]
                    # Diagonal drawn ON TOP of the filled density (which would
                    # otherwise bury it) but semi-transparent so it does not
                    # mask the contours it crosses.
                    ax.plot(lim, lim, color="0.25", lw=1.2, ls="--", zorder=6,
                            alpha=0.85)
                    px, py = 100.0 * prediction, 100.0 * actual
                    if len(px) > 20000:                    # keep the KDE tractable
                        pick = rng.choice(len(px), 20000, replace=False)
                        px, py = px[pick], py[pick]
                    drawn = False
                    try:
                        from scipy.stats import gaussian_kde
                        if len(px) >= 10 and px.std() > 0 and py.std() > 0:
                            gx, gy = np.mgrid[lim[0]:lim[1]:80j, lim[0]:lim[1]:80j]
                            dens = gaussian_kde(np.vstack([px, py]))(
                                np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
                            cmap = LinearSegmentedColormap.from_list(
                                f"_{method}_dens", ["#ffffff", color])
                            ax.contourf(gx, gy, dens, levels=12, cmap=cmap, zorder=2)
                            drawn = True
                    except Exception:  # noqa: BLE001
                        drawn = False
                    if not drawn:                          # hexbin fallback
                        cmap = LinearSegmentedColormap.from_list(
                            f"_{method}_dens", ["#ffffff", color])
                        ax.hexbin(px, py, gridsize=30, mincnt=1, cmap=cmap,
                                  linewidths=0.0, zorder=2)
                    ax.set_xlim(lim)
                    ax.set_ylim(lim)
                    ax.set_aspect("equal", adjustable="box")
                ax.tick_params(labelsize=tick_fs)
                # A ridge plot needs no vertical rules: the faint x-grid line
                # at zero read as a stray artefact exactly where the density
                # starts. The two grey class baselines carry the structure.
                if is_binary:
                    ax.grid(False)          # NB: grid(False, alpha=..) still enables it
                else:
                    ax.grid(True, alpha=0.25)
                if r == 0:
                    ax.set_title(_display_name(method), fontsize=LABEL_FS,
                                 fontweight="bold",
                                 color="crimson" if method in _foundation_methods() else "black")
                if c == 0:
                    label = (f"{_ds_name(dataset)}\nactual outcome" if is_binary
                             else f"{_ds_name(dataset)}\nactual {quantity} (%)")
                    ax.set_ylabel(label, fontsize=tick_fs, fontweight="bold")
                if r == n_rows - 1:
                    ax.set_xlabel(f"predicted {quantity} (%)", fontsize=tick_fs,
                                  fontweight="bold")
        page_note = f" -- page {page} of {n_pages}" if n_pages > 1 else ""
        detail = ("density of the predicted probability for defaulters vs non-defaulters"
                  if is_binary else
                  "2D kernel density of the raw prediction/actual pairs; dashed = perfect calibration")
        fig.suptitle(
            f"{task_name}: prediction density per dataset{page_note}\n{detail}",
            fontsize=TITLE_FS, fontweight="bold",
        )
        # Reserve a FIXED 0.62 in for the two-line suptitle. A fraction of the
        # figure height (the old 0.45/n_rows) asked tight_layout for margins it
        # could not satisfy on short pages, which it warned about and ignored.
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.62 / fig.get_figheight()))
        stem = f"{task}_per_dataset_prediction_density"
        if n_pages > 1:
            stem = f"{stem}_p{page}"
        saved = _save(fig, out_dir, stem)
        if saved:
            paths.append(saved)
    return paths
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
    # Rows in PAPER order (public alphabetical, then proprietary) -- NOT
    # .sort_index(), which sorts by slug and reproduces the old numbering.
    pivot = _order_datasets_index(
        df.pivot_table(index="dataset", columns="method",
                       values=mean_col, aggfunc="mean")
    )
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
                  # xticklabels="auto" (seaborn default) DROPS labels once space
                  # gets tight; every method must stay named. No visual change at
                  # this width -- it is a guard, not a restyle.
                  xticklabels=True, yticklabels=True,
                  cbar_kws={"label": pm, **MATRIX_CBAR_KW})
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
    _style_dataset_axis(ax, "y")           # rows are datasets -> registry labels
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
    sns.barplot(data=sub, x="dataset", y=mean_col, hue="method", ax=ax,
                order=_ds_sorted(sub["dataset"].astype(str).unique()))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    _style_dataset_axis(ax, "x")           # x ticks are datasets -> registry labels
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
    for dataset in _ds_sorted(sub["dataset"].unique()):
        dataset_label = _ds_name(dataset)   # registry label (anonymised if proprietary)
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
        # Filename keeps the on-disk slug so existing \includegraphics paths in
        # the LaTeX keep working (the slug never appears in the compiled PDF).
        stem = f"{task_name.lower()}_{sweep_axis}_{str(dataset).replace('.', '_')}_{metric.lower()}"
        # For a PROPRIETARY dataset also write a neutral, anonymised copy
        # (spec rule 3) so the figure can be \included without the real dataset
        # name appearing in the path. Written BEFORE _save, which closes the
        # figure. Same rendered content -- title/labels are already anonymised.
        if out_dir is not None and _ds_proprietary(dataset):
            neutral = (
                f"{task_name.lower()}_{sweep_axis}_"
                f"{_ds_name(dataset).lower()}_{metric.lower()}"
            )
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            neutral_path = Path(out_dir) / f"{neutral}.pdf"
            fig.savefig(neutral_path, bbox_inches="tight", dpi=200)
            paths.append(neutral_path)
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
    # Dataset rows in PAPER order (see _order_datasets_index). This is the
    # chokepoint for rank_heatmap / method_ranking_bars / rank_boxplots.
    pivot = _order_datasets_index(
        df.pivot_table(index="dataset", columns="method",
                       values=mean_col, aggfunc="mean")
    )
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
                cbar_kws={"label": f"rank by {pm} (1 = best)", **MATRIX_CBAR_KW},
                # see metric_heatmap: never let seaborn drop a method label
                xticklabels=True, yticklabels=True,
                linewidths=0.5, ax=ax)
    ax.set_title(f"{task_name} rank matrix by {pm} (1 = best; best mean rank left)",
                 fontsize=TITLE_FS, fontweight="bold", pad=20)
    ax.set_xlabel("Method", fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel("Dataset", fontsize=LABEL_FS, fontweight="bold")
    _style_method_axis(ax)
    _style_dataset_axis(ax, "y")           # rows are datasets -> registry labels
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
                      [_ds_name(d) for d, yi in zip(piv.index, y) if np.isfinite(yi)])
    plt.tight_layout()
    return _save(fig, out_dir,
                 f"{task_name.lower()}_{fnd_method}_vs_{base_method}_sizetrend_{metric.lower()}")


def foundation_vs_baseline_imbalance_trend(
    df: pd.DataFrame,
    *,
    metric: str,
    fnd_method: str = "tabpfn_v3",
    base_method: str = "catboost",
    relative: bool = True,
    higher_is_better: bool = True,
    task_name: str = "PD",
    figsize: Tuple[int, int] = FIG_WIDE,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Per-dataset method gain versus the processed PD minority proportion.

    A smaller minority proportion means stronger class imbalance. The y-axis
    follows the same relative-improvement definition and color convention as
    :func:`foundation_vs_baseline_size_trend`.
    """
    from src.data.dataset_inventory import minority_proportion

    fnd_disp, base_disp = _display_name(fnd_method), _display_name(base_method)
    mean_col = _resolve_mean_column(df, metric)
    pivot = (
        df[df["method"].isin([fnd_method, base_method])]
        .groupby(["dataset", "method"])[mean_col]
        .mean()
        .unstack("method")
    )
    if fnd_method not in pivot.columns or base_method not in pivot.columns:
        logger.warning(
            "imbalance_trend: need both %s and %s in the data", fnd_method, base_method
        )
        return None
    pivot = pivot.dropna(subset=[fnd_method, base_method])
    proportions = {dataset: minority_proportion("pd", dataset) for dataset in pivot.index}
    pivot = pivot[[proportions.get(dataset) is not None for dataset in pivot.index]]
    if pivot.empty:
        logger.warning("imbalance_trend: no datasets with a known minority proportion")
        return None

    x = 100.0 * np.array([proportions[dataset] for dataset in pivot.index], dtype=float)
    fnd = pivot[fnd_method].to_numpy(dtype=float)
    base = pivot[base_method].to_numpy(dtype=float)
    if relative:
        y = _relative_metric_gain(fnd, base, metric, higher_is_better=higher_is_better)
        ylabel = f"Relative {_pretty_metric(metric)} improvement (%)"
    else:
        y = (fnd - base) if higher_is_better else (base - fnd)
        ylabel = f"{_pretty_metric(metric)} improvement ({fnd_disp} over {base_disp})"
    win = (fnd >= base) if higher_is_better else (fnd <= base)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0.0, color="0.4", lw=1.3, ls="--", zorder=1)
    ax.scatter(
        x[win], y[win], s=PT_NORMAL, c="#2ca02c", edgecolor="black",
        linewidth=EDGE_LW, zorder=3, label=f"{fnd_disp} better",
    )
    ax.scatter(
        x[~win], y[~win], s=PT_NORMAL, c="#d62728", edgecolor="black",
        linewidth=EDGE_LW, zorder=3, label=f"{base_disp} better",
    )
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() >= 2:
        coefficient = np.polyfit(x[finite], y[finite], 1)
        x_line = np.linspace(float(np.min(x[finite])), float(np.max(x[finite])), 100)
        ax.plot(
            x_line, np.polyval(coefficient, x_line), color="#1f4e79",
            lw=CURVE_LW, zorder=2,
            label=f"OLS trend ({coefficient[0]:+.2f} per percentage point)",
        )
    ax.set_xlabel(
        "Minority-class proportion (%) - lower means more imbalanced",
        fontsize=LABEL_FS, fontweight="bold",
    )
    ax.set_ylabel(ylabel, fontsize=LABEL_FS, fontweight="bold")
    ax.set_title(
        f"{task_name}: does {fnd_disp}'s edge over {base_disp} grow as the minority class gets rarer?",
        fontsize=TITLE_FS, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=LEGEND_FS)
    ax.grid(True, alpha=0.25)
    ax.margins(y=0.12)
    _declutter_labels(
        ax,
        [(x_value, y_value) for x_value, y_value in zip(x, y) if np.isfinite(y_value)],
        [
            _ds_name(dataset)
            for dataset, y_value in zip(pivot.index, y)
            if np.isfinite(y_value)
        ],
    )
    fig.tight_layout()
    return _save(
        fig,
        out_dir,
        f"{task_name.lower()}_{fnd_method}_vs_{base_method}_imbalancetrend_{metric.lower()}",
    )


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
    _declutter_labels(ax, list(zip(xb, yf)), [_ds_name(d) for d in piv.index])
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


def rank_summary_text(df: pd.DataFrame, metric: str, *,
                      higher_is_better: bool = True, task_name: str = "PD") -> str:
    """Average per-dataset RANK of every method (1 = best) -- the numbers behind
    the rank heatmap / ranking bars / rank box plots."""
    pivot = _order_datasets_index(
        df.pivot_table(index="dataset", columns="method",
                       values=_resolve_mean_column(df, metric), aggfunc="mean")
    )
    ranks = pivot.rank(axis=1, ascending=not higher_is_better)
    table = pd.DataFrame({
        "mean_rank": ranks.mean(axis=0),
        "std_rank": ranks.std(axis=0, ddof=0),
        "best_rank": ranks.min(axis=0),
        "worst_rank": ranks.max(axis=0),
        "times_rank_1": (ranks == 1).sum(axis=0),
        "datasets": ranks.notna().sum(axis=0),
    }).sort_values("mean_rank")
    table.index = [_display_name(m) for m in table.index]
    header = (f"{task_name} - average rank per method on {_pretty_metric(metric)} "
              f"(1 = best on a dataset; ranked over {ranks.shape[0]} datasets)")
    text = header + "\n" + "-" * len(header) + "\n" + table.round(3).to_string()
    print(text)
    return text


def hpo_effect_text(df_both: pd.DataFrame, metric: str, *,
                    higher_is_better: bool = True, task_name: str = "PD") -> str:
    """Tuned vs untuned ``metric`` per method -- the numbers behind the
    HPO-effect bars.

    ``df_both`` is the summary loaded with ``hpo_mode=None`` so it carries BOTH
    modes. Methods that cannot be tuned appear with a delta of exactly zero
    (their HPO result is a copy of the NO_HPO run, by construction).
    """
    col = _resolve_mean_column(df_both, metric)
    if "hpo_mode" not in df_both.columns:
        text = f"{task_name} - HPO effect: no hpo_mode column in the summary."
        print(text)
        return text
    per = df_both.pivot_table(index="method", columns="hpo_mode", values=col, aggfunc="mean")
    if not {"HPO", "NO_HPO"} <= set(per.columns):
        text = f"{task_name} - HPO effect: need both HPO and NO_HPO rows."
        print(text)
        return text
    table = pd.DataFrame({
        "no_hpo": per["NO_HPO"],
        "hpo": per["HPO"],
        "delta (hpo - no_hpo)": per["HPO"] - per["NO_HPO"],
    })
    table = table.sort_values("delta (hpo - no_hpo)", ascending=not higher_is_better)
    table.index = [_display_name(m) for m in table.index]
    tuned = int((table["delta (hpo - no_hpo)"].abs() > 1e-12).sum())
    header = (f"{task_name} - effect of hyper-parameter tuning on "
              f"{_pretty_metric(metric)} ({tuned} of {len(table)} methods actually "
              f"change; the rest are NO_HPO-only, so their tuned result is a copy)")
    text = header + "\n" + "-" * min(len(header), 100) + "\n" + table.round(4).to_string()
    print(text)
    return text


def head_to_head_text(
    df: pd.DataFrame,
    metric: str,
    *,
    fnd_method: str = "tabpfn_v3",
    base_methods: Sequence[str] = ("catboost",),
    higher_is_better: bool = True,
    task_name: str = "PD",
) -> str:
    """Per-dataset head-to-head of ``fnd_method`` against each baseline -- the
    numbers behind the ``y = x`` scatters and the size/imbalance trend lines.

    Reports, per baseline: the win/loss/tie count over datasets, the mean and
    median difference, and the full per-dataset difference column (positive =
    the foundation model is better on that dataset, for a higher-is-better
    metric).
    """
    col = _resolve_mean_column(df, metric)
    pivot = _order_datasets_index(
        df.pivot_table(index="dataset", columns="method", values=col, aggfunc="mean")
    )
    parts: List[str] = []
    pm = _pretty_metric(metric)
    if fnd_method not in pivot.columns:
        text = f"{task_name} - head-to-head: {fnd_method} absent from the summary."
        print(text)
        return text
    fnd = pivot[fnd_method]
    for base in base_methods:
        if base not in pivot.columns:
            parts.append(f"{_display_name(base)}: absent from the summary.")
            continue
        diff = (fnd - pivot[base]) if higher_is_better else (pivot[base] - fnd)
        valid = diff.dropna()
        wins = int((valid > 0).sum())
        losses = int((valid < 0).sum())
        ties = int((valid == 0).sum())
        block = pd.DataFrame({
            f"{_display_name(fnd_method)}": fnd.reindex(valid.index),
            f"{_display_name(base)}": pivot[base].reindex(valid.index),
            "difference": valid,
        })
        block.index = [_ds_name(d) for d in block.index]
        head = (f"{_display_name(fnd_method)} vs {_display_name(base)} on {pm}: "
                f"{wins} wins / {losses} losses / {ties} ties over {len(valid)} datasets; "
                f"mean difference {valid.mean():+.4f}, median {valid.median():+.4f}")
        parts.append(head + "\n" + block.round(4).to_string())
    header = (f"{task_name} - head-to-head per dataset (positive difference = "
              f"{_display_name(fnd_method)} better)")
    text = header + "\n" + "-" * min(len(header), 100) + "\n" + "\n\n".join(parts)
    print(text)
    return text


def calibration_summary_text(
    bias_table: pd.DataFrame,
    *,
    task_name: str = "PD",
    df: Optional[pd.DataFrame] = None,
    results_root: Optional[Path] = None,
    task: Optional[str] = None,
    experiment: str = "experiment1",
    hpo_mode: str = "HPO",
    methods: Optional[Sequence[str]] = None,
    n_bins: int = 10,
) -> str:
    """Printed counterpart of the WHOLE calibration analysis.

    ``run_notebooks`` harvests stdout into ``results/All_Results.md``, so this
    is what makes the calibration figures' numbers available as text. It covers
    every calibration figure:

    1. **Per-method summary** -- mean observed, mean predicted, signed bias
       (observed - predicted; positive = UNDERprediction), its spread and mean
       absolute value, best-calibrated first. Backs the two-panel summary.
    2. **Per-dataset tables** -- observed mean, predicted mean and signed bias
       per (dataset, method). Backs the dataset-level distribution panel and
       the per-dataset predicted-vs-actual grids.
    3. **Decile calibration table** (only when ``df``/``results_root``/``task``
       are given) -- the mean predicted and mean observed value per decile,
       averaged over datasets: the numbers plotted by
       :func:`calibration_decile_curve`.

    ``bias_table`` is :func:`calibration_bias_table`'s output.
    """
    if bias_table is None or bias_table.empty:
        text = f"{task_name} calibration: no out-of-fold predictions found."
        print(text)
        return text

    grouped = bias_table.groupby("method")
    summary = pd.DataFrame({
        "observed_mean": grouped["observed_mean"].mean(),
        "predicted_mean": grouped["predicted_mean"].mean(),
        "bias_mean (obs-pred)": grouped["calibration_bias"].mean(),
        "bias_std": grouped["calibration_bias"].std(ddof=0),
        "abs_bias_mean": grouped["calibration_bias"].apply(lambda s: float(s.abs().mean())),
        "datasets": grouped["dataset"].nunique(),
    }).sort_values("abs_bias_mean")

    def _pivot(values: str) -> pd.DataFrame:
        """Per-dataset pivot in paper order, labelled with display names."""
        out = _order_datasets_index(
            bias_table.pivot_table(index="dataset", columns="method",
                                   values=values, aggfunc="mean")
        )
        out.index = [_ds_name(d) for d in out.index]
        return out

    header = (f"{task_name} — calibration: observed vs predicted target means per method "
              f"(equal weight per dataset; positive bias = underprediction)")
    parts = [
        header,
        "=" * max(len(header), 64),
        summary.round(4).to_string(),
        "",
        "Per-dataset OBSERVED mean:",
        _pivot("observed_mean").round(4).to_string(),
        "",
        "Per-dataset PREDICTED mean:",
        _pivot("predicted_mean").round(4).to_string(),
        "",
        "Per-dataset signed bias (observed - predicted):",
        _pivot("calibration_bias").round(4).to_string(),
    ]

    # ---- decile table: the numbers behind the decile calibration curve -----
    if df is not None and results_root is not None and task is not None:
        task_l = task.lower()
        if methods is None:
            baseline = "LogReg" if task_l == "pd" else "LinearRegression"
            methods = ("tabpfn_v3", "tabicl_v2", "catboost", baseline)
        available = set(df["method"].astype(str))
        present = [m for m in methods if m in available]
        datasets = _ds_sorted(df["dataset"].astype(str).unique())
        rows = {}
        for method in present:
            preds, obs = [], []
            for dataset in datasets:
                path = _result_npz_path(Path(results_root), experiment=experiment,
                                        task=task_l, dataset=dataset, method=method,
                                        hpo_mode=hpo_mode)
                if not path.exists():
                    continue
                try:
                    y_true, prediction, _ = _load_oof_prediction_pair(path, task_l)
                except (OSError, ValueError):
                    continue
                p, o = _decile_curve_points(y_true, prediction, n_bins)
                preds.append(p)
                obs.append(o)
            if preds:
                rows[(method, "predicted")] = np.mean(np.stack(preds), axis=0)
                rows[(method, "observed")] = np.mean(np.stack(obs), axis=0)
        if rows:
            decile = pd.DataFrame(rows, index=[f"D{i}" for i in range(1, n_bins + 1)]).T
            decile.index = pd.MultiIndex.from_tuples(decile.index,
                                                     names=["method", "series"])
            parts += [
                "",
                f"Decile calibration ({n_bins} equal-count bins, mean over datasets; "
                f"D1 = lowest predicted risk):",
                decile.round(4).to_string(),
            ]

    text = "\n".join(parts)
    print(text)
    return text

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
    "excluded_methods",
    "champion_methods",
    "calibration_bias_table",
    "selected_method_calibration_summary",
    "calibration_decile_curve",
    "per_dataset_calibration_grid",
    "per_dataset_prediction_density",
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
    "foundation_vs_baseline_size_trend",
    "foundation_vs_baseline_imbalance_trend",
    "foundation_vs_baseline_scatter",
    "pd_summary_text",
    "lgd_summary_text",
]
