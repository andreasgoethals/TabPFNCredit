"""Heavy-lifting helpers for ``notebooks/Data_Exploration.ipynb``.

Why this lives in src/
----------------------
The Data_Exploration notebook itself is intentionally thin: it imports
the helpers below and calls them. All loading, profiling, plotting, and
table building is in this module so it can be (a) unit-tested,
(b) imported from scripts, and (c) version-controlled without notebook
output churn.

Public surface
--------------
* :func:`list_raw_datasets`        -- enumerate raw CSVs per task
* :func:`load_raw_csv`             -- robust CSV reader (handles BOMs, delimiters)
* :func:`raw_dataset_summary_table` -- per-dataset overview (rows, cols, dtypes, file size)
* :func:`processed_dataset_summary_table` -- inverse for ``data/processed/``
* :func:`pd_target_balance_table`  -- class balance per PD dataset
* :func:`lgd_target_distribution_table` -- mean/median/skew/zero-rate per LGD dataset
* :func:`numeric_feature_stats`    -- per-feature stats (mean/std/min/max) per dataset
* :func:`plot_dataset_size_bar`    -- bar chart of dataset row counts
* :func:`plot_target_balance`      -- PD class balance plot
* :func:`plot_lgd_target_hists`    -- LGD target histograms (one per dataset)
* :func:`plot_correlation_heatmap` -- per-dataset feature correlation heatmap
* :func:`plot_pca_2d`              -- 2D PCA scatter coloured by target
* :func:`save_or_show`             -- consistent figure saving
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
# Pin the non-interactive Agg backend so the module is safe to import
# under SLURM (no ``$DISPLAY``), CI (no Tk), and Jupyter alike. We then
# render figures inline by manually encoding a PNG with ``fig.savefig``
# and pushing it through ``IPython.display.Image`` -- that works under
# every backend, including Agg, so the figure shows up in the notebook
# without depending on a GUI backend being available.
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
_PROC_DIR = _PROJECT_ROOT / "data" / "processed"


# ============================================================================
#  Discovery
# ============================================================================

def list_raw_datasets(task: str) -> List[Path]:
    """Return sorted list of raw CSV paths under ``data/raw/<task>/``."""
    out_dir = _RAW_DIR / task.lower()
    if not out_dir.exists():
        return []
    return sorted(out_dir.glob("*.csv"))


def list_processed_datasets(task: str) -> List[Path]:
    """Return sorted list of processed dataset directories under ``data/processed/<task>/``.

    Only directories that actually contain a ``y.npy`` (i.e. a successful
    preprocessing run) are returned. Empty stray directories created by an
    aborted preprocessing call are silently skipped so the notebook can't
    blow up on them.
    """
    out_dir = _PROC_DIR / task.lower()
    if not out_dir.exists():
        return []
    return sorted(
        p for p in out_dir.iterdir()
        if p.is_dir() and (p / "y.npy").exists()
    )


def load_raw_csv(path: Path, *, nrows: Optional[int] = None) -> pd.DataFrame:
    """Robust CSV loader that handles BOMs, mixed delimiters, and stray quoting.

    Tries the C engine first (fast; supports ``low_memory``); falls back
    to the python engine for files the C parser chokes on. ``low_memory``
    is only passed on the C engine -- pandas raises ``ValueError`` if you
    combine it with the python engine.
    """
    last_exc: Exception | None = None
    for sep in (",", ";", "\t"):
        for engine, extra in (("c", {"low_memory": False}), ("python", {})):
            try:
                return pd.read_csv(
                    path,
                    sep=sep,
                    nrows=nrows,
                    encoding="utf-8-sig",
                    engine=engine,
                    on_bad_lines="skip",
                    **extra,
                )
            except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
                last_exc = exc
                continue
    raise RuntimeError(f"Could not read {path}: {last_exc}")


def load_processed_dataset(task: str, dataset: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, dict]:
    """Load (N, C, y, info) from the cached TALENT-format dataset."""
    from src.data.preprocessing import preprocess_dataset
    return preprocess_dataset(task, dataset)


# ============================================================================
#  Summary tables
# ============================================================================

def raw_dataset_summary_table(task: str) -> pd.DataFrame:
    """One row per raw CSV: shape, file size, dtype counts, target column."""
    rows = []
    for path in list_raw_datasets(task):
        try:
            head = load_raw_csv(path, nrows=5)
            full_size = path.stat().st_size
            # Read row count without loading everything in memory
            with path.open("r", encoding="utf-8-sig", errors="replace") as f:
                n_rows = sum(1 for _ in f) - 1  # subtract header
            rows.append({
                "dataset": path.stem,
                "task": task.lower(),
                "rows": int(n_rows),
                "cols": int(head.shape[1]),
                "size_MB": round(full_size / 1024 / 1024, 2),
                "n_numeric_cols": int(head.select_dtypes(include="number").shape[1]),
                "n_object_cols": int(head.select_dtypes(include="object").shape[1]),
                "columns_preview": ", ".join(head.columns[:6].tolist()) + ("..." if head.shape[1] > 6 else ""),
            })
        except Exception as exc:
            logger.warning("Failed to summarise %s: %s", path.name, exc)
    return pd.DataFrame(rows)


def processed_dataset_summary_table(task: str) -> pd.DataFrame:
    """One row per processed dataset: shape, n_num/cat features, target stats."""
    rows = []
    for path in list_processed_datasets(task):
        # ``path.name`` (NOT ``path.stem``) -- dataset directories are
        # named like "0001.gmsc"; ``stem`` would strip ".gmsc" thinking
        # it's an extension and return "0001", which then fails to
        # round-trip back into load_processed_dataset.
        dataset = path.name
        try:
            N, C, y, info = load_processed_dataset(task, dataset)
            rows.append({
                "dataset": dataset,
                "task": task.lower(),
                "rows": int(len(y)),
                "n_num_features": int(info.get("n_num_features", 0)),
                "n_cat_features": int(info.get("n_cat_features", 0)),
                "n_total_features": int(info.get("n_num_features", 0) + info.get("n_cat_features", 0)),
                "target_mean": float(np.mean(y)),
                "target_min": float(np.min(y)),
                "target_max": float(np.max(y)),
            })
        except Exception as exc:
            logger.warning("Failed to summarise %s: %s", dataset, exc)
    return pd.DataFrame(rows)


def pd_target_balance_table() -> pd.DataFrame:
    """Class balance for every PD dataset (positive rate + #pos + #neg + imbalance ratio)."""
    rows = []
    for path in list_processed_datasets("pd"):
        dataset = path.name  # see note in processed_dataset_summary_table
        try:
            _, _, y, _ = load_processed_dataset("pd", dataset)
            y = np.asarray(y).astype(int).ravel()
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            n_total = n_pos + n_neg
            pos_rate = n_pos / n_total if n_total else float("nan")
            rows.append({
                "dataset": dataset,
                "n_total": n_total,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "positive_rate": round(pos_rate, 5),
                "imbalance_ratio": round(n_neg / n_pos, 2) if n_pos else float("inf"),
            })
        except Exception as exc:
            logger.warning("Failed to balance %s: %s", dataset, exc)
    df = pd.DataFrame(rows)
    return df.sort_values("positive_rate") if not df.empty else df


def lgd_target_distribution_table() -> pd.DataFrame:
    """Mean / std / skew / % zeros / quantiles for every LGD dataset's target."""
    from scipy import stats
    rows = []
    for path in list_processed_datasets("lgd"):
        dataset = path.name  # see note in processed_dataset_summary_table
        try:
            _, _, y, _ = load_processed_dataset("lgd", dataset)
            y = np.asarray(y).astype(float).ravel()
            zero_rate = float((y == 0).mean())
            one_rate = float((y == 1).mean())
            rows.append({
                "dataset": dataset,
                "n": int(len(y)),
                "mean": round(float(np.mean(y)), 4),
                "std": round(float(np.std(y)), 4),
                "median": round(float(np.median(y)), 4),
                "skew": round(float(stats.skew(y)), 4),
                "q05": round(float(np.quantile(y, 0.05)), 4),
                "q95": round(float(np.quantile(y, 0.95)), 4),
                "pct_zeros": round(100 * zero_rate, 2),
                "pct_ones": round(100 * one_rate, 2),
            })
        except Exception as exc:
            logger.warning("Failed to summarise LGD %s: %s", path.name, exc)
    return pd.DataFrame(rows)


def numeric_feature_stats(task: str, dataset: str) -> pd.DataFrame:
    """Per-feature descriptive stats for a processed dataset's numeric block.

    Returns an empty DataFrame on any error -- the calling cell in the
    notebook stays usable even if one dataset is misconfigured.
    """
    try:
        N, _, _, info = load_processed_dataset(task, dataset)
    except Exception as exc:
        logger.warning("numeric_feature_stats(%s, %s) failed: %s", task, dataset, exc)
        return pd.DataFrame()
    if N is None:
        return pd.DataFrame()
    cols = info.get("numerical_cols") or [f"num_{i}" for i in range(N.shape[1])]
    df = pd.DataFrame(N, columns=cols)
    desc = df.describe().T.reset_index().rename(columns={"index": "feature"})
    desc["dataset"] = dataset
    return desc


# ============================================================================
#  Plot helpers
# ============================================================================

def save_or_show(fig: plt.Figure, out_path: Optional[Path]) -> Optional[Path]:
    """Save ``fig`` to ``<out_path>.pdf`` (PDF only) AND display it inline.

    Behaviour
    ---------
    * **Save**: only PDF (no PNG). The extension on ``out_path`` is
      forced to ``.pdf`` so callers can pass a path with or without one.
    * **Display**: encodes a PNG via ``fig.savefig`` into an in-memory
      buffer and pushes it through ``IPython.display.Image`` so the
      figure renders right at the call site in a notebook. This works
      with the Agg backend (which we pin at import time so SLURM and
      tests do not need a GUI backend), unlike ``display(fig)`` which
      can fall back to printing the figure's text repr when Agg is in
      use. Outside Jupyter every branch is a harmless no-op.
    * **Cleanup**: closes the figure so a notebook calling this in a
      ``for`` loop over N datasets doesn't accumulate N open figures in
      memory.
    """
    pdf_path: Optional[Path] = None
    if out_path is not None:
        pdf_path = Path(out_path).with_suffix(".pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, bbox_inches="tight", dpi=200)   # crisp on disk
        from src.utils.generate_captions import refresh_captions_for_saved_figure
        refresh_captions_for_saved_figure(pdf_path)
    _display_inline(fig, dpi=96)                              # small inline PNG
    plt.close(fig)
    return pdf_path


def _display_inline(fig: plt.Figure, *, dpi: int = 96) -> None:
    """Encode ``fig`` as PNG bytes and push through IPython.display.Image.

    Works with the Agg backend (no GUI required) and is a harmless no-op
    when IPython is not installed. Kept separate so other modules (e.g.
    ``experiment_plots`` and ``calibration_plots``) can re-use the same
    code path. ``dpi`` is kept modest so the PNG embedded in the committed
    notebook stays small (the on-disk PDF is saved crisp separately).
    """
    try:
        from IPython.display import Image, display
    except ImportError:
        return
    import io
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    except Exception:
        return
    buf.seek(0)
    display(Image(data=buf.getvalue(), format="png"))


def plot_dataset_size_bar(
    task: str,
    *,
    out_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Optional[Path]:
    """Horizontal bar chart of row counts per dataset."""
    df = processed_dataset_summary_table(task)
    if df.empty or "rows" not in df.columns:
        logger.warning("plot_dataset_size_bar(%s): no datasets to plot", task)
        return None
    df = df.sort_values("rows", ascending=True)
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(df["dataset"], df["rows"], color=sns.color_palette("viridis", len(df)))
    for bar, value in zip(bars, df["rows"]):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{value:,}", va="center", fontsize=9)
    ax.set_xlabel("Number of rows")
    ax.set_ylabel("Dataset")
    ax.set_title(f"{task.upper()} dataset sizes (processed)")
    ax.set_xscale("log")
    plt.tight_layout()
    return save_or_show(fig, out_path)


def plot_target_balance(
    *,
    out_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Optional[Path]:
    """PD class-balance bar chart (positive rate per dataset)."""
    df = pd_target_balance_table()
    if df.empty:
        logger.warning("plot_target_balance: no PD datasets to plot")
        return None
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(df["dataset"], df["positive_rate"] * 100,
                   color=sns.color_palette("rocket", len(df)))
    for bar, value in zip(bars, df["positive_rate"] * 100):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}%", va="center", fontsize=9)
    ax.set_xlabel("Positive-class rate (%)")
    ax.set_ylabel("Dataset")
    ax.set_title("PD: positive-class rate per dataset (sorted ascending)")
    plt.tight_layout()
    return save_or_show(fig, out_path)


def plot_lgd_target_hists(
    *,
    out_path: Optional[Path] = None,
    ncols: int = 2,
    figsize_per_panel: Tuple[float, float] = (7.5, 4.0),
    bins: int = 40,
) -> Optional[Path]:
    """Grid of LGD target distributions, **two panels per row**.

    Each panel shows the target histogram (as a density), a red kernel-density
    trend line over it, and vertical lines for the **mean** (solid orange) and
    **median** (dashed green) -- distinct colour AND line style so the two are
    still distinguishable in greyscale. The panel title carries the dataset
    name (bold, enlarged), its row count, and the numeric mean and median. The
    colour key is a SINGLE figure-level legend at the top, so it can never sit
    in front of a panel's mean/median lines. Datasets that fail to load are
    skipped (logged at WARNING) so one bad dataset can't blow up the grid.
    """
    from matplotlib.lines import Line2D
    from scipy.stats import gaussian_kde

    # Okabe-Ito colour-blind-safe hues; line style also distinguishes them.
    C_DENSITY, C_MEAN, C_MEDIAN = "red", "#E69F00", "#009E73"  # red / orange / green

    candidates = [p.name for p in list_processed_datasets("lgd")]
    if not candidates:
        return None
    # Pre-load all targets so we know which actually succeeded before
    # building the subplot grid.
    pairs: List[Tuple[str, np.ndarray]] = []
    for dataset in candidates:
        try:
            _, _, y, _ = load_processed_dataset("lgd", dataset)
            pairs.append((dataset, np.asarray(y, dtype=float).ravel()))
        except Exception as exc:
            logger.warning("LGD histogram: skipping %s (%s)", dataset, exc)
    if not pairs:
        return None
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    for ax, (dataset, y) in zip(axes.ravel(), pairs):
        y = y[np.isfinite(y)]
        mean, median = float(np.mean(y)), float(np.median(y))
        ax.hist(y, bins=bins, density=True, color="steelblue",
                edgecolor="white", alpha=0.85)
        # Red density (KDE) trend line. Skip gracefully when the target is
        # degenerate (single value) -- gaussian_kde needs non-zero variance.
        if y.size > 1 and np.ptp(y) > 0:
            try:
                kde = gaussian_kde(y)
                xs = np.linspace(float(y.min()), float(y.max()), 200)
                ax.plot(xs, kde(xs), color=C_DENSITY, lw=2)
            except Exception:  # pragma: no cover -- numerical edge case
                pass
        ax.axvline(mean, color=C_MEAN, lw=2.0)              # orange, solid
        ax.axvline(median, color=C_MEDIAN, lw=2.0, ls="--")  # green, dashed
        # Two-line title: name + n, then the numeric mean / median.
        ax.set_title(
            f"{dataset}   (n = {len(y):,})\nmean = {mean:.3f}   median = {median:.3f}",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("LGD target")
        ax.set_ylabel("density")
    for ax in axes.ravel()[len(pairs):]:
        ax.set_visible(False)
    # ONE shared colour key above all panels -- never overlaps the lines.
    handles = [
        Line2D([0], [0], color=C_DENSITY, lw=2, label="density (KDE)"),
        Line2D([0], [0], color=C_MEAN, lw=2, label="mean"),
        Line2D([0], [0], color=C_MEDIAN, lw=2, ls="--", label="median"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True,
               fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_or_show(fig, out_path)


def plot_correlation_heatmap(
    task: str,
    dataset: str,
    *,
    out_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 9),
) -> Optional[Path]:
    """Feature-correlation heatmap for one processed dataset's numerical block.

    Silently returns ``None`` if the dataset can't be loaded.
    """
    try:
        N, _, _, info = load_processed_dataset(task, dataset)
    except Exception as exc:
        logger.warning("plot_correlation_heatmap(%s, %s) failed: %s", task, dataset, exc)
        return None
    if N is None or N.shape[1] < 2:
        return None
    cols = info.get("numerical_cols") or [f"num_{i}" for i in range(N.shape[1])]
    df = pd.DataFrame(N, columns=cols)
    corr = df.corr().fillna(0)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, vmin=-1, vmax=1, cmap="vlag", center=0, square=True,
                cbar_kws={"label": "Pearson r"}, ax=ax)
    ax.set_title(f"{dataset} -- numerical feature correlations")
    plt.tight_layout()
    return save_or_show(fig, out_path)


def plot_pca_2d(
    task: str,
    dataset: str,
    *,
    out_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (7, 6),
    sample: Optional[int] = 5000,
) -> Optional[Path]:
    """2D PCA scatter, coloured by target. Silently skips on load error."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    try:
        N, _, y, _ = load_processed_dataset(task, dataset)
    except Exception as exc:
        logger.warning("plot_pca_2d(%s, %s) failed: %s", task, dataset, exc)
        return None
    if N is None or N.shape[1] < 2:
        return None
    y = np.asarray(y).ravel()
    # PCA (sklearn >= 1.4) refuses to operate on NaN. Drop any row that
    # carries a NaN in the numerical block before transforming. This is a
    # quick descriptive view -- not feeding a model -- so dropping is fine.
    N = np.asarray(N, dtype=float)
    finite_rows = np.isfinite(N).all(axis=1)
    if not finite_rows.any():
        logger.warning("plot_pca_2d(%s, %s): all rows contain NaN; skipping", task, dataset)
        return None
    N = N[finite_rows]
    y = y[finite_rows]
    if sample is not None and len(y) > sample:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), size=sample, replace=False)
        N = N[idx]
        y = y[idx]
    Xs = StandardScaler().fit_transform(N)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=figsize)
    if task == "pd":
        for label, marker, alpha in [(0, "o", 0.5), (1, "^", 0.7)]:
            mask = y == label
            ax.scatter(pcs[mask, 0], pcs[mask, 1], s=8, marker=marker, alpha=alpha,
                       label=f"y={label} (n={mask.sum()})")
        ax.legend(loc="best")
    else:
        sc = ax.scatter(pcs[:, 0], pcs[:, 1], s=8, c=y, cmap="viridis", alpha=0.7)
        plt.colorbar(sc, ax=ax, label="LGD target")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    ax.set_title(f"{dataset} -- 2D PCA (n_shown={len(y):,})")
    plt.tight_layout()
    return save_or_show(fig, out_path)


__all__ = [
    "list_raw_datasets", "list_processed_datasets",
    "load_raw_csv", "load_processed_dataset",
    "raw_dataset_summary_table", "processed_dataset_summary_table",
    "pd_target_balance_table", "lgd_target_distribution_table",
    "numeric_feature_stats",
    "plot_dataset_size_bar", "plot_target_balance", "plot_lgd_target_hists",
    "plot_correlation_heatmap", "plot_pca_2d",
    "save_or_show",
]
