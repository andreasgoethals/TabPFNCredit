"""Reliability diagrams + ECE bin views (B7).

These plots consume TALENT's calibration outputs. ``RunResult.predict_proba``
provides the probabilities and TALENT's ``ECE`` metric uses the same bins
implementation as :func:`reliability_diagram` here, so the figure and the
metric agree by construction.

Two entry points:

* :func:`reliability_diagram` -- single (method, fold) confidence-vs-accuracy plot.
* :func:`reliability_grid` -- one diagram per method, side-by-side, for a benchmark sweep.

Both write to ``figures/<experiment>/<task>/<dataset>/`` by default and
return the file path so notebooks can ``IPython.display.Image(path)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import matplotlib
matplotlib.use("Agg", force=False)  # safe under SLURM (no display); GUI backends still work locally
import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = 15,
    title: str = "Reliability diagram",
    out_path: Optional[Path] = None,
    ax: Optional[plt.Axes] = None,
) -> Optional[Path]:
    """Plot a reliability diagram for binary or multiclass classification.

    Uses the same equal-width binning as TALENT's
    :func:`~TALENT.model.lib.calibration.expected_calibration_error`, so
    the bar heights here add up (weighted by bin mass) to the reported ECE.
    """
    y_true = np.asarray(y_true).ravel()
    proba = np.asarray(y_proba)
    if proba.ndim == 1:
        # Binary; expand to (N, 2) for argmax
        proba = np.stack([1.0 - proba, proba], axis=1)
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    accuracies = (predictions == y_true.astype(int)).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    accs = np.zeros(n_bins)
    confs = np.zeros(n_bins)
    fracs = np.zeros(n_bins)
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)
        n_in = int(in_bin.sum())
        if n_in > 0:
            accs[i] = float(accuracies[in_bin].mean())
            confs[i] = float(confidences[in_bin].mean())
            fracs[i] = n_in / n

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 5))
    bar_widths = (edges[1:] - edges[:-1]) * 0.9
    ax.bar(centres, accs, width=bar_widths, alpha=0.7, label="Accuracy", edgecolor="black")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Ideal")
    ax.scatter(centres, confs, marker="x", color="red", zorder=5, label="Mean confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    if out_path is not None and own_fig:
        plt.tight_layout()
        plt.savefig(_ensure_dir(out_path), dpi=150)
        plt.close(fig)
        return out_path
    return None


def reliability_grid(
    runs: Mapping[str, Mapping[str, np.ndarray]],
    *,
    n_bins: int = 15,
    out_path: Path,
    ncols: int = 4,
    suptitle: str = "Reliability diagrams",
) -> Path:
    """Plot a grid of reliability diagrams, one per method.

    ``runs`` maps method-name to a dict with keys ``y_true`` and ``y_proba``.
    """
    items = list(runs.items())
    ncols = max(1, min(ncols, len(items)))
    nrows = (len(items) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, (method, data) in zip(axes_arr, items):
        reliability_diagram(
            data["y_true"], data["y_proba"],
            n_bins=n_bins, title=method, ax=ax,
        )
    for ax in axes_arr[len(items):]:
        ax.set_visible(False)
    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    plt.savefig(_ensure_dir(out_path), dpi=150)
    plt.close(fig)
    return out_path


__all__ = ["reliability_diagram", "reliability_grid"]
