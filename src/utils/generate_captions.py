"""Auto-generate a single ``figures/CAPTIONS.md`` for every figure.

Every figure is saved with a structured stem (e.g. ``pd_bar_mean_auc``,
``lgd_tabpfn_v3_vs_catboost_scatter_r2``, ``pd_cd_auc``). This tool parses those
stems, writes a research-paper caption for each, and collects them into ONE
``figures/CAPTIONS.md`` split into chapters -- one per analysis notebook, in
notebook order -- with each chapter's figures listed in the order the notebook
produces them and the figure's file name as the heading. Method names use the
standard display labels (TALENT-free import, so this runs anywhere).

    python -m src.utils.generate_captions
    python -m src.utils.generate_captions --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.methods.method_names import display_name  # noqa: E402
from src.utils.paths import PROJECT_ROOT  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Metric + dataset display
# ---------------------------------------------------------------------------
METRIC_DISPLAY = {
    "auc": "AUC", "brier": "the Brier score", "f1": "F1", "ks": "the KS statistic",
    "ap_normalized": "the prevalence-corrected average precision (AP)",
    "accuracy": "accuracy", "balanced_accuracy": "balanced accuracy", "mcc": "MCC",
    "ece": "the expected calibration error", "logloss": "log-loss",
    "r2": "R²", "rmse": "RMSE", "mae": "MAE",
    "pearson_corr": "Pearson correlation", "spearman_corr": "Spearman correlation",
}
# Notebook ordering of metrics (PD block then LGD block).
METRIC_ORDER = ["auc", "brier", "f1", "ks", "ap_normalized", "accuracy",
                "balanced_accuracy", "mcc", "ece", "logloss",
                "r2", "rmse", "mae", "pearson_corr", "spearman_corr"]
# Longest-first alternation so a multi-token metric matches before its prefix.
_METRIC_RE = "|".join(sorted(METRIC_DISPLAY, key=len, reverse=True))
TASK_DISPLAY = {"pd": "PD", "lgd": "LGD"}


def _metric(m: str) -> str:
    return METRIC_DISPLAY.get(m, m.upper())


def _metric_rank(m: str) -> int:
    return METRIC_ORDER.index(m) if m in METRIC_ORDER else 99


def _dataset(tok: str) -> str:
    """`0003_vehicle_loan` -> `vehicle loan`."""
    return re.sub(r"^\d+[_.]", "", tok).replace("_", " ")


def _size_trend_caption(g) -> str:
    return (
        f"Relative {_metric(g[3])} gain of {display_name(g[1])} over "
        f"{display_name(g[2])} for each dataset, plotted against dataset size "
        f"(rows, log scale). Points above zero indicate datasets where "
        f"{display_name(g[1])} outperforms {display_name(g[2])}; the fitted line "
        f"summarises how the gain changes with dataset size."
    )


# ---------------------------------------------------------------------------
#  Stem -> (sort key, caption). Each rule: (regex, builder); builder returns
#  (sort_key_tuple, caption_str). First matching rule wins.
# ---------------------------------------------------------------------------
# Section indices give the top-level order; tuples after them break ties.
S_DATA, S_MATRIX, S_HPO, S_COMPUTE, S_COSTQ, S_HEAD, S_CURVE, S_PERDS, S_STAT, S_FALLBACK = range(10)
_VIEW = {"heatmap": 0, "bar_mean": 1, "box": 2, "rank_matrix": 3, "ranking": 4, "rank_box": 5}


def _rules() -> List[Tuple[re.Pattern, Callable]]:
    M = _METRIC_RE
    R: List[Tuple[str, Callable]] = []

    # ---- data exploration ----
    R.append((r"^(pd|lgd)_dataset_sizes$",
              lambda g: ((S_DATA, 0, g[0]),
                         f"Number of rows in each {TASK_DISPLAY[g[0]]} dataset (one bar per dataset, "
                         f"log scale).")))
    R.append((r"^(pd|lgd)_target_balance$",
              lambda g: ((S_DATA, 1, g[0]),
                         "Default rate (share of the positive class) of each PD dataset, one bar per "
                         "dataset.")))
    R.append((r"^(pd|lgd)_target_hists$",
              lambda g: ((S_DATA, 1, g[0]),
                         "Histogram of the loss-given-default target value for each LGD dataset.")))
    R.append((r"^corr_(.+)$",
              lambda g: ((S_DATA, 2, g[0]),
                         f"Pairwise feature-correlation heatmap for the {_dataset(g[0])} dataset.")))
    R.append((r"^pca_(.+)$",
              lambda g: ((S_DATA, 3, g[0]),
                         f"First two principal components of the {_dataset(g[0])} dataset "
                         f"(one point per instance).")))

    # ---- matrix-metric views (heatmap / bar / box / rank views) ----
    R.append((rf"^(pd|lgd)_heatmap_({M})$",
              lambda g: ((S_MATRIX, _metric_rank(g[1]), _VIEW["heatmap"]),
                         f"Five-fold mean {_metric(g[1])} for each method on each "
                         f"{TASK_DISPLAY[g[0]]} dataset. Rows are datasets, columns are "
                         f"methods ordered by overall performance, and colour encodes the score.")))
    R.append((rf"^(pd|lgd)_bar_mean_({M})$",
              lambda g: ((S_MATRIX, _metric_rank(g[1]), _VIEW["bar_mean"]),
                         f"Across-dataset mean for {_metric(g[1])} by method on the "
                         f"{TASK_DISPLAY[g[0]]} datasets. Bars are ordered by mean performance and error bars show "
                         f"fold-level variability.")))
    R.append((rf"^(pd|lgd)_box_({M})$",
              lambda g: ((S_MATRIX, _metric_rank(g[1]), _VIEW["box"]),
                         f"Distribution of {_metric(g[1])} by method on the "
                         f"{TASK_DISPLAY[g[0]]} datasets. Boxes summarise fold-level scores, "
                         f"and overlaid points show dataset-level means.")))
    R.append((rf"^(pd|lgd)_rank_matrix_({M})$",
              lambda g: ((S_MATRIX, _metric_rank(g[1]), _VIEW["rank_matrix"]),
                         f"Per-dataset method ranks for {_metric(g[1])} (rank 1 = best). "
                         f"Columns are ordered by average rank across datasets.")))
    R.append((rf"^(pd|lgd)_ranking_({M})$",
              lambda g: ((S_MATRIX, _metric_rank(g[1]), _VIEW["ranking"]),
                         f"Average method rank for {_metric(g[1])} across the "
                         f"{TASK_DISPLAY[g[0]]} datasets (rank 1 = best). Error bars show "
                         f"the standard deviation of per-dataset ranks.")))
    R.append((rf"^(pd|lgd)_rank_box_({M})$",
              lambda g: ((S_MATRIX, _metric_rank(g[1]), _VIEW["rank_box"]),
                         f"Distribution of per-dataset {_metric(g[1])} ranks for each method "
                         f"(rank 1 = best).")))

    # ---- HPO effect ----
    R.append((rf"^(pd|lgd)_hpo_effect_({M})$",
              lambda g: ((S_HPO, _metric_rank(g[1]), 0),
                         f"Effect of hyper-parameter tuning on {_metric(g[1])}, computed as "
                         f"the tuned score minus the untuned score and averaged across datasets. "
                         f"Positive values indicate that tuning improves performance.")))

    # ---- compute time ----
    R.append((r"^(pd|lgd)_bar_compute_time$",
              lambda g: ((S_COMPUTE, 0, 0),
                         f"Mean compute time per method on {TASK_DISPLAY[g[0]]} "
                         f"(train + prediction time per fold, seconds, log scale). For tuned "
                         f"methods, training time includes the hyper-parameter search.")))
    R.append((r"^(pd|lgd)_box_compute_time$",
              lambda g: ((S_COMPUTE, 1, 0),
                         f"Distribution of per-fold compute time (train + prediction time) "
                         f"for each method, with one observation per dataset-fold pair.")))

    # ---- cost / quality ----
    R.append((rf"^(pd|lgd)_cost_quality_({M})$",
              lambda g: ((S_COSTQ, _metric_rank(g[1]), 0),
                         f"Trade-off between mean {_metric(g[1])} and mean compute time "
                         f"(train + prediction time, log scale) for each method on "
                         f"{TASK_DISPLAY[g[0]]}.")))

    # ---- foundation-vs-baseline head-to-head ----
    R.append((rf"^(pd|lgd)_(.+?)_vs_(.+?)_sizetrend_({M})$",
              lambda g: ((S_HEAD, _metric_rank(g[3]), 0), _size_trend_caption(g))))
    R.append((rf"^(pd|lgd)_(.+?)_vs_(.+?)_scatter_({M})$",
              lambda g: ((S_HEAD, _metric_rank(g[3]), 1),
                         f"Per-dataset comparison of {display_name(g[1])} and "
                         f"{display_name(g[2])} on {_metric(g[3])}. Each point is one "
                         f"dataset; points above the diagonal indicate higher performance for "
                         f"{display_name(g[1])}.")))

    # ---- pooled learning / imbalance curves ----
    R.append((rf"^(pd|lgd)_(learning_curve|imbalance_curve)_({M})_combined(_less|_more)?$",
              lambda g: ((S_CURVE, 0 if g[1] == "learning_curve" else 1, _metric_rank(g[2]),
                          _curve_combined_variant_rank(g)),
                         _curve_combined_caption(g))))
    R.append((rf"^(pd|lgd)_(learning_curve|imbalance_curve)_({M})(_relative)?(_smooth(?:_less|_more)?)?(_zoom)?(_logx)?$",
              lambda g: ((S_CURVE, 0 if g[1] == "learning_curve" else 1, _metric_rank(g[2]),
                          _curve_variant_rank(g)),
                         _curve_caption(g))))

    # ---- per-dataset raw-point curves ----
    R.append((rf"^(pd|lgd)_row_limit_(.+)_({M})$",
              lambda g: ((S_PERDS, 0, _dataset(g[1])),
                         f"{TASK_DISPLAY[g[0]]} - {_dataset(g[1])}: {_metric(g[2])} as a function "
                         f"of dataset size. Each line represents one method.")))
    R.append((rf"^(pd|lgd)_minority_proportion_(.+)_({M})$",
              lambda g: ((S_PERDS, 1, _dataset(g[1])),
                         f"{TASK_DISPLAY[g[0]]} - {_dataset(g[1])}: {_metric(g[2])} as a function "
                         f"of the minority-class (default) proportion. Each line represents "
                         f"one method.")))

    # ---- statistical figures ----
    R.append((r"^(pd|lgd)_pama$",
              lambda g: ((S_STAT, 0, 0),
                         "Probability of achieving the maximal accuracy (PAMA): the share of "
                         "dataset-fold observations on which each method attains the best score.")))
    R.append((r"^(pd|lgd)_pama_min2wins$",
              lambda g: ((S_STAT, 0, 1),
                         "Probability of achieving the maximal accuracy (PAMA), restricted to "
                         "methods that attain the best score on at least two dataset-fold "
                         "observations.")))
    R.append((rf"^(pd|lgd)_cd_({M})$",
              lambda g: ((S_STAT, 1, _metric_rank(g[1])),
                         f"Nemenyi critical-difference diagram for average {_metric(g[1])} "
                         f"ranks (rank 1 = best). Methods connected by a horizontal bar are "
                         f"not significantly different at the chosen significance level.")))
    R.append((r"^(pd|lgd)_winloss$",
              lambda g: ((S_STAT, 2, 0),
                         "Pairwise win/loss matrix across datasets. Each cell reports the number "
                         "of datasets on which the row method outperforms versus underperforms "
                         "the column method; ties are omitted.")))
    R.append((r"^(pd|lgd)_significance$",
              lambda g: ((S_STAT, 3, 0),
                         "Pairwise matrix of multiple-comparison-adjusted p-values for method "
                         "comparisons. Cells indicate whether the corresponding pair differs "
                         "significantly after adjustment.")))

    return [(re.compile(p), b) for p, b in R]


def _cap(s: str) -> str:
    """Capitalise the first letter only (keeps acronyms like AUC intact)."""
    return s[:1].upper() + s[1:] if s else s


_LEARNING_XAXIS = "dataset size"
_SMOOTH_CAPTION = {
    "less": "Lines show moving-average trends using a shorter window.",
    "standard": "Lines show moving-average trends using the standard window.",
    "more": "Lines show moving-average trends using a longer window.",
}
_COMBINED_SMOOTH_CAPTION = {
    "less": "Moving-average lines use a shorter window.",
    "standard": "Moving-average lines use the standard window.",
    "more": "Moving-average lines use a longer window.",
}
_SMOOTH_ORDER = {"less": 0, "standard": 1, "more": 2}


def _smooth_variant(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    token = token.lstrip("_")
    if token == "smooth":
        return "standard"
    if token.startswith("smooth_"):
        return token[len("smooth_"):]
    if token in {"less", "more"}:
        return token
    return "standard"


def _curve_caption(g) -> str:
    learning = g[1] == "learning_curve"
    kind = "Learning curve" if learning else "Imbalance-robustness curve"
    xaxis = _LEARNING_XAXIS if learning else "minority-class (default) proportion"
    notes = ["Each line represents one method, averaged across datasets."]
    if g[3]:
        notes.append("Values are shown relative to each method's best observed value.")
    smooth_variant = _smooth_variant(g[4])
    if smooth_variant:
        notes.append(_SMOOTH_CAPTION.get(smooth_variant, _SMOOTH_CAPTION["standard"]))
    if g[5]:
        if learning:
            notes.append("The inset highlights the low-data region.")
        else:
            notes.append("The inset highlights the severe-imbalance region.")
    if len(g) > 6 and g[6]:
        notes.append("The x-axis uses a logarithmic scale.")
    return f"{kind} showing {_metric(g[2])} as a function of {xaxis}. {' '.join(notes)}"


def _curve_combined_caption(g) -> str:
    learning = g[1] == "learning_curve"
    kind = "Learning curve" if learning else "Imbalance-robustness curve"
    xaxis = _LEARNING_XAXIS if learning else "minority-class (default) proportion"
    smooth_variant = _smooth_variant(g[3])
    return (
        f"{kind} showing {_metric(g[2])} as a function of {xaxis}. "
        f"Points show sweep estimates and lines show moving-average trends, "
        f"with one colour per method. "
        f"{_COMBINED_SMOOTH_CAPTION.get(smooth_variant or 'standard')}"
    )


def _curve_combined_variant_rank(g) -> int:
    smooth_variant = _smooth_variant(g[3]) or "standard"
    return 5 + _SMOOTH_ORDER.get(smooth_variant, _SMOOTH_ORDER["standard"])


def _curve_variant_rank(g) -> int:
    """Generation order for pooled curve variants in the notebooks."""
    relative = bool(g[3])
    smooth_variant = _smooth_variant(g[4])
    smooth = smooth_variant is not None
    zoom = bool(g[5])
    logx = bool(g[6]) if len(g) > 6 else False
    if logx:
        return 20 + (4 if relative else 0) + (2 if smooth else 0)
    if relative:
        return 10 + (_SMOOTH_ORDER.get(smooth_variant, 0) + 1 if smooth else 0)
    if smooth:
        return 2 + _SMOOTH_ORDER.get(smooth_variant, _SMOOTH_ORDER["standard"])
    if zoom:
        return 1
    return 0


_RULES = _rules()


def caption_for(stem: str) -> Tuple[Tuple, str]:
    """Return ``(sort_key, caption)`` for a figure stem (no extension)."""
    for pat, builder in _RULES:
        m = pat.match(stem)
        if m:
            return builder(m.groups())
    return ((S_FALLBACK, stem), f"Figure `{stem}`.")


# ---------------------------------------------------------------------------
#  Consolidated CAPTIONS.md  (one file, chapters in notebook order)
# ---------------------------------------------------------------------------
# Each chapter mirrors one analysis notebook, in the same order the notebooks
# appear in notebooks/; the second item is the figure sub-directory that notebook
# writes to under figures/.
CHAPTERS: List[Tuple[str, str]] = [
    ("Data exploration",                           "data_exploration"),
    ("Experiment 0 — pilot coverage",              "experiment0"),
    ("Experiment 1.1 — PD benchmark",              "experiment1/pd"),
    ("Experiment 1.2 — PD statistical analysis",   "experiment1/pd_stats"),
    ("Experiment 1.3 — PD champion-level statistics", "experiment1/pd_family"),
    ("Experiment 1.4 — LGD benchmark",             "experiment1/lgd"),
    ("Experiment 1.5 — LGD statistical analysis",  "experiment1/lgd_stats"),
    ("Experiment 1.6 — LGD champion-level statistics", "experiment1/lgd_family"),
    ("Experiment 2.1 — PD data-efficiency sweep",  "experiment2/pd"),
    ("Experiment 2.2 — LGD data-efficiency sweep", "experiment2/lgd"),
    ("Experiment 3 — imbalance-robustness sweep",  "experiment3"),
]


def _ordered_stems(d: Path) -> List[str]:
    """The ``.pdf`` stems in directory ``d`` in notebook (figure-generation) order."""
    pdfs = [f.stem for f in d.glob("*.pdf")]
    return [s for _key, s in sorted((caption_for(s)[0], s) for s in pdfs)]


def generate_captions(figures_root: Path, experiments: Optional[Sequence[str]] = None,
                      dry_run: bool = False) -> List[Path]:
    """Write a SINGLE ``figures/CAPTIONS.md``: one chapter per notebook (in
    notebook order), each listing its figures in generation order with the figure
    file name as the heading. Any stale per-directory ``CAPTIONS.md`` from the old
    layout is removed. Returns ``[the file]`` (or ``[]`` if there are no figures)."""
    figures_root = Path(figures_root)
    exps = {e.lower() for e in experiments} if experiments else None

    # Old layout wrote one CAPTIONS.md per leaf dir; everything now lives in one
    # file at the figures root, so clear the per-directory ones.
    for stale in figures_root.rglob("CAPTIONS.md"):
        if stale.parent != figures_root and not dry_run:
            stale.unlink(missing_ok=True)

    def _section(title: str, d: Path) -> List[str]:
        stems = _ordered_stems(d)
        if not stems:
            return []
        out = [f"## {title}\n"]
        for stem in stems:
            out.append(f"**`{stem}.pdf`**\n> {caption_for(stem)[1]}\n")
        return out

    body: List[str] = []
    covered = set()
    for title, sub in CHAPTERS:
        if exps and sub.split("/")[0].lower() not in exps:
            continue
        d = figures_root / sub
        covered.add(d.resolve())
        if d.is_dir():
            body += _section(title, d)

    # Catch any figure directory not listed above, so nothing is silently dropped.
    for d in sorted(p for p in figures_root.rglob("*") if p.is_dir()):
        if d.resolve() in covered or not any(d.glob("*.pdf")):
            continue
        rel = "/".join(d.relative_to(figures_root).parts)
        if exps and rel.split("/")[0].lower() not in exps:
            continue
        body += _section(rel, d)

    if not body:
        return []
    header = (
        "<!-- Auto-generated by src/utils/generate_captions.py -- do not edit by hand. -->\n\n"
        "# Figure captions\n\n"
        "Captions for every generated figure, grouped by notebook (in notebook order) and, within "
        "each chapter, in the order the figures are produced; the heading of each entry is the "
        "figure's file name. Conventions: tabular foundation models are shown in red; methods are "
        "ordered best-first; bar error bars are the fold-level standard deviation unless noted.\n"
    )
    out = figures_root / "CAPTIONS.md"
    if not dry_run:
        figures_root.mkdir(parents=True, exist_ok=True)
        out.write_text(header + "\n" + "\n".join(body) + "\n", encoding="utf-8")
    return [out]


def refresh_captions_for_saved_figure(saved_path: Optional[Path]) -> None:
    """Refresh ``figures/CAPTIONS.md`` after a project figure is written.

    This is intentionally best-effort: figure saving should never fail because
    the caption sidecar could not be regenerated. ``run_notebooks`` disables
    this per-save hook and performs one consolidated refresh at the end.
    """
    if saved_path is None:
        return
    if os.environ.get("TABPFNCREDIT_AUTO_CAPTIONS", "1").lower() in {"0", "false", "no"}:
        return

    figures_root = PROJECT_ROOT / "figures"
    try:
        Path(saved_path).resolve().relative_to(figures_root.resolve())
    except (OSError, ValueError):
        return

    try:
        generate_captions(figures_root)
    except Exception as exc:  # noqa: BLE001 -- captions are non-critical sidecars
        logger.warning("Could not refresh figures/CAPTIONS.md after saving %s: %s", saved_path, exc)


def main(argv: Optional[List[str]] = None) -> int:
    from src.utils.paths import PROJECT_ROOT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--figures-root", type=Path, default=PROJECT_ROOT / "figures")
    ap.add_argument("--experiment", nargs="*", help="limit to these top-level dirs (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)
    written = generate_captions(args.figures_root, experiments=args.experiment, dry_run=args.dry_run)
    if written:
        print(f"{'would write' if args.dry_run else 'wrote'} {written[0]}")
    else:
        print("(no figures found)")
    return 0


__all__ = [
    "caption_for",
    "generate_captions",
    "refresh_captions_for_saved_figure",
    "main",
    "CHAPTERS",
    "METRIC_DISPLAY",
]


if __name__ == "__main__":
    raise SystemExit(main())
