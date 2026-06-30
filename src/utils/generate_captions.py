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
import re
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.methods.method_names import display_name  # noqa: E402

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


def _size_trend_is_relative(g) -> bool:
    task, _fnd, base, metric = g
    # The LGD notebook uses a relative R2-over-size plot for the CatBoost
    # comparison only; the linear-regression size trend remains an absolute
    # R2-difference plot. The filename stem does not encode this flag.
    if task == "lgd" and metric == "r2" and base == "LinearRegression":
        return False
    return True


def _size_trend_caption(g) -> str:
    if not _size_trend_is_relative(g):
        return (
            f"Per-dataset {_metric(g[3])} difference of {display_name(g[1])} minus "
            f"{display_name(g[2])} (y) against dataset size in rows (x, log scale); "
            f"points are green where {display_name(g[1])} wins and red where "
            f"{display_name(g[2])} wins, the dashed line marks equal performance, "
            f"and the solid line is the OLS trend."
        )
    return (
        f"Per-dataset relative {_metric(g[3])} improvement of {display_name(g[1])} over "
        f"{display_name(g[2])} (y, %) against dataset size in rows (x, log scale); "
        f"points are green where {display_name(g[1])} wins and red where "
        f"{display_name(g[2])} wins, the dashed line marks equal performance, "
        f"and the solid line is the OLS trend."
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
                         f"{_cap(_metric(g[1]))} of each method (columns) on every {TASK_DISPLAY[g[0]]} "
                         f"dataset (rows), as the five-fold mean of the tuned model. Columns are "
                         f"ordered best-first and shaded green (best) to red (worst); foundation-model "
                         f"names are in red.")))
    R.append((rf"^(pd|lgd)_bar_mean_({M})$",
              lambda g: ((S_MATRIX, _metric_rank(g[1]), _VIEW["bar_mean"]),
                         f"Per-method mean of {_metric(g[1])} across the {TASK_DISPLAY[g[0]]} "
                         f"datasets, ordered best-first; the value is printed above each bar and the "
                         f"error bar is the fold-level standard deviation. Foundation-model names are "
                         f"in red.")))
    R.append((rf"^(pd|lgd)_box_({M})$",
              lambda g: ((S_MATRIX, _metric_rank(g[1]), _VIEW["box"]),
                         f"Per-method distribution of {_metric(g[1])} on the {TASK_DISPLAY[g[0]]} "
                         f"datasets: each box spans the method's per-fold scores and each dot is one "
                         f"dataset's fold-mean, with boxes ordered by the mean.")))
    R.append((rf"^(pd|lgd)_rank_matrix_({M})$",
              lambda g: ((S_MATRIX, _metric_rank(g[1]), _VIEW["rank_matrix"]),
                         f"Per-dataset rank of each method on {_metric(g[1])} (1 = best, green; worse, "
                         f"red), with columns ordered by mean rank. Foundation-model names are in "
                         f"red.")))
    R.append((rf"^(pd|lgd)_ranking_({M})$",
              lambda g: ((S_MATRIX, _metric_rank(g[1]), _VIEW["ranking"]),
                         f"Mean {_metric(g[1])} rank per method across the {TASK_DISPLAY[g[0]]} datasets "
                         f"(1 = best, lower is better), ordered best-first; error bars are the standard "
                         f"deviation of the per-dataset ranks.")))
    R.append((rf"^(pd|lgd)_rank_box_({M})$",
              lambda g: ((S_MATRIX, _metric_rank(g[1]), _VIEW["rank_box"]),
                         f"Distribution of each method's per-dataset {_metric(g[1])} ranks (1 = best, at "
                         f"the top), one box per method.")))

    # ---- HPO effect ----
    R.append((rf"^(pd|lgd)_hpo_effect_({M})$",
              lambda g: ((S_HPO, _metric_rank(g[1]), 0),
                         f"Mean effect of hyper-parameter tuning on {_metric(g[1])} (tuned minus "
                         f"untuned, averaged over datasets); positive (green) means tuning helps. "
                         f"Foundation models are zero by construction, as they are not tuned.")))

    # ---- compute time ----
    R.append((r"^(pd|lgd)_bar_compute_time$",
              lambda g: ((S_COMPUTE, 0, 0),
                         f"Mean compute time per method on {TASK_DISPLAY[g[0]]} (train + predict per "
                         f"fold, seconds, log axis; for tuned methods the train time includes the "
                         f"HPO-search cost), fastest first. Foundation-model names are in red.")))
    R.append((r"^(pd|lgd)_box_compute_time$",
              lambda g: ((S_COMPUTE, 1, 0),
                         f"Distribution of compute time (train + predict) per method, one point per "
                         f"(dataset, fold), on a log axis and ordered fastest first.")))

    # ---- cost / quality ----
    R.append((rf"^(pd|lgd)_cost_quality_({M})$",
              lambda g: ((S_COSTQ, _metric_rank(g[1]), 0),
                         f"Mean {_metric(g[1])} (y) against mean compute time (x; train + predict, log "
                         f"scale) for each method on {TASK_DISPLAY[g[0]]}; foundation models are red "
                         f"stars, the other methods blue circles, and notable methods are labelled.")))

    # ---- foundation-vs-baseline head-to-head ----
    R.append((rf"^(pd|lgd)_(.+?)_vs_(.+?)_sizetrend_({M})$",
              lambda g: ((S_HEAD, _metric_rank(g[3]), 0), _size_trend_caption(g))))
    R.append((rf"^(pd|lgd)_(.+?)_vs_(.+?)_scatter_({M})$",
              lambda g: ((S_HEAD, _metric_rank(g[3]), 1),
                         f"Per-dataset head-to-head of {display_name(g[1])} (y) versus "
                         f"{display_name(g[2])} (x) on {_metric(g[3])}; each point is a dataset and the "
                         f"dashed line is y = x, so points above it are datasets where "
                         f"{display_name(g[1])} wins.")))

    # ---- pooled learning / imbalance curves ----
    R.append((rf"^(pd|lgd)_(learning_curve|imbalance_curve)_({M})(_relative)?(_smooth)?(_logx)?$",
              lambda g: ((S_CURVE, 0 if g[1] == "learning_curve" else 1, _metric_rank(g[2]),
                          (4 if g[3] else 0) + (2 if g[4] else 0) + (1 if g[5] else 0)),
                         _curve_caption(g))))

    # ---- per-dataset raw-point curves ----
    R.append((rf"^(pd|lgd)_row_limit_(.+)_({M})$",
              lambda g: ((S_PERDS, 0, _dataset(g[1])),
                         f"{TASK_DISPLAY[g[0]]} — {_dataset(g[1])}: {_metric(g[2])} versus training-set "
                         f"size (rows), one line per method (raw per-point values, no smoothing).")))
    R.append((rf"^(pd|lgd)_minority_proportion_(.+)_({M})$",
              lambda g: ((S_PERDS, 1, _dataset(g[1])),
                         f"{TASK_DISPLAY[g[0]]} — {_dataset(g[1])}: {_metric(g[2])} versus the "
                         f"minority-class (default) proportion, one line per method (raw per-point "
                         f"values).")))

    # ---- statistical figures ----
    R.append((r"^(pd|lgd)_pama$",
              lambda g: ((S_STAT, 0, 0),
                         "Share of (dataset, fold) observations on which each method achieves the "
                         "single best score (PAMA, Probability of Achieving the MAximal accuracy), one "
                         "bar per method that wins at least once, ordered best-first; foundation-model "
                         "names are in red.")))
    R.append((r"^(pd|lgd)_pama_min2wins$",
              lambda g: ((S_STAT, 0, 1),
                         "Share of (dataset, fold) observations on which each method achieves the "
                         "single best score (PAMA), restricted to methods that win on at least two "
                         "observations, one bar per method, ordered best-first; foundation-model names "
                         "are in red.")))
    R.append((rf"^(pd|lgd)_cd_({M})$",
              lambda g: ((S_STAT, 1, _metric_rank(g[1])),
                         f"Nemenyi critical-difference diagram of the {_metric(g[1])} average ranks "
                         f"(1 = best); methods connected by a bold bar are not significantly different "
                         f"(the bar spans one critical difference). Foundation-model names are in "
                         f"red.")))
    R.append((r"^(pd|lgd)_winloss$",
              lambda g: ((S_STAT, 2, 0),
                         "Pairwise win/loss matrix: each cell 'W/L' is the number of datasets on which "
                         "the row method beats versus loses to the column method (ties omitted), "
                         "coloured by the win-minus-loss margin (red = the row method dominates, blue = "
                         "it is dominated). Methods are ordered by mean rank, so the strongest sit "
                         "top-left; foundation-model names are in red.")))
    R.append((r"^(pd|lgd)_significance$",
              lambda g: ((S_STAT, 3, 0),
                         "Pairwise matrix of the multiple-comparison-adjusted p-values for every pair "
                         "of methods: each cell is green where the two methods differ significantly and "
                         "red where they do not, with the p-value printed in the cell. The matrix is "
                         "symmetric, methods are ordered best-first, and foundation-model names are in "
                         "red.")))

    return [(re.compile(p), b) for p, b in R]


def _cap(s: str) -> str:
    """Capitalise the first letter only (keeps acronyms like AUC intact)."""
    return s[:1].upper() + s[1:] if s else s


def _curve_caption(g) -> str:
    learning = g[1] == "learning_curve"
    kind = "Learning curve" if learning else "Imbalance-robustness curve"
    xaxis = "training-set size (rows)" if learning else "minority-class (default) proportion"
    notes = []
    if g[3]:
        notes.append("relative to each method's own best")
    if g[4]:
        notes.append("moving average")
    if len(g) > 5 and g[5]:
        notes.append("log-scaled x-axis")
    if g[2] == "r2" and not g[3]:
        notes.append("R² below 0 shown at 0")
    notes.append("mean over datasets")
    return (f"{kind}: {_metric(g[2])} versus {xaxis}, one line per method "
            f"({'; '.join(notes)}).")


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


__all__ = ["caption_for", "generate_captions", "main", "CHAPTERS", "METRIC_DISPLAY"]


if __name__ == "__main__":
    raise SystemExit(main())
