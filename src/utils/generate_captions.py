"""Auto-generate the per-figure ``CAPTIONS.md`` files under ``figures/``.

Every figure is saved with a structured stem (e.g. ``pd_bar_mean_auc``,
``lgd_tabpfn_v3_vs_catboost_scatter_r2``, ``pd_cd_auc``). This tool parses those
stems and writes a research-paper caption for each, into one ``CAPTIONS.md`` per
leaf figure directory, ordered like the notebooks (matrix views grouped by
metric, then HPO / compute / cost-quality / head-to-head, then curves,
per-dataset plots and the statistical figures). Method names are rendered with
the standard display labels (TALENT-free import, so this runs anywhere).

    python -m src.utils.generate_captions
    python -m src.utils.generate_captions --experiment experiment1 --dry-run
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
                         f"Distribution of {_metric(g[1])} across the {TASK_DISPLAY[g[0]]} datasets "
                         f"(one point per dataset), one box per method, ordered by the mean.")))
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
              lambda g: ((S_HEAD, _metric_rank(g[3]), 0),
                         f"Per-dataset relative {_metric(g[3])} gain of {display_name(g[1])} over "
                         f"{display_name(g[2])} (y) against dataset size in rows (x, log scale); points "
                         f"are green where {display_name(g[1])} wins and red where {display_name(g[2])} "
                         f"wins, the dashed line marks equal performance, and the solid line is the OLS "
                         f"trend.")))
    R.append((rf"^(pd|lgd)_(.+?)_vs_(.+?)_scatter_({M})$",
              lambda g: ((S_HEAD, _metric_rank(g[3]), 1),
                         f"Per-dataset head-to-head of {display_name(g[1])} (y) versus "
                         f"{display_name(g[2])} (x) on {_metric(g[3])}; each point is a dataset and the "
                         f"dashed line is y = x, so points above it are datasets where "
                         f"{display_name(g[1])} wins.")))

    # ---- pooled learning / imbalance curves ----
    R.append((rf"^(pd|lgd)_(learning_curve|imbalance_curve)_({M})(_relative)?(_smooth)?$",
              lambda g: ((S_CURVE, 0 if g[1] == "learning_curve" else 1, _metric_rank(g[2]),
                          (2 if g[3] else 0) + (1 if g[4] else 0)),
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
                         "bar per method, ordered best-first; foundation-model names are in red.")))
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
#  Per-directory titles + writing
# ---------------------------------------------------------------------------
_DIR_TITLES = {
    "data_exploration": "Data exploration", "experiment0": "Experiment 0",
    "experiment1": "Experiment 1", "experiment2": "Experiment 2", "experiment3": "Experiment 3",
}
_SUB_TITLES = {
    "pd": "PD", "lgd": "LGD", "pd_stats": "PD — statistics", "lgd_stats": "LGD — statistics",
    "pd_family": "PD — champion statistics", "lgd_family": "LGD — champion statistics",
}


def _title(rel_parts: Sequence[str]) -> str:
    head = _DIR_TITLES.get(rel_parts[0], rel_parts[0])
    if len(rel_parts) > 1:
        return f"{head} — {_SUB_TITLES.get(rel_parts[-1], rel_parts[-1])}"
    return head


def generate_captions(figures_root: Path, experiments: Optional[Sequence[str]] = None,
                      dry_run: bool = False) -> List[Path]:
    """Write one ``CAPTIONS.md`` per leaf directory under ``figures_root`` that
    contains ``.pdf`` figures. Returns the list of files written."""
    figures_root = Path(figures_root)
    exps = {e.lower() for e in experiments} if experiments else None
    written: List[Path] = []
    for d in sorted(p for p in figures_root.rglob("*") if p.is_dir()):
        pdfs = sorted(f.stem for f in d.glob("*.pdf"))
        if not pdfs:
            continue
        rel = d.relative_to(figures_root).parts
        if exps and rel[0].lower() not in exps:
            continue
        entries = sorted((caption_for(s) + (s,) for s in pdfs), key=lambda x: x[0])
        lines = [
            f"<!-- Auto-generated by src/utils/generate_captions.py. Figures are listed in "
            f"notebook order. Conventions: tabular foundation models are shown in red; methods "
            f"are ordered best-first (left/top); bar error bars are the fold-level standard "
            f"deviation unless noted. -->\n",
            f"# {_title(rel)} — figure captions\n",
        ]
        for _key, caption, stem in entries:
            lines.append(f"**`{stem}.pdf`**\n> {caption}\n")
        text = "\n".join(lines) + "\n"
        out = d / "CAPTIONS.md"
        if not dry_run:
            out.write_text(text, encoding="utf-8")
        written.append(out)
    return written


def main(argv: Optional[List[str]] = None) -> int:
    from src.utils.paths import PROJECT_ROOT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--figures-root", type=Path, default=PROJECT_ROOT / "figures")
    ap.add_argument("--experiment", nargs="*", help="limit to these top-level dirs (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)
    written = generate_captions(args.figures_root, experiments=args.experiment, dry_run=args.dry_run)
    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {len(written)} CAPTIONS.md file(s):")
    for p in written:
        print(f"  {p.relative_to(args.figures_root.parent)}")
    if not written:
        print("  (no figure directories found)")
    return 0


__all__ = ["caption_for", "generate_captions", "main", "METRIC_DISPLAY"]


if __name__ == "__main__":
    raise SystemExit(main())
