"""Auto-generate LaTeX caption snippets for every saved figure.

Every figure is saved with a structured stem such as ``pd_bar_mean_auc`` or
``lgd_tabpfn_v3_vs_catboost_scatter_r2``. This tool parses those stems, writes
short paper-facing captions, and collects them into one
``figures/CAPTIONS.md`` file grouped by notebook. The generated captions are
intended to sit under the corresponding figure, so they avoid repeating what is
already in the title, axis labels, or legend.

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
    "auc": r"\(\mathrm{AUC}\)",
    "brier": "Brier score",
    "f1": r"\(F_1\)",
    "ks": "KS statistic",
    "ap_normalized": "normalized average precision",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced accuracy",
    "mcc": "MCC",
    "ece": "ECE",
    "logloss": "log loss",
    "r2": r"\(R^2\)",
    "rmse": "RMSE",
    "mae": "MAE",
    "pearson_corr": r"Pearson \(r\)",
    "spearman_corr": r"Spearman \(\rho\)",
}
# Notebook ordering of metrics (PD block then LGD block).
METRIC_ORDER = [
    "auc",
    "brier",
    "f1",
    "ks",
    "ap_normalized",
    "accuracy",
    "balanced_accuracy",
    "mcc",
    "ece",
    "logloss",
    "r2",
    "rmse",
    "mae",
    "pearson_corr",
    "spearman_corr",
]
# Longest-first alternation so a multi-token metric matches before its prefix.
_METRIC_RE = "|".join(sorted(METRIC_DISPLAY, key=len, reverse=True))
TASK_DISPLAY = {"pd": "PD", "lgd": "LGD"}


def _metric(m: str) -> str:
    return METRIC_DISPLAY.get(m, _latex_escape(m.upper()))


def _metric_rank(m: str) -> int:
    return METRIC_ORDER.index(m) if m in METRIC_ORDER else 99


def _latex_escape(text: object) -> str:
    """Escape plain text for use inside ``\\caption{...}``."""
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(text))


def _method(raw: str) -> str:
    return _latex_escape(display_name(raw))


def _dataset(tok: str) -> str:
    """``0003_vehicle_loan`` -> ``vehicle loan``."""
    return _latex_escape(re.sub(r"^\d+[_.]", "", tok).replace("_", " "))


def _with_foundation_note(caption: str) -> str:
    return f"{caption} Foundation models are highlighted in red."


def _size_trend_caption(g) -> str:
    fnd, base = _method(g[1]), _method(g[2])
    return (
        f"Each point is a dataset; green points favour {fnd}, red points favour "
        f"{base}, and the line is an OLS trend over log dataset size."
    )


def _latex_label(stem: str) -> str:
    label = stem.lower().replace("_", "-")
    label = re.sub(r"[^a-z0-9:-]+", "-", label)
    label = re.sub(r"-+", "-", label).strip("-")
    return label or "figure"


def latex_caption_block(stem: str, label_stem: Optional[str] = None) -> str:
    """Return a copy-ready LaTeX caption/label block for one figure stem."""
    caption = caption_for(stem)[1]
    return f"\\caption{{{caption}}}\n\\label{{fig:{_latex_label(label_stem or stem)}}}"


# ---------------------------------------------------------------------------
#  Stem -> (sort key, caption body). Each rule: (regex, builder); builder
#  returns (sort_key_tuple, caption_body). First matching rule wins.
# ---------------------------------------------------------------------------
S_DATA, S_MATRIX, S_HPO, S_COMPUTE, S_COSTQ, S_HEAD, S_CAL, S_CURVE, S_PERDS, S_STAT, S_FALLBACK = range(11)
_VIEW = {"heatmap": 0, "bar_mean": 1, "box": 2, "rank_matrix": 3, "ranking": 4, "rank_box": 5}


def _rules() -> List[Tuple[re.Pattern, Callable]]:
    M = _METRIC_RE
    R: List[Tuple[str, Callable]] = []

    # ---- data exploration ----
    R.append((
        r"^(pd|lgd)_dataset_sizes$",
        lambda g: (
            (S_DATA, 0, g[0]),
            "Sample sizes use a logarithmic scale so small and large datasets remain comparable.",
        ),
    ))
    R.append((
        r"^(pd|lgd)_target_balance$",
        lambda g: ((S_DATA, 1, g[0]), "Datasets are ordered by increasing positive-class rate."),
    ))
    R.append((
        r"^(pd|lgd)_target_hists$",
        lambda g: (
            (S_DATA, 1, g[0]),
            "Red curves show kernel-density estimates; vertical lines mark the mean and median in each panel.",
        ),
    ))
    R.append((
        r"^corr_(.+)$",
        lambda g: (
            (S_DATA, 2, g[0]),
            "Only numerical features are included; color is centered at zero Pearson correlation.",
        ),
    ))
    R.append((
        r"^pca_(.+)$",
        lambda g: (
            (S_DATA, 3, g[0]),
            "PCA is computed after standardizing numerical features; at most 5,000 instances are shown.",
        ),
    ))

    # ---- matrix-metric views (heatmap / bar / box / rank views) ----
    R.append((
        rf"^(pd|lgd)_heatmap_({M})$",
        lambda g: (
            (S_MATRIX, _metric_rank(g[1]), _VIEW["heatmap"]),
            _with_foundation_note(
                "Cell values are five-fold means; method columns are ordered best to worst."
            ),
        ),
    ))
    R.append((
        rf"^(pd|lgd)_bar_mean_({M})$",
        lambda g: (
            (S_MATRIX, _metric_rank(g[1]), _VIEW["bar_mean"]),
            _with_foundation_note(
                "Bars are ordered best to worst; whiskers show pooled fold-level standard deviations."
            ),
        ),
    ))
    R.append((
        rf"^(pd|lgd)_box_({M})$",
        lambda g: (
            (S_MATRIX, _metric_rank(g[1]), _VIEW["box"]),
            _with_foundation_note(
                "Boxes summarize fold-level scores, and overlaid points show dataset means."
            ),
        ),
    ))
    R.append((
        rf"^(pd|lgd)_rank_matrix_({M})$",
        lambda g: (
            (S_MATRIX, _metric_rank(g[1]), _VIEW["rank_matrix"]),
            _with_foundation_note(
                "Lower ranks are better; columns follow average rank across datasets."
            ),
        ),
    ))
    R.append((
        rf"^(pd|lgd)_ranking_({M})$",
        lambda g: (
            (S_MATRIX, _metric_rank(g[1]), _VIEW["ranking"]),
            _with_foundation_note(
                "Lower ranks are better; whiskers show the standard deviation of per-dataset ranks."
            ),
        ),
    ))
    R.append((
        rf"^(pd|lgd)_rank_box_({M})$",
        lambda g: (
            (S_MATRIX, _metric_rank(g[1]), _VIEW["rank_box"]),
            _with_foundation_note(
                "Lower ranks are better; boxes show the distribution of ranks across datasets."
            ),
        ),
    ))

    # ---- HPO effect ----
    R.append((
        rf"^(pd|lgd)_hpo_effect_({M})$",
        lambda g: (
            (S_HPO, _metric_rank(g[1]), 0),
            _with_foundation_note(
                "Bars show tuning improvement relative to the untuned run; positive values favour HPO."
            ),
        ),
    ))

    # ---- compute time ----
    R.append((
        r"^(pd|lgd)_bar_compute_time$",
        lambda g: (
            (S_COMPUTE, 0, 0),
            _with_foundation_note(
                "Bars are ordered by mean runtime; HPO runs include search cost when applicable."
            ),
        ),
    ))
    R.append((
        r"^(pd|lgd)_box_compute_time$",
        lambda g: (
            (S_COMPUTE, 1, 0),
            _with_foundation_note(
                "Each point is one dataset-fold run; boxes summarize total fit and prediction time."
            ),
        ),
    ))

    # ---- cost / quality ----
    R.append((
        rf"^(pd|lgd)_cost_quality_({M})$",
        lambda g: (
            (S_COSTQ, _metric_rank(g[1]), 0),
            "Foundation models use red star markers; labels are limited to selected methods to reduce clutter.",
        ),
    ))

    # ---- foundation-vs-baseline head-to-head ----
    R.append((
        rf"^(pd|lgd)_(.+?)_vs_(.+?)_sizetrend_({M})$",
        lambda g: ((S_HEAD, _metric_rank(g[3]), 0), _size_trend_caption(g)),
    ))
    R.append((
        rf"^(pd|lgd)_(.+?)_vs_(.+?)_imbalancetrend_({M})$",
        lambda g: (
            (S_HEAD, _metric_rank(g[3]), 1),
            (
                f"Each point is a dataset. The horizontal axis is the processed minority-class "
                f"proportion, so lower values indicate stronger imbalance; green points favour "
                f"{_method(g[1])}, red points favour {_method(g[2])}, and the solid line is an "
                f"ordinary least-squares trend."
            ),
        ),
    ))
    R.append((
        rf"^(pd|lgd)_(.+?)_vs_(.+?)_scatter_({M})$",
        lambda g: (
            (S_HEAD, _metric_rank(g[3]), 2),
            (
                f"Each point is a dataset; the diagonal marks equal performance, with green points "
                f"favouring {_method(g[1])} and red points favouring {_method(g[2])}."
            ),
        ),
    ))

    # ---- selected-method calibration summary ----
    R.append((
        r"^(pd|lgd)_selected_calibration_summary$",
        lambda g: (
            (S_CAL, 0, g[0]),
            "The top panels compare equal-dataset macro means and signed differences, with "
            "one-standard-deviation whiskers. The bottom panels show the corresponding "
            "dataset-level distributions. Positive observed-minus-predicted values indicate "
            "underprediction.",
        ),
    ))

    # ---- pooled learning / imbalance curves ----
    R.append((
        rf"^(pd|lgd)_(learning_curve|imbalance_curve)_({M})_combined(_less|_more)?$",
        lambda g: (
            (
                S_CURVE,
                0 if g[1] == "learning_curve" else 1,
                _metric_rank(g[2]),
                _curve_combined_variant_rank(g),
            ),
            _curve_combined_caption(g),
        ),
    ))
    R.append((
        rf"^(pd|lgd)_(learning_curve|imbalance_curve)_({M})(_relative)?(_smooth(?:_less|_more)?)?(_zoom)?(_logx)?$",
        lambda g: (
            (
                S_CURVE,
                0 if g[1] == "learning_curve" else 1,
                _metric_rank(g[2]),
                _curve_variant_rank(g),
            ),
            _curve_caption(g),
        ),
    ))

    # ---- per-dataset raw-point curves ----
    R.append((
        rf"^(pd|lgd)_row_limit_(.+)_({M})$",
        lambda g: (
            (S_PERDS, 0, _dataset(g[1])),
            "Dataset-size sweep within one dataset; no cross-dataset averaging or smoothing is applied.",
        ),
    ))
    R.append((
        rf"^(pd|lgd)_minority_proportion_(.+)_({M})$",
        lambda g: (
            (S_PERDS, 1, _dataset(g[1])),
            "Minority-class proportion is varied within one dataset; no cross-dataset averaging or smoothing is applied.",
        ),
    ))

    # ---- statistical figures ----
    R.append((
        r"^(pd|lgd)_pama$",
        lambda g: (
            (S_STAT, 0, 0),
            _with_foundation_note(
                "Bars are ordered by PAMA; labels report percentages and winning-fold counts."
            ),
        ),
    ))
    R.append((
        r"^(pd|lgd)_pama_min2wins$",
        lambda g: (
            (S_STAT, 0, 1),
            _with_foundation_note(
                "Only methods with at least two winning fold-level observations are shown."
            ),
        ),
    ))
    R.append((
        rf"^(pd|lgd)_cd_({M})$",
        lambda g: (
            (S_STAT, 1, _metric_rank(g[1])),
            _with_foundation_note(
                "Lower ranks are better; horizontal bars connect methods not separated by the Nemenyi test."
            ),
        ),
    ))
    R.append((
        r"^(pd|lgd)_winloss$",
        lambda g: (
            (S_STAT, 2, 0),
            _with_foundation_note(
                "Cells compare row methods against column methods; color encodes the win-loss margin."
            ),
        ),
    ))
    R.append((
        r"^(pd|lgd)_significance$",
        lambda g: (
            (S_STAT, 3, 0),
            _with_foundation_note(
                r"Cells contain adjusted \(p\)-values; green cells mark significant pairwise differences."
            ),
        ),
    ))

    return [(re.compile(p), b) for p, b in R]


_LEARNING_XAXIS = "dataset size"
_SMOOTH_CAPTION = {
    "less": "Moving-average lines use the shorter window.",
    "standard": "Moving-average lines use the standard window.",
    "more": "Moving-average lines use the longer window.",
}
_COMBINED_SMOOTH_CAPTION = {
    "less": "solid lines are centered moving averages using the shorter window.",
    "standard": "solid lines are centered moving averages using the standard window.",
    "more": "solid lines are centered moving averages using the longer window.",
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
    notes = ["Curves average scores across datasets at each sweep value."]
    if g[3]:
        notes.append("Values are normalized to each method's best observed value over the sweep.")
    smooth_variant = _smooth_variant(g[4])
    if smooth_variant:
        notes.append(_SMOOTH_CAPTION.get(smooth_variant, _SMOOTH_CAPTION["standard"]))
    else:
        notes.append("Markers show the unsmoothed sweep values.")
    if g[5]:
        if learning:
            notes.append("The inset repeats the low-data end of the sweep.")
        else:
            notes.append("The inset repeats the severe-imbalance end of the sweep.")
    if len(g) > 6 and g[6]:
        notes.append("The horizontal axis uses a logarithmic scale.")
    return " ".join(notes)


def _curve_combined_caption(g) -> str:
    smooth_variant = _smooth_variant(g[3]) or "standard"
    trend = _COMBINED_SMOOTH_CAPTION.get(
        smooth_variant, _COMBINED_SMOOTH_CAPTION["standard"]
    )
    return f"Transparent points show pooled score estimates; {trend}"


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
    """Return ``(sort_key, caption_body)`` for a figure stem (no extension)."""
    for pat, builder in _RULES:
        m = pat.match(stem)
        if m:
            return builder(m.groups())
    return ((S_FALLBACK, stem), f"Caption for \\texttt{{{_latex_escape(stem)}}}.")


# ---------------------------------------------------------------------------
#  Consolidated CAPTIONS.md (one file, chapters in notebook order)
# ---------------------------------------------------------------------------
# Each chapter mirrors one analysis notebook, in the same order the notebooks
# appear in notebooks/. The second item is the figure sub-directory that
# notebook writes to under figures/.
CHAPTERS: List[Tuple[str, str]] = [
    ("Data exploration", "data_exploration"),
    ("Experiment 0 - pilot coverage", "experiment0"),
    ("Experiment 1.1 - PD benchmark", "experiment1/pd"),
    ("Experiment 1.2 - PD statistical analysis", "experiment1/pd_stats"),
    ("Experiment 1.3 - PD champion-level statistics", "experiment1/pd_family"),
    ("Experiment 1.4 - LGD benchmark", "experiment1/lgd"),
    ("Experiment 1.5 - LGD statistical analysis", "experiment1/lgd_stats"),
    ("Experiment 1.6 - LGD champion-level statistics", "experiment1/lgd_family"),
    ("Experiment 2.1 - PD data-efficiency sweep", "experiment2/pd"),
    ("Experiment 2.2 - LGD data-efficiency sweep", "experiment2/lgd"),
    ("Experiment 3 - imbalance-robustness sweep", "experiment3"),
]


def _ordered_stems(d: Path) -> List[str]:
    """The ``.pdf`` stems in directory ``d`` in notebook generation order."""
    pdfs = [f.stem for f in d.glob("*.pdf")]
    return [s for _key, s in sorted((caption_for(s)[0], s) for s in pdfs)]


def generate_captions(
    figures_root: Path,
    experiments: Optional[Sequence[str]] = None,
    dry_run: bool = False,
) -> List[Path]:
    """Write a single root-level ``figures/CAPTIONS.md`` file."""
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
        rel_prefix = "/".join(d.relative_to(figures_root).parts)
        for stem in stems:
            label_stem = f"{rel_prefix}/{stem}" if rel_prefix else stem
            out.append(
                f"**`{stem}.pdf`**\n```latex\n"
                f"{latex_caption_block(stem, label_stem=label_stem)}\n```\n"
            )
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
        "LaTeX-ready caption snippets for generated figures, grouped by notebook. "
        "Each block belongs to the PDF named above it.\n"
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
    "latex_caption_block",
    "generate_captions",
    "refresh_captions_for_saved_figure",
    "main",
    "CHAPTERS",
    "METRIC_DISPLAY",
]


if __name__ == "__main__":
    raise SystemExit(main())
