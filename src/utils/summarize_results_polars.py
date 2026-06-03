"""Polars-backed result aggregator.

Walks the JSON+npz layout produced by
:func:`src.utils.result_io.save_method` and emits two flat CSVs:

* ``<experiment>_per_fold.csv``   -- one row per (dataset, method, fold)
* ``<experiment>_per_method.csv`` -- one row per (dataset, method) with
  mean / std / median across folds.

Layout being scanned
--------------------
::

    <base>/<experiment>/<task>/<dataset>/<method>.json   # scalars + per-fold metrics
    <base>/<experiment>/<task>/<dataset>/<method>.npz    # arrays (not read here)

The ``method`` filename may carry a sweep suffix (``__HPO``,
``__row20000``, ``__min0p0025``) -- that is preserved as the ``method``
column and parsed out into ``sweep_axis`` / ``sweep_value`` columns when
present, so Experiment 1 (HPO sweep), Experiment 2 (row sweep), and
Experiment 3 (minority sweep) can all be summarised through one entry
point.

Falls back to pandas if ``polars`` is not installed (warns once).
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:  # pragma: no cover
    pl = None  # type: ignore
    _HAS_POLARS = False


def _iter_method_files(base: Path, experiment: str) -> Iterable[Path]:
    """Yield every ``<method>.json`` under ``<base>/<experiment>/<task>/<dataset>/``.

    The layout has exactly four path components below ``base``
    (``<experiment>/<task>/<dataset>/<method>.json``). Anything deeper
    (e.g. ``summaries/`` or ``.checkpoints/``) is ignored.
    """
    root = base / experiment
    if not root.exists():
        return
    for path in root.rglob("*.json"):
        rel = path.relative_to(base).parts
        # Expect: (<experiment>, <task>, <dataset>, "<method>.json")
        if len(rel) != 4:
            continue
        yield path


def _split_sweep_suffix(method_stem: str):
    """Parse ``"xgboost__HPO"`` into ``("xgboost", "HPO_mode", "HPO")``.

    Returns ``(bare_method, sweep_axis, sweep_value)``. ``sweep_axis`` /
    ``sweep_value`` are ``None`` for plain methods (no ``__`` suffix).
    """
    if "__" not in method_stem:
        return method_stem, None, None
    bare, suffix = method_stem.split("__", 1)
    # Heuristic mapping suffix-prefix -> sweep axis
    if suffix.startswith("row"):
        axis = "row_limit"
        try:
            value: Any = int(suffix[3:])
        except ValueError:
            value = suffix[3:]
    elif suffix.startswith("min"):
        axis = "minority_proportion"
        try:
            # "min0p0025" -> 0.0025
            value = float(suffix[3:].replace("p", "."))
        except ValueError:
            value = suffix[3:]
    elif suffix == "HPO":
        axis = "hpo_mode"
        value = "HPO"
    else:
        axis = "sweep"
        value = suffix
    return bare, axis, value


def _rows_from_method_file(path: Path, base: Path) -> Iterable[Dict[str, Any]]:
    """Yield one row per fold inside ``<method>.json``."""
    rel = path.relative_to(base).parts
    _experiment, task, dataset, method_file = rel
    method_stem = method_file[:-5]  # strip ".json"
    bare_method, sweep_axis, sweep_value = _split_sweep_suffix(method_stem)

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Skipping malformed JSON: %s", path)
        return

    # For plain methods (no sweep suffix) the HPO mode is implicit NO_HPO;
    # the ``__HPO`` suffix marks the tuned run.
    hpo_mode = "HPO" if sweep_axis == "hpo_mode" else "NO_HPO"

    folds = payload.get("folds") or {}
    for fold_id_str, fold in folds.items():
        try:
            fold_id = int(fold_id_str)
        except ValueError:
            continue
        row: Dict[str, Any] = {
            "task": task,
            "dataset": dataset,
            "method": bare_method,
            "method_full": method_stem,  # incl. sweep suffix
            "sweep_axis": sweep_axis,
            "sweep_value": sweep_value,
            "hpo_mode": hpo_mode,
            "fold_id": fold_id,
            "train_time": float(fold.get("train_time") or 0.0),
            "predict_time": float(fold.get("predict_time") or 0.0),
            "threshold": fold.get("threshold"),
            "used_hpo": bool(fold.get("used_hpo", False)),
        }
        metrics = fold.get("metrics") or {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                row[f"metric.{k}"] = float(v)
        yield row


def collect_fold_results(base: Path, experiment: str):
    """Scan disk and return a polars (or pandas) DataFrame, one row per fold."""
    rows: List[Dict[str, Any]] = []
    for path in _iter_method_files(base, experiment):
        rows.extend(_rows_from_method_file(path, base))
    if not rows:
        logger.warning("No fold results found under %s", base / experiment)
    if _HAS_POLARS:
        # ``infer_schema_length=None`` lets polars look at every row when
        # inferring nullable columns, so methods that emit a metric only
        # sometimes do not get silently dropped.
        return pl.DataFrame(rows, infer_schema_length=None)
    import pandas as pd
    warnings.warn("polars not installed; falling back to pandas DataFrame.")
    return pd.DataFrame(rows)


def aggregate_by_method(
    df,
    *,
    group_by: tuple = ("task", "dataset", "method", "hpo_mode", "sweep_axis", "sweep_value"),
):
    """Mean / std / median for every metric column, grouped by ``group_by``."""
    group_by_list = [g for g in group_by if g in df.columns]
    if not _HAS_POLARS:
        # pandas path
        metric_cols = [c for c in df.columns if c.startswith("metric.")]
        if not metric_cols:
            return df
        agg = df.groupby(group_by_list, dropna=False)[metric_cols].agg(["mean", "std", "median"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        return agg.reset_index()

    metric_cols = [c for c in df.columns if c.startswith("metric.")]
    if not metric_cols:
        return df

    agg_exprs = []
    for c in metric_cols:
        agg_exprs.append(pl.col(c).mean().alias(f"{c}_mean"))
        agg_exprs.append(pl.col(c).std().alias(f"{c}_std"))
        agg_exprs.append(pl.col(c).median().alias(f"{c}_median"))
    agg_exprs.append(pl.col("train_time").mean().alias("train_time_mean"))
    agg_exprs.append(pl.col("predict_time").mean().alias("predict_time_mean"))
    agg_exprs.append(pl.len().alias("n_folds"))

    return df.group_by(group_by_list).agg(agg_exprs).sort(group_by_list)


def summarize_to_csv(
    base: Path,
    experiment: str,
    out_dir: Path,
) -> List[Path]:
    """End-to-end: scan + aggregate + write two CSVs (per-fold, per-method).

    Returns the paths of the two written CSVs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = collect_fold_results(base, experiment)
    per_fold_path = out_dir / f"{experiment}_per_fold.csv"
    per_method_path = out_dir / f"{experiment}_per_method.csv"

    if _HAS_POLARS:
        df.write_csv(per_fold_path)
        aggregate_by_method(df).write_csv(per_method_path)
    else:
        df.to_csv(per_fold_path, index=False)
        aggregate_by_method(df).to_csv(per_method_path)

    return [per_fold_path, per_method_path]


__all__ = ["collect_fold_results", "aggregate_by_method", "summarize_to_csv"]
