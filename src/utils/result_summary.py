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
    """Parse a (possibly multi-axis) method stem into its sweep components.

    Delegates to :func:`src.utils.result_io.parse_method_name`, which is the
    canonical inverse of ``build_method_name`` and handles MULTIPLE suffixes
    (e.g. ``xgboost__HPO__row5000`` -> base ``xgboost`` + ``{HPO: True,
    row: 5000}``). Returns ``(bare_method, hpo_mode, sweep_axis, sweep_value)``
    where ``sweep_axis``/``sweep_value`` describe the first NON-HPO axis
    (None for a plain or HPO-only method), and ``hpo_mode`` is "HPO" or
    "NO_HPO".
    """
    from src.utils.result_io import parse_method_name

    parsed = parse_method_name(method_stem)
    base = parsed["method"]
    sweep = parsed["sweep"]  # {axis: value, ...}; HPO -> True

    hpo_mode = "HPO" if sweep.get("HPO") is True else "NO_HPO"
    # The primary (non-HPO) sweep axis, mapped to a readable name.
    _axis_names = {"row": "row_limit", "min": "minority_proportion"}
    sweep_axis = None
    sweep_value = None
    for axis, value in sweep.items():
        if axis == "HPO":
            continue
        sweep_axis = _axis_names.get(axis, axis)
        sweep_value = value
        break
    return base, hpo_mode, sweep_axis, sweep_value


def _rows_from_point(method_stem: str, point: Dict[str, Any], task: str, dataset: str
                     ) -> Iterable[Dict[str, Any]]:
    """Yield one row per fold for a single logical result (``method_stem`` + folds)."""
    bare_method, hpo_mode, sweep_axis, sweep_value = _split_sweep_suffix(method_stem)
    folds = point.get("folds") or {}
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


def _rows_from_method_file(path: Path, base: Path) -> Iterable[Dict[str, Any]]:
    """Yield one row per fold inside ``<method>.json``.

    Handles both layouts: a plain single result (top-level ``folds``) and a
    PACKED file (Experiment 2/3 -- many sweep points under ``points``), in
    which case each point's name carries its own sweep suffix.
    """
    rel = path.relative_to(base).parts
    _experiment, task, dataset, method_file = rel
    method_stem = method_file[:-5]  # strip ".json"

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Skipping malformed JSON: %s", path)
        return

    if isinstance(payload, dict) and "points" in payload:
        for point_name, entry in (payload.get("points") or {}).items():
            yield from _rows_from_point(point_name, entry, task, dataset)
    else:
        yield from _rows_from_point(method_stem, payload, task, dataset)


def collect_fold_results(base: Path, experiment: str):
    """Scan disk and return a polars (or pandas) DataFrame, one row per fold."""
    rows: List[Dict[str, Any]] = []
    for path in _iter_method_files(base, experiment):
        rows.extend(_rows_from_method_file(path, base))
    # A sweep point normally lives in exactly one packed shard, but replicated
    # runs (TABPFN_REPLICATE_PARTITIONS) or overlapping resubmissions can write
    # it to more than one shard file. Dedupe on (task, dataset, method_full,
    # fold) -- last writer wins -- so a duplicated point doesn't inflate fold
    # counts or skew the per-method aggregates.
    if rows:
        seen: set = set()
        deduped: List[Dict[str, Any]] = []
        for r in rows:
            key = (r.get("task"), r.get("dataset"), r.get("method_full"), r.get("fold_id"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        rows = deduped
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
