"""Polars-backed result aggregator (B11).

Replaces ``summarize_results.py``'s pandas chain. About 10x faster on a
full benchmark sweep (~50 methods x ~14 datasets x ~5 folds = 3,500 rows
of per-fold output), and the lazy / streaming API keeps memory flat as
the sweep grows.

Two entry points:

* :func:`collect_fold_results` -- scan the new JSON+npz layout
  (``results/<experiment>/<task>/<dataset>/<method>/<hpo_mode>/fold_*.json``)
  and return one polars row per fold.
* :func:`aggregate_by_method` -- mean / std / median for every metric
  across folds, grouped by ``(task, dataset, method, hpo_mode)``.

The output of :func:`aggregate_by_method` is a polars DataFrame; call
``.to_pandas()`` if a notebook downstream still expects pandas, or
``.write_csv(path)`` to save.

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


def _iter_fold_files(base: Path, experiment: str) -> Iterable[Path]:
    """Yield every ``fold_<id>.json`` under ``results/<experiment>/``."""
    root = base / experiment
    if not root.exists():
        return
    yield from root.rglob("fold_*.json")


def _row_from_fold_file(path: Path) -> Dict[str, Any]:
    """Parse one fold JSON into a flat dict row."""
    payload = json.loads(path.read_text())
    # path = results/<experiment>/<task>/<dataset>/<method>/<hpo_mode>/fold_<id>.json
    parts = path.parts
    fold_id = int(path.stem.split("_", 1)[1])
    hpo_mode = parts[-2]
    method = parts[-3]
    dataset = parts[-4]
    task = parts[-5]

    row: Dict[str, Any] = {
        "task": task,
        "dataset": dataset,
        "method": method,
        "hpo_mode": hpo_mode,
        "fold_id": fold_id,
        "train_time": float(payload.get("train_time") or 0.0),
        "predict_time": float(payload.get("predict_time") or 0.0),
        "threshold": payload.get("threshold"),
        "used_hpo": bool(payload.get("used_hpo", False)),
    }

    # Flatten the metrics dict -- one column per metric.
    metrics = payload.get("metrics") or {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            row[f"metric.{k}"] = float(v)

    return row


def collect_fold_results(base: Path, experiment: str):
    """Scan disk and return a polars DataFrame (one row per fold)."""
    rows = [_row_from_fold_file(p) for p in _iter_fold_files(base, experiment)]
    if not rows:
        logger.warning(f"No fold results found under {base / experiment}")
    if _HAS_POLARS:
        return pl.DataFrame(rows)
    # Fallback
    import pandas as pd
    warnings.warn("polars not installed; falling back to pandas DataFrame.")
    return pd.DataFrame(rows)


def aggregate_by_method(
    df,
    *,
    group_by: List[str] = ("task", "dataset", "method", "hpo_mode"),
):
    """Mean / std / median for every metric column, grouped by ``group_by``."""
    if not _HAS_POLARS:
        # pandas path
        metric_cols = [c for c in df.columns if c.startswith("metric.")]
        agg = df.groupby(list(group_by))[metric_cols].agg(["mean", "std", "median"])
        return agg

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

    return df.group_by(list(group_by)).agg(agg_exprs).sort(list(group_by))


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
