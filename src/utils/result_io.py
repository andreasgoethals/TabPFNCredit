"""JSON + .npy result storage (B1).

Replaces the bespoke pickle layout (``{hpo_mode: {method: {fold: {...}}}}``)
with a human-readable layout that lives **alongside** the existing pickles
(both are written; the old pickle path is preserved during migration).

Layout
------
::

    results/<experiment>/<task>/<dataset>/<method>/<hpo_mode>/
        fold_<id>.json        # metrics, threshold, fit_time, predict_time
        fold_<id>.npz         # y_true, y_prob, y_pred, val_y_true, val_y_prob
        summary.json          # aggregated metrics across all folds

* ``fold_<id>.json`` is small (<1 KB) and ``jq``-greppable.
* ``fold_<id>.npz`` is compressed; large arrays live here, not in JSON.
* ``summary.json`` aggregates after every fold completes (under lock).

Concurrent SLURM writers are serialised on ``summary.json`` via the
canonical :class:`~src.utils.file_lock.FileLock`. Individual ``fold_*``
files are write-once-per-fold so they need no locking.

The corresponding :func:`load_fold` round-trips back to the same dict
shape the legacy pickle code produced, so downstream code in
``summarize_results.py`` keeps working unchanged.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from src.utils.file_lock import FileLock

logger = logging.getLogger(__name__)


_ARRAY_KEYS = ("y_true", "y_prob", "y_pred", "val_y_true", "val_y_prob")
_SCALAR_KEYS = (
    "fold_id", "metrics", "train_time", "predict_time", "threshold",
    "used_hpo", "hpo_config", "hpo_n_trials", "info",
    "method", "dataset", "task",
    "n_clipped_below", "n_clipped_above",
)


def _fold_dir(
    base: Path,
    experiment: str,
    task: str,
    dataset: str,
    method: str,
    hpo_mode: str,
) -> Path:
    out = base / experiment / task.lower() / dataset / method / hpo_mode.upper()
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_fold(
    fold: Mapping[str, Any],
    *,
    base: Path,
    experiment: str,
    hpo_mode: str = "NO_HPO",
) -> None:
    """Persist one fold result.

    The fold dict shape is what :func:`src.methods.method_runner.run_talent_method`
    produces (one entry of the returned ``{fold_id: {...}}`` mapping).
    """
    out = _fold_dir(
        base=base,
        experiment=experiment,
        task=fold["task"],
        dataset=fold["dataset"],
        method=fold["method"],
        hpo_mode=hpo_mode,
    )
    fold_id = fold["fold_id"]

    # Arrays -> single compressed .npz
    arrays = {
        k: np.asarray(fold[k]) if fold.get(k) is not None else np.array([])
        for k in _ARRAY_KEYS
    }
    np.savez_compressed(out / f"fold_{fold_id}.npz", **arrays)

    # Scalars -> JSON. Strip numpy types so json.dump succeeds.
    scalars = {k: _to_jsonable(fold.get(k)) for k in _SCALAR_KEYS}
    (out / f"fold_{fold_id}.json").write_text(json.dumps(scalars, indent=2))


def load_fold(
    *,
    base: Path,
    experiment: str,
    task: str,
    dataset: str,
    method: str,
    fold_id: int,
    hpo_mode: str = "NO_HPO",
) -> Dict[str, Any]:
    """Load one fold back into the same dict shape that ``run_talent_method`` returns."""
    out = _fold_dir(base, experiment, task, dataset, method, hpo_mode)
    j = json.loads((out / f"fold_{fold_id}.json").read_text())
    npz = np.load(out / f"fold_{fold_id}.npz", allow_pickle=False)
    result = dict(j)
    for k in _ARRAY_KEYS:
        if k in npz.files:
            arr = npz[k]
            result[k] = arr if arr.size > 0 else None
    return result


def update_method_summary(
    method_results: Mapping[int, Mapping[str, Any]],
    *,
    base: Path,
    experiment: str,
    task: str,
    dataset: str,
    method: str,
    hpo_mode: str = "NO_HPO",
) -> None:
    """Aggregate ``{fold_id: result}`` into a ``summary.json`` under exclusive lock.

    Called after every ``run_talent_method`` invocation; under SLURM
    array concurrency two slots may be appending to the same summary for
    different folds of the same method, so the lock is essential.
    """
    out = _fold_dir(base, experiment, task, dataset, method, hpo_mode)
    summary_path = out / "summary.json"

    with FileLock(summary_path, exclusive=True) as f:
        f.seek(0)
        content = f.read()
        try:
            existing = json.loads(content) if content.strip() else {}
        except json.JSONDecodeError:
            existing = {}

        existing.setdefault("folds", {})
        for fold_id, fold in method_results.items():
            existing["folds"][str(fold_id)] = {
                "metrics": _to_jsonable(fold.get("metrics", {})),
                "train_time": float(fold.get("train_time", 0.0) or 0.0),
                "predict_time": float(fold.get("predict_time", 0.0) or 0.0),
                "threshold": (
                    float(fold["threshold"]) if fold.get("threshold") is not None else None
                ),
                "used_hpo": bool(fold.get("used_hpo", False)),
            }
        existing["method"] = method
        existing["dataset"] = dataset
        existing["task"] = task
        existing["hpo_mode"] = hpo_mode
        existing["n_folds"] = len(existing["folds"])

        # Aggregate mean / std over each metric
        metric_names: set = set()
        for fold_summary in existing["folds"].values():
            metric_names.update(fold_summary.get("metrics", {}).keys())
        aggregates: Dict[str, Dict[str, float]] = {}
        for name in sorted(metric_names):
            values = []
            for fold_summary in existing["folds"].values():
                v = fold_summary.get("metrics", {}).get(name)
                if v is None:
                    continue
                try:
                    fv = float(v)
                    if fv != fv:  # NaN
                        continue
                    values.append(fv)
                except (TypeError, ValueError):
                    continue
            if values:
                aggregates[name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "n": len(values),
                }
        existing["aggregates"] = aggregates

        f.seek(0)
        f.truncate()
        f.write(json.dumps(existing, indent=2))


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serialisable python types."""
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Mapping):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


__all__ = ["save_fold", "load_fold", "update_method_summary"]
