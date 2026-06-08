"""Row-count inventory for the on-disk datasets.

Used by :func:`src.utils.config_reader.load_config` to resolve the
``min_rows`` shorthand in ``CONFIG_DATA.yaml`` into a concrete list of
dataset names. Used by the CLI to know which datasets need preprocessing
before an experiment can run.

The inventory always prefers the cached row count in
``data/processed/<task>/<dataset>/info.json`` (``n_samples``) when that
file exists -- one cheap JSON read per dataset. When the dataset has
not been preprocessed yet, we fall back to counting rows in the raw
file (CSV via line-count, parquet via pyarrow's row-group metadata so
we never load the actual data).

The result is intentionally a plain ``{dataset: n_rows}`` dict so
callers can do their own filtering / sorting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.utils.paths import (
    find_processed_dir,
    find_raw_path,
    processed_task_dirs,
    raw_task_dirs,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _count_rows_csv(path: Path) -> Optional[int]:
    """Count rows in a CSV (minus the header). Best-effort: returns None on error."""
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except OSError as exc:
        logger.warning("Could not count rows in %s: %s", path, exc)
        return None


def _count_rows_parquet(path: Path) -> Optional[int]:
    """Read parquet row count from metadata without loading the data."""
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception as exc:
        logger.warning("Could not read parquet row count from %s: %s", path, exc)
        return None


def _processed_row_count(task: str, dataset: str) -> Optional[int]:
    """Read ``n_samples`` from a processed ``info.json`` (repo or project storage)."""
    for proc_root in processed_task_dirs(task):
        info_path = proc_root / dataset / "info.json"
        if not info_path.exists():
            continue
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            return int(info.get("n_samples", 0)) or None
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Could not read %s: %s", info_path, exc)
            return None
    return None


def _list_raw_dataset_stems(task: str) -> List[str]:
    """Enumerate dataset stems (e.g. ``"0001.gmsc"``) from raw files.

    Looks at every ``data/raw/<task>`` root (repo-local first, then shared
    project storage) and returns the stem of every ``*.csv`` / ``*.parquet``.
    """
    stems: List[str] = []
    for task_dir in raw_task_dirs(task):
        if not task_dir.exists():
            continue
        for suffix in (".csv", ".parquet"):
            for path in sorted(task_dir.glob(f"*{suffix}")):
                stems.append(path.stem)
    return sorted(set(stems))


def list_datasets(task: str) -> List[str]:
    """Return every dataset stem for ``task`` (union of raw + processed)."""
    names = set(_list_raw_dataset_stems(task))
    for proc_dir in processed_task_dirs(task):
        if proc_dir.exists():
            for sub in proc_dir.iterdir():
                if sub.is_dir() and (sub / "y.npy").exists():
                    names.add(sub.name)
    return sorted(names)


def row_count(task: str, dataset: str) -> Optional[int]:
    """Best-effort row count for one ``(task, dataset)``.

    Tries the processed ``info.json`` first (cheapest), then falls back to
    counting rows in the raw CSV or parquet. Both lookups search the repo
    first, then the shared project storage.
    """
    cached = _processed_row_count(task, dataset)
    if cached is not None:
        return cached
    stem = find_raw_path(task, dataset)
    if stem is None:
        return None
    csv_path = stem.with_suffix(".csv")
    if csv_path.exists():
        return _count_rows_csv(csv_path)
    parquet_path = stem.with_suffix(".parquet")
    if parquet_path.exists():
        return _count_rows_parquet(parquet_path)
    return None


def row_counts(task: str) -> Dict[str, int]:
    """Return ``{dataset: n_rows}`` for every dataset of ``task``.

    Datasets where neither the processed cache nor the raw file are
    readable get silently dropped (a warning is logged via
    :func:`row_count`). Empty dict if the task directory does not exist.
    """
    counts: Dict[str, int] = {}
    for dataset in list_datasets(task):
        n = row_count(task, dataset)
        if n is not None:
            counts[dataset] = n
    return counts


def datasets_with_min_rows(task: str, min_rows: int) -> List[str]:
    """Return sorted dataset names with ``n_rows >= min_rows`` for ``task``."""
    return sorted(d for d, n in row_counts(task).items() if n >= min_rows)


# ---------------------------------------------------------------------------
#  Minority-class proportion (PD only) -- used by Experiment 3's dataset filter
# ---------------------------------------------------------------------------

def minority_proportion(task: str, dataset: str) -> Optional[float]:
    """Return the minority-class proportion ``min(p, 1 - p)`` for a PD dataset.

    ``p`` is the positive-class rate of the processed target ``y``. The
    minority proportion is the smaller of the two class rates, i.e. how
    rare the rarer class is.

    Only meaningful for ``task == "pd"`` (binary classification); returns
    ``None`` for LGD or on any failure. The processed ``y.npy`` is loaded
    directly; if it isn't cached yet the dataset is preprocessed on demand
    (Experiment 3 needs it regardless), so this can be a slow first call.
    """
    if task.lower() != "pd":
        return None
    try:
        proc_dir = find_processed_dir(task, dataset)
        if proc_dir is None:
            # Preprocess on demand -- the experiment will need the cache anyway.
            from src.data.preprocessing import preprocess_dataset
            preprocess_dataset(task, dataset)
            proc_dir = find_processed_dir(task, dataset)
        if proc_dir is None:
            return None
        y = np.load(proc_dir / "y.npy")
        y = np.asarray(y).astype(int).ravel()
        n = len(y)
        if n == 0:
            return None
        p = float((y == 1).sum()) / n
        return min(p, 1.0 - p)
    except Exception as exc:
        logger.warning("minority_proportion(%s, %s) failed: %s", task, dataset, exc)
        return None


def datasets_with_min_minority(
    task: str, datasets: List[str], min_minority: float
) -> List[str]:
    """Filter ``datasets`` to those whose minority proportion EXCEEDS ``min_minority``.

    Used by Experiment 3: you can only subsample the minority class
    *down* to reach a target imbalance, so a dataset whose natural
    minority proportion is already <= ``min_minority`` (the top of the
    sweep) can never participate and is dropped (with an INFO log).
    """
    keep: List[str] = []
    for d in datasets:
        mp = minority_proportion(task, d)
        if mp is None:
            logger.warning(
                "Excluding %s/%s from minority filter: could not determine "
                "its minority proportion.", task, d,
            )
            continue
        if mp > min_minority:
            keep.append(d)
        else:
            logger.info(
                "Excluding %s/%s: minority proportion %.4f <= max %.4f "
                "(cannot subsample UP to the sweep ceiling).",
                task, d, mp, min_minority,
            )
    return sorted(keep)


__all__ = [
    "list_datasets",
    "row_count",
    "row_counts",
    "datasets_with_min_rows",
    "minority_proportion",
    "datasets_with_min_minority",
]
