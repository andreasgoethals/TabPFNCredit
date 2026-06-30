# src/data/preprocessing.py
"""
Preprocess and cache TALENT-compatible datasets.

Delegates dataset-specific cleaning to dataset_preprocessing.py.

- Checks if preprocessed data already exists under data/processed/{task}/{dataset}.
- If cached → loads it.
- If not → calls dataset_preprocessing.py for dataset-specific logic,
then performs standard cleaning, caching, and TALENT-format conversion.

Outputs (unsplit):
    N: np.ndarray or None  -> numerical features
    C: np.ndarray or None  -> categorical features
    y: np.ndarray          -> target
    info: dict             -> metadata

No CV or train/val/test splitting here — handled later.
Statistical preprocessing (PCA, outlier removal, constant columns) happens
after splitting in data_feeder.py to prevent data leakage.
"""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from src.utils.paths import find_processed_dir, processed_write_dir

try:
    from src.data.dataset_preprocessing import preprocess_dataset_specific
except ModuleNotFoundError as exc:
    _PRIVATE_PREPROCESSING_IMPORT_ERROR = exc

    def preprocess_dataset_specific(*_args, **_kwargs):
        raise FileNotFoundError(
            "Missing private preprocessing module: src/data/dataset_preprocessing.py. "
            "It is intentionally gitignored because it contains proprietary raw-dataset "
            "schema and cleaning rules. Restore your local copy before preprocessing "
            "raw datasets; already processed datasets can still be loaded."
        ) from _PRIVATE_PREPROCESSING_IMPORT_ERROR

logger = logging.getLogger(__name__)
pd.set_option("future.no_silent_downcasting", True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_ROOT / "data" / "processed"


def _load_or_preprocess(task: str, dataset: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, dict]:
    """
    Load cached dataset if available; otherwise preprocess raw data and cache it.

    Returns
    -------
    N : np.ndarray or None
        Numerical features (float32)
    C : np.ndarray or None
        Categorical features (int64, index-encoded)
    y : np.ndarray
        Target variable
    info : dict
        Metadata about dataset
    """

    # NOTE: do NOT create the directory eagerly. If preprocessing fails
    # (e.g. wrong task for a dataset), the empty dir would survive and
    # confuse `list_processed_datasets` into reporting a non-existent
    # dataset. We only create the directory just before writing files.
    # ----------------------------------------------------------
    # 1. Load cached version if available
    #    (repo-local data/ first, then shared project storage)
    # ----------------------------------------------------------
    cached_dir = find_processed_dir(task, dataset)
    if cached_dir is not None:
        logger.info(f"  Using cached dataset: {cached_dir.name}")
        N = np.load(cached_dir / "N.npy") if (cached_dir / "N.npy").exists() else None
        C = np.load(cached_dir / "C.npy") if (cached_dir / "C.npy").exists() else None
        y = np.load(cached_dir / "y.npy")
        with open(cached_dir / "info.json") as f:
            info = json.load(f)
        return N, C, y, info

    # ----------------------------------------------------------
    # 2. Preprocess from raw
    # ----------------------------------------------------------
    logger.info(f"  Preprocessing {dataset} ({task}) from raw files...")

    # Delegate dataset-specific cleaning (raises FileNotFoundError if the
    # raw file doesn't exist for this task; we deliberately let that
    # propagate so callers know the dataset is misconfigured).
    df, target_col, cat_cols, num_cols = preprocess_dataset_specific(task, dataset, raw_dir=None)

    # Preprocessing succeeded -- write next to wherever the raw file lives
    # (data staged on project storage caches there too, never filling the
    # small general data storage). Safe to create the destination now.
    dataset_dir = processed_write_dir(task, dataset)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Extract target and features
    # ----------------------------------------------------------
    y = df[target_col].to_numpy()
    X = df.drop(columns=[target_col])

    # Separate numeric and categorical features
    N = X[num_cols].to_numpy(dtype=np.float32) if num_cols else None
    if cat_cols:
        C = (
            X[cat_cols]
            .astype("category")
            .apply(lambda s: s.cat.codes)
            .to_numpy(dtype=np.int64)
        )
    else:
        C = None

    # ----------------------------------------------------------
    # 3. Build metadata dictionary
    # ----------------------------------------------------------
    info = {
        "dataset_name": dataset,
        "task_type": "regression" if task == "lgd" else "binclass",
        "n_samples": len(y),
        "n_num_features": N.shape[1] if N is not None else 0,
        "n_cat_features": C.shape[1] if C is not None else 0,
        "numerical_cols": num_cols,
        "categorical_cols": cat_cols,
    }

    # ----------------------------------------------------------
    # 4. Cache arrays in standard TALENT format.
    #    Writes are ATOMIC (temp file + os.replace) and y.npy is written LAST,
    #    because on the cluster several array tasks may preprocess the SAME
    #    not-yet-staged dataset (e.g. 0014.algorithmwatch) concurrently on
    #    different compute nodes that share $VSC_DATA. Atomic replace means a
    #    concurrent reader never sees a half-written file; writing y.npy last
    #    means the "already cached?" gate above only fires once N/C/info are
    #    fully in place. Preprocessing is deterministic, so identical inputs
    #    produce identical bytes and a last-writer-wins race is harmless.
    # ----------------------------------------------------------
    def _atomic_np_save(name: str, arr) -> None:
        tmp = dataset_dir / f".{name}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
        os.replace(tmp, dataset_dir / name)

    if N is not None:
        _atomic_np_save("N.npy", N)
    if C is not None:
        _atomic_np_save("C.npy", C)
    info_tmp = dataset_dir / f".info.json.{os.getpid()}.tmp"
    with open(info_tmp, "w") as f:
        json.dump(info, f, indent=4)
    os.replace(info_tmp, dataset_dir / "info.json")
    _atomic_np_save("y.npy", y)  # LAST -- gates the cached-load check above

    logger.info(f"  Cached preprocessed dataset: {dataset_dir.name}")
    return N, C, y, info


def preprocess_dataset(task: str, dataset: str):
    """
    Entry point: preprocess or load dataset and return unsplit TALENT Level-0 arrays.
    
    No cross-validation or splitting here.
    No statistical preprocessing (PCA, outlier removal, constant columns) - 
    those operations happen after splitting in data_feeder.py to prevent leakage.
    """
    if task not in {"pd", "lgd"}:
        raise ValueError("Task must be 'pd' or 'lgd'.")

    N, C, y, info = _load_or_preprocess(task, dataset)

    return N, C, y, info
