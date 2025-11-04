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

No config reading, CV, or train/val/test splitting here — handled later.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from src.data.dataset_preprocessing import preprocess_dataset_specific

logger = logging.getLogger(__name__)
pd.set_option("future.no_silent_downcasting", True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
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

    # Ensure directory hierarchy exists: processed/{task}/{dataset}/
    dataset_dir = PROC_DIR / task / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # 1. Load cached version if available
    # ----------------------------------------------------------
    if (dataset_dir / "y.npy").exists():
        logger.info(f"📂 Using cached dataset: {dataset_dir}")
        N = np.load(dataset_dir / "N.npy") if (dataset_dir / "N.npy").exists() else None
        C = np.load(dataset_dir / "C.npy") if (dataset_dir / "C.npy").exists() else None
        y = np.load(dataset_dir / "y.npy")
        with open(dataset_dir / "info.json") as f:
            info = json.load(f)
        return N, C, y, info

    # ----------------------------------------------------------
    # 2. Preprocess from raw
    # ----------------------------------------------------------
    logger.info(f"🧪 Preprocessing {dataset} ({task}) from raw files...")

    # Delegate dataset-specific cleaning
    df, target_col, cat_cols, num_cols = preprocess_dataset_specific(task, dataset, RAW_DIR)

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
    # 3. Build metadata dictionary (extended but non-invasive)
    # ----------------------------------------------------------
    info = {
        "dataset_name": dataset,
        "task_type": "regression" if task == "lgd" else "classification",
        "n_samples": len(y),
        "n_num_features": N.shape[1] if N is not None else 0,
        "n_cat_features": C.shape[1] if C is not None else 0,
        "numerical_cols": num_cols,
        "categorical_cols": cat_cols,
    }

    # ----------------------------------------------------------
    # 4. Cache arrays in standard TALENT format
    # ----------------------------------------------------------
    if N is not None:
        np.save(dataset_dir / "N.npy", N)
    if C is not None:
        np.save(dataset_dir / "C.npy", C)
    np.save(dataset_dir / "y.npy", y)
    with open(dataset_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    logger.info(f"✅ Processed dataset cached at: {dataset_dir}")
    return N, C, y, info


def preprocess_dataset(task: str, dataset: str):
    """
    Entry point: preprocess or load dataset and return unsplit TALENT Level-0 arrays.
    No config reading, cross-validation, or splitting here.
    """
    if task not in {"pd", "lgd"}:
        raise ValueError("Task must be 'pd' or 'lgd'.")

    N, C, y, info = _load_or_preprocess(task, dataset)
    logger.info(f"Returning raw TALENT-ready arrays for {dataset} ({task})")
    return N, C, y, info
