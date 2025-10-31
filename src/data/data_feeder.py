"""
data_feeder.py
=================
Creates TALENT-ready Train / Validation / Test folds with cross-validation.

Logic mirrors the original TabPFNCredit repository:
- First removes a test set (outer hold-out)
- Then applies KFold cross-validation on the remaining data
- Within each training fold, a further val_size fraction is reserved for validation

Outputs per fold:
    N_train, C_train, y_train
    N_val,   C_val,   y_val
    N_test,  C_test,  y_test
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import json
import logging

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_ROOT / "data" / "processed"


class DataFeeder:
    def __init__(self, task: str, dataset: str, split_config: Dict[str, Any]):
        """
        Parameters
        ----------
        task : str
            'pd' or 'lgd'
        dataset : str
            Dataset name (e.g., '0001.gmsc')
        split_config : dict
            Must contain:
                test_size, val_size, cv_splits, seed, row_limit
        """
        self.task = task
        self.dataset = dataset
        self.config = split_config
        self.dataset_dir = PROC_DIR / task / dataset

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Processed dataset not found: {self.dataset_dir}")

        self.N, self.C, self.y, self.info = self._load_arrays()
        self._apply_row_limit()

    # --------------------------------------------------------
    def _load_arrays(self):
        """Load TALENT-format arrays."""
        N = self._try_load("N.npy")
        C = self._try_load("C.npy")
        y = np.load(self.dataset_dir / "y.npy")
        with open(self.dataset_dir / "info.json") as f:
            info = json.load(f)
        return N, C, y, info

    def _try_load(self, name):
        path = self.dataset_dir / name
        return np.load(path) if path.exists() else None

    # --------------------------------------------------------
    def _apply_row_limit(self):
        """Subsample before splitting if requested."""
        limit = self.config.get("row_limit")
        seed = self.config.get("seed", 42)
        n = len(self.y)
        if limit is not None and limit < n:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, limit, replace=False)
            self.N = self.N[idx] if self.N is not None else None
            self.C = self.C[idx] if self.C is not None else None
            self.y = self.y[idx]
            logger.info(f"⚙️ Row limit applied: {limit}/{n} samples kept")

    # --------------------------------------------------------
    def make_splits(self) -> List[Dict[str, Any]]:
        """
        Construct train/val/test folds identical to the original CV logic.
        """
        test_size = self.config.get("test_size", 0.2)
        val_size = self.config.get("val_size", 0.2)
        cv_splits = self.config.get("cv_splits", 3)
        seed = self.config.get("seed", 42)

        n_samples = len(self.y)
        all_idx = np.arange(n_samples)

        # --- Outer test split
        trainval_idx, test_idx = train_test_split(
            all_idx, test_size=test_size, random_state=seed, shuffle=True
        )

        folds: List[Dict[str, Any]] = []

        if cv_splits <= 1:
            # Single train/val/test split
            train_idx, val_idx = train_test_split(
                trainval_idx, test_size=val_size, random_state=seed
            )
            folds.append(self._build_fold(train_idx, val_idx, test_idx))
        else:
            # KFold on training data (excluding test set)
            kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
            for fold_id, (train_idx_rel, holdout_idx_rel) in enumerate(kf.split(trainval_idx)):
                # Map relative indices back to full array indices
                train_idx_abs = trainval_idx[train_idx_rel]
                holdout_idx_abs = trainval_idx[holdout_idx_rel]

                # Split training portion further into train / val
                train_idx_abs, val_idx_abs = train_test_split(
                    train_idx_abs, test_size=val_size, random_state=seed
                )

                folds.append(self._build_fold(train_idx_abs, val_idx_abs, holdout_idx_abs))
                logger.info(f"🌀 Fold {fold_id+1}/{cv_splits}: "
                            f"{len(train_idx_abs)} train, {len(val_idx_abs)} val, {len(holdout_idx_abs)} test")

        logger.info(f"✅ Generated {len(folds)} folds for {self.dataset} ({self.task})")
        return folds

    # --------------------------------------------------------
    def _build_fold(self, train_idx, val_idx, test_idx) -> Dict[str, Any]:
        """Extract arrays for one fold."""
        def sub(a, idx): return a[idx] if a is not None else None

        return {
            "N_train": sub(self.N, train_idx),
            "C_train": sub(self.C, train_idx),
            "y_train": sub(self.y, train_idx),
            "N_val":   sub(self.N, val_idx),
            "C_val":   sub(self.C, val_idx),
            "y_val":   sub(self.y, val_idx),
            "N_test":  sub(self.N, test_idx),
            "C_test":  sub(self.C, test_idx),
            "y_test":  sub(self.y, test_idx),
        }
