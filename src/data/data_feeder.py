# src/data/data_feeder.py
"""
DataFeeder — unified data preparation and split manager for TALENT.

This version integrates directly with src.data.preprocessing.preprocess_dataset()
and outputs data in the format that TALENT's Method.fit() and Method.predict()
expect: three dictionaries (N, C, y), each containing 'train', 'val', and 'test' keys.

-------------------------------------------------------------------------------
INPUTS
-------------------------------------------------------------------------------
You only need to provide:
    - task: "pd" (classification) or "lgd" (regression)
    - dataset: dataset name (e.g. "0014.hmeq")
    - test_size: fraction of data reserved for testing (only used when CV_splits =1)
    - val_size: fraction of remaining training data (without test set) reserved for validation 
    - cv_splits: number of cross-validation folds (1 = single split)
    - seed: random seed (for reproducibility)
    - row_limit: (optional) limit number of rows for debugging
    - sampling: (optional, float ∈ (0,1)) desired minority-class proportion (PD only)

-------------------------------------------------------------------------------
OUTPUTS
-------------------------------------------------------------------------------
Returns a dictionary:
    {
        fold_id: ((N, C, y), info)
    }

Each (N, C, y) is a dict in TALENT's expected format:

    N = {'train': X_num_train, 'val': X_num_val, 'test': X_num_test} or None
    C = {'train': X_cat_train, 'val': X_cat_val, 'test': X_cat_test} or None
    y = {'train': y_train, 'val': y_val, 'test': y_test}
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from src.data.preprocessing import preprocess_dataset


class DataFeeder:
    """
    TALENT-compatible data loader and splitter.
    Handles:
        - loading or preprocessing datasets,
        - optional row limiting,
        - optional imbalance resampling (PD only),
        - single or multi-fold splitting (train/val/test),
        - and returns data formatted for TALENT.
    """

    def __init__(
        self,
        task: str,
        dataset: str,
        test_size: float,
        val_size: float,
        cv_splits: int,
        seed: int,
        row_limit: Optional[int] = None,
        sampling: Optional[float] = None,
    ):
        assert task in {"pd", "lgd"}, "task must be 'pd' or 'lgd'"
        self.task = task
        self.dataset = dataset
        self.test_size = test_size
        self.val_size = val_size
        self.cv_splits = cv_splits
        self.seed = seed
        self.row_limit = row_limit
        self.sampling = sampling
        self.task_type = "classification" if task == "pd" else "regression"

    # ----------------------------------------------------------
    # Optional sampling for class imbalance (binary PD only)
    # ----------------------------------------------------------
    def _apply_sampling(
        self,
        X_num: Optional[np.ndarray],
        y: np.ndarray,
        X_cat: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
        """
        Resample majority/minority classes to reach a desired minority proportion.
        Only used for PD (classification) tasks.
        """
        if self.task != "pd" or self.sampling is None:
            return X_num, y, X_cat

        unique, counts = np.unique(y, return_counts=True)
        if len(unique) != 2:
            return X_num, y, X_cat  # only binary supported

        minority, majority = unique[np.argmin(counts)], unique[np.argmax(counts)]
        idx_min, idx_maj = np.where(y == minority)[0], np.where(y == majority)[0]
        n_min, n_maj = len(idx_min), len(idx_maj)
        target_ratio = self.sampling
        current_ratio = n_min / (n_min + n_maj)
        rng = np.random.default_rng(self.seed)

        # undersample the larger class
        if current_ratio < target_ratio:
            desired_maj = int((1 - target_ratio) / target_ratio * n_min)
            idx_maj_new = rng.choice(idx_maj, size=desired_maj, replace=False)
            idx_min_new = idx_min
        elif current_ratio > target_ratio:
            desired_min = int(target_ratio / (1 - target_ratio) * n_maj)
            idx_min_new = rng.choice(idx_min, size=desired_min, replace=False)
            idx_maj_new = idx_maj
        else:
            idx_min_new, idx_maj_new = idx_min, idx_maj

        new_idx = np.concatenate([idx_min_new, idx_maj_new])
        rng.shuffle(new_idx)

        def safe_index(X):
            return X[new_idx] if X is not None else None

        return safe_index(X_num), y[new_idx], safe_index(X_cat)

    # ----------------------------------------------------------
    # Main entry point
    # ----------------------------------------------------------
    def prepare(self) -> Dict[int, Tuple[Tuple[dict, dict, dict], Dict]]:
        """
        Load/preprocess dataset, optionally sample, and generate TALENT-ready splits.

        Returns
        -------
        folds : dict
            Mapping fold_id → ((N, C, y), info)
        """
        # 1️⃣ Load or preprocess dataset
        N, C, y, info = preprocess_dataset(self.task, self.dataset)

        # 2️⃣ Optionally limit number of rows (useful for debugging)
        if self.row_limit is not None:
            N = N[: self.row_limit] if N is not None else None
            C = C[: self.row_limit] if C is not None else None
            y = y[: self.row_limit]

        stratify = self.task == "pd"
        folds: Dict[int, Tuple[Tuple[dict, dict, dict], Dict]] = {}

        # 3️⃣ Decide split strategy
        if self.cv_splits == 1:
            # --- single train/val/test split ---
            idx_all = np.arange(len(y))
            idx_train_full, idx_test = train_test_split(
                idx_all,
                test_size=self.test_size,
                random_state=self.seed,
                stratify=y if stratify else None,
            )

            # extract data subsets
            Xn_train_full = N[idx_train_full] if N is not None else None
            Xc_train_full = C[idx_train_full] if C is not None else None
            y_train_full = y[idx_train_full]

            Xn_test = N[idx_test] if N is not None else None
            Xc_test = C[idx_test] if C is not None else None
            y_test = y[idx_test]

            # optional resampling
            Xn_train_full, y_train_full, Xc_train_full = self._apply_sampling(
                Xn_train_full, y_train_full, Xc_train_full
            )

            # validation split
            idx_train, idx_val = train_test_split(
                np.arange(len(y_train_full)),
                test_size=self.val_size,
                random_state=self.seed,
                stratify=y_train_full if stratify else None,
            )

            Xn_train = Xn_train_full[idx_train] if Xn_train_full is not None else None
            Xn_val = Xn_train_full[idx_val] if Xn_train_full is not None else None
            y_train = y_train_full[idx_train]
            y_val = y_train_full[idx_val]

            if Xc_train_full is not None:
                Xc_train = Xc_train_full[idx_train]
                Xc_val = Xc_train_full[idx_val]
            else:
                Xc_train = Xc_val = None

            # TALENT dicts
            N_dict = {"train": Xn_train, "val": Xn_val, "test": Xn_test} if N is not None else None
            C_dict = {"train": Xc_train, "val": Xc_val, "test": Xc_test} if C is not None else None
            y_dict = {"train": y_train, "val": y_val, "test": y_test}

            info_fold = {
                "task_type": self.task_type,
                "n_num_features": N.shape[1] if N is not None else 0,
                "n_cat_features": C.shape[1] if C is not None else 0,
            }

            folds[1] = ((N_dict, C_dict, y_dict), info_fold)
            return folds

        # 4️⃣ Cross-validation (KFold/StratifiedKFold)
        splitter = (
            StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.seed)
            if stratify
            else KFold(n_splits=self.cv_splits, shuffle=True, random_state=self.seed)
        )

        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(N, y), 1):
            # extract per-fold subsets
            Xn_train_full = N[train_idx] if N is not None else None
            Xc_train_full = C[train_idx] if C is not None else None
            y_train_full = y[train_idx]

            Xn_test = N[test_idx] if N is not None else None
            Xc_test = C[test_idx] if C is not None else None
            y_test = y[test_idx]

            # resampling on training portion
            Xn_train_full, y_train_full, Xc_train_full = self._apply_sampling(
                Xn_train_full, y_train_full, Xc_train_full
            )

            # validation split inside fold
            idx_train, idx_val = train_test_split(
                np.arange(len(y_train_full)),
                test_size=self.val_size,
                random_state=self.seed,
                stratify=y_train_full if stratify else None,
            )

            Xn_train = Xn_train_full[idx_train] if Xn_train_full is not None else None
            Xn_val = Xn_train_full[idx_val] if Xn_train_full is not None else None
            y_train = y_train_full[idx_train]
            y_val = y_train_full[idx_val]

            if Xc_train_full is not None:
                Xc_train = Xc_train_full[idx_train]
                Xc_val = Xc_train_full[idx_val]
            else:
                Xc_train = Xc_val = None

            # build TALENT dicts for this fold
            N_dict = {"train": Xn_train, "val": Xn_val, "test": Xn_test} if N is not None else None
            C_dict = {"train": Xc_train, "val": Xc_val, "test": Xc_test} if C is not None else None
            y_dict = {"train": y_train, "val": y_val, "test": y_test}

            info_fold = {
                "task_type": self.task_type,
                "n_num_features": N.shape[1] if N is not None else 0,
                "n_cat_features": C.shape[1] if C is not None else 0,
            }

            folds[fold_id] = ((N_dict, C_dict, y_dict), info_fold)

        return folds
