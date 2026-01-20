# src/data/data_feeder.py
"""
DataFeeder — unified data preparation and split manager for TALENT.

This version integrates directly with src.data.preprocessing.preprocess_dataset()
and outputs data in the format that TALENT's Method.fit() and Method.predict()
expect: three dictionaries (N, C, y), each containing 'train', 'val', and 'test' keys.

===============================================================================
POST-SPLIT PREPROCESSING (NO LEAKAGE)
===============================================================================
After splitting, the following operations are applied to each fold:
1. Drop near-constant columns (based on training data)
2. Remove extreme outliers (from training data only, using training statistics)
3. Apply PCA dimensionality reduction (fit on training, transform all splits)

These operations are CRITICAL to prevent data leakage and must happen after
train/test splitting, with separate fitting per fold.
===============================================================================

===============================================================================
ROW LIMIT HANDLING
===============================================================================
Two distinct types of row limits are supported:

1. USER/EXPERIMENT LIMIT (row_limit parameter):
   - Applied BEFORE splitting as a global dataset cap
   - Use for quick debugging/testing runs
   - Reduces both train AND test sets

2. METHOD-INTRINSIC LIMIT (train_row_limit parameter):
   - Applied AFTER splitting, only to TRAINING indices
   - Use for methods with architectural constraints (e.g., TabPFN: 10k)
   - Preserves full test/validation sets for robust evaluation
   - Maximizes training data utilization within method constraints

Example:
    - Dataset: 500k rows
    - Method limit: 10k (TabPFN)
    - CV splits: 5-fold

    OLD BEHAVIOR (incorrect): Cap at 10k → 8k train, 2k test per fold
    NEW BEHAVIOR (correct): Full 500k → 100k test, 10k train (subsampled) per fold
===============================================================================

-------------------------------------------------------------------------------
INPUTS
-------------------------------------------------------------------------------
You only need to provide:
    - task: "pd" (classification) or "lgd" (regression)
    - dataset: dataset name (e.g. "0014.hmeq")
    - test_size: fraction of data reserved for testing (only used when CV_splits=1)
    - val_size: fraction of remaining training data (without test set) reserved for validation
    - cv_splits: number of cross-validation folds (1 = single split)
    - seed: random seed (for reproducibility)
    - row_limit: (optional) global limit on rows BEFORE splitting (user/debugging limit)
    - train_row_limit: (optional) limit on training rows AFTER splitting (method constraint)
    - sampling: (optional, float ∈ (0,1)) desired minority-class proportion (PD only)
    - apply_pca: (optional, bool) whether to apply PCA for high-dimensional datasets
    - remove_outliers: (optional, bool) whether to remove extreme outliers

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

And each info = {
    'task_type': 'binclass' or 'regression',
    'n_num_features': int,
    'n_cat_features': int
    }
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.decomposition import PCA
from src.data.preprocessing import preprocess_dataset

# Setup logger
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION CONSTANTS (moved from dataset_preprocessing.py)
# =============================================================================

# PCA settings
MAX_FEATURES_THRESHOLD = 100  # Trigger PCA if total features exceed this
PCA_TARGET_FEATURES = 99       # Target number of PCA components

# Outlier detection settings (hybrid: percentile + magnitude)
OUTLIER_PERCENTILE_THRESHOLD = 0.001  # 0.1% tails (rare)
OUTLIER_MAGNITUDE_MULTIPLIER = 5.0    # Must be 5× IQR from median (extreme)
MIN_ROWS_FOR_OUTLIER_DETECTION = 100  # Need enough data for statistics

# Near-constant column threshold
NEAR_CONSTANT_THRESHOLD = 0.99  # Drop columns where >99% values are identical


class DataFeeder:
    """
    TALENT-compatible data loader and splitter.

    Handles:
        - loading or preprocessing datasets,
        - optional imbalance resampling (PD only) - applied to entire dataset first,
        - optional row limiting (global or training-only),
        - single or multi-fold splitting (train/val/test),
        - POST-SPLIT preprocessing (constant columns, outliers, PCA) per fold,
        - and returns data formatted for TALENT.

    Row Limit Types:
        - row_limit: Global cap applied BEFORE splitting (for debugging/quick runs)
        - train_row_limit: Training-only cap applied AFTER splitting (for method constraints)
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
        train_row_limit: Optional[int] = None,
        sampling: Optional[float] = None,
        apply_pca: bool = True,
        remove_outliers: bool = True,
    ):
        assert task in {"pd", "lgd"}, "task must be 'pd' or 'lgd'"
        self.task = task
        self.dataset = dataset
        self.test_size = test_size
        self.val_size = val_size
        self.cv_splits = cv_splits
        self.seed = seed
        self.row_limit = row_limit  # Global limit (before splitting)
        self.train_row_limit = train_row_limit  # Training-only limit (after splitting)
        self.sampling = sampling
        self.apply_pca = apply_pca
        self.remove_outliers = remove_outliers
        self.task_type = "binclass" if task == "pd" else "regression"

    # =========================================================================
    # PRE-SPLIT OPERATIONS
    # =========================================================================

    def _subsample_training_indices(
        self,
        train_idx: np.ndarray,
        y_train: np.ndarray,
        fold_id: int
    ) -> np.ndarray:
        """
        Subsample training indices to respect train_row_limit.

        This method is called AFTER splitting to apply method-intrinsic limits
        (e.g., TabPFN's 10k row limit) without affecting test/validation sets.

        For classification tasks (PD), uses stratified sampling to preserve
        class distribution in the subsampled training set.

        Args:
            train_idx: Full training indices for this fold
            y_train: Target values for training data (used for stratification)
            fold_id: Fold identifier (used for reproducible random seed)

        Returns:
            Subsampled training indices (or original if no limit needed)
        """
        if self.train_row_limit is None or len(train_idx) <= self.train_row_limit:
            return train_idx

        # Use fold-specific seed for reproducibility across runs
        rng = np.random.default_rng(self.seed + fold_id)

        original_size = len(train_idx)
        target_size = self.train_row_limit

        if self.task == "pd":
            # Stratified subsampling for classification
            # Preserve class distribution in the subsampled training set
            unique_classes, class_counts = np.unique(y_train, return_counts=True)
            class_proportions = class_counts / len(y_train)

            # Calculate target count per class
            target_counts = np.round(class_proportions * target_size).astype(int)

            # Adjust to ensure exact target size
            while target_counts.sum() > target_size:
                idx_max = np.argmax(target_counts)
                target_counts[idx_max] -= 1
            while target_counts.sum() < target_size:
                idx_min = np.argmin(target_counts)
                target_counts[idx_min] += 1

            # Sample from each class
            subsampled_idx = []
            for cls, count in zip(unique_classes, target_counts):
                cls_mask = y_train == cls
                cls_indices = np.where(cls_mask)[0]
                # Handle case where we need more samples than available
                n_to_sample = min(count, len(cls_indices))
                sampled = rng.choice(cls_indices, size=n_to_sample, replace=False)
                subsampled_idx.append(sampled)

            subsampled_local_idx = np.concatenate(subsampled_idx)
            rng.shuffle(subsampled_local_idx)

            # Map back to original indices
            subsampled_train_idx = train_idx[subsampled_local_idx]
        else:
            # Simple random subsampling for regression
            local_idx = rng.choice(len(train_idx), size=target_size, replace=False)
            subsampled_train_idx = train_idx[local_idx]

        logger.info(
            f"  Fold {fold_id}: Subsampled training set from {original_size:,} "
            f"to {len(subsampled_train_idx):,} rows (train_row_limit={self.train_row_limit:,})"
        )

        return subsampled_train_idx

    def _apply_sampling(
        self,
        X_num: Optional[np.ndarray],
        y: np.ndarray,
        X_cat: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
        """
        Resample majority/minority classes to reach a desired minority proportion.
        Only used for PD (classification) tasks.
        Applied to the entire dataset BEFORE splitting to ensure consistent
        class distributions across all folds.
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

    # =========================================================================
    # POST-SPLIT PREPROCESSING (NO LEAKAGE)
    # =========================================================================
    
    def _drop_near_constant_columns_post_split(
        self,
        N_train: Optional[np.ndarray],
        N_val: Optional[np.ndarray],
        N_test: Optional[np.ndarray],
        C_train: Optional[np.ndarray],
        C_val: Optional[np.ndarray],
        C_test: Optional[np.ndarray],
        threshold: float = NEAR_CONSTANT_THRESHOLD
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
               Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
               List[int], List[int]]:
        """
        Drop near-constant columns based on TRAINING data statistics.
        
        Identifies columns where a single value dominates (>threshold) in the
        training set, then drops those same columns from train/val/test.
        
        Returns
        -------
        N_train, N_val, N_test, C_train, C_val, C_test : arrays with columns dropped
        num_cols_to_keep : list of column indices to keep for numerical features
        cat_cols_to_keep : list of column indices to keep for categorical features
        """
        num_cols_to_drop = []
        cat_cols_to_drop = []
        
        # Check numerical columns
        if N_train is not None:
            for col_idx in range(N_train.shape[1]):
                col_data = N_train[:, col_idx]
                # Count non-NaN values
                valid_data = col_data[~np.isnan(col_data)]
                if len(valid_data) > 0:
                    unique_vals, counts = np.unique(valid_data, return_counts=True)
                    max_freq = counts.max() / len(valid_data)
                    if max_freq > threshold:
                        num_cols_to_drop.append(col_idx)
        
        # Check categorical columns
        if C_train is not None:
            for col_idx in range(C_train.shape[1]):
                col_data = C_train[:, col_idx]
                # Count valid values (assuming -1 represents missing)
                valid_data = col_data[col_data >= 0]
                if len(valid_data) > 0:
                    unique_vals, counts = np.unique(valid_data, return_counts=True)
                    max_freq = counts.max() / len(valid_data)
                    if max_freq > threshold:
                        cat_cols_to_drop.append(col_idx)
        
        # Drop identified columns from all splits
        if num_cols_to_drop:
            num_cols_to_keep = [i for i in range(N_train.shape[1]) if i not in num_cols_to_drop]
            N_train = N_train[:, num_cols_to_keep] if N_train is not None else None
            N_val = N_val[:, num_cols_to_keep] if N_val is not None else None
            N_test = N_test[:, num_cols_to_keep] if N_test is not None else None
            logger.info(f"    Dropped {len(num_cols_to_drop)} near-constant numerical columns")
        else:
            num_cols_to_keep = list(range(N_train.shape[1])) if N_train is not None else []
        
        if cat_cols_to_drop:
            cat_cols_to_keep = [i for i in range(C_train.shape[1]) if i not in cat_cols_to_drop]
            C_train = C_train[:, cat_cols_to_keep] if C_train is not None else None
            C_val = C_val[:, cat_cols_to_keep] if C_val is not None else None
            C_test = C_test[:, cat_cols_to_keep] if C_test is not None else None
            logger.info(f"    Dropped {len(cat_cols_to_drop)} near-constant categorical columns")
        else:
            cat_cols_to_keep = list(range(C_train.shape[1])) if C_train is not None else []
        
        return N_train, N_val, N_test, C_train, C_val, C_test, num_cols_to_keep, cat_cols_to_keep

    def _remove_outliers_post_split(
        self,
        N_train: Optional[np.ndarray],
        C_train: Optional[np.ndarray],  # ← ADD THIS
        y_train: np.ndarray,
        percentile_threshold: float = OUTLIER_PERCENTILE_THRESHOLD,
        magnitude_multiplier: float = OUTLIER_MAGNITUDE_MULTIPLIER,
        min_rows: int = MIN_ROWS_FOR_OUTLIER_DETECTION
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, int]:  # ← UPDATE RETURN TYPE
        """
        Remove extreme outliers from TRAINING data only, using training statistics.
        
        Validation and test sets are NOT modified (no rows removed).
        
        A value is flagged as an outlier if it meets BOTH conditions:
        1. RARE: In the extreme percentile tails (default: 0.1% / 99.9%)
        2. EXTREME: Very far from center (default: >5× IQR from median)
        
        Returns
        -------
        N_train : array with outlier rows removed
        C_train : array with outlier rows removed (same rows as N_train)
        y_train : target with outlier rows removed
        n_removed : number of rows removed
        """
        if N_train is None or len(N_train) < min_rows:
            return N_train, C_train, y_train, 0  # ← RETURN C_train
        
        outlier_mask = np.zeros(len(N_train), dtype=bool)
        
        for col_idx in range(N_train.shape[1]):
            col_data = N_train[:, col_idx]
            valid_mask = ~np.isnan(col_data)
            valid_data = col_data[valid_mask]
            
            if len(valid_data) < 20:
                continue
            
            # Calculate statistics from TRAINING data only
            median = np.median(valid_data)
            q1 = np.percentile(valid_data, 25)
            q3 = np.percentile(valid_data, 75)
            iqr = q3 - q1
            
            if iqr == 0:
                continue
            
            # CONDITION 1: Rarity (percentile-based)
            lower_pct = np.percentile(valid_data, percentile_threshold * 100)
            upper_pct = np.percentile(valid_data, (1 - percentile_threshold) * 100)
            is_rare = (col_data < lower_pct) | (col_data > upper_pct)
            
            # CONDITION 2: Extremity (magnitude-based)
            distance_from_median = np.abs(col_data - median)
            is_extreme = distance_from_median > magnitude_multiplier * iqr
            
            # BOTH conditions must be met
            col_outliers = is_rare & is_extreme & valid_mask
            outlier_mask = outlier_mask | col_outliers
        
        n_removed = outlier_mask.sum()
        
        if n_removed > 0:
            N_train = N_train[~outlier_mask]
            C_train = C_train[~outlier_mask] if C_train is not None else None  # ← FIX: Filter C_train too!
            y_train = y_train[~outlier_mask]
            pct_removed = n_removed / (n_removed + len(N_train)) * 100
            logger.info(
                f"    Removed {n_removed} outliers from training data "
                f"({pct_removed:.2f}%)"
            )
        
        return N_train, C_train, y_train, n_removed  # ← RETURN C_train

    def _apply_pca_post_split(
        self,
        N_train: Optional[np.ndarray],
        N_val: Optional[np.ndarray],
        N_test: Optional[np.ndarray],
        C_train: Optional[np.ndarray],
        C_val: Optional[np.ndarray],
        C_test: Optional[np.ndarray],
        target_features: int = PCA_TARGET_FEATURES,
        winsorize_limits: Tuple[float, float] = (0.01, 0.99)
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
               Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply PCA dimensionality reduction, fitted on TRAINING data only.
        
        Process:
        1. Combine numerical and categorical features (encode categoricals as integers)
        2. Impute missing values using TRAINING statistics
        3. Fit PCA on TRAINING data
        4. Transform train/val/test using the fitted PCA
        5. Winsorize PCA components using TRAINING percentiles
        
        Returns
        -------
        N_train, N_val, N_test : transformed numerical features (PCA components)
        C_train, C_val, C_test : None (categoricals absorbed into PCA)
        """
        # Determine total features
        n_num = N_train.shape[1] if N_train is not None else 0
        n_cat = C_train.shape[1] if C_train is not None else 0
        total_features = n_num + n_cat
        
        if total_features <= target_features:
            return N_train, N_val, N_test, C_train, C_val, C_test
        
        logger.info(f"    Applying PCA: {total_features} features → {target_features} components")
        
        # Combine numerical and categorical features
        def combine_features(N, C):
            if N is not None and C is not None:
                return np.concatenate([N, C.astype(np.float32)], axis=1)
            elif N is not None:
                return N
            elif C is not None:
                return C.astype(np.float32)
            return None
        
        X_train = combine_features(N_train, C_train)
        X_val = combine_features(N_val, C_val)
        X_test = combine_features(N_test, C_test)
        
        # Impute missing values using TRAINING statistics
        col_means = np.nanmean(X_train, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)  # Replace NaN means with 0
        
        def impute_with_train_stats(X, col_means):
            X_imputed = X.copy()
            for col_idx in range(X.shape[1]):
                nan_mask = np.isnan(X[:, col_idx])
                X_imputed[nan_mask, col_idx] = col_means[col_idx]
            return X_imputed
        
        X_train_imputed = impute_with_train_stats(X_train, col_means)
        X_val_imputed = impute_with_train_stats(X_val, col_means)
        X_test_imputed = impute_with_train_stats(X_test, col_means)
        
        # Determine number of components
        n_samples = len(X_train_imputed)
        n_features = X_train_imputed.shape[1]
        n_components = min(target_features, n_samples - 1, n_features - 1)
        n_components = max(1, n_components)
        
        # Fit PCA on TRAINING data only
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train_imputed)
        
        # Transform val and test using the fitted PCA
        X_val_pca = pca.transform(X_val_imputed)
        X_test_pca = pca.transform(X_test_imputed)
        
        # Winsorize PCA components using TRAINING percentiles
        n_winsorized = 0
        for col_idx in range(n_components):
            # Calculate percentiles from TRAINING data only
            lower_bound = np.percentile(X_train_pca[:, col_idx], winsorize_limits[0] * 100)
            upper_bound = np.percentile(X_train_pca[:, col_idx], winsorize_limits[1] * 100)
            
            # Count and clip values in train
            n_below = (X_train_pca[:, col_idx] < lower_bound).sum()
            n_above = (X_train_pca[:, col_idx] > upper_bound).sum()
            n_winsorized += n_below + n_above
            X_train_pca[:, col_idx] = np.clip(X_train_pca[:, col_idx], lower_bound, upper_bound)
            
            # Apply same bounds to val and test
            X_val_pca[:, col_idx] = np.clip(X_val_pca[:, col_idx], lower_bound, upper_bound)
            X_test_pca[:, col_idx] = np.clip(X_test_pca[:, col_idx], lower_bound, upper_bound)
        
        if n_winsorized > 0:
            pct_winsorized = n_winsorized / (len(X_train_pca) * n_components) * 100
            logger.info(
                f"    Winsorized {n_winsorized} PCA values ({pct_winsorized:.2f}%) "
                f"to [{winsorize_limits[0]*100:.1f}%, {winsorize_limits[1]*100:.1f}%] percentiles"
            )
        
        var_explained = pca.explained_variance_ratio_.sum() * 100
        logger.info(f"    PCA: {n_components} components, {var_explained:.1f}% variance explained")
        
        # Return PCA-transformed data (categoricals now absorbed into numerical)
        return X_train_pca, X_val_pca, X_test_pca, None, None, None

    def _apply_post_split_preprocessing(
        self,
        N_train: Optional[np.ndarray],
        N_val: Optional[np.ndarray],
        N_test: Optional[np.ndarray],
        C_train: Optional[np.ndarray],
        C_val: Optional[np.ndarray],
        C_test: Optional[np.ndarray],
        y_train: np.ndarray,
        fold_id: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
            Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
            np.ndarray]:
        """
        Apply all post-split preprocessing steps for a single fold.
        """
        logger.info(f"  Fold {fold_id}: Applying post-split preprocessing...")
        
        # Step 1: Drop near-constant columns
        N_train, N_val, N_test, C_train, C_val, C_test, _, _ = \
            self._drop_near_constant_columns_post_split(
                N_train, N_val, N_test, C_train, C_val, C_test
            )
        
        # Step 2: Remove extreme outliers (from training only)
        if self.remove_outliers and N_train is not None:
            N_train, C_train, y_train, _ = self._remove_outliers_post_split(
                N_train, C_train, y_train  # ← PASS C_train
            )
        
        # Step 3: Apply PCA if needed
        if self.apply_pca and N_train is not None:
            total_features = (N_train.shape[1] if N_train is not None else 0) + \
                        (C_train.shape[1] if C_train is not None else 0)
            
            if total_features > MAX_FEATURES_THRESHOLD:
                N_train, N_val, N_test, C_train, C_val, C_test = \
                    self._apply_pca_post_split(
                        N_train, N_val, N_test, C_train, C_val, C_test
                    )
            else:
                N_train, N_val, N_test = self._winsorize_features_post_split(
                    N_train, N_val, N_test
                )
        
        return N_train, N_val, N_test, C_train, C_val, C_test, y_train
    
    def _winsorize_features_post_split(
        self,
        N_train: Optional[np.ndarray],
        N_val: Optional[np.ndarray],
        N_test: Optional[np.ndarray],
        winsorize_limits: Tuple[float, float] = (0.001, 0.999)
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Winsorize extreme values in val/test sets using TRAINING percentiles.
        
        This protects against extreme outliers without removing test samples.
        Training set is NOT modified (outliers already removed separately).
        
        Parameters
        ----------
        winsorize_limits : tuple
            (lower, upper) percentiles for clipping (default: 1%, 99%)
        
        Returns
        -------
        N_train, N_val, N_test : arrays with val/test features clipped
        """
        if N_train is None:
            return N_train, N_val, N_test
        
        n_clipped_val = 0
        n_clipped_test = 0
        
        for col_idx in range(N_train.shape[1]):
            col_data_train = N_train[:, col_idx]
            valid_mask = ~np.isnan(col_data_train)
            valid_data = col_data_train[valid_mask]
            
            if len(valid_data) < 20:
                continue
            
            # Calculate percentiles from TRAINING data only
            lower_bound = np.percentile(valid_data, winsorize_limits[0] * 100)
            upper_bound = np.percentile(valid_data, winsorize_limits[1] * 100)
            
            # Clip val and test (NOT train - already had outliers removed)
            if N_val is not None:
                n_below = (N_val[:, col_idx] < lower_bound).sum()
                n_above = (N_val[:, col_idx] > upper_bound).sum()
                n_clipped_val += n_below + n_above
                N_val[:, col_idx] = np.clip(N_val[:, col_idx], lower_bound, upper_bound)
            
            if N_test is not None:
                n_below = (N_test[:, col_idx] < lower_bound).sum()
                n_above = (N_test[:, col_idx] > upper_bound).sum()
                n_clipped_test += n_below + n_above
                N_test[:, col_idx] = np.clip(N_test[:, col_idx], lower_bound, upper_bound)
        
        if n_clipped_val > 0 or n_clipped_test > 0:
            logger.info(
                f"    Winsorized features: {n_clipped_val} values in val, "
                f"{n_clipped_test} values in test "
                f"to [{winsorize_limits[0]*100:.1f}%, {winsorize_limits[1]*100:.1f}%] percentiles"
            )
        
        return N_train, N_val, N_test

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================
    
    def prepare(self) -> Dict[int, Tuple[Tuple[dict, dict, dict], Dict]]:
        """
        Load/preprocess dataset, optionally sample, split, and apply post-split preprocessing.

        Processing order:
        1. Load/preprocess dataset (dataset-specific cleaning only)
        2. Apply resampling to entire dataset (if requested)
        3. Apply global row_limit (if requested) - BEFORE splitting
        4. Split into folds (stratified CV preserves resampled distribution)
        5. FOR EACH FOLD:
           a. Apply train_row_limit to TRAINING indices only (method constraints)
           b. Apply post-split preprocessing (constant columns, outliers, PCA)

        Returns
        -------
        folds : dict
            Mapping fold_id → ((N, C, y), info)
        """
        logger.info(f"Preparing data: {self.dataset} ({self.task.upper()})")
        
        # 1️⃣ Load or preprocess dataset (NO statistical preprocessing yet)
        N, C, y, info = preprocess_dataset(self.task, self.dataset)

        # 2️⃣ Apply resampling FIRST to entire dataset (ensures consistent distribution across folds)
        N, y, C = self._apply_sampling(N, y, C)

        # 3️⃣ THEN optionally limit number of rows (applied to already-resampled data)
        if self.row_limit is not None:
            # Shuffle first to avoid bias (e.g., taking only oldest records if data is sorted)
            # We re-initialize the RNG here to ensure this shuffle is reproducible 
            # regardless of whether _apply_sampling() was executed previously.
            rng = np.random.default_rng(self.seed)
            perm_idx = rng.permutation(len(y))
            
            N = N[perm_idx] if N is not None else None
            C = C[perm_idx] if C is not None else None
            y = y[perm_idx]

            # Apply the row limit
            N = N[: self.row_limit] if N is not None else None
            C = C[: self.row_limit] if C is not None else None
            y = y[: self.row_limit]


        stratify = self.task == "pd"
        folds: Dict[int, Tuple[Tuple[dict, dict, dict], Dict]] = {}

        # 4️⃣ Decide split strategy
        if self.cv_splits == 1:
            # --- single train/val/test split ---
            idx_all = np.arange(len(y))
            idx_train_full, idx_test = train_test_split(
                idx_all,
                test_size=self.test_size,
                random_state=self.seed,
                stratify=y if stratify else None,
            )

            # Extract TEST data (never subsampled - full size for robust evaluation)
            Xn_test = N[idx_test] if N is not None else None
            Xc_test = C[idx_test] if C is not None else None
            y_test = y[idx_test]

            # Validation split from training data
            idx_train_local, idx_val_local = train_test_split(
                np.arange(len(idx_train_full)),
                test_size=self.val_size,
                random_state=self.seed,
                stratify=y[idx_train_full] if stratify else None,
            )

            # Map local indices back to global indices
            idx_train = idx_train_full[idx_train_local]
            idx_val = idx_train_full[idx_val_local]

            # Extract VALIDATION data (never subsampled - full size for threshold optimization)
            Xn_val = N[idx_val] if N is not None else None
            Xc_val = C[idx_val] if C is not None else None
            y_val = y[idx_val]

            # 5️⃣a Apply train_row_limit to TRAINING indices only (method constraints)
            # This preserves full test/val sets while respecting method's max training rows
            if self.train_row_limit is not None:
                idx_train = self._subsample_training_indices(
                    train_idx=idx_train,
                    y_train=y[idx_train],
                    fold_id=1
                )

            # Extract TRAINING data (potentially subsampled for method constraints)
            Xn_train = N[idx_train] if N is not None else None
            Xc_train = C[idx_train] if C is not None else None
            y_train = y[idx_train]

            # 5️⃣b Apply post-split preprocessing
            Xn_train, Xn_val, Xn_test, Xc_train, Xc_val, Xc_test, y_train = \
                self._apply_post_split_preprocessing(
                    Xn_train, Xn_val, Xn_test,
                    Xc_train, Xc_val, Xc_test,
                    y_train, fold_id=1
                )

            # Build TALENT dicts
            N_dict = {"train": Xn_train, "val": Xn_val, "test": Xn_test} if Xn_train is not None else None
            C_dict = {"train": Xc_train, "val": Xc_val, "test": Xc_test} if Xc_train is not None else None
            y_dict = {"train": y_train, "val": y_val, "test": y_test}

            info_fold = {
                "task_type": self.task_type,
                "n_num_features": Xn_train.shape[1] if Xn_train is not None else 0,
                "n_cat_features": Xc_train.shape[1] if Xc_train is not None else 0,
            }

            folds[1] = ((N_dict, C_dict, y_dict), info_fold)
            
            logger.info(f"Data prepared: {self.dataset} (1 fold)")
            
            return folds

        # 6️⃣ Cross-validation (KFold/StratifiedKFold)
        splitter = (
            StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.seed)
            if stratify
            else KFold(n_splits=self.cv_splits, shuffle=True, random_state=self.seed)
        )

        for fold_id, (train_idx_full, test_idx) in enumerate(splitter.split(N if N is not None else C, y), 1):
            # Extract TEST data (never subsampled - full size for robust evaluation)
            Xn_test = N[test_idx] if N is not None else None
            Xc_test = C[test_idx] if C is not None else None
            y_test = y[test_idx]

            # Validation split from training indices
            idx_train_local, idx_val_local = train_test_split(
                np.arange(len(train_idx_full)),
                test_size=self.val_size,
                random_state=self.seed,
                stratify=y[train_idx_full] if stratify else None,
            )

            # Map local indices back to global indices
            idx_train = train_idx_full[idx_train_local]
            idx_val = train_idx_full[idx_val_local]

            # Extract VALIDATION data (never subsampled - full size for threshold optimization)
            Xn_val = N[idx_val] if N is not None else None
            Xc_val = C[idx_val] if C is not None else None
            y_val = y[idx_val]

            # Apply train_row_limit to TRAINING indices only (method constraints)
            # This preserves full test/val sets while respecting method's max training rows
            if self.train_row_limit is not None:
                idx_train = self._subsample_training_indices(
                    train_idx=idx_train,
                    y_train=y[idx_train],
                    fold_id=fold_id
                )

            # Extract TRAINING data (potentially subsampled for method constraints)
            Xn_train = N[idx_train] if N is not None else None
            Xc_train = C[idx_train] if C is not None else None
            y_train = y[idx_train]

            # Apply post-split preprocessing for this fold
            Xn_train, Xn_val, Xn_test, Xc_train, Xc_val, Xc_test, y_train = \
                self._apply_post_split_preprocessing(
                    Xn_train, Xn_val, Xn_test,
                    Xc_train, Xc_val, Xc_test,
                    y_train, fold_id=fold_id
                )

            # Build TALENT dicts for this fold
            N_dict = {"train": Xn_train, "val": Xn_val, "test": Xn_test} if Xn_train is not None else None
            C_dict = {"train": Xc_train, "val": Xc_val, "test": Xc_test} if Xc_train is not None else None
            y_dict = {"train": y_train, "val": y_val, "test": y_test}

            info_fold = {
                "task_type": self.task_type,
                "n_num_features": Xn_train.shape[1] if Xn_train is not None else 0,
                "n_cat_features": Xc_train.shape[1] if Xc_train is not None else 0,
            }

            folds[fold_id] = ((N_dict, C_dict, y_dict), info_fold)

        logger.info(f"Data prepared: {self.dataset} ({self.cv_splits} folds)")

        return folds