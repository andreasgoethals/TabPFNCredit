"""
processed_data_exploration.py
-----------------------------------
Performs post-preprocessing quality checks on TALENT-formatted datasets.

Expected structure:
data/processed/{task}/{dataset}/
    ├── N.npy   (optional, numerical features)
    ├── C.npy   (optional, categorical features)
    ├── y.npy   (required, target)
    └── info.json  (metadata)

Focuses on detecting conditions TALENT cannot or should not handle:
    - Infinite values in N or C
    - Negative indices in C
    - NaN or inf in targets
    - Constant columns (zero variance)
    - Shape mismatches between arrays
    - Inconsistent metadata

Note:
TALENT can handle NaN values in features → these are NOT flagged.
"""

from __future__ import annotations
import numpy as np
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ProcessedDataInspector:
    """
    Inspector for TALENT-preprocessed datasets.

    Parameters
    ----------
    dataset_name : str
        Name of dataset (e.g., '0001.gmsc')
    task : str
        Either 'pd' or 'lgd'
    processed_root : str | Path
        Root directory of processed datasets (default: 'data/processed/')
    """

    def __init__(self, dataset_name: str, task: str, processed_root: str | Path = "data/processed/"):
        self.dataset_name = dataset_name
        self.task = task
        self.processed_root = Path(processed_root)
        self.dataset_dir = self.processed_root / task / dataset_name

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Processed dataset directory not found: {self.dataset_dir}")

        self.N = self._load_array("N.npy")
        self.C = self._load_array("C.npy")
        self.y = self._load_array("y.npy", required=True)
        self.info = self._load_info()

    # --------------------------------------------------------------
    def _load_array(self, name: str, required: bool = False):
        path = self.dataset_dir / name
        if path.exists():
            try:
                return np.load(path)
            except Exception as e:
                logger.error(f"Failed to load {name} for {self.dataset_name}: {e}")
                raise
        elif required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return None

    def _load_info(self):
        info_path = self.dataset_dir / "info.json"
        if info_path.exists():
            try:
                with open(info_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Could not load info.json for {self.dataset_name}: {e}")
        return {}

    # --------------------------------------------------------------
    def summarize(self) -> dict:
        """Compute TALENT compatibility diagnostics."""
        report = {
            "dataset": self.dataset_name,
            "task": self.task,
            "path": str(self.dataset_dir),
            "n_samples": len(self.y) if self.y is not None else None,
            "issues": [],
            "n_num_features": self.N.shape[1] if self.N is not None else 0,
            "n_cat_features": self.C.shape[1] if self.C is not None else 0,
        }

        # --------------------------------------------
        # 1️⃣ Check shapes & metadata consistency
        # --------------------------------------------
        if self.info:
            expected_n = self.info.get("n_num_features", None)
            expected_c = self.info.get("n_cat_features", None)
            if expected_n != report["n_num_features"] or expected_c != report["n_cat_features"]:
                report["issues"].append(
                    f"Metadata mismatch: info.json ({expected_n}, {expected_c}) vs actual ({report['n_num_features']}, {report['n_cat_features']})"
                )

        # --------------------------------------------
        # 2️⃣ Check for illegal or inconsistent values
        # --------------------------------------------
        # Infinite values
        for arr_name, arr in [("N", self.N), ("C", self.C)]:
            if arr is not None and np.isinf(arr).any():
                bad_indices = np.argwhere(np.isinf(arr))[:5]
                report["issues"].append(
                    f"{arr_name} contains infinite values (e.g., at indices {bad_indices.tolist()})"
                )

        # Negative categorical indices
        if self.C is not None:
            n_missing_codes = int((self.C == -1).sum())
            has_invalid_negatives = (self.C < -1).any()

            if has_invalid_negatives:
                bad_idx = np.argwhere(self.C < -1)[:5]
                report["issues"].append(
                    f"C contains invalid negative category indices (< -1) at positions {bad_idx.tolist()}"
                )
            elif n_missing_codes > 0:
                report["issues"].append(
                    f"C contains {n_missing_codes} entries with -1 (missing category placeholders)."
                )

        # NaN or inf in targets (critical)
        if np.isnan(self.y).any():
            nan_rows = np.argwhere(np.isnan(self.y))[:5].flatten().tolist()
            report["issues"].append(f"Target y contains NaN values at rows {nan_rows}")
        if np.isinf(self.y).any():
            inf_rows = np.argwhere(np.isinf(self.y))[:5].flatten().tolist()
            report["issues"].append(f"Target y contains Inf values at rows {inf_rows}")

        # --------------------------------------------
        # 3️⃣ Constant or degenerate features
        # --------------------------------------------
        for arr_name, arr in [("N", self.N), ("C", self.C)]:
            if arr is not None and arr.ndim == 2 and arr.shape[1] > 0:
                var = np.nanvar(arr, axis=0)
                const_cols = np.where(var == 0)[0].tolist()
                if const_cols:
                    report["issues"].append(
                        f"{len(const_cols)} constant columns in {arr_name} (first few indices: {const_cols[:10]})"
                    )

        # --------------------------------------------
        # 4️⃣ Shape mismatches
        # --------------------------------------------
        n_rows = len(self.y)
        for arr_name, arr in [("N", self.N), ("C", self.C)]:
            if arr is not None and arr.shape[0] != n_rows:
                report["issues"].append(
                    f"Shape mismatch: {arr_name}.shape[0]={arr.shape[0]} != len(y)={n_rows}"
                )

        # --------------------------------------------
        # Finalize
        # --------------------------------------------
        if not report["issues"]:
            report["issues"] = ["✅ All TALENT compatibility checks passed."]

        return report

    # --------------------------------------------------------------
    def pretty_print(self):
        """Pretty-print dataset inspection results."""
        rep = self.summarize()

        print(f"\n📦 Dataset: {rep['dataset']} [{rep['task'].upper()}]")
        print(f"📁 Path: {rep['path']}")
        print(f"🧮 Samples: {rep['n_samples']} | Num features: {rep['n_num_features']} | Cat features: {rep['n_cat_features']}")
        print(f"⚠️ Issues:")
        for issue in rep["issues"]:
            print(f"   - {issue}")
        print("-" * 70)
        return rep
