from __future__ import annotations
import numpy as np
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ProcessedDataInspector:
    """
    Inspector for TALENT-preprocessed datasets.

    Performs post-preprocessing quality and consistency checks on TALENT datasets,
    and adds task-specific target variable summaries.

    Parameters
    ----------
    dataset_name : str
        Name of dataset (e.g., '0001.gmsc')
    task : str
        Either 'pd' (classification) or 'lgd' (regression)
    processed_root : str | Path
        Root directory of processed datasets (default: 'data/processed/')
    """

    def __init__(self, dataset_name: str, task: str, processed_root: str | Path = "data/processed/"):
        self.dataset_name = dataset_name
        self.task = task.lower()
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
    def _summarize_target(self) -> dict:
        """Generate a summary of the target variable depending on task type."""
        if self.y is None:
            return {"error": "Target array y.npy not found."}

        y = np.squeeze(self.y)

        # Classification (PD)
        if self.task == "pd":
            unique, counts = np.unique(y, return_counts=True)
            proportions = counts / counts.sum() * 100
            return {
                "n_classes": len(unique),
                "class_counts": {int(k): int(v) for k, v in zip(unique, counts)},
                "class_distribution_%": {int(k): round(p, 2) for k, p in zip(unique, proportions)},
                "majority_class_ratio": round(counts.max() / counts.min(), 3) if len(unique) > 1 else None,
            }

        # Regression (LGD)
        elif self.task == "lgd":
            return {
                "mean": float(np.nanmean(y)),
                "std": float(np.nanstd(y)),
                "min": float(np.nanmin(y)),
                "max": float(np.nanmax(y)),
                "n_missing": int(np.isnan(y).sum()),
            }

        else:
            return {"warning": f"Unknown task type: {self.task}"}

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
            "target_summary": self._summarize_target(),
        }

        # --------------------------------------------
        # 1️⃣ Check metadata consistency
        # --------------------------------------------
        if self.info:
            expected_n = self.info.get("n_num_features", None)
            expected_c = self.info.get("n_cat_features", None)
            if expected_n != report["n_num_features"] or expected_c != report["n_cat_features"]:
                report["issues"].append(
                    f"Metadata mismatch: info.json ({expected_n}, {expected_c}) vs actual ({report['n_num_features']}, {report['n_cat_features']})"
                )

        # --------------------------------------------
        # 2️⃣ Illegal / inconsistent feature values
        # --------------------------------------------
        for arr_name, arr in [("N", self.N), ("C", self.C)]:
            if arr is not None and np.isinf(arr).any():
                bad_indices = np.argwhere(np.isinf(arr))[:5]
                report["issues"].append(f"{arr_name} contains ∞ values (e.g., at indices {bad_indices.tolist()})")

        if self.C is not None:
            n_missing_codes = int((self.C == -1).sum())
            if (self.C < -1).any():
                bad_idx = np.argwhere(self.C < -1)[:5]
                report["issues"].append(f"C has invalid negative indices (< -1) at {bad_idx.tolist()}")
            elif n_missing_codes > 0:
                report["issues"].append(f"C contains {n_missing_codes} missing category placeholders (-1).")

        if np.isnan(self.y).any():
            nan_rows = np.argwhere(np.isnan(self.y))[:5].flatten().tolist()
            report["issues"].append(f"y contains NaN values at rows {nan_rows}")
        if np.isinf(self.y).any():
            inf_rows = np.argwhere(np.isinf(self.y))[:5].flatten().tolist()
            report["issues"].append(f"y contains Inf values at rows {inf_rows}")

        # --------------------------------------------
        # 3️⃣ Constant features
        # --------------------------------------------
        for arr_name, arr in [("N", self.N), ("C", self.C)]:
            if arr is not None and arr.ndim == 2 and arr.shape[1] > 0:
                const_cols = np.where(np.nanvar(arr, axis=0) == 0)[0].tolist()
                if const_cols:
                    report["issues"].append(
                        f"{len(const_cols)} constant columns in {arr_name} (first few: {const_cols[:10]})"
                    )

        # --------------------------------------------
        # 4️⃣ Shape mismatches
        # --------------------------------------------
        n_rows = len(self.y)
        for arr_name, arr in [("N", self.N), ("C", self.C)]:
            if arr is not None and arr.shape[0] != n_rows:
                report["issues"].append(
                    f"Shape mismatch: {arr_name}.shape[0]={arr.shape[0]} ≠ len(y)={n_rows}"
                )

        if not report["issues"]:
            report["issues"] = ["✅ All TALENT compatibility checks passed."]

        return report

    # --------------------------------------------------------------
    def pretty_print(self):
        """Pretty-print dataset inspection results."""
        rep = self.summarize()

        print(f"\n📦 Dataset: {rep['dataset']} [{rep['task'].upper()}]")
        print(f"📁 Path: {rep['path']}")
        print(f"🧮 Samples: {rep['n_samples']} | Num: {rep['n_num_features']} | Cat: {rep['n_cat_features']}")
        print(f"\n🎯 Target Summary:")
        for k, v in rep["target_summary"].items():
            print(f"   {k}: {v}")
        print(f"\n⚠️ Issues:")
        for issue in rep["issues"]:
            print(f"   - {issue}")
        print("-" * 70)
        return rep
