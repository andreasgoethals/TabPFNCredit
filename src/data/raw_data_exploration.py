import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

class RawDataInspector:
    """
    Automated schema & quality profiler for raw credit datasets.
    Automatically infers task (PD/LGD) and computes dataset-level diagnostics.
    """

    def __init__(self, dataset_name: str, task: Optional[str] = None, raw_root: str = "data/raw/"):
        self.dataset_name = dataset_name
        self.raw_root = Path(raw_root)
        self.task = task or self._infer_task()
        self.file_path = self._find_dataset_path()
        self.df = None
        self.report = {}

    # -------------------------------------------------------------
    def _infer_task(self) -> str:
        """Infer task (pd/lgd) from available subfolders."""
        for t in ["pd", "lgd"]:
            if (self.raw_root / t / f"{self.dataset_name}.csv").exists():
                return t
        raise FileNotFoundError(f"Dataset {self.dataset_name} not found in pd/ or lgd/ folders.")

    def _find_dataset_path(self) -> Path:
        """Return the full path to the dataset CSV."""
        path = self.raw_root / self.task / f"{self.dataset_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {path}")
        return path

    # -------------------------------------------------------------
    def load(self):
        """Load dataset into memory."""
        self.df = pd.read_csv(self.file_path)
        return self.df

    # -------------------------------------------------------------
    def summarize(self) -> dict:
        """Compute dataset-wide statistics."""
        if self.df is None:
            self.load()
        df = self.df

        n_rows, n_cols = df.shape
        dtype_counts = df.dtypes.value_counts().to_dict()

        # Missingness
        col_missing = df.isnull().sum()
        row_missing = df.isnull().any(axis=1).sum()
        n_cols_missing = (col_missing > 0).sum()
        n_rows_missing = row_missing

        # Constant / unique values
        n_constant = sum(df.nunique() == 1)
        high_card_cols = [c for c in df.columns if df[c].nunique() > 0.9 * len(df)]

        # Memory footprint
        mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

        # Numeric overview
        num_df = df.select_dtypes(include=[np.number])
        num_summary = {
            "mean_mean": num_df.mean().mean(),
            "mean_std": num_df.std().mean(),
            "mean_min": num_df.min().mean(),
            "mean_max": num_df.max().mean(),
        } if not num_df.empty else {}

        # Categorical overview
        cat_df = df.select_dtypes(exclude=[np.number])
        cat_summary = {
            "avg_unique_categories": cat_df.nunique().mean(),
            "max_unique_categories": cat_df.nunique().max(),
        } if not cat_df.empty else {}

        # Potential issues
        issues = []

        # Constant columns
        if n_constant > 0:
            constant_cols = df.columns[df.nunique() == 1].tolist()
            issues.append(
                f"{n_constant} constant columns: {constant_cols[:10]}"
                + ("..." if len(constant_cols) > 10 else "")
            )

        # High-cardinality columns (possibly IDs)
        if len(high_card_cols) > 0:
            issues.append(
                f"{len(high_card_cols)} high-cardinality columns (>90% unique): {high_card_cols[:10]}"
                + ("..." if len(high_card_cols) > 10 else "")
            )

        # Duplicate rows
        n_dupes = df.duplicated().sum()
        if n_dupes > 0:
            dupe_indices = df.index[df.duplicated()].tolist()[:10]
            issues.append(
                f"{n_dupes} duplicate rows detected (first few indices: {dupe_indices})"
            )

        # Missing values
        cols_with_missing = df.columns[df.isnull().any()].tolist()
        if cols_with_missing:
            missing_rows = df[df.isnull().any(axis=1)].index[:10].tolist()
            issues.append(
                f"{len(cols_with_missing)} columns contain missing values: {cols_with_missing[:10]}"
                + ("..." if len(cols_with_missing) > 10 else "")
                + f" | Example row indices with NaN: {missing_rows}"
            )

        # If nothing suspicious
        if not issues:
            issues = ["No immediate structural issues detected."]

        self.report = {
            "dataset": self.dataset_name,
            "task": self.task,
            "path": str(self.file_path),
            "n_rows": n_rows,
            "n_columns": n_cols,
            "dtype_counts": dtype_counts,
            "columns_with_missing": int(n_cols_missing),
            "rows_with_missing": int(n_rows_missing),
            "constant_columns": int(n_constant),
            "memory_usage_MB": round(mem_mb, 2),
            "numeric_summary": num_summary,
            "categorical_summary": cat_summary,
            "issues": issues,
        }

        return self.report

    # -------------------------------------------------------------
    def pretty_print(self):
        """Print a neat summary to console."""
        if not self.report:
            self.summarize()

        print(f"🔍 Dataset: {self.dataset_name}  |  Task: {self.task.upper()}")
        print(f"📊 Shape: {self.report['n_rows']} rows × {self.report['n_columns']} columns")
        print(f"🧩 Dtypes: {self.report['dtype_counts']}")
        print(f"❓ Missing: {self.report['columns_with_missing']} cols, {self.report['rows_with_missing']} rows")
        print(f"💾 Memory: {self.report['memory_usage_MB']} MB")
        print(f"⚠️ Issues: {'; '.join(self.report['issues'])}")
        print("-" * 60)
